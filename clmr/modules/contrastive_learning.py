import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from simclr import SimCLR
from simclr.modules import LARS


class ContrastiveLearning(LightningModule):
    def __init__(self, args, encoder: nn.Module):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder = encoder
        if hasattr(self.encoder.fc, "in_features"):
            self.n_features = self.encoder.fc.in_features
        else:
            self.n_features = 512  # Default if `fc` not present

        self.encoder.fc = nn.Identity()
        if not hasattr(self.hparams, "projection_dim"):
            self.hparams.projection_dim = 128  # Add default projection_dim

        self.model = SimCLR(self.encoder, self.hparams.projection_dim, self.n_features)
        self.criterion = self.configure_criterion()

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        _, _, z_i, z_j = self.model(x_i, x_j)
        return self.criterion(z_i, z_j)

    def training_step(self, batch, _) -> Tensor:
        x, _ = batch
        x_i, x_j = x[:, 0, :], x[:, 1, :]
        print(f"Batch input shape: {x.shape}")
        loss = self.forward(x_i, x_j)
        self.log("Train/loss", loss)
        return loss

    def configure_criterion(self) -> nn.Module:
        batch_size = self.hparams.batch_size
        print(f"Configured NT_Xent with batch_size={batch_size}, temperature={self.hparams.temperature}")
        return NT_Xent(batch_size, self.hparams.temperature)

    def configure_optimizers(self) -> dict:
        if self.hparams.optimizer == "Adam":
            print("Using Adam optimizer")
            optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        elif self.hparams.optimizer == "LARS":
            print("Using LARS optimizer")
            lr = 0.3 * self.hparams.batch_size / 256
            optimizer = LARS(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.hparams.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.hparams.max_epochs, eta_min=0
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            raise NotImplementedError(f"Optimizer {self.hparams.optimizer} not implemented.")
        return {"optimizer": optimizer}


class NT_Xent(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss"""

    def __init__(self, batch_size: int, temperature: float):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self._get_correlated_mask(2 * batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_correlated_mask(self, N: int):
        """Creates a mask to exclude positive pairs and diagonal elements"""
        mask = torch.ones((N, N), dtype=torch.bool)  # BoolTensor
        mask.fill_diagonal_(0)  # Remove diagonal elements (self-similarity)
        for i in range(N // 2):
            mask[i, i + N // 2] = 0
            mask[i + N // 2, i] = 0
        return mask

    def similarity_function(self, x: Tensor, y: Tensor) -> Tensor:
        """Cosine similarity"""
        return nn.functional.cosine_similarity(x, y, dim=-1)

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """Compute the loss"""
        # Concatenate embeddings from both augmentations
        z = torch.cat((z_i, z_j), dim=0)
        N = z.size(0)

        # Compute pairwise similarity
        sim = torch.mm(z, z.t().contiguous()) / self.temperature

        # Debugging log
        print(f"sim shape: {sim.shape}, mask shape: {self.mask.shape}")
        print(f"sim.device: {sim.device}, mask.device: {self.mask.device}")

        # Regenerate the mask if needed
        if self.mask.size(0) != N:
            self.mask = self._get_correlated_mask(N).to(sim.device)

        # Dynamically adjust batch size
        batch_size = N // 2  # Calculate batch size dynamically
        print(f"Adjusted batch_size: {batch_size}")

        # Extract positive samples
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i))

        # Check for empty positive samples
        if positive_samples.numel() == 0:
            raise ValueError(
                f"Positive samples tensor is empty. "
                f"sim shape: {sim.shape}, batch_size: {batch_size}"
            )

        # Extract negative samples
        negative_samples = sim[self.mask].view(N, -1)

        # Combine positive and negative samples
        labels = torch.zeros(N, dtype=torch.long).to(z.device)
        logits = torch.cat((positive_samples.unsqueeze(1), negative_samples), dim=1)

        # Compute loss
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
