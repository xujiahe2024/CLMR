import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Audio Augmentations
from torchaudio_augmentations import (
    RandomApply,
    ComposeMany,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)

from clmr.data import ContrastiveDataset
from clmr.datasets import get_dataset
from clmr.evaluation import evaluate
from clmr.models import SampleCNN
import clmr.modules.contrastive_learning as cl
import clmr.modules.supervised_learning as sl
from clmr.utils import yaml_config_hook


if __name__ == "__main__":

    print("Program started...")
    parser = argparse.ArgumentParser(description="CLMR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument("--train", action="store_true", help="Flag to enable training")
    parser.add_argument("--test", action="store_true", help="Flag to enable testing")

    args = parser.parse_args()
    print("Configuration loaded successfully!")
    print(f"Configuration: {args}")

    pl.seed_everything(args.seed)
    print("Random seed set...")

    # ------------
    # data augmentations
    # ------------
    print("Setting up data augmentations...")
    if args.supervised:
        train_transform = [RandomResizedCrop(n_samples=args.audio_length)]
        num_augmented_samples = 1
    else:
        train_transform = [
            RandomResizedCrop(n_samples=args.audio_length),
            RandomApply([PolarityInversion()], p=args.transforms_polarity),
            RandomApply([Noise()], p=args.transforms_noise),
            RandomApply([Gain()], p=args.transforms_gain),
            RandomApply(
                [HighLowPass(sample_rate=args.sample_rate)], p=args.transforms_filters
            ),
            RandomApply([Delay(sample_rate=args.sample_rate)], p=args.transforms_delay),
            RandomApply(
                [
                    PitchShift(
                        n_samples=args.audio_length,
                        sample_rate=args.sample_rate,
                    )
                ],
                p=args.transforms_pitch,
            ),
            RandomApply(
                [Reverb(sample_rate=args.sample_rate)], p=args.transforms_reverb
            ),
        ]
        num_augmented_samples = 2
    print("Data augmentations set up successfully!")

    # ------------
    # dataloaders
    # ------------
    print("Loading datasets...")
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")
    contrastive_train_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=(1, args.audio_length),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
        ),
    )
    contrastive_valid_dataset = ContrastiveDataset(
        valid_dataset,
        input_shape=(1, args.audio_length),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
        ),
    )
    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True,
    )
    valid_loader = DataLoader(
        contrastive_valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=False,
    )
    print("Dataloaders created successfully!")

    # ------------
    # encoder
    # ------------
    print("Initializing encoder...")
    encoder = SampleCNN(
        input_dim=1,
        output_dim=train_dataset.n_classes,
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=args.supervised,
    )
    print("Encoder initialized successfully!")

    # ------------
    # model
    # ------------
    print("Initializing model...")
    if args.supervised:
        module = sl.SupervisedLearning(args, encoder, output_dim=train_dataset.n_classes)
        print("Supervised learning module initialized!")
    else:
        module = cl.ContrastiveLearning(args, encoder)
        print("Contrastive learning module initialized!")

    if args.checkpoint_path:
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        module = module.load_from_checkpoint(
            args.checkpoint_path, encoder=encoder, output_dim=train_dataset.n_classes
        )
        print("Checkpoint loaded successfully!")

    # ------------
    # training
    # ------------
    if args.train:
        print("Starting training...")
        early_stopping = EarlyStopping(monitor="Valid/loss", patience=20)
        trainer = Trainer.from_argparse_args(
            args,
            logger=TensorBoardLogger("runs", name="CLMRv2-{}".format(args.dataset)),
            sync_batchnorm=True,
            max_epochs=args.max_epochs,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            accelerator=args.accelerator,
        )
        trainer.fit(module, train_loader, valid_loader)
        print("Training completed!")

    # ------------
    # testing
    # ------------
    if args.test:
        print("Starting evaluation...")
        test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")
        contrastive_test_dataset = ContrastiveDataset(
            test_dataset,
            input_shape=(1, args.audio_length),
            transform=None,
        )
        device = "cuda:0" if args.gpus else "cpu"
        results = evaluate(
            module.encoder,
            None,
            contrastive_test_dataset,
            args.dataset,
            args.audio_length,
            device=device,
        )
        print("Evaluation completed!")
        print(f"Results: {results}")
