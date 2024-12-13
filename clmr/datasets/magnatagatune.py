import os
import warnings
import subprocess
import torch
import numpy as np
import zipfile
from collections import defaultdict
from typing import Any, Tuple, Optional
from tqdm import tqdm

import soundfile as sf
import torchaudio

# 设置 torchaudio 的音频后端
torchaudio.set_audio_backend("soundfile")
from torch import Tensor, FloatTensor
from torch.hub import download_url_to_file  # 替换旧版的 download_url
from clmr.datasets.dataset import Dataset  # 导入 Dataset 基类

FOLDER_IN_ARCHIVE = "magnatagatune"
_CHECKSUMS = {
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001": "",
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002": "",
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003": "",
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/binary.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/tags.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/test.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/train.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/valid.npy": "",
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/train_gt_mtt.tsv": "",
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/val_gt_mtt.tsv": "",
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/test_gt_mtt.tsv": "",
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/index_mtt.tsv": "",
}


def get_file_list(root, subset, split):
    if subset == "train":
        if split == "pons2017":
            fl = open(os.path.join(root, "train_gt_mtt.tsv")).read().splitlines()
        else:
            fl = np.load(os.path.join(root, "train.npy"))
    elif subset == "valid":
        if split == "pons2017":
            fl = open(os.path.join(root, "val_gt_mtt.tsv")).read().splitlines()
        else:
            fl = np.load(os.path.join(root, "valid.npy"))
    else:
        if split == "pons2017":
            fl = open(os.path.join(root, "test_gt_mtt.tsv")).read().splitlines()
        else:
            fl = np.load(os.path.join(root, "test.npy"))

    if split == "pons2017":
        binary = {}
        index = open(os.path.join(root, "index_mtt.tsv")).read().splitlines()
        fp_dict = {}
        for i in index:
            clip_id, fp = i.split("\t")
            fp_dict[clip_id] = fp

        for idx, f in enumerate(fl):
            clip_id, label = f.split("\t")
            fl[idx] = "{}\t{}".format(clip_id, fp_dict[clip_id])
            clip_id = int(clip_id)
            binary[clip_id] = eval(label)
    else:
        binary = np.load(os.path.join(root, "binary.npy"))

    return fl, binary


class MAGNATAGATUNE(Dataset):
    """Create a Dataset for MagnaTagATune.
    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        folder_in_archive (str, optional): The top-level directory of the dataset.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str, optional): Which subset of the dataset to use.
            One of ``"train"``, ``"valid"``, ``"test"`` or ``None``.
            If ``None``, the entire dataset is used. (default: ``None``).
        split (str, optional): Split type, defaults to "pons2017".
    """

    _ext_audio = ".wav"

    def __init__(
        self,
        root: str,
        folder_in_archive: Optional[str] = FOLDER_IN_ARCHIVE,
        download: Optional[bool] = False,
        subset: Optional[str] = None,
        split: Optional[str] = "pons2017",
    ) -> None:
        super(MAGNATAGATUNE, self).__init__(root)
        self.root = root
        self.folder_in_archive = folder_in_archive
        self.download = download
        self.subset = subset
        self.split = split

        assert subset is None or subset in ["train", "valid", "test"], (
            "When `subset` is not None, it must take a value from "
            + "{'train', 'valid', 'test'}."
        )

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                os.makedirs(self._path)

            zip_files = []
            for url, checksum in _CHECKSUMS.items():
                target_fn = os.path.basename(url)
                target_fp = os.path.join(self._path, target_fn)
                if ".zip" in target_fp:
                    zip_files.append(target_fp)

                if not os.path.exists(target_fp):
                    download_url_to_file(url, target_fp)  # 使用 download_url_to_file 替换旧版

            if not os.path.exists(
                os.path.join(
                    self._path,
                    "f",
                    "american_bach_soloists-j_s__bach_solo_cantatas-01-bwv54__i_aria-30-59.mp3",
                )
            ):
                merged_zip = os.path.join(self._path, "mp3.zip")
                print("Merging zip files...")
                with open(merged_zip, "wb") as f:
                    for filename in zip_files:
                        with open(filename, "rb") as g:
                            f.write(g.read())

                # 使用 zipfile 进行解压
                with zipfile.ZipFile(merged_zip, 'r') as zip_ref:
                    zip_ref.extractall(self._path)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        self.fl, self.binary = get_file_list(self._path, self.subset, self.split)
        self.n_classes = 50

    def file_path(self, n: int) -> str:
        _, fp = self.fl[n].split("\t")
        return os.path.join(self.root, fp)


    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        clip_id, fp = self.fl[n].split("\t")
        label = self.binary[int(clip_id)]

        audio, _ = self.load(n)
        label = FloatTensor(label)
        return audio, label

    def __len__(self) -> int:
        return len(self.fl)
