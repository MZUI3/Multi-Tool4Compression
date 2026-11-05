"""CIFAR-10/100 kept on the device as uint8 tensors, with augmentation done on the device.

Compression runs fine-tune for a few epochs many times over, so the per-epoch cost of a CPU DataLoader
(worker start-up on Windows, PIL transforms) would dominate. The whole training set is 150 MB as uint8,
so it fits on any GPU. Augmentation is the CIFAR recipe of Multi-Tool4Vision without the rotation:
4-pixel padded random crop and horizontal flip. Normalisation uses the same mean and std.
"""
import os
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F

DATASETS = {
    "cifar10": dict(
        num_classes=10,
        folder="cifar-10-batches-py",
        url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        train=[f"data_batch_{i}" for i in range(1, 6)],
        test=["test_batch"],
        label_key=b"labels",
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
    "cifar100": dict(
        num_classes=100,
        folder="cifar-100-python",
        url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        train=["train"],
        test=["test"],
        label_key=b"fine_labels",
        mean=(0.5071, 0.4865, 0.4409),
        std=(0.2673, 0.2564, 0.2762),
    ),
}
IMG_SIZE = 32


def _download(info: dict, root: str) -> None:
    os.makedirs(root, exist_ok=True)
    archive = os.path.join(root, os.path.basename(info["url"]))
    if not os.path.exists(archive):
        print(f"downloading {info['url']}")
        urllib.request.urlretrieve(info["url"], archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root, filter="data")


def _load_split(info: dict, root: str, files) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for name in files:
        with open(os.path.join(root, info["folder"], name), "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        xs.append(batch[b"data"].reshape(-1, 3, 32, 32))
        ys.append(np.asarray(batch[info["label_key"]], dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


@dataclass
class Batches:
    """An iterable over (images, labels) batches of a device-resident split."""

    images: torch.Tensor  # uint8 N,3,32,32
    labels: torch.Tensor
    batch_size: int
    train: bool
    mean: torch.Tensor
    std: torch.Tensor
    limit: int = 0  # max batches per pass, 0 = all
    generator: torch.Generator = None

    def __len__(self) -> int:
        n = (len(self.labels) + self.batch_size - 1) // self.batch_size
        return min(n, self.limit) if self.limit else n

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        padded = F.pad(x, (4, 4, 4, 4))
        dx = torch.randint(0, 9, (n,), device=x.device, generator=self.generator)
        dy = torch.randint(0, 9, (n,), device=x.device, generator=self.generator)
        out = self._crop(padded, dx, dy)
        flip = torch.rand(n, device=x.device, generator=self.generator) < 0.5
        out[flip] = out[flip].flip(3)
        return out

    @staticmethod
    def _crop(padded: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        n = padded.shape[0]
        ar = torch.arange(32, device=padded.device)
        rows = (dy[:, None] + ar[None])  # n,32
        cols = (dx[:, None] + ar[None])  # n,32
        idx_n = torch.arange(n, device=padded.device)[:, None, None]
        out = padded.permute(0, 2, 3, 1)[idx_n, rows[:, :, None], cols[:, None, :]]  # n,32,32,3
        return out.permute(0, 3, 1, 2).contiguous()

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n = len(self.labels)
        if self.train:
            order = torch.randperm(n, device=self.labels.device, generator=self.generator)
        else:
            order = torch.arange(n, device=self.labels.device)
        for i in range(len(self)):
            idx = order[i * self.batch_size:(i + 1) * self.batch_size]
            x = self.images[idx].float().div_(255)
            if self.train:
                x = self._augment(x)
            yield (x - self.mean) / self.std, self.labels[idx]


class CifarData:
    """``train``, ``val`` and ``calib`` batch iterables on one device."""

    def __init__(self, name: str = "cifar10", root: str = "./data", device="cpu", batch_size: int = 128,
                 calib_size: int = 1024, seed: int = 0, download: bool = True):
        if name not in DATASETS:
            raise KeyError(f"Unknown dataset '{name}'. Expected one of {list(DATASETS)}")
        info = DATASETS[name]
        if not os.path.isdir(os.path.join(root, info["folder"])):
            if not download:
                raise FileNotFoundError(f"{info['folder']} not found in {root}")
            _download(info, root)
        self.name = name
        self.num_classes = info["num_classes"]
        self.img_size = IMG_SIZE
        self.device = torch.device(device)
        x_train, y_train = _load_split(info, root, info["train"])
        x_test, y_test = _load_split(info, root, info["test"])
        self.x_train = torch.from_numpy(x_train).to(self.device)
        self.y_train = torch.from_numpy(y_train).to(self.device)
        self.x_val = torch.from_numpy(x_test).to(self.device)
        self.y_val = torch.from_numpy(y_test).to(self.device)
        self.mean = torch.tensor(info["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(info["std"], device=self.device).view(1, 3, 1, 1)
        self.batch_size = batch_size
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        # A fixed calibration subset of the training set (no augmentation), the same for every method.
        g = torch.Generator().manual_seed(1234)
        self.calib_idx = torch.randperm(len(y_train), generator=g)[:calib_size].to(self.device)

    def train(self, batch_size: int = None, limit: int = 0) -> Batches:
        return Batches(self.x_train, self.y_train, batch_size or self.batch_size, True, self.mean, self.std,
                       limit, self.generator)

    def val(self, batch_size: int = 500, limit: int = 0) -> Batches:
        return Batches(self.x_val, self.y_val, batch_size, False, self.mean, self.std, limit)

    def calib(self, batch_size: int = 128, limit: int = 0) -> Batches:
        return Batches(self.x_train[self.calib_idx], self.y_train[self.calib_idx], batch_size, False,
                       self.mean, self.std, limit)

    def example_input(self, batch: int = 1) -> torch.Tensor:
        return torch.randn(batch, 3, self.img_size, self.img_size, device=self.device)
