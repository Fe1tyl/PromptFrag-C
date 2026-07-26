"""CIFAR-10 and CIFAR-10-C dataset loading."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


def resolve_cifar10c_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    candidates = (root / "CIFAR-10-C", root)
    for candidate in candidates:
        if (candidate / "labels.npy").exists():
            return candidate
    raise FileNotFoundError(
        f"CIFAR-10-C not found under {root}. "
        "Run scripts/download_cifar10c.py first."
    )


def fixed_subset_indices(length: int, sample_limit: int | None, seed: int) -> np.ndarray:
    if sample_limit is None or sample_limit >= length:
        return np.arange(length, dtype=np.int64)
    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive or null")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=sample_limit, replace=False)).astype(np.int64)


class CIFAR10CDataset:
    """Memory-mapped view of one CIFAR-10-C corruption severity."""

    def __init__(
        self,
        data_root: str | Path,
        corruption: str,
        severity: int,
        transform: Callable,
        indices: np.ndarray | None = None,
    ) -> None:
        if severity not in {1, 2, 3, 4, 5}:
            raise ValueError("severity must be in {1, 2, 3, 4, 5}")
        root = resolve_cifar10c_root(data_root)
        image_path = root / f"{corruption}.npy"
        if not image_path.exists():
            raise FileNotFoundError(f"missing corruption array: {image_path}")

        self.images = np.load(image_path, mmap_mode="r")
        self.labels = np.load(root / "labels.npy", mmap_mode="r")
        self.start = (severity - 1) * 10_000
        self.stop = severity * 10_000
        if self.images.shape[0] < self.stop:
            raise ValueError(
                f"{image_path} has {self.images.shape[0]} images; "
                f"severity {severity} requires at least {self.stop}"
            )
        if self.labels.shape[0] not in {10_000, self.images.shape[0]}:
            raise ValueError(
                "labels.npy must contain either 10,000 labels or one label per image"
            )
        self.indices = (
            np.arange(10_000, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if np.any(self.indices < 0) or np.any(self.indices >= 10_000):
            raise ValueError("subset index outside [0, 10000)")
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int):
        local_index = int(self.indices[item])
        image = np.array(self.images[self.start + local_index], copy=True)
        if self.labels.shape[0] == 10_000:
            label = int(self.labels[local_index])
        else:
            label = int(self.labels[self.start + local_index])
        return self.transform(Image.fromarray(image)), label


def make_dataset(
    data_root: str | Path,
    corruption: str,
    severity: int,
    transform: Callable,
    sample_limit: int | None,
    seed: int,
):
    indices = fixed_subset_indices(10_000, sample_limit, seed)
    if corruption == "clean":
        from torch.utils.data import Subset
        from torchvision.datasets import CIFAR10

        try:
            dataset = CIFAR10(
                root=str(Path(data_root)),
                train=False,
                transform=transform,
                download=False,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Clean CIFAR-10 is missing or invalid. Run "
                "scripts/download_cifar10.py before inference."
            ) from exc
        return Subset(dataset, indices.tolist())

    return CIFAR10CDataset(
        data_root=data_root,
        corruption=corruption,
        severity=severity,
        transform=transform,
        indices=indices,
    )
