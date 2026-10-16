"""Official partitions, stratified validation, and training-only statistics."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


@dataclass
class DataBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    classes: list[str]
    mean: list[float]
    std: list[float]
    metadata: dict


def split_indices(targets, seed: int = 42):
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=seed, stratify=np.asarray(targets))
    assert not np.intersect1d(train_idx, val_idx).size
    assert len(train_idx) + len(val_idx) == len(targets)
    return train_idx, val_idx


def channel_statistics(raw_data, indices, grayscale: bool) -> tuple[list, list]:
    """Population channel moments computed only on the selected training subset."""
    raw = np.asarray(raw_data)
    total = np.zeros(1 if grayscale else 3, dtype=np.float64)
    square = np.zeros_like(total)
    count = 0
    for start in range(0, len(indices), 512):
        pixels = raw[indices[start:start + 512]].astype(np.float64) / 255.0
        if grayscale:
            pixels = pixels[..., None]
        pixels = pixels.reshape(-1, pixels.shape[-1])
        total += pixels.sum(axis=0)
        square += np.square(pixels).sum(axis=0)
        count += len(pixels)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean ** 2, 1e-12))
    return mean.tolist(), std.tolist()


def image_transform(dataset: str, mean, std, training: bool = False):
    ops = []
    if dataset == 'cifar10' and training:
        ops.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(ops)


def prepare_data(dataset: str, root: Path, batch_size: int, seed: int = 42,
                 download: bool = False, smoke: bool = False, pin_memory: bool = False) -> DataBundle:
    if dataset not in {'mnist', 'cifar10'}:
        raise ValueError(dataset)
    grayscale = dataset == 'mnist'
    classes = [str(i) for i in range(10)] if grayscale else CIFAR_CLASSES
    if smoke:
        # Synthetic data checks plumbing only. It never produces benchmark evidence.
        mean, std = ([0.5], [0.25]) if grayscale else ([0.5] * 3, [0.25] * 3)
        shape = (1, 28, 28) if grayscale else (3, 32, 32)
        parts = [datasets.FakeData(n, shape, 10, transform=image_transform(dataset, mean, std, i == 0),
                                  random_offset=seed + i * 1000) for i, n in enumerate([32, 16, 20])]
        metadata = {'source': 'SYNTHETIC_SMOKE_ONLY', 'split_seed': seed,
                    'sizes': [32, 16, 20], 'split_sha256': 'synthetic-disjoint-offsets'}
    else:
        constructor = datasets.MNIST if grayscale else datasets.CIFAR10
        raw = constructor(root=str(root), train=True, download=download)
        expected_train = 60000 if grayscale else 50000
        if len(raw) != expected_train:
            raise ValueError('Unexpected official training partition size.')
        train_idx, val_idx = split_indices(raw.targets, seed)
        mean, std = channel_statistics(raw.data, train_idx, grayscale)
        training = constructor(root=str(root), train=True, download=False,
                               transform=image_transform(dataset, mean, std, True))
        validation = constructor(root=str(root), train=True, download=False,
                                 transform=image_transform(dataset, mean, std, False))
        test = constructor(root=str(root), train=False, download=download,
                           transform=image_transform(dataset, mean, std, False))
        if len(test) != 10000:
            raise ValueError('Unexpected official test partition size.')
        if not grayscale and list(raw.classes) != CIFAR_CLASSES:
            raise ValueError('Unexpected CIFAR-10 class order.')
        parts = [Subset(training, train_idx.tolist()), Subset(validation, val_idx.tolist()), test]
        metadata = {'source': 'torchvision-official', 'split_seed': seed,
                    'sizes': [len(p) for p in parts],
                    'split_sha256': hashlib.sha256(train_idx.astype('<i8').tobytes() + val_idx.astype('<i8').tobytes()).hexdigest(),
                    'train_indices': train_idx.tolist(), 'validation_indices': val_idx.tolist(),
                    'statistics_source': 'training-subset-only'}
    # num_workers=0 is deliberate: portable in Windows notebooks, deterministic and modest RAM use.
    loaders = [DataLoader(part, batch_size=batch_size, shuffle=i == 0, num_workers=0,
                          pin_memory=pin_memory, generator=torch.Generator().manual_seed(seed + i))
               for i, part in enumerate(parts)]
    return DataBundle(*loaders, classes, mean, std, metadata)
