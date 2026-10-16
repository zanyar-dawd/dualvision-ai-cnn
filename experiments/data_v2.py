"""Training-only candidate preparation. Official test data are never opened here."""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.utilities.data import CIFAR_CLASSES, split_indices, channel_statistics, image_transform


def candidate_transform(dataset, mean, std):
    if dataset == 'mnist':
        augmentation = [transforms.RandomAffine(degrees=10, translate=(0.08, 0.08),
                         scale=(0.9, 1.1), shear=5,
                         interpolation=transforms.InterpolationMode.BILINEAR, fill=0)]
    elif dataset == 'cifar10':
        augmentation = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)]
    else:
        raise ValueError(dataset)
    return transforms.Compose(augmentation + [transforms.ToTensor(), transforms.Normalize(mean, std)])


@dataclass
class CandidateData:
    train: DataLoader
    validation: DataLoader
    clean_train: DataLoader
    classes: list
    mean: list
    std: list
    metadata: dict


def prepare_candidate_data(dataset, root, config, smoke=False):
    seed, batch = config['seed'], config['batch_size']
    gray = dataset == 'mnist'
    classes = [str(i) for i in range(10)] if gray else CIFAR_CLASSES
    if smoke:
        mean, std = ([0.5], [0.25]) if gray else ([0.5] * 3, [0.25] * 3)
        shape = (1, 28, 28) if gray else (3, 32, 32)
        training = datasets.FakeData(16, shape, 10, transform=candidate_transform(dataset, mean, std), random_offset=seed)
        validation = datasets.FakeData(10, shape, 10, transform=image_transform(dataset, mean, std), random_offset=seed+1000)
        clean = datasets.FakeData(16, shape, 10, transform=image_transform(dataset, mean, std), random_offset=seed)
        metadata = {'source': 'SYNTHETIC_SMOKE_ONLY', 'split_sha256': 'synthetic-v2'}
    else:
        constructor = datasets.MNIST if gray else datasets.CIFAR10
        raw = constructor(str(root), train=True, download=False)
        assert len(raw) == (60000 if gray else 50000)
        if not gray:
            assert raw.classes == CIFAR_CLASSES
        train_indices, validation_indices = split_indices(raw.targets, seed)
        mean, std = channel_statistics(raw.data, train_indices, gray)
        training = constructor(str(root), train=True, download=False,
                               transform=candidate_transform(dataset, mean, std))
        clean_source = constructor(str(root), train=True, download=False,
                                   transform=image_transform(dataset, mean, std))
        training = Subset(training, train_indices.tolist())
        validation = Subset(clean_source, validation_indices.tolist())
        # Fixed diagnostic sample; only training indices, measured under validation conditions.
        clean_indices = np.random.default_rng(seed + 800).choice(train_indices, min(2048, len(train_indices)), replace=False)
        clean = Subset(clean_source, clean_indices.tolist())
        metadata = {'source': 'torchvision-official', 'train_indices': train_indices.tolist(),
                    'validation_indices': validation_indices.tolist(), 'clean_train_indices': clean_indices.tolist(),
                    'split_sha256': hashlib.sha256(train_indices.astype('<i8').tobytes() + validation_indices.astype('<i8').tobytes()).hexdigest(),
                    'statistics_source': 'training-subset-only', 'official_test_opened': False}
    loaders = [DataLoader(part, batch_size=batch, shuffle=index == 0, num_workers=0,
                         generator=torch.Generator().manual_seed(seed + index))
               for index, part in enumerate([training, validation, clean])]
    metadata['sizes'] = [len(part) for part in [training, validation, clean]]
    return CandidateData(*loaders, classes, mean, std, metadata)
