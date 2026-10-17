"""Two independently trained custom classifiers; no pretrained weights."""
import torch
from torch import nn


class DigitNet(nn.Module):
    """Compact 28x28 grayscale CNN with a 128-unit classification head."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128),
                                        nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class VisionNet(nn.Module):
    """Narrow residual CNN for 32x32 RGB images and 4 GB GPUs."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU())
        self.stage1 = nn.Sequential(ResidualBlock(32, 32), ResidualBlock(32, 32))
        self.stage2 = nn.Sequential(ResidualBlock(32, 64, 2), ResidualBlock(64, 64))
        self.stage3 = nn.Sequential(ResidualBlock(64, 128, 2), ResidualBlock(128, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, 10))

    def forward(self, x):
        x = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return self.head(self.pool(x))


class DigitNetPlus(nn.Module):
    """Validation-selected six-convolution MNIST model used by the final report."""
    def __init__(self):
        super().__init__()
        layers = []
        previous = 1
        for width in (32, 64, 128):
            for _ in range(2):
                layers.extend([
                    nn.Conv2d(previous, width, 3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                ])
                previous = width
            if width < 128:
                layers.extend([nn.MaxPool2d(2), nn.Dropout2d(0.10)])
        layers.append(nn.AvgPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(), nn.Dropout(0.30), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class WidePreactivationBlock(nn.Module):
    """Preactivation residual block used by the final CIFAR-10 model."""
    def __init__(self, incoming: int, outgoing: int, stride: int = 1, dropout: float = 0.15):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(incoming)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.projection = None if incoming == outgoing and stride == 1 else nn.Conv2d(
            incoming, outgoing, 1, stride, bias=False)

    def forward(self, inputs):
        activated = self.relu(self.bn1(inputs))
        residual = inputs if self.projection is None else self.projection(activated)
        hidden = self.conv1(activated)
        hidden = self.conv2(self.dropout(self.relu(self.bn2(hidden))))
        return residual + hidden


class VisionNetWide(nn.Module):
    """Custom WRN-28-4-style network selected for the final CIFAR-10 report."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.stage1 = self._stage(16, 64, 1)
        self.stage2 = self._stage(64, 128, 2)
        self.stage3 = nn.Sequential(*self._stage(128, 256, 2), nn.BatchNorm2d(256), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256, 10))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    @staticmethod
    def _stage(incoming: int, outgoing: int, stride: int):
        return nn.Sequential(
            WidePreactivationBlock(incoming, outgoing, stride),
            *(WidePreactivationBlock(outgoing, outgoing) for _ in range(3)),
        )

    def forward(self, x):
        x = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return self.head(self.pool(x))


def make_model(dataset: str, architecture: str | None = None) -> nn.Module:
    if dataset == 'mnist':
        if architecture == 'digitnet-plus-v2':
            return DigitNetPlus()
        return DigitNet()
    if dataset == 'cifar10':
        if architecture == 'visionnet-wrn28-4-v2':
            return VisionNetWide()
        return VisionNet()
    raise ValueError(f'Unknown dataset: {dataset}')


def make_candidate(dataset: str) -> nn.Module:
    """Construct the validation-selected architecture for bounded candidate training."""
    architectures = {'mnist': 'digitnet-plus-v2', 'cifar10': 'visionnet-wrn28-4-v2'}
    if dataset not in architectures:
        raise ValueError(f'Unknown dataset: {dataset}')
    return make_model(dataset, architectures[dataset])
