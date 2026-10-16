"""Candidate CNNs trained from scratch; architecture identifiers are checkpointed."""
from torch import nn


class DigitNetPlus(nn.Module):
    """Six small convolutions with a compact spatial classification head."""
    def __init__(self):
        super().__init__()
        layers = []
        previous = 1
        for width in (32, 64, 128):
            for _ in range(2):
                layers.extend([nn.Conv2d(previous, width, 3, padding=1, bias=False),
                               nn.BatchNorm2d(width), nn.ReLU()])
                previous = width
            if width < 128:
                layers.extend([nn.MaxPool2d(2), nn.Dropout2d(0.10)])
        # 7x7 -> 3x3 with fixed pooling: deterministic CUDA backward is supported.
        layers.append(nn.AvgPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 3 * 3, 128),
                                        nn.ReLU(), nn.Dropout(0.30), nn.Linear(128, 10))

    def forward(self, images):
        return self.classifier(self.features(images))


class WidePreactivationBlock(nn.Module):
    """Preactivation residual block following the Wide ResNet design principle."""
    def __init__(self, incoming, outgoing, stride=1, dropout=0.15):
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
    """Custom implementation of a WRN-28-4 style CIFAR network, not pretrained."""
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
    def _stage(incoming, outgoing, stride):
        return nn.Sequential(WidePreactivationBlock(incoming, outgoing, stride),
                             *(WidePreactivationBlock(outgoing, outgoing) for _ in range(3)))

    def forward(self, images):
        features = self.stage3(self.stage2(self.stage1(self.stem(images))))
        return self.head(self.pool(features))


def make_candidate(dataset):
    if dataset == 'mnist':
        return DigitNetPlus()
    if dataset == 'cifar10':
        return VisionNetWide()
    raise ValueError(dataset)
