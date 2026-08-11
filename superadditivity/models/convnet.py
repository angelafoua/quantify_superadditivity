"""Simple 4-layer ConvNet for federated learning benchmarks.

A lightweight convolutional network with four stages of
Conv2d + BatchNorm + ReLU + spatial reduction, followed by a linear
classifier. Suitable for CIFAR-10/100 and similar 32x32 image tasks.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SimpleConvNet(nn.Module):
    """4-layer ConvNet for FL benchmarks.

    Architecture
    ------------
    * layer1: Conv(in_channels, 32) + BN + ReLU + MaxPool(2)
    * layer2: Conv(32, 64) + BN + ReLU + MaxPool(2)
    * layer3: Conv(64, 128) + BN + ReLU + MaxPool(2)
    * layer4: Conv(128, 256) + BN + ReLU + AdaptiveAvgPool(1)
    * fc: Linear(256, num_classes)

    Each layer is an :class:`nn.Sequential` so that forward hooks can be
    registered on individual stages.

    Parameters
    ----------
    num_classes:
        Number of output classes.
    in_channels:
        Number of input image channels (3 for RGB, 1 for grayscale).

    Attributes
    ----------
    feature_dim : int
        Dimensionality of the penultimate representation (256).
    """

    feature_dim: int = 256

    def __init__(self, num_classes: int = 100, in_channels: int = 3) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 256-dim penultimate features (before the classifier).

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``(B, 256)``.
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        return out

    @staticmethod
    def get_layer_names() -> List[str]:
        """Return the canonical ordered list of stage/head names."""
        return ["layer1", "layer2", "layer3", "layer4", "fc"]


def build_convnet(num_classes: int = 100, in_channels: int = 3) -> SimpleConvNet:
    """Construct a 4-layer ConvNet.

    Parameters
    ----------
    num_classes:
        Number of output classes.
    in_channels:
        Number of input image channels.

    Returns
    -------
    SimpleConvNet
        Uninitialized SimpleConvNet model.
    """
    logger.info(
        "Building SimpleConvNet with %d classes, %d input channels",
        num_classes,
        in_channels,
    )
    return SimpleConvNet(num_classes=num_classes, in_channels=in_channels)
