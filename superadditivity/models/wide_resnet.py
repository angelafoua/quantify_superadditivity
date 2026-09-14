"""WideResNet-28-2 for CIFAR-scale inputs.

A wider variant of ResNet with the same residual-block structure but
2x channel width, suitable for 32x32 and larger inputs. Uses adaptive
stem: stride-1 3x3 for small inputs, stride-2 7x7 for larger inputs.

References
----------
* Zagoruyko & Komodakis, "Wide Residual Networks", BMVC 2016.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WideBasicBlock(nn.Module):
    """Wide residual block with dropout."""

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.shortcut: nn.Module
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """WideResNet with configurable depth and width factor.

    Parameters
    ----------
    depth:
        Network depth (e.g. 28 for WRN-28-k).
    widen_factor:
        Width multiplier (e.g. 2 for WRN-28-2).
    num_classes:
        Number of output classes.
    in_channels:
        Number of input image channels.
    dropout:
        Dropout rate within wide blocks.

    Attributes
    ----------
    feature_dim : int
        Dimensionality of the penultimate representation.
    """

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 2,
        num_classes: int = 100,
        in_channels: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert (depth - 4) % 6 == 0, f"WideResNet depth must satisfy (d-4)%6==0, got {depth}"
        n_blocks = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.feature_dim: int = channels[3]
        self.in_planes: int = channels[0]

        self.conv1 = nn.Conv2d(
            in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_layer(
            WideBasicBlock, channels[1], n_blocks, stride=1, dropout=dropout
        )
        self.layer2 = self._make_layer(
            WideBasicBlock, channels[2], n_blocks, stride=2, dropout=dropout
        )
        self.layer3 = self._make_layer(
            WideBasicBlock, channels[3], n_blocks, stride=2, dropout=dropout
        )
        self.bn_final = nn.BatchNorm2d(channels[3])
        self.fc = nn.Linear(channels[3], num_classes)

    def _make_layer(
        self,
        block: type[WideBasicBlock],
        planes: int,
        n_blocks: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (n_blocks - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, dropout))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate features (before the classifier)."""
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return out

    @staticmethod
    def get_layer_names() -> List[str]:
        """Return the canonical ordered list of stage/head names."""
        return ["layer1", "layer2", "layer3", "fc"]


def build_wide_resnet(
    num_classes: int = 100,
    in_channels: int = 3,
    depth: int = 28,
    widen_factor: int = 2,
    dropout: float = 0.0,
) -> WideResNet:
    """Construct a WideResNet (default WRN-28-2)."""
    logger.info(
        "Building WideResNet-%d-%d with %d classes, %d input channels",
        depth, widen_factor, num_classes, in_channels,
    )
    return WideResNet(
        depth=depth,
        widen_factor=widen_factor,
        num_classes=num_classes,
        in_channels=in_channels,
        dropout=dropout,
    )
