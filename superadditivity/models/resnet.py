"""CIFAR-adapted ResNet.

Standard ResNet modified for 32x32 inputs: 3x3 stride-1 stem (no initial
max-pool), four residual stages (64/128/256/512 channels), global average
pooling, and a linear classifier.

References
----------
* He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
* He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """Two 3x3 conv residual block with identity shortcut.

    Attributes
    ----------
    expansion : int
        Channel expansion factor (always 1 for BasicBlock).
    """

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet(nn.Module):
    """ResNet adapted for CIFAR-sized (32x32) inputs.

    Uses a 3x3 stride-1 stem instead of the 7x7 stride-2 + max-pool used
    for ImageNet.  Four residual stages produce 64/128/256/512 channels,
    followed by global average pooling and a linear classifier.

    Parameters
    ----------
    block:
        Residual block class (e.g. :class:`BasicBlock`).
    num_blocks:
        Number of blocks per stage, e.g. ``[2, 2, 2, 2]`` for ResNet-18.
    num_classes:
        Number of output classes.

    Attributes
    ----------
    feature_dim : int
        Dimensionality of the penultimate representation (512).
    """

    feature_dim: int = 512

    def __init__(
        self,
        block: type[BasicBlock],
        num_blocks: List[int],
        num_classes: int = 100,
    ) -> None:
        super().__init__()
        self.in_planes: int = 64

        # 3x3 stride-1 stem — no max-pool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Build one residual stage."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim penultimate features (before the classifier).

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, 3, 32, 32)``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``(B, 512)``.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return out

    @staticmethod
    def get_layer_names() -> List[str]:
        """Return the canonical ordered list of stage/head names."""
        return ["layer1", "layer2", "layer3", "layer4", "fc"]


def build_resnet18_cifar(num_classes: int = 100) -> CIFARResNet:
    """Construct a CIFAR-adapted ResNet-18.

    Parameters
    ----------
    num_classes:
        Number of output classes.

    Returns
    -------
    CIFARResNet
        Uninitialized ResNet-18 model.
    """
    logger.info("Building CIFAR ResNet-18 with %d classes", num_classes)
    return CIFARResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def build_resnet34_cifar(num_classes: int = 100) -> CIFARResNet:
    """Construct a CIFAR-adapted ResNet-34.

    Parameters
    ----------
    num_classes:
        Number of output classes.

    Returns
    -------
    CIFARResNet
        Uninitialized ResNet-34 model.
    """
    logger.info("Building CIFAR ResNet-34 with %d classes", num_classes)
    return CIFARResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
