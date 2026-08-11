"""Thin wrapper owning the loss criterion for local SGD steps."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LocalTrainer:
    """Manages the criterion and executes local SGD steps.

    Parameters
    ----------
    criterion:
        Loss function (default: CrossEntropyLoss).
    """

    def __init__(self, criterion: Optional[nn.Module] = None) -> None:
        self.criterion = criterion or nn.CrossEntropyLoss()

    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Execute one gradient step and return the scalar loss."""
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = self.criterion(output, target)
        loss.backward()
        optimizer.step()
        return loss.item()
