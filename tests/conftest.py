"""Shared test fixtures — all tests run on CPU without dataset downloads."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset


@pytest.fixture
def small_n_clients() -> int:
    return 8


@pytest.fixture
def small_n_communities() -> int:
    return 2


@pytest.fixture
def synthetic_dataset() -> TensorDataset:
    """A small synthetic image dataset for testing (no download required)."""
    n_samples = 200
    images = torch.randn(n_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (n_samples,))
    return TensorDataset(images, labels)


@pytest.fixture
def synthetic_targets() -> np.ndarray:
    """Synthetic targets matching the synthetic dataset."""
    return np.random.randint(0, 10, size=200)


@pytest.fixture
def synthetic_features() -> np.ndarray:
    """Synthetic feature matrix for metric tests."""
    return np.random.randn(50, 64).astype(np.float64)


@pytest.fixture
def small_model() -> torch.nn.Module:
    """A minimal model for testing (not a full ResNet)."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 10),
    )


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")
