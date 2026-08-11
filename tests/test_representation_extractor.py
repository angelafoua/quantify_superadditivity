"""Tests for representation extraction."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from superadditivity.evaluation.representation_extractor import RepresentationExtractor


class _FakeClient:
    """Minimal client stub with a .model attribute."""
    def __init__(self, model: nn.Module):
        self.model = model


class TestRepresentationExtractor:
    def test_extract_single(self, small_model, device):
        """Test _extract_single extracts features from a model."""
        extractor = RepresentationExtractor(layer_names=["0"], device=str(device))
        data = torch.randn(10, 3, 32, 32)
        labels = torch.zeros(10, dtype=torch.long)
        probe_loader = DataLoader(TensorDataset(data, labels), batch_size=5)
        features = extractor._extract_single(small_model, probe_loader)
        assert isinstance(features, dict)
        assert "0" in features
        assert isinstance(features["0"], np.ndarray)

    def test_features_are_float64(self, small_model, device):
        extractor = RepresentationExtractor(layer_names=["0"], device=str(device))
        data = torch.randn(10, 3, 32, 32)
        labels = torch.zeros(10, dtype=torch.long)
        probe_loader = DataLoader(TensorDataset(data, labels), batch_size=5)
        features = extractor._extract_single(small_model, probe_loader)
        for name, feat in features.items():
            assert feat.dtype == np.float64

    def test_feature_batch_dimension(self, small_model, device):
        extractor = RepresentationExtractor(layer_names=["0"], device=str(device))
        n = 10
        data = torch.randn(n, 3, 32, 32)
        labels = torch.zeros(n, dtype=torch.long)
        probe_loader = DataLoader(TensorDataset(data, labels), batch_size=5)
        features = extractor._extract_single(small_model, probe_loader)
        for name, feat in features.items():
            assert feat.shape[0] == n

    def test_extract_from_clients(self, small_model, device):
        """Test extract() with a clients dict."""
        from superadditivity.models.model_utils import clone_model
        clients = {
            0: _FakeClient(clone_model(small_model)),
            1: _FakeClient(clone_model(small_model)),
        }
        extractor = RepresentationExtractor(layer_names=["0"], device=str(device))
        data = torch.randn(8, 3, 32, 32)
        labels = torch.zeros(8, dtype=torch.long)
        probe_loader = DataLoader(TensorDataset(data, labels), batch_size=4)
        result = extractor.extract(clients, probe_loader)
        assert "0" in result
        assert 0 in result["0"]
        assert 1 in result["0"]
        assert result["0"][0].shape[0] == 8
