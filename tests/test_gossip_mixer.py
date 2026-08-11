"""Tests for gossip mixing correctness."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from superadditivity.communication.gossip_mixer import GossipMixer


class _FakeClient:
    """Minimal client stub with a .model attribute."""
    def __init__(self, model: nn.Module):
        self.model = model


def _make_clients_with_values(values: list) -> list:
    """Create clients whose single-param models hold the given float values."""
    clients = []
    for v in values:
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(v)
        clients.append(_FakeClient(model))
    return clients


class TestGossipMixer:
    def test_uniform_mixing_averages(self):
        """With uniform W, mixing should produce the average."""
        n = 4
        W = torch.ones(n, n, dtype=torch.float64) / n
        mixer = GossipMixer(mix_device="cpu")
        clients = _make_clients_with_values([0.0, 1.0, 2.0, 3.0])

        mixer.mix(clients, W)

        expected_mean = 1.5
        for c in clients:
            val = c.model.weight.item()
            np.testing.assert_allclose(val, expected_mean, atol=1e-5)

    def test_identity_mixing_preserves(self):
        """With identity W, mixing should not change anything."""
        n = 4
        W = torch.eye(n, dtype=torch.float64)
        mixer = GossipMixer(mix_device="cpu")
        clients = _make_clients_with_values([0.0, 1.0, 2.0, 3.0])

        mixer.mix(clients, W)

        for i, c in enumerate(clients):
            np.testing.assert_allclose(c.model.weight.item(), float(i), atol=1e-12)

    def test_mixing_preserves_global_mean(self):
        """Mixing with a doubly-stochastic W preserves the global mean."""
        n = 4
        W = torch.tensor([
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
        ], dtype=torch.float64)
        mixer = GossipMixer(mix_device="cpu")
        clients = _make_clients_with_values([0.0, 1.0, 2.0, 3.0])
        global_mean_before = 1.5

        mixer.mix(clients, W)

        vals = [c.model.weight.item() for c in clients]
        global_mean_after = np.mean(vals)
        np.testing.assert_allclose(global_mean_before, global_mean_after, atol=1e-10)
