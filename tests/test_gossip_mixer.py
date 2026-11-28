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

    def test_batchnorm_running_stats_not_mixed(self):
        """BatchNorm running_mean/running_var must stay local per client."""
        n = 3
        W = torch.ones(n, n, dtype=torch.float64) / n
        mixer = GossipMixer(mix_device="cpu")

        # Build small models with BatchNorm
        clients = []
        running_means_before = []
        for i in range(n):
            model = nn.Sequential(
                nn.Conv2d(1, 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(4),
            )
            # Set distinct running_mean per client
            with torch.no_grad():
                model[1].running_mean.fill_(float(i * 10))
            clients.append(_FakeClient(model))
            running_means_before.append(model[1].running_mean.clone())

        mixer.mix(clients, W)

        # Running means should be unchanged (not averaged)
        for i, c in enumerate(clients):
            np.testing.assert_allclose(
                c.model[1].running_mean.numpy(),
                running_means_before[i].numpy(),
                atol=1e-12,
                err_msg=f"Client {i} running_mean was mixed (should be local)",
            )


class TestBufferMixing:
    """mix_buffers=True gossips BatchNorm running stats with the same W."""

    @staticmethod
    def _clients(n=3):
        clients = []
        for i in range(n):
            model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4))
            with torch.no_grad():
                model[1].running_mean.fill_(float(i * 10))
                model[1].running_var.fill_(float(i + 1))
                model[1].num_batches_tracked.fill_(i + 5)
                model[0].weight.fill_(float(i))
            clients.append(_FakeClient(model))
        return clients

    @pytest.mark.parametrize("path", ["in_place", "to_new_dicts"])
    def test_float_buffers_mixed_integer_buffers_kept(self, path):
        n = 3
        W = torch.ones(n, n, dtype=torch.float64) / n
        mixer = GossipMixer(mix_device="cpu", mix_buffers=True)
        clients = self._clients(n)
        if path == "in_place":
            mixer.mix(clients, W)
        else:
            mixer._mix_to_new_dicts(clients, W, n)
        for i, c in enumerate(clients):
            np.testing.assert_allclose(c.model[1].running_mean.numpy(), 10.0)
            np.testing.assert_allclose(c.model[1].running_var.numpy(), 2.0)
            assert int(c.model[1].num_batches_tracked) == i + 5
            np.testing.assert_allclose(c.model[0].weight.detach().numpy(), 1.0)

    def test_default_path_to_new_dicts_keeps_buffers_local(self):
        n = 3
        W = torch.ones(n, n, dtype=torch.float64) / n
        clients = self._clients(n)
        GossipMixer(mix_device="cpu")._mix_to_new_dicts(clients, W, n)
        for i, c in enumerate(clients):
            np.testing.assert_allclose(c.model[1].running_mean.numpy(), float(i * 10))
            np.testing.assert_allclose(c.model[0].weight.detach().numpy(), 1.0)
