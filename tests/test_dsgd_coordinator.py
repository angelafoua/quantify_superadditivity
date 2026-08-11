"""Tests for D-SGD coordinator."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from superadditivity.datasets.client_dataset import ClientDataset
from superadditivity.models.model_utils import clone_model
from superadditivity.training.decentralized_client import DecentralizedClient
from superadditivity.training.dsgd_coordinator import DSGDCoordinator
from superadditivity.training.lr_schedule import CosineDecaySchedule


class TestDSGDCoordinator:
    def test_runs_without_error(self, small_model, synthetic_dataset, device):
        n_clients = 4
        base_model = small_model

        clients = []
        for cid in range(n_clients):
            indices = np.arange(cid * 50, (cid + 1) * 50)
            dataset = ClientDataset(synthetic_dataset, indices, cid, run_seed=42)
            model = clone_model(base_model)
            client = DecentralizedClient(
                client_id=cid, model=model, dataset=dataset,
                device=device, batch_size=16, local_steps=1,
            )
            clients.append(client)

        W = np.ones((n_clients, n_clients), dtype=np.float64) / n_clients
        lr_sched = CosineDecaySchedule(lr_max=0.01, total_rounds=3)

        coordinator = DSGDCoordinator(
            clients=clients, mixing_matrix=W,
            lr_schedule=lr_sched, total_rounds=3,
            eval_every=10, checkpoint_every=10,
        )
        history = coordinator.run()

        assert len(history["round"]) == 3
        assert all(isinstance(l, float) for l in history["mean_loss"])

    def test_loss_decreases_over_rounds(self, small_model, synthetic_dataset, device):
        n_clients = 2
        clients = []
        for cid in range(n_clients):
            indices = np.arange(cid * 100, (cid + 1) * 100)
            dataset = ClientDataset(synthetic_dataset, indices, cid, run_seed=42)
            model = clone_model(small_model)
            client = DecentralizedClient(
                client_id=cid, model=model, dataset=dataset,
                device=device, batch_size=32, local_steps=2,
            )
            clients.append(client)

        W = np.ones((n_clients, n_clients), dtype=np.float64) / n_clients
        lr_sched = CosineDecaySchedule(lr_max=0.01, total_rounds=10)

        coordinator = DSGDCoordinator(
            clients=clients, mixing_matrix=W,
            lr_schedule=lr_sched, total_rounds=10,
        )
        history = coordinator.run()
        assert history["mean_loss"][-1] <= history["mean_loss"][0]
