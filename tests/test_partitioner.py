"""Tests for data partitioning."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from superadditivity.datasets.semantic_partitioner import SemanticPartitioner
from superadditivity.datasets.quantity_skew_partitioner import QuantitySkewPartitioner


class _SyntheticDataset(Dataset):
    """Minimal dataset with a .targets attribute for partitioner tests."""

    def __init__(self, n_samples: int = 200, n_classes: int = 10, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.data = torch.randn(n_samples, 3, 32, 32)
        self.targets = list(rng.randint(0, n_classes, size=n_samples))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TestSemanticPartitioner:
    def _make_community_assignments(self, n_clients: int, n_communities: int):
        """Build a community_assignments dict: {comm_id: [client_ids]}."""
        assignments = {}
        clients_per = n_clients // n_communities
        for c in range(n_communities):
            start = c * clients_per
            end = start + clients_per if c < n_communities - 1 else n_clients
            assignments[c] = list(range(start, end))
        return assignments

    def test_iid_partition_covers_all_samples(self):
        n_clients = 8
        n_communities = 2
        ds = _SyntheticDataset(200, 10, seed=0)
        comm = self._make_community_assignments(n_clients, n_communities)
        part = SemanticPartitioner(
            train_dataset=ds,
            semantic_clusters=None,
            community_assignments=comm,
            alpha=1.0,
            mode="iid",
            run_seed=42,
            num_clients=n_clients,
        )
        result = part.partition()
        all_indices = []
        for indices in result.values():
            all_indices.extend(indices)
        assert len(set(all_indices)) == 200

    def test_dirichlet_partition_covers_all_samples(self):
        n_clients = 8
        n_communities = 2
        ds = _SyntheticDataset(200, 10, seed=0)
        comm = self._make_community_assignments(n_clients, n_communities)
        part = SemanticPartitioner(
            train_dataset=ds,
            semantic_clusters=None,
            community_assignments=comm,
            alpha=0.5,
            mode="dirichlet_semantic",
            run_seed=42,
            num_clients=n_clients,
        )
        result = part.partition()
        all_indices = []
        for indices in result.values():
            all_indices.extend(indices)
        assert len(all_indices) >= 200 * 0.9

    def test_n_clients_correct(self):
        n_clients = 8
        n_communities = 2
        ds = _SyntheticDataset(200, 10, seed=0)
        comm = self._make_community_assignments(n_clients, n_communities)
        part = SemanticPartitioner(
            train_dataset=ds,
            semantic_clusters=None,
            community_assignments=comm,
            alpha=1.0,
            mode="iid",
            run_seed=42,
            num_clients=n_clients,
        )
        result = part.partition()
        assert len(result) == n_clients

    def test_invalid_mode_raises(self):
        n_clients = 8
        ds = _SyntheticDataset(200, 10, seed=0)
        comm = {0: list(range(4)), 1: list(range(4, 8))}
        with pytest.raises(ValueError, match="Unknown partitioner mode"):
            SemanticPartitioner(
                train_dataset=ds,
                semantic_clusters=None,
                community_assignments=comm,
                alpha=1.0,
                mode="bad_mode",
                run_seed=42,
                num_clients=n_clients,
            )


class TestQuantitySkewPartitioner:
    def test_all_clients_get_data(self, synthetic_targets):
        n_clients = 8
        part = QuantitySkewPartitioner(
            n_clients=n_clients,
            alpha=1.0,
            min_samples=2,
        )
        result = part.partition(synthetic_targets, n_communities=2, seed=42)
        for ci in result["client_indices"]:
            assert len(ci) >= 2

    def test_community_assignments(self, synthetic_targets):
        n_clients = 8
        part = QuantitySkewPartitioner(n_clients=n_clients)
        result = part.partition(synthetic_targets, n_communities=2, seed=42)
        assert len(result["community_assignments"]) == n_clients
