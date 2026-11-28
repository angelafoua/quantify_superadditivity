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
        # Every sample exactly once (no duplication across communities).
        assert sorted(all_indices) == list(range(200))

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


class TestBetweenCommunityDirichlet:
    """alpha must control how different the communities' label mixes are."""

    CLUSTERS = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11]}

    def _partitioner(self, alpha, between="dirichlet", clusters=CLUSTERS, n_classes=12):
        targets = np.repeat(np.arange(n_classes), 400)
        ds = _SyntheticDataset(len(targets), n_classes)
        ds.targets = list(targets)
        comm = {c: list(range(4 * c, 4 * c + 4)) for c in range(4)}
        return SemanticPartitioner(
            ds, clusters, comm, alpha, "dirichlet_semantic", 42, 16,
            between_community=between,
        )

    @pytest.mark.parametrize("alpha", [100.0, 1.0, 0.1])
    def test_each_sample_assigned_exactly_once(self, alpha):
        p = self._partitioner(alpha)
        result = p.partition()
        all_idx = sorted(i for v in result.values() for i in v)
        assert all_idx == list(range(len(p.targets)))

    def test_label_distance_decreases_with_alpha(self):
        tv = [
            self._partitioner(a).community_label_distance(self._partitioner(a).partition())
            for a in (0.05, 0.5, 10.0, 1000.0)
        ]
        assert tv == sorted(tv, reverse=True)
        assert tv[0] > 0.8 and tv[-1] < 0.05

    def test_home_community_gets_largest_share(self):
        p = self._partitioner(0.5)
        result = p.partition()
        for comm, classes in self.CLUSTERS.items():
            for cls in classes:
                per_comm = [
                    sum(int(p.targets[i] == cls)
                        for cid in range(4 * c, 4 * c + 4) for i in result[cid])
                    for c in range(4)
                ]
                assert int(np.argmax(per_comm)) == comm

    def test_legacy_disjoint_is_constant_in_alpha(self):
        tv = [
            self._partitioner(a, "disjoint").community_label_distance(
                self._partitioner(a, "disjoint").partition()
            )
            for a in (0.1, 10.0)
        ]
        # ~1.0 (an empty client may borrow one sample from another community)
        assert tv[0] == pytest.approx(1.0, abs=1e-3) and tv[1] == pytest.approx(1.0, abs=1e-3)

    def test_shared_pools_do_not_duplicate(self):
        """3 clusters for 4 communities (PathMNIST-like) must not duplicate data."""
        clusters = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
        p = self._partitioner(0.5, clusters=clusters, n_classes=9)
        result = p.partition()
        all_idx = [i for v in result.values() for i in v]
        assert len(all_idx) == len(set(all_idx)) == len(p.targets)

    def test_split_counts_sum_and_rounding(self):
        counts = SemanticPartitioner._split_counts(10, np.array([0.34, 0.33, 0.33]))
        assert counts.sum() == 10 and list(counts) == [4, 3, 3]

    def test_invalid_between_mode(self):
        with pytest.raises(ValueError, match="between_community"):
            self._partitioner(1.0, between="bogus")


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
