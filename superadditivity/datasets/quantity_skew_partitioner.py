"""Quantity-skew partitioner — heterogeneity through sample-count imbalance only.

This serves as a negative control: labels are distributed IID, but the
*number* of samples per client varies via a Dirichlet draw on quantities.
If superadditivity is driven by label imbalance (not just dataset-size
differences), quantity skew should yield near-zero interaction I.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class QuantitySkewPartitioner:
    """Partition data IID by labels but with Dirichlet-skewed sample counts.

    Parameters
    ----------
    n_clients:
        Total number of clients.
    alpha:
        Dirichlet concentration for *quantity* allocation.
        Lower alpha → more imbalance in sample counts.
    min_samples:
        Minimum samples any client must receive.
    """

    def __init__(
        self,
        n_clients: int = 128,
        alpha: float = 1.0,
        min_samples: int = 10,
    ) -> None:
        self.n_clients = n_clients
        self.alpha = alpha
        self.min_samples = min_samples

    def partition(
        self,
        targets: np.ndarray,
        n_communities: int = 4,
        seed: Optional[int] = None,
    ) -> dict:
        """Assign samples to clients with IID labels but skewed quantities.

        Parameters
        ----------
        targets:
            Integer label array of shape ``(N,)``.
        n_communities:
            Number of communities (clients are split evenly across them).
        seed:
            Random seed for reproducibility.

        Returns
        -------
        dict with keys:
            ``"client_indices"`` — list of arrays, one per client.
            ``"community_assignments"`` — array mapping client → community.
            ``"samples_per_client"`` — array of sample counts.
        """
        rng = np.random.RandomState(seed)
        n_total = len(targets)

        proportions = rng.dirichlet(np.full(self.n_clients, self.alpha))
        raw_counts = (proportions * n_total).astype(np.int64)
        raw_counts = np.maximum(raw_counts, self.min_samples)

        total_assigned = raw_counts.sum()
        if total_assigned > n_total:
            excess = total_assigned - n_total
            sorted_idx = np.argsort(-raw_counts)
            for i in sorted_idx:
                can_remove = raw_counts[i] - self.min_samples
                remove = min(can_remove, excess)
                raw_counts[i] -= remove
                excess -= remove
                if excess <= 0:
                    break

        all_indices = rng.permutation(n_total)

        client_indices = []
        offset = 0
        for count in raw_counts:
            count = int(count)
            end = min(offset + count, n_total)
            client_indices.append(all_indices[offset:end])
            offset = end

        if offset < n_total:
            leftover = all_indices[offset:]
            for i, idx in enumerate(leftover):
                cid = i % self.n_clients
                client_indices[cid] = np.concatenate(
                    [client_indices[cid], [idx]]
                )

        clients_per_community = self.n_clients // n_communities
        community_assignments = np.zeros(self.n_clients, dtype=np.int64)
        for c in range(n_communities):
            start = c * clients_per_community
            end = start + clients_per_community if c < n_communities - 1 else self.n_clients
            community_assignments[start:end] = c

        samples_per_client = np.array([len(ci) for ci in client_indices])

        logger.info(
            "Quantity-skew partition: %d clients, alpha=%.2f, "
            "samples range [%d, %d], mean=%.1f",
            self.n_clients, self.alpha,
            samples_per_client.min(), samples_per_client.max(),
            samples_per_client.mean(),
        )

        return {
            "client_indices": client_indices,
            "community_assignments": community_assignments,
            "samples_per_client": samples_per_client,
        }
