"""Per-client dataset with deterministic per-epoch shuffling."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from superadditivity.utils.seed import derived_seed

logger = logging.getLogger(__name__)


class ClientDataset(Dataset):
    """Wraps a subset of a base dataset with deterministic per-epoch shuffling.

    Parameters
    ----------
    base_dataset:
        The full training dataset (e.g. CIFAR-100).
    indices:
        Indices into ``base_dataset`` assigned to this client.
    client_id:
        Numeric client identifier (used for seed derivation).
    run_seed:
        Master run seed for reproducibility.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices: np.ndarray,
        client_id: int,
        run_seed: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.client_id = client_id
        self.run_seed = run_seed
        self._current_round: int = 0
        self._shuffled_indices: Optional[np.ndarray] = None
        self._shuffle_for_round(0)

    def set_round(self, round_num: int) -> None:
        """Update the communication round, triggering a deterministic re-shuffle."""
        if round_num != self._current_round:
            self._current_round = round_num
            self._shuffle_for_round(round_num)

    def _shuffle_for_round(self, round_num: int) -> None:
        seed = derived_seed(
            self.run_seed, kind="shuffle",
            client_id=self.client_id, round_num=round_num,
        )
        rng = np.random.RandomState(seed)
        self._shuffled_indices = rng.permutation(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self._shuffled_indices[idx])
        return self.base_dataset[real_idx]
