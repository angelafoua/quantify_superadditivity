"""Community-averaged checkpointing with retention policy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from superadditivity.utils.io import ensure_dir

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Save and load community-averaged model checkpoints.

    Parameters
    ----------
    output_dir:
        Directory for checkpoint files.
    community_assignments:
        Array mapping client index → community index.
    keep_last:
        Number of most recent checkpoints to keep (older ones are deleted).
    """

    def __init__(
        self,
        output_dir: str,
        community_assignments: np.ndarray,
        keep_last: int = 5,
    ) -> None:
        self.output_dir = ensure_dir(Path(output_dir))
        self.community_assignments = np.asarray(community_assignments)
        self.keep_last = keep_last
        self._saved_rounds: List[int] = []

    def save(
        self,
        state_dicts: List[Dict[str, torch.Tensor]],
        round_num: int,
    ) -> None:
        """Save community-averaged state dicts.

        For each community, averages the state dicts of its members and
        saves the result as a ``.pt`` file.
        """
        n_communities = int(self.community_assignments.max()) + 1

        for cid in range(n_communities):
            members = np.where(self.community_assignments == cid)[0]
            if len(members) == 0:
                continue

            avg_state = {}
            for key in state_dicts[members[0]]:
                tensors = [state_dicts[m][key].to(torch.float64) for m in members]
                avg_state[key] = (torch.stack(tensors).mean(dim=0)).to(torch.float32)

            path = self.output_dir / f"checkpoint_r{round_num:04d}_c{cid}.pt"
            torch.save(avg_state, path)

        self._saved_rounds.append(round_num)
        self._enforce_retention()
        logger.debug("Checkpoint saved for round %d", round_num)

    def _enforce_retention(self) -> None:
        """Delete checkpoints beyond the retention window."""
        if len(self._saved_rounds) <= self.keep_last:
            return

        to_delete = self._saved_rounds[:-self.keep_last]
        for r in to_delete:
            for path in self.output_dir.glob(f"checkpoint_r{r:04d}_c*.pt"):
                path.unlink(missing_ok=True)
                logger.debug("Deleted old checkpoint: %s", path)

        self._saved_rounds = self._saved_rounds[-self.keep_last:]

    def load(
        self,
        round_num: int,
        community_id: int,
    ) -> Dict[str, torch.Tensor]:
        """Load a community-averaged checkpoint."""
        path = self.output_dir / f"checkpoint_r{round_num:04d}_c{community_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)
