"""Community-averaged checkpointing with retention policy and full training-state resume."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from superadditivity.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def _config_hash(config: dict) -> str:
    """Deterministic SHA-256 of a JSON-serialisable config dict.

    Used to detect incompatible configuration changes between the run that
    saved the checkpoint and the run that tries to resume from it.
    """
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


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


class TrainingCheckpointManager:
    """Full training-state checkpoint manager for mid-run resume.

    Saves **per-client** model and optimizer states, the current round,
    RNG states, drift-tracker state, and a config hash so that an
    interrupted run can be resumed exactly or safely rejected when the
    configuration has changed.

    Parameters
    ----------
    output_dir:
        Directory for resume checkpoint files.
    config_dict:
        The resolved Hydra config dict, used to compute a compatibility hash.
    keep_last:
        Number of most recent resume checkpoints to keep.
    """

    FILENAME_PREFIX = "resume_ckpt"

    def __init__(
        self,
        output_dir: str,
        config_dict: Optional[dict] = None,
        keep_last: int = 2,
    ) -> None:
        self.output_dir = ensure_dir(Path(output_dir))
        self._config_hash = _config_hash(config_dict) if config_dict else ""
        self.keep_last = keep_last
        self._saved_rounds: List[int] = []

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_training_state(
        self,
        clients: list,
        round_num: int,
        history: Dict[str, list],
        drift_tracker_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist everything needed to resume training from ``round_num``.

        Writes to a temporary file first, then renames atomically.

        Parameters
        ----------
        clients:
            List of :class:`DecentralizedClient` instances.
        round_num:
            The round that was just completed (0-indexed).
        history:
            Coordinator history dict accumulated so far.
        drift_tracker_state:
            Output of ``DriftTracker.get_state()``.

        Returns
        -------
        Path
            Path of the saved checkpoint.
        """
        client_states = []
        for c in clients:
            client_states.append({
                "model": {k: v.detach().cpu().clone()
                          for k, v in c.model.state_dict().items()},
                "optimizer": c.optimizer.state_dict(),
            })

        # Save per-client DataLoader generator state for exact data-order
        # reproducibility across resume.  The generator may not exist if the
        # client's loader has not been initialised yet (first checkpoint
        # before any local_step).
        dl_gen_states = []
        for c in clients:
            loader = getattr(c, "_loader", None)
            if loader is not None and hasattr(loader, "generator") and loader.generator is not None:
                dl_gen_states.append(loader.generator.get_state())
            else:
                dl_gen_states.append(None)

        payload = {
            "round_num": round_num,
            "config_hash": self._config_hash,
            "n_clients": len(clients),
            "client_states": client_states,
            "history": history,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch_cpu": torch.random.get_rng_state(),
            "dataloader_generator_states": dl_gen_states,
        }
        if torch.cuda.is_available():
            payload["rng_torch_cuda"] = [
                torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
            ]
        if drift_tracker_state is not None:
            payload["drift_tracker_state"] = drift_tracker_state

        target = self.output_dir / f"{self.FILENAME_PREFIX}_r{round_num:04d}.pt"
        tmp = target.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(target)

        self._saved_rounds.append(round_num)
        self._enforce_retention()
        logger.info("Resume checkpoint saved: round %d → %s", round_num, target)
        return target

    # ------------------------------------------------------------------
    # Load / discover
    # ------------------------------------------------------------------

    def find_latest(self) -> Optional[Path]:
        """Return the path of the newest valid, config-compatible checkpoint.

        Iterates from newest to oldest.  Corrupt files are skipped (with a
        warning).  The **first** structurally valid checkpoint is tested for
        config compatibility; if it mismatches, a ``RuntimeError`` is raised
        immediately — older checkpoints are **not** tried because all
        checkpoints within a single run share the same config hash, so a
        mismatch means the *config* changed, not the checkpoint.

        If corrupt checkpoints are skipped and an older valid one is found,
        a warning notes the progress that will be replayed.
        """
        candidates = sorted(
            self.output_dir.glob(f"{self.FILENAME_PREFIX}_r*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        skipped_corrupt: List[Path] = []
        for path in candidates:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                if "round_num" not in ckpt or "client_states" not in ckpt:
                    skipped_corrupt.append(path)
                    logger.warning(
                        "Skipping structurally invalid checkpoint: %s", path,
                    )
                    continue
                # Config compatibility — fail fast, don't fall through.
                saved_hash = ckpt.get("config_hash", "")
                if (
                    self._config_hash
                    and saved_hash
                    and saved_hash != self._config_hash
                ):
                    raise RuntimeError(
                        f"Configuration changed since checkpoint was saved "
                        f"(saved hash={saved_hash}, current hash="
                        f"{self._config_hash}). Delete the checkpoint "
                        f"directory to start fresh, or restore the original "
                        f"configuration."
                    )
                if skipped_corrupt:
                    logger.warning(
                        "Falling back to older checkpoint %s (skipped %d "
                        "corrupt: %s). Rounds %d+ will be replayed.",
                        path.name,
                        len(skipped_corrupt),
                        [p.name for p in skipped_corrupt],
                        ckpt["round_num"] + 1,
                    )
                return path
            except RuntimeError:
                raise  # config mismatch — propagate
            except Exception as exc:
                skipped_corrupt.append(path)
                logger.warning("Skipping corrupt checkpoint %s: %s", path, exc)
        return None

    def has_any_checkpoint_files(self) -> bool:
        """Return True if the checkpoint directory contains any ``.pt`` files."""
        return any(self.output_dir.glob(f"{self.FILENAME_PREFIX}_r*.pt"))

    def load_training_state(
        self,
        path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Load a resume checkpoint and validate config compatibility.

        Parameters
        ----------
        path:
            Explicit checkpoint path; if None the latest is used.

        Returns
        -------
        Dict[str, Any]
            The full checkpoint payload.

        Raises
        ------
        FileNotFoundError
            No checkpoint found.
        RuntimeError
            Config hash mismatch (incompatible configuration).
        """
        if path is None:
            path = self.find_latest()
        if path is None:
            raise FileNotFoundError("No resume checkpoint found.")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        saved_hash = ckpt.get("config_hash", "")
        if self._config_hash and saved_hash and saved_hash != self._config_hash:
            raise RuntimeError(
                f"Configuration changed since the checkpoint was saved "
                f"(saved hash={saved_hash}, current hash={self._config_hash}). "
                f"Delete the checkpoint directory to start fresh, or restore "
                f"the original configuration."
            )

        logger.info(
            "Loaded resume checkpoint: round %d from %s",
            ckpt["round_num"], path,
        )
        return ckpt

    def restore_clients(
        self,
        clients: list,
        ckpt: Dict[str, Any],
    ) -> None:
        """Restore per-client model and optimizer states from *ckpt*.

        Parameters
        ----------
        clients:
            The list of :class:`DecentralizedClient` whose state will be
            overwritten.
        ckpt:
            Checkpoint payload returned by :meth:`load_training_state`.

        Raises
        ------
        RuntimeError
            Client count mismatch.
        """
        if ckpt["n_clients"] != len(clients):
            raise RuntimeError(
                f"Client count mismatch: checkpoint has {ckpt['n_clients']} "
                f"but the current run has {len(clients)}."
            )
        for i, c in enumerate(clients):
            c.model.load_state_dict(ckpt["client_states"][i]["model"])
            c.optimizer.load_state_dict(ckpt["client_states"][i]["optimizer"])
        logger.info("Restored %d client states.", len(clients))

    @staticmethod
    def restore_rng(ckpt: Dict[str, Any]) -> None:
        """Restore Python, NumPy, and PyTorch RNG states from *ckpt*."""
        random.setstate(ckpt["rng_python"])
        np.random.set_state(ckpt["rng_numpy"])
        torch.random.set_rng_state(ckpt["rng_torch_cpu"])
        if "rng_torch_cuda" in ckpt and torch.cuda.is_available():
            for i, state in enumerate(ckpt["rng_torch_cuda"]):
                torch.cuda.set_rng_state(state, i)
        logger.info("RNG states restored.")

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def _enforce_retention(self) -> None:
        if len(self._saved_rounds) <= self.keep_last:
            return
        to_delete = self._saved_rounds[:-self.keep_last]
        for r in to_delete:
            path = self.output_dir / f"{self.FILENAME_PREFIX}_r{r:04d}.pt"
            path.unlink(missing_ok=True)
            logger.debug("Deleted old resume checkpoint: %s", path)
        self._saved_rounds = self._saved_rounds[-self.keep_last:]
