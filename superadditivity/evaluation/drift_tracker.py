"""Temporal metric storage for tracking representation drift over rounds.

Accumulates per-round scalar and matrix metrics, optional raw embeddings,
and provides serialisation to HDF5 for post-hoc analysis.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from superadditivity.utils.io import write_hdf5_dataset, write_hdf5_attrs

logger = logging.getLogger(__name__)


class DriftTracker:
    """Track evaluation metrics across training rounds.

    Records scalar and array metrics at each round, and optionally stores
    raw embedding snapshots at selected rounds.

    Examples
    --------
    >>> tracker = DriftTracker()
    >>> tracker.record(0, {"cka_mean": 0.95, "fisher": 1.2})
    >>> tracker.record(1, {"cka_mean": 0.87, "fisher": 2.1})
    >>> trajectory = tracker.get_trajectory("cka_mean")
    >>> tracker.save("results/metrics.h5")
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, List[Any]] = {}
        self._rounds: Dict[str, List[int]] = {}
        self._embeddings: Dict[int, Dict[str, Any]] = {}
        self._recorded_keys: Set[str] = set()

    def record(self, round_num: int, metrics: Dict[str, Any]) -> None:
        """Record metrics for a single round.

        Parameters
        ----------
        round_num:
            Training round number.
        metrics:
            Mapping from metric name to value (scalar, ndarray, etc.).
        """
        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
                self._rounds[key] = []
            self._metrics[key].append(
                np.asarray(value, dtype=np.float64)
                if isinstance(value, (np.ndarray, list))
                else value
            )
            self._rounds[key].append(round_num)
            self._recorded_keys.add(key)

        logger.debug("Recorded %d metrics at round %d", len(metrics), round_num)

    def record_embeddings(
        self,
        round_num: int,
        layer_embeddings: Dict[str, Dict[int, np.ndarray]],
    ) -> None:
        """Store raw embedding snapshots for a round.

        Use sparingly -- raw embeddings consume significant memory. Intended
        for post-hoc trajectory analysis at selected checkpoints.

        Parameters
        ----------
        round_num:
            Training round number.
        layer_embeddings:
            Nested mapping: ``layer_name -> community_id -> embeddings``.
        """
        snapshot: Dict[str, Any] = {}
        for layer_name, comm_embs in layer_embeddings.items():
            snapshot[layer_name] = {
                cid: np.asarray(emb, dtype=np.float64)
                for cid, emb in comm_embs.items()
            }
        self._embeddings[round_num] = snapshot
        logger.debug("Stored embedding snapshot at round %d", round_num)

    def get_trajectory(self, metric_key: str) -> np.ndarray:
        """Get the trajectory of a metric across all recorded rounds.

        Parameters
        ----------
        metric_key:
            Name of the metric.

        Returns
        -------
        np.ndarray
            Stacked array of shape ``(n_rounds, ...)`` where the trailing
            dimensions depend on the metric type.

        Raises
        ------
        KeyError
            If the metric was never recorded.
        """
        if metric_key not in self._metrics:
            raise KeyError(f"Metric {metric_key!r} not found. "
                           f"Available: {sorted(self._recorded_keys)}")
        return np.array(self._metrics[metric_key])

    def get_rounds(self, metric_key: str) -> np.ndarray:
        """Get the round numbers at which a metric was recorded.

        Parameters
        ----------
        metric_key:
            Name of the metric.

        Returns
        -------
        np.ndarray
            1-D array of round numbers.

        Raises
        ------
        KeyError
            If the metric was never recorded.
        """
        if metric_key not in self._rounds:
            raise KeyError(f"Metric {metric_key!r} not found. "
                           f"Available: {sorted(self._recorded_keys)}")
        return np.array(self._rounds[metric_key])

    def get_state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of all tracked state.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing metrics, rounds, and embeddings.
            Can be passed to :meth:`load_state` to restore.
        """
        return {
            "metrics": copy.deepcopy(self._metrics),
            "rounds": copy.deepcopy(self._rounds),
            "embeddings": copy.deepcopy(self._embeddings),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore state from a previous :meth:`get_state` call.

        Parameters
        ----------
        state:
            State dictionary previously produced by :meth:`get_state`.
        """
        self._metrics = copy.deepcopy(state["metrics"])
        self._rounds = copy.deepcopy(state["rounds"])
        self._embeddings = copy.deepcopy(state.get("embeddings", {}))
        self._recorded_keys = set(self._metrics.keys())
        logger.info(
            "Loaded state with %d metric keys and %d embedding snapshots",
            len(self._recorded_keys),
            len(self._embeddings),
        )

    def save(self, path: str) -> None:
        """Persist all tracked metrics and embeddings to an HDF5 file.

        Parameters
        ----------
        path:
            File path for the output HDF5 file.
        """
        logger.info("Saving drift tracker to %s", path)

        # Save scalar / array metrics
        for key in sorted(self._recorded_keys):
            trajectory = self.get_trajectory(key)
            rounds = self.get_rounds(key)
            write_hdf5_dataset(
                path,
                f"metrics/{key}/values",
                np.asarray(trajectory, dtype=np.float64),
            )
            write_hdf5_dataset(
                path,
                f"metrics/{key}/rounds",
                np.asarray(rounds, dtype=np.int64),
            )

        # Save embedding snapshots
        for round_num, snapshot in sorted(self._embeddings.items()):
            for layer_name, comm_embs in snapshot.items():
                for cid, emb in comm_embs.items():
                    write_hdf5_dataset(
                        path,
                        f"embeddings/round_{round_num}/{layer_name}/community_{cid}",
                        np.asarray(emb, dtype=np.float64),
                    )

        # Save metadata
        write_hdf5_attrs(
            path,
            "metadata",
            {
                "n_metrics": len(self._recorded_keys),
                "metric_keys": ",".join(sorted(self._recorded_keys)),
                "n_embedding_snapshots": len(self._embeddings),
            },
        )

        logger.info(
            "Saved %d metrics and %d embedding snapshots to %s",
            len(self._recorded_keys),
            len(self._embeddings),
            path,
        )
