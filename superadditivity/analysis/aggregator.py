"""Load and aggregate results from multiple experiment runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from superadditivity.utils.io import load_json, read_hdf5_dataset

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Aggregate experiment summaries and HDF5 metric trajectories.

    Parameters
    ----------
    results_dir:
        Root directory containing per-run subdirectories.
    """

    def __init__(self, results_dir: str) -> None:
        self.results_dir = Path(results_dir)

    def load_summaries(self) -> pd.DataFrame:
        """Load all ``summary.json`` files into a DataFrame."""
        records: List[Dict[str, Any]] = []
        for summary_path in sorted(self.results_dir.rglob("summary.json")):
            try:
                data = load_json(summary_path)
                data["_run_dir"] = str(summary_path.parent)
                records.append(data)
            except Exception as e:
                logger.warning("Failed to load %s: %s", summary_path, e)

        if not records:
            logger.warning("No summaries found in %s", self.results_dir)
            return pd.DataFrame()

        df = pd.DataFrame(records)
        logger.info("Loaded %d run summaries from %s", len(df), self.results_dir)
        return df

    def load_trajectories(
        self,
        metric: str = "cka",
        layer: str = "layer4",
    ) -> Dict[str, np.ndarray]:
        """Load HDF5 metric trajectories keyed by run directory name.

        Parameters
        ----------
        metric:
            Metric name (e.g. ``"cka"``, ``"rsa"``).
        layer:
            Layer name (e.g. ``"layer4"``).

        Returns
        -------
        dict mapping run-directory name → trajectory array.
        """
        key = f"metrics/{metric}/{layer}"
        trajectories: Dict[str, np.ndarray] = {}

        for h5_path in sorted(self.results_dir.rglob("drift_metrics.h5")):
            run_name = h5_path.parent.name
            try:
                arr = read_hdf5_dataset(h5_path, key)
                trajectories[run_name] = arr
            except (KeyError, Exception) as e:
                logger.debug("No key %s in %s: %s", key, h5_path, e)

        logger.info(
            "Loaded %d trajectories for %s/%s", len(trajectories), metric, layer
        )
        return trajectories

    def build_factorial_table(
        self,
        data_col: str = "data_regime",
        network_col: str = "network_regime",
        metric_col: str = "final_cka_cross_community",
    ) -> pd.DataFrame:
        """Build a table suitable for factorial ANOVA.

        Returns a DataFrame with columns: ``[data_regime, network_regime, metric, run_seed, graph_seed]``.
        """
        df = self.load_summaries()
        if df.empty:
            return df

        required = [data_col, network_col, metric_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in summaries: {missing}")

        out = df[[data_col, network_col, metric_col]].copy()
        out.columns = ["data_regime", "network_regime", "metric"]

        for seed_col in ["run_seed", "graph_seed"]:
            if seed_col in df.columns:
                out[seed_col] = df[seed_col]

        return out
