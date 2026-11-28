"""Load and aggregate results from multiple experiment runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from superadditivity.analysis.drift_metrics import (
    DriftMetric,
    assign_factorial_cells,
    resolve_drift_metric,
)
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

    def _find_summaries(self) -> List[Path]:
        """Find all ``summary.json`` files under results_dir.

        If results_dir itself contains summaries (nested layout), return those.
        Otherwise, fall back to sibling directories whose name starts with
        ``results_dir.name + '_'`` (flat layout produced before the nesting fix).
        """
        found = sorted(self.results_dir.rglob("summary.json"))
        if found:
            return found
        parent = self.results_dir.parent
        prefix = self.results_dir.name + "_"
        if parent.is_dir():
            for sibling in sorted(parent.iterdir()):
                if sibling.is_dir() and sibling.name.startswith(prefix):
                    found.extend(sorted(sibling.rglob("summary.json")))
        return found

    def load_summaries(self) -> pd.DataFrame:
        """Load all ``summary.json`` files into a DataFrame."""
        import re

        records: List[Dict[str, Any]] = []
        for summary_path in self._find_summaries():
            try:
                data = load_json(summary_path)
                data["_run_dir"] = str(summary_path.parent)

                # Parse p_out and data regime from ancestor folder names.
                # Collect names from the summary up to (not including)
                # results_dir.parent so both nested and sibling layouts work.
                stop = self.results_dir.parent
                ancestor_names = [
                    p.name
                    for p in summary_path.parents
                    if p != stop and stop in p.parents or p == summary_path.parent
                ]
                combined = " ".join(ancestor_names)
                if "p_out" not in data:
                    m = re.search(r"pout([0-9]+(?:\.[0-9]+)?)", combined)
                    if m:
                        data["p_out"] = float(m.group(1))
                if "data_regime" not in data:
                    for regime in (
                        "severe_noniid", "moderate_noniid", "mild_noniid",
                        "noniid", "iid",
                    ):
                        if regime in combined:
                            data["data_regime"] = regime
                            break
                # IID runs have no Dirichlet alpha; they are the alpha -> inf
                # limit and the reference level of the p_out surface.
                if data.get("dirichlet_alpha") is None and (
                    str(data.get("data_regime", "")).lower() == "iid"
                ):
                    data["dirichlet_alpha"] = float("inf")
                # Map data regime to dirichlet_alpha when absent
                if "dirichlet_alpha" not in data and "data_regime" in data:
                    _alpha_map = {
                        "iid": float("inf"),
                        # Must match configs/data/*.yaml.
                        "mild_noniid": 10.0,
                        "moderate_noniid": 0.5,
                        "noniid": 0.5,
                        "severe_noniid": 0.1,
                    }
                    data["dirichlet_alpha"] = _alpha_map.get(data["data_regime"])

                # Backfill final drift metrics from drift_metrics.h5 when absent.
                if (
                    "final_cka_cross_community" not in data
                    or "final_mmd_paired_drift" not in data
                ):
                    data = self._backfill_drift_metrics(data, summary_path.parent)

                records.append(data)
            except Exception as e:
                logger.warning("Failed to load %s: %s", summary_path, e)

        if not records:
            logger.warning("No summaries found in %s", self.results_dir)
            return pd.DataFrame()

        df = pd.DataFrame(records)
        logger.info("Loaded %d run summaries from %s", len(df), self.results_dir)
        return df

    @staticmethod
    def _backfill_drift_metrics(
        data: Dict[str, Any], run_dir: Path
    ) -> Dict[str, Any]:
        """Extract final drift metrics from drift_metrics.h5 into *data*."""
        import h5py

        h5_path = run_dir / "drift_metrics.h5"
        if not h5_path.exists():
            return data

        try:
            with h5py.File(h5_path, "r") as fh:
                metrics_grp = fh.get("metrics")
                if metrics_grp is None:
                    return data

                # Find the primary-layer CKA mean key (e.g. cka_layer4_mean).
                cka_mean_keys = sorted(
                    k for k in metrics_grp.keys()
                    if k.startswith("cka_") and k.endswith("_mean")
                )

                for key in metrics_grp.keys():
                    values_ds = metrics_grp.get(f"{key}/values")
                    if values_ds is None:
                        continue
                    arr = values_ds[()]
                    if arr.ndim == 1 and len(arr) > 0:
                        data.setdefault(f"final_{key}", float(arr[-1]))

                # Alias the deepest CKA layer mean as the canonical column.
                # The primary layer is the second-to-last in sorted order
                # (layer4 before fc, layer3 before fc, etc.), matching
                # run_experiment.py's layer_names[-2].
                if cka_mean_keys:
                    primary_key = cka_mean_keys[-1]
                    values_ds = metrics_grp.get(f"{primary_key}/values")
                    if values_ds is not None:
                        arr = values_ds[()]
                        if arr.ndim == 1 and len(arr) > 0:
                            data.setdefault(
                                "final_cka_cross_community", float(arr[-1])
                            )
                    # Alias paired MMD drift on the same primary layer.
                    layer = primary_key[len("cka_"):-len("_mean")]
                    mmd_key = f"final_mmd_paired_{layer}_drift"
                    if mmd_key in data:
                        data.setdefault("final_mmd_paired_drift", data[mmd_key])
        except Exception as exc:
            logger.debug("Could not backfill from %s: %s", h5_path, exc)

        return data

    def load_trajectories(
        self,
        metric_key: str,
    ) -> List[Dict[str, Any]]:
        """Load one per-round metric from every run's ``drift_metrics.h5``.

        Parameters
        ----------
        metric_key:
            Key as recorded by the drift tracker, e.g. ``"cka_layer4_mean"``
            or ``"mmd_paired_layer4_drift"``. Read from
            ``metrics/<metric_key>/values`` and ``.../rounds``.

        Returns
        -------
        list of dict
            One entry per run that has the key, with ``"run_dir"``,
            ``"rounds"`` and ``"values"`` (1-D arrays), and
            ``"data_regime"`` / ``"network_regime"`` from the run's
            ``summary.json`` (``None`` if absent).
        """
        runs: List[Dict[str, Any]] = []
        for h5_path in sorted(self.results_dir.rglob("drift_metrics.h5")):
            try:
                values = np.asarray(
                    read_hdf5_dataset(h5_path, f"metrics/{metric_key}/values"),
                    dtype=np.float64,
                )
                rounds = np.asarray(
                    read_hdf5_dataset(h5_path, f"metrics/{metric_key}/rounds")
                )
            except Exception as e:
                logger.debug("No key %s in %s: %s", metric_key, h5_path, e)
                continue
            if values.ndim != 1:
                logger.debug("%s in %s is not a scalar trajectory", metric_key, h5_path)
                continue
            summary_path = h5_path.parent / "summary.json"
            summary: Dict[str, Any] = {}
            if summary_path.exists():
                try:
                    summary = load_json(summary_path)
                except Exception as e:
                    logger.debug("Could not read %s: %s", summary_path, e)
            runs.append({
                "run_dir": str(h5_path.parent),
                "rounds": rounds,
                "values": values,
                "data_regime": summary.get("data_regime"),
                "network_regime": summary.get("network_regime"),
            })

        logger.info("Loaded %d trajectories for %s", len(runs), metric_key)
        return runs

    def build_factorial_table(
        self,
        data_col: str = "data_regime",
        network_col: str = "network_regime",
        drift_metric: Optional[DriftMetric] = None,
    ) -> pd.DataFrame:
        """Build a table suitable for factorial ANOVA and the interaction test.

        Parameters
        ----------
        data_col, network_col:
            Summary columns holding the two factors.
        drift_metric:
            Metric to analyse; defaults to CKA. The ``metric`` column is in
            drift orientation (higher = more drift, see
            :mod:`superadditivity.analysis.drift_metrics`), and the raw
            value is kept in ``metric_raw``.

        Returns
        -------
        pd.DataFrame
            Columns ``[data_regime, network_regime, metric, metric_raw,
            cell, run_seed, graph_seed]``. ``cell`` is A/B/C/D. Runs
            missing the metric are dropped with a warning.
        """
        metric = drift_metric or resolve_drift_metric()
        df = self.load_summaries()
        if df.empty:
            return df

        required = [data_col, network_col, metric.column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in summaries: {missing}")

        n_before = len(df)
        df = df[df[metric.column].notna()]
        if len(df) < n_before:
            logger.warning(
                "%d of %d runs have no %s and are excluded",
                n_before - len(df), n_before, metric.column,
            )

        out = df[[data_col, network_col]].copy()
        out.columns = ["data_regime", "network_regime"]
        out["metric_raw"] = df[metric.column].astype(np.float64)
        out["metric"] = metric.to_drift(out["metric_raw"])
        out["cell"] = assign_factorial_cells(out)

        for seed_col in ["run_seed", "graph_seed"]:
            if seed_col in df.columns:
                out[seed_col] = df[seed_col]

        return out
