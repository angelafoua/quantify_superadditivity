"""Drift metrics available to the statistical analysis, and factorial cells.

The interaction term ``I = Drift(D) - Drift(B) - Drift(C) + Drift(A)`` and
its one-sided test ``H0: I <= 0`` assume that **larger values mean more
drift**. Metrics are recorded in their natural form: CKA is a
*similarity* (1 = identical representations), MMD is a *distance*. This
module converts each to drift orientation before analysis:

- similarity metrics: ``drift = 1 - value``. For CKA this is the "CKA
  divergence" used in the paper. ``I`` changes sign compared with raw CKA.
- distance metrics: ``drift = value``.

It also assigns runs to the 2x2 factorial cells from the labels that
``scripts/run_experiment.py`` writes (``data_regime`` = the partition
method, ``network_regime`` = the graph topology), and from the older
``*_noniid`` labels inferred from directory names.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SIMILARITY = "similarity"
DISTANCE = "distance"
ORIENTATIONS = (SIMILARITY, DISTANCE)


@dataclass(frozen=True)
class DriftMetric:
    """A summary column and how to turn it into a drift value.

    Attributes
    ----------
    name:
        Short name used on the command line and in output paths.
    column:
        Column in the run summaries (``summary.json``).
    orientation:
        ``"similarity"`` (drift = 1 - value) or ``"distance"``
        (drift = value).
    description:
        Human-readable description for reports.
    trajectory_key:
        Per-round key in ``drift_metrics.h5`` (``metrics/<key>/values``),
        with ``{layer}`` as a placeholder for the layer name. ``None`` if
        the metric has no per-round trajectory.
    """

    name: str
    column: str
    orientation: str
    description: str
    trajectory_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.orientation not in ORIENTATIONS:
            raise ValueError(
                f"orientation must be one of {ORIENTATIONS}, got {self.orientation!r}"
            )

    def to_drift(self, values: Iterable[float]) -> np.ndarray:
        """Convert raw metric values to drift orientation (higher = more drift).

        Parameters
        ----------
        values:
            Raw metric values.

        Returns
        -------
        np.ndarray
            Float64 drift values.
        """
        arr = np.asarray(list(values) if not hasattr(values, "__len__") else values,
                         dtype=np.float64)
        return 1.0 - arr if self.orientation == SIMILARITY else arr

    def h5_key(self, layer: str) -> Optional[str]:
        """Per-round HDF5 metric key for ``layer`` (``None`` if unavailable)."""
        if self.trajectory_key is None:
            return None
        return self.trajectory_key.format(layer=layer)

    @property
    def label(self) -> str:
        """Axis label for plots, e.g. ``"Drift (1 - CKA)"``."""
        if self.orientation == SIMILARITY:
            return f"Drift (1 - {self.name.upper() if self.name in DRIFT_METRICS else self.name})"
        return f"Drift ({self.name.upper() if self.name in DRIFT_METRICS else self.name})"

    @property
    def drift_definition(self) -> str:
        """Formula relating drift to the raw column, for reports."""
        if self.orientation == SIMILARITY:
            return f"Drift = 1 - {self.column}"
        return f"Drift = {self.column}"


#: Named metrics. ``cka`` is the paper's primary metric.
DRIFT_METRICS: Dict[str, DriftMetric] = {
    "cka": DriftMetric(
        name="cka",
        column="final_cka_cross_community",
        orientation=SIMILARITY,
        description=(
            "Primary-layer linear CKA between community-averaged embeddings "
            "(mean off-diagonal), final round"
        ),
        trajectory_key="cka_{layer}_mean",
    ),
    "mmd": DriftMetric(
        name="mmd",
        column="final_mmd_paired_drift",
        orientation=DISTANCE,
        description=(
            "Primary-layer client-level paired MMD^2, between-community minus "
            "within-community mean (B - W), median-heuristic bandwidth, final round"
        ),
        trajectory_key="mmd_paired_{layer}_drift",
    ),
}

DEFAULT_METRIC = "cka"


def resolve_drift_metric(
    metric: str = DEFAULT_METRIC,
    orientation: Optional[str] = None,
) -> DriftMetric:
    """Look up a named metric, or build one for a raw summary column.

    Parameters
    ----------
    metric:
        A name in :data:`DRIFT_METRICS` (e.g. ``"cka"``, ``"mmd"``) or a
        summary column name (e.g. ``"final_mmd_paired_layer4_drift_bw0p5"``).
    orientation:
        Required for a raw column: ``"similarity"`` or ``"distance"``. For
        a named metric it must be omitted or match the metric's
        orientation.

    Returns
    -------
    DriftMetric

    Raises
    ------
    ValueError
        If a raw column has no orientation, or an orientation contradicts
        a named metric.
    """
    if metric in DRIFT_METRICS:
        known = DRIFT_METRICS[metric]
        if orientation is not None and orientation != known.orientation:
            raise ValueError(
                f"Metric {metric!r} is a {known.orientation}; "
                f"--orientation {orientation} contradicts it"
            )
        return known
    if orientation is None:
        raise ValueError(
            f"{metric!r} is not a named metric ({sorted(DRIFT_METRICS)}). For a "
            "raw summary column, pass its orientation: 'similarity' "
            "(higher = more alike, e.g. CKA/RSA) or 'distance' "
            "(higher = more drift, e.g. MMD)."
        )
    return DriftMetric(
        name=metric,
        column=metric,
        orientation=orientation,
        description=f"Summary column {metric} ({orientation})",
        # A final_<key> summary column is the last value of metrics/<key>.
        trajectory_key=metric[len("final_"):] if metric.startswith("final_") else None,
    )


def output_subdir(metric: DriftMetric, base: str) -> str:
    """Name of a per-metric output folder: ``base`` for CKA, else ``base_<metric>``.

    Parameters
    ----------
    metric:
        The drift metric.
    base:
        Base folder name, e.g. ``"analysis"`` or ``"figures"``.

    Returns
    -------
    str
        Folder name (no path).
    """
    if metric.name == DEFAULT_METRIC:
        return base
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric.name)
    return f"{base}_{safe}"


# ----------------------------------------------------------------------
# Factorial cells
# ----------------------------------------------------------------------

#: Display labels for the factorial cells.
CELL_LABELS = {
    "A": "IID + Dense",
    "B": "Non-IID + Dense",
    "C": "IID + Community",
    "D": "Non-IID + Community",
}

#: Topologies treated as the dense (no community structure) level.
DENSE_TOPOLOGIES = frozenset({"erdos_renyi", "er", "dense", "complete"})
#: Topologies treated as the community-structured level.
COMMUNITY_TOPOLOGIES = frozenset({"sbm", "ring_of_cliques", "community"})


def _is_iid(label: str) -> bool:
    return str(label).strip().lower() == "iid"


def _network_level(label: str) -> Optional[str]:
    lab = str(label).strip().lower()
    if lab in DENSE_TOPOLOGIES:
        return "dense"
    if lab in COMMUNITY_TOPOLOGIES:
        return "community"
    return None


def assign_factorial_cells(
    df: pd.DataFrame,
    data_col: str = "data_regime",
    network_col: str = "network_regime",
) -> pd.Series:
    """Label each run with its 2x2 factorial cell (A, B, C or D).

    - A = IID + dense, B = non-IID + dense, C = IID + community,
      D = non-IID + community.
    - Data level: ``"iid"`` is IID. Any other label is non-IID, for example
      ``dirichlet_semantic`` or ``moderate_noniid``.
    - Network level: :data:`DENSE_TOPOLOGIES` and
      :data:`COMMUNITY_TOPOLOGIES`.

    Parameters
    ----------
    df:
        Table with one row per run.
    data_col, network_col:
        Column names of the two factors.

    Returns
    -------
    pd.Series
        Cell label per row, aligned with ``df.index``.

    Raises
    ------
    ValueError
        If a topology is not recognised, more than one non-IID data
        regime or more than one topology per level is present (the design
        is not a 2x2), or a cell has no runs.
    """
    unknown = sorted({
        str(v) for v in df[network_col].unique() if _network_level(v) is None
    })
    if unknown:
        raise ValueError(
            f"Unrecognised network_regime values {unknown}; expected one of "
            f"{sorted(DENSE_TOPOLOGIES)} (dense) or "
            f"{sorted(COMMUNITY_TOPOLOGIES)} (community)"
        )

    noniid_labels = sorted({str(v) for v in df[data_col].unique() if not _is_iid(v)})
    if len(noniid_labels) > 1:
        raise ValueError(
            f"More than one non-IID data regime {noniid_labels}: not a 2x2 design. "
            "Analyse one non-IID level at a time."
        )
    for level in ("dense", "community"):
        topos = sorted({
            str(v) for v in df[network_col].unique() if _network_level(v) == level
        })
        if len(topos) > 1:
            raise ValueError(
                f"More than one {level} topology {topos}: not a 2x2 design."
            )

    iid = df[data_col].map(_is_iid)
    net = df[network_col].map(_network_level)
    cells = pd.Series(index=df.index, dtype=object)
    cells[iid & (net == "dense")] = "A"
    cells[~iid & (net == "dense")] = "B"
    cells[iid & (net == "community")] = "C"
    cells[~iid & (net == "community")] = "D"

    missing = [c for c in "ABCD" if not (cells == c).any()]
    if missing:
        raise ValueError(
            f"Factorial cells {missing} have no runs "
            f"(data regimes {sorted(map(str, df[data_col].unique()))}, "
            f"network regimes {sorted(map(str, df[network_col].unique()))})"
        )
    return cells
