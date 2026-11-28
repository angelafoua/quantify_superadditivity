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
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SIMILARITY = "similarity"
DISTANCE = "distance"
ORIENTATIONS = (SIMILARITY, DISTANCE)

#: Monotone transforms applied to drift before analysis. An interaction
#: measured on a bounded or non-linear scale can appear or vanish under a
#: monotone rescaling, so a superadditivity claim should hold on more than
#: one scale (e.g. ``none`` and ``logit`` for 1 - CKA).
TRANSFORMS = ("none", "log", "logit")


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
    transform:
        Monotone transform applied after orientation: ``"none"``,
        ``"log"`` (drift must be > 0) or ``"logit"`` (drift must be in
        (0, 1), e.g. 1 - CKA).
    display:
        Human-readable drift name for plot labels (default derived from
        ``name``).
    """

    name: str
    column: str
    orientation: str
    description: str
    trajectory_key: Optional[str] = None
    transform: str = "none"
    display: Optional[str] = None

    def __post_init__(self) -> None:
        if self.orientation not in ORIENTATIONS:
            raise ValueError(
                f"orientation must be one of {ORIENTATIONS}, got {self.orientation!r}"
            )
        if self.transform not in TRANSFORMS:
            raise ValueError(
                f"transform must be one of {TRANSFORMS}, got {self.transform!r}"
            )

    @property
    def key(self) -> str:
        """Identifier including the transform, used for output folders."""
        return self.name if self.transform == "none" else f"{self.name}_{self.transform}"

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
        drift = 1.0 - arr if self.orientation == SIMILARITY else arr
        if self.transform == "none":
            return drift
        if self.transform == "log":
            bad = ~(drift > 0)
            if bad.any():
                raise ValueError(
                    f"log transform needs drift > 0; {int(bad.sum())} value(s) of "
                    f"{self.column} give drift <= 0 (min {np.nanmin(drift):.3g})"
                )
            return np.log(drift)
        bad = ~((drift > 0) & (drift < 1))
        if bad.any():
            raise ValueError(
                f"logit transform needs 0 < drift < 1; {int(bad.sum())} value(s) "
                f"of {self.column} are outside (range {np.nanmin(drift):.3g} to "
                f"{np.nanmax(drift):.3g})"
            )
        return np.log(drift) - np.log1p(-drift)

    def h5_key(self, layer: str) -> Optional[str]:
        """Per-round HDF5 metric key for ``layer`` (``None`` if unavailable)."""
        if self.trajectory_key is None:
            return None
        return self.trajectory_key.format(layer=layer)

    @property
    def label(self) -> str:
        """Axis label for plots, e.g. ``"Drift (1 - CKA)"``."""
        if self.display is not None:
            inner = self.display
        else:
            inner = f"1 - {self.name}" if self.orientation == SIMILARITY else self.name
        if self.transform == "none":
            return f"Drift ({inner})"
        return f"{self.transform}(Drift ({inner}))"

    @property
    def drift_definition(self) -> str:
        """Formula relating drift to the raw column, for reports."""
        inner = (
            f"1 - {self.column}" if self.orientation == SIMILARITY else self.column
        )
        if self.transform == "none":
            return f"Drift = {inner}"
        return f"Drift = {self.transform}({inner})"


#: Primary-layer summary aliases: ``<alias>`` -> per-layer tracker key.
#: ``scripts/run_experiment.py`` writes ``final_<alias>`` (last evaluation)
#: and ``window_<alias>`` (mean over the last ``summary_window_rounds``
#: rounds) for each; the aggregator backfills ``final_<alias>`` for older
#: runs from ``drift_metrics.h5``.
PRIMARY_ALIASES: Dict[str, str] = {
    "cka_client_between": "cka_client_{layer}_between",
    "cka_client_drift": "cka_client_{layer}_drift",
    "cka_cross_community": "cka_{layer}_mean",
    "mmd_paired_between": "mmd_paired_{layer}_between",
    "mmd_paired_drift": "mmd_paired_{layer}_drift",
}

_BASE_METRICS = [
    # (name, alias, orientation, display, description)
    ("cka", "cka_client_between", SIMILARITY, "1 - CKA, between communities",
     "Primary-layer linear CKA averaged over client pairs in different "
     "communities (client level, as in the proposal)"),
    ("cka_bw", "cka_client_drift", DISTANCE, "CKA divergence, between - within",
     "Primary-layer client CKA divergence between communities minus within "
     "communities: (1 - CKA_between) - (1 - CKA_within)"),
    ("cka_community", "cka_cross_community", SIMILARITY, "1 - CKA, community-averaged",
     "Legacy: primary-layer linear CKA between community-averaged "
     "embeddings (mean off-diagonal)"),
    ("mmd", "mmd_paired_between", DISTANCE, "MMD^2, between communities",
     "Primary-layer paired MMD^2 averaged over client pairs in different "
     "communities"),
    ("mmd_bw", "mmd_paired_drift", DISTANCE, "MMD^2, between - within",
     "Primary-layer paired MMD^2 between communities minus within "
     "communities (B - W)"),
]

#: Named metrics. ``cka`` (client-level, between communities) is the
#: paper's primary metric. Each also has a ``<name>_window`` variant: the
#: mean over the last ``evaluation.summary_window_rounds`` rounds.
DRIFT_METRICS: Dict[str, DriftMetric] = {}
for _name, _alias, _orient, _display, _desc in _BASE_METRICS:
    DRIFT_METRICS[_name] = DriftMetric(
        name=_name,
        column=f"final_{_alias}",
        orientation=_orient,
        description=f"{_desc}, final evaluation",
        trajectory_key=PRIMARY_ALIASES[_alias],
        display=_display,
    )
    DRIFT_METRICS[f"{_name}_window"] = DriftMetric(
        name=f"{_name}_window",
        column=f"window_{_alias}",
        orientation=_orient,
        description=f"{_desc}, mean over the last evaluation.summary_window_rounds rounds",
        trajectory_key=PRIMARY_ALIASES[_alias],
        display=f"{_display} (window)",
    )

DEFAULT_METRIC = "cka"


def resolve_drift_metric(
    metric: str = DEFAULT_METRIC,
    orientation: Optional[str] = None,
    transform: str = "none",
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
    transform:
        One of :data:`TRANSFORMS`, applied after orientation.

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
        return replace(known, transform=transform)
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
        transform=transform,
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
    if metric.key == DEFAULT_METRIC:
        return base
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric.key)
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
    alpha_col: str = "dirichlet_alpha",
    pout_col: str = "p_out",
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
    alpha_col, pout_col:
        Optional level columns. ``data_regime`` is only the partition
        method and ``network_regime`` only the topology name, so different
        Dirichlet alphas, or different SBM ``p_out`` values, share a label.
        If these columns are present, more than one level among non-IID
        runs (alpha) or community-structured runs (p_out) is an error
        rather than being pooled into one cell.

    Returns
    -------
    pd.Series
        Cell label per row, aligned with ``df.index``.

    Raises
    ------
    ValueError
        If a topology is not recognised, more than one non-IID data
        regime or more than one topology per level is present (the design
        is not a 2x2), levels of alpha or p_out would be pooled, or a cell
        has no runs.
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

    def distinct(col: str, mask: pd.Series) -> List[float]:
        if col not in df.columns:
            return []
        vals = pd.to_numeric(df.loc[mask, col], errors="coerce").dropna().unique()
        return sorted(float(v) for v in vals)

    alphas = distinct(alpha_col, ~iid)
    if len(alphas) > 1:
        raise ValueError(
            f"Non-IID runs have {len(alphas)} different {alpha_col} values {alphas}: "
            "not a 2x2 design. Analyse one alpha level at a time (e.g. point "
            "--results_dir at a folder with a single level)."
        )
    pouts = distinct(pout_col, net == "community")
    if len(pouts) > 1:
        raise ValueError(
            f"Community-structured runs have {len(pouts)} different {pout_col} "
            f"values {pouts}: not a 2x2 design. Analyse one p_out level at a time, "
            "or use --analysis surface for sweeps."
        )
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
