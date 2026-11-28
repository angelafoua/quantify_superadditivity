"""Bar plots and grids for the 2x2 factorial interaction effect."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superadditivity.analysis.drift_metrics import (
    CELL_LABELS,
    DriftMetric,
    resolve_drift_metric,
)
from superadditivity.visualization.figure_style import WONG_COLORS, set_publication_style

logger = logging.getLogger(__name__)


def factorial_cell_stats(
    table: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]], Dict[str, int]]:
    """Per-cell mean drift and 95% t-based CI.

    Parameters
    ----------
    table:
        Output of :meth:`ResultAggregator.build_factorial_table`, with
        columns ``cell`` (A-D) and ``metric`` (drift orientation).

    Returns
    -------
    tuple
        ``(means, cis, counts)`` keyed by cell. CI = mean +/- t(0.975,
        n - 1) * SEM. A cell with a single run gets a zero-width CI.
    """
    means: Dict[str, float] = {}
    cis: Dict[str, Tuple[float, float]] = {}
    counts: Dict[str, int] = {}
    for cell in "ABCD":
        vals = table.loc[table["cell"] == cell, "metric"].to_numpy(dtype=np.float64)
        if len(vals) == 0:
            continue
        from scipy import stats

        m = float(vals.mean())
        if len(vals) > 1:
            half = float(
                stats.t.ppf(0.975, df=len(vals) - 1) * vals.std(ddof=1) / np.sqrt(len(vals))
            )
        else:
            half = 0.0
        means[cell] = m
        cis[cell] = (m - half, m + half)
        counts[cell] = int(len(vals))
    return means, cis, counts


def plot_interaction_bar(
    cell_means: Dict[str, float],
    cell_cis: Optional[Dict[str, Tuple[float, float]]] = None,
    interaction_I: Optional[float] = None,
    title: str = "2x2 Factorial: Superadditive Interaction",
    save_path: Optional[str] = None,
    ylabel: str = "Drift (1 - CKA)",
    interaction_ci: Optional[Tuple[float, float]] = None,
    interaction_p: Optional[float] = None,
    show_additive_prediction: bool = True,
    interaction_n_bootstrap: Optional[int] = None,
) -> plt.Figure:
    """Bar plot of the 2x2 factorial cells with the interaction term.

    Values must be in drift orientation (higher = more drift), so that
    ``I = D - B - C + A > 0`` is superadditive. The additive prediction for
    D (``B + C - A``) is drawn as a dashed line over the D bar; the gap
    between the bar and the line is ``I``.

    Parameters
    ----------
    cell_means:
        Dict with keys ``"A"`` (IID+Dense), ``"B"`` (Non-IID+Dense),
        ``"C"`` (IID+Community), ``"D"`` (Non-IID+Community).
    cell_cis:
        Optional dict of (lower, upper) confidence intervals per cell.
    interaction_I:
        The interaction term value, annotated on the plot.
    title:
        Plot title.
    save_path:
        If given, save the figure.
    ylabel:
        Y-axis label naming the drift metric.
    interaction_ci:
        Optional (lower, upper) bootstrap CI of ``I`` for the annotation.
    interaction_p:
        Optional one-sided p-value (H0: I <= 0) for the annotation.
    show_additive_prediction:
        Draw the additive prediction ``B + C - A`` over the D bar.
    interaction_n_bootstrap:
        Number of bootstrap replicates behind ``interaction_p``. A p-value
        of 0 is then shown as ``p < 1/n_bootstrap``.

    Returns
    -------
    The matplotlib Figure.
    """
    set_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    labels_map = CELL_LABELS
    cell_keys = ["A", "B", "C", "D"]
    x = np.arange(len(cell_keys))
    means = [cell_means.get(k, 0) for k in cell_keys]
    colors = [WONG_COLORS[i] for i in range(4)]

    ax.bar(x, means, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    if cell_cis:
        for i, k in enumerate(cell_keys):
            if k in cell_cis:
                lo, hi = cell_cis[k]
                ax.errorbar(
                    x[i], means[i], yerr=[[means[i] - lo], [hi - means[i]]],
                    fmt="none", color="black", capsize=4,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([labels_map[k] for k in cell_keys], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    tops = [
        (cell_cis[k][1] if cell_cis and k in cell_cis else cell_means.get(k, 0))
        for k in cell_keys
    ]
    top = max(max(tops), max(means), 0.0)
    bottom = min(min(means), 0.0)
    if top > bottom:
        # Leave the upper ~35% of the axes free for the legend and annotation.
        ax.set_ylim(bottom, bottom + (top - bottom) / 0.65)

    if show_additive_prediction and all(k in cell_means for k in cell_keys):
        additive = cell_means["B"] + cell_means["C"] - cell_means["A"]
        ax.hlines(
            additive, x[3] - 0.45, x[3] + 0.45,
            colors="black", linestyles="--", linewidth=1.5,
            label="Additive prediction (B + C - A)",
        )
        ax.legend(loc="upper left")

    if interaction_I is not None:
        text = f"I = {interaction_I:.4g}"
        if interaction_ci is not None:
            text += f"\n95% CI [{interaction_ci[0]:.4g}, {interaction_ci[1]:.4g}]"
        if interaction_p is not None:
            if interaction_p == 0 and interaction_n_bootstrap:
                text += f"\np (I ≤ 0) < {1.0 / interaction_n_bootstrap:.1g}"
            else:
                text += f"\np (I ≤ 0) = {interaction_p:.3g}"
        ax.text(
            0.95, 0.95,
            text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Interaction bar plot saved to %s", save_path)

    return fig


def plot_factorial_grid(
    results_df,
    data_col: str = "data_regime",
    network_col: str = "network_regime",
    drift_metric: Optional[DriftMetric] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of mean drift for all factor-level combinations.

    Parameters
    ----------
    results_df:
        Run summaries with factor columns and ``drift_metric.column``.
    data_col, network_col:
        Factor columns.
    drift_metric:
        Metric to show, converted to drift orientation (higher = more
        drift). Defaults to CKA, shown as ``1 - CKA``.
    title:
        Plot title (default names the metric).
    save_path:
        If given, save the figure.

    Returns
    -------
    The matplotlib Figure.
    """
    set_publication_style()
    metric = drift_metric or resolve_drift_metric()
    if metric.column not in results_df.columns:
        raise ValueError(f"Missing column in summaries: {metric.column}")
    df = results_df[results_df[metric.column].notna()].copy()
    df["_drift"] = metric.to_drift(df[metric.column])

    # data_regime / network_regime don't distinguish alpha or p_out levels;
    # label rows and columns by level when those columns exist.
    def level_label(row, base_col, level_col, fmt):
        val = row.get(level_col) if level_col in df.columns else None
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return str(row[base_col])
        return f"{row[base_col]} ({fmt.format(val)})"

    df["_data_level"] = df.apply(
        lambda r: level_label(r, data_col, "dirichlet_alpha", "α={:g}"), axis=1
    )
    df["_net_level"] = df.apply(
        lambda r: level_label(r, network_col, "p_out", "p_out={:g}"), axis=1
    )
    pivot = df.pivot_table(
        values="_drift", index="_data_level", columns="_net_level", aggfunc="mean",
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".3g",
        cmap="YlOrRd", linewidths=0.5, square=True,
        cbar_kws={"label": metric.label},
    )
    ax.set_title(title or f"Factorial Grid: Mean {metric.label}")
    ax.set_xlabel("Network Regime")
    ax.set_ylabel("Data Regime")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Factorial grid saved to %s", save_path)

    return fig
