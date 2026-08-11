"""Bar plots and grids for the 2x2 factorial interaction effect."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from superadditivity.visualization.figure_style import WONG_COLORS, set_publication_style

logger = logging.getLogger(__name__)


def plot_interaction_bar(
    cell_means: Dict[str, float],
    cell_cis: Optional[Dict[str, Tuple[float, float]]] = None,
    interaction_I: Optional[float] = None,
    title: str = "2x2 Factorial: Superadditive Interaction",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar plot of the 2x2 factorial cells with the interaction term.

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

    Returns
    -------
    The matplotlib Figure.
    """
    set_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    labels_map = {
        "A": "IID + Dense",
        "B": "Non-IID + Dense",
        "C": "IID + Community",
        "D": "Non-IID + Community",
    }
    cell_keys = ["A", "B", "C", "D"]
    x = np.arange(len(cell_keys))
    means = [cell_means.get(k, 0) for k in cell_keys]
    colors = [WONG_COLORS[i] for i in range(4)]

    bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

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
    ax.set_ylabel("Cross-Community CKA Drift")
    ax.set_title(title)

    if interaction_I is not None:
        ax.text(
            0.95, 0.95,
            f"I = {interaction_I:.4f}",
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
    metric_col: str = "final_cka_cross_community",
    title: str = "Factorial Grid: CKA Drift",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grid plot showing metric means for all factor-level combinations.

    Parameters
    ----------
    results_df:
        DataFrame with factor and metric columns.
    data_col, network_col:
        Factor columns.
    metric_col:
        Metric column.
    title:
        Plot title.
    save_path:
        If given, save the figure.

    Returns
    -------
    The matplotlib Figure.
    """
    set_publication_style()
    import pandas as pd

    pivot = results_df.pivot_table(
        values=metric_col, index=data_col, columns=network_col, aggfunc="mean",
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".4f",
        cmap="YlOrRd", linewidths=0.5, square=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Network Regime")
    ax.set_ylabel("Data Regime")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Factorial grid saved to %s", save_path)

    return fig
