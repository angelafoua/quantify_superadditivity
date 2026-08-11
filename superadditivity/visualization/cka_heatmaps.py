"""CKA cross-community heatmaps."""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from superadditivity.visualization.figure_style import set_publication_style

logger = logging.getLogger(__name__)


def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    labels: Optional[list] = None,
    title: str = "Cross-Community CKA",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> plt.Figure:
    """Plot a CKA similarity matrix as a heatmap.

    Parameters
    ----------
    cka_matrix:
        Square matrix of CKA similarities.
    labels:
        Row/column labels (e.g. community names).
    title:
        Plot title.
    save_path:
        If given, save the figure.
    ax:
        Optional axes.
    vmin, vmax:
        Colour scale limits.

    Returns
    -------
    The matplotlib Figure.
    """
    set_publication_style()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    else:
        fig = ax.figure

    n = cka_matrix.shape[0]
    if labels is None:
        labels = [f"C{i}" for i in range(n)]

    sns.heatmap(
        cka_matrix,
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        vmin=vmin,
        vmax=vmax,
        cmap="RdYlBu_r",
        annot=True,
        fmt=".3f",
        square=True,
        linewidths=0.5,
    )
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("CKA heatmap saved to %s", save_path)

    return fig
