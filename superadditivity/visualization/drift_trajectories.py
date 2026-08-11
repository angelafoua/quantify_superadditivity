"""Drift metric trajectory plots over communication rounds."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from superadditivity.visualization.figure_style import WONG_COLORS, set_publication_style

logger = logging.getLogger(__name__)


def plot_drift_trajectories(
    trajectories: Dict[str, np.ndarray],
    rounds: Optional[np.ndarray] = None,
    metric_name: str = "CKA",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show_ci: bool = True,
) -> plt.Figure:
    """Plot drift metric trajectories for multiple conditions.

    Parameters
    ----------
    trajectories:
        Mapping of condition label → array of shape ``(n_seeds, n_rounds)``
        or ``(n_rounds,)``.
    rounds:
        Round numbers for the x-axis. If None, uses 0-indexed.
    metric_name:
        Name of the metric for axis labels.
    title:
        Plot title.
    save_path:
        If given, save the figure.
    ax:
        Optional axes.
    show_ci:
        If True and trajectories have multiple seeds, show 95% CI band.

    Returns
    -------
    The matplotlib Figure.
    """
    set_publication_style()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.figure

    for i, (label, data) in enumerate(trajectories.items()):
        color = WONG_COLORS[i % len(WONG_COLORS)]
        data = np.atleast_2d(data)

        if rounds is None:
            x = np.arange(data.shape[1])
        else:
            x = rounds[:data.shape[1]]

        mean = data.mean(axis=0)
        ax.plot(x, mean, label=label, color=color, linewidth=1.8)

        if show_ci and data.shape[0] > 1:
            sem = data.std(axis=0) / np.sqrt(data.shape[0])
            ax.fill_between(
                x, mean - 1.96 * sem, mean + 1.96 * sem,
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Communication Round")
    ax.set_ylabel(f"Cross-Community {metric_name}")
    ax.set_title(title or f"{metric_name} Drift Trajectories")
    ax.legend(loc="best")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Drift trajectory plot saved to %s", save_path)

    return fig
