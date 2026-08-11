"""Plots for p_out sweep experiments — drift vs network heterogeneity."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from superadditivity.visualization.figure_style import WONG_COLORS, set_publication_style

logger = logging.getLogger(__name__)


def plot_pout_sweep(
    pout_values: np.ndarray,
    drift_by_data_regime: Dict[str, np.ndarray],
    spectral_gaps: Optional[np.ndarray] = None,
    metric_name: str = "CKA",
    title: str = "Drift vs Network Heterogeneity",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot drift as a function of p_out for multiple data regimes.

    Parameters
    ----------
    pout_values:
        Array of p_out values.
    drift_by_data_regime:
        Mapping of data-regime label → array of drift values
        (one per p_out value).
    spectral_gaps:
        If given, overlay spectral gap on a secondary y-axis.
    metric_name:
        Name of the drift metric.
    title:
        Plot title.
    save_path:
        If given, save the figure.

    Returns
    -------
    The matplotlib Figure.
    """
    set_publication_style()
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    for i, (label, drift) in enumerate(drift_by_data_regime.items()):
        color = WONG_COLORS[i % len(WONG_COLORS)]
        ax1.plot(pout_values, drift, "o-", label=label, color=color, linewidth=1.8)

    ax1.set_xlabel("p_out (inter-community edge probability)")
    ax1.set_ylabel(f"Cross-Community {metric_name}")
    ax1.set_title(title)
    ax1.legend(loc="upper left")

    if spectral_gaps is not None:
        ax2 = ax1.twinx()
        ax2.plot(
            pout_values, spectral_gaps, "k--",
            label="Spectral Gap", alpha=0.5, linewidth=1.2,
        )
        ax2.set_ylabel("Spectral Gap (1 - |λ₂|)")
        ax2.legend(loc="upper right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("p_out sweep plot saved to %s", save_path)

    return fig
