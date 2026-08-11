"""3D surface and heatmap of the interaction term I(alpha, beta)."""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from superadditivity.visualization.figure_style import set_publication_style

logger = logging.getLogger(__name__)


def plot_interaction_surface(
    alpha_values: np.ndarray,
    beta_values: np.ndarray,
    I_values: np.ndarray,
    title: str = "Interaction Surface I(α, β)",
    save_path: Optional[str] = None,
    mode: str = "both",
) -> plt.Figure:
    """Plot the interaction term as a 3D surface and/or 2D heatmap.

    Parameters
    ----------
    alpha_values:
        Dirichlet alpha values (1-D).
    beta_values:
        Network heterogeneity parameter values (1-D).
    I_values:
        Interaction term values (1-D, same length).
    title:
        Plot title.
    save_path:
        If given, save the figure.
    mode:
        ``"3d"``, ``"heatmap"``, or ``"both"``.

    Returns
    -------
    The matplotlib Figure.
    """
    set_publication_style()

    unique_alpha = np.unique(alpha_values)
    unique_beta = np.unique(beta_values)

    I_grid = np.full((len(unique_alpha), len(unique_beta)), np.nan)
    for a, b, I in zip(alpha_values, beta_values, I_values):
        i = np.searchsorted(unique_alpha, a)
        j = np.searchsorted(unique_beta, b)
        if i < len(unique_alpha) and j < len(unique_beta):
            I_grid[i, j] = I

    if mode == "both":
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(16, 6),
            subplot_kw={"projection": None},
        )
        fig.delaxes(ax1)
        ax1 = fig.add_subplot(121, projection="3d")
        _plot_3d(ax1, unique_alpha, unique_beta, I_grid, title)
        _plot_heatmap(ax2, unique_alpha, unique_beta, I_grid, title)
    elif mode == "3d":
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        _plot_3d(ax, unique_alpha, unique_beta, I_grid, title)
    elif mode == "heatmap":
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        _plot_heatmap(ax, unique_alpha, unique_beta, I_grid, title)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Interaction surface saved to %s", save_path)

    return fig


def _plot_3d(
    ax,
    alphas: np.ndarray,
    betas: np.ndarray,
    I_grid: np.ndarray,
    title: str,
) -> None:
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    mask = ~np.isnan(I_grid)

    if mask.all():
        ax.plot_surface(A, B, I_grid, cmap="viridis", alpha=0.8, edgecolor="none")
    else:
        ax.scatter(
            A[mask], B[mask], I_grid[mask],
            c=I_grid[mask], cmap="viridis", s=40,
        )

    ax.set_xlabel("α (Dirichlet)")
    ax.set_ylabel("β (p_out)")
    ax.set_zlabel("I")
    ax.set_title(title)


def _plot_heatmap(
    ax,
    alphas: np.ndarray,
    betas: np.ndarray,
    I_grid: np.ndarray,
    title: str,
) -> None:
    import seaborn as sns
    import pandas as pd

    df = pd.DataFrame(
        I_grid,
        index=[f"α={a:.2g}" for a in alphas],
        columns=[f"β={b:.3g}" for b in betas],
    )

    sns.heatmap(
        df, ax=ax, annot=True, fmt=".3f",
        cmap="RdYlBu_r", linewidths=0.5,
        center=0,
    )
    ax.set_title(f"{title} (Heatmap)")
    ax.set_xlabel("Network Heterogeneity (p_out)")
    ax.set_ylabel("Data Heterogeneity (α)")
