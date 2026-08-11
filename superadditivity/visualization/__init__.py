"""Publication-quality figures for superadditivity experiments."""

from __future__ import annotations

from superadditivity.visualization.figure_style import set_publication_style, WONG_PALETTE
from superadditivity.visualization.topology_plots import plot_topology
from superadditivity.visualization.cka_heatmaps import plot_cka_heatmap
from superadditivity.visualization.drift_trajectories import plot_drift_trajectories
from superadditivity.visualization.embedding_projections import plot_embedding_projection
from superadditivity.visualization.interaction_plots import plot_interaction_bar, plot_factorial_grid
from superadditivity.visualization.interaction_surface import plot_interaction_surface
from superadditivity.visualization.pout_sweep_plots import plot_pout_sweep

__all__ = [
    "set_publication_style",
    "WONG_PALETTE",
    "plot_topology",
    "plot_cka_heatmap",
    "plot_drift_trajectories",
    "plot_embedding_projection",
    "plot_interaction_bar",
    "plot_factorial_grid",
    "plot_interaction_surface",
    "plot_pout_sweep",
]
