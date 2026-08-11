"""Network topology visualisation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from superadditivity.visualization.figure_style import WONG_COLORS, set_publication_style

logger = logging.getLogger(__name__)


def plot_topology(
    G: nx.Graph,
    community_assignments: Optional[np.ndarray] = None,
    title: str = "Network Topology",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Draw a network graph coloured by community.

    Parameters
    ----------
    G:
        NetworkX graph.
    community_assignments:
        Array mapping node → community (used for colouring).
    title:
        Plot title.
    save_path:
        If given, save the figure to this path.
    ax:
        Optional axes to draw on.

    Returns
    -------
    The matplotlib Figure.
    """
    set_publication_style()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure

    pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(G.number_of_nodes()))

    if community_assignments is not None:
        n_communities = int(community_assignments.max()) + 1
        colors = [WONG_COLORS[community_assignments[i] % len(WONG_COLORS)]
                  for i in range(G.number_of_nodes())]
    else:
        colors = [WONG_COLORS[0]] * G.number_of_nodes()

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=40, alpha=0.8,
    )
    ax.set_title(title)
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Topology plot saved to %s", save_path)

    return fig
