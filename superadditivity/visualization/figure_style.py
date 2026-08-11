"""Publication figure style using the Wong colour-blind-safe palette."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

WONG_PALETTE = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

WONG_COLORS = list(WONG_PALETTE.values())


def set_publication_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.style.use("seaborn-v0_8-whitegrid")

    mpl.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "axes.prop_cycle": mpl.cycler(color=WONG_COLORS),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
