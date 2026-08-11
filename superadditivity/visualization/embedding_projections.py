"""t-SNE and UMAP projections of latent representations."""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from superadditivity.visualization.figure_style import WONG_COLORS, set_publication_style

logger = logging.getLogger(__name__)


def plot_embedding_projection(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    community_ids: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    perplexity: int = 30,
    seed: int = 42,
) -> plt.Figure:
    """Project features to 2D and plot, coloured by label or community.

    Parameters
    ----------
    features:
        Feature matrix of shape ``(n_samples, feature_dim)``.
    labels:
        Class labels for colouring (shape ``(n_samples,)``).
    method:
        ``"tsne"`` or ``"umap"``.
    community_ids:
        If given, use marker shapes to distinguish communities.
    title:
        Plot title.
    save_path:
        If given, save the figure.
    perplexity:
        t-SNE perplexity.
    seed:
        Random seed for projection.

    Returns
    -------
    The matplotlib Figure.
    """
    set_publication_style()

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2, perplexity=perplexity,
            random_state=seed, init="pca", learning_rate="auto",
        )
        emb = reducer.fit_transform(features)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=seed)
            emb = reducer.fit_transform(features)
        except ImportError:
            logger.warning("umap-learn not installed; falling back to t-SNE.")
            return plot_embedding_projection(
                features, labels, method="tsne",
                community_ids=community_ids, title=title,
                save_path=save_path, perplexity=perplexity, seed=seed,
            )
    else:
        raise ValueError(f"Unknown projection method: {method!r}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    unique_labels = np.unique(labels)
    cmap = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=[cmap[i]], s=8, alpha=0.6, label=f"Class {lab}",
        )

    ax.set_title(title or f"{method.upper()} Projection")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    if len(unique_labels) <= 20:
        ax.legend(markerscale=3, fontsize=7, ncol=2, loc="best")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Embedding projection saved to %s", save_path)

    return fig
