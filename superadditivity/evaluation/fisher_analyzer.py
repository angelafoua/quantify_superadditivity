"""Fisher discriminant ratio for measuring class/community separation.

Computes tr(S_B) / tr(S_W) where S_B is the between-class scatter matrix
and S_W is the within-class scatter matrix. A higher ratio indicates
greater separation between communities in representation space.

All computations use float64 for numerical stability.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def fisher_ratio(embeddings: Dict[int, np.ndarray]) -> float:
    """Compute Fisher discriminant ratio: tr(S_B) / tr(S_W).

    Parameters
    ----------
    embeddings:
        Mapping from community/class ID to embedding matrix ``(n_k, d)``.
        Each matrix contains the representations of all samples in that
        group.

    Returns
    -------
    float
        Fisher ratio.  Returns 0.0 if S_W has zero trace (all within-class
        variance is zero).

    Raises
    ------
    ValueError
        If fewer than 2 groups are provided.
    """
    if len(embeddings) < 2:
        raise ValueError(
            f"Need at least 2 groups for Fisher ratio, got {len(embeddings)}"
        )

    # Stack all embeddings to compute the global mean
    all_emb = [np.asarray(v, dtype=np.float64) for v in embeddings.values()]
    global_mean = np.vstack(all_emb).mean(axis=0)  # (d,)

    d = global_mean.shape[0]
    S_B = np.zeros((d, d), dtype=np.float64)
    S_W = np.zeros((d, d), dtype=np.float64)

    for emb in all_emb:
        n_k = emb.shape[0]
        mean_k = emb.mean(axis=0)  # (d,)

        # Between-class scatter
        diff = (mean_k - global_mean).reshape(-1, 1)  # (d, 1)
        S_B += n_k * (diff @ diff.T)

        # Within-class scatter
        centered = emb - mean_k  # (n_k, d)
        S_W += centered.T @ centered

    tr_sw = np.trace(S_W)
    tr_sb = np.trace(S_B)

    if tr_sw < 1e-15:
        logger.warning("Within-class scatter trace near zero; returning 0.0")
        return 0.0

    return float(tr_sb / tr_sw)


class FisherAnalyzer:
    """Analyzer that computes the Fisher discriminant ratio.

    Examples
    --------
    >>> analyzer = FisherAnalyzer()
    >>> ratio = analyzer.compute(embeddings)
    """

    def compute(self, embeddings: Dict[int, np.ndarray]) -> float:
        """Compute Fisher discriminant ratio for the given embeddings.

        Parameters
        ----------
        embeddings:
            Mapping from community ID to embedding matrix ``(n_k, d)``.

        Returns
        -------
        float
            tr(S_B) / tr(S_W).
        """
        return fisher_ratio(embeddings)
