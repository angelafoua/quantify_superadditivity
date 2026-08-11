"""Linear CKA (Centered Kernel Alignment) for comparing representations.

Implements the efficient feature-space form of linear CKA from Kornblith
et al., "Similarity of Neural Network Representations Revisited" (ICML 2019).

All computations use float64 for numerical stability.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    Uses the efficient feature-space form:

        CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Both matrices are column-centered before computation.

    Parameters
    ----------
    X:
        Representation matrix of shape ``(n, p1)``, one row per sample.
    Y:
        Representation matrix of shape ``(n, p2)``, one row per sample.

    Returns
    -------
    float
        CKA similarity in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the matrices have different numbers of samples.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Sample counts must match: X has {X.shape[0]}, Y has {Y.shape[0]}"
        )

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # Column-center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Feature-space form
    YtX = Y.T @ X  # (p2, p1)
    XtX = X.T @ X  # (p1, p1)
    YtY = Y.T @ Y  # (p2, p2)

    numerator = np.linalg.norm(YtX, "fro") ** 2
    denominator = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")

    if denominator < 1e-15:
        logger.warning("CKA denominator near zero; returning 0.0")
        return 0.0

    return float(numerator / denominator)


class CKAAnalyzer:
    """Analyzer that computes linear CKA across communities and time steps.

    Examples
    --------
    >>> analyzer = CKAAnalyzer()
    >>> sim = analyzer.compute(X, Y)
    >>> mat = analyzer.compute_community_matrix(embeddings)
    """

    def compute(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute linear CKA between two representation matrices.

        Parameters
        ----------
        X:
            Shape ``(n, p1)``.
        Y:
            Shape ``(n, p2)``.

        Returns
        -------
        float
            CKA similarity.
        """
        return linear_cka(X, Y)

    def compute_community_matrix(
        self, embeddings: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """Compute pairwise CKA between all communities.

        Parameters
        ----------
        embeddings:
            Mapping from community ID to embedding matrix ``(n, d)``.

        Returns
        -------
        np.ndarray
            Symmetric ``(K, K)`` matrix of CKA similarities.
        """
        keys = sorted(embeddings.keys())
        K = len(keys)
        matrix = np.ones((K, K), dtype=np.float64)

        for i in range(K):
            for j in range(i + 1, K):
                sim = linear_cka(embeddings[keys[i]], embeddings[keys[j]])
                matrix[i, j] = sim
                matrix[j, i] = sim

        return matrix

    def compute_temporal(
        self,
        emb_t: Dict[int, np.ndarray],
        emb_prev: Dict[int, np.ndarray],
    ) -> Dict[int, float]:
        """Compute per-community CKA self-similarity between time steps.

        Parameters
        ----------
        emb_t:
            Current-round embeddings per community.
        emb_prev:
            Previous-round embeddings per community.

        Returns
        -------
        Dict[int, float]
            CKA similarity for each community present in both dicts.
        """
        result: Dict[int, float] = {}
        for cid in sorted(set(emb_t) & set(emb_prev)):
            result[cid] = linear_cka(emb_t[cid], emb_prev[cid])
        return result

    @staticmethod
    def mean_off_diagonal(matrix: np.ndarray) -> float:
        """Compute the mean of the upper-triangular entries.

        Parameters
        ----------
        matrix:
            Square symmetric matrix.

        Returns
        -------
        float
            Mean of upper-triangular (off-diagonal) elements.
        """
        K = matrix.shape[0]
        if K < 2:
            return float("nan")
        idx = np.triu_indices(K, k=1)
        return float(np.mean(matrix[idx]))
