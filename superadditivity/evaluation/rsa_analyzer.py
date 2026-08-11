"""Representational Similarity Analysis (RSA).

Computes the Spearman rank correlation between Euclidean representational
dissimilarity matrices (RDMs). Following Kriegeskorte et al. (2008),
"Representational Similarity Analysis -- Connecting the Branches of
Systems Neuroscience".

All computations use float64 for numerical stability.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def rsa(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute RSA between two representation matrices.

    Builds Euclidean RDMs from ``X`` and ``Y``, then returns the Spearman
    rank correlation between their upper-triangular entries.

    Parameters
    ----------
    X:
        Representation matrix of shape ``(n, p1)``.
    Y:
        Representation matrix of shape ``(n, p2)``.

    Returns
    -------
    float
        Spearman correlation of the vectorised RDMs, in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If the matrices have different numbers of samples, or fewer than 3
        samples (insufficient for a meaningful correlation).
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Sample counts must match: X has {X.shape[0]}, Y has {Y.shape[0]}"
        )
    if X.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 samples for RSA, got {X.shape[0]}"
        )

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    rdm_x = pdist(X, metric="euclidean")
    rdm_y = pdist(Y, metric="euclidean")

    corr, _ = spearmanr(rdm_x, rdm_y)
    return float(corr)


class RSAAnalyzer:
    """Analyzer that computes RSA across communities.

    Examples
    --------
    >>> analyzer = RSAAnalyzer()
    >>> sim = analyzer.compute(X, Y)
    >>> mat = analyzer.compute_community_matrix(embeddings)
    """

    def compute(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute RSA between two representation matrices.

        Parameters
        ----------
        X:
            Shape ``(n, p1)``.
        Y:
            Shape ``(n, p2)``.

        Returns
        -------
        float
            Spearman correlation of RDMs.
        """
        return rsa(X, Y)

    def compute_community_matrix(
        self, embeddings: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """Compute pairwise RSA between all communities.

        Parameters
        ----------
        embeddings:
            Mapping from community ID to embedding matrix ``(n, d)``.

        Returns
        -------
        np.ndarray
            Symmetric ``(K, K)`` matrix of RSA correlations.
        """
        keys = sorted(embeddings.keys())
        K = len(keys)
        matrix = np.ones((K, K), dtype=np.float64)

        for i in range(K):
            for j in range(i + 1, K):
                sim = rsa(embeddings[keys[i]], embeddings[keys[j]])
                matrix[i, j] = sim
                matrix[j, i] = sim

        return matrix

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
