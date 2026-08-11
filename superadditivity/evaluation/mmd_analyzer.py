"""Maximum Mean Discrepancy (MMD) with permutation test.

Implements the unbiased quadratic-time MMD^2 estimator with a Gaussian RBF
kernel whose bandwidth is set via the median heuristic. Statistical
significance is assessed via a permutation test.

All computations use float64 for numerical stability.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.distance import cdist, pdist

logger = logging.getLogger(__name__)


def _median_bandwidth(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Gaussian RBF bandwidth via the median heuristic.

    The bandwidth is the median of pairwise Euclidean distances in the
    pooled sample ``[X; Y]``.

    Parameters
    ----------
    X:
        Shape ``(n, d)``.
    Y:
        Shape ``(m, d)``.

    Returns
    -------
    float
        Bandwidth (sigma) for the RBF kernel.
    """
    pooled = np.vstack([X, Y])
    dists = pdist(pooled, metric="euclidean")
    median = float(np.median(dists))
    return max(median, 1e-10)  # avoid zero bandwidth


def _rbf_kernel(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF kernel matrix.

    Parameters
    ----------
    A:
        Shape ``(n, d)``.
    B:
        Shape ``(m, d)``.
    sigma:
        Bandwidth.

    Returns
    -------
    np.ndarray
        Kernel matrix of shape ``(n, m)``.
    """
    sq_dists = cdist(A, B, metric="sqeuclidean")
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def _unbiased_mmd2(K_XX: np.ndarray, K_YY: np.ndarray, K_XY: np.ndarray) -> float:
    """Unbiased estimator of MMD^2.

    Parameters
    ----------
    K_XX:
        Kernel matrix within X, shape ``(n, n)``.
    K_YY:
        Kernel matrix within Y, shape ``(m, m)``.
    K_XY:
        Kernel matrix between X and Y, shape ``(n, m)``.

    Returns
    -------
    float
        Unbiased MMD^2 estimate.
    """
    n = K_XX.shape[0]
    m = K_YY.shape[0]

    # Zero the diagonals for the unbiased estimator
    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)

    term_xx = K_XX.sum() / (n * (n - 1)) if n > 1 else 0.0
    term_yy = K_YY.sum() / (m * (m - 1)) if m > 1 else 0.0
    term_xy = K_XY.sum() / (n * m)

    return float(term_xx + term_yy - 2.0 * term_xy)


def mmd_with_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    n_permutations: int = 200,
    seed: int = 0,
) -> Tuple[float, float]:
    """Compute MMD^2 with a permutation test for significance.

    Uses a Gaussian RBF kernel with median-heuristic bandwidth and the
    unbiased quadratic-time estimator.

    Parameters
    ----------
    X:
        Representation matrix of shape ``(n, d)``.
    Y:
        Representation matrix of shape ``(m, d)``.
    n_permutations:
        Number of permutations for the null distribution.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float]
        ``(mmd2, p_value)`` where ``mmd2`` is the unbiased MMD^2 estimate
        and ``p_value`` is the fraction of permuted statistics >= observed.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    n = X.shape[0]
    m = Y.shape[0]
    sigma = _median_bandwidth(X, Y)

    pooled = np.vstack([X, Y])
    total = n + m

    # Full kernel matrix on pooled data
    K_full = _rbf_kernel(pooled, pooled, sigma)

    # Observed statistic
    K_XX = K_full[:n, :n].copy()
    K_YY = K_full[n:, n:].copy()
    K_XY = K_full[:n, n:].copy()
    observed = _unbiased_mmd2(K_XX, K_YY, K_XY)

    # Permutation test
    rng = np.random.RandomState(seed)
    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(total)
        K_perm = K_full[np.ix_(perm, perm)]
        K_pp_XX = K_perm[:n, :n].copy()
        K_pp_YY = K_perm[n:, n:].copy()
        K_pp_XY = K_perm[:n, n:].copy()
        perm_stat = _unbiased_mmd2(K_pp_XX, K_pp_YY, K_pp_XY)
        if perm_stat >= observed:
            count_ge += 1

    p_value = float((count_ge + 1) / (n_permutations + 1))

    return observed, p_value


class MMDAnalyzer:
    """Analyzer that computes MMD across communities with permutation testing.

    Parameters
    ----------
    n_permutations:
        Number of permutations per test.
    seed:
        Base random seed for reproducibility.

    Examples
    --------
    >>> analyzer = MMDAnalyzer(n_permutations=100, seed=42)
    >>> mmd2, p = analyzer.compute(X, Y)
    >>> mmd_mat, p_mat = analyzer.compute_community_matrix(embeddings)
    """

    def __init__(self, n_permutations: int = 200, seed: int = 0) -> None:
        self.n_permutations = n_permutations
        self.seed = seed

    def compute(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[float, float]:
        """Compute MMD^2 and permutation p-value between two samples.

        Parameters
        ----------
        X:
            Shape ``(n, d)``.
        Y:
            Shape ``(m, d)``.

        Returns
        -------
        Tuple[float, float]
            ``(mmd2, p_value)``.
        """
        return mmd_with_permutation_test(
            X, Y,
            n_permutations=self.n_permutations,
            seed=self.seed,
        )

    def compute_community_matrix(
        self, embeddings: Dict[int, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute pairwise MMD and p-values between all communities.

        Parameters
        ----------
        embeddings:
            Mapping from community ID to embedding matrix ``(n, d)``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(mmd_mat, p_mat)`` -- both ``(K, K)`` symmetric matrices.
            Diagonal entries of ``mmd_mat`` are 0; diagonal entries of
            ``p_mat`` are 1.
        """
        keys = sorted(embeddings.keys())
        K = len(keys)
        mmd_mat = np.zeros((K, K), dtype=np.float64)
        p_mat = np.ones((K, K), dtype=np.float64)

        for i in range(K):
            for j in range(i + 1, K):
                mmd2, p = mmd_with_permutation_test(
                    embeddings[keys[i]],
                    embeddings[keys[j]],
                    n_permutations=self.n_permutations,
                    seed=self.seed + i * K + j,
                )
                mmd_mat[i, j] = mmd2
                mmd_mat[j, i] = mmd2
                p_mat[i, j] = p
                p_mat[j, i] = p

        return mmd_mat, p_mat

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
