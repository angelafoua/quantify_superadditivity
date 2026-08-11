"""Mixing-matrix construction and validation.

The Metropolis-Hastings weighting scheme yields a symmetric, doubly-stochastic
mixing matrix from any connected undirected graph, which is the standard
choice for decentralized SGD convergence analysis.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def metropolis_hastings_weights(adjacency: np.ndarray) -> np.ndarray:
    """Construct a Metropolis-Hastings doubly-stochastic mixing matrix.

    For every edge ``(i, j)``::

        W[i, j] = 1 / (1 + max(deg(i), deg(j)))

    The diagonal is set so that each row (and column) sums to 1.

    Parameters
    ----------
    adjacency : np.ndarray
        Symmetric binary adjacency matrix of shape ``(n, n)``, float64.

    Returns
    -------
    np.ndarray
        Symmetric doubly-stochastic mixing matrix, dtype float64.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(np.float64)
    W = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0:
                weight = 1.0 / (1.0 + max(degrees[i], degrees[j]))
                W[i, j] = weight
                W[j, i] = weight

    # Diagonal: row sums to 1
    for i in range(n):
        W[i, i] = 1.0 - W[i, :].sum()

    logger.debug(
        "Metropolis-Hastings matrix: n=%d, min off-diag weight=%.6f, "
        "max off-diag weight=%.6f.",
        n,
        W[W > 0].min() if np.any(W > 0) else 0.0,
        np.max(W - np.diag(np.diag(W))),
    )
    return W


def is_doubly_stochastic(W: np.ndarray, tol: float = 1e-9) -> bool:
    """Check whether *W* is doubly stochastic.

    Verifies:
    - All entries are non-negative.
    - Every row sums to 1 within *tol*.
    - Every column sums to 1 within *tol*.
    - The matrix is symmetric within *tol*.

    Parameters
    ----------
    W : np.ndarray
        Square matrix to check.
    tol : float
        Numerical tolerance for the checks.

    Returns
    -------
    bool
        ``True`` if all checks pass.
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        return False

    # Non-negativity
    if np.any(W < -tol):
        return False

    # Row sums
    row_sums = W.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        return False

    # Column sums
    col_sums = W.sum(axis=0)
    if not np.allclose(col_sums, 1.0, atol=tol):
        return False

    # Symmetry
    if not np.allclose(W, W.T, atol=tol):
        return False

    return True


def uniform_mixing_matrix(n: int) -> np.ndarray:
    """Create a uniform (fully-connected) mixing matrix.

    Returns ``(1/n) * ones(n, n)``, which is the consensus matrix for a
    complete graph.

    Parameters
    ----------
    n : int
        Number of nodes.

    Returns
    -------
    np.ndarray
        Uniform mixing matrix, dtype float64.
    """
    return np.full((n, n), 1.0 / n, dtype=np.float64)


def identity_mixing_matrix(n: int) -> np.ndarray:
    """Create an identity mixing matrix (no mixing).

    Represents a fully disconnected topology where each node keeps only
    its own parameters.

    Parameters
    ----------
    n : int
        Number of nodes.

    Returns
    -------
    np.ndarray
        Identity matrix, dtype float64.
    """
    return np.eye(n, dtype=np.float64)
