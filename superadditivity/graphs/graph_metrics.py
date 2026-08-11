"""Graph-theoretic metrics: spectral gap, modularity, conductance.

All computations use float64 for numerical stability.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def compute_spectral_gap(W: np.ndarray) -> float:
    """Compute the spectral gap of a mixing matrix.

    The spectral gap is defined as ``1 - |lambda_2|``, where ``lambda_2``
    is the second-largest eigenvalue of *W* in absolute value.  A larger
    spectral gap implies faster consensus.

    Parameters
    ----------
    W : np.ndarray
        Symmetric doubly-stochastic mixing matrix, dtype float64.

    Returns
    -------
    float
        The spectral gap.
    """
    eigenvalues = np.linalg.eigvalsh(W.astype(np.float64))
    # Sort by absolute value descending
    sorted_abs = np.sort(np.abs(eigenvalues))[::-1]
    # The largest eigenvalue of a doubly-stochastic matrix is 1;
    # the spectral gap is 1 - |second largest|.
    lambda_2_abs = float(sorted_abs[1]) if len(sorted_abs) > 1 else 0.0
    gap = 1.0 - lambda_2_abs

    logger.debug("Spectral gap: %.6f (|lambda_2|=%.6f).", gap, lambda_2_abs)
    return float(gap)


def compute_modularity(
    adjacency: np.ndarray,
    community_map: Dict[int, List[int]],
) -> float:
    """Compute Newman modularity for a given community assignment.

    .. math::

        Q = \\frac{1}{2m} \\sum_{ij} \\left[A_{ij}
            - \\frac{k_i k_j}{2m}\\right] \\delta(c_i, c_j)

    Parameters
    ----------
    adjacency : np.ndarray
        Symmetric binary adjacency matrix, dtype float64.
    community_map : dict[int, list[int]]
        Mapping from community index to list of node indices.

    Returns
    -------
    float
        Modularity score in ``[-0.5, 1.0]``.
    """
    A = adjacency.astype(np.float64)
    degrees = A.sum(axis=1)
    m2 = degrees.sum()  # 2m

    if m2 == 0.0:
        return 0.0

    Q = 0.0
    for members in community_map.values():
        members_arr = np.array(members, dtype=int)
        # Sum of A[i,j] for i,j in community
        intra_edges = A[np.ix_(members_arr, members_arr)].sum()
        # Sum of degrees in community
        degree_sum = degrees[members_arr].sum()
        Q += intra_edges / m2 - (degree_sum / m2) ** 2

    logger.debug("Modularity: %.6f.", Q)
    return float(Q)


def compute_conductance(
    adjacency: np.ndarray,
    community_map: Dict[int, List[int]],
) -> float:
    """Compute the mean conductance across communities.

    For community *S*, conductance is::

        phi(S) = cut(S, V\\S) / min(vol(S), vol(V\\S))

    where ``cut`` is the number of edges crossing the partition and
    ``vol`` is the sum of degrees within the set.

    Parameters
    ----------
    adjacency : np.ndarray
        Symmetric binary adjacency matrix, dtype float64.
    community_map : dict[int, list[int]]
        Mapping from community index to list of node indices.

    Returns
    -------
    float
        Mean conductance across all communities.  Lower values indicate
        better-separated communities.
    """
    A = adjacency.astype(np.float64)
    degrees = A.sum(axis=1)
    n = A.shape[0]
    all_nodes = set(range(n))

    conductances: list[float] = []
    for members in community_map.values():
        S = set(members)
        complement = all_nodes - S

        if not S or not complement:
            continue

        S_arr = np.array(sorted(S), dtype=int)
        C_arr = np.array(sorted(complement), dtype=int)

        cut = A[np.ix_(S_arr, C_arr)].sum()
        vol_S = degrees[S_arr].sum()
        vol_C = degrees[C_arr].sum()
        min_vol = min(vol_S, vol_C)

        if min_vol == 0.0:
            conductances.append(0.0)
        else:
            conductances.append(cut / min_vol)

    mean_cond = float(np.mean(conductances)) if conductances else 0.0
    logger.debug("Mean conductance: %.6f.", mean_cond)
    return mean_cond


def compute_all_metrics(
    adjacency: np.ndarray,
    W: np.ndarray,
    community_map: Dict[int, List[int]],
) -> Dict[str, float]:
    """Compute all graph metrics in one call.

    Returns a dictionary containing spectral gap, modularity, conductance,
    and basic degree statistics.

    Parameters
    ----------
    adjacency : np.ndarray
        Symmetric binary adjacency matrix.
    W : np.ndarray
        Doubly-stochastic mixing matrix.
    community_map : dict[int, list[int]]
        Community assignment.

    Returns
    -------
    dict[str, float]
        Keys: ``spectral_gap``, ``modularity``, ``conductance``,
        ``degree_mean``, ``degree_std``, ``degree_min``, ``degree_max``,
        ``n_edges``.
    """
    degrees = adjacency.astype(np.float64).sum(axis=1)

    metrics: Dict[str, float] = {
        "spectral_gap": compute_spectral_gap(W),
        "modularity": compute_modularity(adjacency, community_map),
        "conductance": compute_conductance(adjacency, community_map),
        "degree_mean": float(np.mean(degrees)),
        "degree_std": float(np.std(degrees)),
        "degree_min": float(np.min(degrees)),
        "degree_max": float(np.max(degrees)),
        "n_edges": float(np.sum(adjacency) / 2.0),
    }

    logger.info(
        "Graph metrics: spectral_gap=%.4f, modularity=%.4f, "
        "conductance=%.4f, mean_degree=%.1f.",
        metrics["spectral_gap"],
        metrics["modularity"],
        metrics["conductance"],
        metrics["degree_mean"],
    )
    return metrics
