"""Community centroid distances for measuring representation drift.

Computes per-community centroid locations and all pairwise Euclidean
distances between them. Useful for tracking how communities move apart
(or converge) in representation space over training.

All computations use float64 for numerical stability.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


def centroid_distances(
    embeddings: Dict[int, np.ndarray],
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Compute community centroids and pairwise Euclidean distances.

    Parameters
    ----------
    embeddings:
        Mapping from community ID to embedding matrix ``(n_k, d)``.

    Returns
    -------
    Tuple[Dict[int, np.ndarray], np.ndarray]
        ``(centroids, distance_matrix)`` where ``centroids`` maps each
        community ID to its mean embedding ``(d,)`` and ``distance_matrix``
        is a ``(K, K)`` symmetric matrix of pairwise Euclidean distances,
        with communities ordered by sorted ID.
    """
    keys = sorted(embeddings.keys())
    centroids: Dict[int, np.ndarray] = {}
    centroid_list = []

    for cid in keys:
        emb = np.asarray(embeddings[cid], dtype=np.float64)
        c = emb.mean(axis=0)
        centroids[cid] = c
        centroid_list.append(c)

    centroid_stack = np.vstack(centroid_list)  # (K, d)
    if len(keys) < 2:
        dist_mat = np.zeros((len(keys), len(keys)), dtype=np.float64)
    else:
        dist_mat = squareform(pdist(centroid_stack, metric="euclidean"))

    return centroids, dist_mat


class CentroidAnalyzer:
    """Analyzer for community centroid distances.

    Examples
    --------
    >>> analyzer = CentroidAnalyzer()
    >>> centroids, dist_mat = analyzer.compute(embeddings)
    >>> mean_d = analyzer.mean_distance(dist_mat)
    """

    def compute(
        self, embeddings: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """Compute centroids and pairwise distance matrix.

        Parameters
        ----------
        embeddings:
            Mapping from community ID to embedding matrix ``(n_k, d)``.

        Returns
        -------
        Tuple[Dict[int, np.ndarray], np.ndarray]
            ``(centroids, distance_matrix)``.
        """
        return centroid_distances(embeddings)

    @staticmethod
    def mean_distance(distance_matrix: np.ndarray) -> float:
        """Mean of upper-triangular entries in the distance matrix.

        Parameters
        ----------
        distance_matrix:
            Square symmetric matrix of pairwise distances.

        Returns
        -------
        float
            Mean pairwise distance.
        """
        K = distance_matrix.shape[0]
        if K < 2:
            return float("nan")
        idx = np.triu_indices(K, k=1)
        return float(np.mean(distance_matrix[idx]))

    @staticmethod
    def stack_centroids(
        centroids: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Stack centroids into a ``(K, d)`` matrix ordered by community ID.

        Parameters
        ----------
        centroids:
            Mapping from community ID to centroid vector ``(d,)``.

        Returns
        -------
        np.ndarray
            Shape ``(K, d)`` with rows ordered by sorted community ID.
        """
        keys = sorted(centroids.keys())
        return np.vstack([centroids[k] for k in keys]).astype(np.float64)
