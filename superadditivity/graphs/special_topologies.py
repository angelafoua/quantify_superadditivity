"""Deterministic special topologies: ring-of-cliques and exponential graph.

These generators produce fixed (non-random) communication topologies that
serve as analytic baselines in the superadditivity experiments.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def ring_of_cliques(
    n_clients: int,
    n_communities: int,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """Build a ring-of-cliques topology.

    Each of the *n_communities* groups forms a complete subgraph (clique).
    Adjacent cliques in a circular arrangement are connected by a single
    bridge edge between the last node of one clique and the first node of
    the next.

    Parameters
    ----------
    n_clients : int
        Total number of nodes (must be divisible by *n_communities*).
    n_communities : int
        Number of cliques (at least 2).

    Returns
    -------
    adjacency : np.ndarray
        Symmetric binary adjacency matrix, dtype float64, zero diagonal.
    community_map : dict[int, list[int]]
        Mapping from community index to sorted list of member node indices.

    Raises
    ------
    ValueError
        If *n_clients* is not divisible by *n_communities* or
        *n_communities* < 2.
    """
    if n_communities < 2:
        raise ValueError(
            f"ring_of_cliques requires at least 2 communities, got {n_communities}."
        )
    if n_clients % n_communities != 0:
        raise ValueError(
            f"n_clients ({n_clients}) must be divisible by "
            f"n_communities ({n_communities})."
        )

    clique_size = n_clients // n_communities
    adjacency = np.zeros((n_clients, n_clients), dtype=np.float64)
    community_map: Dict[int, List[int]] = {}

    # Build each clique as a complete subgraph
    for c in range(n_communities):
        start = c * clique_size
        end = start + clique_size
        community_map[c] = list(range(start, end))
        for i in range(start, end):
            for j in range(i + 1, end):
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

    # Bridge edges: last node of clique c to first node of clique (c+1) % K
    for c in range(n_communities):
        last_node = (c + 1) * clique_size - 1
        first_next = ((c + 1) % n_communities) * clique_size
        adjacency[last_node, first_next] = 1.0
        adjacency[first_next, last_node] = 1.0

    logger.info(
        "Ring-of-cliques: %d nodes, %d cliques of size %d, %d bridge edges.",
        n_clients,
        n_communities,
        clique_size,
        n_communities,
    )
    return adjacency, community_map


def exponential_graph(
    n_clients: int,
    n_communities: int = 4,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """Build an exponential (base-2) graph.

    Node *i* is connected to nodes ``i +/- 2^k`` (mod *n_clients*) for
    ``k = 0, 1, 2, ...`` as long as ``2^k < n_clients``.  This gives
    ``O(log n)`` diameter with ``O(n log n)`` edges.

    Parameters
    ----------
    n_clients : int
        Total number of nodes.
    n_communities : int
        Number of nominal communities (contiguous blocks; the exponential
        graph has no planted community structure).

    Returns
    -------
    adjacency : np.ndarray
        Symmetric binary adjacency matrix, dtype float64, zero diagonal.
    community_map : dict[int, list[int]]
        Nominal contiguous-block community assignment.
    """
    adjacency = np.zeros((n_clients, n_clients), dtype=np.float64)

    for i in range(n_clients):
        k = 0
        while 2**k < n_clients:
            offset = 2**k
            j_forward = (i + offset) % n_clients
            j_backward = (i - offset) % n_clients
            if j_forward != i:
                adjacency[i, j_forward] = 1.0
                adjacency[j_forward, i] = 1.0
            if j_backward != i:
                adjacency[i, j_backward] = 1.0
                adjacency[j_backward, i] = 1.0
            k += 1

    np.fill_diagonal(adjacency, 0.0)

    # Nominal community map: contiguous blocks
    community_size = n_clients // n_communities
    remainder = n_clients % n_communities
    community_map: Dict[int, List[int]] = {}
    offset = 0
    for c in range(n_communities):
        size = community_size + (1 if c < remainder else 0)
        community_map[c] = list(range(offset, offset + size))
        offset += size

    n_edges = int(np.sum(adjacency) / 2)
    logger.info(
        "Exponential graph: %d nodes, %d edges, nominal %d communities.",
        n_clients,
        n_edges,
        n_communities,
    )
    return adjacency, community_map
