"""Watts-Strogatz small-world graph generator.

Produces graphs that interpolate between a ring lattice (*rewire_prob=0*)
and an Erdos-Renyi random graph (*rewire_prob=1*), giving tunable
clustering and short path lengths.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class WattsStrogatzGenerator:
    """Generate a connected Watts-Strogatz small-world graph.

    Parameters
    ----------
    n_clients : int
        Total number of nodes.
    k_neighbors : int
        Each node is initially connected to *k_neighbors* nearest neighbours
        in a ring (must be even).
    rewire_prob : float
        Probability of rewiring each edge.  ``0`` yields a ring lattice;
        ``1`` yields an essentially random graph.
    seed : int
        RNG seed for reproducibility.
    n_communities : int
        Number of nominal communities for the returned community map
        (contiguous blocks; the WS model has no planted structure).
    max_attempts : int
        Maximum retries to obtain a connected graph.
    """

    def __init__(
        self,
        n_clients: int,
        k_neighbors: int,
        rewire_prob: float,
        seed: int,
        n_communities: int = 4,
        max_attempts: int = 100,
    ) -> None:
        if k_neighbors % 2 != 0:
            raise ValueError(
                f"k_neighbors must be even, got {k_neighbors}."
            )
        if k_neighbors >= n_clients:
            raise ValueError(
                f"k_neighbors ({k_neighbors}) must be less than "
                f"n_clients ({n_clients})."
            )
        self.n_clients = n_clients
        self.k_neighbors = k_neighbors
        self.rewire_prob = rewire_prob
        self.seed = seed
        self.n_communities = n_communities
        self.max_attempts = max_attempts

    def generate(self) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """Generate a connected Watts-Strogatz graph.

        Returns
        -------
        adjacency : np.ndarray
            Symmetric binary adjacency matrix, dtype float64, zero diagonal.
        community_map : dict[int, list[int]]
            Contiguous-block community assignment (nominal).

        Raises
        ------
        RuntimeError
            If a connected graph is not found within *max_attempts*.
        """
        rng = np.random.RandomState(self.seed)

        for attempt in range(1, self.max_attempts + 1):
            attempt_seed = int(rng.randint(0, 2**31))
            G = nx.watts_strogatz_graph(
                self.n_clients,
                self.k_neighbors,
                self.rewire_prob,
                seed=attempt_seed,
            )

            if nx.is_connected(G):
                logger.info(
                    "Watts-Strogatz graph connected on attempt %d "
                    "(k=%d, p=%.4f, seed=%d).",
                    attempt,
                    self.k_neighbors,
                    self.rewire_prob,
                    attempt_seed,
                )
                break
            logger.debug(
                "WS attempt %d/%d not connected (seed=%d), retrying.",
                attempt,
                self.max_attempts,
                attempt_seed,
            )
        else:
            raise RuntimeError(
                f"Failed to generate a connected Watts-Strogatz graph after "
                f"{self.max_attempts} attempts."
            )

        adjacency = nx.to_numpy_array(G, dtype=np.float64)
        np.fill_diagonal(adjacency, 0.0)

        # Nominal community map: contiguous blocks
        community_size = self.n_clients // self.n_communities
        remainder = self.n_clients % self.n_communities
        community_map: Dict[int, List[int]] = {}
        offset = 0
        for c in range(self.n_communities):
            size = community_size + (1 if c < remainder else 0)
            community_map[c] = list(range(offset, offset + size))
            offset += size

        return adjacency, community_map
