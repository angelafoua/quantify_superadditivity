"""Stochastic Block Model (SBM) graph generator.

Generates planted-partition graphs via :func:`networkx.stochastic_block_model`,
retrying until the realised graph is connected.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class SBMGenerator:
    """Generate a connected Stochastic Block Model graph.

    Parameters
    ----------
    n_clients : int
        Total number of nodes (must be divisible by *n_communities*).
    n_communities : int
        Number of equal-sized communities.
    p_in : float
        Intra-community edge probability.
    p_out : float
        Inter-community edge probability.
    seed : int
        RNG seed for reproducibility.
    max_attempts : int
        Maximum retries to obtain a connected graph.
    """

    def __init__(
        self,
        n_clients: int,
        n_communities: int,
        p_in: float,
        p_out: float,
        seed: int,
        max_attempts: int = 100,
    ) -> None:
        if n_clients % n_communities != 0:
            raise ValueError(
                f"n_clients ({n_clients}) must be divisible by "
                f"n_communities ({n_communities})."
            )
        self.n_clients = n_clients
        self.n_communities = n_communities
        self.p_in = p_in
        self.p_out = p_out
        self.seed = seed
        self.max_attempts = max_attempts

    def generate(self) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """Generate a connected SBM graph.

        Returns
        -------
        adjacency : np.ndarray
            Symmetric binary adjacency matrix of shape ``(n_clients, n_clients)``
            with zero diagonal, dtype float64.
        community_map : dict[int, list[int]]
            Mapping from community index to sorted list of member node indices.

        Raises
        ------
        RuntimeError
            If a connected graph is not found within *max_attempts*.
        """
        community_size = self.n_clients // self.n_communities
        sizes = [community_size] * self.n_communities

        # Build the probability matrix: p_in on diagonal blocks, p_out elsewhere
        prob_matrix = np.full(
            (self.n_communities, self.n_communities), self.p_out, dtype=np.float64
        )
        np.fill_diagonal(prob_matrix, self.p_in)

        rng = np.random.RandomState(self.seed)

        for attempt in range(1, self.max_attempts + 1):
            attempt_seed = int(rng.randint(0, 2**31))
            G = nx.stochastic_block_model(
                sizes, prob_matrix.tolist(), seed=attempt_seed
            )

            if nx.is_connected(G):
                logger.info(
                    "SBM graph connected on attempt %d (seed=%d).",
                    attempt,
                    attempt_seed,
                )
                break
            logger.debug(
                "SBM attempt %d/%d not connected (seed=%d), retrying.",
                attempt,
                self.max_attempts,
                attempt_seed,
            )
        else:
            raise RuntimeError(
                f"Failed to generate a connected SBM graph after "
                f"{self.max_attempts} attempts."
            )

        adjacency = nx.to_numpy_array(G, dtype=np.float64)
        np.fill_diagonal(adjacency, 0.0)

        # Build community map from contiguous blocks
        community_map: Dict[int, List[int]] = {}
        offset = 0
        for c in range(self.n_communities):
            community_map[c] = list(range(offset, offset + sizes[c]))
            offset += sizes[c]

        return adjacency, community_map
