"""Erdos-Renyi graph generator with degree-matched probability.

Provides a homogeneous random graph whose expected degree matches that of
a reference SBM, enabling controlled comparisons of community structure
versus uniform connectivity.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def matched_er_probability(
    n_clients: int,
    n_communities: int,
    p_in: float,
    p_out: float,
) -> float:
    """Compute the ER edge probability that matches the expected degree of an SBM.

    In an SBM with *K* equal-sized communities of size *m = n/K*, each node has
    expected degree ``(m - 1) * p_in + (n - m) * p_out``.  The matched ER
    probability satisfies ``p_er * (n - 1) = expected_degree``.

    Parameters
    ----------
    n_clients : int
        Total number of nodes.
    n_communities : int
        Number of equal-sized SBM communities.
    p_in : float
        Intra-community edge probability.
    p_out : float
        Inter-community edge probability.

    Returns
    -------
    float
        The ER edge probability.
    """
    m = n_clients // n_communities
    expected_degree = (m - 1) * p_in + (n_clients - m) * p_out
    p_er = expected_degree / (n_clients - 1)
    logger.debug(
        "Matched ER probability: %.6f (expected degree=%.2f, n=%d, K=%d).",
        p_er,
        expected_degree,
        n_clients,
        n_communities,
    )
    return float(p_er)


class ErdosRenyiGenerator:
    """Generate a connected Erdos-Renyi random graph.

    When *p* is ``None`` the edge probability is automatically computed to
    match the expected degree of an SBM with the given reference parameters.

    Parameters
    ----------
    n_clients : int
        Total number of nodes.
    seed : int
        RNG seed for reproducibility.
    p : float or None
        Edge probability.  ``None`` triggers degree-matching.
    n_communities : int
        Number of nominal communities (used for degree-matching and the
        returned community map).
    max_attempts : int
        Maximum retries to obtain a connected graph.
    ref_p_in : float
        Reference intra-community probability for degree matching.
    ref_p_out : float
        Reference inter-community probability for degree matching.
    """

    def __init__(
        self,
        n_clients: int,
        seed: int,
        p: Optional[float] = None,
        n_communities: int = 4,
        max_attempts: int = 100,
        ref_p_in: float = 0.25,
        ref_p_out: float = 0.01,
    ) -> None:
        self.n_clients = n_clients
        self.seed = seed
        self.n_communities = n_communities
        self.max_attempts = max_attempts

        if p is None:
            self.p = matched_er_probability(
                n_clients, n_communities, ref_p_in, ref_p_out
            )
        else:
            self.p = p

    def generate(self) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """Generate a connected ER graph.

        Returns
        -------
        adjacency : np.ndarray
            Symmetric binary adjacency matrix, dtype float64.
        nominal_community_map : dict[int, list[int]]
            Contiguous-block community assignment (nominal, as ER has no
            planted communities).

        Raises
        ------
        RuntimeError
            If a connected graph is not found within *max_attempts*.
        """
        rng = np.random.RandomState(self.seed)

        for attempt in range(1, self.max_attempts + 1):
            attempt_seed = int(rng.randint(0, 2**31))
            G = nx.erdos_renyi_graph(self.n_clients, self.p, seed=attempt_seed)

            if nx.is_connected(G):
                logger.info(
                    "ER graph connected on attempt %d (p=%.4f, seed=%d).",
                    attempt,
                    self.p,
                    attempt_seed,
                )
                break
            logger.debug(
                "ER attempt %d/%d not connected (seed=%d), retrying.",
                attempt,
                self.max_attempts,
                attempt_seed,
            )
        else:
            raise RuntimeError(
                f"Failed to generate a connected ER graph after "
                f"{self.max_attempts} attempts (p={self.p:.4f})."
            )

        adjacency = nx.to_numpy_array(G, dtype=np.float64)
        np.fill_diagonal(adjacency, 0.0)

        # Nominal community map: contiguous blocks
        community_size = self.n_clients // self.n_communities
        remainder = self.n_clients % self.n_communities
        nominal_community_map: Dict[int, List[int]] = {}
        offset = 0
        for c in range(self.n_communities):
            size = community_size + (1 if c < remainder else 0)
            nominal_community_map[c] = list(range(offset, offset + size))
            offset += size

        return adjacency, nominal_community_map
