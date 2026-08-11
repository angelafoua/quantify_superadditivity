"""GraphManager: high-level orchestrator for topology construction.

Dispatches to the appropriate generator based on ``config.topology``,
builds the mixing matrix, computes metrics, and validates invariants.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

from superadditivity.graphs.erdos_renyi_generator import ErdosRenyiGenerator
from superadditivity.graphs.graph_metrics import compute_all_metrics
from superadditivity.graphs.mixing_matrix import (
    is_doubly_stochastic,
    metropolis_hastings_weights,
)
from superadditivity.graphs.sbm_generator import SBMGenerator
from superadditivity.graphs.special_topologies import exponential_graph, ring_of_cliques
from superadditivity.graphs.watts_strogatz_generator import WattsStrogatzGenerator
from superadditivity.utils.io import ensure_dir

logger = logging.getLogger(__name__)


class GraphManager:
    """Build, validate, and persist communication topologies.

    Parameters
    ----------
    config : DictConfig
        Hydra configuration.  Expected keys depend on ``config.topology``:

        - ``"sbm"``: ``n_clients``, ``n_communities``, ``p_in``, ``p_out``
        - ``"erdos_renyi"``: ``n_clients``, ``n_communities`` (and optionally
          ``p``, ``ref_p_in``, ``ref_p_out``)
        - ``"watts_strogatz"``: ``n_clients``, ``k_neighbors``,
          ``rewire_prob``, ``n_communities``
        - ``"ring_of_cliques"``: ``n_clients``, ``n_communities``
        - ``"exponential"``: ``n_clients``, ``n_communities``
    graph_seed : int
        Seed for random topology generation.
    """

    _SUPPORTED_TOPOLOGIES = frozenset(
        {"sbm", "erdos_renyi", "watts_strogatz", "ring_of_cliques", "exponential"}
    )

    def __init__(self, config: DictConfig, graph_seed: int) -> None:
        self.config = config
        self.graph_seed = graph_seed

        self._adjacency: Optional[np.ndarray] = None
        self._mixing_matrix: Optional[np.ndarray] = None
        self._communities: Optional[Dict[int, List[int]]] = None
        self._metrics: Optional[Dict[str, float]] = None
        self._topology: Optional[str] = None

    def build(self) -> None:
        """Build the topology, mixing matrix, and metrics.

        Dispatches to the generator indicated by ``config.topology``,
        constructs the Metropolis-Hastings mixing matrix, validates it,
        and computes all graph metrics.

        Raises
        ------
        ValueError
            If ``config.topology`` is not recognised.
        """
        topology = str(self.config.topology)
        if topology not in self._SUPPORTED_TOPOLOGIES:
            raise ValueError(
                f"Unknown topology {topology!r}.  "
                f"Supported: {sorted(self._SUPPORTED_TOPOLOGIES)}."
            )
        self._topology = topology

        logger.info("Building topology %r (seed=%d).", topology, self.graph_seed)
        adjacency, communities = self._dispatch(topology)

        self._adjacency = adjacency
        self._communities = communities
        self._mixing_matrix = metropolis_hastings_weights(adjacency)
        self.validate()
        self._metrics = compute_all_metrics(adjacency, self._mixing_matrix, communities)

        logger.info(
            "Topology %r built: %d nodes, %d edges, spectral_gap=%.4f.",
            topology,
            adjacency.shape[0],
            int(self._metrics["n_edges"]),
            self._metrics["spectral_gap"],
        )

    def _dispatch(
        self, topology: str
    ) -> tuple[np.ndarray, Dict[int, List[int]]]:
        """Route to the correct generator."""
        cfg = self.config

        if topology == "sbm":
            gen = SBMGenerator(
                n_clients=int(cfg.n_clients),
                n_communities=int(cfg.n_communities),
                p_in=float(cfg.p_in),
                p_out=float(cfg.p_out),
                seed=self.graph_seed,
            )
            return gen.generate()

        if topology == "erdos_renyi":
            gen_er = ErdosRenyiGenerator(
                n_clients=int(cfg.n_clients),
                seed=self.graph_seed,
                p=float(cfg.p) if hasattr(cfg, "p") and cfg.p is not None else None,
                n_communities=int(cfg.n_communities),
                ref_p_in=float(getattr(cfg, "ref_p_in", 0.25)),
                ref_p_out=float(getattr(cfg, "ref_p_out", 0.01)),
            )
            return gen_er.generate()

        if topology == "watts_strogatz":
            gen_ws = WattsStrogatzGenerator(
                n_clients=int(cfg.n_clients),
                k_neighbors=int(cfg.k_neighbors),
                rewire_prob=float(cfg.rewire_prob),
                seed=self.graph_seed,
                n_communities=int(cfg.n_communities),
            )
            return gen_ws.generate()

        if topology == "ring_of_cliques":
            return ring_of_cliques(
                n_clients=int(cfg.n_clients),
                n_communities=int(cfg.n_communities),
            )

        # topology == "exponential"
        return exponential_graph(
            n_clients=int(cfg.n_clients),
            n_communities=int(getattr(cfg, "n_communities", 4)),
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_mixing_matrix(self) -> np.ndarray:
        """Return the doubly-stochastic mixing matrix.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called.
        """
        if self._mixing_matrix is None:
            raise RuntimeError("Call build() before get_mixing_matrix().")
        return self._mixing_matrix

    def get_adjacency(self) -> np.ndarray:
        """Return the binary adjacency matrix.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called.
        """
        if self._adjacency is None:
            raise RuntimeError("Call build() before get_adjacency().")
        return self._adjacency

    def get_communities(self) -> Dict[int, List[int]]:
        """Return the community map.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called.
        """
        if self._communities is None:
            raise RuntimeError("Call build() before get_communities().")
        return self._communities

    def get_metrics(self) -> Dict[str, float]:
        """Return the computed graph metrics.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called.
        """
        if self._metrics is None:
            raise RuntimeError("Call build() before get_metrics().")
        return self._metrics

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Assert mixing-matrix invariants.

        Checks:
        1. *W* is symmetric.
        2. *W* is doubly stochastic.
        3. Non-zero off-diagonal entries of *W* correspond to edges in *A*.
        4. Zero diagonal in *A*.

        Raises
        ------
        AssertionError
            If any invariant is violated.
        """
        W = self._mixing_matrix
        A = self._adjacency
        assert W is not None and A is not None, "Nothing to validate -- call build()."

        # Symmetry
        assert np.allclose(W, W.T, atol=1e-12), (
            "Mixing matrix is not symmetric."
        )

        # Doubly stochastic
        assert is_doubly_stochastic(W, tol=1e-9), (
            "Mixing matrix is not doubly stochastic."
        )

        # Weights only on edges or diagonal
        n = W.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and W[i, j] > 1e-12:
                    assert A[i, j] > 0, (
                        f"W[{i},{j}]={W[i,j]:.6f} but A[{i},{j}]=0 (no edge)."
                    )

        # Zero diagonal in adjacency
        assert np.allclose(np.diag(A), 0.0), (
            "Adjacency matrix has non-zero diagonal."
        )

        logger.debug("Mixing matrix validation passed.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the graph state to a ``.npz`` file.

        Parameters
        ----------
        path : str
            Destination file path (should end in ``.npz``).
        """
        if self._adjacency is None or self._mixing_matrix is None:
            raise RuntimeError("Call build() before save().")

        p = Path(path)
        ensure_dir(p.parent)

        # Serialise community map as two arrays: keys and flat member lists
        comm_keys = np.array(sorted(self._communities.keys()), dtype=np.int64)  # type: ignore[union-attr]
        comm_values = np.array(
            [self._communities[k] for k in comm_keys], dtype=object  # type: ignore[union-attr]
        )
        # Store lengths and flat values for reliable reconstruction
        comm_lengths = np.array(
            [len(self._communities[k]) for k in comm_keys], dtype=np.int64  # type: ignore[union-attr]
        )
        comm_flat = np.concatenate(
            [np.array(self._communities[k], dtype=np.int64) for k in comm_keys]  # type: ignore[union-attr]
        )

        # Serialise metrics
        metric_keys: list[str] = []
        metric_values: list[float] = []
        if self._metrics is not None:
            for mk, mv in sorted(self._metrics.items()):
                metric_keys.append(mk)
                metric_values.append(mv)

        np.savez(
            p,
            adjacency=self._adjacency,
            mixing_matrix=self._mixing_matrix,
            comm_keys=comm_keys,
            comm_lengths=comm_lengths,
            comm_flat=comm_flat,
            metric_keys=np.array(metric_keys, dtype=str),
            metric_values=np.array(metric_values, dtype=np.float64),
            topology=np.array([self._topology or ""], dtype=str),
            graph_seed=np.array([self.graph_seed], dtype=np.int64),
        )
        logger.info("Graph state saved to %s.", p)

    @classmethod
    def load(cls, path: str) -> "GraphManager":
        """Load a graph state from a ``.npz`` file.

        Parameters
        ----------
        path : str
            Source file path.

        Returns
        -------
        GraphManager
            A manager instance with pre-loaded adjacency, mixing matrix,
            communities, and metrics.  No Hydra config is attached.
        """
        data = np.load(path, allow_pickle=False)

        # Reconstruct without a real config
        instance = object.__new__(cls)
        instance.config = None  # type: ignore[assignment]
        instance.graph_seed = int(data["graph_seed"][0])
        instance._adjacency = data["adjacency"].astype(np.float64)
        instance._mixing_matrix = data["mixing_matrix"].astype(np.float64)
        instance._topology = str(data["topology"][0]) if "topology" in data else None

        # Reconstruct community map
        comm_keys = data["comm_keys"]
        comm_lengths = data["comm_lengths"]
        comm_flat = data["comm_flat"]
        communities: Dict[int, List[int]] = {}
        offset = 0
        for idx, key in enumerate(comm_keys):
            length = int(comm_lengths[idx])
            communities[int(key)] = comm_flat[offset : offset + length].tolist()
            offset += length
        instance._communities = communities

        # Reconstruct metrics
        if "metric_keys" in data and len(data["metric_keys"]) > 0:
            instance._metrics = {
                str(k): float(v)
                for k, v in zip(data["metric_keys"], data["metric_values"])
            }
        else:
            instance._metrics = None

        logger.info("Graph state loaded from %s.", path)
        return instance
