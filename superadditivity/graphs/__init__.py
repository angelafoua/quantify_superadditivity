"""Network topology generators, mixing matrices, and graph metrics.

This module provides multiple communication-graph generators for
decentralized federated learning experiments, along with
Metropolis-Hastings mixing-matrix construction and spectral / community
analysis tools.

Generators
----------
- :class:`SBMGenerator` -- Stochastic Block Model (planted partition).
- :class:`ErdosRenyiGenerator` -- Erdos-Renyi with degree-matched probability.
- :class:`WattsStrogatzGenerator` -- Watts-Strogatz small-world.
- :func:`ring_of_cliques` -- Deterministic ring-of-cliques.
- :func:`exponential_graph` -- Deterministic exponential (base-2) graph.

Mixing matrices
---------------
- :func:`metropolis_hastings_weights` -- Doubly-stochastic MH weights.

Metrics
-------
- :func:`compute_spectral_gap`, :func:`compute_modularity`,
  :func:`compute_conductance`.

Orchestrator
------------
- :class:`GraphManager` -- Config-driven build / validate / persist.
"""

from superadditivity.graphs.erdos_renyi_generator import ErdosRenyiGenerator
from superadditivity.graphs.graph_manager import GraphManager
from superadditivity.graphs.graph_metrics import (
    compute_conductance,
    compute_modularity,
    compute_spectral_gap,
)
from superadditivity.graphs.mixing_matrix import metropolis_hastings_weights
from superadditivity.graphs.sbm_generator import SBMGenerator
from superadditivity.graphs.special_topologies import exponential_graph, ring_of_cliques
from superadditivity.graphs.watts_strogatz_generator import WattsStrogatzGenerator

__all__ = [
    "SBMGenerator",
    "ErdosRenyiGenerator",
    "WattsStrogatzGenerator",
    "ring_of_cliques",
    "exponential_graph",
    "metropolis_hastings_weights",
    "compute_spectral_gap",
    "compute_modularity",
    "compute_conductance",
    "GraphManager",
]
