"""superadditivity: Beyond Additive Heterogeneity — Quantifying Superadditive
Interaction Effects in Decentralized Federated Learning.

This package implements the full experimental pipeline for measuring the
interaction term I between data heterogeneity and network topology in DFL,
using layer-wise CKA as the primary drift metric.

Modules
-------
datasets       -- Multi-dataset loading (CIFAR-100, CIFAR-10, EMNIST), semantic
                  clustering, Dirichlet/quantity-skew partitioning.
graphs         -- SBM / ER / Watts-Strogatz / ring-of-cliques / exponential
                  topologies and mixing matrices.
models         -- ResNet-18 and 4-layer ConvNet for CIFAR, with feature extraction.
training       -- Decentralized clients and D-SGD / FedAvg / local coordinators.
communication  -- W-weighted gossip mixing.
evaluation     -- Representation extraction and drift metrics (CKA/RSA/MMD/Fisher).
logging        -- W&B / CSV logging, checkpointing, metadata.
analysis       -- Post-hoc statistical analysis and superadditivity quantification.
visualization  -- Publication-quality figures.
utils          -- Seeding, device management, I/O helpers.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
