"""Deterministic seeding utilities.

Reproducibility is a first-class requirement. Every stochastic operation
derives its seed from a small set of top-level seeds via :func:`derived_seed`.

Seed decomposition
------------------
Each run is identified by a ``(run_seed, graph_seed)`` pair:

* ``run_seed``  -- weight initialisation, data shuffling, Dirichlet sampling.
* ``graph_seed`` -- SBM / ER / Watts-Strogatz edge generation only.

Derived seeds:

* weight init seed                : ``run_seed``
* Dirichlet partition seed        : ``run_seed + 1000``
* data shuffle seed (per client/round): ``run_seed + client_id * 10000 + round_num``
* graph generation seed           : ``graph_seed``
* probe-set selection seed        : ``999`` (fixed across ALL experiments)
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

PROBE_SEED: int = 999

_DIRICHLET_OFFSET = 1000
_CLIENT_STRIDE = 10000


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed every source of randomness used in the project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    logger.debug("All seeds set to %d (deterministic=%s)", seed, deterministic)


def derived_seed(
    run_seed: int,
    *,
    kind: str = "weight_init",
    client_id: int = 0,
    round_num: int = 0,
) -> int:
    """Compute a derived seed from a master ``run_seed``.

    Parameters
    ----------
    run_seed:
        The master run seed.
    kind:
        One of ``{"weight_init", "dirichlet", "shuffle"}``.
    client_id:
        Client index (used for ``"shuffle"``).
    round_num:
        Communication round (used for ``"shuffle"``).
    """
    if kind == "weight_init":
        return int(run_seed)
    if kind == "dirichlet":
        return int(run_seed + _DIRICHLET_OFFSET)
    if kind == "shuffle":
        return int(run_seed + client_id * _CLIENT_STRIDE + round_num)
    raise ValueError(f"Unknown derived-seed kind: {kind!r}")


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for :class:`torch.utils.data.DataLoader`."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
