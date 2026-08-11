"""FedAvg coordinator — uniform averaging as a special case of D-SGD."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

from superadditivity.training.decentralized_client import DecentralizedClient
from superadditivity.training.dsgd_coordinator import DSGDCoordinator
from superadditivity.training.lr_schedule import CosineDecaySchedule

logger = logging.getLogger(__name__)


class FedAvgCoordinator(DSGDCoordinator):
    """D-SGD with a uniform (fully-connected) mixing matrix.

    This is equivalent to FedAvg: every round, all clients average their
    models uniformly.

    Parameters
    ----------
    clients:
        List of :class:`DecentralizedClient` instances.
    lr_schedule:
        Cosine-decay schedule.
    total_rounds:
        Number of communication rounds.
    eval_every:
        Evaluate drift metrics every this many rounds.
    checkpoint_every:
        Save checkpoints every this many rounds.
    output_dir:
        Directory for saving checkpoints and logs.
    evaluator:
        Optional experiment evaluator.
    loggers:
        Optional list of loggers.
    checkpoint_manager:
        Optional checkpoint manager.
    """

    def __init__(
        self,
        clients: List[DecentralizedClient],
        lr_schedule: CosineDecaySchedule,
        total_rounds: int = 300,
        eval_every: int = 10,
        checkpoint_every: int = 50,
        output_dir: Optional[str] = None,
        evaluator: Any = None,
        loggers: Optional[list] = None,
        checkpoint_manager: Any = None,
    ) -> None:
        n = len(clients)
        W = np.ones((n, n), dtype=np.float64) / n
        super().__init__(
            clients=clients,
            mixing_matrix=W,
            lr_schedule=lr_schedule,
            total_rounds=total_rounds,
            eval_every=eval_every,
            checkpoint_every=checkpoint_every,
            output_dir=output_dir,
            evaluator=evaluator,
            loggers=loggers,
            checkpoint_manager=checkpoint_manager,
        )
        logger.info("FedAvg coordinator: uniform W (%d×%d)", n, n)
