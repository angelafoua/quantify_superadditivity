"""Local-only coordinator — identity mixing (no communication)."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

from superadditivity.training.decentralized_client import DecentralizedClient
from superadditivity.training.dsgd_coordinator import DSGDCoordinator
from superadditivity.training.lr_schedule import CosineDecaySchedule

logger = logging.getLogger(__name__)


class LocalOnlyCoordinator(DSGDCoordinator):
    """D-SGD with identity mixing matrix — each client trains independently.

    This baseline isolates the effect of communication: clients see the
    same data partition but never mix parameters.

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
        W = np.eye(n, dtype=np.float64)
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
        logger.info("Local-only coordinator: identity W (%d×%d)", n, n)
