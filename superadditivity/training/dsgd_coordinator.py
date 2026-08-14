"""D-SGD coordinator: the main training loop.

Each round: LR schedule → local SGD → gossip mixing → eval → checkpoint → log.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from superadditivity.communication.gossip_mixer import GossipMixer
from superadditivity.training.decentralized_client import DecentralizedClient
from superadditivity.training.lr_schedule import CosineDecaySchedule

logger = logging.getLogger(__name__)


class DSGDCoordinator:
    """Orchestrates Decentralized SGD across all clients.

    Parameters
    ----------
    clients:
        List of :class:`DecentralizedClient` instances.
    mixing_matrix:
        Doubly-stochastic mixing matrix ``W`` of shape ``(n, n)``.
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
        Optional experiment evaluator for drift metrics.
    loggers:
        Optional list of logger objects (CSV, W&B).
    checkpoint_manager:
        Optional checkpoint manager.
    """

    def __init__(
        self,
        clients: List[DecentralizedClient],
        mixing_matrix: np.ndarray,
        lr_schedule: CosineDecaySchedule,
        total_rounds: int = 300,
        eval_every: int = 10,
        checkpoint_every: int = 50,
        output_dir: Optional[str] = None,
        evaluator: Any = None,
        loggers: Optional[list] = None,
        checkpoint_manager: Any = None,
    ) -> None:
        self.clients = clients
        self.n_clients = len(clients)
        self.mixer = GossipMixer(mix_device="cpu")
        self.W = torch.from_numpy(
            np.asarray(mixing_matrix, dtype=np.float64)
        )
        self.lr_schedule = lr_schedule
        self.total_rounds = total_rounds
        self.eval_every = eval_every
        self.checkpoint_every = checkpoint_every
        self.output_dir = Path(output_dir) if output_dir else None
        self.evaluator = evaluator
        self.loggers = loggers or []
        self.checkpoint_manager = checkpoint_manager

    def run(self) -> Dict[str, Any]:
        """Execute the full D-SGD training loop."""
        logger.info(
            "Starting D-SGD: %d clients, %d rounds", self.n_clients, self.total_rounds
        )
        history: Dict[str, list] = {"round": [], "lr": [], "mean_loss": [], "wall_time": []}
        t0 = time.time()

        for round_num in range(self.total_rounds):
            lr = self.lr_schedule.get_lr(round_num)

            for client in self.clients:
                client.set_lr(lr)

            losses = []
            for client in self.clients:
                loss = client.local_step(round_num)
                losses.append(loss)
            mean_loss = float(np.mean(losses))

            self.mixer.mix(self.clients, self.W)

            elapsed = time.time() - t0
            history["round"].append(round_num)
            history["lr"].append(lr)
            history["mean_loss"].append(mean_loss)
            history["wall_time"].append(elapsed)

            if round_num % 10 == 0 or round_num == self.total_rounds - 1:
                logger.info(
                    "Round %d/%d  lr=%.6f  loss=%.4f  time=%.1fs",
                    round_num, self.total_rounds, lr, mean_loss, elapsed,
                )

            for lg in self.loggers:
                lg.log({
                    "round": round_num, "lr": lr,
                    "mean_loss": mean_loss, "wall_time": elapsed,
                })

            if self.evaluator and (
                round_num % self.eval_every == 0
                or round_num == self.total_rounds - 1
            ):
                clients_dict = {i: c for i, c in enumerate(self.clients)}
                self.evaluator.evaluate(clients_dict, round_num)

            if self.checkpoint_manager and (
                round_num % self.checkpoint_every == 0
                or round_num == self.total_rounds - 1
            ):
                state_dicts_cp = [c.get_state_dict() for c in self.clients]
                self.checkpoint_manager.save(state_dicts_cp, round_num)

        logger.info("D-SGD complete. Total wall time: %.1fs", time.time() - t0)
        return history
