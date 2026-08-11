"""Cosine-decay learning-rate schedule with optional linear warmup."""

from __future__ import annotations

import math
from typing import Optional


class CosineDecaySchedule:
    """Cosine annealing from ``lr_max`` to ``lr_min`` over ``total_rounds``.

    Parameters
    ----------
    lr_max:
        Peak learning rate (reached after warmup, if any).
    lr_min:
        Final learning rate.
    total_rounds:
        Total number of communication rounds.
    warmup_rounds:
        Linear warmup from ``lr_min`` to ``lr_max`` over this many rounds.
    """

    def __init__(
        self,
        lr_max: float = 0.1,
        lr_min: float = 1e-4,
        total_rounds: int = 300,
        warmup_rounds: int = 0,
    ) -> None:
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.total_rounds = total_rounds
        self.warmup_rounds = warmup_rounds

    def get_lr(self, round_num: int) -> float:
        """Return the learning rate for ``round_num`` (0-indexed)."""
        if round_num < self.warmup_rounds:
            frac = round_num / max(self.warmup_rounds, 1)
            return self.lr_min + (self.lr_max - self.lr_min) * frac

        decay_round = round_num - self.warmup_rounds
        decay_total = self.total_rounds - self.warmup_rounds
        cosine = 0.5 * (1 + math.cos(math.pi * decay_round / max(decay_total, 1)))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine
