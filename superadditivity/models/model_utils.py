"""Model utility functions.

Deterministic initialization, cloning, parameter counting, state-dict
averaging, and consensus-error computation for federated learning.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from superadditivity.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


def init_weights(model: nn.Module, seed: int) -> None:
    """Deterministic weight initialisation.

    Applies Kaiming-uniform for :class:`nn.Conv2d`, ones/zeros for
    :class:`nn.BatchNorm2d`, and normal(0, 0.01) for :class:`nn.Linear`.

    Parameters
    ----------
    model:
        The model to initialise in-place.
    seed:
        RNG seed for reproducibility.
    """
    set_all_seeds(seed)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    logger.debug("Weights initialised with seed %d", seed)


def clone_model(model: nn.Module) -> nn.Module:
    """Deep-copy a model (architecture + parameters + buffers).

    Parameters
    ----------
    model:
        Source model.

    Returns
    -------
    nn.Module
        An independent copy of *model*.
    """
    return copy.deepcopy(model)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Parameters
    ----------
    model:
        The model to inspect.
    trainable_only:
        If ``True`` (default), count only parameters with
        ``requires_grad=True``.

    Returns
    -------
    int
        Total number of scalar parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def average_state_dicts(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute a (weighted) average of state dicts in float64.

    Parameters
    ----------
    state_dicts:
        Sequence of ``state_dict()`` outputs with identical keys.
    weights:
        Per-model weights (must sum to 1). ``None`` for uniform average.

    Returns
    -------
    dict
        Averaged state dict with the same keys, cast back to each key's
        original dtype.

    Raises
    ------
    ValueError
        If *state_dicts* is empty or *weights* length mismatches.
    """
    n = len(state_dicts)
    if n == 0:
        raise ValueError("state_dicts must be non-empty")

    if weights is None:
        w = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(
                f"weights length ({len(weights)}) != state_dicts length ({n})"
            )
        w = list(weights)

    avg: Dict[str, torch.Tensor] = {}
    keys = list(state_dicts[0].keys())

    for key in keys:
        original_dtype = state_dicts[0][key].dtype
        acc = torch.zeros_like(state_dicts[0][key], dtype=torch.float64)
        for i in range(n):
            acc.add_(state_dicts[i][key].to(dtype=torch.float64), alpha=w[i])
        avg[key] = acc.to(dtype=original_dtype)

    return avg


def consensus_error(models: Sequence[nn.Module]) -> float:
    """Compute mean L2 deviation from the global average of flattened params.

    Uses a two-pass streaming approach on CPU to avoid allocating an
    ``(N, D)`` matrix that could cause OOM with many large models.

    Pass 1: accumulate the global mean parameter vector.
    Pass 2: accumulate squared deviations from the mean.

    Parameters
    ----------
    models:
        Sequence of models with identical architectures.

    Returns
    -------
    float
        Mean L2 norm of the deviation of each model's flattened parameters
        from the global average.

    Raises
    ------
    ValueError
        If *models* is empty.
    """
    n = len(models)
    if n == 0:
        raise ValueError("models must be non-empty")

    def _flatten(model: nn.Module) -> torch.Tensor:
        """Flatten all parameters into a single 1-D float64 CPU vector."""
        return torch.cat(
            [p.detach().cpu().to(torch.float64).reshape(-1) for p in model.parameters()]
        )

    # Pass 1: compute global mean (streaming accumulation)
    mean_vec: Optional[torch.Tensor] = None
    for model in models:
        flat = _flatten(model)
        if mean_vec is None:
            mean_vec = torch.zeros_like(flat)
        mean_vec.add_(flat)
    assert mean_vec is not None
    mean_vec.div_(n)

    # Pass 2: compute mean L2 deviation
    total_l2 = 0.0
    for model in models:
        flat = _flatten(model)
        diff = flat - mean_vec
        total_l2 += torch.norm(diff, p=2).item()

    result = total_l2 / n
    logger.debug("Consensus error across %d models: %.6e", n, result)
    return result
