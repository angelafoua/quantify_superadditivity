"""W-weighted gossip mixing for decentralized SGD.

Implements the core gossip averaging step: given a doubly-stochastic mixing
matrix ``W`` and a set of clients, each client's parameters are replaced by
the weighted combination ``new[i] = sum_j W[i,j] * params[j]``.

This is the single most failure-prone step in decentralized FL; an error
here masquerades as representation drift. The implementation is tested
extensively in ``tests/test_gossip_mixer.py``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Protocol, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _HasModel(Protocol):
    """Structural type for objects that expose a ``.model`` attribute."""

    model: nn.Module


class GossipMixer:
    """Apply doubly-stochastic mixing matrix ``W`` to client parameters.

    Only trainable parameters are mixed.  BatchNorm running statistics
    (``running_mean``, ``running_var``, ``num_batches_tracked``) are
    buffers, not parameters, and are excluded from the gossip average so
    each client retains its own local batch-statistics estimate.

    Parameters
    ----------
    mix_device:
        Device on which to perform the mixing linear algebra (typically
        ``"cpu"`` to avoid GPU-memory pressure).
    dtype:
        Floating-point dtype for the mixing computation. Must be at least
        float64 for numerical stability with large networks.
    """

    def __init__(
        self,
        mix_device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.mix_device = torch.device(mix_device)
        self.dtype = dtype
        logger.info(
            "GossipMixer initialised (device=%s, dtype=%s)",
            self.mix_device,
            self.dtype,
        )

    @staticmethod
    def _param_keys(model: nn.Module) -> frozenset[str]:
        """Return the state-dict keys that correspond to trainable parameters."""
        return frozenset(name for name, _ in model.named_parameters())

    def mix(
        self,
        clients: Sequence[_HasModel],
        W: torch.Tensor,
    ) -> None:
        """Apply gossip mixing in-place to all client models.

        For every parameter tensor, stacks the N client copies into an
        ``(N, P)`` matrix, left-multiplies by ``W``, and writes the result
        back into each client's model.

        Parameters
        ----------
        clients:
            Sequence of client objects, each with a ``.model`` attribute
            that is an :class:`nn.Module`.
        W:
            Doubly-stochastic mixing matrix of shape ``(N, N)`` where
            ``N = len(clients)``.

        Raises
        ------
        ValueError
            If ``W`` shape does not match the number of clients.
        """
        n = len(clients)
        if W.shape != (n, n):
            raise ValueError(
                f"W shape {W.shape} does not match number of clients ({n})"
            )

        W_mix = W.to(device=self.mix_device, dtype=self.dtype)

        # Check whether models already live on the mixing device
        first_param = next(clients[0].model.parameters())
        models_on_mix_device = first_param.device == self.mix_device

        if models_on_mix_device:
            self._mix_in_place(clients, W_mix, n)
        else:
            self._mix_to_new_dicts(clients, W_mix, n)

        logger.debug("Gossip mixing applied to %d clients", n)

    def _mix_in_place(
        self,
        clients: Sequence[_HasModel],
        W: torch.Tensor,
        n: int,
    ) -> None:
        """Mix when all models are already on the mixing device."""
        param_keys = self._param_keys(clients[0].model)

        state_dicts: List[Dict[str, torch.Tensor]] = [
            clients[i].model.state_dict() for i in range(n)
        ]
        keys = [k for k in state_dicts[0].keys() if k in param_keys]

        for key in keys:
            tensors = [state_dicts[i][key] for i in range(n)]
            original_shape = tensors[0].shape
            original_dtype = tensors[0].dtype

            flat = torch.stack(
                [t.to(dtype=self.dtype).reshape(-1) for t in tensors], dim=0
            )  # (N, P)

            # W @ flat => (N, P), new[i] = sum_j W[i,j] * params[j]
            mixed = W @ flat  # (N, P)

            for i in range(n):
                new_val = mixed[i].reshape(original_shape).to(dtype=original_dtype)
                state_dicts[i][key] = new_val

        for i in range(n):
            clients[i].model.load_state_dict(state_dicts[i])

    def _mix_to_new_dicts(
        self,
        clients: Sequence[_HasModel],
        W: torch.Tensor,
        n: int,
    ) -> None:
        """Mix when models live on a different device than the mixer.

        Pulls parameters to the mixing device one key at a time, computes
        the weighted combination, and writes results back in-place via
        ``param.data.copy_()``.  This avoids allocating a full duplicate
        set of model tensors on the original device, which would otherwise
        double GPU memory for all N clients simultaneously.

        Buffers (e.g. BatchNorm running stats) are not parameters and are
        left untouched — each client keeps its own local batch-statistics.
        """
        param_keys = self._param_keys(clients[0].model)

        # Build a {name: param} lookup for every client so we can write
        # mixed values back in-place without load_state_dict.
        client_params: List[Dict[str, nn.Parameter]] = [
            dict(clients[i].model.named_parameters()) for i in range(n)
        ]

        for name in client_params[0]:
            if name not in param_keys:
                continue

            original_dtype = client_params[0][name].dtype
            original_device = client_params[0][name].device
            original_shape = client_params[0][name].shape

            # Pull this parameter from every client to the mixing device
            flat = torch.stack(
                [
                    client_params[i][name]
                    .detach()
                    .to(device=self.mix_device, dtype=self.dtype)
                    .reshape(-1)
                    for i in range(n)
                ],
                dim=0,
            )  # (N, P)

            mixed = W @ flat  # (N, P)

            # Write back in-place — no new GPU allocation
            for i in range(n):
                client_params[i][name].data.copy_(
                    mixed[i].reshape(original_shape).to(
                        dtype=original_dtype, device=original_device
                    )
                )
