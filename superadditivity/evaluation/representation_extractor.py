"""Representation extraction via forward hooks.

Uses PyTorch forward hooks to capture intermediate layer activations from
arbitrary named modules. Convolutional outputs are spatially pooled via
global average pooling (mean over spatial dims). The ``fc`` layer captures
its *input* (i.e. the penultimate representation).

All extracted representations are returned in float64.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RepresentationExtractor:
    """Extract intermediate representations from models via forward hooks.

    Parameters
    ----------
    layer_names:
        Names of the layers to extract from (e.g. ``["layer1", "layer4", "fc"]``).
        Must correspond to named children / modules of the model.
    device:
        Device for inference (e.g. ``"cpu"`` or ``"cuda:0"``).

    Examples
    --------
    >>> extractor = RepresentationExtractor(["layer4", "fc"], device="cpu")
    >>> layer_emb = extractor.extract(clients, probe_loader)
    >>> community_emb = extractor.extract_community_embeddings(
    ...     clients, probe_loader, community_map
    ... )
    """

    def __init__(
        self,
        layer_names: Sequence[str],
        device: str = "cpu",
    ) -> None:
        self.layer_names = list(layer_names)
        self.device = torch.device(device)

    def _extract_single(
        self,
        model: nn.Module,
        probe_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, np.ndarray]:
        """Extract representations from a single model.

        Registers forward hooks on each requested layer, runs the probe set
        through the model, then removes all hooks.

        For convolutional layers the output is globally average-pooled over
        spatial dimensions ``(H, W)``. For the ``fc`` layer the hook
        captures the *input* to the layer (penultimate features).

        Parameters
        ----------
        model:
            The neural network to probe.
        probe_loader:
            DataLoader yielding ``(images, labels)`` batches for the
            fixed probe set.

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from layer name to activation matrix ``(n_probes, d)``
            in float64.
        """
        model = model.to(self.device)
        model.eval()

        activations: Dict[str, List[np.ndarray]] = {
            name: [] for name in self.layer_names
        }
        handles: List[torch.utils.hooks.RemovableHook] = []

        try:
            for name in self.layer_names:
                module = dict(model.named_modules()).get(name)
                if module is None:
                    raise ValueError(
                        f"Layer {name!r} not found in model. "
                        f"Available: {[n for n, _ in model.named_modules() if n]}"
                    )

                if name == "fc":
                    # Capture the input to the fc layer (penultimate features)
                    hook = self._make_input_hook(activations, name)
                else:
                    # Capture the output; GAP for conv layers
                    hook = self._make_output_hook(activations, name)

                handle = module.register_forward_hook(hook)
                handles.append(handle)

            # Run the probe set
            with torch.no_grad():
                for batch in probe_loader:
                    images = batch[0].to(self.device)
                    _ = model(images)

        finally:
            # Always remove hooks
            for handle in handles:
                handle.remove()

        # Concatenate and convert to float64
        result: Dict[str, np.ndarray] = {}
        for name in self.layer_names:
            result[name] = np.concatenate(activations[name], axis=0).astype(
                np.float64
            )

        return result

    @staticmethod
    def _make_output_hook(
        activations: Dict[str, List[np.ndarray]],
        name: str,
    ) -> Callable[..., None]:
        """Create a forward hook that captures (and GAPs) the layer output.

        Parameters
        ----------
        activations:
            Accumulator dict to append activations into.
        name:
            Layer name key.

        Returns
        -------
        Callable
            Hook function.
        """

        def hook(
            module: nn.Module,
            input: Any,
            output: torch.Tensor,
        ) -> None:
            act = output.detach()
            # Global average pooling for conv outputs: (B, C, H, W) -> (B, C)
            if act.dim() == 4:
                act = act.mean(dim=(2, 3))
            activations[name].append(act.cpu().numpy())

        return hook

    @staticmethod
    def _make_input_hook(
        activations: Dict[str, List[np.ndarray]],
        name: str,
    ) -> Callable[..., None]:
        """Create a forward hook that captures the input to a layer.

        Parameters
        ----------
        activations:
            Accumulator dict to append activations into.
        name:
            Layer name key.

        Returns
        -------
        Callable
            Hook function.
        """

        def hook(
            module: nn.Module,
            input: Any,
            output: torch.Tensor,
        ) -> None:
            # input is a tuple; take the first element
            act = input[0].detach()
            if act.dim() == 4:
                act = act.mean(dim=(2, 3))
            activations[name].append(act.cpu().numpy())

        return hook

    def extract(
        self,
        clients: Dict[int, Any],
        probe_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Extract representations from all clients.

        Parameters
        ----------
        clients:
            Mapping from client ID to client object. Each client must have
            a ``.model`` attribute that is an ``nn.Module``.
        probe_loader:
            DataLoader for the fixed probe set.

        Returns
        -------
        Dict[str, Dict[int, np.ndarray]]
            Nested mapping: ``layer_name -> client_id -> activations (n, d)``.
        """
        result: Dict[str, Dict[int, np.ndarray]] = {
            name: {} for name in self.layer_names
        }

        for client_id in sorted(clients.keys()):
            client = clients[client_id]
            model = client.model
            layer_activations = self._extract_single(model, probe_loader)

            for name in self.layer_names:
                result[name][client_id] = layer_activations[name]

        logger.info(
            "Extracted representations from %d clients at layers %s",
            len(clients),
            self.layer_names,
        )
        return result

    def extract_community_embeddings(
        self,
        clients: Dict[int, Any],
        probe_loader: torch.utils.data.DataLoader,
        community_map: Dict[int, int],
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Extract per-community averaged embeddings.

        First extracts per-client representations, then averages them
        within each community.

        Parameters
        ----------
        clients:
            Mapping from client ID to client object with a ``.model``
            attribute.
        probe_loader:
            DataLoader for the fixed probe set.
        community_map:
            Mapping from client ID to community ID.

        Returns
        -------
        Dict[str, Dict[int, np.ndarray]]
            Nested mapping: ``layer_name -> community_id -> averaged
            activations (n_probes, d)`` in float64.
        """
        # Extract per-client first
        all_client_emb = self.extract(clients, probe_loader)

        result: Dict[str, Dict[int, np.ndarray]] = {
            name: {} for name in self.layer_names
        }

        # Group client IDs by community
        communities: Dict[int, List[int]] = {}
        for client_id, comm_id in community_map.items():
            if client_id in clients:
                communities.setdefault(comm_id, []).append(client_id)

        for name in self.layer_names:
            for comm_id, client_ids in sorted(communities.items()):
                # Average across clients in this community
                embs = [all_client_emb[name][cid] for cid in client_ids]
                avg = np.mean(embs, axis=0).astype(np.float64)
                result[name][comm_id] = avg

        logger.info(
            "Computed community embeddings for %d communities at layers %s",
            len(communities),
            self.layer_names,
        )
        return result
