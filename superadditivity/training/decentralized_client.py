"""A single decentralised client owning a model, data, and optimizer."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from superadditivity.datasets.client_dataset import ClientDataset
from superadditivity.training.local_trainer import LocalTrainer
from superadditivity.utils.seed import seed_worker

logger = logging.getLogger(__name__)


class DecentralizedClient:
    """One node in the decentralised federation.

    Parameters
    ----------
    client_id:
        Unique numeric identifier.
    model:
        The client's local model (already initialised).
    dataset:
        A :class:`ClientDataset` assigned to this client.
    device:
        Compute device.
    lr:
        Learning rate (updated externally by the coordinator).
    momentum:
        SGD momentum.
    weight_decay:
        L2 regularisation.
    batch_size:
        Mini-batch size for the local DataLoader.
    local_steps:
        Number of SGD steps per communication round.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataset: ClientDataset,
        device: torch.device,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        local_steps: int = 1,
    ) -> None:
        self.client_id = client_id
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.local_steps = local_steps

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.trainer = LocalTrainer()
        self._loader: Optional[DataLoader] = None

    def _get_loader(self) -> DataLoader:
        if self._loader is None:
            g = torch.Generator()
            g.manual_seed(self.client_id)
            self._loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=g,
            )
        return self._loader

    def set_lr(self, lr: float) -> None:
        """Update the learning rate for this client's optimizer."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def local_step(self, round_num: int) -> float:
        """Perform ``local_steps`` SGD updates and return the mean loss."""
        self.dataset.set_round(round_num)
        loader = self._get_loader()
        total_loss = 0.0
        steps_done = 0

        self.model.to(self.device)
        data_iter = iter(loader)

        for _ in range(self.local_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            data, target = batch[0].to(self.device), batch[1].to(self.device)
            loss = self.trainer.train_step(
                self.model, self.optimizer, data, target,
            )
            total_loss += loss
            steps_done += 1

        return total_loss / max(steps_done, 1)

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a copy of the model state dict (on CPU)."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load parameters into the local model."""
        self.model.load_state_dict(state_dict)
