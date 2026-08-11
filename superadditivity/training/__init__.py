"""Training coordinators, clients, and learning-rate schedules for D-SGD."""

from __future__ import annotations

from superadditivity.training.decentralized_client import DecentralizedClient
from superadditivity.training.dsgd_coordinator import DSGDCoordinator
from superadditivity.training.fedavg_coordinator import FedAvgCoordinator
from superadditivity.training.local_only_coordinator import LocalOnlyCoordinator
from superadditivity.training.local_trainer import LocalTrainer
from superadditivity.training.lr_schedule import CosineDecaySchedule

__all__ = [
    "DecentralizedClient",
    "DSGDCoordinator",
    "FedAvgCoordinator",
    "LocalOnlyCoordinator",
    "LocalTrainer",
    "CosineDecaySchedule",
]
