"""Device (GPU/CPU) selection and inspection helpers."""

from __future__ import annotations

import logging
import platform
import sys
from typing import Optional, Sequence

import torch

logger = logging.getLogger(__name__)


def select_device(gpu_ids: Optional[Sequence[int]] = None) -> torch.device:
    """Select a compute device."""
    if gpu_ids and torch.cuda.is_available():
        for gid in gpu_ids:
            if gid < torch.cuda.device_count():
                device = torch.device(f"cuda:{gid}")
                logger.info("Selected CUDA device %s (%s)", device, torch.cuda.get_device_name(gid))
                return device
        logger.warning("Requested GPU ids %s unavailable; falling back.", list(gpu_ids))

    if torch.cuda.is_available():
        logger.info("Using default CUDA device cuda:0")
        return torch.device("cuda:0")

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        logger.info("CUDA unavailable; using Apple MPS device.")
        return torch.device("mps")

    logger.info("No GPU available; using CPU.")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Return a dictionary describing the runtime environment."""
    info: dict = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    else:
        info["gpu_count"] = 0
        info["gpu_names"] = []
    return info
