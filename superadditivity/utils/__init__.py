"""Shared, dependency-free utilities used across the entire package."""

from __future__ import annotations

from superadditivity.utils.device import get_device_info, select_device
from superadditivity.utils.io import (
    ensure_dir,
    get_git_hash,
    load_json,
    read_hdf5_dataset,
    save_json,
    write_hdf5_dataset,
)
from superadditivity.utils.seed import (
    derived_seed,
    seed_worker,
    set_all_seeds,
)

__all__ = [
    "set_all_seeds",
    "derived_seed",
    "seed_worker",
    "select_device",
    "get_device_info",
    "ensure_dir",
    "get_git_hash",
    "load_json",
    "save_json",
    "read_hdf5_dataset",
    "write_hdf5_dataset",
]
