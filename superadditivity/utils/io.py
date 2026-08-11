"""I/O helpers: HDF5 datasets, JSON, git metadata and path utilities."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional, Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create ``path`` (and parents) if needed and return it as a :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: PathLike, indent: int = 2) -> None:
    """Serialise ``obj`` to JSON at ``path``."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    logger.debug("Wrote JSON to %s", p)


def load_json(path: PathLike) -> Any:
    """Load JSON from ``path``."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serialisable")


def write_hdf5_dataset(
    path: PathLike,
    key: str,
    array: np.ndarray,
    attrs: Optional[dict] = None,
    compression: Optional[str] = "gzip",
) -> None:
    """Write ``array`` to ``path`` under hierarchical ``key``."""
    p = Path(path)
    ensure_dir(p.parent)
    with h5py.File(p, "a") as fh:
        if key in fh:
            del fh[key]
        dset = fh.create_dataset(key, data=array, compression=compression)
        if attrs:
            for k, v in attrs.items():
                dset.attrs[k] = v


def read_hdf5_dataset(path: PathLike, key: str) -> np.ndarray:
    """Read the dataset stored at hierarchical ``key`` in ``path``."""
    with h5py.File(Path(path), "r") as fh:
        if key not in fh:
            raise KeyError(f"Key {key!r} not found in {path}")
        return fh[key][()]


def write_hdf5_attrs(path: PathLike, group: str, attrs: dict) -> None:
    """Attach attributes to a (possibly new) group in an HDF5 file."""
    p = Path(path)
    ensure_dir(p.parent)
    with h5py.File(p, "a") as fh:
        grp = fh.require_group(group)
        for k, v in attrs.items():
            grp.attrs[k] = v


def get_git_hash(short: bool = True) -> str:
    """Return the current git commit hash, or ``"unknown"``."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_branch() -> str:
    """Return the current git branch name, or ``"unknown"``."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_git_dirty() -> bool:
    """Return ``True`` if the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return bool(out.decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
