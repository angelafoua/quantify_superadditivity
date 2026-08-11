"""Provenance metadata store — captures git, host, and seed info."""

from __future__ import annotations

import logging
import platform
import sys
import time
from typing import Any, Dict, Optional

from superadditivity.utils.device import get_device_info
from superadditivity.utils.io import get_git_branch, get_git_hash, is_git_dirty, save_json

logger = logging.getLogger(__name__)


class MetadataStore:
    """Collect and persist experiment provenance metadata.

    Parameters
    ----------
    output_path:
        Path to save the metadata JSON.
    """

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self._data: Dict[str, Any] = {}

    def collect(
        self,
        run_seed: int,
        graph_seed: int,
        config: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Gather all provenance information."""
        self._data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "git_hash": get_git_hash(short=True),
            "git_branch": get_git_branch(),
            "git_dirty": is_git_dirty(),
            "run_seed": run_seed,
            "graph_seed": graph_seed,
            "device_info": get_device_info(),
        }
        if config is not None:
            self._data["config"] = config

        return self._data

    def save(self) -> None:
        """Persist metadata to JSON."""
        if not self._data:
            logger.warning("No metadata collected; nothing to save.")
            return
        save_json(self._data, self.output_path)
        logger.info("Metadata saved to %s", self.output_path)
