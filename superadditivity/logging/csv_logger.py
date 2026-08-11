"""CSV logger with union-of-columns buffering."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from superadditivity.utils.io import ensure_dir

logger = logging.getLogger(__name__)


class CSVLogger:
    """Buffer rows and flush to a CSV file with the union of all columns.

    Parameters
    ----------
    path:
        Output CSV file path.
    flush_every:
        Flush buffer to disk every this many rows.
    """

    def __init__(self, path: str, flush_every: int = 50) -> None:
        self.path = Path(path)
        self.flush_every = flush_every
        self._buffer: List[Dict[str, Any]] = []
        self._all_keys: List[str] = []
        self._key_set: set = set()
        ensure_dir(self.path.parent)

    def log(self, row: Dict[str, Any]) -> None:
        """Append a row to the buffer."""
        for k in row:
            if k not in self._key_set:
                self._key_set.add(k)
                self._all_keys.append(k)
        self._buffer.append(row)

        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write all buffered rows to disk."""
        if not self._buffer:
            return

        file_exists = self.path.exists()
        existing_rows: List[Dict[str, Any]] = []

        if file_exists:
            with self.path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    existing_rows.append(r)
                    for k in r:
                        if k not in self._key_set:
                            self._key_set.add(k)
                            self._all_keys.append(k)

        all_rows = existing_rows + self._buffer

        with self.path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._all_keys, extrasaction="ignore")
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

        logger.debug("Flushed %d rows to %s", len(self._buffer), self.path)
        self._buffer.clear()

    def close(self) -> None:
        """Flush remaining rows."""
        self.flush()
