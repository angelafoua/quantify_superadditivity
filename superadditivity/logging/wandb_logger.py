"""Weights & Biases logger and composite logger."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WandbLogger:
    """Thin wrapper around ``wandb.log``.

    Parameters
    ----------
    project:
        W&B project name.
    run_name:
        W&B run name.
    config:
        Configuration dict to log.
    enabled:
        If ``False``, all calls are no-ops.
    """

    def __init__(
        self,
        project: str = "superadditivity",
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self._run = None

        if not enabled:
            return

        try:
            import wandb
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                reinit=True,
            )
            logger.info("W&B run initialized: %s/%s", project, run_name)
        except ImportError:
            logger.warning("wandb not installed; logging disabled.")
            self.enabled = False
        except Exception as e:
            logger.warning("wandb init failed: %s; logging disabled.", e)
            self.enabled = False

    def log(self, row: Dict[str, Any]) -> None:
        """Log a dictionary of metrics."""
        if not self.enabled:
            return
        try:
            import wandb
            wandb.log(row)
        except Exception as e:
            logger.debug("wandb.log failed: %s", e)

    def close(self) -> None:
        """Finish the W&B run."""
        if not self.enabled or self._run is None:
            return
        try:
            self._run.finish()
        except Exception as e:
            logger.debug("wandb finish failed: %s", e)


class CompositeLogger:
    """Fan out ``log()`` calls to multiple loggers.

    Parameters
    ----------
    loggers:
        List of logger objects (each must have a ``.log(dict)`` method).
    """

    def __init__(self, loggers: Optional[List] = None) -> None:
        self.loggers = loggers or []

    def log(self, row: Dict[str, Any]) -> None:
        for lg in self.loggers:
            lg.log(row)

    def close(self) -> None:
        for lg in self.loggers:
            if hasattr(lg, "close"):
                lg.close()
