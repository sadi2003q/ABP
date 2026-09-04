"""
Logging utilities for EventCameraProject.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.config import load_config

_INITIALIZED = False


def setup_logging() -> None:
    """
    Configure project-wide logging.
    """

    global _INITIALIZED

    if _INITIALIZED:
        return

    cfg = load_config()

    handlers = []

    if cfg.logging.console:
        handlers.append(logging.StreamHandler())

    if cfg.logging.save_to_file:

        log_file = (
            Path(cfg.experiment.log_dir)
            / cfg.logging.filename
        )

        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format,
        datefmt=cfg.logging.datefmt,
        handlers=handlers,
    )

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
    """

    setup_logging()

    return logging.getLogger(name)