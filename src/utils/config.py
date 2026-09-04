"""
Configuration management for EventCameraProject.

This module loads all YAML configuration files from the project's
`configs/` directory, merges them into a single OmegaConf object,
expands user paths, and creates common project directories.

Example
-------
>>> from src.utils.config import load_config
>>> cfg = load_config()
>>> print(cfg.dataset.root)
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration files are invalid."""


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Configuration directory
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config() -> DictConfig:
    """
    Load and merge all YAML configuration files.

    Every `.yaml` file inside the configs directory is loaded in
    alphabetical order and merged into a single configuration object.

    Returns
    -------
    DictConfig
        Unified project configuration.
    """
    config_files = sorted(CONFIG_DIR.glob("*.yaml"))

    if not config_files:
        raise ConfigError(f"No configuration files found in: {CONFIG_DIR}")

    logger.info("Loading configuration files...")

    configs = []

    for file in config_files:
        logger.info("Loading %s", file.name)
        configs.append(OmegaConf.load(file))

    cfg = OmegaConf.merge(*configs)

    _expand_paths(cfg)
    _create_directories(cfg)

    logger.info("Configuration loaded successfully.")

    return cfg


def project_root() -> Path:
    """
    Return the project root directory.

    Returns
    -------
    Path
        Project root.
    """
    return PROJECT_ROOT


def config_directory() -> Path:
    """
    Return the configuration directory.

    Returns
    -------
    Path
        Configuration directory.
    """
    return CONFIG_DIR


def _expand_paths(cfg: DictConfig) -> None:
    """
    Expand '~' to the user's home directory for all string paths.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration.
    """

    def recurse(node):
        if isinstance(node, DictConfig):
            for key in node.keys():
                value = node[key]

                if isinstance(value, DictConfig):
                    recurse(value)

                elif isinstance(value, str) and value.startswith("~"):
                    node[key] = str(Path(value).expanduser())

    recurse(cfg)


def _create_directories(cfg: DictConfig) -> None:
    """
    Create project directories defined in the configuration.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration.
    """

    directories = []

    if "experiment" in cfg:

        for key, value in cfg.experiment.items():

            if key.endswith("_dir"):

                directories.append(PROJECT_ROOT / value)

    for directory in directories:

        directory.mkdir(parents=True, exist_ok=True)

        logger.info("Directory ready: %s", directory)