"""
Abstract base class for all datasets.

Every supported dataset (MVSEC, DSEC, EVIMO, etc.) must inherit from
BaseDataset and implement the required interface.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path

from omegaconf import DictConfig

from src.utils.logger import get_logger


class BaseDataset(ABC):
    """
    Abstract dataset interface.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration.
    """

    NAME = "base"

    def __init__(self, cfg: DictConfig):

        self.cfg = cfg

        self.logger = get_logger(self.__class__.__name__)

        dataset_cfg = cfg.dataset

        self.dataset_root = Path(dataset_cfg.root)

    # ------------------------------------------------------------------
    # Required Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def download(self, subset: str = "all") -> None:
        """Download the dataset."""

    @abstractmethod
    def inspect(self):
        """Inspect the dataset."""

    @abstractmethod
    def prepare(self):
        """Prepare metadata."""

    @abstractmethod
    def verify(self):
        """Verify processed files."""

    @abstractmethod
    def clean(self):
        """Remove temporary files."""

    @abstractmethod
    def info(self):
        """Return dataset information."""

    @abstractmethod
    def visualize(self):
        """Visualize dataset samples."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """
        Check whether the dataset root exists.

        Returns
        -------
        bool
        """
        return self.dataset_root.exists()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root='{self.dataset_root}')"