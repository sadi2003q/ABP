"""
MVSEC dataset implementation.

This module provides the MVSEC dataset interface used by the
EventCameraProject.
"""
from __future__ import annotations

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))




from omegaconf import DictConfig

from .base import BaseDataset
from src.utils.downloader import GoogleDriveDownloader

class MVSEC(BaseDataset):
    """
    MVSEC dataset implementation.
    """

    NAME = "mvsec"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        cfg_mvsec = cfg.mvsec

        # ------------------------------------------------------------------
        # Dataset directories
        # ------------------------------------------------------------------

        self.root = self.dataset_root / cfg_mvsec.dataset_dir

        self.raw_dir = self.root / cfg_mvsec.raw_dir

        self.processed_dir = self.root / cfg_mvsec.processed_dir

        self.metadata_dir = self.root / cfg_mvsec.metadata_dir

        self.cache_dir = self.root / cfg_mvsec.cache_dir

        # Download configuration
        self.download_source = cfg_mvsec.download.source

        self.download_url = cfg_mvsec.download.url

    # ------------------------------------------------------------------
    # Required Interface
    # ------------------------------------------------------------------

    def download(self, subset: str = "all") -> None:
        """
        Download the MVSEC dataset.

        Currently downloads the complete Google Drive folder.
        """

        self.logger.info("Preparing MVSEC download...")

        self.create_directories()

        downloader = GoogleDriveDownloader()

        downloader.download(
            url=self.download_url,
            output=self.raw_dir,
        )

        self.logger.info("MVSEC download finished.")

    def inspect(self):

        self.logger.info("Inspecting MVSEC dataset...")

        raise NotImplementedError

    def prepare(self):

        self.logger.info("Preparing MVSEC dataset...")

        raise NotImplementedError

    def verify(self):

        self.logger.info("Verifying MVSEC dataset...")

        raise NotImplementedError

    def clean(self):

        self.logger.info("Cleaning MVSEC dataset...")

        raise NotImplementedError

    def visualize(self):

        self.logger.info("Launching visualization...")

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------

    def info(self) -> dict:
        """
        Return basic information about the dataset.
        """

        return {

            "name": self.NAME,

            "root": str(self.root),

            "raw_dir": str(self.raw_dir),

            "processed_dir": str(self.processed_dir),

            "metadata_dir": str(self.metadata_dir),

            "cache_dir": str(self.cache_dir),

            "exists": self.root.exists(),

        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def create_directories(self) -> None:
        """
        Create dataset directories if they do not exist.
        """

        directories = [

            self.root,

            self.raw_dir,

            self.processed_dir,

            self.metadata_dir,

            self.cache_dir,

        ]

        for directory in directories:

            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:

        return f"MVSEC(root='{self.root}')"