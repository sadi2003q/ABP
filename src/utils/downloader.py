"""
Downloader utilities for EventCameraProject.

This module provides a common interface for downloading datasets,
pretrained models, and other project assets.

Currently supported:
    - Google Drive folders/files (via gdown)

Future:
    - HTTP/HTTPS
    - Kaggle
    - HuggingFace
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import gdown

logger = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """Raised when a download fails."""


class BaseDownloader(ABC):
    """
    Abstract downloader interface.
    """

    @abstractmethod
    def download(
        self,
        url: str,
        output: Path,
        overwrite: bool = False,
    ) -> None:
        """
        Download data.

        Parameters
        ----------
        url : str
            Download URL.

        output : Path
            Destination directory.

        overwrite : bool
            Overwrite existing files.
        """
        raise NotImplementedError

class GoogleDriveDownloader(BaseDownloader):
    """
    Google Drive downloader using gdown.
    """

    def download(
        self,
        url: str,
        output: Path,
        overwrite: bool = False,
    ) -> None:

        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        logger.info("-" * 80)
        logger.info("Google Drive Download")
        logger.info("URL    : %s", url)
        logger.info("Output : %s", output)
        logger.info("-" * 80)

        try:

            if "folders" in url:

                files = gdown.download_folder(
                    url=url,
                    output=str(output),
                    quiet=False,
                    resume=True,
                )

            else:

                files = [
                    gdown.download(
                        url=url,
                        output=str(output),
                        quiet=False,
                        fuzzy=True,
                    )
                ]

        except Exception as exc:

            raise DownloadError(str(exc)) from exc

        if files is None:
            raise DownloadError("Download failed.")

        logger.info("Downloaded %d file(s).", len(files))