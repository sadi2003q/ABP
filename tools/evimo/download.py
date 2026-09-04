#!/usr/bin/env python3
"""
Download the complete EVIMO2 dataset.

Example
-------
Download dataset

    python tools/evimo/download.py

Delete downloaded archives after extraction

    python tools/evimo/download.py --delete

Only download (no extraction)

    python tools/evimo/download.py --no-extract
"""

from pathlib import Path
import argparse
import tarfile
import urllib.request
import shutil

from tqdm import tqdm


# ==============================================================================
# Configuration
# ==============================================================================

BASE_URL = "https://obj.umiacs.umd.edu/evimo2v2"

FILES = [
    "flea3_7_imo.tar.gz",
    "flea3_7_sanity.tar.gz",
    "flea3_7_sanity_ll.tar.gz",
    "flea3_7_sfm.tar.gz",
    "flea3_7_sfm_ll.tar.gz",
    "left_camera_imo.tar.gz",
    "left_camera_imo_ll.tar.gz",
    "left_camera_sanity.tar.gz",
    "left_camera_sanity_ll.tar.gz",
    "left_camera_sfm.tar.gz",
    "left_camera_sfm_ll.tar.gz",
    "right_camera_imo.tar.gz",
    "right_camera_imo_ll.tar.gz",
    "right_camera_sanity.tar.gz",
    "right_camera_sanity_ll.tar.gz",
    "right_camera_sfm.tar.gz",
    "right_camera_sfm_ll.tar.gz",
    "samsung_mono_imo.tar.gz",
    "samsung_mono_imo_ll.tar.gz",
    "samsung_mono_sanity.tar.gz",
    "samsung_mono_sanity_ll.tar.gz",
    "samsung_mono_sfm.tar.gz",
    "samsung_mono_sfm_ll.tar.gz",
]


# ==============================================================================
# Progress Bar
# ==============================================================================

class DownloadProgressBar(tqdm):

    def update_to(self, blocks=1, block_size=1, total_size=None):

        if total_size is not None:
            self.total = total_size

        self.update(blocks * block_size - self.n)


# ==============================================================================
# Download
# ==============================================================================

def download_file(url: str, output_file: Path):

    if output_file.exists():

        print(f"[SKIP] {output_file.name}")

        return

    print(f"\nDownloading {output_file.name}")

    with DownloadProgressBar(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=output_file.name,
    ) as progress:

        urllib.request.urlretrieve(
            url,
            filename=output_file,
            reporthook=progress.update_to,
        )


# ==============================================================================
# Extract
# ==============================================================================

def extract_archive(archive: Path, output_dir: Path):

    print(f"Extracting {archive.name}")

    with tarfile.open(archive, "r:gz") as tar:

        tar.extractall(output_dir)


# ==============================================================================
# Main
# ==============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--format",
        default="npz",
        choices=["npz", "txt"],
        help="Dataset format",
    )

    parser.add_argument(
        "--output",
        default="~/HDD/EventDatasets/EVIMO2/raw",
        help="Output directory",
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete archives after extraction",
    )

    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only download archives",
    )

    args = parser.parse_args()

    output_dir = Path(args.output).expanduser()

    archive_dir = output_dir / "_archives"

    output_dir.mkdir(parents=True, exist_ok=True)

    archive_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"{BASE_URL}{args.format}"

    print("=" * 80)
    print("EVIMO2 Downloader")
    print("=" * 80)
    print("Format :", args.format)
    print("Output :", output_dir)
    print()

    for filename in FILES:

        dataset_file = f"{args.format}_{filename}"

        url = f"{base_url}/{dataset_file}"

        archive = archive_dir / dataset_file

        download_file(url, archive)

        if not args.no_extract:

            extract_archive(archive, output_dir)

            if args.delete:

                archive.unlink()

    print()
    print("=" * 80)
    print("Finished.")
    print("=" * 80)


if __name__ == "__main__":

    main()