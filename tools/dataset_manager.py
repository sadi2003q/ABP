"""
Dataset Manager for EventCameraProject.

This module provides a unified command line interface for managing
datasets used by the project.
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import get_logger
from tools.dataset.registry import (
    get_dataset_class,
    list_datasets,
)

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command line argument parser.
    """

    parser = argparse.ArgumentParser(
        description="EventCameraProject Dataset Manager"
    )

    parser.add_argument(
        "command",
        choices=[
            "list",
            "info",
            "download",
            "inspect",
            "prepare",
            "verify",
            "clean",
        ],
        help="Command to execute.",
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name.",
    )

    parser.add_argument(
        "--subset",
        default="all",
        help="Subset or sequence to download.",
    )

    return parser


def main() -> int:

    parser = build_parser()

    args = parser.parse_args()

    # -------------------------------------------------------
    # List supported datasets
    # -------------------------------------------------------

    if args.command == "list":

        print()

        print("Supported datasets")

        print("------------------")

        for name in list_datasets():
            print(name)

        print()

        return 0

    # -------------------------------------------------------

    if args.dataset is None:

        parser.error("Dataset name is required.")

    cfg = load_config()

    dataset_class = get_dataset_class(args.dataset)

    dataset = dataset_class(cfg)

    # -------------------------------------------------------

    if args.command == "info":

        print(
            json.dumps(
                dataset.info(),
                indent=4,
            )
        )

    elif args.command == "download":

        dataset.download(
            subset=args.subset,
        )

    elif args.command == "inspect":

        dataset.inspect()

    elif args.command == "prepare":

        dataset.prepare()

    elif args.command == "verify":

        dataset.verify()

    elif args.command == "clean":

        dataset.clean()

    else:

        raise RuntimeError("Unknown command.")

    return 0


if __name__ == "__main__":

    sys.exit(main())