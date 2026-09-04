#!/usr/bin/env python3
"""
EVIMO2 Dataset Explorer

This script summarizes the dataset structure without making assumptions
about the internal organization.

Example
-------
python tools/evimo/inspect_dataset.py \
    --root ~/HDD/EventDatasets/EVIMO2/raw
"""

from pathlib import Path
import argparse
import numpy as np


# ---------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------

def hr():
    print("=" * 90)


def section(title):
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)


# ---------------------------------------------------------------------
# Sequence inspection
# ---------------------------------------------------------------------

def inspect_sequence(sequence_dir: Path):

    npz_files = sorted(sequence_dir.rglob("*.npz"))

    summary = {
        "npz_files": len(npz_files),
        "arrays": 0,
        "events": 0,
        "frames": 0,
        "depth": 0,
        "poses": 0,
        "memory": 0,
    }

    print()
    hr()
    print(f"Sequence : {sequence_dir.name}")
    hr()

    if len(npz_files) == 0:
        print("No npz files found.")
        return summary

    for file in npz_files:

        try:
            data = np.load(file, allow_pickle=True)

        except Exception as e:
            print(f"[FAILED] {file.name}")
            print(e)
            continue

        summary["arrays"] += len(data.files)

        print(f"\n{file.relative_to(sequence_dir)}")

        for key in data.files:

            arr = data[key]

            summary["memory"] += arr.nbytes

            shape = arr.shape
            dtype = arr.dtype

            line = f"    {key:<30}"

            line += f"{str(shape):<20}"

            line += f"{str(dtype):<12}"

            if arr.ndim > 0:
                line += f"{arr.size:>12,d}"

            print(line)

            k = key.lower()

            if "event" in k:
                summary["events"] += arr.shape[0]

            if "image" in k or "frame" in k:
                summary["frames"] += arr.shape[0]

            if "depth" in k:
                summary["depth"] += arr.shape[0]

            if "pose" in k:
                summary["poses"] += arr.shape[0]

    section("Sequence Summary")

    print(f"NPZ files        : {summary['npz_files']}")
    print(f"Arrays           : {summary['arrays']}")
    print(f"Event samples    : {summary['events']:,}")
    print(f"Image frames     : {summary['frames']:,}")
    print(f"Depth frames     : {summary['depth']:,}")
    print(f"Pose samples     : {summary['poses']:,}")
    print(f"Memory           : {summary['memory']/1024**2:.2f} MB")

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
        help="EVIMO2 raw dataset root",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser()

    if not root.exists():
        raise FileNotFoundError(root)

    hr()
    print("EVIMO2 DATASET SUMMARY")
    hr()

    sequences = sorted([d for d in root.iterdir() if d.is_dir()])

    print(f"Dataset Root : {root}")
    print(f"Sequences    : {len(sequences)}")

    total = {
        "npz_files": 0,
        "arrays": 0,
        "events": 0,
        "frames": 0,
        "depth": 0,
        "poses": 0,
        "memory": 0,
    }

    for seq in sequences:

        s = inspect_sequence(seq)

        for k in total:
            total[k] += s[k]

    hr()
    print("OVERALL DATASET")
    hr()

    print(f"Sequences        : {len(sequences)}")
    print(f"NPZ files        : {total['npz_files']}")
    print(f"Arrays           : {total['arrays']}")
    print(f"Events           : {total['events']:,}")
    print(f"Images           : {total['frames']:,}")
    print(f"Depth            : {total['depth']:,}")
    print(f"Poses            : {total['poses']:,}")
    print(f"Dataset Memory   : {total['memory']/1024**3:.2f} GB")

    hr()


if __name__ == "__main__":
    main()