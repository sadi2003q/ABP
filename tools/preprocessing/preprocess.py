"""
Preprocess every EVIMO2 sequence.

For every valid sequence directory this script generates all cached
preprocessing files required by the Dataset.

Currently generated caches

    cache/
        event_index.npz
        frame_motion.npz
        camera_motion.npz

Future preprocessing stages can be added without changing the Dataset.

Example
-------
python tools/preprocessing/preprocess.py \
    ~/HDD/EventDatasets/EVIMO2_official
"""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback

import src.preprocessing.event_index as event_index

import src.preprocessing.frame_motion as frame_motion
import src.preprocessing.camera_motion as camera_motion
import src.preprocessing.imu_index as imu_index

# ==========================================================
# Required files identifying an EVIMO2 sequence
# ==========================================================

REQUIRED_FILES = (
    "dataset_info.npz",
    "dataset_events_xy.npy",
    "dataset_events_t.npy",
    "dataset_events_p.npy",
)


# ==========================================================
# Sequence discovery
# ==========================================================

def is_sequence_directory(
    directory: Path,
) -> bool:
    """
    Return True if directory looks like an EVIMO2 sequence.
    """

    return all(
        (directory / f).exists()
        for f in REQUIRED_FILES
    )


def discover_sequences(
    dataset_root: Path,
):
    """
    Recursively discover all EVIMO2 sequences.
    """

    sequences = []

    for path in dataset_root.rglob("*"):

        if not path.is_dir():
            continue

        if is_sequence_directory(path):
            sequences.append(path)

    sequences.sort()

    return sequences


# ==========================================================
# Sequence preprocessing
# ==========================================================

def preprocess_sequence(
    sequence_dir: Path,
    overwrite: bool = False,
    event_index_only: bool = False,
    frame_motion_only: bool = False,
    camera_motion_only: bool = False,
    imu_index_only: bool = False,
):
    """
    Preprocess one sequence.
    """

    selected = (
        event_index_only
        or frame_motion_only
        or camera_motion_only
        or imu_index_only
    )

    run_event = event_index_only or not selected
    run_frame = frame_motion_only or not selected
    run_camera = camera_motion_only or not selected
    run_imu = imu_index_only or not selected

    cache_dir = sequence_dir / "cache"

    event_cache = cache_dir / "event_index.npz"
    frame_cache = cache_dir / "frame_motion.npz"
    camera_cache = cache_dir / "camera_motion.npz"
    imu_cache = cache_dir / "imu_index.npz"

    #
    # Event index
    #
    if run_event:

        if overwrite or not event_cache.exists():

            print("  Event index...")

            event_index.generate_event_index(
                sequence_dir,
            )

        else:

            print("  Event index already exists.")

    #
    # Frame motion
    #
    if run_frame:

        if overwrite or not frame_cache.exists():

            print("  Frame motion...")

            frame_motion.generate_frame_motion(
                sequence_dir,
            )

        else:

            print("  Frame motion already exists.")

    #
    # Camera motion
    #
    if run_camera:

        if overwrite or not camera_cache.exists():

            print("  Camera motion...")

            camera_motion.generate_camera_motion(
                sequence_dir,
            )

        else:

            print("  Camera motion already exists.")


    #
    # IMU index
    #
    if run_imu:

        if overwrite or not imu_cache.exists():

            print("  IMU index...")

            imu_index.generate_imu_index(
                sequence_dir,
            )

        else:

            print("  IMU index already exists.")
# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_root",
        type=Path,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--event-index",
        action="store_true",
        help="Generate event index cache only.",
    )

    parser.add_argument(
        "--frame-motion",
        action="store_true",
        help="Generate object/frame motion cache only.",
    )

    parser.add_argument(
        "--camera-motion",
        action="store_true",
        help="Generate camera motion cache only.",
    )

    parser.add_argument(
        "--imu-index",
        action="store_true",
        help="Generate IMU lookup cache only.",
    )

    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Process only the sequence whose relative path matches this string.",
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser()

    print("=" * 90)
    print("EVIMO2 PREPROCESSING")
    print("=" * 90)

    sequences = discover_sequences(
        dataset_root,
    )

    if args.sequence is not None:
        sequences = [
            s for s in sequences
            if args.sequence in str(s.relative_to(dataset_root))
        ]

    print(f"Selected {len(sequences)} sequence(s).")

    print(f"Found {len(sequences)} sequences.\n")

    success = 0
    failed = 0

    for i, sequence in enumerate(sequences, start=1):

        print(
            f"[{i}/{len(sequences)}] "
            f"{sequence.relative_to(dataset_root)}"
        )

        try:

            preprocess_sequence(
                sequence,
                overwrite=args.overwrite,
                event_index_only=args.event_index,
                frame_motion_only=args.frame_motion,
                camera_motion_only=args.camera_motion,
                imu_index_only=args.imu_index,
            )

            success += 1

            print("  ✓ Done\n")

        except Exception:

            failed += 1

            print("  ✗ Failed\n")

            traceback.print_exc()

    print("=" * 90)

    print("Finished")

    print(f"Successful : {success}")

    print(f"Failed     : {failed}")

    print("=" * 90)


if __name__ == "__main__":

    main()