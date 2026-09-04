"""
Frame-to-IMU lookup cache.

For every frame, stores the range of IMU measurements that fall
within that frame interval.

This mirrors the event index cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


CACHE_FILENAME = "imu_index.npz"


@dataclass(slots=True)
class IMUIndexCache:

    imu_start: np.ndarray
    """
    Shape
    -----
    (N_frames,)
    """

    imu_end: np.ndarray
    """
    Shape
    -----
    (N_frames,)
    """


def build_imu_index(
    frame_timestamps: np.ndarray,
    imu_timestamps: np.ndarray,
) -> IMUIndexCache:
    """
    Build frame -> imu lookup.

    Every frame owns all IMU samples in

        [frame_i, frame_{i+1})

    The final frame owns all remaining IMU samples.
    """

    n_frames = len(frame_timestamps)

    imu_start = np.zeros(
        n_frames,
        dtype=np.uint32,
    )

    imu_end = np.zeros(
        n_frames,
        dtype=np.uint32,
    )

    imu_ptr = 0

    n_imu = len(imu_timestamps)

    for i in range(n_frames):

        t0 = frame_timestamps[i]

        if i + 1 < n_frames:
            t1 = frame_timestamps[i + 1]
        else:
            t1 = np.inf

        while imu_ptr < n_imu and imu_timestamps[imu_ptr] < t0:
            imu_ptr += 1

        imu_start[i] = imu_ptr

        while imu_ptr < n_imu and imu_timestamps[imu_ptr] < t1:
            imu_ptr += 1

        imu_end[i] = imu_ptr

    return IMUIndexCache(
        imu_start=imu_start,
        imu_end=imu_end,
    )


def save_imu_index(
    sequence_dir: Path,
    cache: IMUIndexCache,
):

    cache_dir = sequence_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    np.savez_compressed(
        cache_dir / CACHE_FILENAME,
        imu_start=cache.imu_start,
        imu_end=cache.imu_end,
    )


def load_imu_index(
    sequence_dir: Path,
) -> IMUIndexCache:

    data = np.load(
        sequence_dir / "cache" / CACHE_FILENAME
    )

    return IMUIndexCache(
        imu_start=data["imu_start"],
        imu_end=data["imu_end"],
    )

from src.data.evimo2._metadata import parse_sequence_metadata


def generate_imu_index(
    sequence_dir: str | Path,
):
    """
    Generate and save the IMU lookup cache for one sequence.

    Samsung Mono sequences contain no IMU and are skipped.
    """

    sequence_dir = Path(sequence_dir)

    frames, _, imu = parse_sequence_metadata(sequence_dir)

    sensor = sequence_dir.parts[-4]

    #
    # Samsung Mono has no IMU
    #
    if sensor not in imu:

        return

    frame_timestamps = np.asarray(
        [f.timestamp for f in frames],
        dtype=np.float64,
    )

    imu_timestamps = imu[sensor]["timestamps"]

    cache = build_imu_index(
        frame_timestamps,
        imu_timestamps,
    )

    save_imu_index(
        sequence_dir,
        cache,
    )