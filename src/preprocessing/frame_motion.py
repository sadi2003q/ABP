"""
Frame-to-frame object motion computation.

This module computes per-frame object motion in the world frame.

Unlike object_motion.py, which reconstructs complete trajectories,
this module computes incremental motion between consecutive frames.

The cached motion quantities are objective geometric measurements and
contain no thresholding or dynamic/static classification. Those
decisions are intentionally deferred to the Dataset during training.

Cache Contents
--------------
frame_ids
timestamps
object_ids
delta_position
speed
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pathlib import Path

from src.preprocessing.cache import (
    save_npz,
    load_npz,
)

from src.preprocessing.object_motion import (
    ObjectTrajectory,
    collect_world_trajectories,
)


CACHE_FILENAME = "frame_motion.npz"

@dataclass(slots=True)
class FrameMotionCache:
    """
    Cached frame-to-frame object motion.

    Parameters
    ----------
    frame_ids
        Frame identifiers.

    timestamps
        Frame timestamps.

    object_ids
        Object IDs in column order.

    delta_position
        World-frame displacement between consecutive frames.

        Shape
        -----
        (N_frames, N_objects, 3)

        Units
        -----
        meters

    speed
        Instantaneous object speed.

        Shape
        -----
        (N_frames, N_objects)

        Units
        -----
        meters / second
    """

    frame_ids: np.ndarray
    timestamps: np.ndarray

    object_ids: np.ndarray

    delta_position: np.ndarray
    speed: np.ndarray

def save_frame_motion(
    sequence_dir: str | Path,
    cache: FrameMotionCache,
) -> Path:
    """
    Save the computed frame-motion cache.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.

    cache
        Frame-motion cache returned by
        compute_frame_motion().

    Returns
    -------
    Path
        Saved cache file.
    """

    return save_npz(
        sequence_dir,
        CACHE_FILENAME,
        frame_ids=cache.frame_ids,
        timestamps=cache.timestamps,
        object_ids=cache.object_ids,
        delta_position=cache.delta_position,
        speed=cache.speed,
    )

def load_frame_motion(
    sequence_dir: str | Path,
) -> FrameMotionCache:
    """
    Load a previously generated frame-motion cache.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.

    Returns
    -------
    FrameMotionCache
    """

    data = load_npz(
        sequence_dir,
        CACHE_FILENAME,
    )

    return FrameMotionCache(
        frame_ids=data["frame_ids"],
        timestamps=data["timestamps"],
        object_ids=data["object_ids"],
        delta_position=data["delta_position"],
        speed=data["speed"],
    )


def compute_frame_motion(
    frames,
    objects,
) -> FrameMotionCache:
    """
    Compute per-frame object motion.

    Motion is represented as the displacement between consecutive
    observations of each object in the world frame.

    The first observation of every object has zero motion.

    Parameters
    ----------
    frames
        Sequence frames.

    objects
        Dictionary of RawObject instances.

    Returns
    -------
    FrameMotionCache
    """

    trajectories = collect_world_trajectories(
        frames,
        objects,
    )

    frame_ids = np.asarray(
        [frame.frame_id for frame in frames],
        dtype=np.int32,
    )

    timestamps = np.asarray(
        [frame.timestamp for frame in frames],
        dtype=np.float64,
    )

    object_ids = np.asarray(
        sorted(trajectories.keys()),
        dtype=np.int32,
    )

    n_frames = len(frame_ids)
    n_objects = len(object_ids)

    delta_position = np.zeros(
        (n_frames, n_objects, 3),
        dtype=np.float32,
    )

    speed = np.zeros(
        (n_frames, n_objects),
        dtype=np.float32,
    )

    #
    # Build a lookup from frame timestamp
    # to frame index.
    #
    timestamp_to_index = {
        frame.timestamp: i
        for i, frame in enumerate(frames)
    }

    #
    # Process every object independently.
    #
    for obj_col, object_id in enumerate(object_ids):

        traj = trajectories[object_id]

        positions = traj.positions
        times = traj.timestamps

        #
        # First pose has zero motion.
        #
        for k in range(1, len(times)):

            frame_index = timestamp_to_index[times[k]]

            delta = positions[k] - positions[k - 1]

            dt = times[k] - times[k - 1]

            delta_position[
                frame_index,
                obj_col,
            ] = delta.astype(np.float32)

            if dt > 0:

                speed[
                    frame_index,
                    obj_col,
                ] = (
                    np.linalg.norm(delta)
                    / dt
                )

    return FrameMotionCache(
        frame_ids=frame_ids,
        timestamps=timestamps,
        object_ids=object_ids,
        delta_position=delta_position,
        speed=speed,
    )


def generate_frame_motion(
    sequence_dir: str | Path,
) -> Path:
    """
    Generate and save the frame-motion cache for one sequence.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.

    Returns
    -------
    Path
        Path to the generated cache.
    """

    from src.data.evimo2.parser import EVIMO2Parser

    dataset = EVIMO2Parser(sequence_dir)

    cache = compute_frame_motion(
        dataset.frames,
        dataset.objects,
    )

    output_path = save_frame_motion(
        sequence_dir,
        cache,
    )

    return output_path



def verify_frame_motion(
    sequence_dir: str | Path,
):
    """
    Generate, save and verify frame-motion cache.
    """

    sequence_dir = Path(sequence_dir)

    print("=" * 90)
    print("FRAME MOTION VERIFICATION")
    print("=" * 90)

    cache_path = generate_frame_motion(
        sequence_dir
    )

    cache = load_frame_motion(
        cache_path
    )

    print()
    print(f"Cache file : {cache_path.name}")
    print(f"Frames     : {len(cache.frame_ids)}")
    print(f"Objects    : {len(cache.object_ids)}")

    print()

    print("Object Statistics")
    print("-" * 90)

    header = (
        f"{'ID':>4} "
        f"{'MeanSpeed':>12} "
        f"{'MaxSpeed':>12} "
        f"{'MeanStep(cm)':>15} "
        f"{'MaxStep(cm)':>15}"
    )

    print(header)
    print("-" * 90)

    for column, object_id in enumerate(cache.object_ids):

        delta = cache.delta_position[:, column]

        step = np.linalg.norm(
            delta,
            axis=1,
        )

        speed = cache.speed[:, column]

        print(
            f"{object_id:4d} "
            f"{speed.mean():12.3f} "
            f"{speed.max():12.3f} "
            f"{100*step.mean():15.2f} "
            f"{100*step.max():15.2f}"
        )

    print("-" * 90)

    global_step = np.linalg.norm(
        cache.delta_position,
        axis=2,
    )

    print()

    print("Global Statistics")
    print("-" * 40)

    print(
        f"Maximum speed : {cache.speed.max():.3f} m/s"
    )

    print(
        f"Maximum step  : {global_step.max():.4f} m"
    )

    print()

    print("Sanity Checks")
    print("-" * 40)

    assert np.allclose(
        cache.delta_position[0],
        0.0,
    )

    print("✓ First frame has zero motion")

    assert cache.speed.shape == (
        len(cache.frame_ids),
        len(cache.object_ids),
    )

    print("✓ Speed dimensions correct")

    assert cache.delta_position.shape == (
        len(cache.frame_ids),
        len(cache.object_ids),
        3,
    )

    print("✓ Delta position dimensions correct")

    assert np.all(
        cache.speed >= 0
    )

    print("✓ Non-negative speeds")

    print()

    print("Verification completed successfully.")
    print("=" * 90)



if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python -m src.preprocessing.frame_motion <sequence_dir>"
        )

        sys.exit(1)

    verify_frame_motion(
        sys.argv[1]
    )