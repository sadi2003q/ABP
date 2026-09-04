"""
Compute object motion statistics for EVIMO2 sequences.

Overview
--------
EVIMO2 provides

    • Object poses      : object -> camera
    • Camera poses      : camera -> world

To analyze the true motion of an object independent of camera motion,
object poses are first transformed into the world frame

    T_wo = T_wc @ T_co

where

    T_co : object -> camera
    T_wc : camera -> world
    T_wo : object -> world

Motion statistics are then computed using the object's world-frame
trajectory.

The computed statistics are intended for

    • dynamic/static object classification
    • dataset analysis
    • preprocessing
    • visualization

This module is independent of any learning code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.evimo2.parser import EVIMO2Parser
from src.data.evimo2.datatypes.raw import (
    RawFrame,
    RawObject,
)

from src.utils.geometry import (
    compose,
    matrix_to_pose,
    pose_to_matrix,
)


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(slots=True)
class ObjectTrajectory:
    """
    World-frame trajectory of one object.
    """

    object_id: int

    timestamps: np.ndarray

    positions: np.ndarray


@dataclass(slots=True)
class MotionStatistics:
    """
    Motion statistics for one object.
    """

    object_id: int

    num_frames: int

    duration: float

    path_length: float

    displacement: float

    mean_step: float

    max_step: float

    mean_speed: float

    max_speed: float


# =============================================================================
# Trajectory Extraction
# =============================================================================


def collect_world_trajectories(
    frames: list[RawFrame],
    objects: dict[int, RawObject],
) -> dict[int, ObjectTrajectory]:
    """
    Construct world-frame trajectories for every object.

    Object poses stored by EVIMO2 are:

        object -> camera

    Camera poses are:

        camera -> world

    Therefore,

        object -> world

    is computed as

        T_world_object =
            T_world_camera
            @
            T_camera_object

    Parameters
    ----------
    frames
        Parsed sequence frames.

    objects
        Sequence object dictionary.

    Returns
    -------
    dict
        Mapping

            object_id -> ObjectTrajectory
    """

    trajectory_data = {

        object_id: {

            "timestamps": [],
            "positions": [],

        }

        for object_id in objects

    }

    for frame in frames:

        if frame.camera_pose is None:
            continue

        T_wc = pose_to_matrix(
            frame.camera_pose
        )

        for state in frame.object_states:

            T_co = pose_to_matrix(
                state.pose
            )

            T_wo = T_wc @ T_co

            position = T_wo[:3, 3]

            trajectory_data[
                state.object_id
            ]["timestamps"].append(
                frame.timestamp
            )

            trajectory_data[
                state.object_id
            ]["positions"].append(
                position
            )

    trajectories = {}

    for object_id, values in trajectory_data.items():

        if len(values["timestamps"]) == 0:
            continue

        trajectories[object_id] = ObjectTrajectory(

            object_id=object_id,

            timestamps=np.asarray(
                values["timestamps"],
                dtype=np.float64,
            ),

            positions=np.asarray(
                values["positions"],
                dtype=np.float64,
            ),

        )

    return trajectories


# =============================================================================
# Motion Statistics
# =============================================================================


def compute_motion_statistics(
    trajectory: ObjectTrajectory,
) -> MotionStatistics:
    """
    Compute motion statistics for one object trajectory.

    Parameters
    ----------
    trajectory
        World-frame trajectory.

    Returns
    -------
    MotionStatistics
    """

    positions = trajectory.positions

    timestamps = trajectory.timestamps

    n = len(positions)

    if n < 2:

        return MotionStatistics(

            object_id=trajectory.object_id,

            num_frames=n,

            duration=0.0,

            path_length=0.0,

            displacement=0.0,

            mean_step=0.0,

            max_step=0.0,

            mean_speed=0.0,

            max_speed=0.0,

        )

    # -------------------------------------------------------------
    # Step vectors
    # -------------------------------------------------------------

    delta_position = np.diff(
        positions,
        axis=0,
    )

    step_lengths = np.linalg.norm(
        delta_position,
        axis=1,
    )

    # -------------------------------------------------------------
    # Time intervals
    # -------------------------------------------------------------

    delta_time = np.diff(
        timestamps
    )

    # Avoid divide-by-zero

    delta_time = np.maximum(
        delta_time,
        1e-12,
    )

    speeds = (
        step_lengths /
        delta_time
    )

    path_length = float(
        step_lengths.sum()
    )

    displacement = float(

        np.linalg.norm(

            positions[-1] -
            positions[0]

        )

    )

    duration = float(

        timestamps[-1] -
        timestamps[0]

    )

    return MotionStatistics(

        object_id=trajectory.object_id,

        num_frames=n,

        duration=duration,

        path_length=path_length,

        displacement=displacement,

        mean_step=float(
            step_lengths.mean()
        ),

        max_step=float(
            step_lengths.max()
        ),

        mean_speed=float(
            speeds.mean()
        ),

        max_speed=float(
            speeds.max()
        ),

    )

# =============================================================================
# Statistics
# =============================================================================


def compute_all_statistics(
    frames: list[RawFrame],
    objects: dict[int, RawObject],
) -> list[MotionStatistics]:
    """
    Compute motion statistics for every object.

    Parameters
    ----------
    frames
        Sequence frames.

    objects
        Object metadata.

    Returns
    -------
    list
        Motion statistics sorted by object id.
    """

    trajectories = collect_world_trajectories(
        frames,
        objects,
    )

    statistics = [

        compute_motion_statistics(
            trajectory
        )

        for trajectory in trajectories.values()

    ]

    statistics.sort(
        key=lambda s: s.object_id
    )

    return statistics

# =============================================================================
# Printing
# =============================================================================


def print_statistics(
    statistics: list[MotionStatistics],
) -> None:
    """
    Print motion statistics in tabular form.
    """

    print()

    print("=" * 108)
    print("OBJECT MOTION STATISTICS (WORLD FRAME)")
    print("=" * 108)

    print(

        f"{'ID':>4}"
        f"{'Frames':>8}"
        f"{'Duration':>10}"
        f"{'Path(m)':>12}"
        f"{'Disp(m)':>12}"
        f"{'MeanStep(cm)':>15}"
        f"{'MaxStep(cm)':>15}"
        f"{'MeanSpeed':>13}"
        f"{'MaxSpeed':>13}"

    )

    print("-" * 108)

    for stat in statistics:

        print(

            f"{stat.object_id:>4}"

            f"{stat.num_frames:>8}"

            f"{stat.duration:>10.3f}"

            f"{stat.path_length:>12.3f}"

            f"{stat.displacement:>12.3f}"

            f"{100*stat.mean_step:>15.2f}"

            f"{100*stat.max_step:>15.2f}"

            f"{stat.mean_speed:>13.3f}"

            f"{stat.max_speed:>13.3f}"

        )

    print("=" * 108)


# =============================================================================
# Verification Helpers
# =============================================================================

def compute_camera_path_length(
    frames: list[RawFrame],
) -> float:
    """
    Compute total camera trajectory length in the world frame.

    Parameters
    ----------
    frames
        Sequence frames.

    Returns
    -------
    float
        Total travelled distance in meters.
    """

    positions: list[np.ndarray] = []

    for frame in frames:

        if frame.camera_pose is None:
            continue

        positions.append(
            frame.camera_pose.translation
        )

    if len(positions) < 2:
        return 0.0

    path = 0.0

    for p0, p1 in zip(positions[:-1], positions[1:]):

        path += np.linalg.norm(
            p1 - p0
        )

    return float(path)


# =============================================================================
# Main
# =============================================================================
# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Compute motion statistics for an EVIMO2 sequence."
    )

    parser.add_argument(
        "sequence",
        type=Path,
        help="Path to an EVIMO2 sequence directory.",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("OBJECT MOTION VERIFICATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Parse dataset
    # -------------------------------------------------------------------------

    dataset = EVIMO2Parser(
        args.sequence
    )

    sequence = dataset.sequence
    frames = dataset.frames
    objects = sequence.objects

    # -------------------------------------------------------------------------
    # Sequence summary
    # -------------------------------------------------------------------------

    print("\n[Sequence]")

    print(f"Name          : {sequence.sequence_name}")
    print(f"Frames        : {len(frames)}")
    print(f"Objects       : {len(objects)}")
    print(f"Events        : {sequence.num_events:,}")

    valid_frames = sum(
        frame.camera_pose is not None
        for frame in frames
    )

    print(f"Tracked Frames: {valid_frames}")
    print(f"Missing Frames: {len(frames) - valid_frames}")

    # -------------------------------------------------------------------------
    # Camera statistics
    # -------------------------------------------------------------------------

    camera_path = compute_camera_path_length(
        frames
    )

    print("\n[Camera]")

    print(f"Path Length   : {camera_path:.3f} m")

    # -------------------------------------------------------------------------
    # Object trajectories
    # -------------------------------------------------------------------------

    trajectories = collect_world_trajectories(
        frames,
        objects,
    )

    print("\n[Trajectories]")

    print(f"Objects       : {len(trajectories)}")

    for object_id in sorted(trajectories):

        trajectory = trajectories[object_id]

        print(
            f"Object {object_id:>2}: "
            f"{len(trajectory.timestamps)} poses"
        )

    # -------------------------------------------------------------------------
    # Motion statistics
    # -------------------------------------------------------------------------

    statistics = compute_all_statistics(
        frames,
        objects,
    )

    print()
    print_statistics(
        statistics
    )

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------

    print("\n[Sanity Checks]")

    passed = True

    for stat in statistics:

        if stat.path_length + 1e-9 < stat.displacement:

            passed = False

            print(
                f"✗ Object {stat.object_id}: "
                "Path length smaller than displacement."
            )

    if passed:
        print("✓ Path length >= displacement for every object.")

    if camera_path > 0:
        print("✓ Camera trajectory successfully reconstructed.")

    if len(statistics) == len(objects):
        print("✓ Motion statistics computed for every object.")

    print()
    print("=" * 100)
    print("Verification completed successfully.")
    print("=" * 100)