"""
EVIMO2v2 metadata parser.

This module converts the raw dataset_info.npz metadata
into project datatypes.

Only the event-camera stream is considered.

Classical camera data (flea3_7) is intentionally ignored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .datatypes.raw import (
    RawFrame,
    RawObject,
    RawObjectState,
)
from .datatypes.common import Pose


# =============================================================================
# Pose parsing
# =============================================================================

def parse_pose(data: dict) -> Pose:
    """
    Parse EVIMO pose dictionary.

    EVIMO format:

    {
        "t":
            {"x","y","z"},

        "q":
            {"x","y","z","w"}
    }

    """

    t = data["t"]

    q = data["q"]

    translation = np.array(
        [
            t["x"],
            t["y"],
            t["z"],
        ],
        dtype=np.float32,
    )

    quaternion = np.array(
        [
            q["x"],
            q["y"],
            q["z"],
            q["w"],
        ],
        dtype=np.float32,
    )

    return Pose(
        translation=translation,
        quaternion=quaternion,
    )


# =============================================================================
# Available frame indexing
# =============================================================================

def extract_available_ids(
    path: Path,
    prefix: str,
) -> set[int]:
    """
    Extract available frame IDs from EVIMO2v2 npz.

    Example:

    mask_0000000046

    becomes:

    46
    """

    if not path.exists():
        return set()


    data = np.load(path)

    ids = set()

    for key in data.files:

        if key.startswith(prefix):

            frame_id = int(
                key.split("_")[1]
            )

            ids.add(frame_id)

    return ids



# =============================================================================
# Frame parsing
# =============================================================================

def parse_frames(
    metadata: dict,
    depth_ids: set[int],
    mask_ids: set[int],
) -> list[RawFrame]:

    frames = []

    for frame in metadata["frames"]:

        frame_id = int(
            frame["id"]
        )

        timestamp = float(
            frame["ts"]
        )


        # -------------------------------------------------
        # Camera pose
        # -------------------------------------------------

        camera_pose = None

        if "cam" in frame:

            camera_pose = parse_pose(
                frame["cam"]["pos"]
            )


        # -------------------------------------------------
        # Objects
        # -------------------------------------------------

        objects = []


        for key, value in frame.items():

            if key in [
                "cam",
                "id",
                "ts",
                "classical_frame",
                "gt_frame",
            ]:
                continue


            # object entry

            if isinstance(value, dict):

                if "pos" in value:

                    objects.append(
                        RawObjectState(
                            object_id=int(key),

                            pose=parse_pose(
                                value["pos"]
                            ),

                            visible=True,
                        )
                    )


        frames.append(
            RawFrame(
                frame_id=frame_id,

                timestamp=timestamp,

                camera_pose=camera_pose,

                object_states=tuple(objects),

                depth_available=(
                    frame_id in depth_ids
                ),

                mask_available=(
                    frame_id in mask_ids
                ),
            )
        )


    return frames



# =============================================================================
# IMU parsing
# =============================================================================

def parse_imu(metadata: dict) -> dict[str, dict[str, np.ndarray]]:
    """
    Parse EVIMO2 IMU measurements.

    Returns
    -------
    dict

        {
            "left_camera":
            {
                "timestamps": (N,),
                "gyro": (N,3),
                "acceleration": (N,3),
            },

            "right_camera":
            {
                "timestamps": (N,),
                "gyro": (N,3),
                "acceleration": (N,3),
            },
        }

    Samsung Mono sequences contain no IMU and are therefore omitted.
    """

    imu_data = {}

    sensor_map = {
        "/prophesee/left/imu": "left_camera",
        "/prophesee/right/imu": "right_camera",
    }

    raw_imu = metadata.get("imu", {})

    for raw_key, sensor_name in sensor_map.items():

        if raw_key not in raw_imu:
            continue

        measurements = raw_imu[raw_key]

        timestamps = np.asarray(
            [m["ts"] for m in measurements],
            dtype=np.float64,
        )

        gyro = np.asarray(
            [
                [
                    m["angular_velocity"]["x"],
                    m["angular_velocity"]["y"],
                    m["angular_velocity"]["z"],
                ]
                for m in measurements
            ],
            dtype=np.float32,
        )

        acceleration = np.asarray(
            [
                [
                    m["linear_acceleration"]["x"],
                    m["linear_acceleration"]["y"],
                    m["linear_acceleration"]["z"],
                ]
                for m in measurements
            ],
            dtype=np.float32,
        )

        imu_data[sensor_name] = {
            "timestamps": timestamps,
            "gyro": gyro,
            "acceleration": acceleration,
        }

    return imu_data

# =============================================================================
# Main parser
# =============================================================================

def parse_sequence_metadata(
    sequence_dir: str | Path,
) -> tuple[list[RawFrame], dict[int, RawObject]]:

    sequence_dir = Path(sequence_dir)


    info_path = (
        sequence_dir /
        "dataset_info.npz"
    )


    data = np.load(
        info_path,
        allow_pickle=True,
    )


    meta = data["meta"].item()


    # -------------------------------------------------
    # Available GT frames
    # -------------------------------------------------

    depth_ids = extract_available_ids(
        sequence_dir /
        "dataset_depth.npz",

        "depth",
    )


    mask_ids = extract_available_ids(
        sequence_dir /
        "dataset_mask.npz",

        "mask",
    )


    frames = parse_frames(
        meta,

        depth_ids,

        mask_ids,
    )

    imu = parse_imu(meta)
    # -------------------------------------------------
    # Object list
    # -------------------------------------------------

    objects = {}

    for frame in frames:

        for obj in frame.object_states:

            if obj.object_id not in objects:

                objects[obj.object_id] = RawObject(
                    object_id=obj.object_id,

                    name=f"object_{obj.object_id}",
                )


    # -------------------------------------------------
    # Consistency check
    # -------------------------------------------------

    frame_object_ids = {
        obj.object_id
        for frame in frames
        for obj in frame.object_states
    }

    missing_objects = (
        frame_object_ids -
        set(objects.keys())
    )

    assert not missing_objects, (
        "Missing objects in parser: "
        f"{missing_objects}"
    )

    return frames, objects, imu


    