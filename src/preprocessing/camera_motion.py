"""
Per-frame camera ego-motion computation.

This module computes the camera motion between consecutive frames
using the ground-truth camera poses provided by EVIMO2.

Unlike object motion, camera motion represents the motion of the
sensor itself (ego-motion).

No thresholding or motion classification is performed here.
Only geometric quantities are cached.

Cache Contents
--------------
frame_ids
timestamps

translation
quaternion

delta_translation
linear_speed

delta_quaternion
angular_speed
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


CACHE_FILENAME = "camera_motion.npz"

from src.preprocessing.cache import (
    save_npz,
    load_npz,
)


# ============================================================
# Dataclass
# ============================================================

@dataclass(slots=True)
class CameraMotionCache:
    """
    Cached camera motion.

    Parameters
    ----------
    frame_ids
        Frame identifiers.

    timestamps
        Frame timestamps.
    
    frame_dt: np.ndarray

    pose_available
        True if a valid camera pose exists for the frame.

        Frames without a pose retain zero translation,
        identity quaternion, and zero motion.

    translation
        Camera translation in world coordinates.

        Shape
        -----
        (N,3)

    quaternion
        Camera orientation.

        Quaternion order
        ----------------
        (x,y,z,w)

        Shape
        -----
        (N,4)

    delta_translation
        Translation between consecutive frames.

        Shape
        -----
        (N,3)

    linear_speed
        Instantaneous camera speed.

        Units
        -----
        meters / second

    delta_quaternion
        Relative rotation from previous frame.

        Shape
        -----
        (N,4)

    angular_speed
        Instantaneous angular velocity.

        Units
        -----
        radians / second
    """

    frame_ids: np.ndarray
    timestamps: np.ndarray
    frame_dt: np.ndarray

    pose_available: np.ndarray

    translation: np.ndarray
    quaternion: np.ndarray

    delta_translation: np.ndarray
    linear_speed: np.ndarray

    delta_quaternion: np.ndarray
    angular_speed: np.ndarray


# ============================================================
# Quaternion utilities
# ============================================================

def quaternion_inverse(
    q: np.ndarray,
) -> np.ndarray:
    """
    Invert unit quaternion.

    Quaternion format
    -----------------
    (x,y,z,w)
    """

    x, y, z, w = q

    return np.array(
        [-x, -y, -z, w],
        dtype=np.float32,
    )


def quaternion_multiply(
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    """
    Hamilton product.

    Quaternion format
    -----------------
    (x,y,z,w)
    """

    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array(
        [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ],
        dtype=np.float32,
    )


def quaternion_angle(
    q: np.ndarray,
) -> float:
    """
    Rotation angle represented by a unit quaternion.

    Returns
    -------
    radians
    """

    w = np.clip(q[3], -1.0, 1.0)

    return 2.0 * np.arccos(w)


# ============================================================
# Main computation
# ============================================================

def compute_camera_motion(
    frames,
) -> CameraMotionCache:
    """
    Compute camera motion between consecutive frames.

    Parameters
    ----------
    frames
        Sequence of RawFrame objects.

    Returns
    -------
    CameraMotionCache
    """

    n = len(frames)

    frame_ids = np.empty(
        n,
        dtype=np.int32,
    )

    timestamps = np.empty(
        n,
        dtype=np.float64,
    )

    frame_dt = np.zeros(
        n,
        dtype=np.float32,
    )

    pose_available = np.zeros(
        n,
        dtype=bool,
    )

    translation = np.zeros(
        (n,3),
        dtype=np.float32,
    )

    quaternion = np.zeros(
        (n,4),
        dtype=np.float32,
    )
    #
    # Missing poses use identity rotation.
    #
    quaternion[:, 3] = 1.0

    delta_translation = np.zeros(
        (n,3),
        dtype=np.float32,
    )

    linear_speed = np.zeros(
        n,
        dtype=np.float32,
    )

    delta_quaternion = np.zeros(
        (n,4),
        dtype=np.float32,
    )

    delta_quaternion[:,3] = 1.0

    angular_speed = np.zeros(
        n,
        dtype=np.float32,
    )

    #
    # Store poses
    #
    for i, frame in enumerate(frames):

        frame_ids[i] = frame.frame_id

        timestamps[i] = frame.timestamp

        if frame.camera_pose is None:
            continue

        pose_available[i] = True

        translation[i] = frame.camera_pose.translation

        quaternion[i] = frame.camera_pose.quaternion

    #
    # Frame interval.
    #
    if n > 1:

        frame_dt[:-1] = np.diff(timestamps)

        frame_dt[-1] = frame_dt[-2]

    #
    # Compute motion
    #
    previous_valid = None

    for i in range(n):

        #
        # Ignore frames without pose.
        #
        if not pose_available[i]:
            continue

        #
        # First valid pose.
        #
        if previous_valid is None:
            previous_valid = i
            continue

        j = previous_valid

        dt = timestamps[i] - timestamps[j]

        if dt <= 0:
            previous_valid = i
            continue

        #
        # Translation
        #
        delta = translation[i] - translation[j]

        delta_translation[i] = delta

        linear_speed[i] = (
            np.linalg.norm(delta)
            / dt
        )

        #
        # Rotation
        #
        dq = quaternion_multiply(
            quaternion_inverse(quaternion[j]),
            quaternion[i],
        )

        norm = np.linalg.norm(dq)

        if norm > 0:
            dq /= norm

        delta_quaternion[i] = dq

        angular_speed[i] = (
            quaternion_angle(dq)
            / dt
        )

        previous_valid = i

    return CameraMotionCache(
        frame_ids=frame_ids,
        timestamps=timestamps,
        frame_dt=frame_dt,

        pose_available=pose_available,

        translation=translation,
        quaternion=quaternion,

        delta_translation=delta_translation,
        linear_speed=linear_speed,

        delta_quaternion=delta_quaternion,
        angular_speed=angular_speed,
    )


def save_camera_motion(
    sequence_dir: str | Path,
    cache: CameraMotionCache,
) -> Path:
    """
    Save camera-motion cache.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.

    cache
        Computed camera-motion cache.

    Returns
    -------
    Path
        Saved cache path.
    """

    return save_npz(
        sequence_dir,
        CACHE_FILENAME,

        frame_ids=cache.frame_ids,
        timestamps=cache.timestamps,
        frame_dt=cache.frame_dt,

        pose_available=cache.pose_available,

        translation=cache.translation,
        quaternion=cache.quaternion,

        delta_translation=cache.delta_translation,
        linear_speed=cache.linear_speed,

        delta_quaternion=cache.delta_quaternion,
        angular_speed=cache.angular_speed,
    )


def load_camera_motion(
    sequence_dir: str | Path,
) -> CameraMotionCache:
    """
    Load previously generated camera-motion cache.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.

    Returns
    -------
    CameraMotionCache
    """

    data = load_npz(
        sequence_dir,
        CACHE_FILENAME,
    )

    return CameraMotionCache(

        frame_ids=data["frame_ids"],
        timestamps=data["timestamps"],
        frame_dt=data["frame_dt"],

        pose_available=data["pose_available"],

        translation=data["translation"],
        quaternion=data["quaternion"],

        delta_translation=data["delta_translation"],
        linear_speed=data["linear_speed"],

        delta_quaternion=data["delta_quaternion"],
        angular_speed=data["angular_speed"],
    )


# ============================================================
# Generation
# ============================================================

def generate_camera_motion(
    sequence_dir: str | Path,
) -> Path:
    """
    Generate and save the camera-motion cache.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.

    Returns
    -------
    Path
        Path to generated cache.
    """

    from src.data.evimo2.parser import EVIMO2Parser

    dataset = EVIMO2Parser(
        sequence_dir,
    )

    cache = compute_camera_motion(
        dataset.frames,
    )

    output_path = save_camera_motion(
        sequence_dir,
        cache,
    )

    return output_path