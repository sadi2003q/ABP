"""
Dataclasses shared across the EVIMO2 data pipeline.

These classes represent the information returned by the Dataset and
passed through the DataLoader.

No loading logic should exist in this module.
Only lightweight immutable data containers.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class SequenceReference:
    """
    Maps a global dataset index to a frame in a sequence.
    """

    sequence_id: int
    local_frame_index: int


@dataclass(slots=True)
class CameraMotion:
    translation: np.ndarray
    """
    Shape
    -----
    (3,)

    World-frame camera position.
    """

    quaternion: np.ndarray
    """
    Shape
    -----
    (4,)

    Quaternion order:
    (x, y, z, w)
    """

    delta_translation: np.ndarray
    """
    Shape
    -----
    (3,)

    Translation between consecutive frames.
    """

    delta_quaternion: np.ndarray
    """
    Shape
    -----
    (4,)

    Relative rotation from previous frame.

    Quaternion order:
    (x, y, z, w)
    """

    linear_speed: float
    angular_speed: float

    dt: float

    pose_available: bool



@dataclass(slots=True)
class FrameMotion:
    """
    Per-frame object motion loaded from the preprocessing cache.

    Motion is represented as the displacement of every object
    since its previous observation in the world frame.

    Dynamic/static classification is intentionally NOT stored
    here. The Dataset (or transforms) can compute it using the
    object speeds and a configurable threshold.
    """

    object_ids: np.ndarray

    delta_position: np.ndarray

    speed: np.ndarray


@dataclass(slots=True)
class IMUWindow:
    """
    IMU measurements belonging to one frame interval.

    The measurements span the same temporal window as the event
    packet returned for this frame.
    """

    timestamps: np.ndarray
    """
    Shape
    -----
    (N,)

    dtype
    -----
    float64

    IMU timestamps.
    """

    angular_velocity: np.ndarray
    """
    Shape
    -----
    (N,3)

    dtype
    -----
    float32

    Angular velocity (rad/s).
    """

    linear_acceleration: np.ndarray
    """
    Shape
    -----
    (N,3)

    dtype
    -----
    float32

    Linear acceleration (m/s²).
    """


@dataclass(slots=True)
class EVIMO2Sample:

    """
    One training sample.
    """

    #
    # Identification
    #
    sequence_name: str

    sensor: str

    #
    # Camera calibration
    #

    camera_intrinsics: np.ndarray
    """
    Shape
    -----
    (3,3)

    Camera intrinsic matrix K.
    """

    camera_distortion: np.ndarray
    """
    Shape
    -----
    (4,)

    Distortion coefficients:

        [k1,k2,p1,p2]

    """

    local_frame_index: int

    frame_id: int

    #
    # Time
    #
    timestamp: float

    #
    # Events
    #
    events_xy: np.ndarray
    """
    Shape
    -----
    (N, 2)

    dtype
    -----
    uint16

    Pixel coordinates.
    """

    events_t: np.ndarray
    """
    Shape
    -----
    (N,)

    dtype
    -----
    float64

    Event timestamps in seconds.
    """

    events_p: np.ndarray
    """
    Shape
    -----
    (N,)

    dtype
    -----
    bool

    Event polarity.
    """

    #
    # Camera motion
    #
    camera_motion: CameraMotion


    #
    # IMU
    #
    imu: IMUWindow


    #
    # Object motion
    #
    frame_motion: FrameMotion

    #
    # Ground truth
    #
    depth: np.ndarray | None
    """
    Shape
    -----
    (H, W)

    dtype
    -----
    float32

    Depth map in meters.
    """

    mask: np.ndarray | None
    """
    Shape
    -----
    (H, W)

    dtype
    -----
    uint16

    Per-pixel object instance IDs.
    """

    #
    # RGB (visualization only)
    #
    rgb: np.ndarray | None
    """
    Shape
    -----
    (H, W, 3)

    dtype
    -----
    uint8

    RGB frame.

    Returned for visualization only.
    """


# ==========================================================
# Batched EVIMO2 samples
# ==========================================================

@dataclass(slots=True)
class EVIMO2Batch:

    """
    Mini-batch returned by the custom collate function.

    Event arrays are concatenated into one continuous event stream.

    The event_sample_indices array indicates which sample each
    event belongs to.

    CameraMotion and FrameMotion remain as lists because they are
    lightweight metadata objects and preserving their structure
    simplifies downstream processing.
    """

    #
    # ------------------------------------------------------
    # Sample metadata
    # ------------------------------------------------------
    #

    sequence_names: list[str]

    sensors: list[str]

    local_frame_indices: np.ndarray

    frame_ids: np.ndarray

    timestamps: np.ndarray

    #
    # ------------------------------------------------------
    # Events
    # ------------------------------------------------------
    #

    events_xy: np.ndarray
    """
    Shape
    -----
    (N_events, 2)
    """

    events_t: np.ndarray
    """
    Shape
    -----
    (N_events,)
    """

    events_p: np.ndarray
    """
    Shape
    -----
    (N_events,)
    """

    event_sample_indices: np.ndarray
    """
    Shape
    -----
    (N_events,)

    Indicates which sample each event belongs to.

    Example
    -------
    [0,0,0,0,1,1,1,2,2,...]
    """

    #
    # ------------------------------------------------------
    # Motion
    # ------------------------------------------------------
    #

    camera_motion: list[CameraMotion]

    frame_motion: list[FrameMotion]


    #
    # ------------------------------------------------------
    # Camera calibration
    # ------------------------------------------------------
    #

    camera_intrinsics: list[np.ndarray]
    """
    One K matrix per sample.

    Each element:

    (3,3)
    """


    camera_distortion: list[np.ndarray]
    """
    One distortion vector per sample.

    Each element:

    (4,)
    """

    #
    # ------------------------------------------------------
    # IMU
    # ------------------------------------------------------
    #

    imu_timestamps: np.ndarray
    """
    Shape
    -----
    (N_imu,)
    """

    imu_angular_velocity: np.ndarray
    """
    Shape
    -----
    (N_imu,3)
    """

    imu_linear_acceleration: np.ndarray
    """
    Shape
    -----
    (N_imu,3)
    """

    imu_sample_indices: np.ndarray
    """
    Shape
    -----
    (N_imu,)

    Exactly analogous to event_sample_indices.
    """
    
    #
    # ------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------
    #

    depth: list[np.ndarray | None]
    """
    One depth image per sample.

    Length
    ------
    batch_size

    Each element is either

        (H,W) float32

    or None.
    """

    mask: list[np.ndarray | None]
    """
    One instance mask per sample.

    Length
    ------
    batch_size

    Each element is either

        (H,W) uint16

    or None.
    """

    rgb: list[np.ndarray | None]
    """
    One RGB image per sample.

    Length
    ------
    batch_size

    Each element is either

        (H,W,3)

    or None.
    """


@dataclass(slots=True)
class TemporalEVIMO2Batch:
    """
    Mini-batch produced by TemporalEVIMO2Dataset.

    One EVIMO2Batch is stored for every temporal position.

    frames[0] corresponds to history_offsets[0].

    frames[-1] corresponds to the reference frame.
    """

    frames: list[EVIMO2Batch]

    history_offsets: tuple[int, ...]
