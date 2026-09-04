"""
Raw EVIMO2v2 data structures.

These dataclasses describe only information directly available
from the original EVIMO2v2 event-camera dataset.

No derived quantities are stored here.

Examples of derived information NOT stored:

- event windows
- voxel grids
- dynamic labels
- motion statistics
- verification results
"""

from __future__ import annotations

from dataclasses import dataclass

from ..datatypes.common import CameraIntrinsics, Pose


# =============================================================================
# Raw Object
# =============================================================================

@dataclass(frozen=True, slots=True)
class RawObject:
    """
    Object metadata.

    Parameters
    ----------
    object_id:
        EVIMO object id.

    name:
        Human-readable name.
    """

    object_id: int
    name: str


# =============================================================================
# Raw Object State
# =============================================================================

@dataclass(frozen=True, slots=True)
class RawObjectState:
    """
    Object state at one timestamp.

    Parameters
    ----------
    object_id:
        Object identifier.

    pose:
        Object pose.

    visible:
        Whether object exists in this frame.
    """

    object_id: int

    pose: Pose | None

    visible: bool


# =============================================================================
# Raw Frame
# =============================================================================

@dataclass(frozen=True, slots=True)
class RawFrame:
    """
    One EVIMO2 event-camera ground-truth frame.

    This follows the left_camera timeline.

    Classical camera frames are intentionally not included.
    """

    frame_id: int

    timestamp: float

    camera_pose: Pose | None

    object_states: tuple[RawObjectState, ...]

    depth_available: bool

    mask_available: bool


# =============================================================================
# Raw Sequence
# =============================================================================

@dataclass(frozen=True, slots=True)
class RawSequence:
    """
    Sequence-level metadata.
    """

    sequence_name: str

    camera: CameraIntrinsics

    num_events: int

    num_frames: int

    start_time: float

    end_time: float

    objects: dict[int, RawObject]