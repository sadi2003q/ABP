"""
Common data structures shared across all event-camera datasets.

This module intentionally contains only dataset-agnostic types.
These classes describe generic concepts such as poses, camera
calibration and events, and should not contain any dataset-specific
assumptions.

All dataclasses are immutable (frozen=True) and memory efficient
(slots=True).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =============================================================================
# Camera Geometry
# =============================================================================

@dataclass(frozen=True, slots=True)
class Pose:
    """
    Rigid body pose represented by translation and quaternion.

    Parameters
    ----------
    translation : np.ndarray
        Translation vector of shape (3,) in meters.

    quaternion : np.ndarray
        Unit quaternion of shape (4,) stored as (x, y, z, w).
    """

    translation: np.ndarray
    quaternion: np.ndarray


@dataclass(slots=True)
class CameraIntrinsics:

    fx: float
    fy: float

    cx: float
    cy: float

    width: int
    height: int

    distortion: np.ndarray

    def matrix(self):

        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def distortion_coefficients(self):

        return self.distortion


# =============================================================================
# Image Size
# =============================================================================

@dataclass(frozen=True, slots=True)
class Resolution:
    """
    Image resolution.

    Parameters
    ----------
    width : int
        Image width.

    height : int
        Image height.
    """

    width: int
    height: int

