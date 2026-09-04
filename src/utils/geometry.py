"""
Geometric utilities for SE(3) transformations.

This module provides lightweight utilities for manipulating poses,
quaternions and homogeneous transformation matrices.

Coordinate Convention
---------------------

Throughout this project, poses represent rigid-body transforms.

A transform

    T_ab

maps coordinates from frame **b** into frame **a**.

For example

    camera <- object

means

    p_camera = T_camera_object @ p_object

Functions in this module intentionally avoid any dependency on ROS,
OpenCV, SciPy, or external robotics libraries. Only NumPy is required.

Notation
--------

R
    3×3 rotation matrix

t
    Translation vector

T
    4×4 homogeneous transform

q
    Quaternion stored as

        (x, y, z, w)

which follows the convention used by EVIMO2.
"""

from __future__ import annotations

import numpy as np

from src.data.evimo2.datatypes.common import Pose



# =============================================================================
# Quaternion
# =============================================================================


def quaternion_to_rotation(
    quaternion: np.ndarray,
) -> np.ndarray:
    """
    Convert a quaternion into a rotation matrix.

    Parameters
    ----------
    quaternion

        Quaternion stored as

            (x, y, z, w)

    Returns
    -------
    ndarray

        Rotation matrix of shape (3,3).
    """

    x, y, z, w = quaternion

    xx = x * x
    yy = y * y
    zz = z * z

    xy = x * y
    xz = x * z
    yz = y * z

    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(

        [

            [
                1 - 2 * (yy + zz),
                2 * (xy - wz),
                2 * (xz + wy),
            ],

            [
                2 * (xy + wz),
                1 - 2 * (xx + zz),
                2 * (yz - wx),
            ],

            [
                2 * (xz - wy),
                2 * (yz + wx),
                1 - 2 * (xx + yy),
            ],

        ],

        dtype=np.float64,

    )


def rotation_to_quaternion(
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Convert a rotation matrix into a quaternion.

    Parameters
    ----------
    rotation

        Rotation matrix.

    Returns
    -------
    ndarray

        Quaternion stored as

            (x, y, z, w)
    """

    R = rotation

    trace = np.trace(R)

    if trace > 0:

        s = np.sqrt(trace + 1.0) * 2.0

        w = 0.25 * s

        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:

        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2

        w = (R[2, 1] - R[1, 2]) / s

        x = 0.25 * s

        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:

        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2

        w = (R[0, 2] - R[2, 0]) / s

        x = (R[0, 1] + R[1, 0]) / s

        y = 0.25 * s

        z = (R[1, 2] + R[2, 1]) / s

    else:

        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2

        w = (R[1, 0] - R[0, 1]) / s

        x = (R[0, 2] + R[2, 0]) / s

        y = (R[1, 2] + R[2, 1]) / s

        z = 0.25 * s

    return np.array(
        [x, y, z, w],
        dtype=np.float64,
    )

# =============================================================================
# Pose <-> Homogeneous Matrix
# =============================================================================


def pose_to_matrix(
    pose: Pose,
) -> np.ndarray:
    """
    Convert a Pose into a 4×4 homogeneous transformation matrix.

    Parameters
    ----------
    pose
        Pose represented by translation and quaternion.

    Returns
    -------
    ndarray

        Homogeneous transform

            [[R t]
             [0 1]]

        of shape (4,4).
    """

    T = np.eye(
        4,
        dtype=np.float64,
    )

    T[:3, :3] = quaternion_to_rotation(
        pose.quaternion
    )

    T[:3, 3] = pose.translation

    return T


def matrix_to_pose(
    transform: np.ndarray,
) -> Pose:
    """
    Convert a homogeneous transformation matrix into a Pose.

    Parameters
    ----------
    transform

        4×4 homogeneous transform.

    Returns
    -------
    Pose
    """

    return Pose(

        translation=transform[:3, 3].copy(),

        quaternion=rotation_to_quaternion(

            transform[:3, :3]

        ),

    )

# =============================================================================
# Transform Operations
# =============================================================================


def compose(
    transform_a: np.ndarray,
    transform_b: np.ndarray,
) -> np.ndarray:
    """
    Compose two homogeneous transformations.

    Parameters
    ----------
    transform_a

        First transform.

    transform_b

        Second transform.

    Returns
    -------
    ndarray

        Matrix product

            transform_a @ transform_b
    """

    return transform_a @ transform_b

def inverse(
    transform: np.ndarray,
) -> np.ndarray:
    """
    Compute the inverse of a rigid-body transform.

    Parameters
    ----------
    transform

        4×4 homogeneous transformation matrix.

    Returns
    -------
    ndarray

        Inverse transform.
    """

    R = transform[:3, :3]

    t = transform[:3, 3]

    inv = np.eye(
        4,
        dtype=np.float64,
    )

    inv[:3, :3] = R.T

    inv[:3, 3] = -R.T @ t

    return inv

def transform_point(
    transform: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    """
    Apply a homogeneous transformation to a 3D point.

    Parameters
    ----------
    transform

        4×4 transform.

    point

        Shape (3,).

    Returns
    -------
    ndarray

        Transformed point of shape (3,).
    """

    homogeneous = np.append(
        point,
        1.0,
    )

    transformed = transform @ homogeneous

    return transformed[:3]

def transform_points(
    transform: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """
    Transform multiple 3D points.

    Parameters
    ----------
    transform

        4×4 transform.

    points

        Shape

            (N,3)

    Returns
    -------
    ndarray

        Transformed points with shape

            (N,3)
    """

    ones = np.ones(
        (points.shape[0], 1),
        dtype=np.float64,
    )

    homogeneous = np.concatenate(
        [points, ones],
        axis=1,
    )

    transformed = (
        transform @ homogeneous.T
    ).T

    return transformed[:, :3]



if __name__ == "__main__":

    print("=" * 70)
    print("GEOMETRY VERIFICATION")
    print("=" * 70)

    pose = Pose(

        translation=np.array(
            [1.0, 2.0, 3.0]
        ),

        quaternion=np.array(
            [0.0, 0.0, 0.0, 1.0]
        ),

    )

    T = pose_to_matrix(
        pose
    )

    recovered = matrix_to_pose(
        T
    )

    print("\nPose -> Matrix -> Pose")

    print("translation error:",
          np.linalg.norm(
              pose.translation -
              recovered.translation
          ))

    print("quaternion error:",
          min(
              np.linalg.norm(
                  pose.quaternion -
                  recovered.quaternion
              ),
              np.linalg.norm(
                  pose.quaternion +
                  recovered.quaternion
              ),
          ))

    T_inv = inverse(T)

    identity = compose(
        T,
        T_inv,
    )

    print("\nInverse check:")

    print(
        np.linalg.norm(
            identity -
            np.eye(4)
        )
    )

    point = np.array(
        [2.0, 0.0, 0.0]
    )

    transformed = transform_point(
        T,
        point,
    )

    print("\nPoint transform:")

    print(point)

    print(transformed)