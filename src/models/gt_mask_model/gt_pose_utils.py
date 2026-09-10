"""
Ground-truth relative pose utilities.

The dataset gives us, per frame, a `CameraMotion` object with the
camera's pose IN THE WORLD FRAME:

    translation : (3,)   camera position in world coordinates
    quaternion  : (4,)   camera orientation, world <- camera, (x,y,z,w)

To warp a source frame's features/depth into the target frame (the
operation `LatentRenderer` performs), we need the relative transform

    T_target_source  =  T_world_target^-1  @  T_world_source

i.e. "where was every 3-D point of the source camera's view, expressed
in the target camera's coordinate frame". This is standard visual-
odometry relative-pose composition and is EXACT (no learning, no
ambiguity) because EVIMO2's GT poses are metric ground truth from a
motion-capture rig.

This module is intentionally independent of the rest of the training
pipeline (no torch autograd needed for GT poses — they are constants
for a given batch) and operates on plain tensors so it can be unit
tested and reused by any script (training, evaluation, notebooks).

Rotation convention
--------------------
Quaternions follow the EVIMO2 / project-wide convention of (x, y, z, w)
(see src/utils/geometry.py). This module intentionally reimplements
the tiny quaternion->rotation conversion in pure torch (batched, GPU
friendly, autograd-safe even though GT poses don't need grad) instead
of importing the numpy version in src/utils/geometry.py.
"""

from __future__ import annotations

import torch


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Batched quaternion (x, y, z, w) -> rotation matrix.

    Parameters
    ----------
    q : (..., 4) tensor, order (x, y, z, w). Need not be pre-normalized.

    Returns
    -------
    (..., 3, 3) rotation matrix.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)  # (..., 3, 3)


def pose_to_se3(translation: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    """
    Build a batched 4x4 homogeneous transform from world-frame
    translation + quaternion.

    Parameters
    ----------
    translation : (B, 3)
    quaternion  : (B, 4)  order (x, y, z, w)

    Returns
    -------
    (B, 4, 4)
    """
    B = translation.shape[0]
    device, dtype = translation.device, translation.dtype

    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = quaternion_to_matrix(quaternion.to(dtype))
    T[:, :3, 3] = translation
    return T


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Batched rigid-transform inverse. T: (B, 4, 4)."""
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, t.unsqueeze(-1)).squeeze(-1)

    T_inv = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.shape[0], 1, 1)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3] = t_inv
    return T_inv


def relative_transform(
    translation_target: torch.Tensor,
    quaternion_target: torch.Tensor,
    translation_source: torch.Tensor,
    quaternion_source: torch.Tensor,
) -> torch.Tensor:
    """
    Relative SE(3) transform mapping a point expressed in the SOURCE
    camera frame into the TARGET camera frame:

        T_target_source = T_world_target^-1 @ T_world_source

    All inputs are (B, 3) / (B, 4) world-frame GT camera poses.

    Returns
    -------
    (B, 4, 4)
    """
    T_world_target = pose_to_se3(translation_target, quaternion_target)
    T_world_source = pose_to_se3(translation_source, quaternion_source)
    return torch.bmm(se3_inverse(T_world_target), T_world_source)


def se3_to_6dof_translation_rotmat(T: torch.Tensor):
    """Split a (B,4,4) transform into translation (B,3) and rotation (B,3,3)."""
    return T[:, :3, 3], T[:, :3, :3]


def matrix_to_9d_pose(T: torch.Tensor) -> torch.Tensor:
    """
    Convert a (B,4,4) SE(3) transform into the 9-DoF pose vector
    [tx,ty,tz, a1x,a1y,a1z, a2x,a2y,a2z] expected by
    `LatentRenderer` (rotation_type='6d'), using the first two
    columns of R as the 6D rotation representation (Zhou et al.).
    Because R is a genuine rotation matrix (from a real quaternion),
    this round-trips through LatentRenderer.pose_to_matrix exactly
    (Gram-Schmidt on two already-orthonormal columns is a no-op).

    Parameters
    ----------
    T : (B, 4, 4)

    Returns
    -------
    (B, 9)
    """
    t = T[:, :3, 3]
    R = T[:, :3, :3]
    a1 = R[:, :, 0]
    a2 = R[:, :, 1]
    return torch.cat([t, a1, a2], dim=1)


def batch_gt_relative_pose_9d(
    camera_motions_target,
    camera_motions_source,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convenience wrapper: build the 9-DoF GT relative pose
    (source -> target) directly from lists of `CameraMotion` objects
    (one per batch sample), matching `LatentRenderer`'s expected
    input format.

    Parameters
    ----------
    camera_motions_target : list[CameraMotion], length B
        GT camera pose at the frame we are warping INTO (e.g. frame t).
    camera_motions_source : list[CameraMotion], length B
        GT camera pose at the frame we are warping FROM (e.g. frame t-1).

    Returns
    -------
    (B, 9) pose tensor, ready for LatentRenderer(rotation_type='6d').
    """
    import numpy as np

    t_tgt = torch.as_tensor(
        np.stack([cm.translation for cm in camera_motions_target]), device=device, dtype=dtype
    )
    q_tgt = torch.as_tensor(
        np.stack([cm.quaternion for cm in camera_motions_target]), device=device, dtype=dtype
    )
    t_src = torch.as_tensor(
        np.stack([cm.translation for cm in camera_motions_source]), device=device, dtype=dtype
    )
    q_src = torch.as_tensor(
        np.stack([cm.quaternion for cm in camera_motions_source]), device=device, dtype=dtype
    )

    T_rel = relative_transform(t_tgt, q_tgt, t_src, q_src)
    return matrix_to_9d_pose(T_rel)


def gt_pose_validity(camera_motions) -> torch.Tensor:
    """Boolean tensor (B,) — True where `CameraMotion.pose_available`."""
    return torch.tensor([bool(cm.pose_available) for cm in camera_motions], dtype=torch.bool)
