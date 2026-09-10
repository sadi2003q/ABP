"""
IMU rotation integration for v3.

Turns raw per-frame gyroscope samples into an exact (non-learned)
rotation-matrix prior via piecewise-constant angular-velocity
integration (rectangular rule on the exponential map).

This is deliberately NOT a learned component: the whole point is to
give the pose head a physically grounded rotation estimate that
gradient descent cannot "explain away" by re-deriving pose purely
from photometric error, which is the failure mode diagnosed in v1/v2
(pose collapsed to a self-consistent-but-arbitrary solution because
nothing outside the photometric loss constrained it).

Math
----
Between two consecutive IMU samples i, i+1 with timestamps t_i, t_i+1
and angular velocity omega_i (measured at sample i, rad/s), the
rotation accumulated over that sub-interval is approximated as
constant-rate:

    dtheta_i = omega_i * (t_{i+1} - t_i)                      (B,3)
    dR_i     = exp( [dtheta_i]_x )     (Rodrigues / SO(3) exp map)

The frame's total rotation is the ordered composition:

    R_imu = dR_0 @ dR_1 @ ... @ dR_{n-1}

For a frame with zero or one IMU sample (no interval to integrate),
R_imu falls back to the identity matrix — i.e. "no information",
which composes as a no-op with whatever the residual head predicts.

This is intentionally simple (no bias estimation, no gravity
handling — gyro bias/gravity don't corrupt *rotation-rate*
integration the way they corrupt accelerometer double-integration
for position, so a first-order integrator is a reasonable prior
here, unlike the accel case).
"""

from __future__ import annotations

import torch


def _so3_exp(theta: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues' rotation formula: axis-angle vector -> rotation matrix.

    Parameters
    ----------
    theta : (..., 3)
        Axis-angle vector (rotation axis * angle, radians).

    Returns
    -------
    R : (..., 3, 3)
    """
    eps = 1e-8

    angle = torch.linalg.norm(theta, dim=-1, keepdim=True)  # (...,1)
    axis = theta / (angle + eps)

    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]

    zeros = torch.zeros_like(x)

    K = torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )  # (...,3,3), skew-symmetric

    angle = angle.unsqueeze(-1)  # (...,1,1)
    sin = torch.sin(angle)
    cos = torch.cos(angle)

    eye = torch.eye(
        3,
        device=theta.device,
        dtype=theta.dtype,
    ).expand(K.shape)

    K2 = torch.matmul(K, K)

    R = eye + sin * K + (1 - cos) * K2

    #
    # For near-zero angle, K/K2 terms vanish correctly (sin, 1-cos -> 0),
    # so this is numerically safe without a separate small-angle branch.
    #

    return R


def integrate_frame_rotation(
    imu_timestamps,
    imu_angular_velocity,
    imu_sample_indices,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Integrate raw gyro samples into one rotation matrix per batch item.

    IMPORTANT — timestamp units
    ---------------------------
    Callers MUST pass REAL (unnormalized) timestamps in seconds, e.g.
    `frame.metadata.imu_timestamps` (the untouched EVIMO2Batch stored
    on VoxelFrameBatch.metadata), NOT `frame.imu_timestamps` off the
    tensor/voxel batch. The latter has been rescaled to [0, 1] per
    frame by the NormalizeIMU transform (first sample -> 0, last -> 1),
    independently for every frame — using it here would make
    `omega * dt` an arbitrary, frame-inconsistent quantity instead of
    a real rotation angle in radians, silently producing a rotation
    prior with no physical meaning (valid-looking matrices, wrong
    content). `metadata` is preserved specifically because ToTensor's
    docstring says "Metadata ... remain inside the original
    EVIMO2Batch" — i.e. untouched by any later normalization.

    Parameters
    ----------
    imu_timestamps : (N_imu,)
        Real timestamps in seconds. numpy array or torch tensor.
    imu_angular_velocity : (N_imu, 3)
        Gyro reading (rad/s) at each timestamp. numpy array or torch
        tensor.
    imu_sample_indices : (N_imu,)
        Which batch item each IMU sample belongs to. numpy array or
        torch tensor.
    batch_size : int
    device : torch.device
        Where to place the output (and any input arrays that need
        converting to tensors).
    dtype : torch.dtype
        Floating dtype for the output and for converted inputs.

    Returns
    -------
    R_imu : (batch_size, 3, 3)
        Integrated rotation matrix per batch item. Identity for any
        item with fewer than 2 IMU samples (nothing to integrate).
    """
    if not torch.is_tensor(imu_timestamps):
        imu_timestamps = torch.as_tensor(imu_timestamps, dtype=dtype)
    if not torch.is_tensor(imu_angular_velocity):
        imu_angular_velocity = torch.as_tensor(imu_angular_velocity, dtype=dtype)
    if not torch.is_tensor(imu_sample_indices):
        imu_sample_indices = torch.as_tensor(imu_sample_indices, dtype=torch.long)

    imu_timestamps = imu_timestamps.to(device=device, dtype=dtype)
    imu_angular_velocity = imu_angular_velocity.to(device=device, dtype=dtype)
    imu_sample_indices = imu_sample_indices.to(device=device, dtype=torch.long)

    rotations = []

    for sample_index in range(batch_size):

        mask = imu_sample_indices == sample_index

        t = imu_timestamps[mask]
        omega = imu_angular_velocity[mask]

        n = t.shape[0]

        if n < 2:
            #
            # Not enough samples to integrate an interval (e.g. the
            # final frame in a sequence, which owns no IMU samples
            # because nothing follows it). Fall back to identity:
            # the residual rotation predicted by the network becomes
            # the sole source of rotation for this frame, exactly as
            # in v2 before this change.
            #
            rotations.append(
                torch.eye(3, device=device, dtype=dtype)
            )
            continue

        dt = t[1:] - t[:-1]  # (n-1,)
        dt = dt.clamp(min=0.0)  # guard against any out-of-order samples

        #
        # Piecewise-constant angular velocity: use the rate measured
        # at the START of each sub-interval.
        #
        omega_step = omega[:-1]  # (n-1, 3)

        dtheta = omega_step * dt.unsqueeze(-1)  # (n-1, 3)

        dR = _so3_exp(dtheta)  # (n-1, 3, 3)

        R = torch.eye(3, device=device, dtype=dtype)
        for i in range(dR.shape[0]):
            R = R @ dR[i]

        rotations.append(R)

    return torch.stack(rotations, dim=0)  # (B, 3, 3)


def rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Extract the Zhou et al. 6D rotation representation from a
    rotation matrix: simply its first two columns.

    Parameters
    ----------
    R : (..., 3, 3)

    Returns
    -------
    (..., 6)   -> [a1(3), a2(3)]
    """
    a1 = R[..., :, 0]
    a2 = R[..., :, 1]
    return torch.cat([a1, a2], dim=-1)
