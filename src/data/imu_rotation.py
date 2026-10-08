"""
Camera rotation from the camera's own inertial sensor (gyroscope), for
STAGE B0: pseudo-labels without motion-capture geometry.

1. integrate_gyro: angular velocity samples over one frame window
   -> rotation vector (axis * angle) in the sensor's own axes.

2. calibrate_axes: the dataset gives no transform between the inertial
   sensor's axes and the camera's axes. We find it WITHOUT ground truth:
   for each of the 48 ways to map three axes onto three axes (with
   signs), rotate-compensate the events of a few frames and measure how
   sharp the resulting event image is. The true mapping makes the static
   scene collapse onto sharp edges, so it gives the highest sharpness.
   (Including mirrored mappings also covers the "which direction does the
   rotation act" sign convention.)
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial.transform import Rotation

from selfsup_signal_check import pixels_to_normalized, project


def integrate_gyro(timestamps, angular_velocity, t0: float, t1: float):
    """
    Rotation vector (rad) accumulated over [t0, t1] from gyroscope samples
    (piecewise-constant angular velocity around each sample).
    Returns None if no sample falls inside the window.
    """
    ts = np.asarray(timestamps, np.float64)
    w = np.asarray(angular_velocity, np.float64).reshape(-1, 3)
    if len(ts) == 0 or t1 <= t0:
        return None
    order = np.argsort(ts)
    ts, w = ts[order], w[order]
    inside = (ts >= t0) & (ts <= t1)
    if not inside.any():
        return None
    ts, w = ts[inside], w[inside]
    mids = (ts[:-1] + ts[1:]) / 2.0
    edges = np.concatenate([[t0], mids, [t1]])
    dt = np.diff(edges)
    return (w * dt[:, None]).sum(axis=0)


def signed_permutations() -> list[np.ndarray]:
    """All 48 3x3 matrices with one +/-1 per row and column."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                M[row, col] = s
            mats.append(M)
    return mats


def rotation_from_imu(theta_imu: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Rotation matrix R (static point P_end = R @ P_start, camera axes)."""
    return Rotation.from_rotvec(axes @ np.asarray(theta_imu, np.float64)).as_matrix()


def _event_image_sharpness(x, y, H, W):
    """Events-per-event concentration of the warped event image (higher = sharper)."""
    xi = np.round(x).astype(np.int64)
    yi = np.round(y).astype(np.int64)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    if ok.sum() == 0:
        return 0.0
    img = np.bincount(yi[ok] * W + xi[ok], minlength=H * W).astype(np.float64)
    return float((img ** 2).sum() / max(1.0, img.sum()))


class FrameForCalibration:
    """Precomputed per-frame data so each candidate mapping is cheap to score."""

    def __init__(self, xy, t, t0, t1, K, dist, theta_imu, max_events=100_000, seed=0):
        xy = np.asarray(xy, np.float64)
        t = np.asarray(t, np.float64)
        if len(t) > max_events:
            keep = np.random.default_rng(seed).choice(len(t), max_events, replace=False)
            xy, t = xy[keep], t[keep]
        self.x, self.y = xy[:, 0], xy[:, 1]
        self.alpha = np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
        nx, ny = pixels_to_normalized(self.x, self.y, K, dist)
        self.rays = np.stack([nx, ny, np.ones_like(nx)], axis=1)
        self.K, self.dist = K, dist
        self.theta_imu = np.asarray(theta_imu, np.float64)

    def sharpness(self, R, H, W):
        if R is None:
            return _event_image_sharpness(self.x, self.y, H, W)
        moved = self.rays @ R.T
        proj = project(moved, self.K, self.dist)
        fx = proj[:, 0] - self.x
        fy = proj[:, 1] - self.y
        bad = ~np.isfinite(fx) | ~np.isfinite(fy) | (moved[:, 2] <= 1e-3)
        fx[bad] = 0.0
        fy[bad] = 0.0
        return _event_image_sharpness(self.x - self.alpha * fx, self.y - self.alpha * fy, H, W)


def calibrate_axes(frames: list[FrameForCalibration], H: int, W: int):
    """
    Returns (best_axes (3x3), table) where table is a list of
    (mean sharpness gain over no compensation, axes) sorted best first.
    """
    if not frames:
        raise ValueError("No frames available for axis calibration.")
    base = np.array([f.sharpness(None, H, W) for f in frames])
    base = np.maximum(base, 1e-9)
    table = []
    for M in signed_permutations():
        s = np.array([f.sharpness(rotation_from_imu(f.theta_imu, M), H, W) for f in frames])
        table.append((float(np.mean(s / base)), M))
    table.sort(key=lambda r: -r[0])
    return table[0][1], table


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))
