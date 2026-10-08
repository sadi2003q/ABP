"""
Stage B0 tests (CPU only, a few seconds):

    python tests/test_imu_rotation.py

1. Gyroscope integration gives the right rotation for a known motion.
2. There are exactly 48 distinct candidate axis mappings.
3. KEY CHECK: events from a purely rotating camera are generated with a
   hidden inertial-to-camera axis mapping; the calibration must recover
   that mapping from event sharpness alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.imu_rotation import (
    integrate_gyro, signed_permutations, FrameForCalibration, calibrate_axes, rotation_from_imu,
)

H, W = 120, 160
K = np.array([[150.0, 0, W / 2], [0, 150.0, H / 2], [0, 0, 1]])
DIST = np.zeros(4)


def test_integrate_gyro():
    ts = np.linspace(0.0, 0.05, 51)                      # 1 kHz
    w = np.tile([0.2, -0.1, 0.4], (51, 1))
    theta = integrate_gyro(ts, w, 0.0, 0.05)
    assert np.allclose(theta, np.array([0.2, -0.1, 0.4]) * 0.05, atol=1e-9)
    assert integrate_gyro(ts, w, 1.0, 2.0) is None, "no samples in window -> None"
    print("  gyroscope integration ............... ok")


def test_candidates():
    mats = signed_permutations()
    assert len(mats) == 48
    assert len({m.tobytes() for m in mats}) == 48
    assert all(np.allclose(m @ m.T, np.eye(3)) for m in mats)
    print("  48 axis mappings .................... ok")


def make_frame(rng, theta_cam, M_true, n_points=1500, events_per_point=8):
    """Static scene, camera rotating by theta_cam over the window (P_end = R P_start)."""
    u = rng.uniform(5, W - 5, n_points)
    v = rng.uniform(5, H - 5, n_points)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(n_points)], 1)
    xs, ts = [], []
    for _ in range(events_per_point):
        a = rng.uniform(0, 1, n_points)
        P = np.einsum("nij,nj->ni", Rotation.from_rotvec(a[:, None] * theta_cam).as_matrix(), rays)
        xs.append(np.stack([K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2], K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]], 1))
        ts.append(a)
    xy, t = np.concatenate(xs), np.concatenate(ts)
    keep = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
    theta_imu = M_true.T @ theta_cam                     # what the gyroscope would report
    return FrameForCalibration(xy[keep], t[keep], 0.0, 1.0, K, DIST, theta_imu)


def test_calibration_recovers_hidden_axes():
    rng = np.random.default_rng(0)
    mats = signed_permutations()
    for trial, idx in enumerate((7, 30)):                # two different hidden mappings
        M_true = mats[idx]
        frames = [make_frame(rng, rng.uniform(-0.06, 0.06, 3), M_true) for _ in range(6)]
        best, table = calibrate_axes(frames, H, W)
        assert np.allclose(best, M_true), f"trial {trial}: wrong mapping recovered\n{best}\nvs\n{M_true}"
        assert table[0][0] > 1.0, "true mapping must sharpen the events"
        print(f"  hidden mapping {idx:2d} recovered (sharpness x{table[0][0]:.2f}, "
              f"next best x{table[1][0]:.2f}) ... ok")


if __name__ == "__main__":
    print("Stage B0 (gyroscope rotation) self-test")
    test_integrate_gyro()
    test_candidates()
    test_calibration_recovers_hidden_axes()
    print("ALL TESTS PASSED")
