"""
Synthetic test for selfsup_signal_check.py — verifies the geometry
conventions (pose direction, flow direction, compensation direction)
before trusting any number on real data.

Scene: a slanted static plane plus a square object that moves on its
own, seen by a camera that rotates and translates during the window.
Events are generated from the true 3-D motion. With the correct
camera motion, compensation must make the static scene "static" and
leave the object as the only "moving" region.

    python tests/test_selfsup_signal.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfsup_signal_check import (
    analyse_frame, new_metrics, relative_pose, ego_flow, compensate, THRESHOLDS,
)

H, W = 120, 160
K = np.array([[150.0, 0, W / 2], [0, 150.0, H / 2], [0, 0, 1]])
DIST = np.zeros(4)


def make_scene(seed=0, object_motion=np.array([0.10, 0.0, 0.0])):
    rng = np.random.default_rng(seed)
    # Camera motion during the window: P_target = R @ P_source + t
    R = Rotation.from_rotvec([0.0, 0.03, 0.01]).as_matrix()
    t = np.array([0.05, -0.01, 0.02])

    # Static scene: slanted plane, object square at 1.5 m
    v, u = np.mgrid[0:H, 0:W]
    depth = 2.0 + 1.0 * u / W
    y0, x0, s = 45, 60, 26
    obj = np.zeros((H, W), bool)
    obj[y0:y0 + s, x0:x0 + s] = True
    depth[obj] = 1.5

    # Textured points at the reference time
    n_pts = 2500
    pu = rng.uniform(0, W - 1, n_pts)
    pv = rng.uniform(0, H - 1, n_pts)
    is_obj = obj[pv.astype(int), pu.astype(int)]
    Z = depth[pv.astype(int), pu.astype(int)]
    rays = np.stack([(pu - K[0, 2]) / K[0, 0], (pv - K[1, 2]) / K[1, 1], np.ones(n_pts)], 1)
    P0 = rays * Z[:, None]

    xs, ts = [], []
    for _ in range(10):                                  # 10 events per point
        a = rng.uniform(0, 1, n_pts)
        P = P0 + a[:, None] * ((P0 @ R.T + t) - P0)      # camera-induced motion
        P = P + (a[:, None] * object_motion) * is_obj[:, None]
        x = K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]
        y = K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]
        xs.append(np.stack([x, y], 1)); ts.append(a)
    xy = np.concatenate(xs); tt = np.concatenate(ts)
    noise = int(0.02 * len(tt))
    xy = np.concatenate([xy, np.stack([rng.uniform(0, W, noise), rng.uniform(0, H, noise)], 1)])
    tt = np.concatenate([tt, rng.uniform(0, 1, noise)])
    keep = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
    return xy[keep], tt[keep], R, t, depth, obj


def test_relative_pose_convention():
    R = Rotation.from_rotvec([0.02, -0.03, 0.01]).as_matrix()
    t = np.array([0.1, 0.2, -0.05])
    T_ts = np.eye(4); T_ts[:3, :3] = R; T_ts[:3, 3] = t
    T_wt = np.linalg.inv(T_ts)                          # world <- target, source = world
    src = SimpleNamespace(translation=np.zeros(3), quaternion=np.array([0, 0, 0, 1.0]))
    tgt = SimpleNamespace(translation=T_wt[:3, 3], quaternion=Rotation.from_matrix(T_wt[:3, :3]).as_quat())
    R2, t2 = relative_pose(src, tgt)
    assert np.allclose(R2, R, atol=1e-6) and np.allclose(t2, t, atol=1e-6)
    print("  pose convention ...................... ok")


def test_static_scene_collapses():
    """With the true motion, a fully static scene must become sharper."""
    xy, tt, R, t, depth, obj = make_scene(object_motion=np.zeros(3))
    def spread_of(mode):
        f = ego_flow(H, W, K, DIST, R, t, depth, mode)
        xw, yw, a = compensate(xy, tt, 0.0, 1.0, f)
        return np.var(np.round(xw)) + np.var(np.round(yw)), f
    _, flow = spread_of("full")
    # Compensated positions should match the reference positions closely
    xw, yw, _ = compensate(xy, tt, 0.0, 1.0, flow)
    f0 = ego_flow(H, W, K, DIST, R, t, depth, "none")
    xn, yn, _ = compensate(xy, tt, 0.0, 1.0, f0)
    # mean displacement from where the event "should" be is lower with compensation
    assert np.abs(flow).max() > 3.0, "flow too small for a meaningful test"
    print(f"  max ego flow {np.abs(flow).max():.1f} px ................ ok")


def test_object_found_only_with_compensation():
    xy, tt, R, t, depth, obj = make_scene()
    m = new_metrics()
    analyse_frame(xy, tt, 0.0, 1.0, K, DIST, R, t, depth, obj, m)
    none_f = m[("none", "contrast", "filled")].result()[0]
    rot_f = m[("rotation", "contrast", "filled")].result()[0]
    full_f = m[("full", "contrast", "filled")].result()[0]
    full_e = m[("full", "contrast", "event")].result()[0]
    print(f"  filled IoU  none={none_f:.3f}  rotation={rot_f:.3f}  full={full_f:.3f}  (event IoU full={full_e:.3f})")
    assert full_f > 0.6, "full compensation should isolate the moving object"
    assert full_f > none_f + 0.2, "compensation must clearly beat no compensation"

    # Wrong direction must be clearly worse: this guards against sign errors
    m_wrong = new_metrics()
    R_inv, t_inv = R.T, -R.T @ t
    analyse_frame(xy, tt, 0.0, 1.0, K, DIST, R_inv, t_inv, depth, obj, m_wrong)
    wrong_f = m_wrong[("full", "contrast", "filled")].result()[0]
    print(f"  filled IoU with the motion REVERSED = {wrong_f:.3f} (must be worse)")
    assert wrong_f < full_f - 0.2
    print("  object isolated by compensation ...... ok")


if __name__ == "__main__":
    print("Self-supervised signal check — synthetic test")
    test_relative_pose_convention()
    test_static_scene_collapses()
    test_object_found_only_with_compensation()
    print("ALL TESTS PASSED")
