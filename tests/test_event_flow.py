"""
Stage B1 tests (CPU only, under a minute):

    python tests/test_event_flow.py

1. The analytic sharpness gradient matches a finite-difference estimate.
2. A known whole-image shift is recovered from events alone.
3. KEY CHECK: on the synthetic scene used for stage A (slanted surface,
   camera rotating AND translating, plus an independently moving
   square), the flow fitted from events alone is close to the true
   background flow, and the resulting pseudo-label finds the square.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.event_flow import _Problem, fit_background_flow, flow_field
from selfsup_signal_check import ego_flow
from generate_pseudo_labels import score_from_flow, score_to_label
from test_selfsup_signal import make_scene, K, DIST, H, W


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    n = 20_000
    x, y = rng.uniform(0, W, n), rng.uniform(0, H, n)
    # structure: events concentrated on a few vertical lines, so sharpness varies with the shift
    x = np.round(x / 20) * 20 + rng.normal(0, 0.5, n)
    alpha = rng.uniform(0, 1, n)
    prob = _Problem(x, y, alpha, H, W, "planar")
    theta = np.array([2.3, 0.7, -0.4, -1.1, 0.5, 0.3, 0.2, -0.3])
    _, g = prob.value_and_gradient(theta, 2.0)
    eps = 1e-3
    fd = np.zeros_like(theta)
    for k in range(len(theta)):
        d = np.zeros_like(theta); d[k] = eps
        fd[k] = (prob.value_and_gradient(theta + d, 2.0)[0] - prob.value_and_gradient(theta - d, 2.0)[0]) / (2 * eps)
    rel = np.linalg.norm(g - fd) / max(np.linalg.norm(fd), 1e-12)
    print(f"  gradient vs finite difference: relative difference {rel:.3f}")
    assert rel < 0.1, f"analytic gradient wrong\n{g}\nvs\n{fd}"
    print("  analytic gradient ................... ok")


def test_recovers_known_shift():
    rng = np.random.default_rng(2)
    n_pts, shift = 2000, np.array([7.0, -5.0])
    px, py = rng.uniform(10, W - 10, n_pts), rng.uniform(10, H - 10, n_pts)
    xs, ts = [], []
    for _ in range(10):
        a = rng.uniform(0, 1, n_pts)
        xs.append(np.stack([px + a * shift[0], py + a * shift[1]], 1)); ts.append(a)
    xy, t = np.concatenate(xs), np.concatenate(ts)
    for model in ("translation", "planar"):
        theta, info = fit_background_flow(xy, t, 0.0, 1.0, H, W, model=model)
        f = flow_field(theta, H, W, model)
        err = np.linalg.norm(f - shift[None, None, :], axis=2).max()
        print(f"  {model:<11} shift recovered: max error {err:.2f} px (sharpness x{info['gain']:.2f})")
        assert err < 1.0, f"{model}: wrong shift {f[H // 2, W // 2]} vs {shift}"
    print("  known shift ......................... ok")


def test_synthetic_scene():
    xy, tt, R, t, depth, obj = make_scene()
    true_flow = ego_flow(H, W, K, DIST, R, t, depth=depth, mode="full")
    static = ~obj
    moves = np.median(np.linalg.norm(true_flow, axis=2)[static])

    theta, info = fit_background_flow(xy, tt, 0.0, 1.0, H, W, model="planar")
    flow = flow_field(theta, H, W, "planar")
    err = np.median(np.linalg.norm(flow - true_flow, axis=2)[static])
    print(f"  background moves {moves:.2f} px; fitted flow error {err:.2f} px "
          f"(sharpness x{info['gain']:.2f}, parameter at limit: {info['at_bound']})")
    assert err < 1.0, "fitted background flow too far from the truth"

    def iou(flow_used):
        lab = score_to_label(score_from_flow(xy, tt, 0.0, 1.0, flow_used), 0.3, 0.003)
        return (lab & obj).sum() / max(1, (lab | obj).sum())

    iou_fit, iou_true, iou_none = iou(flow), iou(true_flow), iou(np.zeros_like(flow))
    print(f"  pseudo-label IoU: events-only flow {iou_fit:.3f} | true flow {iou_true:.3f} | "
          f"no compensation {iou_none:.3f}")
    assert iou_fit > 0.5, "events-only pseudo-label should find the moving square"
    assert iou_fit > iou_none, "fitted flow should beat no compensation"
    print("  synthetic scene ..................... ok")


if __name__ == "__main__":
    print("Stage B1 (background flow from events) self-test")
    test_gradient_matches_finite_difference()
    test_recovers_known_shift()
    test_synthetic_scene()
    print("ALL TESTS PASSED")
