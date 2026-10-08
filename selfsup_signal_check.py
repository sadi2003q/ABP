"""
selfsup_signal_check.py — measure how much "moving object" information
an EVENT-NATIVE self-supervised signal contains, BEFORE training anything.

Why this exists
---------------
The earlier self-supervised attempts used a photometric loss: warp the
event voxel grid of frame t-1 into frame t and compare. That assumes
the same scene point produces the same events in both windows. Event
cameras break that assumption — an edge fires events only when its
brightness CHANGES, so the event pattern depends on how the camera
moves relative to the edge. The residual is therefore large almost
everywhere, even with perfect (ground-truth) depth and pose, which
matches what was observed.

The event-native signal is MOTION COMPENSATION inside one window:
warp every event back to the window's reference time using the camera
motion. Events from the static scene collapse onto sharp edges, and
each such pixel collects events from the WHOLE time window. Events
from an independently moving object follow a different motion, stay
smeared into trails, and each pixel on a trail only sees a narrow
slice of the window. That difference needs no brightness constancy.

What this script measures
-------------------------
For every frame with a ground-truth mask it builds three versions of
the warp ("compensation modes"):
    none      : no compensation (baseline)
    rotation  : camera rotation only — what an inertial sensor's
                gyroscope can provide without any learning
    full      : rotation + translation + depth — the ceiling for any
                method that predicts depth and camera motion
and per-pixel "moving" scores computed from the warped events:
    contrast  : local contrast maximisation (main score). Try small extra
                motions on top of the camera motion; if one makes the
                local events much sharper than "no extra motion", the
                region moves on its own. Robust to object texture.
    spread    : 1 - (variance of event times at the pixel / 1/12).
                Static, well-compensated pixels see times spread over
                the whole window (variance ~1/12); moving-object pixels
                see a narrow slice (variance ~0).
    tgrad     : gradient of the mean event-time image (Mitrokhin et al.
                2018, "Event-based moving object detection and tracking").
Each score is compared with the real (speed-aware) moving-object mask:
    event IoU  : only at pixels that received events (signal quality)
    filled IoU : after closing/hole-filling the thresholded score into
                 blobs (what a pseudo-label would look like)
Ground-truth depth and pose are used here ONLY to measure the ceiling
of the signal; the self-supervised model will have to estimate them.

Usage
-----
python selfsup_signal_check.py \
    --dataset-root /content/drive/MyDrive/single_seq_root \
    --sensors left_camera --subset imo --split train \
    --sequence scene15_dyn_test_06_000000 \
    --out-dir /content/drive/MyDrive/runs/selfsup_signal_check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger("selfsup_signal_check")

MODES = ("none", "rotation", "full")
SCORES = ("contrast", "spread", "tgrad")
THRESHOLDS = np.round(np.arange(0.1, 0.95, 0.1), 2)

try:
    import cv2  # optional: lens distortion
except Exception:  # pragma: no cover
    cv2 = None


# ======================================================================
# Geometry
# ======================================================================

def relative_pose(camera_source, camera_target):
    """(R, t) such that a static point P_src in the source camera frame
    is at R @ P_src + t in the target camera frame (same convention as
    src/models/gt_mask_model/gt_pose_utils.relative_transform)."""
    import torch
    from src.models.gt_mask_model.gt_pose_utils import relative_transform

    T = relative_transform(
        torch.as_tensor(np.asarray(camera_target.translation, np.float64))[None],
        torch.as_tensor(np.asarray(camera_target.quaternion, np.float64))[None],
        torch.as_tensor(np.asarray(camera_source.translation, np.float64))[None],
        torch.as_tensor(np.asarray(camera_source.quaternion, np.float64))[None],
    )[0].numpy()
    return T[:3, :3], T[:3, 3]


def _use_distortion(dist) -> bool:
    return cv2 is not None and dist is not None and np.any(np.abs(np.asarray(dist)) > 1e-9)


def pixels_to_normalized(u, v, K, dist):
    pts = np.stack([u, v], axis=1).astype(np.float64)
    if _use_distortion(dist):
        n = cv2.undistortPoints(pts.reshape(-1, 1, 2), K.astype(np.float64),
                                np.asarray(dist, np.float64)).reshape(-1, 2)
        return n[:, 0], n[:, 1]
    return (pts[:, 0] - K[0, 2]) / K[0, 0], (pts[:, 1] - K[1, 2]) / K[1, 1]


def project(P, K, dist):
    if _use_distortion(dist):
        img, _ = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                   K.astype(np.float64), np.asarray(dist, np.float64))
        return img.reshape(-1, 2)
    Z = np.where(np.abs(P[:, 2]) < 1e-9, 1e-9, P[:, 2])
    return np.stack([K[0, 0] * P[:, 0] / Z + K[0, 2], K[1, 1] * P[:, 1] / Z + K[1, 2]], axis=1)


def ego_flow(H, W, K, dist, R, t, depth=None, mode="full",
             min_depth=0.05, max_depth=20.0):
    """Displacement (H, W, 2), in pixels, of static scene points over the
    full window, induced by the camera motion (R, t). Pixels without
    valid depth fall back to rotation-only (point at infinity)."""
    if mode == "none":
        return np.zeros((H, W, 2), np.float32)
    v, u = np.mgrid[0:H, 0:W]
    u, v = u.ravel().astype(np.float64), v.ravel().astype(np.float64)
    x, y = pixels_to_normalized(u, v, K, dist)
    rays = np.stack([x, y, np.ones_like(x)], axis=1)

    rotated = rays @ R.T                       # point at infinity: translation has no effect
    moved = rotated
    if mode == "full" and depth is not None:
        Z = depth.ravel().astype(np.float64)
        ok = np.isfinite(Z) & (Z > min_depth) & (Z < max_depth)
        full = (rays * Z[:, None]) @ R.T + t[None, :]
        moved = np.where(ok[:, None], full, rotated)

    in_front = moved[:, 2] > 1e-3
    proj = project(np.where(in_front[:, None], moved, rays), K, dist)
    flow = proj - np.stack([u, v], axis=1)
    flow[~in_front] = 0.0
    flow[~np.isfinite(flow).all(axis=1)] = 0.0
    return flow.reshape(H, W, 2).astype(np.float32)


# ======================================================================
# Motion compensation and scores
# ======================================================================

def compensate(xy, t, t0, t1, flow):
    """Warp events back to the reference time t0 along the ego-motion flow."""
    H, W = flow.shape[:2]
    alpha = np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)
    xi = np.clip(np.round(x).astype(int), 0, W - 1)
    yi = np.clip(np.round(y).astype(int), 0, H - 1)
    f = flow[yi, xi]
    return x - alpha * f[:, 0], y - alpha * f[:, 1], alpha


def splat(x, y, w, H, W):
    """Bilinear accumulation of weights w at real-valued (x, y)."""
    img = np.zeros((H + 1) * (W + 1), np.float64)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                       (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xx, yy = x0 + dx, y0 + dy
        ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        np.add.at(img, yy[ok] * (W + 1) + xx[ok], (w * wt)[ok])
    return img.reshape(H + 1, W + 1)[:H, :W]


def scores_from_warped(xw, yw, alpha, H, W, sigma=1.0):
    count = ndimage.gaussian_filter(splat(xw, yw, np.ones_like(alpha), H, W), sigma)
    s1 = ndimage.gaussian_filter(splat(xw, yw, alpha, H, W), sigma)
    s2 = ndimage.gaussian_filter(splat(xw, yw, alpha ** 2, H, W), sigma)

    eps = 1e-6
    mean_t = s1 / (count + eps)
    var_t = np.clip(s2 / (count + eps) - mean_t ** 2, 0.0, None)

    support = count[count > 1e-3]
    # Confidence = "enough events NEARBY", measured over a wider window.
    # It must not use the per-pixel count: compensated static edges pile
    # many events onto one pixel while a moving object's trail spreads
    # its events thinly, so a per-pixel count would suppress exactly the
    # moving pixels. Only isolated noise events should be suppressed.
    density = ndimage.gaussian_filter(splat(xw, yw, np.ones_like(alpha), H, W), 2.0)
    d_ref = 1.0 / (np.sqrt(2 * np.pi) * 2.0)   # density of a 1-event-per-pixel line
    confidence = 1.0 - np.exp(-density / (0.5 * d_ref))

    # The time variance is only defined where events actually landed.
    # Without this, empty pixels (variance 0) look "moving" — e.g. the
    # places static events were moved AWAY from by compensation.
    single_event_peak = 1.0 / (2 * np.pi * sigma ** 2)
    support_weight = count / (count + 0.3 * single_event_peak)
    confidence = confidence * support_weight

    spread = np.clip(1.0 - var_t / (1.0 / 12.0), 0.0, 1.0) * confidence

    gx = ndimage.sobel(mean_t, axis=1)
    gy = ndimage.sobel(mean_t, axis=0)
    tgrad = np.hypot(gx, gy) * confidence
    tgrad = ndimage.gaussian_filter(tgrad, 1.5)
    hi = np.percentile(tgrad[count > 1e-3], 99) if support.size else 1.0
    tgrad = np.clip(tgrad / max(hi, 1e-9), 0.0, 1.0)

    out = {"spread": spread, "tgrad": tgrad, "count": count}
    out["contrast"] = contrast_score(xw, yw, alpha, H, W) * (density > 0.25 * d_ref)
    return out


def _nearest_image(x, y, H, W):
    xi = np.round(x).astype(np.int64)
    yi = np.round(y).astype(np.int64)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    return np.bincount(yi[ok] * W + xi[ok], minlength=H * W).reshape(H, W).astype(np.float64)


def contrast_score(xw, yw, alpha, H, W, max_residual=16, step=4, sigma=3.0):
    """
    Local contrast maximisation. After ego-motion compensation, try a
    grid of extra constant-velocity motions d (pixels over the window):
        x' = x - alpha * d
    and measure local sharpness  G*(I^2) / G*I  (events per event in the
    neighbourhood; higher when events pile up). Static, well-compensated
    regions are sharpest at d = 0; an independently moving region becomes
    much sharper at some d != 0. Score = 1 - sharpness(d=0) / best sharpness.
    """
    eps = 1e-6
    best = None
    s0 = None
    r = np.arange(-max_residual, max_residual + 1, step)
    for dy in r:
        for dx in r:
            img = _nearest_image(xw - alpha * dx, yw - alpha * dy, H, W)
            sharp = ndimage.gaussian_filter(img ** 2, sigma) / (ndimage.gaussian_filter(img, sigma) + eps)
            if dx == 0 and dy == 0:
                s0 = sharp
            best = sharp if best is None else np.maximum(best, sharp)
    return np.clip(1.0 - s0 / (best + eps), 0.0, 1.0)


def fill_blobs(binary, close_radius=5, min_area_fraction=0.001):
    r = close_radius
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2) <= r * r
    closed = ndimage.binary_closing(binary, structure=disk)
    filled = ndimage.binary_fill_holes(closed)
    labels, n = ndimage.label(filled)
    if n == 0:
        return filled
    sizes = ndimage.sum(filled, labels, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_area_fraction * binary.size
    return keep[labels]


class PooledIoU:
    def __init__(self):
        self.tp = np.zeros(len(THRESHOLDS))
        self.fp = np.zeros(len(THRESHOLDS))
        self.fn = np.zeros(len(THRESHOLDS))

    def update(self, i, pred, gt):
        self.tp[i] += np.sum(pred & gt)
        self.fp[i] += np.sum(pred & ~gt)
        self.fn[i] += np.sum(~pred & gt)

    def result(self):
        iou = self.tp / np.maximum(self.tp + self.fp + self.fn, 1)
        k = int(np.argmax(iou))
        return float(iou[k]), float(THRESHOLDS[k])


# ======================================================================
# Per-frame analysis (pure numpy; also used by the synthetic test)
# ======================================================================

def analyse_frame(xy, t, t0, t1, K, dist, R, trans, depth, gt_mask, metrics, raw_pixels=None):
    """Update `metrics[(mode, score, kind)]` with one frame. Returns maps for plotting."""
    H, W = gt_mask.shape
    if raw_pixels is None:
        raw_pixels = np.zeros((H, W), bool)
        xi = np.clip(xy[:, 0].astype(int), 0, W - 1)
        yi = np.clip(xy[:, 1].astype(int), 0, H - 1)
        raw_pixels[yi, xi] = True

    maps = {}
    for mode in MODES:
        flow = ego_flow(H, W, K, dist, R, trans, depth=depth, mode=mode)
        xw, yw, alpha = compensate(xy, t, t0, t1, flow)
        sc = scores_from_warped(xw, yw, alpha, H, W)
        maps[mode] = sc
        for name in SCORES:
            s = sc[name]
            for i, thr in enumerate(THRESHOLDS):
                pred = s > thr
                metrics[(mode, name, "event")].update(i, pred[raw_pixels], gt_mask[raw_pixels])
                metrics[(mode, name, "filled")].update(i, fill_blobs(pred), gt_mask)
    return maps


def new_metrics():
    return {(m, s, k): PooledIoU() for m in MODES for s in SCORES for k in ("event", "filled")}


# ======================================================================
# Main (real data)
# ======================================================================

def _depth_in_metres(depth):
    if depth is None:
        return None
    d = depth.astype(np.float64)
    valid = d[np.isfinite(d) & (d > 0)]
    if valid.size and np.median(valid) > 50:   # stored in millimetres
        d = d / 1000.0
    return d


def main():
    p = argparse.ArgumentParser(description="Measure the motion-compensation signal against ground-truth masks")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--sensors", nargs="+", default=["left_camera"])
    p.add_argument("--subset", default="imo")
    p.add_argument("--split", default="train")
    p.add_argument("--sequence", nargs="+", default=None)
    p.add_argument("--max-frames", type=int, default=None, help="Limit frames for a quick run.")
    p.add_argument("--stride", type=int, default=1, help="Use every Nth frame.")
    p.add_argument("--num-plots", type=int, default=6)
    p.add_argument("--out-dir", default="runs/selfsup_signal_check")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "signal_check.log")], force=True)

    from src.data.dataset import EVIMO2Dataset
    from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic

    ds = EVIMO2Dataset(dataset_root=args.dataset_root, sensors=tuple(args.sensors),
                       split=args.split, load_depth=True, load_mask=True,
                       subset=args.subset, sequence=args.sequence)
    refs = ds.index.references
    logger.info(f"Distortion handling: {'OpenCV' if cv2 is not None else 'pinhole only (OpenCV missing)'}")

    metrics = new_metrics()
    trivial = PooledIoU()
    used, plotted, skipped = 0, 0, {"no_mask": 0, "no_depth": 0, "no_pose": 0, "no_moving": 0, "no_events": 0}

    for i in range(0, len(ds) - 1, args.stride):
        if args.max_frames is not None and used >= args.max_frames:
            break
        if refs[i].sequence_id != refs[i + 1].sequence_id:
            continue
        a, b = ds[i], ds[i + 1]
        if a.mask is None:
            skipped["no_mask"] += 1; continue
        if a.depth is None:
            skipped["no_depth"] += 1; continue
        if not (a.camera_motion.pose_available and b.camera_motion.pose_available):
            skipped["no_pose"] += 1; continue
        if len(a.events_t) == 0:
            skipped["no_events"] += 1; continue

        gt = evimo2_mask_to_binary_dynamic(a.mask, get_dynamic_object_ids(a.frame_motion)).numpy()
        while gt.ndim > 2:
            gt = gt[0]
        if not gt.any():
            skipped["no_moving"] += 1; continue

        R, trans = relative_pose(a.camera_motion, b.camera_motion)
        depth = _depth_in_metres(a.depth)
        K = np.asarray(a.camera_intrinsics, np.float64)
        dist = np.asarray(a.camera_distortion, np.float64)
        xy = np.asarray(a.events_xy)
        t = np.asarray(a.events_t, np.float64)

        H, W = gt.shape
        raw = np.zeros((H, W), bool)
        raw[np.clip(xy[:, 1].astype(int), 0, H - 1), np.clip(xy[:, 0].astype(int), 0, W - 1)] = True
        for k in range(len(THRESHOLDS)):
            trivial.update(k, raw[raw], gt[raw])

        maps = analyse_frame(xy, t, a.timestamp, b.timestamp, K, dist, R, trans, depth, gt, metrics, raw)
        used += 1

        if plotted < args.num_plots:
            _plot(out / f"frame_{i:05d}.png", raw, maps, gt, depth)
            plotted += 1
        if used % 10 == 0:
            logger.info(f"  processed {used} frames with moving objects")

    logger.info(f"Frames used: {used} | skipped: {skipped}")
    if used == 0:
        logger.error("No usable frames (need mask + depth + pose + moving object).")
        return

    report = {"frames_used": used, "skipped": skipped,
              "trivial_all_event_pixels_iou": trivial.result()[0], "results": {}}
    logger.info("=" * 78)
    logger.info("Trivial baseline (every pixel with an event = moving): "
                f"event IoU = {report['trivial_all_event_pixels_iou']:.4f}")
    logger.info(f"{'compensation':<13}{'score':<8}{'event IoU':>11}{'thr':>6}{'filled IoU':>13}{'thr':>6}")
    for mode in MODES:
        for name in SCORES:
            e_iou, e_thr = metrics[(mode, name, "event")].result()
            f_iou, f_thr = metrics[(mode, name, "filled")].result()
            report["results"][f"{mode}/{name}"] = {"event_iou": e_iou, "event_thr": e_thr,
                                                   "filled_iou": f_iou, "filled_thr": f_thr}
            logger.info(f"{mode:<13}{name:<8}{e_iou:>11.4f}{e_thr:>6.1f}{f_iou:>13.4f}{f_thr:>6.1f}")
    logger.info("=" * 78)
    with open(out / "signal_check.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved {out / 'signal_check.json'} and {plotted} figure(s).")


def _plot(path, raw, maps, gt, depth):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    panels = [
        ("Events, no compensation", maps["none"]["count"], "gray"),
        ("Events, rotation compensated", maps["rotation"]["count"], "gray"),
        ("Events, full compensation", maps["full"]["count"], "gray"),
        ("True moving-object mask", gt.astype(float), "gray"),
        ("Moving score: none", maps["none"]["contrast"], "magma"),
        ("Moving score: rotation", maps["rotation"]["contrast"], "magma"),
        ("Moving score: full", maps["full"]["contrast"], "magma"),
        ("Depth (metres)", depth if depth is not None else np.zeros_like(gt, float), "viridis"),
    ]
    for ax, (title, img, cmap) in zip(axes.ravel(), panels):
        if "Events" in title:
            img = np.clip(img / (np.percentile(img[img > 0], 99) if (img > 0).any() else 1), 0, 1)
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
