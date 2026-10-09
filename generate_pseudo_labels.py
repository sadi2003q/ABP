"""
generate_pseudo_labels.py — STAGE A of the self-supervised route.

Creates moving-object masks WITHOUT human mask labels, from the
event-native motion-compensation signal measured by
selfsup_signal_check.py:

    1. warp each frame's events back to the frame time using the
       camera motion between this frame and the next, and the depth
       (stage A: the dataset's motion-capture depth and pose);
    2. local contrast maximisation -> per-pixel "moves on its own" score;
    3. threshold + close + fill holes + drop small blobs -> pseudo-label.

Pseudo-labels are written for EVERY frame that has depth and pose,
whether or not it has a true mask. True masks are read only to REPORT
pseudo-label quality; they never influence the labels.

The report also scores a small grid of settings (score threshold x
minimum blob size) in the same pass, so the trade-off is visible.
Note: choosing the setting by looking at that report uses the true
masks of the sequence being labelled. For a paper, choose it on a
different sequence than the one you evaluate on.

Where the camera motion in step 1 comes from (--geometry):
    mocap         motion-capture depth + pose (stage A)
    imu_rotation  gyroscope rotation only (stage B0)
    events        background flow fitted to the events alone (stage B1;
                  no other sensor; depth/pose loaded only for the report)
    none          no compensation (baseline)

Usage
-----
python generate_pseudo_labels.py \
    --dataset-root /kaggle/input/datasets/makkisakib1/evimo2/single_seq_root \
    --sensors right_camera --subset imo --split train \
    --out-dir /kaggle/working/pseudo_labels_right
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from selfsup_signal_check import (
    relative_pose, ego_flow, compensate, scores_from_warped, fill_blobs, _depth_in_metres,
)
from src.data.pseudo_labels import save_sequence_labels
from src.data.imu_rotation import (
    integrate_gyro, rotation_from_imu, rotation_angle_deg, FrameForCalibration, calibrate_axes,
)
from src.data.event_flow import MODELS as FLOW_MODELS, fit_background_flow, flow_field

logger = logging.getLogger("generate_pseudo_labels")

GRID_THRESHOLDS = (0.2, 0.3, 0.4)
GRID_MIN_AREAS = (0.001, 0.003, 0.01)

GEOMETRY_DESCRIPTION = {
    "mocap": "motion-capture depth + pose (stage A)",
    "imu_rotation": "gyroscope rotation only, axes found from event sharpness (stage B0)",
    "events": "background flow fitted to the events alone (stage B1)",
    "none": "no motion compensation (baseline)",
}


def score_from_flow(xy, t, t0, t1, flow):
    """Contrast score map after undoing the background displacement `flow` (H, W, 2)."""
    H, W = flow.shape[:2]
    xw, yw, alpha = compensate(xy, t, t0, t1, flow)
    return scores_from_warped(xw, yw, alpha, H, W)["contrast"]


def pseudo_score(xy, t, t0, t1, K, dist, R, trans, depth, H=None, W=None, mode="full"):
    """mode="full": rotation + translation + depth (stage A, motion capture).
    mode="rotation": rotation only, no depth needed (stage B0, inertial sensor)."""
    if depth is not None:
        H, W = depth.shape
    flow = ego_flow(H, W, K, dist, R, trans, depth=depth, mode=mode)
    return score_from_flow(xy, t, t0, t1, flow)


def score_to_label(score, threshold, min_area_fraction, close_radius=5):
    return fill_blobs(score > threshold, close_radius=close_radius, min_area_fraction=min_area_fraction)


class Pooled:
    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.static_frames = self.static_frames_with_fp = 0

    def update(self, pred, gt):
        self.tp += int(np.sum(pred & gt)); self.fp += int(np.sum(pred & ~gt)); self.fn += int(np.sum(~pred & gt))
        if not gt.any():
            self.static_frames += 1
            self.static_frames_with_fp += int(pred.any())

    def iou(self):
        return self.tp / max(1, self.tp + self.fp + self.fn)


def _calibrate_imu_axes(ds, refs, args):
    """
    Inertial-to-camera axis mapping per camera, found from event sharpness
    alone (no ground truth). Returns ({sensor: 3x3}, report dict).
    """
    if args.imu_axes is not None:
        M = np.asarray(args.imu_axes, np.float64).reshape(3, 3)
        logger.info(f"Inertial-to-camera axes given on the command line:\n{M}")
        return {s: M for s in args.sensors}, {"source": "command line", "axes": M.tolist()}

    axes, report = {}, {}
    for sensor in args.sensors:
        frames = []
        stride = max(1, (len(refs) - 1) // max(1, 4 * args.calib_frames))
        for i in range(0, len(refs) - 1, stride):
            if len(frames) >= args.calib_frames:
                break
            if refs[i].sequence_id != refs[i + 1].sequence_id:
                continue
            a = ds[i]
            if a.sensor != sensor or len(a.events_t) < 2000:
                continue
            b = ds[i + 1]
            theta = integrate_gyro(a.imu.timestamps, a.imu.angular_velocity, a.timestamp, b.timestamp)
            if theta is None or np.linalg.norm(theta) < np.radians(0.05):
                continue          # too little rotation to tell the mappings apart
            frames.append(FrameForCalibration(
                a.events_xy, a.events_t, a.timestamp, b.timestamp,
                np.asarray(a.camera_intrinsics, np.float64),
                np.asarray(a.camera_distortion, np.float64), theta, seed=len(frames)))
        if not frames:
            raise RuntimeError(f"[{sensor}] no frames with enough rotation and events to find the "
                               f"inertial-to-camera axes; pass --imu-axes explicitly.")
        logger.info(f"[{sensor}] finding inertial-to-camera axes from {len(frames)} frames "
                    f"(48 candidates, event sharpness only)...")
        best, table = calibrate_axes(frames, args.height, args.width)
        axes[sensor] = best
        gap = table[0][0] / max(table[1][0], 1e-9)
        logger.info(f"[{sensor}] best axes (sharpness x{table[0][0]:.3f} vs no compensation; "
                    f"next best x{table[1][0]:.3f}; margin {gap:.3f}):\n{best}")
        if gap < 1.01:
            logger.warning(f"[{sensor}] the best and second-best axis mappings are nearly tied; "
                           f"the camera may rotate too little in this sequence to calibrate reliably.")
        report[sensor] = {"frames": len(frames), "axes": best.tolist(),
                          "top5": [{"gain": g, "axes": M.tolist()} for g, M in table[:5]]}
    return axes, report


def main():
    p = argparse.ArgumentParser(description="Pseudo-labels from motion compensation (stages A, B0, B1)")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--sensors", nargs="+", default=["left_camera"])
    p.add_argument("--subset", default="imo")
    p.add_argument("--split", default="train")
    p.add_argument("--sequence", nargs="+", default=None)
    p.add_argument("--threshold", type=float, default=0.3, help="Contrast-score threshold for the written labels.")
    p.add_argument("--min-area", type=float, default=0.003,
                   help="Smallest blob kept, as a fraction of the image area.")
    p.add_argument("--out-dir", required=True)
    g = p.add_argument_group("geometry")
    g.add_argument("--geometry", choices=["mocap", "imu_rotation", "events", "none"], default="mocap",
                   help="mocap: motion-capture depth + camera pose (stage A). "
                        "imu_rotation: camera rotation from the camera's own gyroscope only, "
                        "no depth, no translation, no motion capture (stage B0). "
                        "events: background flow fitted to each window's events alone, no other "
                        "sensor (stage B1). none: no compensation at all (baseline).")
    g.add_argument("--flow-model", choices=sorted(FLOW_MODELS), default="planar",
                   help="--geometry events: background flow formula (translation: 2 numbers, "
                        "affine: 6, planar: 8). See src/data/event_flow.py.")
    g.add_argument("--flow-max-events", type=int, default=80_000,
                   help="--geometry events: events used per window to fit the flow (random subsample).")
    p.add_argument("--limit-frames", type=int, default=0,
                   help="Stop after this many labelled frames (0 = all). For a quick check.")
    g.add_argument("--imu-axes", type=float, nargs=9, default=None,
                   help="Fixed 3x3 inertial-to-camera axis matrix (row-major). "
                        "Default: found automatically from event sharpness.")
    g.add_argument("--calib-frames", type=int, default=40,
                   help="Frames used to find the inertial-to-camera axis mapping.")
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--width", type=int, default=640)
    args = p.parse_args()
    geometry = args.geometry
    use_imu = geometry == "imu_rotation"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "generate.log")], force=True)

    from src.data.dataset import EVIMO2Dataset
    from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic

    # events: depth is loaded ONLY for the motion-capture comparison in the report
    ds = EVIMO2Dataset(dataset_root=args.dataset_root, sensors=tuple(args.sensors),
                       split=args.split, load_depth=geometry in ("mocap", "events"), load_mask=True,
                       subset=args.subset, sequence=args.sequence)
    refs = ds.index.references

    imu_axes, calib_report, mocap_check = {}, {}, {}
    if use_imu:
        imu_axes, calib_report = _calibrate_imu_axes(ds, refs, args)

    grid = {(thr, a): Pooled() for thr in GRID_THRESHOLDS for a in GRID_MIN_AREAS}
    chosen = Pooled()
    labels_by_seq: dict[tuple, dict[int, np.ndarray]] = {}
    n_written, n_skipped, t_start = 0, 0, time.time()
    prev_theta: dict[tuple, np.ndarray] = {}        # events: previous window's flow, as a starting guess
    fit_stats = {"gain": [], "at_bound": 0, "not_fitted": 0}
    flow_check = {"error_px": [], "mocap_px": []}    # events: report only

    for i in range(len(ds) - 1):
        if refs[i].sequence_id != refs[i + 1].sequence_id:
            continue
        if args.limit_frames and n_written >= args.limit_frames:
            break
        a, b = ds[i], ds[i + 1]
        K = np.asarray(a.camera_intrinsics, np.float64)
        dist = np.asarray(a.camera_distortion, np.float64)
        xy = np.asarray(a.events_xy)
        t = np.asarray(a.events_t, np.float64)

        gt = None
        if a.mask is not None:        # quality report only
            gt = evimo2_mask_to_binary_dynamic(a.mask, get_dynamic_object_ids(a.frame_motion)).numpy()
            while gt.ndim > 2:
                gt = gt[0]
            gt = gt.astype(bool)

        if geometry == "imu_rotation":
            theta = integrate_gyro(a.imu.timestamps, a.imu.angular_velocity, a.timestamp, b.timestamp)
            if theta is None:
                n_skipped += 1
                continue
            R = rotation_from_imu(theta, imu_axes[a.sensor])
            H, W = args.height, args.width
            if a.camera_motion.pose_available and b.camera_motion.pose_available:
                # report only: how close the gyroscope rotation is to motion capture
                R_mc, _ = relative_pose(a.camera_motion, b.camera_motion)
                mocap_check.setdefault("error_deg", []).append(rotation_angle_deg(R.T @ R_mc))
                mocap_check.setdefault("mocap_deg", []).append(rotation_angle_deg(R_mc))
            flow = ego_flow(H, W, K, dist, R, np.zeros(3), depth=None, mode="rotation")
        elif geometry == "mocap":
            if a.depth is None or not (a.camera_motion.pose_available and b.camera_motion.pose_available):
                n_skipped += 1
                continue
            depth = _depth_in_metres(a.depth)
            R, trans = relative_pose(a.camera_motion, b.camera_motion)
            H, W = depth.shape
            flow = ego_flow(H, W, K, dist, R, trans, depth=depth, mode="full")
        elif geometry == "events":
            H, W = args.height, args.width
            key = (a.sensor, a.sequence_name)
            theta, info = fit_background_flow(xy, t, a.timestamp, b.timestamp, H, W,
                                              model=args.flow_model, init=prev_theta.get(key),
                                              max_events=args.flow_max_events, seed=i)
            if info["fitted"]:
                prev_theta[key] = theta
                fit_stats["gain"].append(info["gain"])
                fit_stats["at_bound"] += int(info["at_bound"])
            else:
                fit_stats["not_fitted"] += 1
            flow = flow_field(theta, H, W, args.flow_model)
            # report only: compare with the motion-capture background flow on static pixels
            if (a.depth is not None and a.camera_motion.pose_available and b.camera_motion.pose_available
                    and np.asarray(a.depth).shape[-2:] == (H, W)):
                depth = _depth_in_metres(a.depth)
                while depth.ndim > 2:
                    depth = depth[0]
                R_mc, t_mc = relative_pose(a.camera_motion, b.camera_motion)
                flow_mc = ego_flow(H, W, K, dist, R_mc, t_mc, depth=depth, mode="full")
                valid = np.isfinite(depth) & (depth > 0.05) & (depth < 20.0)
                if gt is not None:
                    valid &= ~gt
                if valid.sum() > 100:
                    flow_check["error_px"].append(float(np.median(np.linalg.norm(flow - flow_mc, axis=2)[valid])))
                    flow_check["mocap_px"].append(float(np.median(np.linalg.norm(flow_mc, axis=2)[valid])))
        else:   # none
            H, W = args.height, args.width
            flow = np.zeros((H, W, 2), np.float32)

        if len(t) == 0:
            label = np.zeros((H, W), bool)
            score = np.zeros((H, W))
        else:
            score = score_from_flow(xy, t, a.timestamp, b.timestamp, flow)
            label = score_to_label(score, args.threshold, args.min_area)

        labels_by_seq.setdefault((a.sensor, a.sequence_name), {})[int(a.local_frame_index)] = label
        n_written += 1

        if gt is not None:            # quality report only
            chosen.update(label, gt)
            for (thr, area), pooled in grid.items():
                pooled.update(score_to_label(score, thr, area) if len(t) else label, gt)

        if n_written % 25 == 0:
            rate = (time.time() - t_start) / n_written
            logger.info(f"  labelled {n_written} frames ({rate:.1f} s/frame)")

    for (sensor, seq), labels in labels_by_seq.items():
        save_sequence_labels(out, sensor, seq, labels)

    skip_reason = {"imu_rotation": "no gyroscope samples", "mocap": "no depth/pose"}.get(geometry, "none expected")
    logger.info(f"Pseudo-labels written: {n_written} frames | skipped ({skip_reason}): {n_skipped}")
    if mocap_check.get("error_deg"):
        logger.info(f"Gyroscope vs motion-capture rotation (report only): median error "
                    f"{np.median(mocap_check['error_deg']):.3f} deg per frame, while the camera "
                    f"rotates {np.median(mocap_check['mocap_deg']):.3f} deg per frame (median)")
    if geometry == "events":
        if fit_stats["gain"]:
            logger.info(f"Event-fitted background flow ({args.flow_model}): sharpness x"
                        f"{np.median(fit_stats['gain']):.3f} vs no compensation (median over "
                        f"{len(fit_stats['gain'])} windows) | windows with a parameter at its limit: "
                        f"{fit_stats['at_bound']} | windows too sparse to fit: {fit_stats['not_fitted']}")
        if flow_check["error_px"]:
            logger.info(f"Event-fitted vs motion-capture background flow (report only, static pixels): "
                        f"median error {np.median(flow_check['error_px']):.2f} px per window, while the "
                        f"background moves {np.median(flow_check['mocap_px']):.2f} px per window (median)")
        else:
            logger.info("Event-fitted vs motion-capture background flow: no frames with depth + pose "
                        "at the event image size, comparison skipped")
    report = {
        "settings": {"threshold": args.threshold, "min_area": args.min_area,
                     "geometry": GEOMETRY_DESCRIPTION[geometry],
                     "flow_model": args.flow_model if geometry == "events" else None,
                     "limit_frames": args.limit_frames},
        "imu_calibration": calib_report,
        "event_flow": ({
            "median_sharpness_gain": float(np.median(fit_stats["gain"])) if fit_stats["gain"] else None,
            "windows_with_parameter_at_limit": fit_stats["at_bound"],
            "windows_too_sparse": fit_stats["not_fitted"],
        } if geometry == "events" else None),
        "event_flow_vs_mocap_report_only": ({
            "frames": len(flow_check["error_px"]),
            "median_error_px": float(np.median(flow_check["error_px"])),
            "median_mocap_flow_px": float(np.median(flow_check["mocap_px"])),
        } if flow_check["error_px"] else None),
        "imu_vs_mocap_rotation_report_only": ({
            "frames": len(mocap_check.get("error_deg", [])),
            "median_error_deg": float(np.median(mocap_check["error_deg"])),
            "median_mocap_rotation_deg": float(np.median(mocap_check["mocap_deg"])),
        } if mocap_check.get("error_deg") else None),
        "frames_labelled": n_written,
        "quality_vs_true_masks": {
            "iou": chosen.iou(),
            "static_frames": chosen.static_frames,
            "static_frames_with_false_blobs": chosen.static_frames_with_fp,
        },
        "grid": {f"thr={k[0]}_minarea={k[1]}": {"iou": v.iou(), "static_frames_with_false_blobs": v.static_frames_with_fp}
                 for k, v in grid.items()},
    }
    with open(out / "pseudo_label_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 70)
    logger.info("Pseudo-label quality against the TRUE masks (report only)")
    logger.info(f"  chosen setting thr={args.threshold} min_area={args.min_area}: IoU = {chosen.iou():.4f} | "
                f"static frames with false blobs: {chosen.static_frames_with_fp}/{chosen.static_frames}")
    logger.info(f"  {'threshold':>9} {'min area':>9} {'IoU':>8} {'false-blob frames':>18}")
    for (thr, area), v in grid.items():
        logger.info(f"  {thr:>9.1f} {area:>9.3f} {v.iou():>8.4f} {v.static_frames_with_fp:>11d}/{v.static_frames}")
    logger.info(f"  saved to {out}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
