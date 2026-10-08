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

logger = logging.getLogger("generate_pseudo_labels")

GRID_THRESHOLDS = (0.2, 0.3, 0.4)
GRID_MIN_AREAS = (0.001, 0.003, 0.01)


def pseudo_score(xy, t, t0, t1, K, dist, R, trans, depth, H=None, W=None, mode="full"):
    """mode="full": rotation + translation + depth (stage A, motion capture).
    mode="rotation": rotation only, no depth needed (stage B0, inertial sensor)."""
    if depth is not None:
        H, W = depth.shape
    flow = ego_flow(H, W, K, dist, R, trans, depth=depth, mode=mode)
    xw, yw, alpha = compensate(xy, t, t0, t1, flow)
    return scores_from_warped(xw, yw, alpha, H, W)["contrast"]


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
    p = argparse.ArgumentParser(description="Stage A: pseudo-labels from motion compensation")
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
    g.add_argument("--geometry", choices=["mocap", "imu_rotation"], default="mocap",
                   help="mocap: motion-capture depth + camera pose (stage A). "
                        "imu_rotation: camera rotation from the camera's own gyroscope only, "
                        "no depth, no translation, no motion capture (stage B0).")
    g.add_argument("--imu-axes", type=float, nargs=9, default=None,
                   help="Fixed 3x3 inertial-to-camera axis matrix (row-major). "
                        "Default: found automatically from event sharpness.")
    g.add_argument("--calib-frames", type=int, default=40,
                   help="Frames used to find the inertial-to-camera axis mapping.")
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--width", type=int, default=640)
    args = p.parse_args()
    use_imu = args.geometry == "imu_rotation"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "generate.log")], force=True)

    from src.data.dataset import EVIMO2Dataset
    from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic

    ds = EVIMO2Dataset(dataset_root=args.dataset_root, sensors=tuple(args.sensors),
                       split=args.split, load_depth=not use_imu, load_mask=True,
                       subset=args.subset, sequence=args.sequence)
    refs = ds.index.references

    imu_axes, calib_report, mocap_check = {}, {}, {}
    if use_imu:
        imu_axes, calib_report = _calibrate_imu_axes(ds, refs, args)

    grid = {(thr, a): Pooled() for thr in GRID_THRESHOLDS for a in GRID_MIN_AREAS}
    chosen = Pooled()
    labels_by_seq: dict[tuple, dict[int, np.ndarray]] = {}
    n_written, n_skipped, t_start = 0, 0, time.time()

    for i in range(len(ds) - 1):
        if refs[i].sequence_id != refs[i + 1].sequence_id:
            continue
        a, b = ds[i], ds[i + 1]
        if use_imu:
            theta = integrate_gyro(a.imu.timestamps, a.imu.angular_velocity, a.timestamp, b.timestamp)
            if theta is None:
                n_skipped += 1
                continue
            R = rotation_from_imu(theta, imu_axes[a.sensor])
            trans, depth, mode = np.zeros(3), None, "rotation"
            H, W = args.height, args.width
            if a.camera_motion.pose_available and b.camera_motion.pose_available:
                # report only: how close the gyroscope rotation is to motion capture
                R_mc, _ = relative_pose(a.camera_motion, b.camera_motion)
                mocap_check.setdefault("error_deg", []).append(rotation_angle_deg(R.T @ R_mc))
                mocap_check.setdefault("mocap_deg", []).append(rotation_angle_deg(R_mc))
        else:
            if a.depth is None or not (a.camera_motion.pose_available and b.camera_motion.pose_available):
                n_skipped += 1
                continue
            depth = _depth_in_metres(a.depth)
            R, trans = relative_pose(a.camera_motion, b.camera_motion)
            mode = "full"
            H, W = depth.shape
        xy = np.asarray(a.events_xy)
        t = np.asarray(a.events_t, np.float64)
        if len(t) == 0:
            label = np.zeros((H, W), bool)
            score = np.zeros((H, W))
        else:
            score = pseudo_score(xy, t, a.timestamp, b.timestamp,
                                 np.asarray(a.camera_intrinsics, np.float64),
                                 np.asarray(a.camera_distortion, np.float64), R, trans, depth,
                                 H=H, W=W, mode=mode)
            label = score_to_label(score, args.threshold, args.min_area)

        labels_by_seq.setdefault((a.sensor, a.sequence_name), {})[int(a.local_frame_index)] = label
        n_written += 1

        if a.mask is not None:        # quality report only
            gt = evimo2_mask_to_binary_dynamic(a.mask, get_dynamic_object_ids(a.frame_motion)).numpy()
            while gt.ndim > 2:
                gt = gt[0]
            gt = gt.astype(bool)
            chosen.update(label, gt)
            for (thr, area), pooled in grid.items():
                pooled.update(score_to_label(score, thr, area) if len(t) else label, gt)

        if n_written % 25 == 0:
            rate = (time.time() - t_start) / n_written
            logger.info(f"  labelled {n_written} frames ({rate:.1f} s/frame)")

    for (sensor, seq), labels in labels_by_seq.items():
        save_sequence_labels(out, sensor, seq, labels)

    skip_reason = "no gyroscope samples" if use_imu else "no depth/pose"
    logger.info(f"Pseudo-labels written: {n_written} frames | skipped ({skip_reason}): {n_skipped}")
    if mocap_check.get("error_deg"):
        logger.info(f"Gyroscope vs motion-capture rotation (report only): median error "
                    f"{np.median(mocap_check['error_deg']):.3f} deg per frame, while the camera "
                    f"rotates {np.median(mocap_check['mocap_deg']):.3f} deg per frame (median)")
    report = {
        "settings": {"threshold": args.threshold, "min_area": args.min_area, "geometry": ("gyroscope rotation only, axes found from event sharpness (stage B0)" if use_imu
                                 else "motion-capture depth + pose (stage A)")},
        "imu_calibration": calib_report,
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
