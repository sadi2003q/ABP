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

Longer time windows (--window-frames, --chain, --frame-step)
-----------------------------------------------------------
By default each label uses the events of ONE frame window (~17 ms at
60 Hz). In many EVIMO2 recordings the camera moves less than 1 pixel in
that time, so motion compensation has nothing to undo and moving
objects move too little to stand out. --window-frames K (odd) uses the
events of K frames centred on the labelled frame, carried to the time
of that frame (the time of its true mask). K = 1 is exactly the old
behaviour. Contrast-maximisation segmentation on EVIMO2 is commonly run
on ~50 ms slices (Aoki et al., 2025, arXiv 2504.18447), i.e. K = 3.

--chain (with --window-frames > 1, --geometry events): one flow formula
for a long window fits badly (K = 9: 1.85 px error vs 0.5 px for one
frame). With --chain the background flow is fitted for EACH frame on
its own, and the events of the other frames are carried to the
labelled frame's time through those per-frame flows, one frame at a
time. Each frame's flow is fitted once and reused by later windows.

--frame-step N labels only every N-th frame (quick checks). Do not use
it before refine_pseudo_labels.py: refinement needs neighbouring labels.

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


def score_from_flow(xy, t, t0, t1, flow, t_ref=None):
    """Contrast score map after undoing the background displacement `flow` (H, W, 2).
    `flow` is the displacement over the whole window [t0, t1].
    t_ref = None: events are warped to t0 (old behaviour).
    t_ref given:  events are warped to t_ref (e.g. the labelled frame in the
                  middle of a multi-frame window)."""
    H, W = flow.shape[:2]
    if t_ref is None:
        xw, yw, alpha = compensate(xy, t, t0, t1, flow)
    else:
        xw, yw, alpha = compensate_to_time(xy, t, t0, t1, t_ref, flow)
    return scores_from_warped(xw, yw, alpha, H, W)["contrast"]


def compensate_to_time(xy, t, t0, t1, t_ref, flow):
    """Warp events to the time t_ref (anywhere inside [t0, t1]).
    With t_ref = t0 this gives exactly the same result as compensate()."""
    # Step 1: image size, e.g. H = 480, W = 640
    H, W = flow.shape[:2]

    # Step 2: length of the whole window in seconds, e.g. 0.083
    duration = max(t1 - t0, 1e-9)

    # Step 3: signed fraction of the window between each event and t_ref
    #         negative = event before t_ref, positive = after it
    #         e.g. t_ref in the middle -> values between -0.5 and +0.5
    alpha = (t - t_ref) / duration
    lowest = (t0 - t_ref) / duration
    highest = (t1 - t_ref) / duration
    alpha = np.clip(alpha, lowest, highest)

    # Step 4: event pixel positions
    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)

    # Step 5: background displacement at each event's pixel
    x_pixel = np.clip(np.round(x).astype(int), 0, W - 1)
    y_pixel = np.clip(np.round(y).astype(int), 0, H - 1)
    displacement = flow[y_pixel, x_pixel]

    # Step 6: move each event back (or forward) to where it was at t_ref
    x_warped = x - alpha * displacement[:, 0]
    y_warped = y - alpha * displacement[:, 1]
    return x_warped, y_warped, alpha


def get_sample(ds, index, cache):
    """Load frame `index` from the dataset, or reuse it if already loaded."""
    # Step 1: load it only the first time it is needed
    if index not in cache:
        cache[index] = ds[index]
    return cache[index]


def drop_old_samples(cache, keep_from):
    """Forget loaded frames with an index below `keep_from` (saves memory)."""
    # Step 1: find the old indices
    old_indices = [index for index in cache if index < keep_from]

    # Step 2: remove them
    for index in old_indices:
        del cache[index]


def gather_window(ds, refs, i, half, cache):
    """
    Events of the frames i-half ... i+half (all in the same sequence).

    Returns None when the window would leave the sequence (first/last
    frames of a sequence), otherwise a dictionary with:
      xy, t          all events of the window
      t_start, t_end start and end time of the window
      first, end     the first frame and the frame right after the last one
                     (used only for the motion-capture comparison in the report)
    With half = 0 this is exactly frame i's own window (old behaviour).
    """
    # Step 1: frame indices, e.g. i = 10, half = 2 -> frames 8, 9, 10, 11, 12
    first_index = i - half
    last_index = i + half

    # Step 2: we also need the frame after the last one (its time = window end)
    after_last_index = last_index + 1

    # Step 3: the window must stay inside the dataset
    if first_index < 0 or after_last_index >= len(refs):
        return None

    # Step 4: every frame must come from the same sequence as frame i
    sequence_id = refs[i].sequence_id
    for j in range(first_index, after_last_index + 1):
        if refs[j].sequence_id != sequence_id:
            return None

    # Step 5: collect the events of every frame in the window
    xy_parts = []
    t_parts = []
    for j in range(first_index, last_index + 1):
        sample = get_sample(ds, j, cache)
        xy_parts.append(np.asarray(sample.events_xy).reshape(-1, 2))
        t_parts.append(np.asarray(sample.events_t, np.float64).reshape(-1))

    # Step 6: join them into one list of events
    xy = np.concatenate(xy_parts, axis=0)
    t = np.concatenate(t_parts, axis=0)

    # Step 7: start and end time of the window
    first_sample = get_sample(ds, first_index, cache)
    end_sample = get_sample(ds, after_last_index, cache)

    # Step 8: start time of every frame, plus the end of the last one
    #         e.g. 5 frames -> 6 times
    frame_times = []
    for j in range(first_index, after_last_index + 1):
        frame_times.append(get_sample(ds, j, cache).timestamp)

    return {
        "xy": xy,
        "t": t,
        "xy_parts": xy_parts,
        "t_parts": t_parts,
        "frame_times": frame_times,
        "first_index": first_index,
        "t_start": first_sample.timestamp,
        "t_end": end_sample.timestamp,
        "first": first_sample,
        "end": end_sample,
    }


def pseudo_score(xy, t, t0, t1, K, dist, R, trans, depth, H=None, W=None, mode="full"):
    """mode="full": rotation + translation + depth (stage A, motion capture).
    mode="rotation": rotation only, no depth needed (stage B0, inertial sensor)."""
    if depth is not None:
        H, W = depth.shape
    flow = ego_flow(H, W, K, dist, R, trans, depth=depth, mode=mode)
    return score_from_flow(xy, t, t0, t1, flow)


def score_to_label(score, threshold, min_area_fraction, close_radius=5):
    return fill_blobs(score > threshold, close_radius=close_radius, min_area_fraction=min_area_fraction)


def flow_at_events(flow, x, y):
    """Background displacement (pixels over one frame) at each event position."""
    # Step 1: image size
    H, W = flow.shape[:2]

    # Step 2: nearest pixel of each event, kept inside the image
    x_pixel = np.clip(np.round(x).astype(int), 0, W - 1)
    y_pixel = np.clip(np.round(y).astype(int), 0, H - 1)

    # Step 3: read the flow there, shape (number of events, 2)
    return flow[y_pixel, x_pixel]


def warp_chained(xy_parts, t_parts, frame_flows, frame_times, ref_position):
    """
    Carry the events of every frame in the window to the start time of the
    labelled frame, using each frame's OWN background flow.

    xy_parts[k], t_parts[k] : events of frame k of the window
    frame_flows[k]          : background displacement over frame k (H, W, 2)
    frame_times[k]          : start time of frame k (one extra entry = end of the last frame)
    ref_position            : position of the labelled frame in the window, e.g. 2 of 0..4

    With one frame (ref_position = 0) this is exactly compensate().
    """
    x_parts = []
    y_parts = []
    for k in range(len(xy_parts)):
        # Step 1: event positions of frame k
        x = xy_parts[k][:, 0].astype(np.float64)
        y = xy_parts[k][:, 1].astype(np.float64)

        # Step 2: how far through frame k each event happened (0 = start, 1 = end)
        frame_length = max(frame_times[k + 1] - frame_times[k], 1e-9)
        fraction = np.clip((t_parts[k] - frame_times[k]) / frame_length, 0.0, 1.0)

        if k >= ref_position:
            # Step 3a: frame k is the labelled frame or later:
            #          move each event back to the START of frame k
            displacement = flow_at_events(frame_flows[k], x, y)
            x = x - fraction * displacement[:, 0]
            y = y - fraction * displacement[:, 1]

            # Step 4a: then back one whole frame at a time, down to the labelled frame
            for m in range(k - 1, ref_position - 1, -1):
                displacement = flow_at_events(frame_flows[m], x, y)
                x = x - displacement[:, 0]
                y = y - displacement[:, 1]
        else:
            # Step 3b: frame k is before the labelled frame:
            #          move each event forward to the END of frame k
            displacement = flow_at_events(frame_flows[k], x, y)
            x = x + (1.0 - fraction) * displacement[:, 0]
            y = y + (1.0 - fraction) * displacement[:, 1]

            # Step 4b: then forward one whole frame at a time, up to the labelled frame
            for m in range(k + 1, ref_position):
                displacement = flow_at_events(frame_flows[m], x, y)
                x = x + displacement[:, 0]
                y = y + displacement[:, 1]

        x_parts.append(x)
        y_parts.append(y)

    # Step 5: join all frames back into one list
    return np.concatenate(x_parts), np.concatenate(y_parts)


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
    p.add_argument("--window-frames", type=int, default=1,
                   help="Odd number of frames whose events are used for each label, centred on the "
                        "labelled frame (--geometry events / none only). 1 = old behaviour.")
    p.add_argument("--chain", action="store_true",
                   help="With --window-frames > 1 and --geometry events: fit the background flow of "
                        "each frame separately and chain them, instead of one flow for the whole window.")
    p.add_argument("--frame-step", type=int, default=1,
                   help="Label only every N-th frame (quick checks only; refinement needs every frame).")
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
    if args.window_frames < 1 or args.window_frames % 2 == 0:
        p.error("--window-frames must be an odd number (1, 3, 5, ...)")
    if args.window_frames > 1 and geometry not in ("events", "none"):
        p.error("--window-frames > 1 is only supported with --geometry events or none")
    if args.chain and geometry != "events":
        p.error("--chain is only supported with --geometry events")
    if args.frame_step < 1:
        p.error("--frame-step must be 1 or more")
    half = args.window_frames // 2      # e.g. 5 frames -> 2 on each side

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

    cache = {}                                       # loaded frames, reused by neighbouring windows
    flow_cache = {}                                  # --chain: fitted flow of each frame, by dataset index
    fit_stats["background_px"] = []                  # events: fitted background motion over the window
    logger.info(f"Window: {args.window_frames} frame(s) per label"
                f"{' (per-frame flows, chained)' if args.chain else ''} | labelling every "
                f"{args.frame_step} frame(s)")

    for i in range(len(ds) - 1):
        if refs[i].sequence_id != refs[i + 1].sequence_id:
            continue
        if args.limit_frames and n_written >= args.limit_frames:
            break
        if int(refs[i].local_frame_index) % args.frame_step != 0:
            continue

        # forget frames that no later window can use
        drop_old_samples(cache, keep_from=i - half)
        drop_old_samples(flow_cache, keep_from=i - half)
        chained_score = None

        a = get_sample(ds, i, cache)
        b = get_sample(ds, i + 1, cache)
        K = np.asarray(a.camera_intrinsics, np.float64)
        dist = np.asarray(a.camera_distortion, np.float64)
        xy = np.asarray(a.events_xy)
        t = np.asarray(a.events_t, np.float64)

        # events / none: events of the (possibly multi-frame) window around frame i
        t0, t1, t_ref = a.timestamp, b.timestamp, None
        first, end = a, b
        if geometry in ("events", "none"):
            window = gather_window(ds, refs, i, half, cache)
            if window is None:
                n_skipped += 1               # too close to the start/end of the sequence
                continue
            xy, t = window["xy"], window["t"]
            t0, t1 = window["t_start"], window["t_end"]
            first, end = window["first"], window["end"]
            if half > 0:
                t_ref = a.timestamp          # warp to the labelled frame's time

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
        elif geometry == "events" and args.chain:
            H, W = args.height, args.width
            key = (a.sensor, a.sequence_name)

            # Step 1: fitted flow of every frame in the window (each frame fitted only once)
            frame_flows = []
            for k in range(len(window["xy_parts"])):
                j = window["first_index"] + k            # dataset index of frame k
                if j not in flow_cache:
                    theta, info = fit_background_flow(
                        window["xy_parts"][k], window["t_parts"][k],
                        window["frame_times"][k], window["frame_times"][k + 1], H, W,
                        model=args.flow_model, init=prev_theta.get(key),
                        max_events=args.flow_max_events, seed=j)
                    if info["fitted"]:
                        prev_theta[key] = theta
                        fit_stats["gain"].append(info["gain"])
                        fit_stats["at_bound"] += int(info["at_bound"])
                    else:
                        fit_stats["not_fitted"] += 1
                    flow_cache[j] = flow_field(theta, H, W, args.flow_model)
                frame_flows.append(flow_cache[j])

            # Step 2: background motion over the whole window (report, no ground truth)
            total_flow = np.sum(np.stack(frame_flows), axis=0)
            fit_stats["background_px"].append(float(np.median(np.linalg.norm(total_flow, axis=2))))

            # Step 3: the labelled frame's own flow, for the motion-capture comparison below
            flow = frame_flows[half]
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

            # Step 4: carry all events to the labelled frame's time and score them
            if len(t) > 0:
                x_warped, y_warped = warp_chained(window["xy_parts"], window["t_parts"],
                                                  frame_flows, window["frame_times"], half)
                # signed fraction of the whole window between each event and the labelled frame
                alpha = (t - a.timestamp) / max(t1 - t0, 1e-9)
                chained_score = scores_from_warped(x_warped, y_warped, alpha, H, W)["contrast"]
        elif geometry == "events":
            H, W = args.height, args.width
            key = (a.sensor, a.sequence_name)
            theta, info = fit_background_flow(xy, t, t0, t1, H, W,
                                              model=args.flow_model, init=prev_theta.get(key),
                                              max_events=args.flow_max_events, seed=i)
            if info["fitted"]:
                prev_theta[key] = theta
                fit_stats["gain"].append(info["gain"])
                fit_stats["at_bound"] += int(info["at_bound"])
            else:
                fit_stats["not_fitted"] += 1
            flow = flow_field(theta, H, W, args.flow_model)
            fit_stats["background_px"].append(float(np.median(np.linalg.norm(flow, axis=2))))
            # report only: compare with the motion-capture background flow on static pixels
            # (over the same time span: first frame of the window -> frame after the last)
            if (first.depth is not None and first.camera_motion.pose_available
                    and end.camera_motion.pose_available
                    and np.asarray(first.depth).shape[-2:] == (H, W)):
                depth = _depth_in_metres(first.depth)
                while depth.ndim > 2:
                    depth = depth[0]
                R_mc, t_mc = relative_pose(first.camera_motion, end.camera_motion)
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
            if chained_score is not None:
                score = chained_score
            else:
                score = score_from_flow(xy, t, t0, t1, flow, t_ref=t_ref)
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

    skip_reason = {"imu_rotation": "no gyroscope samples", "mocap": "no depth/pose"}.get(
        geometry, "window reaches past the start/end of a sequence")
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
        if fit_stats["background_px"]:
            logger.info(f"Fitted background motion over the window (no ground truth): median "
                        f"{np.median(fit_stats['background_px']):.2f} px")
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
                     "limit_frames": args.limit_frames,
                     "window_frames": args.window_frames, "frame_step": args.frame_step,
                     "chain": args.chain},
        "imu_calibration": calib_report,
        "event_flow": ({
            "median_sharpness_gain": float(np.median(fit_stats["gain"])) if fit_stats["gain"] else None,
            "windows_with_parameter_at_limit": fit_stats["at_bound"],
            "windows_too_sparse": fit_stats["not_fitted"],
            "median_background_motion_px": (float(np.median(fit_stats["background_px"]))
                                            if fit_stats["background_px"] else None),
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
