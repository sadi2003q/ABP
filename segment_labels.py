"""
segment_labels.py — automatic moving-object labels decided per SEGMENT,
with tracking over time (plan A, 16 October 2026).

Why a new label maker
---------------------
The frozen labels (flow_labels.py, "smoothed" rule) score only 0.31 in the
published style (event pixels, frames with a moving object), while the best
recent event-only method scores about 0.77 on the same kind of EVIMO2 scenes
(normal flow + planar background + clustering + merging over time,
arXiv 2507.14500). Our method already has the planar background and the
parallax (epipolar) check. It lacks two of their ingredients:

1. Decisions per SEGMENT, not per pixel. The event pixels are split into
   about 30 small segments (K-means on position and flow, as in the paper).
   A whole segment is called "moving" when its typical (median) difference
   from the background is large. One pixel cannot decide on its own, so
   speckle noise disappears and small objects survive.

2. Tracking over time. A segment that lies where a moving object was found
   in the previous labelled frame needs only HALF the usual difference to
   count as moving ("found once stays found", a simple hysteresis).

A third change follows from these: the old rule threw away every blob
smaller than 1 % of the picture. EVIMO2 scene13 objects cover about 0.3 %,
so they could never survive. Segment decisions remove speckle themselves,
so the smallest kept blob is now 0.1 % of the picture (the default of
fill_blobs).

All settings below were FIXED BEFORE the first run (paper values or the
existing noise rules). They must not be tuned with the true masks.
True masks are read ONLY to print scores.

Small test (scene13, every 5th frame, about 15 minutes)
------------------------------------------------------
python segment_labels.py \
    --dataset-root /kaggle/working/evimo_root \
    --sequence scene13_dyn_test_01_000000 scene13_dyn_test_02_000000 scene13_dyn_test_03_000000 \
    --frame-step 5 \
    --out-dir /kaggle/working/segment_test_scene13
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))

import torch

from flow_labels import (events_of_frames, event_picture, load_raft, dense_flow, cell_flows,
                         ransac_background, cell_residuals, fit_essential, to_normalised,
                         translation_parallax, epipolar_distance, model_flow,
                         event_counts, EventScore)
from generate_pseudo_labels import Pooled, get_sample, drop_old_samples
from selfsup_signal_check import fill_blobs
from src.data.pseudo_labels import save_sequence_labels


# Frozen "smoothed" rule, same frames (every 5th), published-style score,
# mean per frame (run of 16 Oct 2026). Printed next to the new numbers.
FROZEN_REFERENCE = {
    "scene13_dyn_test_01_000000": 0.000,
    "scene13_dyn_test_02_000000": 0.100,
    "scene13_dyn_test_03_000000": 0.307,
    "scene14_dyn_test_00_000000": 0.288,
    "scene14_dyn_test_01_000000": 0.364,
    "scene14_dyn_test_02_000000": 0.327,
    "scene15_dyn_test_00_000000": 0.398,
    "scene15_dyn_test_03_000000": 0.549,
    "scene15_dyn_test_04_000000": 0.116,
    "scene15_dyn_test_06_000000": 0.718,
    "scene6_dyn_train_03_000000": 0.289,
}


# ----------------------------------------------------------------------
# 1. Split the event pixels into small segments
# ----------------------------------------------------------------------

def over_segment(cols, rows, flow_u, flow_v, n_segments, flow_weight, seed):
    """
    K-means on (x, y, flow_u, flow_v) of every event pixel.

    cols, rows     : pixel positions, e.g. 40 000 values each
    flow_u, flow_v : their flow in pixels
    n_segments     : wanted number of segments, e.g. 30
    flow_weight    : how much the flow counts compared to position, e.g. 0.5

    Returns one segment number (0 .. k-1) per pixel.
    """
    from sklearn.cluster import MiniBatchKMeans

    # Step 1: put the four columns side by side, shape (N, 4)
    features = np.stack([cols, rows, flow_u, flow_v], axis=1).astype(np.float64)

    # Step 2: give every column the same spread, so pixels and flow are comparable
    spread = features.std(axis=0)
    spread[spread == 0] = 1.0
    features = features / spread

    # Step 3: flow counts less than position (paper value 0.5)
    features[:, 2] = features[:, 2] * flow_weight
    features[:, 3] = features[:, 3] * flow_weight

    # Step 4: not more segments than the pixels allow (at least 20 pixels each)
    number_of_pixels = len(cols)
    k = min(n_segments, max(2, number_of_pixels // 20))

    # Step 5: run K-means (the mini-batch version is fast for many pixels)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096)
    segment_ids = kmeans.fit_predict(features)
    return segment_ids


def segment_medians(values, segment_ids, number_of_segments):
    """Median of `values` inside every segment, e.g. [0.4, 0.3, 5.2, ...] pixels."""
    medians = np.zeros(number_of_segments)
    for k in range(number_of_segments):
        inside = values[segment_ids == k]
        if len(inside) > 0:
            medians[k] = float(np.median(inside))
    return medians


# ----------------------------------------------------------------------
# 2. Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Segment-level moving-object labels with tracking")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--sensors", nargs="+", default=["right_camera"])
    p.add_argument("--subset", default="imo")
    p.add_argument("--split", default="train")
    p.add_argument("--sequence", nargs="+", required=True)
    p.add_argument("--frame-step", type=int, default=5, help="Label every N-th frame.")
    p.add_argument("--gap", type=int, default=3, help="Frames of events in each of the two pictures.")
    p.add_argument("--inlier-px", type=float, default=1.0)
    p.add_argument("--min-threshold", type=float, default=1.0)
    p.add_argument("--parallax-window", type=int, default=30)
    p.add_argument("--parallax-px", type=float, default=6.0)
    # ---- new settings (fixed before the first run) ----
    p.add_argument("--segments", type=int, default=30, help="Segments per frame (paper: 30).")
    p.add_argument("--flow-weight", type=float, default=0.5, help="Weight of flow vs position (paper: 0.5).")
    p.add_argument("--track-radius", type=int, default=15, help="Pixels around the previous moving object.")
    p.add_argument("--track-overlap", type=float, default=0.3,
                   help="Share of a segment that must lie near the previous object to use the lower bar.")
    p.add_argument("--track-max-gap", type=int, default=10,
                   help="Use the previous label only if it is at most this many frames old.")
    p.add_argument("--min-area", type=float, default=0.001, help="Smallest kept blob (share of picture).")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    H = args.height
    W = args.width
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    from src.data.dataset import EVIMO2Dataset
    from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic

    # Step 1: data
    ds = EVIMO2Dataset(dataset_root=args.dataset_root, sensors=tuple(args.sensors),
                       split=args.split, load_depth=False, load_mask=True,
                       subset=args.subset, sequence=tuple(args.sequence))
    refs = ds.index.references

    # Step 2: flow network
    raft, transforms = load_raft(device, "large")
    print("RAFT (large) loaded", flush=True)

    # ==================================================================
    # PASS 1: per frame, segments and their two difference scores
    # ==================================================================
    store = []                 # one entry per labelled frame
    cache = {}
    n_done = 0
    t_start = time.time()

    for i in range(len(ds) - 1):
        if int(refs[i].local_frame_index) % args.frame_step != 0:
            continue
        drop_old_samples(cache, keep_from=i - args.gap)

        # Step 3: events before and from frame i
        before = events_of_frames(ds, refs, i - args.gap, i, cache)
        after = events_of_frames(ds, refs, i, i + args.gap, cache)
        if before is None or after is None or len(before) < 500 or len(after) < 500:
            continue
        a = get_sample(ds, i, cache)

        # Step 4: true mask (scoring only, never used to decide)
        gt = np.zeros((H, W), bool)
        if a.mask is not None:
            gt = evimo2_mask_to_binary_dynamic(a.mask, get_dynamic_object_ids(a.frame_motion)).numpy()
            while gt.ndim > 2:
                gt = gt[0]
            gt = gt.astype(bool)

        # Step 5: dense flow between the two event pictures
        picture_1, count_1 = event_picture(before, H, W)
        picture_2, count_2 = event_picture(after, H, W)
        flow = dense_flow(raft, transforms, picture_1, picture_2, device)

        # Step 6: planar background motion agreed by most cells (unchanged)
        xs, ys, us, vs = cell_flows(flow, count_1 > 0)
        theta, agree = ransac_background(xs, ys, us, vs, H, W, model="planar",
                                         inlier_px=args.inlier_px, seed=i)
        if theta is None:
            continue
        background = model_flow(theta, H, W, "planar")

        # Step 7: planar noise level (same rule as the frozen method)
        residuals = cell_residuals(theta, xs, ys, us, vs, H, W, "planar")
        background_cells = residuals[residuals < args.inlier_px]
        if len(background_cells) > 0:
            spread = 1.4826 * float(np.median(background_cells))
        else:
            spread = 1.0
        planar_threshold = max(args.min_threshold, 4.0 * spread)

        # Step 8: the event pixels to segment (pixels of the first picture)
        rows, cols = np.nonzero(count_1 > 0)
        if len(rows) < 100:
            continue
        flow_u = flow[rows, cols, 0]
        flow_v = flow[rows, cols, 1]

        # Step 9: planar difference of every event pixel
        planar_difference = np.hypot(flow_u - background[rows, cols, 0],
                                     flow_v - background[rows, cols, 1])

        # Step 10: epipolar distance of every event pixel (handles parallax)
        K = np.asarray(a.camera_intrinsics, np.float64)
        dist = np.asarray(a.camera_distortion, np.float64)
        E, cell_inliers = fit_essential(xs, ys, us, vs, K, dist, args.inlier_px)
        epipolar_values = None
        epipolar_threshold = None
        parallax = 0.0
        if E is not None:
            start = to_normalised(np.stack([cols, rows], 1), K, dist)
            end = to_normalised(np.stack([cols + flow_u, rows + flow_v], 1), K, dist)
            epipolar_values = epipolar_distance(E, start, end, float(K[0, 0]))

            # noise rule: 4 x robust spread of the agreeing cells' distances,
            # at least 2 px (same as the frozen "epipolar 2.0 px" candidate)
            cell_start = to_normalised(np.stack([xs, ys], 1), K, dist)
            cell_end = to_normalised(np.stack([xs + us, ys + vs], 1), K, dist)
            cell_distance = epipolar_distance(E, cell_start, cell_end, float(K[0, 0]))[cell_inliers]
            if len(cell_distance) > 0:
                cell_spread = 1.4826 * float(np.median(cell_distance))
            else:
                cell_spread = 1.0
            epipolar_threshold = max(2.0, max(args.min_threshold, 4.0 * cell_spread))
            parallax = translation_parallax(E, cell_start, cell_end, cell_inliers, float(K[0, 0]))

        # Step 11: split the event pixels into segments
        segment_ids = over_segment(cols, rows, flow_u, flow_v,
                                   args.segments, args.flow_weight, seed=i)
        number_of_segments = int(segment_ids.max()) + 1

        # Step 12: one score per segment = its median difference
        planar_scores = segment_medians(planar_difference, segment_ids, number_of_segments)
        epipolar_scores = None
        if epipolar_values is not None:
            epipolar_scores = segment_medians(epipolar_values, segment_ids, number_of_segments)

        # Step 13: pixels with events in frame i itself (published-style scoring only)
        event_pixels = None
        if gt.any():
            events_now = events_of_frames(ds, refs, i, i + 1, cache)
            _, count_now = event_picture(events_now, H, W)
            event_pixels = count_now > 0

        # Step 14: keep what pass 2 needs
        store.append({
            "sensor": a.sensor,
            "sequence": a.sequence_name,
            "frame": int(a.local_frame_index),
            "rows": rows.astype(np.int16),
            "cols": cols.astype(np.int16),
            "segment_ids": segment_ids.astype(np.int16),
            "number_of_segments": number_of_segments,
            "planar_scores": planar_scores,
            "planar_threshold": planar_threshold,
            "epipolar_scores": epipolar_scores,
            "epipolar_threshold": epipolar_threshold,
            "parallax": parallax,
            "gt": gt,
            "event_pixels": event_pixels,
        })

        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done} frames ({(time.time() - t_start) / n_done:.1f} s/frame)", flush=True)

    # ==================================================================
    # PASS 2: per sequence in time order: choose score, track, label
    # ==================================================================
    labels_by_seq = {}
    all_pixel = {}              # sequence -> Pooled (old all-pixel score)
    event_score = {}            # sequence -> EventScore (published style)
    track_rescued = {}          # sequence -> number of segments kept only thanks to tracking
    disk_size = 2 * args.track_radius + 1
    near_structure = np.ones((disk_size, disk_size), bool)

    for sequence in args.sequence:
        # Step 15: this sequence's frames, in time order
        entries = [e for e in store if e["sequence"] == sequence]
        entries.sort(key=lambda e: e["frame"])

        previous_label = None
        previous_frame = None

        for entry in entries:
            # Step 16: smoothed parallax rule (unchanged): epipolar when the
            # camera clearly moves through the scene, planar otherwise
            nearby = [o["parallax"] for o in entries
                      if abs(o["frame"] - entry["frame"]) <= args.parallax_window]
            typical_parallax = float(np.median(nearby))
            if typical_parallax > args.parallax_px and entry["epipolar_scores"] is not None:
                scores = entry["epipolar_scores"]
                strong_bar = entry["epipolar_threshold"]
            else:
                scores = entry["planar_scores"]
                strong_bar = entry["planar_threshold"]

            # Step 17: the lower bar for segments near the previous moving object
            weak_bar = 0.5 * strong_bar

            # Step 18: area near the previous label (only if it is recent)
            near_previous = None
            if previous_label is not None and entry["frame"] - previous_frame <= args.track_max_gap:
                near_previous = ndimage.binary_dilation(previous_label, structure=near_structure)

            # Step 19: decide every segment
            rows = entry["rows"].astype(np.int64)
            cols = entry["cols"].astype(np.int64)
            segment_ids = entry["segment_ids"]
            moving_pixels = np.zeros(len(rows), bool)
            for k in range(entry["number_of_segments"]):
                in_segment = segment_ids == k
                if not in_segment.any():
                    continue
                score = scores[k]

                # Step 19a: clearly moving on its own
                if score > strong_bar:
                    moving_pixels[in_segment] = True
                    continue

                # Step 19b: moderately different AND where an object was just found
                if near_previous is not None and score > weak_bar:
                    share_near = float(near_previous[rows[in_segment], cols[in_segment]].mean())
                    if share_near >= args.track_overlap:
                        moving_pixels[in_segment] = True
                        track_rescued[sequence] = track_rescued.get(sequence, 0) + 1

            # Step 20: pixel picture of the moving segments, closed and filled
            binary = np.zeros((H, W), bool)
            binary[rows[moving_pixels], cols[moving_pixels]] = True
            label = fill_blobs(binary, close_radius=5, min_area_fraction=args.min_area)

            # Step 21: keep the label and remember it for tracking
            key = (entry["sensor"], sequence)
            labels_by_seq.setdefault(key, {})[entry["frame"]] = label
            previous_label = label
            previous_frame = entry["frame"]

            # Step 22: scores (true masks: report only)
            gt = entry["gt"]
            all_pixel.setdefault(sequence, Pooled()).update(label, gt)
            all_pixel.setdefault("ALL", Pooled()).update(label, gt)
            if entry["event_pixels"] is not None:
                counts = event_counts(label, gt, entry["event_pixels"])
                event_score.setdefault(sequence, EventScore()).add(*counts)
                event_score.setdefault("ALL", EventScore()).add(*counts)

    # Step 23: save the labels (same format as flow_labels.py, usable for training)
    for (sensor, sequence), labels in labels_by_seq.items():
        save_sequence_labels(out, sensor, sequence, labels)

    # Step 24: result table
    print("\n" + "=" * 100)
    print("SEGMENT LABELS (segments + tracking) | PUBLISHED-STYLE SCORE = event pixels, frames with a moving object")
    print("Frozen 'smoothed' rule on the same frames is shown for comparison. True masks: report only.")
    print("=" * 100)
    print(f"{'sequence':30s} {'NEW per-frame':>14s} {'NEW pooled':>11s} {'FROZEN per-frame':>17s} "
          f"{'NEW all-pixel':>14s} {'false-blob frames':>18s} {'tracked segs':>13s}")
    report = {}
    per_sequence_new = []
    per_sequence_old = []
    for sequence in list(args.sequence) + ["ALL"]:
        score = event_score.get(sequence)
        pooled = all_pixel.get(sequence)
        if score is None or pooled is None:
            print(f"{sequence:30s} (no scored frames)")
            continue
        old = FROZEN_REFERENCE.get(sequence)
        old_text = f"{old:.3f}" if old is not None else "-"
        blob_text = f"{pooled.static_frames_with_fp}/{pooled.static_frames}"
        print(f"{sequence:30s} {score.mean_per_frame():14.3f} {score.pooled():11.3f} {old_text:>17s} "
              f"{pooled.iou():14.3f} {blob_text:>18s} {track_rescued.get(sequence, 0):13d}")
        report[sequence] = {
            "new_event_mean_per_frame": score.mean_per_frame(),
            "new_event_pooled": score.pooled(),
            "new_all_pixel_iou": pooled.iou(),
            "frozen_event_mean_per_frame": old,
            "frames_scored": score.frames(),
        }
        if sequence != "ALL":
            per_sequence_new.append(score.mean_per_frame())
            if old is not None:
                per_sequence_old.append(old)

    # Step 25: the number the decision is based on (fixed before running)
    if per_sequence_new:
        new_average = float(np.mean(per_sequence_new))
        print(f"\nAverage over {len(per_sequence_new)} sequences (event pixels, mean per frame):")
        print(f"  NEW segment labels : {new_average:.3f}")
        if per_sequence_old:
            print(f"  FROZEN labels      : {float(np.mean(per_sequence_old)):.3f}")
        print("  Stop rule (fixed before running): continue plan A only if NEW >= 0.50 on scene13.")
        report["average_new"] = new_average

    with open(out / "segment_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nLabels saved: {out}")


if __name__ == "__main__":
    main()
