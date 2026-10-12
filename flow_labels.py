"""
flow_labels.py — SMALL TEST of dense-flow pseudo-labels (stage B4).

Why
---
B1 finds moving objects by their blur, which needs fast relative motion.
DINOv2 appearance on event pictures splits by wall texture, not objects.
Methods that work on EVIMO2 without labels (Arja et al. 2024, 79 % IoU)
all use DENSE optical flow: how every pixel moves. This test checks that cue.

How (per labelled frame i)
--------------------------
1. Two event pictures: events of the `gap` frames BEFORE frame i, and of
   the `gap` frames FROM frame i on (e.g. gap 3 = about 50 ms each).
2. Dense flow between the two pictures with RAFT (torchvision, trained on
   synthetic pictures with known flow; it has never seen EVIMO2 or any of
   its masks).
3. Background motion: an affine motion fitted with RANSAC over image CELLS
   (16 x 16 pixels). Every cell with events counts ONCE, however many
   events it has, so a large textured moving object cannot outvote the
   background by producing more events (the failure of B1).
4. Moving pixels: event pixels whose flow differs from the background
   motion by more than a threshold; several thresholds are scored side by
   side. Then close gaps, fill holes and drop small blobs. A frame with no
   blob left gets an EMPTY label ("no moving object").

The true masks are used ONLY to score the labels.
Needs Internet ON in Kaggle the first time (downloads RAFT weights, ~20 MB).

Usage (small test)
------------------
python flow_labels.py \
    --dataset-root /kaggle/working/evimo_root \
    --sequence scene14_dyn_test_01_000000 scene15_dyn_test_00_000000 scene6_dyn_train_03_000000 \
    --frame-step 10 --out-dir /kaggle/working/flow_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))

import torch

from generate_pseudo_labels import Pooled, get_sample, drop_old_samples
from selfsup_signal_check import fill_blobs
from src.data.event_flow import _basis
from src.data.pseudo_labels import save_sequence_labels

THRESHOLDS = (1.0, 1.5, 2.0, 3.0)      # pixels of flow difference over the picture gap


# ----------------------------------------------------------------------
# 1. Event pictures
# ----------------------------------------------------------------------

def events_of_frames(ds, refs, start, stop, cache):
    """All events of frames start .. stop-1 (one sequence). None if outside the sequence."""
    # Step 1: inside the dataset?
    if start < 0 or stop > len(refs):
        return None

    # Step 2: all frames from the same sequence?
    sequence_id = refs[start].sequence_id
    for j in range(start, stop):
        if refs[j].sequence_id != sequence_id:
            return None

    # Step 3: join their events
    parts = []
    for j in range(start, stop):
        sample = get_sample(ds, j, cache)
        parts.append(np.asarray(sample.events_xy).reshape(-1, 2))
    return np.concatenate(parts, axis=0)


def event_picture(xy, H, W):
    """
    Events -> picture (H, W) with values 0..1, plus the raw count per pixel.
    Log scale so busy edges do not drown quiet ones; slight blur so RAFT sees shapes.
    """
    # Step 1: whole-pixel positions
    x = np.clip(np.round(xy[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(xy[:, 1]).astype(int), 0, H - 1)

    # Step 2: events per pixel
    count = np.zeros((H, W), np.float64)
    np.add.at(count, (y, x), 1.0)

    # Step 3: log scale + light blur, then 0..1
    picture = ndimage.gaussian_filter(np.log1p(count), 1.0)
    picture = picture / max(picture.max(), 1e-9)
    return picture.astype(np.float32), count


# ----------------------------------------------------------------------
# 2. Dense flow with RAFT
# ----------------------------------------------------------------------

def load_raft(device, size):
    """RAFT from torchvision with its pretrained weights (synthetic training data)."""
    from torchvision.models.optical_flow import raft_large, raft_small
    from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
    if size == "small":
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights)
    else:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights)
    model.eval()
    return model.to(device), weights.transforms()


@torch.no_grad()
def dense_flow(model, transforms, picture_1, picture_2, device):
    """Flow (H, W, 2) in pixels: where each pixel of picture_1 moved to in picture_2."""
    # Step 1: grey pictures -> 3 identical colour channels, shape (1, 3, H, W)
    image_1 = torch.from_numpy(picture_1)[None, None].repeat(1, 3, 1, 1)
    image_2 = torch.from_numpy(picture_2)[None, None].repeat(1, 3, 1, 1)

    # Step 2: the scaling RAFT was trained with
    image_1, image_2 = transforms(image_1, image_2)

    # Step 3: RAFT refines its estimate several times; keep the last one
    predictions = model(image_1.to(device), image_2.to(device))
    flow = predictions[-1][0]                     # (2, H, W)
    return flow.permute(1, 2, 0).cpu().numpy()


# ----------------------------------------------------------------------
# 3. Background motion: RANSAC over cells
# ----------------------------------------------------------------------

def cell_flows(flow, has_events, cell=16, min_pixels=8):
    """
    One flow value per cell (median over the cell's event pixels).
    Returns centre x, centre y, flow u, flow v for cells with enough event pixels.
    """
    H, W = has_events.shape
    xs, ys, us, vs = [], [], [], []
    for top in range(0, H - cell + 1, cell):
        for left in range(0, W - cell + 1, cell):
            # Step 1: event pixels in this cell
            inside = has_events[top:top + cell, left:left + cell]
            if inside.sum() < min_pixels:
                continue

            # Step 2: median flow over them (robust to a few bad pixels)
            block = flow[top:top + cell, left:left + cell][inside]
            xs.append(left + cell / 2.0)
            ys.append(top + cell / 2.0)
            us.append(float(np.median(block[:, 0])))
            vs.append(float(np.median(block[:, 1])))
    return np.array(xs), np.array(ys), np.array(us), np.array(vs)


def solve_motion(xs, ys, us, vs, H, W, model):
    """Least-squares motion parameters from cell flows."""
    Bu, Bv = _basis(xs, ys, H, W, model)
    matrix = np.concatenate([Bu, Bv], axis=0)
    target = np.concatenate([us, vs])
    theta, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return theta


def cell_residuals(theta, xs, ys, us, vs, H, W, model):
    """Distance (pixels) between each cell's flow and the motion model."""
    Bu, Bv = _basis(xs, ys, H, W, model)
    return np.hypot(us - Bu @ theta, vs - Bv @ theta)


def ransac_background(xs, ys, us, vs, H, W, model="affine", iterations=300,
                      inlier_px=1.0, seed=0):
    """
    The motion that agrees with the MOST CELLS (not the most events).
    Returns (theta, share of cells that agree) or (None, 0) if too few cells.
    """
    n_params = {"translation": 2, "affine": 6, "planar": 8}[model]
    n_needed = n_params // 2                     # each cell gives 2 equations
    n_cells = len(xs)
    if n_cells < max(10, 3 * n_needed):
        return None, 0.0

    rng = np.random.default_rng(seed)
    best_inliers = None
    for step in range(iterations):
        # Step 1: a random minimal set of cells
        pick = rng.choice(n_cells, size=n_needed, replace=False)
        theta = solve_motion(xs[pick], ys[pick], us[pick], vs[pick], H, W, model)

        # Step 2: how many cells agree with it?
        inliers = cell_residuals(theta, xs, ys, us, vs, H, W, model) < inlier_px
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    # Step 3: refit on all agreeing cells
    theta = solve_motion(xs[best_inliers], ys[best_inliers], us[best_inliers],
                         vs[best_inliers], H, W, model)
    return theta, float(best_inliers.mean())


def model_flow(theta, H, W, model):
    """Motion model -> flow (H, W, 2) for every pixel."""
    v, u = np.mgrid[0:H, 0:W]
    Bu, Bv = _basis(u.ravel(), v.ravel(), H, W, model)
    return np.stack([Bu @ theta, Bv @ theta], 1).reshape(H, W, 2)


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Small test: dense-flow (RAFT) pseudo-labels")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--sensors", nargs="+", default=["right_camera"])
    p.add_argument("--subset", default="imo")
    p.add_argument("--split", default="train")
    p.add_argument("--sequence", nargs="+", required=True)
    p.add_argument("--frame-step", type=int, default=10, help="Label every N-th frame.")
    p.add_argument("--gap", type=int, default=3, help="Frames of events in each of the two pictures.")
    p.add_argument("--background-model", choices=["translation", "affine", "planar"], default="affine")
    p.add_argument("--inlier-px", type=float, default=1.0, help="RANSAC: cell agrees within this many pixels.")
    p.add_argument("--threshold", type=float, default=1.5,
                   help="Flow difference (pixels) for the WRITTEN labels; all THRESHOLDS are reported.")
    p.add_argument("--min-area", type=float, default=0.01, help="Smallest blob kept (fraction of image).")
    p.add_argument("--raft", choices=["large", "small"], default="large")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    H, W = args.height, args.width
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    from src.data.dataset import EVIMO2Dataset
    from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic

    # Step 1: data (no depth needed)
    ds = EVIMO2Dataset(dataset_root=args.dataset_root, sensors=tuple(args.sensors),
                       split=args.split, load_depth=False, load_mask=True,
                       subset=args.subset, sequence=tuple(args.sequence))
    refs = ds.index.references

    # Step 2: flow network
    raft, transforms = load_raft(device, args.raft)
    print(f"RAFT ({args.raft}) loaded", flush=True)

    scores = {}                  # (sequence, threshold) -> Pooled
    labels_by_seq = {}
    agree_shares = {}            # sequence -> list of RANSAC agreement shares
    examples = []
    cache = {}
    n_done = 0
    t_start = time.time()
    dilate = np.ones((5, 5), bool)

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
        sequence = a.sequence_name

        # Step 4: true mask (scoring only)
        gt = None
        if a.mask is not None:
            gt = evimo2_mask_to_binary_dynamic(a.mask, get_dynamic_object_ids(a.frame_motion)).numpy()
            while gt.ndim > 2:
                gt = gt[0]
            gt = gt.astype(bool)

        # Step 5: two pictures and the dense flow between them
        picture_1, count_1 = event_picture(before, H, W)
        picture_2, count_2 = event_picture(after, H, W)
        flow = dense_flow(raft, transforms, picture_1, picture_2, device)

        # Step 6: pixels where flow can be trusted (events nearby)
        has_events = ndimage.binary_dilation(count_1 > 0, structure=dilate)

        # Step 7: background motion agreed by most cells
        xs, ys, us, vs = cell_flows(flow, count_1 > 0)
        theta, agree = ransac_background(xs, ys, us, vs, H, W, model=args.background_model,
                                         inlier_px=args.inlier_px, seed=i)
        if theta is None:
            continue
        agree_shares.setdefault(sequence, []).append(agree)

        # Step 8: how different is each pixel's flow from the background?
        background = model_flow(theta, H, W, args.background_model)
        difference = np.linalg.norm(flow - background, axis=2)

        # Step 9: one label per threshold
        frame_labels = {}
        for threshold in THRESHOLDS:
            moving = (difference > threshold) & has_events
            frame_labels[threshold] = fill_blobs(moving, close_radius=5, min_area_fraction=args.min_area)
        if gt is not None:
            for threshold, label in frame_labels.items():
                scores.setdefault((sequence, threshold), Pooled()).update(label, gt)

        # Step 10: keep the label of the chosen threshold
        chosen = frame_labels.get(args.threshold)
        if chosen is None:
            moving = (difference > args.threshold) & has_events
            chosen = fill_blobs(moving, close_radius=5, min_area_fraction=args.min_area)
        labels_by_seq.setdefault((a.sensor, sequence), {})[int(a.local_frame_index)] = chosen

        # Step 11: frames for the check picture
        if gt is not None and gt.any() and len(examples) < 24:
            examples.append((sequence, int(a.local_frame_index), picture_1, gt,
                             np.linalg.norm(flow, axis=2), np.clip(difference, 0, 5), chosen))

        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done} frames ({(time.time() - t_start) / n_done:.1f} s/frame)", flush=True)

    # Step 12: save the written labels
    for (sensor, sequence), labels in labels_by_seq.items():
        save_sequence_labels(out, sensor, sequence, labels)

    # Step 13: result table
    print("\n" + "=" * 96)
    print("SMALL TEST: dense-flow labels against the TRUE masks (report only)")
    print("Reference, B1 (one motion + dither) on the same sequences: 14_01 0.278 | 15_00 0.003 | 6_03 0.016")
    print("=" * 96)
    report = {}
    for sequence in args.sequence:
        shares = agree_shares.get(sequence, [])
        median_share = float(np.median(shares)) if shares else float("nan")
        print(f"--- {sequence}   (cells agreeing with the background: median {median_share:.2f}) ---")
        for threshold in THRESHOLDS:
            pooled = scores.get((sequence, threshold))
            if pooled is None:
                continue
            precision = pooled.tp / max(1, pooled.tp + pooled.fp)
            recall = pooled.tp / max(1, pooled.tp + pooled.fn)
            print(f"  flow difference > {threshold:.1f} px   IoU {pooled.iou():.3f}   precision {precision:.3f}   "
                  f"recall {recall:.3f}   false-blob frames {pooled.static_frames_with_fp}/{pooled.static_frames}")
            report[f"{sequence} | thr={threshold}"] = {"iou": pooled.iou(), "precision": precision,
                                                        "recall": recall}
    with open(out / "flow_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Step 14: check picture
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picked = []
    for sequence in args.sequence:
        own = [e for e in examples if e[0] == sequence]
        picked += own[:: max(1, len(own) // 2)][:2]
    if picked:
        titles = ["events (before)", "true moving mask", "flow size (px)",
                  "difference from background (0-5 px)", f"label (> {args.threshold} px)"]
        figure, axes = plt.subplots(len(picked), 5, figsize=(20, 3.2 * len(picked)), squeeze=False)
        for row, (sequence, frame_index, events_img, gt, flow_size, diff, label) in enumerate(picked):
            for column, picture in enumerate([events_img, gt, flow_size, diff, label]):
                axes[row, column].imshow(picture, cmap="gray" if column in (0, 1, 4) else "magma")
                axes[row, column].set_title(f"{titles[column]}\n{sequence[:12]} frame {frame_index}", fontsize=8)
                axes[row, column].axis("off")
        plt.tight_layout()
        plt.savefig(out / "flow_check.png", dpi=70)
        print(f"\nPicture saved: {out / 'flow_check.png'}")
    print(f"Labels saved (threshold {args.threshold} px): {out}")


if __name__ == "__main__":
    main()
