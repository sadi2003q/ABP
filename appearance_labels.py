"""
appearance_labels.py — SMALL TEST of appearance + motion pseudo-labels (stage B3).

Why
---
Stage B1 finds moving objects only by their BLUR after undoing the camera
motion. It fails when objects move slowly or when one big object makes
most of the events (see claude/self-supervised-status.md). This test
groups the image by APPEARANCE instead, and uses motion only to decide
which group is the moving object.

How (per labelled frame)
------------------------
1. Event image: events of a few frames around the labelled frame, turned
   into a 3-channel picture (event count, latest event time, blurred count).
2. Appearance: DINOv2 (a vision model trained WITHOUT labels on images)
   gives one feature vector per 14 x 14 pixel patch.
3. Grouping: normalized cut (TokenCut, Wang et al. 2022) splits the patches
   into two groups that look different.
4. Which group is the moving object? Several rules are scored side by side:
     contrast : group with the higher B1 "moves on its own" score
     density  : group with more events per pixel
     smaller  : the smaller group (objects are usually smaller than the background)
     oracle   : group that matches the TRUE mask better — REPORT ONLY, it uses
                the true masks and is never a usable label. It tells how good
                the appearance split itself is.
   The B1 label (one motion + dither) is scored on the same frames for comparison.

Nothing here trains a network. The true masks are used ONLY for scoring
(and for the clearly marked "oracle" upper bound).

Related work: Arja et al. 2024 (arXiv 2405.15209: DINO features on event
frames + normalized cuts, 79 % IoU on EVIMO2, uses a supervised flow network
and takes seconds per frame); TokenCut (arXiv 2209.00383).

Needs Internet ON in Kaggle the first time (downloads DINOv2 weights).

Usage (small test)
------------------
python appearance_labels.py \
    --dataset-root /kaggle/working/evimo_root \
    --sequence scene14_dyn_test_01_000000 scene15_dyn_test_00_000000 scene6_dyn_train_03_000000 \
    --frame-step 10 --out-dir /kaggle/working/appearance_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from generate_pseudo_labels import (
    Pooled, get_sample, drop_old_samples, gather_window,
    dither_positions, score_from_flow, score_to_label,
)
from src.data.event_flow import fit_background_flow, flow_field
from src.data.pseudo_labels import save_sequence_labels

RULES = ["contrast", "density", "smaller", "oracle"]

# ImageNet colour statistics that DINOv2 expects
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ----------------------------------------------------------------------
# 1. Event image
# ----------------------------------------------------------------------

def make_event_image(xy, t, H, W):
    """
    Turn events into a 3-channel picture with values in [0, 1].
      channel 0: how many events hit each pixel (log scale)
      channel 1: how recently each pixel fired (1 = at the end of the window)
      channel 2: channel 0 blurred, so thin edges become wider shapes
    """
    # Step 1: whole-pixel positions inside the image
    x = np.clip(np.round(xy[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(xy[:, 1]).astype(int), 0, H - 1)

    # Step 2: event count per pixel, e.g. 0, 1, 2, ...
    count = np.zeros((H, W), np.float64)
    np.add.at(count, (y, x), 1.0)

    # Step 3: log scale, then 0..1
    count = np.log1p(count)
    count = count / max(count.max(), 1e-9)

    # Step 4: latest event time per pixel, 0..1 (0 = no event)
    latest = np.zeros((H, W), np.float64)
    if len(t) > 0:
        t_norm = (t - t.min()) / max(t.max() - t.min(), 1e-9)
        np.maximum.at(latest, (y, x), t_norm)

    # Step 5: blurred count
    blurred = ndimage.gaussian_filter(count, 2.0)
    blurred = blurred / max(blurred.max(), 1e-9)

    # Step 6: stack into (3, H, W)
    return np.stack([count, latest, blurred]).astype(np.float32)


# ----------------------------------------------------------------------
# 2. DINOv2 patch features
# ----------------------------------------------------------------------

def load_dinov2(device):
    """DINOv2 small model (ViT-S/14), trained without labels. Downloads once."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    return model.to(device)


@torch.no_grad()
def patch_features(model, image, image_h, image_w, device):
    """
    image: (3, H, W) values in [0, 1].
    Returns features (rows, cols, C) with unit length, e.g. (32, 42, 384).
    """
    # Step 1: resize to a size that is a multiple of 14 pixels
    tensor = torch.from_numpy(image).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(image_h, image_w), mode="bilinear", align_corners=False)

    # Step 2: normalise like ImageNet pictures
    tensor = (tensor[0] - IMAGENET_MEAN) / IMAGENET_STD
    tensor = tensor.unsqueeze(0).to(device)

    # Step 3: one feature vector per 14 x 14 patch
    output = model.forward_features(tensor)
    tokens = output["x_norm_patchtokens"][0]           # (rows * cols, C)

    # Step 4: unit length, so a dot product = cosine similarity
    tokens = F.normalize(tokens, dim=1)
    rows = image_h // 14
    cols = image_w // 14
    return tokens.reshape(rows, cols, -1).cpu().numpy()


# ----------------------------------------------------------------------
# 3. Normalized cut (TokenCut)
# ----------------------------------------------------------------------

def normalized_cut(features, tau=0.2, eps=1e-5):
    """
    Split patches into two groups that look different.
    features: (rows, cols, C) unit vectors. Returns a boolean (rows, cols) array.
    """
    rows, cols, channels = features.shape
    flat = features.reshape(-1, channels)

    # Step 1: similarity between every pair of patches (cosine)
    similarity = flat @ flat.T

    # Step 2: graph edges: strong link if similar enough, almost none otherwise (TokenCut)
    affinity = np.where(similarity > tau, 1.0, eps)

    # Step 3: degree of each patch = sum of its links
    degree = np.diag(affinity.sum(axis=1))

    # Step 4: second-smallest solution of (D - A) v = lambda D v  (the normalized cut)
    values, vectors = scipy.linalg.eigh(degree - affinity, degree, subset_by_index=[1, 1])
    second = vectors[:, 0]

    # Step 5: split at the average value
    group = second > second.mean()
    return group.reshape(rows, cols)


# ----------------------------------------------------------------------
# 4. Helpers
# ----------------------------------------------------------------------

def to_patch_grid(image_map, rows, cols):
    """Average an (H, W) map over each patch -> (rows, cols)."""
    tensor = torch.from_numpy(np.asarray(image_map, np.float32))[None, None]
    pooled = F.adaptive_avg_pool2d(tensor, (rows, cols))
    return pooled[0, 0].numpy()


def to_pixels(patch_mask, H, W):
    """Boolean (rows, cols) -> boolean (H, W) by nearest-neighbour enlarging."""
    tensor = torch.from_numpy(patch_mask.astype(np.float32))[None, None]
    big = F.interpolate(tensor, size=(H, W), mode="nearest")
    return big[0, 0].numpy() > 0.5


def iou_of(prediction, truth):
    union = np.logical_or(prediction, truth).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(prediction, truth).sum() / union)


def pick_group(rule, group, contrast_patches, density_patches, gt_pixels, H, W):
    """Return the patch mask (rows, cols) chosen as 'moving object' by one rule."""
    side_1 = group
    side_2 = ~group
    if side_1.sum() == 0 or side_2.sum() == 0:
        return np.zeros_like(group)

    if rule == "contrast":
        if contrast_patches[side_1].mean() >= contrast_patches[side_2].mean():
            return side_1
        return side_2
    if rule == "density":
        if density_patches[side_1].mean() >= density_patches[side_2].mean():
            return side_1
        return side_2
    if rule == "smaller":
        if side_1.sum() <= side_2.sum():
            return side_1
        return side_2
    if rule == "oracle":   # REPORT ONLY: uses the true mask
        if gt_pixels is None:
            return np.zeros_like(group)
        if iou_of(to_pixels(side_1, H, W), gt_pixels) >= iou_of(to_pixels(side_2, H, W), gt_pixels):
            return side_1
        return side_2
    raise ValueError(rule)


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Small test: appearance (DINOv2) + motion pseudo-labels")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--sensors", nargs="+", default=["right_camera"])
    p.add_argument("--subset", default="imo")
    p.add_argument("--split", default="train")
    p.add_argument("--sequence", nargs="+", required=True)
    p.add_argument("--frame-step", type=int, default=10, help="Label every N-th frame.")
    p.add_argument("--appearance-frames", type=int, default=5,
                   help="Odd number of frames whose events form the event image.")
    p.add_argument("--select", choices=["contrast", "density", "smaller"], default="contrast",
                   help="Rule whose labels are WRITTEN (all rules are reported).")
    p.add_argument("--tau", type=float, default=0.2, help="Similarity threshold of the normalized cut.")
    p.add_argument("--image-height", type=int, default=448, help="Multiple of 14.")
    p.add_argument("--image-width", type=int, default=588, help="Multiple of 14.")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--figure-frames", type=int, default=6, help="Frames drawn in the check picture.")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    H, W = args.height, args.width
    half = args.appearance_frames // 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    from src.data.dataset import EVIMO2Dataset
    from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic

    # Step 1: data (depth not needed)
    ds = EVIMO2Dataset(dataset_root=args.dataset_root, sensors=tuple(args.sensors),
                       split=args.split, load_depth=False, load_mask=True,
                       subset=args.subset, sequence=tuple(args.sequence))
    refs = ds.index.references

    # Step 2: the appearance model
    model = load_dinov2(device)
    print("DINOv2 loaded", flush=True)

    # Step 3: scores per sequence: B1 label and every rule
    names = ["B1 (one motion + dither)"] + [f"appearance + {r}" for r in RULES]
    scores = {}                        # (sequence, name) -> Pooled
    labels_by_seq = {}                 # labels written for --select
    examples = []                      # frames for the check picture
    prev_theta = {}
    cache = {}
    n_done = 0
    t_start = time.time()

    for i in range(len(ds) - 1):
        if refs[i].sequence_id != refs[i + 1].sequence_id:
            continue
        if int(refs[i].local_frame_index) % args.frame_step != 0:
            continue
        drop_old_samples(cache, keep_from=i - half)

        # Step 4: events around this frame (for the event image)
        window = gather_window(ds, refs, i, half, cache)
        if window is None:
            continue
        a = get_sample(ds, i, cache)
        b = get_sample(ds, i + 1, cache)
        sequence = a.sequence_name

        # Step 5: true mask (scoring only)
        gt = None
        if a.mask is not None:
            gt = evimo2_mask_to_binary_dynamic(a.mask, get_dynamic_object_ids(a.frame_motion)).numpy()
            while gt.ndim > 2:
                gt = gt[0]
            gt = gt.astype(bool)

        # Step 6: B1 motion cue on this frame's own events (one motion + dither)
        xy = np.asarray(a.events_xy).reshape(-1, 2)
        t = np.asarray(a.events_t, np.float64)
        if len(t) < 200:
            continue
        key = (a.sensor, sequence)
        theta, info = fit_background_flow(dither_positions(xy, seed=i), t, a.timestamp, b.timestamp,
                                          H, W, model="planar", init=prev_theta.get(key), seed=i)
        if info["fitted"]:
            prev_theta[key] = theta
        flow = flow_field(theta, H, W, "planar")
        contrast = score_from_flow(xy, t, a.timestamp, b.timestamp, flow)
        b1_label = score_to_label(contrast, 0.4, 0.01)

        # Step 7: appearance features and the normalized cut
        image = make_event_image(window["xy"], window["t"], H, W)
        features = patch_features(model, image, args.image_height, args.image_width, device)
        rows, cols = features.shape[:2]
        group = normalized_cut(features, tau=args.tau)

        # Step 8: motion cues per patch
        contrast_patches = to_patch_grid(contrast, rows, cols)
        density_patches = to_patch_grid(image[0], rows, cols)

        # Step 9: one label per rule, scored against the true mask
        frame_labels = {"B1 (one motion + dither)": b1_label}
        for rule in RULES:
            chosen = pick_group(rule, group, contrast_patches, density_patches, gt, H, W)
            frame_labels[f"appearance + {rule}"] = to_pixels(chosen, H, W)
        if gt is not None:
            for name, label in frame_labels.items():
                scores.setdefault((sequence, name), Pooled()).update(label, gt)

        # Step 10: keep the label of the --select rule
        written = frame_labels[f"appearance + {args.select}"]
        labels_by_seq.setdefault((a.sensor, sequence), {})[int(a.local_frame_index)] = written

        # Step 11: remember a few frames with a moving object for the picture
        if gt is not None and gt.any() and len(examples) < 4 * args.figure_frames:
            examples.append((sequence, int(a.local_frame_index), image[0], gt,
                             to_pixels(group, H, W), b1_label,
                             frame_labels["appearance + contrast"]))

        n_done += 1
        if n_done % 10 == 0:
            rate = (time.time() - t_start) / n_done
            print(f"  {n_done} frames ({rate:.1f} s/frame)", flush=True)

    # Step 12: save the written labels
    for (sensor, sequence), labels in labels_by_seq.items():
        save_sequence_labels(out, sensor, sequence, labels)

    # Step 13: result table
    print("\n" + "=" * 92)
    print("SMALL TEST: label quality against the TRUE masks (report only)")
    print("'oracle' uses the true mask to pick the group: upper bound of the appearance split, NOT a usable label")
    print("=" * 92)
    report = {}
    for sequence in args.sequence:
        print(f"--- {sequence} ---")
        for name in names:
            pooled = scores.get((sequence, name))
            if pooled is None:
                continue
            precision = pooled.tp / max(1, pooled.tp + pooled.fp)
            recall = pooled.tp / max(1, pooled.tp + pooled.fn)
            print(f"  {name:<28} IoU {pooled.iou():.3f}   precision {precision:.3f}   recall {recall:.3f}   "
                  f"false-blob frames {pooled.static_frames_with_fp}/{pooled.static_frames}")
            report[f"{sequence} | {name}"] = {"iou": pooled.iou(), "precision": precision, "recall": recall}
    with open(out / "appearance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Step 14: check picture (a few frames per sequence)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picked = []
    for sequence in args.sequence:
        own = [e for e in examples if e[0] == sequence]
        picked += own[:: max(1, len(own) // 2)][:2]
    if picked:
        titles = ["events", "true moving mask", "appearance split (2 groups)", "B1 label", "appearance + contrast"]
        figure, axes = plt.subplots(len(picked), 5, figsize=(20, 3.2 * len(picked)), squeeze=False)
        for row, (sequence, frame_index, events_img, gt, split, b1, app) in enumerate(picked):
            for column, picture in enumerate([events_img, gt, split, b1, app]):
                axes[row, column].imshow(picture, cmap="gray")
                axes[row, column].set_title(f"{titles[column]}\n{sequence[:12]} frame {frame_index}", fontsize=8)
                axes[row, column].axis("off")
        plt.tight_layout()
        plt.savefig(out / "appearance_check.png", dpi=70)
        print(f"\nPicture saved: {out / 'appearance_check.png'}")
    print(f"Labels saved ({args.select} rule): {out}")


if __name__ == "__main__":
    main()
