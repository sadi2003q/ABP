#!/usr/bin/env python3
"""
evaluate_with_corrected_gt.py

Purpose
-------
Re-computes IoU/F1 using the STRICTER ground-truth definition from
get_dynamic_object_ids_silhouette_filtered() (added to
src/utils/metrics.py), instead of the original speed-only GT.

An object only counts as "dynamic" here if BOTH:
    1. Its 3D speed exceeds MOTION_THRESHOLD_SPEED (original rule), AND
    2. Its silhouette actually changed between this frame and the
       previous frame (self-IoU <= 0.85 by default) — i.e. it visibly
       moved on screen, not just in noisy 3D pose data.

This script prints and saves BOTH numbers side by side:
    - "original"  : IoU/F1 using the existing speed-only GT
                    (matches your training log / trainer_v2.py exactly)
    - "corrected" : IoU/F1 using the silhouette-filtered GT

This does not change training or trainer_v2.py at all — it's a
standalone comparison for your own investigation / to show your
supervisor.

Usage
-----
python evaluate_with_corrected_gt.py \
    --dataset-root /content/drive/MyDrive/single_seq_root \
    --checkpoint /content/runs/exp_v2/checkpoints/epoch_030.pth \
    --sensors left_camera --split train --overfit --no-ema \
    --output-dir ./corrected_gt_eval
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from trainer_v2 import TrainerV2, TrainConfigV2
from src.utils.metrics import (
    get_dynamic_object_ids,
    get_dynamic_object_ids_silhouette_filtered,
    evimo2_mask_to_binary_dynamic,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def aggregate_metrics(samples, gt_key, threshold):
    """Pixel-level aggregate IoU/F1 across every sample."""
    total_tp = total_fp = total_fn = 0

    for s in samples:
        gt = s[gt_key].astype(bool)
        pred = s["probability"] >= threshold

        total_tp += np.logical_and(gt, pred).sum()
        total_fp += np.logical_and(~gt, pred).sum()
        total_fn += np.logical_and(gt, ~pred).sum()

    union = total_tp + total_fp + total_fn
    iou = total_tp / union if union > 0 else 1.0

    precision_den = total_tp + total_fp
    recall_den = total_tp + total_fn
    precision = total_tp / precision_den if precision_den > 0 else 0.0
    recall = total_tp / recall_den if recall_den > 0 else 0.0

    f1_den = precision + recall
    f1 = 2 * precision * recall / f1_den if f1_den > 0 else 0.0

    return {"iou": float(iou), "f1": float(f1),
            "precision": float(precision), "recall": float(recall)}


# -------------------------------------------------------------------------
# Main evaluation loop
# -------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions_both_gts(trainer, loader, self_iou_threshold):
    """
    Runs the model once and computes BOTH the original speed-only GT
    and the silhouette-filtered GT for every frame, keyed by the true
    frame_id (not dataloader/shuffle order) so the silhouette check
    compares against the correct previous frame.
    """
    model = trainer.model
    model.eval()

    by_frame_id = {}
    raw_mask_by_frame_id = {}

    for raw in loader:
        vb = trainer.transform(raw)

        vox = torch.stack(
            [f.voxel_grid for f in vb.frames],
            dim=1,
        ).to(trainer.device)

        vb = vb.to(trainer.device)

        out = model(vox, vb)
        prediction_probability = torch.sigmoid(out["mask"])

        ref_frame_ids = raw.frames[-1].frame_ids
        raw_masks = raw.frames[-1].mask
        frame_motions = raw.frames[-1].frame_motion

        for prob, raw_gt, motion, fid in zip(
            prediction_probability,
            raw_masks,
            frame_motions,
            ref_frame_ids,
        ):
            if raw_gt is None:
                continue

            fid = int(fid)
            prob = prob.squeeze()

            by_frame_id[fid] = {
                "probability": prob.detach().cpu().numpy().astype(np.float32),
                "raw_mask": raw_gt,
                "motion": motion,
            }
            raw_mask_by_frame_id[fid] = raw_gt

    # Second pass, in TRUE frame order, so the silhouette check always
    # compares against the immediately preceding real frame.
    sorted_ids = sorted(by_frame_id.keys())
    prev_raw_mask = None
    prev_id = None

    samples = []

    for fid in sorted_ids:
        entry = by_frame_id[fid]
        raw_gt = entry["raw_mask"]
        motion = entry["motion"]
        prob = entry["probability"]

        # Only treat as "previous frame available" if truly adjacent.
        prev_for_this = prev_raw_mask if (prev_id is not None and fid == prev_id + 1) else None

        original_ids = get_dynamic_object_ids(motion)
        corrected_ids = get_dynamic_object_ids_silhouette_filtered(
            motion, raw_gt, prev_for_this, self_iou_threshold=self_iou_threshold
        )

        gt_original = evimo2_mask_to_binary_dynamic(raw_gt, original_ids)
        gt_corrected = evimo2_mask_to_binary_dynamic(raw_gt, corrected_ids)

        gt_original = torch.as_tensor(gt_original).squeeze()
        gt_corrected = torch.as_tensor(gt_corrected).squeeze()

        prob_t = torch.as_tensor(prob)
        gt_hw = tuple(gt_original.shape[-2:])
        if tuple(prob_t.shape[-2:]) != gt_hw:
            prob_t = F.interpolate(
                prob_t.unsqueeze(0).unsqueeze(0),
                size=gt_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            prob = prob_t.numpy()

        samples.append({
            "frame_id": fid,
            "probability": prob,
            "ground_truth_original": gt_original.numpy().astype(np.uint8),
            "ground_truth_corrected": gt_corrected.numpy().astype(np.uint8),
        })

        prev_raw_mask = raw_gt
        prev_id = fid

    return samples


def plot_comparison(original_results, corrected_results, output_dir):
    if not HAS_MPL:
        return

    thresholds = [r["threshold"] for r in original_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

    axes[0].plot(thresholds, [r["iou"] for r in original_results],
                 marker="o", label="Original GT (speed-only)", color="#1f77b4")
    axes[0].plot(thresholds, [r["iou"] for r in corrected_results],
                 marker="s", label="Corrected GT (speed + silhouette)", color="#2ca02c")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("IoU")
    axes[0].set_title("IoU: original vs corrected GT")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(thresholds, [r["f1"] for r in original_results],
                 marker="o", label="Original GT (speed-only)", color="#1f77b4")
    axes[1].plot(thresholds, [r["f1"] for r in corrected_results],
                 marker="s", label="Corrected GT (speed + silhouette)", color="#2ca02c")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("F1")
    axes[1].set_title("F1: original vs corrected GT")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "original_vs_corrected_gt.png")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare model IoU/F1 using the original speed-only "
                     "GT vs a corrected silhouette-filtered GT."
    )
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./corrected_gt_eval")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sensors", nargs="+", default=["left_camera"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset", type=str, default="imo")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--self-iou-threshold", type=float, default=0.85,
                         help="Silhouette self-IoU above which an object is "
                              "considered 'not visibly moving' and dropped "
                              "from the corrected GT.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("ORIGINAL vs CORRECTED (SILHOUETTE-FILTERED) GROUND TRUTH")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    cfg_kwargs = dict(
        dataset_root=args.dataset_root,
        sensors=tuple(args.sensors),
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overfit_mode=args.overfit,
        use_ema=not args.no_ema,
    )
    if "subset" in TrainConfigV2.__dataclass_fields__:
        cfg_kwargs["subset"] = args.subset

    cfg = TrainConfigV2(**cfg_kwargs)
    trainer = TrainerV2(cfg)

    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    load_result = trainer.model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"  WARNING: {len(load_result.missing_keys)} missing, "
              f"{len(load_result.unexpected_keys)} unexpected keys.")
    else:
        print("  All checkpoint weights matched model parameters exactly.")

    trainer.model.to(device)
    trainer.model.eval()

    loader = trainer.val_loader if args.split == "val" else trainer.train_loader
    print(f"Dataloader ready ({len(loader)} batches).")

    print()
    print("Running model and computing both GT definitions...")
    samples = collect_predictions_both_gts(trainer, loader, args.self_iou_threshold)

    n_original_dynamic = sum(1 for s in samples if s["ground_truth_original"].sum() > 0)
    n_corrected_dynamic = sum(1 for s in samples if s["ground_truth_corrected"].sum() > 0)

    print(f"Total frames evaluated: {len(samples)}")
    print(f"  frames with dynamic GT (original)  : {n_original_dynamic}")
    print(f"  frames with dynamic GT (corrected) : {n_corrected_dynamic}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 90)
    print(f"{'Threshold':>10} | {'IoU(orig)':>10} | {'IoU(corr)':>10} | "
          f"{'F1(orig)':>9} | {'F1(corr)':>9} | {'Delta IoU':>10}")
    print("=" * 90)

    original_results, corrected_results = [], []

    for threshold in THRESHOLDS:
        r_orig = aggregate_metrics(samples, "ground_truth_original", threshold)
        r_corr = aggregate_metrics(samples, "ground_truth_corrected", threshold)

        r_orig["threshold"] = threshold
        r_corr["threshold"] = threshold
        original_results.append(r_orig)
        corrected_results.append(r_corr)

        delta = r_corr["iou"] - r_orig["iou"]
        print(f"{threshold:>10.2f} | {r_orig['iou']:>10.4f} | {r_corr['iou']:>10.4f} | "
              f"{r_orig['f1']:>9.4f} | {r_corr['f1']:>9.4f} | {delta:>+10.4f}")

    if HAS_MPL:
        plot_comparison(original_results, corrected_results, output_dir)
        print()
        print(f"Saved comparison plot: {output_dir / 'original_vs_corrected_gt.png'}")

    report_path = output_dir / "original_vs_corrected_report.txt"
    with open(report_path, "w") as f:
        f.write("ORIGINAL vs CORRECTED GROUND TRUTH\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total frames: {len(samples)}\n")
        f.write(f"Frames with dynamic GT (original)  : {n_original_dynamic}\n")
        f.write(f"Frames with dynamic GT (corrected) : {n_corrected_dynamic}\n\n")
        for r_o, r_c in zip(original_results, corrected_results):
            f.write(f"threshold={r_o['threshold']:.2f}  "
                     f"orig_iou={r_o['iou']:.4f}  corr_iou={r_c['iou']:.4f}  "
                     f"orig_f1={r_o['f1']:.4f}  corr_f1={r_c['f1']:.4f}\n")

    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()