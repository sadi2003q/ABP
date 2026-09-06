
"""
evaluate_clean_vs_suspect.py

Purpose
-------
Option 3 from the GT diagnosis: report the model's IoU/F1 SEPARATELY
for two groups of frames —

    "clean"    frames: GT dynamic mask looks trustworthy (either no
               dynamic object, or a dynamic object whose silhouette
               actually moved between frames).

    "suspect"  frames: GT dynamic mask covers a large fraction of the
               frame AND the flagged-dynamic objects' silhouettes are
               nearly unchanged from the previous frame — i.e. the
               3D-speed threshold was very likely crossed by pose
               noise/jitter rather than real, visible motion.

This does NOT change training, the model, or the GT-generation code.
It only reports metrics split by frame group, using the exact same
suspect-detection rule as diagnose_dynamic_mask.py, so the two scripts
agree with each other.

Usage
-----
python evaluate_clean_vs_suspect.py \
    --dataset-root /content/drive/MyDrive/single_seq_root \
    --checkpoint /content/runs/exp_v2/checkpoints/epoch_030.pth \
    --sensors left_camera --split train --overfit --no-ema \
    --output-dir ./clean_vs_suspect
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from trainer_v2 import TrainerV2, TrainConfigV2
from src.utils.metrics import (
    get_dynamic_object_ids,
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
MOTION_THRESHOLD_SPEED = 0.05  # matches src/utils/metrics.py exactly


# -------------------------------------------------------------------------
# Suspect-frame classification (same rule as diagnose_dynamic_mask.py)
# -------------------------------------------------------------------------

def classify_frame(gt_mask, prev_gt_mask, pct_threshold, self_iou_threshold):
    """
    gt_mask, prev_gt_mask : boolean (H, W) dynamic masks for this frame
                             and the immediately preceding frame.

    Returns "suspect" if GT coverage is large AND the mask barely
    changed from the previous frame (near-static silhouette despite
    being labeled dynamic). Returns "clean" otherwise (includes
    frames with 0 dynamic pixels, and frames with real motion).
    """
    total_px = gt_mask.size
    pct = 100.0 * gt_mask.sum() / total_px

    if pct <= pct_threshold or prev_gt_mask is None:
        return "clean"

    union = np.logical_or(gt_mask, prev_gt_mask).sum()
    if union == 0:
        return "clean"

    self_iou = np.logical_and(gt_mask, prev_gt_mask).sum() / union

    if self_iou > self_iou_threshold:
        return "suspect"
    return "clean"


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def calculate_metrics(gt, pred):
    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    union = tp + fp + fn
    iou = tp / union if union > 0 else 1.0

    precision_den = tp + fp
    recall_den = tp + fn
    precision = tp / precision_den if precision_den > 0 else 0.0
    recall = tp / recall_den if recall_den > 0 else 0.0

    f1_den = precision + recall
    f1 = 2.0 * precision * recall / f1_den if f1_den > 0 else 0.0

    return iou, f1, precision, recall, tp, fp, fn


def aggregate_group_metrics(group_samples, threshold):
    """Pixel-level aggregate IoU/F1 across every sample in a group."""
    total_tp = total_fp = total_fn = 0

    for s in group_samples:
        gt = s["ground_truth"].astype(bool)
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
            "precision": float(precision), "recall": float(recall),
            "n_frames": len(group_samples)}


# -------------------------------------------------------------------------
# Main evaluation loop
# -------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions_with_frame_ids(trainer, loader, pct_threshold, self_iou_threshold):
    """
    Same prediction logic as visualize_gt_vs_prediction.py, but also
    tracks the TRUE frame_id for each sample (not a shuffled loader
    counter) so clean/suspect classification lines up with the actual
    sequence frame, and computes that classification using the
    previous REAL frame's GT mask (by frame_id, not batch order).
    """
    model = trainer.model
    model.eval()

    # First pass: collect everything keyed by true frame_id.
    by_frame_id = {}

    for raw in loader:
        vb = trainer.transform(raw)

        vox = torch.stack(
            [f.voxel_grid for f in vb.frames],
            dim=1,
        ).to(trainer.device)

        vb = vb.to(trainer.device)

        out = model(vox, vb)
        prediction_probability = torch.sigmoid(out["mask"])

        ref_frame_ids = raw.frames[-1].frame_ids  # (B,) true frame indices
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

            dynamic_ids = get_dynamic_object_ids(motion)
            gt = evimo2_mask_to_binary_dynamic(raw_gt, dynamic_ids)
            gt = torch.as_tensor(gt).squeeze()

            prob = prob.squeeze()
            gt_hw = tuple(gt.shape[-2:])
            if tuple(prob.shape[-2:]) != gt_hw:
                prob = F.interpolate(
                    prob.unsqueeze(0).unsqueeze(0),
                    size=gt_hw,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            by_frame_id[int(fid)] = {
                "probability": prob.detach().cpu().numpy().astype(np.float32),
                "ground_truth": gt.detach().cpu().numpy().astype(np.uint8),
                "frame_id": int(fid),
            }

    # Second pass: classify each frame using the previous frame_id's
    # GT mask (in true sequence order, not dataloader/shuffle order).
    sorted_ids = sorted(by_frame_id.keys())
    prev_gt = None
    prev_id = None

    for fid in sorted_ids:
        sample = by_frame_id[fid]
        gt_bool = sample["ground_truth"].astype(bool)

        # Only compare to the previous frame if it's truly adjacent
        # (fid - 1). If frames were skipped, treat as no previous
        # frame available (classify conservatively as "clean").
        prev_for_this = prev_gt if (prev_id is not None and fid == prev_id + 1) else None

        sample["group"] = classify_frame(
            gt_bool, prev_for_this, pct_threshold, self_iou_threshold
        )

        prev_gt = gt_bool
        prev_id = fid

    return [by_frame_id[fid] for fid in sorted_ids]


def plot_comparison(clean_results, suspect_results, all_results, output_dir):
    if not HAS_MPL:
        return

    thresholds = [r["threshold"] for r in all_results]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    ax.plot(thresholds, [r["iou"] for r in all_results],
            marker="o", label="IoU (all frames)", color="#1f77b4")
    ax.plot(thresholds, [r["iou"] for r in clean_results],
            marker="o", linestyle="--", label="IoU (clean only)", color="#2ca02c")
    ax.plot(thresholds, [r["iou"] for r in suspect_results],
            marker="o", linestyle=":", label="IoU (suspect only)", color="#d62728")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("IoU")
    ax.set_title("IoU vs Threshold — All vs Clean-only vs Suspect-only frames")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "clean_vs_suspect_iou.png")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report IoU/F1 separately for clean vs suspect "
                     "ground-truth frames (Option 3 in the GT diagnosis)."
    )
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./clean_vs_suspect")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sensors", nargs="+", default=["left_camera"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset", type=str, default="imo")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--pct-threshold", type=float, default=15.0,
                         help="GT coverage %% above which a frame is 'large' "
                              "(matches diagnose_dynamic_mask.py default).")
    parser.add_argument("--self-iou-threshold", type=float, default=0.85,
                         help="Silhouette self-IoU above which an object is "
                              "'not visibly moving' (matches diagnose_dynamic_mask.py).")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("CLEAN vs SUSPECT GROUND-TRUTH — SPLIT METRICS")
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
    # Only pass subset if this TrainConfigV2 build supports it (keeps
    # this script working even before the subset patch is applied).
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
              f"{len(load_result.unexpected_keys)} unexpected keys. "
              f"Double-check the checkpoint matches this model.")
    else:
        print("  All checkpoint weights matched model parameters exactly.")

    trainer.model.to(device)
    trainer.model.eval()

    loader = trainer.val_loader if args.split == "val" else trainer.train_loader
    print(f"Dataloader ready ({len(loader)} batches).")

    print()
    print("Running model and classifying frames...")
    samples = collect_predictions_with_frame_ids(
        trainer, loader, args.pct_threshold, args.self_iou_threshold
    )

    clean = [s for s in samples if s["group"] == "clean"]
    suspect = [s for s in samples if s["group"] == "suspect"]

    print(f"Total frames evaluated : {len(samples)}")
    print(f"  clean   : {len(clean)}")
    print(f"  suspect : {len(suspect)}")
    print()
    print("Suspect frame_ids:", sorted(s['frame_id'] for s in suspect))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print(f"{'Threshold':>10} | {'IoU(all)':>9} | {'IoU(clean)':>10} | {'IoU(suspect)':>12} | "
          f"{'F1(all)':>8} | {'F1(clean)':>9} | {'F1(suspect)':>11}")
    print("=" * 70)

    all_results, clean_results, suspect_results = [], [], []

    for threshold in THRESHOLDS:
        r_all = aggregate_group_metrics(samples, threshold)
        r_clean = aggregate_group_metrics(clean, threshold) if clean else None
        r_suspect = aggregate_group_metrics(suspect, threshold) if suspect else None

        r_all["threshold"] = threshold
        all_results.append(r_all)

        if r_clean:
            r_clean["threshold"] = threshold
            clean_results.append(r_clean)
        if r_suspect:
            r_suspect["threshold"] = threshold
            suspect_results.append(r_suspect)

        iou_c = f"{r_clean['iou']:.4f}" if r_clean else "n/a"
        iou_s = f"{r_suspect['iou']:.4f}" if r_suspect else "n/a"
        f1_c = f"{r_clean['f1']:.4f}" if r_clean else "n/a"
        f1_s = f"{r_suspect['f1']:.4f}" if r_suspect else "n/a"

        print(f"{threshold:>10.2f} | {r_all['iou']:>9.4f} | {iou_c:>10} | {iou_s:>12} | "
              f"{r_all['f1']:>8.4f} | {f1_c:>9} | {f1_s:>11}")

    if HAS_MPL and clean_results and suspect_results:
        plot_comparison(clean_results, suspect_results, all_results, output_dir)
        print()
        print(f"Saved comparison plot: {output_dir / 'clean_vs_suspect_iou.png'}")

    report_path = output_dir / "clean_vs_suspect_report.txt"
    with open(report_path, "w") as f:
        f.write("CLEAN vs SUSPECT GROUND-TRUTH — SPLIT METRICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total frames: {len(samples)} (clean={len(clean)}, suspect={len(suspect)})\n")
        f.write(f"Suspect frame_ids: {sorted(s['frame_id'] for s in suspect)}\n\n")
        for r in all_results:
            f.write(f"threshold={r['threshold']:.2f}  all_iou={r['iou']:.4f}  all_f1={r['f1']:.4f}\n")
        f.write("\n")
        for r in clean_results:
            f.write(f"threshold={r['threshold']:.2f}  clean_iou={r['iou']:.4f}  clean_f1={r['f1']:.4f}\n")
        f.write("\n")
        for r in suspect_results:
            f.write(f"threshold={r['threshold']:.2f}  suspect_iou={r['iou']:.4f}  suspect_f1={r['f1']:.4f}\n")

    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()