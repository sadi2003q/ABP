#!/usr/bin/env python3
"""
visualize_gt_vs_prediction.py

Purpose
-------
Produce clear, presentation-ready figures answering exactly the question
your supervisor asked:

    "What is the model predicting, and how does it compare to the
     Ground Truth used to compute the F1 / IoU score?"

For each visualized frame this script saves ONE combined figure with
four panels, side by side:

    [ Event input ]  [ Ground Truth ]  [ Prediction ]  [ Overlay: TP/FP/FN ]

- Event input        : the actual event-camera voxel grid the model saw,
                        rendered as a red/blue polarity image (this
                        dataset has no classical/RGB frames — EVIMO2
                        "left_camera" sequences are event-only).
- Ground Truth        : the EVIMO2 dynamic-object mask (speed-thresholded,
                         same logic used for the F1/IoU number).
- Prediction          : sigmoid(model_output["mask"]) thresholded at the
                         best IoU threshold — the EXACT quantity F1/IoU
                         is computed from in trainer_v2.py.
- Overlay             : green = correct detection (TP), red = false
                         alarm (FP), blue = missed detection (FN),
                         black = correct background (TN).

It also saves:
    - metrics.txt               (global IoU/F1/precision/recall + best threshold)
    - per-sample metrics.txt    (inside each sample folder)
    - threshold_sweep.png       (IoU/F1 vs threshold — shows the model
                                  isn't just getting lucky at one cutoff)

This script is standalone and does not modify the training code.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trainer_v2 import TrainerV2, TrainConfigV2
from src.utils.metrics import (
    get_dynamic_object_ids,
    evimo2_mask_to_binary_dynamic,
)


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


# -------------------------------------------------------------------------
# Event voxel -> viewable image
# -------------------------------------------------------------------------

def voxel_to_event_image(voxel, percentile=99.0, gamma=0.5):
    """
    Convert an event voxel grid (num_bins, H, W) into a viewable RGB
    image using the standard event-camera convention:

        red   = net positive polarity activity at that pixel
        blue  = net negative polarity activity at that pixel
        black = no events

    Event cameras routinely have a handful of "hot pixels" that fire
    far more often than every real, meaningful pixel (e.g. one pixel
    at 80+ net events while the 99th percentile of the rest of the
    frame is ~6). Normalizing by the raw max would make everything
    else look black. Instead we clip to a high percentile of the
    nonzero activity (robust to hot pixels) and apply a gamma curve
    (events are naturally sparse/skewed, so a linear map still looks
    mostly black — gamma<1 brightens midtones for visibility).

    Parameters
    ----------
    voxel : torch.Tensor or np.ndarray, shape (num_bins, H, W)
    percentile : float
        Percentile of nonzero |activity| used as the clipping ceiling.
    gamma : float
        Exponent applied after normalizing to [0, 1]. <1 brightens.

    Returns
    -------
    np.ndarray uint8, shape (H, W, 3)
    """
    if torch.is_tensor(voxel):
        voxel = voxel.detach().cpu().numpy()

    voxel = np.asarray(voxel, dtype=np.float32)

    # Sum across time bins -> net polarity per pixel.
    net = voxel.sum(axis=0)  # (H, W)

    pos = np.clip(net, 0, None)
    neg = np.clip(-net, 0, None)

    def normalize(x):
        nonzero = x[x > 0]
        if nonzero.size == 0:
            return x
        ceiling = np.percentile(nonzero, percentile)
        if ceiling <= 1e-8:
            ceiling = nonzero.max()
        x = np.clip(x / ceiling, 0, 1)
        return x ** gamma

    pos = normalize(pos)
    neg = normalize(neg)

    pos = normalize(pos)
    neg = normalize(neg)

    image = np.zeros((*net.shape, 3), dtype=np.float32)
    image[..., 0] = pos   # red channel   = positive events
    image[..., 2] = neg   # blue channel  = negative events

    return (image * 255.0).clip(0, 255).astype(np.uint8)


def make_overlay(gt, pred):
    """
    TP/FP/FN/TN visualization.

        GREEN = True Positive  (correct dynamic-object detection)
        RED   = False Positive (false alarm)
        BLUE  = False Negative (missed detection)
        BLACK = True Negative  (correct background)
    """
    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)

    tp = gt & pred
    fp = (~gt) & pred
    fn = gt & (~pred)

    overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
    overlay[tp] = [0, 255, 0]
    overlay[fp] = [255, 0, 0]
    overlay[fn] = [0, 0, 255]

    return overlay


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

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


# -------------------------------------------------------------------------
# Main evaluation loop
# -------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(trainer, loader, max_batches=None):
    """
    Run the model over the loader and collect EXACTLY the same
    prediction representation used by trainer_v2's SegmentationMetrics:

        prediction_probability = sigmoid(model_output["mask"])

    Returns a list of dicts: probability, ground_truth, event_image,
    sample_index.
    """
    model = trainer.model
    model.eval()

    samples = []
    global_index = 0

    for batch_index, raw in enumerate(loader):

        if max_batches is not None and batch_index >= max_batches:
            break

        # Same transformation used by training/evaluation.
        vb = trainer.transform(raw)

        # Same voxel construction used by trainer_v2.
        vox = torch.stack(
            [f.voxel_grid for f in vb.frames],
            dim=1,
        ).to(trainer.device)

        vb = vb.to(trainer.device)

        # -------------------------------------------------------------
        # THIS IS THE EXACT PREDICTION F1/IoU IS CALCULATED FROM.
        # -------------------------------------------------------------
        out = model(vox, vb)
        mask_logits = out["mask"]
        prediction_probability = torch.sigmoid(mask_logits)
        # -------------------------------------------------------------

        # Photometric residual that feeds the mask-refinement head —
        # shows WHY the model predicted what it predicted.
        residual = out.get("residual")
        if residual is None:
            residual = torch.zeros_like(mask_logits)

        # Reference-frame voxel grid, for the event input image
        # (last timestep = the frame the mask/prediction correspond to).
        ref_voxels = vb.frames[-1].voxel_grid  # (B, num_bins, H, W)

        raw_masks = raw.frames[-1].mask
        frame_motions = raw.frames[-1].frame_motion

        for prob, raw_gt, motion, voxel, res in zip(
            prediction_probability,
            raw_masks,
            frame_motions,
            ref_voxels,
            residual,
        ):
            # Some EVIMO frames legitimately have no mask.
            if raw_gt is None:
                global_index += 1
                continue

            # Dynamic object IDs, determined from object speed —
            # exactly like the project's evaluation.
            dynamic_ids = get_dynamic_object_ids(motion)

            gt = evimo2_mask_to_binary_dynamic(raw_gt, dynamic_ids)
            gt = torch.as_tensor(gt).squeeze()  # -> (H, W)

            prob = prob.squeeze()  # -> (H, W)

            # Match spatial resizing used by evaluation (only if needed).
            gt_hw = tuple(gt.shape[-2:])
            if tuple(prob.shape[-2:]) != gt_hw:
                prob = F.interpolate(
                    prob.unsqueeze(0).unsqueeze(0),
                    size=gt_hw,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            event_image = voxel_to_event_image(voxel)
            residual_np = res.squeeze().detach().cpu().numpy().astype(np.float32)

            samples.append(
                {
                    "probability": prob.detach().cpu().numpy().astype(np.float32),
                    "ground_truth": gt.detach().cpu().numpy().astype(np.uint8),
                    "event_image": event_image,
                    "residual": residual_np,
                    "sample_index": global_index,
                }
            )

            global_index += 1

    return samples


def find_best_threshold(samples):
    """
    Aggregate IoU/F1 across ALL pixels of ALL samples, for every
    threshold. This is the global, dataset-level version of the metric
    (matches how trainer_v2's SegmentationMetrics aggregates).
    """
    results = []

    for threshold in THRESHOLDS:
        total_tp = total_fp = total_fn = 0

        for sample in samples:
            gt = sample["ground_truth"].astype(bool)
            pred = sample["probability"] >= threshold

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

        results.append({
            "threshold": threshold,
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        })

    best = max(results, key=lambda x: x["iou"])
    return best, results


def plot_threshold_sweep(all_results, output_dir):
    thresholds = [r["threshold"] for r in all_results]
    ious = [r["iou"] for r in all_results]
    f1s = [r["f1"] for r in all_results]

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    ax.plot(thresholds, ious, marker="o", label="IoU", color="#1f77b4")
    ax.plot(thresholds, f1s, marker="s", label="F1", color="#ff7f0e")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("IoU / F1 vs Decision Threshold")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "threshold_sweep.png")
    plt.close(fig)


# -------------------------------------------------------------------------
# Combined 4-panel figure
# -------------------------------------------------------------------------

def save_combined_figure(sample, threshold, path):
    """
    Save one figure with 5 panels:
        Event input | Residual | Ground Truth | Prediction | Overlay (TP/FP/FN)
    """
    probability = sample["probability"]
    gt = sample["ground_truth"].astype(bool)
    prediction = probability >= threshold

    overlay = make_overlay(gt, prediction)
    metrics = calculate_metrics(gt, prediction)

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.3), dpi=150)

    axes[0].imshow(sample["event_image"])
    axes[0].set_title("Event input\n(red=+ / blue=-)")

    axes[1].imshow(sample["residual"], cmap="inferno")
    axes[1].set_title("Photometric residual\n(feeds mask-refinement head)")

    axes[2].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth\n(dynamic-object mask)")

    axes[3].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Prediction (thr={threshold:.2f})\nsigmoid(model_output['mask'])")

    axes[4].imshow(overlay)
    axes[4].set_title(
        f"Overlay: IoU={metrics['iou']:.3f} F1={metrics['f1']:.3f}\n"
        f"green=TP red=FP blue=FN"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Sample {sample['sample_index']}", y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    return metrics


def save_visualizations(samples, output_dir, threshold, max_images):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Most EVIMO2 frames legitimately have ZERO dynamic-object pixels
    # (objects present but not moving fast enough at that instant —
    # only ~20-25% of frames in a typical sequence have any moving
    # object at all). Selecting samples[:max_images] in dataloader
    # order (shuffled) mostly shows those empty frames by chance,
    # which looks like a broken model even when it isn't.
    #
    # Instead, rank samples by how much dynamic GT they contain and
    # show a MIX: the most GT-active frames first (so you can actually
    # see the model detect something), followed by a couple of
    # GT-empty frames (so you can also confirm the model correctly
    # predicts "nothing" when there's nothing — that's a real, useful
    # result too, not a bug).
    def gt_pixel_count(s):
        return int(s["ground_truth"].sum())

    ranked = sorted(samples, key=gt_pixel_count, reverse=True)

    n_active = max(1, int(max_images * 0.8))
    active = [s for s in ranked if gt_pixel_count(s) > 0][:n_active]
    empty = [s for s in ranked if gt_pixel_count(s) == 0][: max_images - len(active)]
    selected = (active + empty)[:max_images]

    if not active:
        print("  NOTE: none of the collected samples have any dynamic-object "
              "GT pixels. Showing empty frames only — try --num-images higher "
              "or a different --split to catch frames with real motion.")

    for sample_number, sample in enumerate(selected):
        sample_dir = output_dir / f"sample_{sample_number:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        metrics = save_combined_figure(
            sample,
            threshold,
            sample_dir / "comparison.png",
        )

        with open(sample_dir / "metrics.txt", "w") as f:
            f.write(f"Sample: {sample['sample_index']}\n")
            f.write(f"Threshold: {threshold:.2f}\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize EVIMO2 Ground Truth versus the exact V2 model "
            "prediction used for F1/IoU."
        )
    )

    parser.add_argument("--dataset-root", type=str, required=True,
                         help="Path to EVIMO2 dataset root.")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to trained V2 checkpoint.")
    parser.add_argument("--output-dir", type=str, default="./gt_vs_prediction",
                         help="Directory for visualization output.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-images", type=int, default=12,
                         help="Number of frames to visualize.")
    parser.add_argument("--sensors", nargs="+", default=["left_camera"])
    parser.add_argument("--num-workers", type=int, default=4,
                         help="Number of dataloader worker processes.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--overfit", action="store_true",
                         help="Use the project's overfit dataset behavior.")
    parser.add_argument("--no-ema", action="store_true",
                         help="Evaluate current model instead of EMA weights.")
    parser.add_argument("--max-batches", type=int, default=None,
                         help="Optional cap on number of batches to run "
                              "(useful for a quick sanity check).")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("EVIMO2 — GROUND TRUTH vs MODEL PREDICTION")
    print("=" * 70)
    print()
    print("Dataset root :", args.dataset_root)
    print("Checkpoint   :", args.checkpoint)
    print("Output       :", args.output_dir)
    print("Thresholds   :", THRESHOLDS)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    cfg = TrainConfigV2(
        dataset_root=args.dataset_root,
        sensors=tuple(args.sensors),
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overfit_mode=args.overfit,
        use_ema=not args.no_ema,
    )

    trainer = TrainerV2(cfg)

    print()
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # trainer_v2.py's _save() writes:
    #   {"epoch": ..., "model_state_dict": ..., "optimizer_state_dict": ...}
    # Check that key FIRST — the previous version of this script checked
    # for "model"/"state_dict" instead, matched neither, and silently
    # fell back to loading the checkpoint's raw dict as if it were a
    # state_dict (with strict=False swallowing the mismatch). That left
    # the model at its random initial weights while still "succeeding",
    # which is why IoU came out near 0 instead of matching training.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    load_result = trainer.model.load_state_dict(state_dict, strict=False)

    if load_result.missing_keys:
        print(f"  WARNING: {len(load_result.missing_keys)} missing keys "
              f"(model parameters that were NOT loaded from the checkpoint):")
        for k in load_result.missing_keys[:10]:
            print(f"    - {k}")
        if len(load_result.missing_keys) > 10:
            print(f"    ... and {len(load_result.missing_keys) - 10} more")

    if load_result.unexpected_keys:
        print(f"  WARNING: {len(load_result.unexpected_keys)} unexpected keys "
              f"(checkpoint entries that did NOT match any model parameter):")
        for k in load_result.unexpected_keys[:10]:
            print(f"    - {k}")
        if len(load_result.unexpected_keys) > 10:
            print(f"    ... and {len(load_result.unexpected_keys) - 10} more")

    if not load_result.missing_keys and not load_result.unexpected_keys:
        print("  All checkpoint weights matched model parameters exactly.")
    else:
        n_model_params = len(list(trainer.model.state_dict().keys()))
        n_loaded = n_model_params - len(load_result.missing_keys)
        print(f"  Loaded {n_loaded}/{n_model_params} parameter tensors. "
              f"If this is far below {n_model_params}, the checkpoint is "
              f"NOT being applied correctly — treat downstream metrics "
              f"as untrustworthy until this is fixed.")

    trainer.model.to(device)
    trainer.model.eval()
    print("Checkpoint loaded.")

    print()
    print("Using EVIMO2 dataloader built by TrainerV2...")

    # TrainerV2.__init__ already calls self._build_dataloaders(), which
    # sets self.train_loader / self.val_loader as attributes.
    if args.split == "val":
        if trainer.val_loader is None:
            raise RuntimeError(
                "No validation loader was built (val split may be empty "
                "or overfit_mode is enabled). Use --split train instead."
            )
        loader = trainer.val_loader
    else:
        loader = trainer.train_loader

    print(f"Dataloader ready ({len(loader)} batches).")

    print()
    print("Running model...")
    print()
    print("Prediction used:")
    print("    sigmoid(model_output['mask'])")
    print()

    samples = collect_predictions(trainer, loader, max_batches=args.max_batches)

    if len(samples) == 0:
        raise RuntimeError(
            "No valid EVIMO2 samples with Ground Truth masks were found."
        )

    print(f"Collected {len(samples)} valid samples.")

    best, all_results = find_best_threshold(samples)

    print()
    print("=" * 70)
    print("THRESHOLD RESULTS")
    print("=" * 70)
    for result in all_results:
        print(
            f"Threshold {result['threshold']:.2f} | "
            f"IoU {result['iou']:.4f} | "
            f"F1 {result['f1']:.4f} | "
            f"Precision {result['precision']:.4f} | "
            f"Recall {result['recall']:.4f}"
        )

    print()
    print(f"BEST THRESHOLD: {best['threshold']:.2f}")
    print(f"BEST IoU      : {best['iou']:.4f}")
    print(f"BEST F1       : {best['f1']:.4f}")

    print()
    print("Saving visualizations...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_visualizations(
        samples=samples,
        output_dir=output_dir,
        threshold=best["threshold"],
        max_images=args.num_images,
    )

    plot_threshold_sweep(all_results, output_dir)

    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("EVIMO2 Ground Truth vs Prediction\n")
        f.write("=================================\n\n")
        f.write("Prediction:\n")
        f.write("sigmoid(model_output['mask'])\n\n")
        f.write("Thresholds:\n")
        f.write(f"{THRESHOLDS}\n\n")
        f.write(f"Best threshold: {best['threshold']:.4f}\n")
        f.write(f"Best IoU: {best['iou']:.6f}\n")
        f.write(f"Best F1: {best['f1']:.6f}\n")
        f.write(f"Best precision: {best['precision']:.6f}\n")
        f.write(f"Best recall: {best['recall']:.6f}\n")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print()
    print("Visualizations saved to:", output_dir.resolve())
    print()
    print("Open comparison.png inside each sample_XXXX folder — that is")
    print("the single image showing input / ground truth / prediction /")
    print("overlay side by side.")


if __name__ == "__main__":
    main()