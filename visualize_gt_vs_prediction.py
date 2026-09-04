#!/usr/bin/env python3
"""
visualize_gt_vs_prediction.py

Purpose
-------
Visualize EXACTLY what the V2 model predicts for dynamic-object segmentation
and compare it against the EVIMO2 Ground Truth.

This script is deliberately standalone. It does not modify the training code.

The prediction used here is:

    prediction_probability = sigmoid(model_output["mask"])

This is the same prediction representation used by trainer_v2.py for
SegmentationMetrics.

The script evaluates thresholds:

    0.3, 0.4, 0.5, 0.6, 0.7

and selects the threshold with the highest IoU, matching the existing
evaluation logic.

Output
------
For selected frames:

    input.png
    ground_truth.png
    prediction_probability.png
    prediction_binary.png
    comparison.png

comparison.png uses:

    GREEN = True Positive
    RED   = False Positive
    BLUE  = False Negative
    BLACK = True Negative

It also writes:

    metrics.txt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# -------------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------------

from trainer_v2 import TrainerV2, TrainerConfig
from src.utils.metrics import (
    get_dynamic_object_ids,
    evimo2_mask_to_binary_dynamic,
)


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


# -------------------------------------------------------------------------
# Image utilities
# -------------------------------------------------------------------------

def normalize_to_uint8(x):
    """
    Convert a tensor/array into uint8 [0, 255].
    """
    x = np.asarray(x, dtype=np.float32)

    x_min = x.min()
    x_max = x.max()

    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros_like(x)

    return (x * 255.0).clip(0, 255).astype(np.uint8)


def save_gray(array, path):
    """
    Save a grayscale image.
    """
    array = normalize_to_uint8(array)
    Image.fromarray(array, mode="L").save(path)


def save_binary(mask, path):
    """
    Save binary mask as black/white.
    """
    mask = np.asarray(mask).astype(np.uint8)
    image = mask * 255
    Image.fromarray(image, mode="L").save(path)


def make_comparison(gt, pred):
    """
    Create TP/FP/FN/TN visualization.

    GREEN = TP
    RED   = FP
    BLUE  = FN
    BLACK = TN
    """

    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)

    tp = gt & pred
    fp = (~gt) & pred
    fn = gt & (~pred)

    comparison = np.zeros(
        (gt.shape[0], gt.shape[1], 3),
        dtype=np.uint8,
    )

    # True Positive = green
    comparison[tp] = [0, 255, 0]

    # False Positive = red
    comparison[fp] = [255, 0, 0]

    # False Negative = blue
    comparison[fn] = [0, 0, 255]

    return comparison


def save_input_image(frame, path):
    """
    Try to save an EVIMO frame if available.

    This function handles common tensor/array formats.
    """

    if frame is None:
        return False

    x = frame

    if torch.is_tensor(x):
        x = x.detach().cpu()

        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)

        x = x.numpy()

    x = np.asarray(x)

    if x.ndim == 2:
        Image.fromarray(normalize_to_uint8(x), mode="L").save(path)
        return True

    if x.ndim == 3:

        # CHW -> HWC
        if x.shape[0] in (1, 3):
            x = np.transpose(x, (1, 2, 0))

        if x.shape[-1] == 1:
            Image.fromarray(
                normalize_to_uint8(x[..., 0]),
                mode="L",
            ).save(path)
            return True

        if x.shape[-1] == 3:
            x = x.astype(np.float32)

            # Handle [0,1]
            if x.max() <= 1.0:
                x = x * 255.0

            # Handle arbitrary normalized input
            if x.min() < 0:
                x = normalize_to_uint8(x)
            else:
                x = x.clip(0, 255).astype(np.uint8)

            Image.fromarray(x, mode="RGB").save(path)
            return True

    return False


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def calculate_metrics(gt, pred):
    """
    Calculate binary segmentation metrics.
    """

    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    union = tp + fp + fn

    if union > 0:
        iou = tp / union
    else:
        iou = 1.0

    precision_den = tp + fp
    recall_den = tp + fn

    precision = (
        tp / precision_den
        if precision_den > 0
        else 0.0
    )

    recall = (
        tp / recall_den
        if recall_den > 0
        else 0.0
    )

    f1_den = precision + recall

    f1 = (
        2.0 * precision * recall / f1_den
        if f1_den > 0
        else 0.0
    )

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
# Main evaluation
# -------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(trainer, loader):
    """
    Run the model over the loader and collect EXACTLY the same type of
    prediction used by trainer_v2 evaluation.

    Returns
    -------
    samples:
        list of dictionaries containing:

            probability
            ground_truth
            input_frame
            sample_index
    """

    model = trainer.model
    model.eval()

    samples = []

    global_index = 0

    for raw in loader:

        # Same transformation used by training/evaluation.
        vb = trainer.transform(raw)

        # Same voxel construction used by trainer_v2.
        vox = torch.stack(
            [f.voxel_grid for f in vb.frames],
            dim=1,
        ).to(trainer.device)

        vb = vb.to(trainer.device)

        # -------------------------------------------------------------
        # THIS IS THE IMPORTANT PART
        # -------------------------------------------------------------
        #
        # This is the model prediction that F1/IoU is calculated from.
        #
        out = model(vox, vb)

        mask_logits = out["mask"]

        prediction_probability = torch.sigmoid(mask_logits)

        # -------------------------------------------------------------

        # EVIMO2 GT
        raw_masks = raw.frames[-1].mask
        frame_motions = raw.frames[-1].frame_motion

        for batch_index, (prob, raw_gt, motion) in enumerate(
            zip(
                prediction_probability,
                raw_masks,
                frame_motions,
            )
        ):

            # Some EVIMO frames legitimately have no mask.
            if raw_gt is None:
                global_index += 1
                continue

            # Dynamic object IDs are determined from object speed,
            # exactly like the project's evaluation.
            dynamic_ids = get_dynamic_object_ids(motion)

            gt = evimo2_mask_to_binary_dynamic(
                raw_gt,
                dynamic_ids,
            )

            # Convert prediction to [H,W].
            prob = prob.squeeze()

            # Match the exact spatial resizing used by evaluation.
            if tuple(prob.shape[-2:]) != tuple(gt.shape):

                prob = F.interpolate(
                    prob.unsqueeze(0).unsqueeze(0),
                    size=gt.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            probability_np = (
                prob.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            gt_np = (
                gt.detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
                if torch.is_tensor(gt)
                else np.asarray(gt).astype(np.uint8)
            )

            samples.append(
                {
                    "probability": probability_np,
                    "ground_truth": gt_np,
                    "sample_index": global_index,
                    "input": None,
                }
            )

            global_index += 1

    return samples


# -------------------------------------------------------------------------
# Find best threshold
# -------------------------------------------------------------------------

def find_best_threshold(samples):
    """
    Calculate aggregate IoU/F1 for every threshold.

    This follows the same threshold sweep used by SegmentationMetrics.
    """

    results = []

    for threshold in THRESHOLDS:

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for sample in samples:

            gt = sample["ground_truth"].astype(bool)

            pred = (
                sample["probability"] >= threshold
            )

            total_tp += np.logical_and(
                gt,
                pred,
            ).sum()

            total_fp += np.logical_and(
                ~gt,
                pred,
            ).sum()

            total_fn += np.logical_and(
                gt,
                ~pred,
            ).sum()

        union = total_tp + total_fp + total_fn

        iou = (
            total_tp / union
            if union > 0
            else 1.0
        )

        precision_den = total_tp + total_fp
        recall_den = total_tp + total_fn

        precision = (
            total_tp / precision_den
            if precision_den > 0
            else 0.0
        )

        recall = (
            total_tp / recall_den
            if recall_den > 0
            else 0.0
        )

        f1_den = precision + recall

        f1 = (
            2 * precision * recall / f1_den
            if f1_den > 0
            else 0.0
        )

        results.append(
            {
                "threshold": threshold,
                "iou": float(iou),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
        )

    # IMPORTANT:
    # The project's SegmentationMetrics selects best IoU threshold.
    best = max(
        results,
        key=lambda x: x["iou"],
    )

    return best, results


# -------------------------------------------------------------------------
# Save visualizations
# -------------------------------------------------------------------------

def save_visualizations(
    samples,
    output_dir,
    threshold,
    max_images,
):
    """
    Save Ground Truth vs Prediction visualizations.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = samples[:max_images]

    for sample_number, sample in enumerate(selected):

        probability = sample["probability"]
        gt = sample["ground_truth"]

        prediction = (
            probability >= threshold
        ).astype(np.uint8)

        sample_dir = (
            output_dir
            / f"sample_{sample_number:04d}"
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -------------------------------------------------------------
        # 1. Ground Truth
        # -------------------------------------------------------------

        save_binary(
            gt,
            sample_dir / "ground_truth.png",
        )

        # -------------------------------------------------------------
        # 2. Prediction probability
        # -------------------------------------------------------------

        save_gray(
            probability,
            sample_dir / "prediction_probability.png",
        )

        # -------------------------------------------------------------
        # 3. Binary prediction
        # -------------------------------------------------------------

        save_binary(
            prediction,
            sample_dir / "prediction_binary.png",
        )

        # -------------------------------------------------------------
        # 4. GT vs Prediction comparison
        # -------------------------------------------------------------

        comparison = make_comparison(
            gt,
            prediction,
        )

        Image.fromarray(
            comparison,
            mode="RGB",
        ).save(
            sample_dir / "comparison.png"
        )

        # -------------------------------------------------------------
        # 5. Metrics for this frame
        # -------------------------------------------------------------

        frame_metrics = calculate_metrics(
            gt,
            prediction,
        )

        with open(
            sample_dir / "metrics.txt",
            "w",
        ) as f:

            f.write(
                f"Sample: {sample['sample_index']}\n"
            )

            f.write(
                f"Threshold: {threshold:.2f}\n\n"
            )

            for key, value in frame_metrics.items():
                f.write(
                    f"{key}: {value}\n"
                )


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Visualize EVIMO2 Ground Truth versus "
            "the exact V2 model prediction used "
            "for F1/IoU."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to EVIMO2 dataset root.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained V2 checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./gt_vs_prediction",
        help="Directory for visualization output.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=12,
        help="Number of frames to visualize.",
    )

    parser.add_argument(
        "--sensors",
        nargs="+",
        default=["left_camera"],
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )

    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Use the project's overfit dataset behavior.",
    )

    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Evaluate current model instead of EMA weights.",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # -----------------------------------------------------------------
    # Build TrainerConfig using the project's existing configuration.
    # -----------------------------------------------------------------

    cfg = TrainerConfig(
        dataset_root=args.dataset_root,
        sensors=args.sensors,
        split=args.split,
        batch_size=args.batch_size,
        device=device,
        use_ema=not args.no_ema,
    )

    # -----------------------------------------------------------------
    # Create trainer.
    # -----------------------------------------------------------------

    trainer = TrainerV2(cfg)

    # -----------------------------------------------------------------
    # Load checkpoint.
    # -----------------------------------------------------------------

    print()
    print("Loading checkpoint...")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    if "model" in checkpoint:
        trainer.model.load_state_dict(
            checkpoint["model"],
            strict=False,
        )

    elif "state_dict" in checkpoint:
        trainer.model.load_state_dict(
            checkpoint["state_dict"],
            strict=False,
        )

    else:
        trainer.model.load_state_dict(
            checkpoint,
            strict=False,
        )

    trainer.model.to(device)
    trainer.model.eval()

    print("Checkpoint loaded.")

    # -----------------------------------------------------------------
    # Build data loader.
    # -----------------------------------------------------------------

    print()
    print("Building EVIMO2 dataloader...")

    loader = trainer._build_dataloader(
        train=True
    )

    print("Dataloader ready.")

    # -----------------------------------------------------------------
    # Collect EXACT predictions.
    # -----------------------------------------------------------------

    print()
    print("Running model...")
    print()
    print("Prediction used:")
    print("    sigmoid(model_output['mask'])")
    print()

    samples = collect_predictions(
        trainer,
        loader,
    )

    if len(samples) == 0:
        raise RuntimeError(
            "No valid EVIMO2 samples with Ground Truth masks were found."
        )

    print(
        f"Collected {len(samples)} valid samples."
    )

    # -----------------------------------------------------------------
    # Find best threshold.
    # -----------------------------------------------------------------

    best, all_results = find_best_threshold(
        samples
    )

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
    print(
        f"BEST THRESHOLD: {best['threshold']:.2f}"
    )

    print(
        f"BEST IoU      : {best['iou']:.4f}"
    )

    print(
        f"BEST F1       : {best['f1']:.4f}"
    )

    # -----------------------------------------------------------------
    # Save visualizations.
    # -----------------------------------------------------------------

    print()
    print("Saving visualizations...")

    save_visualizations(
        samples=samples,
        output_dir=args.output_dir,
        threshold=best["threshold"],
        max_images=args.num_images,
    )

    # -----------------------------------------------------------------
    # Save global metrics.
    # -----------------------------------------------------------------

    metrics_path = (
        Path(args.output_dir)
        / "metrics.txt"
    )

    with open(metrics_path, "w") as f:

        f.write(
            "EVIMO2 Ground Truth vs Prediction\n"
        )

        f.write(
            "=================================\n\n"
        )

        f.write(
            "Prediction:\n"
        )

        f.write(
            "sigmoid(model_output['mask'])\n\n"
        )

        f.write(
            "Thresholds:\n"
        )

        f.write(
            f"{THRESHOLDS}\n\n"
        )

        f.write(
            f"Best threshold: "
            f"{best['threshold']:.4f}\n"
        )

        f.write(
            f"Best IoU: "
            f"{best['iou']:.6f}\n"
        )

        f.write(
            f"Best F1: "
            f"{best['f1']:.6f}\n"
        )

        f.write(
            f"Best precision: "
            f"{best['precision']:.6f}\n"
        )

        f.write(
            f"Best recall: "
            f"{best['recall']:.6f}\n"
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print(
        "Visualizations saved to:"
    )

    print(
        Path(args.output_dir).resolve()
    )

    print()
    print(
        "Open comparison.png inside each sample directory."
    )


if __name__ == "__main__":
    main()