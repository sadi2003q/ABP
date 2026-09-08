#!/usr/bin/env python3
"""
evaluate_with_gt_depth.py

Purpose
-------
Diagnostic to isolate WHY IoU is low: is it bad depth/pose prediction,
or a bad mask-refinement head?

Normally WorldModelV2 predicts depth AND pose, warps voxel(t-1) -> t
with them, turns the photometric residual into a mask, and that mask
is what gets compared to GT for IoU. If IoU is stuck near-zero, you
can't tell from that number alone whether the geometry (depth/pose) or
the mask head is the broken part.

This script re-runs evaluation twice per checkpoint:
    1. "predicted" -- normal forward pass (predicted depth used for warp)
    2. "gt_depth"  -- ground-truth EVIMO2 depth substituted for the
                      predicted depth in the warp/residual step, while
                      pose is STILL predicted by the model (see note below)

If "gt_depth" IoU is much higher than "predicted" IoU, your depth
prediction is a major bottleneck. If it's barely different, the
problem is downstream: either pose, or the residual->mask refinement
head itself (in which case --gt-mask-sanity is the more useful tool).

IMPORTANT — scale mismatch
---------------------------
The model's predicted depth is UNSUPERVISED and arbitrary-scale
(DepthHead starts at 1.0, has no metric grounding). Predicted pose
is also arbitrary-scale (jointly consistent with predicted depth,
not with metric GT depth). EVIMO2 ground-truth depth is METRIC
(meters). Substituting metric GT depth while keeping the model's
own arbitrary-scale predicted pose can introduce a scale mismatch
in the warp (translation magnitude no longer matches depth
magnitude), which can make the "gt_depth" result look WORSE than
it should, not better.

To sanity-check this, the script also reports a "gt_depth_scaled"
variant that rescales GT depth per-sample to match the predicted
depth's median (crude, but removes the global scale ambiguity so
you're testing depth SHAPE/accuracy, not absolute scale).

Usage
-----
python evaluate_with_gt_depth.py \\
    --dataset-root /path/to/dataset_root \\
    --checkpoint runs/exp_v2/checkpoints/epoch_030.pth \\
    --sensors left_camera --split val --subset imo --no-ema
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from trainer_v2 import TrainerV2, TrainConfigV2
from src.utils.metrics import SegmentationMetrics

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def build_gt_depth_tensor(raw_batch, target_hw, device):
    """
    Stack per-sample GT depth (meters, (H,W), may contain None) from
    every frame in the temporal window into (B, T, 1, h, w), resized
    to `target_hw`. Missing frames get filled with the batch's own
    median depth so the warp doesn't blow up on a stray None.

    Returns (gt_depth_tensor, valid_mask) where valid_mask (B,) is
    False for samples where the REFERENCE (last) frame had no GT depth
    at all -- those samples should be excluded from the IoU comparison
    just like the mask-only eval already excludes GT-mask-less samples.
    """
    T = len(raw_batch.frames)
    B = len(raw_batch.frames[0].depth)

    per_frame = []
    ref_valid = [True] * B

    for t, frame in enumerate(raw_batch.frames):
        frame_depths = []
        for i, d in enumerate(frame.depth):
            if d is None:
                frame_depths.append(None)
                if t == T - 1:
                    ref_valid[i] = False
            else:
                dt = torch.as_tensor(d, dtype=torch.float32, device=device)
                if dt.shape[-2:] != tuple(target_hw):
                    dt = F.interpolate(
                        dt.unsqueeze(0).unsqueeze(0), size=target_hw,
                        mode="nearest",
                    ).squeeze(0).squeeze(0)
                frame_depths.append(dt)
        per_frame.append(frame_depths)

    # Fill any None with the median of whatever's valid in that frame,
    # or 1.0 if the whole frame is empty (keeps shapes/warp finite;
    # those samples are excluded from metrics via ref_valid anyway).
    stacked = torch.zeros(B, T, 1, *target_hw, device=device)
    for t in range(T):
        valid_vals = [d for d in per_frame[t] if d is not None]
        fallback = torch.stack(valid_vals).median(dim=0).values if valid_vals else torch.ones(target_hw, device=device)
        for i in range(B):
            d = per_frame[t][i]
            stacked[i, t, 0] = d if d is not None else fallback

    valid_mask = torch.tensor(ref_valid, device=device)
    return stacked, valid_mask


def rescale_to_match_predicted(gt_depth, pred_depth):
    """
    Crude per-sample global scale correction: multiply GT depth so its
    median matches the median of the model's own predicted depth
    (evaluated at the reference frame). Removes scale ambiguity so
    the comparison probes depth SHAPE, not absolute metric scale.
    gt_depth, pred_depth: (B, T, 1, h, w) and (B, 1, h, w) respectively,
    already resized to the same (h, w).
    """
    B = gt_depth.shape[0]
    gt_ref = gt_depth[:, -1]  # (B, 1, h, w)
    gt_med = gt_ref.flatten(1).median(dim=1).values.clamp_min(1e-6)  # (B,)
    pred_med = pred_depth.flatten(1).median(dim=1).values.clamp_min(1e-6)  # (B,)
    scale = (pred_med / gt_med).view(B, 1, 1, 1, 1)
    return gt_depth * scale


@torch.no_grad()
def evaluate(model, loader, transform, device, mode, ema=None):
    """
    mode: "predicted" | "gt_depth" | "gt_depth_scaled"
    """
    model.eval()
    metrics = SegmentationMetrics(THRESHOLDS)
    n = 0

    for raw in loader:
        vb = transform(raw)
        vox = torch.stack([f.voxel_grid for f in vb.frames], dim=1).to(device)
        vb = vb.to(device)

        if mode == "predicted":
            out = model(vox, vb)
        else:
            # First pass: get the model's predicted depth at its native
            # resolution so we know target_hw for resizing GT depth,
            # and (for gt_depth_scaled) the reference depth to match scale to.
            probe = model(vox, vb)
            low_hw = probe["depths"].shape[-2:]
            gt_depth, valid_ref = build_gt_depth_tensor(raw, low_hw, device)

            if mode == "gt_depth_scaled":
                gt_depth = rescale_to_match_predicted(gt_depth, probe["depth"])

            out = model(vox, vb, gt_depths=gt_depth)

        probs = torch.sigmoid(out["mask"])
        gts = raw.frames[-1].mask
        fms = raw.frames[-1].frame_motion
        valid = [(p, g, fm) for p, g, fm in zip(probs, gts, fms) if g is not None]
        if not valid:
            continue
        vp = torch.stack([p for p, _, _ in valid])
        vg = [g for _, g, _ in valid]
        vf = [fm for _, _, fm in valid]
        if vp.shape[-2:] != vg[0].shape:
            vp = F.interpolate(vp, size=vg[0].shape, mode="bilinear", align_corners=False)
        metrics.update(vp, vg, frame_motions=vf)
        n += len(valid)

    r = metrics.compute()
    return r, n


def main():
    p = argparse.ArgumentParser(description="Evaluate IoU with GT depth substituted for predicted depth")
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sensors", type=str, nargs="+", default=["left_camera"])
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--subset", type=str, default="imo")
    p.add_argument("--sequence", type=str, nargs="+", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--history-offsets", type=int, nargs="+", default=[-12, -8, -4, 0])
    p.add_argument("--num-bins", type=int, default=5)
    p.add_argument("--overfit", action="store_true",
                    help="Evaluate on the train loader (matches trainer_v2 --overfit convention).")
    p.add_argument("--no-ema", action="store_true")
    args = p.parse_args()

    cfg = TrainConfigV2(
        dataset_root=args.dataset_root, sensors=tuple(args.sensors),
        split=args.split if not args.overfit else "train",
        val_split=None,
        subset=args.subset,
        sequence=tuple(args.sequence) if args.sequence else None,
        history_offsets=tuple(args.history_offsets),
        num_bins=args.num_bins,
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_ema=not args.no_ema,
        overfit_mode=args.overfit,
    )
    trainer = TrainerV2(cfg)

    ckpt = torch.load(args.checkpoint, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    loader = trainer.train_loader if args.overfit else (trainer.val_loader or trainer.train_loader)

    results = {}
    for mode in ["predicted", "gt_depth", "gt_depth_scaled"]:
        r, n = evaluate(trainer.model, loader, trainer.transform, trainer.device, mode)
        results[mode] = r
        logger.info(
            f"[{mode:16s}] n={n:5d} IoU={r['best_iou']:.4f} F1={r['best_f1']:.4f} "
            f"dr={r['pred_dynamic_ratio']:.4f}"
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"predicted depth+pose  IoU: {results['predicted']['best_iou']:.4f}")
    logger.info(f"GT depth, pred pose   IoU: {results['gt_depth']['best_iou']:.4f}")
    logger.info(f"GT depth (scale-matched), pred pose IoU: {results['gt_depth_scaled']['best_iou']:.4f}")
    logger.info("")
    logger.info("Interpretation:")
    logger.info(" - gt_depth/gt_depth_scaled >> predicted  -> depth prediction is the bottleneck")
    logger.info(" - all three similarly low                -> depth isn't (only) the problem;")
    logger.info("   suspect pose prediction and/or the mask-refinement head.")
    logger.info("   (pose is still PREDICTED in all three modes above --")
    logger.info("   this script only isolates depth, not pose.)")


if __name__ == "__main__":
    main()