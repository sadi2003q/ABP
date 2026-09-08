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


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    q: (B, 4) in (x, y, z, w) order (EVIMO2 convention throughout this
    codebase). Returns (B, 3, 3) rotation matrices.
    """
    x, y, z, w = q.unbind(-1)
    B = q.shape[0]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack([
        1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx),
            2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy),
    ], dim=-1).reshape(B, 3, 3)
    return R


def rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    (B, 3, 3) -> (B, 6) using the same convention LatentRenderer expects:
    pose[3:6] = R[:, 0] (first COLUMN of R), pose[6:9] = R[:, 1] (second
    column) -- see LatentRenderer.pose_to_matrix, which builds
    R = stack([b1, b2, b3], dim=2), i.e. b1/b2 are columns of R.
    """
    a1 = R[:, :, 0]
    a2 = R[:, :, 1]
    return torch.cat([a1, a2], dim=-1)


def build_gt_pose_tensor(raw_batch, device):
    """
    Build the ground-truth RELATIVE pose for the reference pair
    (t-1 -> t) in the (B, 9) [tx,ty,tz, a1(3), a2(3)] layout
    WorldModelV2 predicts, from the absolute camera-to-world
    translation/quaternion in the last two frames' CameraMotion.

    LatentRenderer applies T directly to points backprojected in the
    TARGET camera's frame (using depth(t)) to sample into the SOURCE
    image (voxel(t-1)) -- i.e. T must map camera(t) coordinates into
    camera(t-1) coordinates (SfMLearner/Monodepth2 "T_{t-1<-t}"
    convention). With R_w2c = R_c2w^T for a camera-to-world rotation
    R_c2w, and p_world = R_i @ p_cam_i + t_i:

        p_cam_{t-1} = R_{t-1}^T @ (R_t @ p_cam_t + t_t - t_{t-1})
                    = (R_{t-1}^T @ R_t) @ p_cam_t + R_{t-1}^T @ (t_t - t_{t-1})

    So T = [R_rel | t_rel] with:
        R_rel = R_{t-1}^T @ R_t
        t_rel = R_{t-1}^T @ (t_t - t_{t-1})

    Returns (gt_pose, valid) where valid (B,) is False for samples
    where either frame lacks a valid GT pose (CameraMotion.pose_available).
    """
    frame_prev = raw_batch.frames[-2]
    frame_curr = raw_batch.frames[-1]
    B = len(frame_curr.camera_motion)

    t_prev = torch.stack([
        torch.as_tensor(cm.translation, dtype=torch.float32) for cm in frame_prev.camera_motion
    ]).to(device)
    t_curr = torch.stack([
        torch.as_tensor(cm.translation, dtype=torch.float32) for cm in frame_curr.camera_motion
    ]).to(device)
    q_prev = torch.stack([
        torch.as_tensor(cm.quaternion, dtype=torch.float32) for cm in frame_prev.camera_motion
    ]).to(device)
    q_curr = torch.stack([
        torch.as_tensor(cm.quaternion, dtype=torch.float32) for cm in frame_curr.camera_motion
    ]).to(device)

    valid = torch.tensor([
        bool(cm_p.pose_available) and bool(cm_c.pose_available)
        for cm_p, cm_c in zip(frame_prev.camera_motion, frame_curr.camera_motion)
    ], device=device)

    R_prev = quaternion_to_matrix(q_prev)  # (B,3,3)
    R_curr = quaternion_to_matrix(q_curr)

    R_rel = torch.bmm(R_prev.transpose(1, 2), R_curr)  # R_{t-1}^T @ R_t
    t_rel = torch.bmm(
        R_prev.transpose(1, 2), (t_curr - t_prev).unsqueeze(-1)
    ).squeeze(-1)  # R_{t-1}^T @ (t_t - t_{t-1})

    rot6d = rotation_matrix_to_6d(R_rel)  # (B, 6)
    gt_pose = torch.cat([t_rel, rot6d], dim=-1)  # (B, 9)

    return gt_pose, valid


def rescale_pose_translation_to_match_predicted(gt_pose, pred_pose):
    """
    Crude per-sample scale correction for GT pose translation:
    rescale ||t_gt|| to match ||t_pred|| (rotation is unaffected by
    scale so it's left as-is). Same rationale as
    rescale_to_match_predicted for depth: predicted pose/depth share
    an arbitrary, jointly-learned scale that GT metric values don't
    match, so a raw metric substitution can look artificially worse.
    gt_pose, pred_pose: (B, 9).
    """
    t_gt = gt_pose[:, :3]
    t_pred = pred_pose[:, :3]
    gt_norm = t_gt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    pred_norm = t_pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = pred_norm / gt_norm
    t_gt_scaled = t_gt * scale
    return torch.cat([t_gt_scaled, gt_pose[:, 3:]], dim=-1)


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
    mode: "predicted" | "gt_depth" | "gt_depth_scaled" |
          "gt_pose" | "gt_pose_scaled" |
          "gt_pose_depth" | "gt_pose_depth_scaled"
    """
    model.eval()
    metrics = SegmentationMetrics(THRESHOLDS)
    n = 0

    use_gt_depth = mode in ("gt_depth", "gt_depth_scaled", "gt_pose_depth", "gt_pose_depth_scaled")
    use_gt_pose = mode in ("gt_pose", "gt_pose_scaled", "gt_pose_depth", "gt_pose_depth_scaled")
    use_scaled = mode.endswith("_scaled")

    for raw in loader:
        vb = transform(raw)
        vox = torch.stack([f.voxel_grid for f in vb.frames], dim=1).to(device)
        vb = vb.to(device)

        if mode == "predicted":
            out = model(vox, vb)
        else:
            # First pass: get the model's predicted depth/pose so we
            # know target_hw for resizing GT depth, and (for *_scaled
            # modes) the reference values to match scale to.
            probe = model(vox, vb)

            gt_depths_arg = None
            if use_gt_depth:
                low_hw = probe["depths"].shape[-2:]
                gt_depth, _ = build_gt_depth_tensor(raw, low_hw, device)
                if use_scaled:
                    gt_depth = rescale_to_match_predicted(gt_depth, probe["depth"])
                gt_depths_arg = gt_depth

            gt_pose_arg = None
            if use_gt_pose:
                gt_pose, _ = build_gt_pose_tensor(raw, device)
                if use_scaled:
                    gt_pose = rescale_pose_translation_to_match_predicted(gt_pose, probe["pose"])
                gt_pose_arg = gt_pose

            out = model(vox, vb, gt_depths=gt_depths_arg, gt_pose=gt_pose_arg)

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

    modes = [
        "predicted",
        "gt_depth", "gt_depth_scaled",
        "gt_pose", "gt_pose_scaled",
        "gt_pose_depth", "gt_pose_depth_scaled",
    ]
    results = {}
    for mode in modes:
        r, n = evaluate(trainer.model, loader, trainer.transform, trainer.device, mode)
        results[mode] = r
        logger.info(
            f"[{mode:20s}] n={n:5d} IoU={r['best_iou']:.4f} F1={r['best_f1']:.4f} "
            f"dr={r['pred_dynamic_ratio']:.4f}"
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"predicted depth, predicted pose            IoU: {results['predicted']['best_iou']:.4f}")
    logger.info(f"GT depth,        predicted pose             IoU: {results['gt_depth']['best_iou']:.4f}")
    logger.info(f"GT depth (scaled), predicted pose            IoU: {results['gt_depth_scaled']['best_iou']:.4f}")
    logger.info(f"predicted depth, GT pose                    IoU: {results['gt_pose']['best_iou']:.4f}")
    logger.info(f"predicted depth, GT pose (scaled)           IoU: {results['gt_pose_scaled']['best_iou']:.4f}")
    logger.info(f"GT depth,        GT pose                    IoU: {results['gt_pose_depth']['best_iou']:.4f}")
    logger.info(f"GT depth (scaled), GT pose (scaled)         IoU: {results['gt_pose_depth_scaled']['best_iou']:.4f}")
    logger.info("")
    logger.info("Interpretation:")
    logger.info(" - gt_depth* >> predicted, gt_pose* ~= predicted  -> depth is the bottleneck")
    logger.info(" - gt_pose*  >> predicted, gt_depth* ~= predicted -> pose is the bottleneck")
    logger.info(" - gt_pose_depth* >> both individual gt_* results -> depth AND pose both matter")
    logger.info("   (their errors partially cancel/compound with each other)")
    logger.info(" - even gt_pose_depth_scaled stays near 'predicted' -> geometry (depth+pose)")
    logger.info("   isn't the bottleneck at all; suspect the mask-refinement head itself")
    logger.info("   (--gt-mask-sanity is the tool for that).")


if __name__ == "__main__":
    main()