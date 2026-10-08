"""
Loss for SupervisedMotionSegmenter.

    total = bce_weight   * BinaryCrossEntropy(mask, gt_mask, pos_weight)
          + dice_weight  * SoftDice(mask, gt_mask)            [batch-global]
          + depth_weight * ScaleInvariantLog(depth, gt_depth)   (helper)
          + pose_weight  * SmoothL1(pose, gt_relative_pose)     (helper)

Design notes
------------
* Dice is computed over the WHOLE batch (all valid pixels pooled),
  not averaged per sample. The evaluation metric (SegmentationMetrics)
  also pools true/false positives over all pixels, so this matches the
  thing being measured, and it stays well-defined when individual
  samples contain no moving object. If the batch contains no moving
  pixel at all, Dice is skipped for that batch (it would only measure
  false positives, which binary cross-entropy already does).
* Every term is computed in float32 even under mixed precision.
* Every term carries its own per-sample validity mask, so a sample
  missing ground-truth depth still contributes to the mask loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0,
                         weight: torch.Tensor | None = None) -> torch.Tensor:
    """Batch-global soft Dice. Pixels with weight 0 are ignored entirely."""
    probs = torch.sigmoid(logits)
    if weight is not None:
        probs = probs * weight
        target = target * weight
    intersection = (probs * target).sum()
    denominator = probs.sum() + target.sum()
    return 1.0 - (2.0 * intersection + eps) / (denominator + eps)


def scale_invariant_log_loss(
    pred_log: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, variance_focus: float = 0.85,
) -> torch.Tensor:
    """Eigen et al. 2014 / BTS formulation, over valid pixels only."""
    diff = pred_log[valid] - torch.log(gt[valid])
    if diff.numel() == 0:
        return pred_log.sum() * 0.0
    value = (diff ** 2).mean() - variance_focus * diff.mean() ** 2
    return torch.sqrt(value.clamp_min(1e-8))


class SupervisedLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
        depth_weight: float = 0.1,
        pose_weight: float = 0.1,
        translation_scale: float = 100.0,
        rotation_scale: float = 10.0,
        min_depth: float = 0.05,
        max_depth: float = 20.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.depth_weight = depth_weight
        self.pose_weight = pose_weight
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, outputs: dict, targets: dict) -> dict:
        mask_logits = outputs["mask"].float()
        device = mask_logits.device
        zero = mask_logits.sum() * 0.0

        result = {}

        # ---------------- main task: moving-object mask -------------------
        mv = targets["mask_valid"].to(device)
        if mv.any():
            logits = mask_logits[mv]
            gt = targets["gt_mask"].to(device)[mv].float()
            pw = None
            if self.pos_weight is not None:
                pw = torch.tensor(self.pos_weight, device=device, dtype=torch.float32)
            w = targets.get("mask_weight")
            if w is not None:
                # Pseudo-label training: weight-0 pixels are "unknown" and
                # take no part in either loss term.
                w = w.to(device)[mv].float()
                per_pixel = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pw, reduction="none")
                bce = (per_pixel * w).sum() / w.sum().clamp_min(1.0)
                result["ignored_fraction"] = (1.0 - w.mean()).detach()
            else:
                bce = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pw)
            if gt.sum() > 0:
                dice = batch_soft_dice_loss(logits, gt, weight=w)
            else:
                dice = zero
            result["pred_dynamic_ratio"] = torch.sigmoid(logits).mean().detach()
            result["gt_dynamic_ratio"] = gt.mean().detach()
        else:
            bce, dice = zero, zero
            result["pred_dynamic_ratio"] = zero.detach()
            result["gt_dynamic_ratio"] = zero.detach()

        mask_loss = self.bce_weight * bce + self.dice_weight * dice

        # ---------------- helper task: depth -------------------------------
        depth_loss = zero
        if self.depth_weight > 0 and outputs.get("depth_log") is not None and "gt_depth" in targets:
            dv = targets["depth_valid"].to(device)
            if dv.any():
                pred = outputs["depth_log"].float()[dv]
                gt_d = targets["gt_depth"].to(device).float()[dv]
                pixel_valid = (
                    torch.isfinite(gt_d) & (gt_d > self.min_depth) & (gt_d < self.max_depth)
                )
                depth_loss = scale_invariant_log_loss(pred, gt_d, pixel_valid)

        # ---------------- helper task: camera motion -----------------------
        pose_loss = zero
        if self.pose_weight > 0 and outputs.get("pose") is not None and "gt_pose" in targets:
            pv = targets["pose_valid"].to(device)
            if pv.any():
                pred = outputs["pose"].float()[pv]
                gt_p = targets["gt_pose"].to(device).float()[pv]
                scale = torch.tensor(
                    [self.translation_scale] * 3 + [self.rotation_scale] * 6,
                    device=device, dtype=torch.float32,
                )
                pose_loss = F.smooth_l1_loss(pred * scale, gt_p * scale, beta=0.1)

        total = mask_loss + self.depth_weight * depth_loss + self.pose_weight * pose_loss

        result.update({
            "loss": total,
            "mask_loss": mask_loss.detach(),
            "bce_loss": bce.detach(),
            "dice_loss": dice.detach(),
            "depth_loss": depth_loss.detach(),
            "pose_loss": pose_loss.detach(),
        })
        return result
