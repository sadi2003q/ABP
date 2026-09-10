"""
Direct supervised loss for GTMaskModel.

Since geometry is ground truth, there is no photometric/self-
supervised objective here at all -- just a standard binary
segmentation loss against the real EVIMO2 dynamic-object mask:

    L = BCE-with-logits(mask, gt) + dice_weight * SoftDiceLoss(mask, gt)

BCE gives a well-behaved per-pixel gradient; Dice directly optimizes
overlap (IoU-like) and helps a lot when the dynamic mask is sparse
(most pixels are static background), which is the norm here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class GTMaskLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, pos_weight: float | None = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight

    def forward(self, mask_logits: torch.Tensor, gt_mask: torch.Tensor) -> dict:
        """
        mask_logits : (B, 1, H, W)
        gt_mask     : (B, 1, H, W) float in {0, 1}
        """
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=mask_logits.device, dtype=mask_logits.dtype)

        bce = F.binary_cross_entropy_with_logits(mask_logits, gt_mask, pos_weight=pos_weight)
        dice = soft_dice_loss(mask_logits, gt_mask)

        loss = self.bce_weight * bce + self.dice_weight * dice

        return {
            "loss": loss,
            "bce_loss": bce.detach(),
            "dice_loss": dice.detach(),
            "pred_dynamic_ratio": torch.sigmoid(mask_logits).mean().detach(),
            "gt_dynamic_ratio": gt_mask.mean().detach(),
        }
