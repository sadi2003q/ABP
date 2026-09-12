"""
Self-supervised PHOTOMETRIC loss for GTMaskModel, matching the actual
recipe already used (and validated) in this project's V2/V3 pipeline
-- see src/losses/total_loss_v2.py and
src/losses/dynamic_mask_regularization_loss.py.

Why this design, not a naive "explainability mask"
-----------------------------------------------------
The project's own PhotometricLoss (src/losses/photometric_loss.py)
supports an "explainability mask" that down-weights photometric loss
where the mask predicts "dynamic" (Monodepth2-style). Its docstring
records that this was tried and explicitly DISABLED: it let the mask
grow toward 100% to trivially zero out the photometric loss -- a
degenerate solution, not real learning.

What V2 actually does instead (TotalLossV2):
  1. Photometric loss trains depth+pose ONLY, using warped-voxel vs
     actual-voxel error (SSIM+L1), with NO mask weighting at all.
  2. The mask is trained SEPARATELY, via BCE against a PSEUDO-LABEL:
     the (detached) warp residual itself, clamped to [0,1]. Large
     residual = "this looks dynamic"; small residual = "this looks
     static". This never uses the real GT mask.
  3. A sparsity + confidence regularizer (matching
     DynamicMaskRegularizationLoss) prevents the mask from collapsing
     to all-static or drifting to all-dynamic, and pushes predictions
     toward confident 0/1 values instead of hovering at 0.5.

This file reimplements that same recipe for GTMaskModel. The one
structural difference: in V2, depth+pose are LEARNED, so photometric
loss has two jobs (teach geometry AND generate the mask's pseudo-
label). Here depth+pose are GROUND TRUTH, so there's no geometry left
to teach -- the warp residual GTMaskModel already computes (from
EXACT GT depth+pose) is a much cleaner, near-noise-free pseudo-label
than V2's residual ever could be, since it isn't corrupted by
imperfect learned geometry.

No real GT mask is used anywhere in this loss -- only the model's own
GT-geometry-derived residual and its own predicted mask. Real GT mask
is used ONLY for evaluation (see trainer), same as the mask-
consistency loss variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicMaskRegularization(nn.Module):
    """Sparsity + confidence regularizer, ported from
    src/losses/dynamic_mask_regularization_loss.py so this file has
    no import dependency on the V2 loss stack (keeps GTMaskModel's
    loss code self-contained, matching the rest of gt_mask_model/)."""

    def __init__(self, target_dynamic_ratio: float = 0.05,
                 sparsity_weight: float = 1.0, confidence_weight: float = 1.0):
        super().__init__()
        self.target_dynamic_ratio = target_dynamic_ratio
        self.sparsity_weight = sparsity_weight
        self.confidence_weight = confidence_weight

    def forward(self, mask_logits: torch.Tensor) -> dict:
        probs = torch.sigmoid(mask_logits)
        dynamic_ratio = probs.mean()
        sparsity_loss = (dynamic_ratio - self.target_dynamic_ratio).abs()
        confidence_loss = (probs * (1.0 - probs)).mean()
        loss = self.sparsity_weight * sparsity_loss + self.confidence_weight * confidence_loss
        return {
            "loss": loss, "sparsity_loss": sparsity_loss,
            "confidence_loss": confidence_loss, "dynamic_ratio": dynamic_ratio,
        }


class PhotometricGTMaskLoss(nn.Module):
    """
    Self-supervised loss for GTMaskModel: trains the mask head using
    ONLY the GT-geometry warp residual as a pseudo-label (never the
    real GT mask), plus sparsity/confidence regularization.

    Parameters
    ----------
    residual_weight : weight of the residual-pseudo-label BCE term
        (analogous to TotalLossV2's residual_mask_weight).
    sparsity_weight, confidence_weight : passed to
        DynamicMaskRegularization.
    target_dynamic_ratio : expected fraction of dynamic pixels
        (project convention: ~0.05 indoor scenes).
    """

    def __init__(
        self,
        residual_weight: float = 1.0,
        sparsity_weight: float = 5.0,
        confidence_weight: float = 1.0,
        target_dynamic_ratio: float = 0.05,
    ):
        super().__init__()
        self.residual_weight = residual_weight
        self.regularizer = DynamicMaskRegularization(
            target_dynamic_ratio=target_dynamic_ratio,
            sparsity_weight=sparsity_weight, confidence_weight=confidence_weight,
        )

    def forward(self, mask_logits: torch.Tensor, residual: torch.Tensor) -> dict:
        """
        mask_logits : (B,1,H,W) raw logits from GTMaskModel.forward()["mask"]
        residual : (B,1[+1],H,W) from GTMaskModel.forward()["residual"]
            (already detached by the model). If it has 2 channels
            (mask_extra_input="photometric"), both are averaged into
            one pseudo-label channel.
        """
        if residual.shape[1] > 1:
            residual = residual.mean(dim=1, keepdim=True)

        # Per-sample min-max normalize so the pseudo-label's scale is
        # comparable to a probability target, matching how the model
        # itself already normalizes the residual before its mask head
        # sees it (GTMaskModel._normalize_residual).
        B = residual.shape[0]
        flat = residual.flatten(1)
        r_min = flat.min(dim=1).values.view(B, 1, 1, 1)
        r_max = flat.max(dim=1).values.view(B, 1, 1, 1)
        pseudo_mask = ((residual - r_min) / (r_max - r_min + 1e-6)).clamp(0.0, 1.0)

        if pseudo_mask.shape[-2:] != mask_logits.shape[-2:]:
            pseudo_mask = F.interpolate(
                pseudo_mask, size=mask_logits.shape[-2:], mode="bilinear", align_corners=False,
            )

        residual_loss = F.binary_cross_entropy_with_logits(mask_logits, pseudo_mask)
        weighted_residual_loss = self.residual_weight * residual_loss

        reg = self.regularizer(mask_logits)

        loss = weighted_residual_loss + reg["loss"]

        return {
            "loss": loss,
            "residual_loss": residual_loss.detach(),
            "weighted_residual_loss": weighted_residual_loss.detach(),
            "sparsity_loss": reg["sparsity_loss"].detach(),
            "confidence_loss": reg["confidence_loss"].detach(),
            "pred_dynamic_ratio": reg["dynamic_ratio"].detach(),
            "pseudo_mask_mean": pseudo_mask.mean().detach(),
        }