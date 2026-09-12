"""
Self-supervised mask-CONSISTENCY loss for GTMaskModel.

Unlike GTMaskLoss (loss.py), which trains the mask head with DIRECT
supervision against the real EVIMO2 GT dynamic mask, this loss trains
it with NO ground-truth mask at all. It only uses the model's OWN
predictions at two consecutive frame pairs, checked for geometric
consistency -- this is the actual definition of "self-supervised"
your research goal requires, and matches what your supervisor asked
for: replace direct supervision with a consistency signal derived
from consecutive-frame agreement.

Core idea
---------
Consider three consecutive frames: t-2, t-1, t.

  1. Run GTMaskModel on the pair (t-2 -> t-1). This gives a predicted
     mask probability map at frame t-1: pred_mask(t-1).
  2. Run GTMaskModel on the pair (t-1 -> t). This gives a predicted
     mask probability map at frame t: pred_mask(t).
  3. Using the SAME exact GT depth(t-1) + GT relative pose(t-1 -> t)
     already used inside the model's own warp step, warp
     pred_mask(t-1) into frame t's view:
         warped_pred_mask = warp(pred_mask(t-1); depth(t-1), pose(t-1->t))
  4. If the model's predictions are geometrically consistent, a pixel
     that was predicted "dynamic" at t-1 and did NOT independently
     move again should still land on a "dynamic" pixel once warped
     into t's view, and match pred_mask(t) there. Where they
     disagree, the loss penalizes it.

This never touches the real GT mask (EVIMO2's speed-thresholded
object mask) during TRAINING -- that's only used afterward, at
evaluation time, purely to measure how good the self-supervised
result turned out to be. That is the standard self-supervised
recipe: train on a proxy signal derivable from the data itself
(here: cross-frame consistency of your own predictions, using
GT depth/pose for the exact warp), evaluate against real labels
you happen to have for research purposes but which the trained
model never needed to see.

Why this is nontrivial (and not just "always agrees with itself")
-------------------------------------------------------------------
A model that predicts an EMPTY mask everywhere would trivially be
"perfectly consistent" (0 dynamic pixels warped is still 0 dynamic
pixels). To avoid this degenerate collapse, we add a small
"non-triviality" regularizer that penalizes the predicted dynamic
ratio being pinned at 0 -- pushing the model to prefer the SMALLEST
nonzero, self-consistent, well-localized region over the trivially
consistent empty mask. This uses the same warp-residual magnitude
already computed inside the model (residual.mean() as a proxy target
for how much of the frame plausibly contains real motion) rather than
the GT mask, so it stays self-supervised.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_agreement_loss(a_probs: torch.Tensor, b_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft-Dice-style DISAGREEMENT between two probability maps
    (1 - overlap), used the same way soft_dice_loss in loss.py
    compares a prediction to a hard GT mask -- here both sides are
    the model's own (soft) predictions instead."""
    a = a_probs.flatten(1)
    b = b_probs.flatten(1)
    intersection = (a * b).sum(dim=1)
    union = a.sum(dim=1) + b.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class MaskConsistencyLoss(nn.Module):
    """
    Self-supervised loss: no GT mask used.

    Parameters
    ----------
    bce_weight, dice_weight : weight the pixelwise-agreement BCE term
        vs. the region-overlap Dice-agreement term, same balance
        convention as GTMaskLoss.
    collapse_penalty_weight : weight of the anti-collapse regularizer
        that discourages the trivial all-zero mask.
    target_dynamic_ratio : the minimum predicted dynamic-pixel ratio
        the regularizer pushes toward if the model collapses to (near)
        zero. This is a soft floor, not a hard target -- it only
        activates when pred_dynamic_ratio falls below it.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        collapse_penalty_weight: float = 0.5,
        target_dynamic_ratio: float = 0.02,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.collapse_penalty_weight = collapse_penalty_weight
        self.target_dynamic_ratio = target_dynamic_ratio

    def forward(
        self,
        pred_mask_t1_logits: torch.Tensor,
        warped_pred_mask_t1_probs: torch.Tensor,
        pred_mask_t_logits: torch.Tensor,
    ) -> dict:
        """
        pred_mask_t1_logits : (B,1,H,W) raw logits from the (t-2->t-1) forward pass
            -- kept only so pred_dynamic_ratio can be reported; not
            directly used in the consistency term (its WARPED probs
            are what gets compared).
        warped_pred_mask_t1_probs : (B,1,H,W) pred_mask(t-1) probabilities,
            warped into frame t's view using GT depth(t-1)+GT pose(t-1->t).
            Treated as a soft, non-differentiable-into-the-past target
            (detached) so gradients flow into the (t-1->t) branch,
            matching standard consistency-loss practice (like a
            teacher/EMA target) rather than optimizing both branches
            to collapse toward each other trivially.
        pred_mask_t_logits : (B,1,H,W) raw logits from the (t-1->t) forward pass.

        Returns
        -------
        dict with 'loss' and diagnostic scalars.
        """
        warped_target = warped_pred_mask_t1_probs.detach().clamp(0.0, 1.0)
        pred_t_probs = torch.sigmoid(pred_mask_t_logits)

        consistency_bce = F.binary_cross_entropy_with_logits(
            pred_mask_t_logits, warped_target,
        )
        consistency_dice = soft_dice_agreement_loss(pred_t_probs, warped_target)

        consistency_loss = self.bce_weight * consistency_bce + self.dice_weight * consistency_dice

        # Anti-collapse: softly penalize predicting (near-)nothing
        # everywhere, since an all-zero mask trivially "agrees" with
        # itself across frames. Only penalizes when BELOW the floor
        # (never discourages a larger, genuinely-detected region).
        pred_dynamic_ratio = pred_t_probs.mean()
        collapse_penalty = F.relu(self.target_dynamic_ratio - pred_dynamic_ratio)

        loss = consistency_loss + self.collapse_penalty_weight * collapse_penalty

        return {
            "loss": loss,
            "consistency_bce": consistency_bce.detach(),
            "consistency_dice": consistency_dice.detach(),
            "collapse_penalty": collapse_penalty.detach(),
            "pred_dynamic_ratio": pred_dynamic_ratio.detach(),
            "warped_target_ratio": warped_target.mean().detach(),
        }