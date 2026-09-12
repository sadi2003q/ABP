"""
Feature-space CONTRASTIVE loss for self-supervised motion segmentation
in GTMaskModel -- following the "unsupervised moving object
segmentation via motion consistency" line of work (Yang et al. and
successors), adapted here to use EXACT GT-geometry warping instead of
optical flow.

Why this differs from consistency_loss.py and photometric_gt_loss.py
-----------------------------------------------------------------------
- photometric_gt_loss.py: trains the MASK directly via BCE against the
  raw warp residual treated as a per-pixel pseudo-label. Weak signal
  in practice -- a single frame pair's residual is noisy (occlusion,
  event sparsity, sensor noise), so the mask head partly just learns
  the residual's noise statistics rather than true object boundaries.

- consistency_loss.py: trains the MASK via agreement between the
  model's own mask predictions at two consecutive frame pairs. Also a
  single, potentially noisy signal (both pairs derived from the same
  encoder, so it's possible for the mask head to find a self-
  consistent but wrong solution).

- THIS FILE: trains the ENCODER'S FEATURE SPACE directly using a
  contrastive/triplet objective, not the mask. The idea (matching the
  published unsupervised-motion-segmentation recipe): after an EXACT
  geometric warp, a static pixel's warped feature and its actual
  feature at the next frame are two views of "the same 3-D point" --
  they should be close in feature space (positive pair). A pixel
  where the warp is confidently wrong (large, spatially-coherent
  residual -- i.e. a real moving object, not warp/sensor noise) gives
  a feature pair that should NOT be forced close -- these are used as
  hard negatives in the triplet loss. The residual is used only to
  SELECT confident anchor/positive/negative examples (not as a direct
  per-pixel supervisory label), so a single noisy pixel can't directly
  poison the mask the way BCE-against-residual does; the loss instead
  shapes the whole feature space, which the mask head then reads off.

Confident-anchor selection
----------------------------
Rather than treating every pixel as a training example (which lets
per-pixel noise dominate), we rank pixels by warp-residual magnitude
per sample and select:
  - STATIC anchors: the bottom `static_percentile` fraction (most
    confidently well-explained by pure camera motion).
  - DYNAMIC anchors: the top `dynamic_percentile` fraction, but only
    among pixels forming spatially-coherent high-residual REGIONS
    (a single hot pixel is more likely noise than a real object; a
    contiguous patch of hot pixels is more likely a real object) --
    enforced with a cheap local-average smoothing of the residual
    before ranking.

Loss
-----
For each sample, sample N static anchors and M dynamic anchors:
  - Static anchor pixel p: pull warped_feat(p) and actual_feat(p)
    together (this IS the "positive pair" -- same 3-D point, two
    views). L_pos = ||warped_feat(p) - actual_feat(p)||^2
  - Dynamic anchor pixel q: push warped_feat(q) and actual_feat(q)
    apart, but only up to a margin (we don't want to blow up the
    embedding space arbitrarily -- we just want them clearly
    separated from the static cluster).
    L_neg = relu(margin - ||warped_feat(q) - actual_feat(q)||_2)

Total: L = mean(L_pos over static anchors) + neg_weight * mean(L_neg
over dynamic anchors).

The predicted MASK is trained separately (still self-supervised, no
GT) with a lightweight BCE against a per-pixel distance-based pseudo-
label derived from this now-better-shaped feature space: pixels whose
warped-vs-actual feature distance is large (post-training, once the
feature space has actually separated static/dynamic) are dynamic.
This mirrors how the published approach turns a learned embedding
into a segmentation: cluster/threshold in feature space, not directly
supervise pixels from a single frame's raw residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureContrastiveLoss(nn.Module):
    """
    Parameters
    ----------
    static_percentile : fraction (0-1) of lowest-residual pixels
        treated as confident static anchors.
    dynamic_percentile : fraction (0-1) of highest-residual pixels
        (after spatial smoothing) treated as confident dynamic anchors.
    smooth_kernel : odd int, size of the local-average smoothing
        window applied to the residual before ranking for dynamic
        anchors, to prefer spatially-coherent regions over lone noisy
        pixels.
    margin : minimum desired L2 distance between warped/actual
        features at dynamic anchors.
    neg_weight : weight of the dynamic (negative-pair) term relative
        to the static (positive-pair) term.
    mask_bce_weight : weight of the auxiliary mask-training BCE term
        (mask vs. normalized warped-actual feature distance, upsampled
        to mask resolution). Set to 0 to train ONLY the feature space
        and skip mask supervision entirely in this loss (useful if you
        want to inspect the feature-space effect in isolation first).
    """

    def __init__(
        self,
        static_percentile: float = 0.3,
        dynamic_percentile: float = 0.1,
        smooth_kernel: int = 3,
        margin: float = 1.0,
        neg_weight: float = 1.0,
        mask_bce_weight: float = 1.0,
    ):
        super().__init__()
        assert 0.0 < static_percentile < 1.0
        assert 0.0 < dynamic_percentile < 1.0
        self.static_percentile = static_percentile
        self.dynamic_percentile = dynamic_percentile
        self.smooth_kernel = smooth_kernel
        self.margin = margin
        self.neg_weight = neg_weight
        self.mask_bce_weight = mask_bce_weight

    def forward(
        self,
        warped_feat: torch.Tensor,
        feat_t1: torch.Tensor,
        mask_logits: torch.Tensor,
    ) -> dict:
        """
        warped_feat, feat_t1 : (B, C, Hf, Wf) -- feature-space warp
            output and actual target features, WITH gradients (from
            GTMaskModel.forward()'s "warped_feat"/"feat_t1" outputs).
        mask_logits : (B, 1, H, W) -- full-resolution mask prediction,
            from GTMaskModel.forward()'s "mask" output.
        """
        B, C, Hf, Wf = warped_feat.shape

        # Per-pixel feature distance (L2 over channels) -- this IS the
        # signal used both to pick anchors and (after smoothing) to
        # derive the mask's pseudo-label.
        dist = (warped_feat - feat_t1).norm(p=2, dim=1)  # (B, Hf, Wf)

        # Spatially smooth for RANKING only (dynamic-anchor selection
        # and mask pseudo-label), never for the raw per-pixel pull/push
        # terms themselves -- those use the exact per-pixel distance.
        pad = self.smooth_kernel // 2
        dist_smooth = F.avg_pool2d(
            dist.unsqueeze(1), self.smooth_kernel, stride=1, padding=pad, count_include_pad=False,
        ).squeeze(1)  # (B, Hf, Wf)

        flat_dist = dist.reshape(B, -1)
        flat_dist_smooth = dist_smooth.reshape(B, -1)
        n_pixels = flat_dist.shape[1]
        n_static = max(1, int(n_pixels * self.static_percentile))
        n_dynamic = max(1, int(n_pixels * self.dynamic_percentile))

        pos_losses = []
        neg_losses = []
        for b in range(B):
            static_idx = torch.topk(flat_dist[b], n_static, largest=False).indices
            dynamic_idx = torch.topk(flat_dist_smooth[b], n_dynamic, largest=True).indices

            pos_losses.append((flat_dist[b, static_idx] ** 2).mean())
            neg_losses.append(F.relu(self.margin - flat_dist[b, dynamic_idx]).mean())

        pos_loss = torch.stack(pos_losses).mean()
        neg_loss = torch.stack(neg_losses).mean()
        contrastive_loss = pos_loss + self.neg_weight * neg_loss

        dist_detached = dist.detach().unsqueeze(1)
        flat = dist_detached.flatten(1)
        d_min = flat.min(dim=1).values.view(B, 1, 1, 1)
        d_max = flat.max(dim=1).values.view(B, 1, 1, 1)
        pseudo_mask = ((dist_detached - d_min) / (d_max - d_min + 1e-6)).clamp(0.0, 1.0)
        if pseudo_mask.shape[-2:] != mask_logits.shape[-2:]:
            pseudo_mask = F.interpolate(
                pseudo_mask, size=mask_logits.shape[-2:], mode="bilinear", align_corners=False,
            )
        mask_bce = F.binary_cross_entropy_with_logits(mask_logits, pseudo_mask)
        weighted_mask_bce = self.mask_bce_weight * mask_bce

        loss = contrastive_loss + weighted_mask_bce

        return {
            "loss": loss,
            "pos_loss": pos_loss.detach(),
            "neg_loss": neg_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
            "mask_bce": mask_bce.detach(),
            "weighted_mask_bce": weighted_mask_bce.detach(),
            "pred_dynamic_ratio": torch.sigmoid(mask_logits).mean().detach(),
            "pseudo_mask_mean": pseudo_mask.mean().detach(),
        }