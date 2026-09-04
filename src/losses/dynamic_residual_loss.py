"""
Dynamic Residual Pseudo-Label Loss.

This is the PRIMARY spatial supervision for the mask head.

Core idea
---------
The LatentRenderer warps the predicted latent using predicted depth
and camera pose. For static background pixels, the warp correctly
aligns the latent with the observed current-frame latent, so the
residual is low. For independently moving objects, the camera-motion-
compensated warp fails (because object motion is not modeled), so
the residual is high.

We turn this residual into a soft pseudo-label in [0, 1] and train
the mask head to predict it via BCE-with-logits. The pseudo-label is
detached so gradients flow ONLY into the mask head, not into the
depth/pose/transition (which would create a shortcut where the model
minimizes the residual by predicting trivial depth).

Math
-----
    residual = |rendered_state - target_state|.mean(dim=C)
                                              # (B,1,H_low,W_low)
    r_min    = residual.amin(dim=[1,2,3], keepdim=True)   # per-sample
    r_max    = residual.amax(dim=[1,2,3], keepdim=True)   # per-sample
    pseudo   = (residual - r_min) / (r_max - r_min + eps)  # in [0,1]
    pseudo   = pseudo.detach()                # stop-gradient

    # Match mask resolution (decoder upsamples 16x)
    pseudo   = F.interpolate(pseudo, (H, W), mode="bilinear")
    pseudo   = pseudo.clamp(0, 1)             # bilinear can overshoot

    # BCE-with-logits (autocast-safe, numerically stable)
    L = F.binary_cross_entropy_with_logits(mask_logits, pseudo)

Per-sample min-max normalization makes the pseudo-label robust to
the absolute residual magnitude (which shrinks as training
progresses and the renderer improves). Only the RELATIVE
high-residual pixels are labeled dynamic.

Why the resolution mismatch is OK
---------------------------------
The residual lives at the latent resolution (H/16, W/16) by design:
the renderer operates there for efficiency and the depth is predicted
there too. The mask, however, is at full image resolution because
WorldDecoder upsamples 16x to produce sharp outputs.

This is NOT a problem because:
1. The pseudo-label is a noisy TEACHER, not the answer. A coarse
   teacher is fine; it just tells the mask head WHERE motion exists.
2. The conv stack (alignment -> temporal_memory -> decoder) is
   responsible for sharpening the mask using its learned features.
3. This is exactly the SfMLearner recipe: supervise at coarse
   resolution, predict at full resolution, let the decoder refine.

Important
---------
- `target_state` MUST be detached (stop-gradient) by the CALLER.
  We do NOT detach here to avoid double-detach confusion; the caller
  controls the stop-gradient.
- The pseudo-label is detached internally so the mask loss cannot
  backprop into the renderer/depth/pose (which would create a
  shortcut).
- `mask` is EXPECTED TO BE LOGITS (not probabilities). The mask head
  was updated to NOT apply sigmoid; BCE-with-logits handles the
  sigmoid internally. This is autocast-safe and numerically stable.
- This loss should be WARMED UP (e.g. disabled for the first ~1
  epoch, or ramped from 0 -> 1.0 over the first 10% of training)
  because at step 0 the residuals are pure noise. Implement the
  warmup in the trainer by scaling `residual_loss_weight`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicResidualLoss(nn.Module):
    """
    Self-supervised spatial mask loss via rendering residual.

    Parameters
    ----------
    eps : float
        Small constant for numerical stability in normalization.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        # If the pseudo-label's mean exceeds this, the residual is
        # considered "noise" (renderer hasn't converged) and the loss
        # is suppressed for this step. This prevents the residual loss
        # from corrupting the mask when the renderer is bad.
        # A meaningful pseudo-label (motion concentrated in few pixels)
        # has mean < 0.15. Noise (uniform residual) has mean ~0.33.
        noise_threshold: float = 0.20,
    ):
        super().__init__()
        self.eps = eps
        self.noise_threshold = noise_threshold

    def forward(
        self,
        mask: torch.Tensor,
        rendered_state: torch.Tensor,
        target_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        mask : (B, 1, H, W)
            Predicted mask LOGITS (NOT probabilities). The mask head
            was updated to output raw logits (no sigmoid).
        rendered_state : (B, C, H_low, W_low)
            Latent warped by LatentRenderer using depth + pose.
            H_low = H / 16, W_low = W / 16.
        target_state : (B, C, H_low, W_low)
            Encoded current-frame latent. Caller MUST detach this
            before passing (we do NOT detach here to avoid double-
            detach confusion; the caller controls the stop-gradient).

        Returns
        -------
        dict with keys:
            loss            : scalar tensor (may be 0 if residual is noise)
            residual        : (B,1,H_low,W_low) detached, for logging
            pseudo_label    : (B,1,H,W) detached, in [0,1], for logging
            mask_probs      : (B,1,H,W) detached, sigmoid(mask) for logging
            pseudo_mean     : scalar, mean of pseudo_label (for logging)
            is_noise        : bool, whether the residual was suppressed
        """

        if mask.ndim != 4:
            raise ValueError(
                f"Expected mask shape (B,1,H,W), got {tuple(mask.shape)}"
            )
        if rendered_state.shape != target_state.shape:
            raise ValueError(
                f"rendered_state {tuple(rendered_state.shape)} != "
                f"target_state {tuple(target_state.shape)}"
            )

        # --------------------------------------------------
        # Per-pixel residual: average over channel dimension
        # --------------------------------------------------
        residual = (
            rendered_state - target_state
        ).abs().mean(dim=1, keepdim=True)  # (B,1,H_low,W_low)

        # --------------------------------------------------
        # Per-sample min-max normalization -> [0, 1]
        # --------------------------------------------------
        r_min = residual.amin(
            dim=[1, 2, 3], keepdim=True
        )  # (B,1,1,1)
        r_max = residual.amax(
            dim=[1, 2, 3], keepdim=True
        )  # (B,1,1,1)
        pseudo_label = (residual - r_min) / (r_max - r_min + self.eps)

        # --------------------------------------------------
        # Sparsity-promoting power transform.
        # --------------------------------------------------
        pseudo_label = pseudo_label.pow(2)

        # Stop gradient on the pseudo-label
        pseudo_label = pseudo_label.detach()

        # --------------------------------------------------
        # Match mask spatial resolution
        # --------------------------------------------------
        mask_h, mask_w = mask.shape[-2], mask.shape[-1]
        res_h, res_w = pseudo_label.shape[-2], pseudo_label.shape[-1]

        if (mask_h != res_h) or (mask_w != res_w):
            pseudo_label = F.interpolate(
                pseudo_label,
                size=(mask_h, mask_w),
                mode="bilinear",
                align_corners=False,
            )
            pseudo_label = pseudo_label.clamp(min=0.0, max=1.0)

        # --------------------------------------------------
        # NOISE DETECTION: suppress loss if pseudo-label is uniform
        # --------------------------------------------------
        # When the renderer hasn't converged, the residual is noise:
        #   |warp(random) - obs| ~ uniform random
        # After min-max + power transform, this gives a pseudo-label
        # with mean ~0.33 (not sparse).
        #
        # A meaningful pseudo-label (real motion concentrated in few
        # pixels) has mean < 0.15.
        #
        # If the mean exceeds `noise_threshold`, we ZERO OUT the loss
        # for this step. This prevents the residual loss from corrupting
        # the mask (which was happening: dr kept rising, IoU collapsed).
        #
        # The trainer's `residual_loss_weight` warmup (0 -> 1 over 10%)
        # handles the first ~10% of training. This noise detection
        # handles the case where the renderer takes longer than 10%
        # to converge (which is common for self-supervised depth/pose).
        # --------------------------------------------------
        pseudo_mean = pseudo_label.mean().item()
        is_noise = pseudo_mean > self.noise_threshold

        if is_noise:
            # Suppress: don't update the mask head this step.
            # Return 0 loss (still differentiable in case anyone checks).
            loss = mask.sum() * 0.0  # zero but keeps graph connected
        else:
            # --------------------------------------------------
            # BCE-with-logits (autocast-safe, numerically stable)
            # --------------------------------------------------
            loss = F.binary_cross_entropy_with_logits(
                mask, pseudo_label, reduction="mean"
            )

        # --------------------------------------------------
        # For logging: compute mask probabilities
        # --------------------------------------------------
        mask_probs = torch.sigmoid(mask).detach()

        return {
            "loss": loss,
            "residual": residual.detach(),
            "pseudo_label": pseudo_label.detach(),
            "mask_probs": mask_probs,
            "pseudo_mean": pseudo_mean,
            "is_noise": is_noise,
        }
