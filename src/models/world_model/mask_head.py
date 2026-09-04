"""
Dynamic object mask prediction head.

Predicts (per-pixel) whether each pixel belongs to a dynamic object.

Input
-----
(B, 16, H, W)

Output
------
(B, 1, H, W)

The output is RAW LOGITS (no sigmoid applied). This is the modern
convention because:

  1. Numerical stability: BCE-with-logits avoids log(0) when the
     sigmoid saturates to exactly 0 or 1.
  2. Autocast compatibility: torch.nn.functional.binary_cross_entropy
     is NOT safe to autocast (raises RuntimeError under bf16/fp16),
     but binary_cross_entropy_with_logits IS safe.
  3. Cleaner API: downstream consumers (metrics, visualization)
     explicitly call torch.sigmoid(logits) when they need
     probabilities, making the conversion explicit.

To get probabilities:
    probs = torch.sigmoid(model_output)

BIAS INITIALIZATION
-------------------
The final 1x1 conv's bias is initialized to -3.0, so that at random
init (when the conv weights produce ~0 output), the logit is ~-3
and sigmoid(-3) ≈ 0.047 ≈ 0.05.

This matches the target_dynamic_ratio (0.05) used by the sparsity
loss, giving the sparsity loss a "head start" — the mask starts
already sparse (predicting ~0 everywhere), and the residual loss
only needs to INCREASE logits where motion actually exists.

Without this initialization, sigmoid(0) = 0.5, so the mask starts
at 0.5 everywhere (50% dynamic). The sparsity loss then has to
"dig" the mask down from 0.5 to 0.05, which is slow and fights
against the residual loss (which may be pushing toward 0.5 due to
min-max normalization of the pseudo-label).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# Bias for the final conv so sigmoid(bias) ≈ 0.05 (the sparsity target).
# sigmoid(-3.0) = 0.0474 ≈ 0.05
MASK_BIAS_INIT = -3.0


class DynamicMaskHead(nn.Module):

    def __init__(
        self,
        in_channels: int = 16,
    ):

        super().__init__()

        self.head = nn.Sequential(

            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                min(16, in_channels),
                in_channels,
            ),

            nn.GELU(),

            nn.Conv2d(
                in_channels,
                1,
                kernel_size=1,
            ),

            # NO Sigmoid here -- output is logits.
            # See module docstring for rationale.
        )

        # --------------------------------------------------
        # Initialize the final conv's bias to -3 so sigmoid(-3) ≈ 0.05.
        # This gives the sparsity loss a head start: the mask starts
        # already sparse (predicting ~0 everywhere), and the residual
        # loss only needs to INCREASE logits where motion exists.
        #
        # The final layer is head[-1] (the 1x1 conv with 1 output channel).
        # --------------------------------------------------
        final_conv = self.head[-1]
        if isinstance(final_conv, nn.Conv2d):
            nn.init.constant_(final_conv.bias, MASK_BIAS_INIT)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.head(x)