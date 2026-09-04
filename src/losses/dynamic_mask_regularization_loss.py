"""
Dynamic Mask Regularization Loss.

This loss provides regularization for the predicted dynamic segmentation
mask in the absence of pixel-level supervision.

The latent consistency losses already determine *where* dynamic objects
should appear because moving regions are required to explain the latent
representation over time.

This loss only prevents degenerate solutions by introducing two weak
priors on the predicted mask.

-----------------------------------------------------------------------
1. Sparsity Prior
-----------------------------------------------------------------------

Most pixels in an event-camera scene belong to the static background.

Rather than forcing every pixel toward zero, we encourage the average
dynamic probability to remain close to an expected dynamic ratio.

Let

    M(x,y) ∈ [0,1]

denote the predicted dynamic probability.

The average dynamic ratio is

                1
r = ---------------------- Σ M(x,y)
      (H × W)

The sparsity objective becomes

L_sparsity = (r - ρ)^2

where

ρ = expected proportion of dynamic pixels.

Typical values

    Indoor scenes
        ρ ≈ 0.05

    Driving scenes
        ρ ≈ 0.10

This prior is much more stable than minimizing mean(mask), since it
prevents collapse toward an all-static prediction while still encouraging
compact dynamic regions.

-----------------------------------------------------------------------
2. Confidence Prior
-----------------------------------------------------------------------

A segmentation network often produces uncertain probabilities around
0.5 during early training.

To encourage confident predictions we minimize

L_confidence = mean(M(1-M))

Properties

    M = 0      → loss = 0
    M = 1      → loss = 0
    M = 0.5    → maximum loss

This pushes the network toward confident binary masks while remaining
fully differentiable.

-----------------------------------------------------------------------
Total Loss
-----------------------------------------------------------------------

L_mask

    = λs L_sparsity

    + λc L_confidence

where

λs

    sparsity weight

λc

    confidence weight

-----------------------------------------------------------------------
Why this loss works
-----------------------------------------------------------------------

Notice that this loss NEVER tells the network which pixels are dynamic.

Instead,

LatentConsistencyLoss determines where motion must exist,

while this regularization simply encourages the resulting segmentation to

• occupy a realistic image area,

• avoid trivial all-static solutions,

• avoid noisy probabilistic masks,

• converge toward clean binary segmentations.

Therefore it acts purely as a weak Bayesian prior rather than a source
of supervision.

-----------------------------------------------------------------------
Inputs
-----------------------------------------------------------------------

mask

    Predicted dynamic probabilities

    Shape

        (B,1,H,W)

-----------------------------------------------------------------------
Returns
-----------------------------------------------------------------------

dict

    loss

        Total mask regularization loss.

    sparsity_loss

        Dynamic ratio prior.

    confidence_loss

        Binary confidence prior.

    dynamic_ratio

        Average predicted dynamic probability.

-----------------------------------------------------------------------
Example Usage
-----------------------------------------------------------------------

mask_losses = mask_regularization_loss(

    outputs["mask"],

)

total_loss = (

    latent_losses["loss"]

    +

    depth_smoothness_losses["loss"]

    +

    pose_temporal_losses

    +

    depth_temporal_losses["loss"]

    +

    mask_losses["loss"]

)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicMaskRegularizationLoss(nn.Module):
    """
    Weak regularization for the dynamic segmentation mask.

    IMPORTANT: After Patch Set 4, the mask head outputs LOGITS
    (no sigmoid). This loss must apply sigmoid internally before
    computing the sparsity/confidence regularizers, because:

      - `mean(probs)` should be in [0, 1] (a ratio of dynamic pixels)
      - `probs * (1 - probs)` is the binary confidence loss, which
        only makes sense for probabilities (max at p=0.5, min at
        p=0 or p=1).

    If we computed these on raw logits:
      - `mean(logits)` is unbounded (can be very negative)
      - `logits * (1 - logits) = logits - logits^2` is a downward
        parabola with minimum at logits -> +/- infinity, where it
        goes to -infinity. This would REWARD the network for pushing
        logits to extreme values, destroying all spatial structure.

    The fix: apply sigmoid at the start. The gradient still flows
    through sigmoid into the mask head's logits, so training works
    correctly.
    """

    def __init__(
        self,
        target_dynamic_ratio: float = 0.05,
        sparsity_weight: float = 1.0,
        confidence_weight: float = 1.0,
    ):
        super().__init__()

        self.target_dynamic_ratio = target_dynamic_ratio

        self.sparsity_weight = sparsity_weight

        self.confidence_weight = confidence_weight

    def forward(
        self,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        mask : torch.Tensor
            Predicted dynamic mask LOGITS (no sigmoid applied).
            Shape: (B, 1, H, W).

            We apply sigmoid internally to convert to probabilities
            before computing the regularizers. This ensures:
              - dynamic_ratio is in [0, 1] (a meaningful ratio)
              - confidence_loss = p*(1-p) is bounded in [0, 0.25]

        Returns
        -------
        dict
        """

        if mask.ndim != 4:

            raise ValueError(
                "Expected mask shape (B,1,H,W)"
            )

        # --------------------------------------------------
        # Convert logits -> probabilities.
        #
        # All regularization math below operates on `probs` (in [0,1]).
        # The sigmoid is differentiable, so gradients still flow
        # back into the mask head's logits correctly.
        # --------------------------------------------------
        probs = torch.sigmoid(mask)

        #
        # Estimated proportion of dynamic pixels.
        # This is now a meaningful ratio in [0, 1].
        #
        dynamic_ratio = probs.mean()

        #
        # Encourage approximately rho fraction
        # of pixels to be dynamic.
        #
        # Using L1 (abs) instead of L2 (squared) because the squared
        # version is too weak near the target. At dr=0.10 (2x target):
        #   L2: (0.10 - 0.05)^2 = 0.0025 (weak, residual loss wins)
        #   L1: |0.10 - 0.05|  = 0.0500 (20x stronger)
        #
        # With L1 and weight=5.0:
        #   dr=0.05 -> loss = 0.0  (no penalty, optimal)
        #   dr=0.10 -> loss = 0.25 (strong penalty)
        #   dr=0.20 -> loss = 0.75 (very strong)
        #
        # This makes the sparsity loss competitive with the residual
        # BCE loss (~0.4), preventing the residual from pushing dr
        # above 0.10.
        sparsity_loss = (
            dynamic_ratio
            - self.target_dynamic_ratio
        ).abs()

        #
        # Encourage confident predictions.
        #
        # For probabilities: p*(1-p) is bounded in [0, 0.25]:
        #   - p=0   -> 0   (confident static, no loss)
        #   - p=1   -> 0   (confident dynamic, no loss)
        #   - p=0.5 -> 0.25 (max uncertainty, max loss)
        #
        # This is what we want: penalize uncertain predictions
        # (probabilities near 0.5), reward confident ones (near 0 or 1).
        #
        confidence_loss = (
            probs * (1.0 - probs)
        ).mean()

        #
        # Total
        #
        loss = (

            self.sparsity_weight
            * sparsity_loss

            +

            self.confidence_weight
            * confidence_loss

        )

        return {

            "loss": loss,

            "sparsity_loss": sparsity_loss,

            "confidence_loss": confidence_loss,

            # Return the probability-based ratio (in [0,1]) for
            # meaningful logging. Previously this was mean(logits)
            # which could be negative and was meaningless.
            "dynamic_ratio": dynamic_ratio,

        }