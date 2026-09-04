"""
Total loss for the self-supervised event-camera world model.

This module provides a single loss interface for the complete world
model training objective.

The TotalLoss class receives the complete dictionary returned by
WorldModel.forward() and internally evaluates all individual loss
components.

## Current loss components

1. Latent consistency loss

   * Prediction loss
   * Rendering loss
   * Agreement loss

2. Depth smoothness loss

3. Pose temporal consistency loss

4. Depth temporal consistency loss

5. Dynamic mask regularization loss

   * Sparsity loss
   * Confidence loss

The individual losses remain implemented as separate modules. This
class only coordinates them and combines their weighted contributions.

This separation is intentional:

```
Individual loss modules
    -> define individual mathematical objectives

TotalLoss
    -> defines how those objectives are combined

Training loop
    -> performs optimization
```

This makes it possible to change loss weights or add new loss terms
without modifying the training loop.

## Expected model outputs

The model is expected to return a dictionary containing at least:

```
outputs["predicted_state"]
outputs["rendered_state"]
outputs["temporal_features"]

outputs["depth"]
outputs["depths"]

outputs["poses"]

outputs["mask"]
```

The TotalLoss class does not perform model inference itself.

It only consumes the already-computed model outputs.

## Loss weighting

The constructor accepts independent weights for every loss family:

```
latent_weight
depth_smoothness_weight
pose_temporal_weight
depth_temporal_weight
dynamic_mask_weight
```

The total objective is:

```
L_total =
    λ_latent      L_latent
  + λ_smooth      L_depth_smoothness
  + λ_pose        L_pose_temporal
  + λ_depth_temp  L_depth_temporal
  + λ_mask        L_dynamic_mask
```

All weights default to 1.0.

The weights are stored as ordinary Python floats rather than trainable
parameters because they define the training objective rather than
learned model parameters.
"""

from __future__ import annotations


import torch
import torch.nn as nn

from src.losses.latent_consistency_loss import LatentConsistencyLoss
from src.losses.depth_smoothness_loss import DepthSmoothnessLoss
from src.losses.pose_temporal_consistency_loss import (
PoseTemporalConsistencyLoss,
)
from src.losses.depth_temporal_consistency_loss import (
DepthTemporalConsistencyLoss,
)
from src.losses.dynamic_mask_regularization_loss import (
DynamicMaskRegularizationLoss,
)
from src.losses.dynamic_residual_loss import DynamicResidualLoss
from src.losses.photometric_loss import PhotometricLoss


class TotalLoss(nn.Module):
    """
    Combine all self-supervised world-model loss components.

    The individual loss modules return unweighted loss values.
    TotalLoss is responsible for applying the experiment-specific
    weights and computing the final training objective.

    Parameters
    ----------
    latent_weight : float
        Weight applied to the latent consistency objective.

    depth_smoothness_weight : float
        Weight applied to the depth smoothness objective.

    pose_temporal_weight : float
        Weight applied to pose temporal consistency.

    depth_temporal_weight : float
        Weight applied to depth temporal consistency.

    dynamic_mask_weight : float
        Weight applied to dynamic-mask regularization.

    Notes
    -----
    TotalLoss does not contain trainable model parameters.

    It is implemented as an nn.Module so that it behaves naturally
    alongside the model and individual loss modules in the training
    framework.

    The individual loss modules are deliberately kept independent
    of these experiment-level weights. This allows the same loss
    implementations to be reused across different experiments while
    changing only the relative importance of each objective here.
    """

    def __init__(
        self,
        prediction_loss_weight: float = 1.0,
        rendering_loss_weight: float = 1.0,
        agreement_loss_weight: float = 1.0,
        depth_smoothness_weight: float = 1.0,
        pose_temporal_weight: float = 1.0,
        depth_temporal_weight: float = 1.0,
        sparsity_loss_weight: float = 5.0,
        # Reduced from 1.0 to 0.1 — the confidence loss (p*(1-p)) was
        # fighting against the sparsity loss. With weight=1.0, the
        # confidence loss wants per-pixel extremes (0 or 1), which
        # combined with the residual loss pushing toward 0.5 results
        # in the mask settling at uniform ~0.4-0.5 (worst of both).
        # With weight=0.1, the sparsity loss dominates, allowing the
        # mask to be sparse (matching target_dynamic_ratio=0.05).
        confidence_loss_weight: float = 0.1,
        dynamic_ratio_weight: float = 0.0,
        # Reduced from 1.0 to 0.5 — when the residual is meaningful
        # (renderer converged), BCE-with-logits produces values ~0.5
        # which is comparable to the sparsity loss * 5.0 = 5 * 0.002 = 0.01.
        # With residual_loss_weight=0.5, the residual can still carve
        # out motion regions without overwhelming the sparsity prior.
        residual_loss_weight: float = 0.5,
        # Photometric loss weight — THE CORE SUPERVISION SIGNAL for
        # self-supervised depth+pose. This is what makes the renderer
        # converge. Without it, depth/pose have no gradient signal and
        # the residual loss is permanently noise.
        # Default 10.0 (not 1.0) because the latent loss can be shortcut
        # by the transition learning identity; the photometric loss must
        # DOMINATE to force depth+pose convergence.
        photometric_loss_weight: float = 10.0,
        ):
        super().__init__()

        # ------------------------------------------------------
        # Loss modules
        # ------------------------------------------------------

        self.latent_loss = LatentConsistencyLoss()

        self.depth_smoothness_loss = (
            DepthSmoothnessLoss()
        )

        self.pose_temporal_loss = (
            PoseTemporalConsistencyLoss()
        )

        self.depth_temporal_loss = (
            DepthTemporalConsistencyLoss()
        )

        self.dynamic_mask_loss = (
            DynamicMaskRegularizationLoss()
        )

        # Spatial pseudo-label loss -- the PRIMARY signal that
        # teaches the mask head WHERE dynamic objects are.
        # See src/losses/dynamic_residual_loss.py for details.
        self.residual_loss = DynamicResidualLoss()

        # Photometric event reconstruction loss — THE CORE SUPERVISION
        # for self-supervised depth+pose. Warps voxel(t-1) to t using
        # depth+pose, compares to observed voxel(t). This directly
        # supervises depth+pose from observations (can't be shortcut).
        self.photometric_loss = PhotometricLoss()

        # ------------------------------------------------------
        # Loss weights
        # ------------------------------------------------------

        
        self.prediction_loss_weight = float(
            prediction_loss_weight)
        self.rendering_loss_weight = float(
            rendering_loss_weight)
        self.agreement_loss_weight = float(
            agreement_loss_weight)

        self.depth_smoothness_weight = float(
            depth_smoothness_weight
        )

        self.pose_temporal_weight = float(
            pose_temporal_weight
        )

        self.depth_temporal_weight = float(
            depth_temporal_weight
        )

        self.sparsity_loss_weight = float(
            sparsity_loss_weight
        )
        self.confidence_loss_weight = float(
            confidence_loss_weight
        )
        self.dynamic_ratio_weight = float(
            dynamic_ratio_weight
        )
        self.residual_loss_weight = float(
            residual_loss_weight
        )
        self.photometric_loss_weight = float(
            photometric_loss_weight
        )


    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the complete weighted training objective.

        Parameters
        ----------
        outputs : dict[str, torch.Tensor]
            Dictionary returned by WorldModel.forward().
        inputs : dict[str, torch.Tensor] | None
            Optional dict with input tensors. If provided and contains
            "voxel_grid", the photometric loss and edge-aware depth
            smoothness will be computed.

            voxel_grid : (B, T, C, H, W) — input event voxels

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the final weighted objective,
            weighted individual contributions, and raw unweighted
            component losses.
        """

        # ======================================================
        # Latent consistency
        # ======================================================

        target_state = (
            outputs["temporal_features"][:, -1].detach()
        )

        latent = self.latent_loss(
            predicted_state=outputs["predicted_state"],
            rendered_state=outputs["rendered_state"],
            target_state=target_state,
        )

        # ------------------------------------------------------
        # Apply individual latent-loss weights
        # ------------------------------------------------------

        prediction_loss = (
            self.prediction_loss_weight
            * latent["prediction_loss"]
        )

        rendering_loss = (
            self.rendering_loss_weight
            * latent["rendering_loss"]
        )

        agreement_loss = (
            self.agreement_loss_weight
            * latent["agreement_loss"]
        )

        latent_loss = (
            prediction_loss
            + rendering_loss
            + agreement_loss
        )

        # ======================================================
        # Depth smoothness (edge-aware, with depth normalization)
        # ======================================================
        # If inputs contains voxel_grid, use it for edge-aware weighting.
        # Otherwise falls back to unweighted smoothness.
        voxel_grid = inputs.get("voxel_grid") if inputs else None

        depth_smoothness = (
            self.depth_smoothness_loss(
                outputs["depth"],
                voxel_grid=voxel_grid,
            )
        )

        depth_smoothness_loss = (
            self.depth_smoothness_weight
            * depth_smoothness["loss"]
        )

        # ======================================================
        # Photometric event reconstruction loss
        # ======================================================
        # THE CORE SUPERVISION SIGNAL for self-supervised depth+pose.
        # Warps voxel(t-1) to t using depth+pose, compares to observed
        # voxel(t). This directly supervises depth+pose from observations
        # and CANNOT be shortcut (unlike the latent consistency loss).
        #
        # Only computed if inputs contains voxel_grid.
        # ======================================================
        if voxel_grid is not None and self.photometric_loss_weight > 0:
            # Use K_original (unscaled) — the photometric loss scales K
            # internally for each resolution level
            K_for_photo = outputs.get("K_original", outputs["K"])

            # Get mask probabilities for explainability weighting.
            # DISABLED: the explainability mask caused the mask head
            # to grow toward 100% (degenerate solution). We pass
            # None so the photometric loss applies equally everywhere.
            mask_probs = None

            photo_out = self.photometric_loss(
                voxel_grid=voxel_grid,
                depths=outputs["depths"],
                poses=outputs["poses"],
                K=K_for_photo,
                distortion=outputs["distortion"],
                mask_probs=mask_probs,
            )
            photometric_loss = (
                self.photometric_loss_weight
                * photo_out["loss"]
            )
        else:
            photo_out = {"loss": torch.tensor(0.0, device=outputs["depth"].device)}
            photometric_loss = torch.tensor(0.0, device=outputs["depth"].device)

        # ======================================================
        # Pose temporal consistency
        # ======================================================

        pose_temporal = (
            self.pose_temporal_loss(
                outputs["poses"]
            )
        )

        pose_temporal_loss = (
            self.pose_temporal_weight
            * pose_temporal["loss"]
        )

        # ======================================================
        # Depth temporal consistency
        # ======================================================

        depth_temporal = (
            self.depth_temporal_loss(
                outputs["depths"]
            )
        )

        depth_temporal_loss = (
            self.depth_temporal_weight
            * depth_temporal["loss"]
        )

        # ======================================================
        # Dynamic mask regularization
        # ======================================================

        dynamic_mask = (
            self.dynamic_mask_loss(
                outputs["mask"]
            )
        )

        # ------------------------------------------------------
        # Apply individual dynamic-mask weights
        # ------------------------------------------------------

        sparsity_loss = (
            self.sparsity_loss_weight
            * dynamic_mask["sparsity_loss"]
        )

        confidence_loss = (
            self.confidence_loss_weight
            * dynamic_mask["confidence_loss"]
        )

        dynamic_ratio_loss = (
            self.dynamic_ratio_weight
            * dynamic_mask["dynamic_ratio"]
        )

        dynamic_mask_loss = (
            sparsity_loss
            + confidence_loss
            + dynamic_ratio_loss
        )

        # ======================================================
        # Dynamic residual pseudo-label loss (SPATIAL signal)
        # ======================================================
        # This is the PRIMARY supervision for the mask head.
        # It teaches the mask WHERE dynamic objects are by using
        # the rendering residual as a self-supervised pseudo-label.
        # See src/losses/dynamic_residual_loss.py for details.
        #
        # `mask` is now logits (no sigmoid in mask_head). The loss
        # uses binary_cross_entropy_with_logits internally.
        # ======================================================

        residual_out = self.residual_loss(
            mask=outputs["mask"],
            rendered_state=outputs["rendered_state"],
            target_state=target_state,  # already detached above
        )

        residual_loss = (
            self.residual_loss_weight
            * residual_out["loss"]
        )

        # ======================================================
        # Total weighted objective
        # ======================================================

        total_loss = (
            latent_loss
            + depth_smoothness_loss
            + pose_temporal_loss
            + depth_temporal_loss
            + dynamic_mask_loss
            + residual_loss
            + photometric_loss
        )

        # ======================================================
        # Return
        # ======================================================

        return {

            # --------------------------------------------------
            # Final weighted objective
            # --------------------------------------------------

            "loss": total_loss,

            # --------------------------------------------------
            # Weighted high-level contributions
            # --------------------------------------------------

            "latent_loss": latent_loss,

            "depth_smoothness_loss": (
                depth_smoothness_loss
            ),

            "pose_temporal_loss": (
                pose_temporal_loss
            ),

            "depth_temporal_loss": (
                depth_temporal_loss
            ),

            "dynamic_mask_loss": (
                dynamic_mask_loss
            ),

            # --------------------------------------------------
            # Raw latent components
            #
            # These are the ORIGINAL unweighted values.
            # --------------------------------------------------

            "prediction_loss": (
                latent["prediction_loss"]
            ),

            "rendering_loss": (
                latent["rendering_loss"]
            ),

            "agreement_loss": (
                latent["agreement_loss"]
            ),

            # --------------------------------------------------
            # Weighted latent components
            #
            # Useful for checking what actually contributes
            # to the optimization objective.
            # --------------------------------------------------

            "weighted_prediction_loss": (
                prediction_loss
            ),

            "weighted_rendering_loss": (
                rendering_loss
            ),

            "weighted_agreement_loss": (
                agreement_loss
            ),

            # --------------------------------------------------
            # Raw depth loss
            # --------------------------------------------------

            "depth_smoothness_loss": (
                depth_smoothness["loss"]
            ),

            # --------------------------------------------------
            # Raw pose loss
            # --------------------------------------------------

            "pose_temporal_loss": (
                pose_temporal["loss"]
            ),

            # --------------------------------------------------
            # Raw depth temporal loss
            # --------------------------------------------------

            "depth_temporal_loss": (
                depth_temporal["loss"]
            ),

            # --------------------------------------------------
            # Raw dynamic-mask components
            # --------------------------------------------------

            "mask_sparsity_loss": (
                dynamic_mask["sparsity_loss"]
            ),

            "mask_confidence_loss": (
                dynamic_mask["confidence_loss"]
            ),

            "dynamic_ratio": (
                dynamic_mask["dynamic_ratio"]
            ),

            # --------------------------------------------------
            # Weighted dynamic-mask components
            # --------------------------------------------------

            "weighted_sparsity_loss": (
                sparsity_loss
            ),

            "weighted_confidence_loss": (
                confidence_loss
            ),

            "weighted_dynamic_ratio_loss": (
                dynamic_ratio_loss
            ),

            # --------------------------------------------------
            # Residual pseudo-label loss (spatial mask signal)
            # --------------------------------------------------
            "residual_loss": residual_out["loss"],

            "weighted_residual_loss": residual_loss,

            "residual": residual_out["residual"],

            "pseudo_label": residual_out["pseudo_label"],

            # Sigmoid'd mask probabilities (for visualization + metrics)
            # The mask head now outputs logits; this is sigmoid(logits).
            "mask_probs": residual_out["mask_probs"],

            # Noise detection status (forwarded from DynamicResidualLoss)
            "is_noise": residual_out.get("is_noise", False),
            "pseudo_mean": residual_out.get("pseudo_mean", 0.0),

            # --------------------------------------------------
            # Photometric loss (the core depth+pose supervision)
            # --------------------------------------------------
            "photometric_loss": photo_out["loss"],
            "weighted_photometric_loss": photometric_loss,
        }
