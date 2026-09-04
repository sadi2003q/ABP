"""
Total Loss v2 — with depth diversity + relative residual.

Fixes the depth-collapse problem where depth stays constant
(allowing pose to compensate) and the mask collapses to zero.

Two new additions:
  1. DEPTH DIVERSITY LOSS — penalizes constant depth
     -depth_var = depth.std() / (depth.mean() + eps)
     -loss = max(0, target_diversity - depth_var)
     This REQUIRES depth to have spatial variation.

  2. The mask_refinement head already takes the residual as input,
     but the residual is ABSOLUTE. As depth+pose improve, the
     residual decreases everywhere. The mask can't distinguish
     dynamic from static because there's no contrast.

     Fix: in model_v2.py, we normalize the residual RELATIVE to
     the median (not min-max). This highlights pixels where the
     warp fails MORE than average.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.photometric_loss import PhotometricLoss
from src.losses.depth_smoothness_loss import DepthSmoothnessLoss
from src.losses.pose_temporal_consistency_loss import PoseTemporalConsistencyLoss
from src.losses.dynamic_mask_regularization_loss import DynamicMaskRegularizationLoss


class TotalLossV2(nn.Module):
    """Loss for v2 model with depth diversity."""

    def __init__(
        self,
        photometric_loss_weight: float = 10.0,
        depth_smoothness_weight: float = 1.0,
        pose_temporal_weight: float = 1.0,
        sparsity_loss_weight: float = 5.0,
        depth_diversity_weight: float = 5.0,
        target_depth_diversity: float = 0.3,
        residual_mask_weight: float = 0.5,
    ):
        super().__init__()

        self.photometric_loss = PhotometricLoss()
        self.depth_smoothness_loss = DepthSmoothnessLoss()
        self.pose_temporal_loss = PoseTemporalConsistencyLoss()
        self.dynamic_mask_loss = DynamicMaskRegularizationLoss(
            target_dynamic_ratio=0.05,
        )

        self.photometric_loss_weight = float(photometric_loss_weight)
        self.depth_smoothness_weight = float(depth_smoothness_weight)
        self.pose_temporal_weight = float(pose_temporal_weight)
        self.sparsity_loss_weight = float(sparsity_loss_weight)
        self.depth_diversity_weight = float(depth_diversity_weight)
        self.target_depth_diversity = float(target_depth_diversity)
        self.residual_mask_weight = float(residual_mask_weight)

    def forward(
        self,
        outputs: dict,
        inputs: dict | None = None,
    ) -> dict:
        voxel_grid = inputs.get("voxel_grid") if inputs else None
        device = outputs["depth"].device

        # === Photometric loss ===
        if voxel_grid is not None and self.photometric_loss_weight > 0:
            K_for_photo = outputs.get("K_original", outputs["K"])
            photo_out = self.photometric_loss(
                voxel_grid=voxel_grid,
                depths=outputs["depths"],
                poses=outputs["poses"],
                K=K_for_photo,
                distortion=outputs["distortion"],
                mask_probs=None,
            )
            photometric_loss = self.photometric_loss_weight * photo_out["loss"]
        else:
            photo_out = {"loss": torch.tensor(0.0, device=device)}
            photometric_loss = torch.tensor(0.0, device=device)

        # === Depth smoothness (edge-aware) ===
        depth_smoothness = self.depth_smoothness_loss(
            outputs["depth"], voxel_grid=voxel_grid,
        )
        depth_smoothness_loss = self.depth_smoothness_weight * depth_smoothness["loss"]

        # === DEPTH DIVERSITY (NEW) ===
        # Penalize constant depth. When depth is constant:
        #   std = 0 → diversity = 0 → loss = target
        # When depth has spatial variation:
        #   std > 0 → diversity > 0 → loss decreases
        #
        # This is CRITICAL because without it, the model can
        # minimize photometric loss with constant depth + learned
        # pose (a global affine warp), without ever learning
        # spatial depth structure.
        depth = outputs["depth"]  # (B, 1, H, W)
        depth_var = depth.std(dim=[1, 2, 3]) / (depth.mean(dim=[1, 2, 3]) + 1e-7)
        depth_diversity = depth_var.mean()  # scalar
        # Hinge loss: only penalize if diversity < target
        depth_diversity_loss = self.depth_diversity_weight * torch.relu(
            self.target_depth_diversity - depth_diversity
        )

        # === Pose temporal ===
        pose_temporal = self.pose_temporal_loss(outputs["poses"])
        pose_temporal_loss = self.pose_temporal_weight * pose_temporal["loss"]

        # === Mask sparsity ===
        dynamic_mask = self.dynamic_mask_loss(outputs["mask"])
        sparsity_loss = self.sparsity_loss_weight * dynamic_mask["sparsity_loss"]

        # === Residual -> mask pseudo-supervision ===
        # The model already exposes a detached residual_full. Use it as a
        # conservative pseudo-target, but keep the explicit sparsity prior
        # strong enough to prevent mask expansion.
        residual_mask_loss = torch.tensor(0.0, device=device)
        weighted_residual_mask_loss = residual_mask_loss
        residual = outputs.get("residual")

        if residual is not None:
            pseudo_mask = residual.detach().clamp(0.0, 1.0)

            if pseudo_mask.shape[-2:] != outputs["mask"].shape[-2:]:
                pseudo_mask = F.interpolate(
                    pseudo_mask,
                    size=outputs["mask"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            residual_mask_loss = F.binary_cross_entropy_with_logits(
                outputs["mask"],
                pseudo_mask,
            )
            weighted_residual_mask_loss = (
                self.residual_mask_weight * residual_mask_loss
            )

        # === Total ===
        total_loss = (
            photometric_loss
            + depth_smoothness_loss
            + depth_diversity_loss
            + pose_temporal_loss
            + sparsity_loss
            + weighted_residual_mask_loss
        )

        return {
            "loss": total_loss,
            "photometric_loss": photo_out["loss"],
            "weighted_photometric_loss": photometric_loss,
            "depth_smoothness_loss": depth_smoothness["loss"],
            "weighted_depth_smoothness_loss": depth_smoothness_loss,
            "depth_diversity": depth_diversity.detach(),
            "depth_diversity_loss": depth_diversity_loss,
            "pose_temporal_loss": pose_temporal["loss"],
            "weighted_pose_temporal_loss": pose_temporal_loss,
            "sparsity_loss": dynamic_mask["sparsity_loss"],
            "weighted_sparsity_loss": sparsity_loss,
            "residual_mask_loss": residual_mask_loss,
            "weighted_residual_mask_loss": weighted_residual_mask_loss,
            "dynamic_ratio": dynamic_mask["dynamic_ratio"],
            "mask_probs": outputs.get("mask_probs"),
        }
