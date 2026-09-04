"""
Edge-Aware Depth Smoothness Loss with Depth Normalization.

This loss regularizes the predicted depth map by encouraging locally
smooth depth while preserving sharp discontinuities at object boundaries
(where the input event voxel grid has strong gradients).

Two key improvements over the old unweighted TV:

1. DEPTH NORMALIZATION (Monodepth2-style)
   ----------------------------------------
   Before computing the smoothness, we normalize the depth:
       d_norm = d / (d.mean() + 1e-7)

   Without this, the network can trivially minimize the smoothness
   loss by shrinking all depths toward zero (since |d_x| → 0 as
   d → 0). The normalization makes the smoothness loss scale-
   invariant, so the network can't cheat by scaling depth down.

2. EDGE-AWARE WEIGHTING
   --------------------
   We weight the depth smoothness by the INVERSE of the input
   voxel gradient:
       weight(x,y) = exp(-|∇V(x,y)|)

   where V is the input event voxel grid (summed over bins).

   This means:
     - In SMOOTH regions (low event gradient): full smoothness
       penalty → depth is encouraged to be smooth
     - At EDGES (high event gradient): reduced smoothness penalty
       → depth is allowed to have discontinuities

   This is the standard Monodepth2 / SfMLearner recipe. The event
   voxel grid serves the same role as the intensity image in RGB SSL:
   it tells us WHERE object boundaries are.

Inputs
------
depth : (B, 1, H, W)
    Predicted depth map.
voxel_grid : (B, T, C, H, W) or (B, C, H, W)
    Input event voxel grid. If (B, T, C, H, W), we use the LAST
    frame's voxel (t = T-1) to match the depth (which is predicted
    for the last frame).
    Used to compute edge weights.

Returns
-------
dict with keys:
    loss            : scalar — edge-aware smooth depth TV
    smoothness_loss : scalar — unweighted smoothness (for logging)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthSmoothnessLoss(nn.Module):
    """
    Edge-aware depth smoothness with depth normalization.

    Parameters
    ----------
    weight : float
        Weight applied to the smoothness loss.
    edge_weight_decay : float
        Controls how strongly edges suppress the smoothness penalty.
        Higher = edges more strongly preserved (less smoothing at edges).
        The weight at each pixel is exp(-edge_weight_decay * |∇V|).
    """

    def __init__(
        self,
        weight: float = 1.0,
        edge_weight_decay: float = 1.0,
    ):
        super().__init__()
        self.weight = weight
        self.edge_weight_decay = edge_weight_decay

    def forward(
        self,
        depth: torch.Tensor,
        voxel_grid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        depth : (B, 1, H, W)
            Predicted depth.
        voxel_grid : (B, T, C, H_full, W_full) or (B, C, H, W) or None
            Input event voxel grid for edge-aware weighting.
            If None, falls back to UNWEIGHTED smoothness (old behavior).
            If (B, T, C, H, W), uses last frame and downsamples/upsamples
            to match depth resolution.

        Returns
        -------
        dict
        """

        if depth.ndim != 4:
            raise ValueError(
                "depth must have shape (B,1,H,W)"
            )

        # --------------------------------------------------
        # 1. DEPTH NORMALIZATION (Monodepth2-style)
        # --------------------------------------------------
        # d_norm = d / (d.mean() + eps)
        # This makes the smoothness loss scale-invariant.
        # Without it, the network can shrink depth to minimize |d_x|.
        # --------------------------------------------------
        depth_norm = depth / (depth.mean() + 1e-7)

        # --------------------------------------------------
        # 2. Compute depth gradients
        # --------------------------------------------------
        dx = torch.abs(
            depth_norm[:, :, :, 1:]
            - depth_norm[:, :, :, :-1]
        )  # (B, 1, H, W-1)

        dy = torch.abs(
            depth_norm[:, :, 1:, :]
            - depth_norm[:, :, :-1, :]
        )  # (B, 1, H-1, W)

        # --------------------------------------------------
        # 3. Edge-aware weighting (if voxel_grid is provided)
        # --------------------------------------------------
        if voxel_grid is not None:
            # Normalize voxel_grid shape to (B, C, H, W)
            if voxel_grid.ndim == 5:
                # (B, T, C, H, W) → use last frame
                voxel_grid = voxel_grid[:, -1]  # (B, C, H, W)

            B_v, C_v, H_v, W_v = voxel_grid.shape

            # Sum over bins to get a single-channel "intensity" image
            # (B, 1, H, W)
            voxel_intensity = voxel_grid.sum(dim=1, keepdim=True)

            # Match depth resolution
            if (H_v, W_v) != depth.shape[-2:]:
                voxel_intensity = F.interpolate(
                    voxel_intensity,
                    size=depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # Compute voxel gradients (same shape as dx, dy)
            voxel_dx = torch.abs(
                voxel_intensity[:, :, :, 1:]
                - voxel_intensity[:, :, :, :-1]
            )  # (B, 1, H, W-1)

            voxel_dy = torch.abs(
                voxel_intensity[:, :, 1:, :]
                - voxel_intensity[:, :, :-1, :]
            )  # (B, 1, H-1, W)

            # Edge weights: exp(-decay * |∇V|)
            # At smooth regions (|∇V| ≈ 0): weight ≈ 1 (full smoothing)
            # At edges (|∇V| large): weight ≈ 0 (no smoothing, preserve edge)
            wx = torch.exp(
                -self.edge_weight_decay * voxel_dx
            )  # (B, 1, H, W-1)

            wy = torch.exp(
                -self.edge_weight_decay * voxel_dy
            )  # (B, 1, H-1, W)

            # Apply weights
            dx_weighted = dx * wx
            dy_weighted = dy * wy

            smoothness_loss = (
                dx_weighted.mean()
                + dy_weighted.mean()
            )
        else:
            # Fallback: unweighted (old behavior)
            smoothness_loss = (
                dx.mean()
                + dy.mean()
            )

        total_loss = (
            self.weight
            *
            smoothness_loss
        )

        return {
            "loss": total_loss,
            "smoothness_loss": smoothness_loss,
        }
