"""
Multi-scale Photometric Loss with Explainability Mask + Forward-Backward Consistency + SSIM.

Three key improvements over the previous version:

1. EXPLAINABILITY MASK (Monodeptsh2-style)
   ----------------------------------------
   Weight the per-pixel photometric loss by (1 - mask_probs).
   Where the mask is high (dynamic object), down-weight the photometric loss.
   Where the mask is low (static background), full photometric loss.

   This creates a feedback loop:
   - mask starts sparse (~0.05) → photo loss applies to 95% of pixels
   - photo loss teaches depth+pose on static regions
   - residual loss teaches mask where depth+pose fail
   - As mask improves, photo loss focuses on explainable regions

2. FORWARD-BACKWARD CONSISTENCY
   ------------------------------
   Compute the photometric loss in BOTH directions:
   - Forward: warp voxel(t-1) → t using depth(t) + pose(t)
   - Backward: warp voxel(t) → t-1 using depth(t-1) + inv_pose(t)

   Take the MIN per-pixel loss (allows the model to "give up" on
   hard regions in one direction).

   This prevents the identity-warp local minimum because:
   - Forward warp with identity: L_fwd = |voxel(t-1) - voxel(t)|
   - Backward warp with identity: L_bwd = |voxel(t) - voxel(t-1)|
   - Both have the same loss, but DIFFERENT gradients
   - The backward warp provides additional gradient signal to
     depth(t-1) and pose(t-1)

3. SSIM (Structural Similarity)
   ------------------------------
   Replace smooth_l1 with SSIM for the photometric comparison.
   SSIM is more sensitive to structural misalignment than L1:
   - L1: difference between "correct warp" and "identity warp" is tiny
   - SSIM: structural difference is amplified (local statistics change)

   We use a 3x3 window SSIM, averaged with L1 for robustness
   (Monodepth2 uses 0.85*SSIM + 0.15*L1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.world_model.latent_renderer import LatentRenderer


def invert_pose_6dof(pose: torch.Tensor) -> torch.Tensor:
    """Invert a 6-DoF pose [tx,ty,tz,rx,ry,rz].

    For SE(3) transform T = [R, t; 0, 1]:
        T_inv = [R^T, -R^T @ t; 0, 1]

    For axis-angle: inv_r = -r (same axis, opposite angle)
    For translation: inv_t = -R^T @ t
    """
    B = pose.shape[0]
    t = pose[:, :3]  # (B, 3)

    if pose.shape[1] == 9:
        # 6D rotation: build R, invert as R^T
        a1 = pose[:, 3:6]
        a2 = pose[:, 6:9]
        b1 = F.normalize(a1, p=2, dim=1, eps=1e-6)
        b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
        b2 = F.normalize(b2, p=2, dim=1, eps=1e-6)
        b3 = torch.cross(b1, b2, dim=1)
        R = torch.stack([b1, b2, b3], dim=2)  # (B, 3, 3)

        # Inverse translation: -R^T @ t
        inv_t = -torch.bmm(R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)

        # Inverse rotation in 6D: just swap a1, a2 (columns of R^T)
        # R_inv = R^T, whose columns are the ROWS of R
        # The 6D rep of R^T is [R^T[:,0], R^T[:,1]] = [R[0,:], R[1,:]]
        inv_a1 = R[:, 0, :]  # first row of R = first column of R^T
        inv_a2 = R[:, 1, :]  # second row of R = second column of R^T

        return torch.cat([inv_t, inv_a1, inv_a2], dim=1)

    else:
        # Axis-angle (legacy)
        r = pose[:, 3:]
        inv_r = -r
        theta = torch.norm(r, dim=1, keepdim=True)
        axis = r / (theta + 1e-8)
        x, y, z = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        R = torch.zeros(B, 3, 3, device=pose.device, dtype=pose.dtype)
        R[:, 0, 0:1] = cos + x * x * (1 - cos)
        R[:, 0, 1:2] = x * y * (1 - cos) - z * sin
        R[:, 0, 2:3] = x * z * (1 - cos) + y * sin
        R[:, 1, 0:1] = y * x * (1 - cos) + z * sin
        R[:, 1, 1:2] = cos + y * y * (1 - cos)
        R[:, 1, 2:3] = y * z * (1 - cos) - x * sin
        R[:, 2, 0:1] = z * x * (1 - cos) - y * sin
        R[:, 2, 1:2] = z * y * (1 - cos) + x * sin
        R[:, 2, 2:3] = cos + z * z * (1 - cos)
        inv_t = -torch.bmm(R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)
        return torch.cat([inv_t, inv_r], dim=1)


def ssim(x: torch.Tensor, y: torch.Tensor, win_size: int = 3) -> torch.Tensor:
    """Structural Similarity Index (SSIM) loss — sparse-aware.

    For event voxels (which are ~95% zero), standard SSIM gives
    SSIM(0,0)=1.0 (maximum loss) for all zero-zero pixel pairs.
    This dominates the loss and prevents the model from learning.

    Fix: use a VALIDITY MASK that only counts regions where at
    least one of x or y has non-zero values. In zero-zero regions,
    the loss is set to 0 (not 1).

    Returns (B, 1, H', W') — per-pixel SSIM loss (1 - SSIM)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pad = win_size // 2
    x_padded = F.pad(x, [pad, pad, pad, pad], mode="reflect")
    y_padded = F.pad(y, [pad, pad, pad, pad], mode="reflect")

    mu_x = F.avg_pool2d(x_padded, win_size, stride=1)
    mu_y = F.avg_pool2d(y_padded, win_size, stride=1)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(x_padded * x_padded, win_size, stride=1) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y_padded * y_padded, win_size, stride=1) - mu_y_sq
    sigma_xy = F.avg_pool2d(x_padded * y_padded, win_size, stride=1) - mu_xy

    ssim_val = (2 * mu_xy + C1) * (2 * sigma_xy + C2) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )

    loss = 1 - ssim_val  # (B, C, H', W')

    # --------------------------------------------------
    # SPARSE-AWARE MASKING: zero-out regions where both x and y
    # are zero (no events). In these regions, SSIM(0,0) = 1.0
    # (maximum loss) which dominates and prevents learning.
    # --------------------------------------------------
    # Compute validity: 1 where at least one of x or y is non-zero
    # Use a local sum to account for the SSIM window
    validity = (x_padded.abs().sum(dim=1, keepdim=True) +
                y_padded.abs().sum(dim=1, keepdim=True)) > 1e-6
    validity = F.avg_pool2d(
        validity.float(), win_size, stride=1
    ) > 0  # 1 if any pixel in window is non-zero

    # Apply: set loss to 0 in zero-zero regions
    loss = loss * validity.float()

    # Average across channels
    if loss.shape[1] > 1:
        loss = loss.mean(dim=1, keepdim=True)

    return loss  # (B, 1, H', W')


class PhotometricLoss(nn.Module):
    """Multi-scale photometric loss with explainability mask + fwd-bwd + SSIM."""

    def __init__(self):
        super().__init__()
        self.renderer = LatentRenderer()

    def forward(
        self,
        voxel_grid: torch.Tensor,
        depths: torch.Tensor,
        poses: torch.Tensor,
        K: torch.Tensor,
        distortion: torch.Tensor,
        mask_probs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        voxel_grid : (B, T, C, H, W)
        depths : (B, T, 1, H_low, W_low)
        poses : (B, T, 6)
        K : (B, 3, 3) — ORIGINAL resolution intrinsics
        distortion : (B, 4)
        mask_probs : (B, 1, H, W) or None
            Predicted mask probabilities (sigmoid of logits).
            Used as explainability mask: weight = (1 - mask_probs).
            If None, no masking (all pixels weighted equally).
        """
        B, T, C, H, W = voxel_grid.shape
        H_low = depths.shape[-2]
        W_low = depths.shape[-1]

        orig_w = 2.0 * K[:, 0, 2]
        orig_h = 2.0 * K[:, 1, 2]

        # Multi-scale: H/4, H/8, H/16
        scales = [
            (H // 4, W // 4, 4.0),
            (H // 8, W // 8, 8.0),
            (H_low, W_low, None),
        ]

        total_loss = torch.tensor(0.0, device=voxel_grid.device)
        per_scale_losses = {}

        for scale_h, scale_w, downsample_factor in scales:
            if scale_h < 4 or scale_w < 4:
                continue

            # Downsample voxel grid
            voxels_scale = F.adaptive_avg_pool2d(
                voxel_grid.reshape(B * T, C, H, W),
                (scale_h, scale_w),
            ).reshape(B, T, C, scale_h, scale_w)

            # Normalize
            flat = voxels_scale.reshape(B * T, C, scale_h, scale_w)
            mean = flat.mean(dim=[1, 2, 3], keepdim=True)
            std = flat.std(dim=[1, 2, 3], keepdim=True)
            flat = (flat - mean) / (std + 1e-7)
            voxels_scale = flat.reshape(B, T, C, scale_h, scale_w)

            # Upsample depth
            if downsample_factor is not None:
                depth_scale = F.interpolate(
                    depths.reshape(B * T, 1, H_low, W_low),
                    size=(scale_h, scale_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(B, T, 1, scale_h, scale_w)
            else:
                depth_scale = depths

            # Scale K
            scale_x = orig_w / scale_w
            scale_y = orig_h / scale_h
            K_scale = K.clone()
            K_scale[:, 0, 0] = K[:, 0, 0] / scale_x
            K_scale[:, 1, 1] = K[:, 1, 1] / scale_y
            K_scale[:, 0, 2] = K[:, 0, 2] / scale_x
            K_scale[:, 1, 2] = K[:, 1, 2] / scale_y

            # Downsample mask for explainability weighting
            if mask_probs is not None:
                mask_scale = F.interpolate(
                    mask_probs,
                    size=(scale_h, scale_w),
                    mode="bilinear",
                    align_corners=False,
                )
                # Explainability weight: 1 - mask (downweight dynamic regions)
                explain_weight = 1.0 - mask_scale
                # Detach so the photometric loss doesn't affect the mask
                # (the mask is supervised by the residual loss, not here)
                explain_weight = explain_weight.detach()
            else:
                explain_weight = None

            scale_loss = torch.tensor(0.0, device=voxel_grid.device)
            n_pairs = 0

            for t in range(1, T):
                # === FORWARD WARP: voxel(t-1) → t ===
                voxel_prev = voxels_scale[:, t - 1]
                voxel_curr = voxels_scale[:, t]
                depth_t = depth_scale[:, t]
                pose_t = poses[:, t]

                warped_fwd = self.renderer(
                    feature=voxel_prev,
                    depth=depth_t,
                    pose=pose_t,
                    K=K_scale,
                    distortion=distortion,
                )

                loss_fwd = self._compute_pair_loss(
                    warped_fwd, voxel_curr, explain_weight,
                )

                # === BACKWARD WARP: voxel(t) → t-1 ===
                depth_prev = depth_scale[:, t - 1]
                pose_inv = invert_pose_6dof(pose_t)

                warped_bwd = self.renderer(
                    feature=voxel_curr,
                    depth=depth_prev,
                    pose=pose_inv,
                    K=K_scale,
                    distortion=distortion,
                )

                loss_bwd = self._compute_pair_loss(
                    warped_bwd, voxel_prev, explain_weight,
                )

                # Take min per-pixel (Monodepth2 trick — allows giving up
                # on hard regions in one direction)
                pair_loss = torch.min(loss_fwd, loss_bwd)
                pair_loss = pair_loss.mean()

                scale_loss = scale_loss + pair_loss
                n_pairs += 1

            if n_pairs > 0:
                scale_loss = scale_loss / n_pairs

            if downsample_factor == 4.0:
                weight = 1.0
            elif downsample_factor == 8.0:
                weight = 0.5
            else:
                weight = 0.25

            total_loss = total_loss + weight * scale_loss
            per_scale_losses[f"scale_{scale_h}x{scale_w}"] = scale_loss.detach()

        return {
            "loss": total_loss,
            "per_scale": per_scale_losses,
        }

    def _compute_pair_loss(
        self,
        warped: torch.Tensor,
        target: torch.Tensor,
        explain_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute per-pixel photometric loss (sparse-aware SSIM + L1).

        Returns (B, 1, H, W) per-pixel loss.

        NOTE: The explainability mask is DISABLED by default
        (explain_weight=None). It caused the mask head to grow
        toward 100% to zero out the photometric loss — a degenerate
        solution. We keep the parameter for future use but don't
        use it in the current training.
        """
        # SSIM loss (sparse-aware — zero in regions with no events)
        ssim_loss = ssim(warped, target, win_size=3)
        # ssim already returns (B, 1, H, W) with sparse masking applied

        # L1 loss (per-pixel, also sparse-aware)
        l1_loss = (warped - target).abs().mean(dim=1, keepdim=True)
        # Mask L1 the same way: zero in regions with no events
        event_mask = (warped.abs().sum(dim=1, keepdim=True) +
                      target.abs().sum(dim=1, keepdim=True)) > 1e-6
        l1_loss = l1_loss * event_mask.float()

        # Combined: 0.85*SSIM + 0.15*L1 (Monodepth2 recipe)
        combined = 0.85 * ssim_loss + 0.15 * l1_loss

        # Explainability mask is DISABLED — see docstring
        # if explain_weight is not None:
        #     combined = combined * explain_weight

        return combined
