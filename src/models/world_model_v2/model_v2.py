"""
World Model v2 — Direct Residual Mask.

Simplified architecture that directly computes the motion mask from
the photometric residual, instead of learning it through a complex
decoder pipeline.

Architecture
-------------
1. EventEncoder → pyramid features (l1, l2, l3, l4)
2. DepthHead(l4) → depth at H/16
3. PoseHead(imu) → pose
4. For each pair (t-1, t):
   a. Warp voxel(t-1) → t using depth(t) + pose(t) + K
   b. residual = |warped - voxel(t)|  (geometric, not learned)
5. MaskRefinementHead(residual, encoder_skips) → mask logits

Key difference from v1:
- The mask is computed DIRECTLY from the photometric residual
- No world transition, no latent renderer, no alignment, no temporal
  memory, no decoder
- The mask is a CONSEQUENCE of depth+pose quality, not a separate
  learned output
- 10x fewer parameters in the mask path

This prevents:
- Identity shortcut (mask from learned features that can be anything)
- Circular dependency (mask weighting photometric loss that trains mask)
- Degenerate solutions (mask growing to 100% to zero out photometric)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.world_model.event_encoder import EventEncoder
from src.models.world_model.imu_encoder import IMUEncoder
from src.models.world_model.depth_head import DepthHead
from src.models.world_model_v2.pose_head_v2 import PoseHeadV2
from src.models.world_model.latent_renderer import LatentRenderer
from src.models.world_model_v2.mask_refinement_head import MaskRefinementHead


class WorldModelV2(nn.Module):
    """Direct residual mask model with temporal fusion."""

    def __init__(
        self,
        num_bins: int = 5,
        event_channels: int = 256,
        imu_hidden: int = 64,
        imu_embedding: int = 128,
        memory_type: str = "transformer",
    ):
        super().__init__()

        self.num_bins = num_bins

        # Encoder (same as v1)
        self.event_encoder = EventEncoder(input_channels=num_bins)
        self.imu_encoder = IMUEncoder(
            hidden_channels=imu_hidden,
            embedding_dim=imu_embedding,
        )

        # Temporal fusion for depth: ConvGRU that processes the
        # encoder features sequentially, giving depth temporal context.
        # Without this, each frame's depth is predicted independently,
        # causing temporal jitter and inconsistent warps.
        from src.models.world_model.temporal_memory import ConvGRUCell
        self.temporal_fusion = ConvGRUCell(
            input_channels=event_channels,
            hidden_channels=event_channels,
            kernel_size=3,
        )

        # Depth head (from TEMPORALLY FUSED features)
        self.depth_head = DepthHead(input_channels=event_channels)

        # Pose head v2: FUSES event features + IMU embeddings
        self.pose_head = PoseHeadV2(
            event_channels=event_channels,
            imu_embedding_dim=imu_embedding,
        )

        # Renderer for warping voxels
        self.renderer = LatentRenderer(rotation_type="6d")

        # Mask refinement (NEW — small, replaces decoder + mask_head + temporal_memory)
        self.mask_refinement = MaskRefinementHead(
            residual_channels=1,
            skip_channels=(128, 64, 32),  # l3, l2, l1 from encoder
            hidden=32,
            out_channels=1,
        )

        # Kaiming/Xavier init
        self._init_weights()

    def _init_weights(self):
        """Apply scaled Kaiming/Xavier initialization.

        Using a scale factor of 0.1 on top of Kaiming to prevent
        exploding gradients in the first forward pass (the 6D rotation
        + multi-scale photometric loss amplifies large init values).
        """
        scale = 0.1
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
                module.weight.data *= scale
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                module.weight.data *= scale
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Pose head: initialize translation at zero and 6D rotation at identity.
        pose_final = self.pose_head.network[-1]
        if isinstance(pose_final, nn.Linear):
            nn.init.zeros_(pose_final.bias)
            with torch.no_grad():
                pose_final.bias[0:3] = 0.0
                pose_final.bias[3:6] = torch.tensor(
                    [1.0, 0.0, 0.0], device=pose_final.bias.device
                )
                pose_final.bias[6:9] = torch.tensor(
                    [0.0, 1.0, 0.0], device=pose_final.bias.device
                )

        # Mask refinement: bias=-3 (sparse mask init)
        mask_final = self.mask_refinement.final
        if isinstance(mask_final, nn.Conv2d):
            nn.init.constant_(mask_final.bias, -3.0)

    def forward(self, voxel_batch: torch.Tensor, batch) -> dict:
        """
        Parameters
        ----------
        voxel_batch : (B, T, C, H, W)
            Input event voxel grids.
        batch : TemporalEVIMO2Batch
            Contains IMU data, camera intrinsics, etc.

        Returns
        -------
        dict with keys:
            depth, depths, poses, pose, K, K_original, distortion
            mask (logits), mask_probs, residual, warped
        """
        B, T, C, H, W = voxel_batch.shape

        # === 1. Encode events ===
        event_pyramid = self.event_encoder(voxel_batch)
        event_features = event_pyramid[-1]  # (B, T, 256, H/16, W/16)

        # === 2. Encode IMU ===
        motion_embeddings = []
        for frame in batch.frames:
            embedding = self.imu_encoder(
                frame=frame,
                batch_size=B,
            )
            motion_embeddings.append(embedding)
        motion_embeddings = torch.stack(motion_embeddings, dim=1)  # (B, T, 128)

        # === 3. Temporal fusion + depth prediction ===
        # Process encoder features through ConvGRU sequentially,
        # giving each frame's depth prediction access to temporal
        # context from previous frames. This prevents per-frame
        # depth jitter and produces temporally consistent warps.
        hidden = None
        fused_features_list = []
        for t in range(T):
            feat_t = event_features[:, t]  # (B, C, H, W)
            out = self.temporal_fusion(feat_t, hidden)
            hidden = out[1]  # GRU hidden state
            fused_features_list.append(out[0])  # h_new
        fused_features = torch.stack(fused_features_list, dim=1)  # (B, T, C, H, W)

        # Depth from TEMPORALLY FUSED features
        depths = self.depth_head(fused_features)  # (B, T, 1, H/16, W/16)
        depth = depths[:, -1]  # reference frame

        # === 4. Pose: FUSE temporally-fused event features + IMU ===
        # Pose ALSO uses temporally fused features — this gives the pose
        # head information about how the scene is evolving over time,
        # which helps estimate camera motion more accurately.
        # (SC-DepthPL and Monodepth2 both use multi-frame context for pose.)
        poses = self.pose_head(fused_features, motion_embeddings)  # (B, T, 9)
        pose = poses[:, -1]

        # === 5. Compute K (scaled to feature resolution) ===
        K_raw = torch.stack([
            torch.as_tensor(k, device=voxel_batch.device, dtype=torch.float32)
            for k in batch.frames[-1].camera_intrinsics
        ]).clone()

        orig_w = 2.0 * K_raw[:, 0, 2]
        orig_h = 2.0 * K_raw[:, 1, 2]
        feature_h = event_features.shape[-2]
        feature_w = event_features.shape[-1]
        scale_x = orig_w / feature_w
        scale_y = orig_h / feature_h

        K = K_raw.clone()
        K[:, 0, 0] = K[:, 0, 0] / scale_x
        K[:, 1, 1] = K[:, 1, 1] / scale_y
        K[:, 0, 2] = K[:, 0, 2] / scale_x
        K[:, 1, 2] = K[:, 1, 2] / scale_y

        distortion = torch.stack([
            torch.as_tensor(d, device=voxel_batch.device, dtype=torch.float32)
            for d in batch.frames[-1].camera_distortion
        ])

        # === 6. Compute multi-scale photometric residual ===
        # For each pair (t-1, t), warp voxel(t-1) → t and compute residual
        # We compute at H/4 (finest scale where we have depth after upsampling)
        residual_full = self._compute_residual(
            voxel_batch, depths, poses, K_raw, distortion, H, W
        )  # (B, 1, H, W)

        # === 7. Refine mask from residual + encoder features ===
        skip_l1 = event_pyramid[0][:, -1]  # (B, 32, H/2, W/2)
        skip_l2 = event_pyramid[1][:, -1]  # (B, 64, H/4, W/4)
        skip_l3 = event_pyramid[2][:, -1]  # (B, 128, H/8, W/8)

        mask_logits = self.mask_refinement(
            residual=residual_full,
            skips=[skip_l3, skip_l2, skip_l1],
        )  # (B, 1, H, W)

        mask_probs = torch.sigmoid(mask_logits).detach()

        return {
            "mask": mask_logits,
            "mask_probs": mask_probs,
            "residual": residual_full.detach(),
            "depth": depth,
            "depths": depths,
            "poses": poses,
            "pose": pose,
            "K": K,
            "K_original": K_raw,
            "distortion": distortion,
            "event_features": event_features,
            "event_pyramid": event_pyramid,
        }

    def _compute_residual(
        self,
        voxel_batch: torch.Tensor,
        depths: torch.Tensor,
        poses: torch.Tensor,
        K_raw: torch.Tensor,
        distortion: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Compute multi-scale photometric residual at full resolution.

        For each pair (t-1, t):
        1. Warp voxel(t-1) → t at H/4 using depth(t) + pose(t)
        2. residual = |warped - voxel(t)| (per-pixel, channel-averaged)
        3. Upsample to full resolution

        Returns (B, 1, H, W) — per-pixel residual, normalized per sample.
        """
        B, T, C = voxel_batch.shape[:3]
        device = voxel_batch.device

        # Target resolution for warping: H/4 (good balance of detail + cost)
        res_h = H // 4
        res_w = W // 4

        # Scale K from original to H/4
        orig_w = 2.0 * K_raw[:, 0, 2]
        orig_h = 2.0 * K_raw[:, 1, 2]
        sx = orig_w / res_w
        sy = orig_h / res_h
        K_res = K_raw.clone()
        K_res[:, 0, 0] = K_raw[:, 0, 0] / sx
        K_res[:, 1, 1] = K_raw[:, 1, 1] / sy
        K_res[:, 0, 2] = K_raw[:, 0, 2] / sx
        K_res[:, 1, 2] = K_raw[:, 1, 2] / sy

        # Downsample voxels to H/4
        voxels_res = F.adaptive_avg_pool2d(
            voxel_batch.reshape(B * T, C, H, W),
            (res_h, res_w),
        ).reshape(B, T, C, res_h, res_w)

        # Normalize (per-frame, zero-mean unit-variance)
        flat = voxels_res.reshape(B * T, C, res_h, res_w)
        mean = flat.mean(dim=[1, 2, 3], keepdim=True)
        std = flat.std(dim=[1, 2, 3], keepdim=True)
        flat = (flat - mean) / (std + 1e-7)
        voxels_res = flat.reshape(B, T, C, res_h, res_w)

        # Upsample depth from H/16 to H/4
        H_low = depths.shape[-2]
        W_low = depths.shape[-1]
        depth_res = F.interpolate(
            depths.reshape(B * T, 1, H_low, W_low),
            size=(res_h, res_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, T, 1, res_h, res_w)

        # Compute residual for ONLY the last pair (t=-1 → t=0)
        # This saves memory (only 2 renderer calls instead of 6)
        # and is sufficient for mask prediction — we only need to
        # know where the warp fails for the reference frame.
        t = T - 1  # last pair
        voxel_prev = voxels_res[:, t - 1]
        voxel_curr = voxels_res[:, t]
        depth_t = depth_res[:, t]
        pose_t = poses[:, t]

        # Forward warp: voxel(t-1) → t
        warped_fwd = self.renderer(
            feature=voxel_prev,
            depth=depth_t,
            pose=pose_t,
            K=K_res,
            distortion=distortion,
        )
        res_fwd = (warped_fwd - voxel_curr).abs().mean(
            dim=1, keepdim=True
        )

        # Backward warp: voxel(t) → t-1
        depth_prev = depth_res[:, t - 1]
        pose_inv = self._invert_pose(pose_t)
        warped_bwd = self.renderer(
            feature=voxel_curr,
            depth=depth_prev,
            pose=pose_inv,
            K=K_res,
            distortion=distortion,
        )
        res_bwd = (warped_bwd - voxel_prev).abs().mean(
            dim=1, keepdim=True
        )

        # Take min per-pixel
        residual_accum = torch.min(res_fwd, res_bwd)

        # --------------------------------------------------
        # MEDIAN-RELATIVE normalization (not min-max).
        #
        # Min-max normalization stretches the residual to [0,1],
        # but the MAX residual is usually an outlier (occlusion,
        # noise). This makes the normalization unstable.
        #
        # Instead, we normalize relative to the MEDIAN:
        #   residual_norm = residual / (median + eps)
        #
        # This means:
        # - Pixels with average residual → ~1.0 (normalized)
        # - Pixels with LOW residual (static, good warp) → ~0
        # - Pixels with HIGH residual (dynamic, bad warp) → >>1
        #
        # Then clip to [0, 1] and apply power transform for sparsity.
        # --------------------------------------------------
        # Robust per-sample median normalization.
        r_median = residual_accum.flatten(1).median(
            dim=1
        ).values.view(B, 1, 1, 1)
        residual_norm = residual_accum / (r_median + 1e-6)

        # Suppress weak residuals so early depth/pose errors do not
        # become a dense pseudo-mask. Strong violations are retained.
        residual_norm = torch.clamp(
            (residual_norm - 1.5) / 1.5,
            min=0.0,
            max=1.0,
        )

        # Make strong residuals more confident.
        residual_norm = residual_norm.pow(2.0)

        # Upsample to full resolution
        residual_full = F.interpolate(
            residual_norm,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        return residual_full

    @staticmethod
    def _invert_pose(pose: torch.Tensor) -> torch.Tensor:
        """Invert a 6-DoF pose [tx,ty,tz,rx,ry,rz]."""
        from src.losses.photometric_loss import invert_pose_6dof
        return invert_pose_6dof(pose)
