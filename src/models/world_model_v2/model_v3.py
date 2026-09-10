"""
World Model v3 — IMU-Anchored Rotation.

v3 is v2 (direct-residual-mask architecture) with exactly ONE change:
the pose head's rotation is no longer a free network output. It is
composed as

    R_final = R_imu @ R_residual

where R_imu is computed by exact integration of raw gyroscope
samples (imu_integration.py — deterministic, not learned) and
R_residual is a small learned correction (PoseHeadV3).

Why: `evaluate_with_gt_depth.py` on v2 checkpoints showed
  - substituting GT depth barely changes IoU  -> depth isn't the bottleneck
  - substituting GT pose makes IoU catastrophically WORSE (0.0000)
This means v2's depth+pose jointly satisfy the photometric loss via
a self-consistent but physically arbitrary solution — the classic
monocular scale/rotation ambiguity. Nothing in v2 stops gradient
descent from "inventing" a rotation that minimizes reconstruction
error on this particular sequence. Anchoring rotation to a real
sensor measurement removes that degree of freedom: the network can
only apply a small correction on top of a real measurement, not
fabricate rotation wholesale.

`evaluate_with_gt_depth.py`'s --gt-mask-sanity result (IoU > 90%)
confirmed the mask/decoder head has plenty of capacity, so this file
deliberately reuses EventEncoder, DepthHead, MaskRefinementHead, and
the residual computation UNCHANGED from v2 — only pose changes.

What v3 does NOT yet address (by design, staged rollout):
  - Absolute translation/depth scale. Photometric loss is
    scale-invariant, so fixing rotation alone does not fix scale.
    The plan is to re-run evaluate_with_gt_depth.py against a v3
    checkpoint first, to confirm rotation-anchoring alone measurably
    helps, before adding a scale anchor (e.g. drift-aware
    accelerometer integration) as a follow-up. Shipping both at once
    would make it impossible to tell which change was responsible for
    any IoU delta — the exact ambiguity that cost multiple patch
    cycles in v1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.world_model.event_encoder import EventEncoder
from src.models.world_model.imu_encoder import IMUEncoder
from src.models.world_model.depth_head import DepthHead
from src.models.world_model_v2.pose_head_v3 import PoseHeadV3
from src.models.world_model.latent_renderer import LatentRenderer
from src.models.world_model_v2.mask_refinement_head import MaskRefinementHead
from src.models.world_model_v2.imu_integration import integrate_frame_rotation


class WorldModelV3(nn.Module):
    """Direct residual mask model with IMU-anchored rotation."""

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

        # Encoder (unchanged from v1/v2)
        self.event_encoder = EventEncoder(input_channels=num_bins)
        self.imu_encoder = IMUEncoder(
            hidden_channels=imu_hidden,
            embedding_dim=imu_embedding,
        )

        # Temporal fusion for depth (unchanged from v2)
        from src.models.world_model.temporal_memory import ConvGRUCell
        self.temporal_fusion = ConvGRUCell(
            input_channels=event_channels,
            hidden_channels=event_channels,
            kernel_size=3,
        )

        # Depth head (unchanged from v2)
        self.depth_head = DepthHead(input_channels=event_channels)

        # Pose head v3: rotation anchored to integrated gyro, only
        # translation + a small rotation correction are learned.
        self.pose_head = PoseHeadV3(
            event_channels=event_channels,
            imu_embedding_dim=imu_embedding,
        )

        # Renderer for warping voxels (unchanged from v2)
        self.renderer = LatentRenderer(rotation_type="6d")

        # Mask refinement (unchanged from v2 — confirmed >90% IoU
        # capacity via --gt-mask-sanity, so it is NOT touched here)
        self.mask_refinement = MaskRefinementHead(
            residual_channels=1,
            skip_channels=(128, 64, 32),
            hidden=32,
            out_channels=1,
        )

        self._init_weights()

    def _init_weights(self):
        """Same scaled Kaiming/Xavier init as v2. PoseHeadV3 and
        MaskRefinementHead already set their own final-layer init in
        their constructors, so we skip re-touching pose_head here
        (unlike v2, which overwrote it) to avoid clobbering the
        near-identity residual-rotation init PoseHeadV3 sets up."""
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
                # Skip the pose head's own final layer: PoseHeadV3.__init__
                # already initialized it deliberately (small residual-
                # rotation weights + identity bias). Re-running Xavier
                # init here would overwrite that on the pose head's
                # EARLIER linear layers too, which is fine (same as v2's
                # treatment of its hidden layers) — only the final layer
                # is protected.
                if module is self.pose_head.network[-1]:
                    continue
                nn.init.xavier_uniform_(module.weight)
                module.weight.data *= scale
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Mask refinement: bias=-3 (sparse mask init), same as v2.
        mask_final = self.mask_refinement.final
        if isinstance(mask_final, nn.Conv2d):
            nn.init.constant_(mask_final.bias, -3.0)

    def forward(self, voxel_batch: torch.Tensor, batch, gt_depths: torch.Tensor | None = None) -> dict:
        """
        Parameters
        ----------
        voxel_batch : (B, T, C, H, W)
        batch : TemporalEVIMO2Batch
        gt_depths : (B, T, 1, H_low, W_low) or None
            DIAGNOSTIC ONLY — identical semantics to v2 (see model_v2.py).

        Returns
        -------
        dict — same keys as v2, plus:
            R_imu : (B, T, 3, 3)  the integrated-gyro rotation prior
                actually used this forward pass (for logging/diagnosis).
        """
        B, T, C, H, W = voxel_batch.shape

        # === 1. Encode events ===
        event_pyramid = self.event_encoder(voxel_batch)
        event_features = event_pyramid[-1]  # (B, T, 256, H/16, W/16)

        # === 2. Encode IMU (learned embedding, as context) ===
        motion_embeddings = []
        for frame in batch.frames:
            embedding = self.imu_encoder(
                frame=frame,
                batch_size=B,
            )
            motion_embeddings.append(embedding)
        motion_embeddings = torch.stack(motion_embeddings, dim=1)  # (B, T, 128)

        # === 2b. Integrate IMU (exact rotation prior, NOT learned) ===
        # IMPORTANT: use frame.metadata.imu_timestamps (real seconds,
        # untouched by NormalizeIMU), NOT frame.imu_timestamps (which
        # has been rescaled to [0,1] per frame and would make the
        # integrated angle physically meaningless). See
        # imu_integration.integrate_frame_rotation's docstring.
        R_imu_list = []
        for frame in batch.frames:
            R_imu_t = integrate_frame_rotation(
                imu_timestamps=frame.metadata.imu_timestamps,
                imu_angular_velocity=frame.metadata.imu_angular_velocity,
                imu_sample_indices=frame.metadata.imu_sample_indices,
                batch_size=B,
                device=voxel_batch.device,
                dtype=torch.float32,
            )  # (B, 3, 3)
            R_imu_list.append(R_imu_t)
        R_imu = torch.stack(R_imu_list, dim=1)  # (B, T, 3, 3)

        # === 3. Temporal fusion + depth prediction (unchanged) ===
        hidden = None
        fused_features_list = []
        for t in range(T):
            feat_t = event_features[:, t]
            out = self.temporal_fusion(feat_t, hidden)
            hidden = out[1]
            fused_features_list.append(out[0])
        fused_features = torch.stack(fused_features_list, dim=1)  # (B, T, C, H, W)

        depths = self.depth_head(fused_features)  # (B, T, 1, H/16, W/16)
        depth = depths[:, -1]

        # === 4. Pose: IMU-anchored rotation + learned translation ===
        poses = self.pose_head(fused_features, motion_embeddings, R_imu)  # (B, T, 9)
        pose = poses[:, -1]

        # === 5. Compute K (unchanged) ===
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

        # === 6. Photometric residual (unchanged from v2) ===
        depths_for_residual = gt_depths if gt_depths is not None else depths
        residual_full = self._compute_residual(
            voxel_batch, depths_for_residual, poses, K_raw, distortion, H, W
        )

        # === 7. Mask refinement (unchanged from v2) ===
        skip_l1 = event_pyramid[0][:, -1]
        skip_l2 = event_pyramid[1][:, -1]
        skip_l3 = event_pyramid[2][:, -1]

        mask_logits = self.mask_refinement(
            residual=residual_full,
            skips=[skip_l3, skip_l2, skip_l1],
        )

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
            "R_imu": R_imu,
        }

    # _compute_residual and _invert_pose are IDENTICAL to v2 — reused
    # verbatim since the diagnostic showed this part of the pipeline
    # is not the bottleneck.

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
        B, T, C = voxel_batch.shape[:3]

        res_h = H // 4
        res_w = W // 4

        orig_w = 2.0 * K_raw[:, 0, 2]
        orig_h = 2.0 * K_raw[:, 1, 2]
        sx = orig_w / res_w
        sy = orig_h / res_h
        K_res = K_raw.clone()
        K_res[:, 0, 0] = K_raw[:, 0, 0] / sx
        K_res[:, 1, 1] = K_raw[:, 1, 1] / sy
        K_res[:, 0, 2] = K_raw[:, 0, 2] / sx
        K_res[:, 1, 2] = K_raw[:, 1, 2] / sy

        voxels_res = F.adaptive_avg_pool2d(
            voxel_batch.reshape(B * T, C, H, W),
            (res_h, res_w),
        ).reshape(B, T, C, res_h, res_w)

        flat = voxels_res.reshape(B * T, C, res_h, res_w)
        mean = flat.mean(dim=[1, 2, 3], keepdim=True)
        std = flat.std(dim=[1, 2, 3], keepdim=True)
        flat = (flat - mean) / (std + 1e-7)
        voxels_res = flat.reshape(B, T, C, res_h, res_w)

        H_low = depths.shape[-2]
        W_low = depths.shape[-1]
        depth_res = F.interpolate(
            depths.reshape(B * T, 1, H_low, W_low),
            size=(res_h, res_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, T, 1, res_h, res_w)

        t = T - 1
        voxel_prev = voxels_res[:, t - 1]
        voxel_curr = voxels_res[:, t]
        depth_t = depth_res[:, t]
        pose_t = poses[:, t]

        warped_fwd = self.renderer(
            feature=voxel_prev,
            depth=depth_t,
            pose=pose_t,
            K=K_res,
            distortion=distortion,
        )
        res_fwd = (warped_fwd - voxel_curr).abs().mean(dim=1, keepdim=True)

        depth_prev = depth_res[:, t - 1]
        pose_inv = self._invert_pose(pose_t)
        warped_bwd = self.renderer(
            feature=voxel_curr,
            depth=depth_prev,
            pose=pose_inv,
            K=K_res,
            distortion=distortion,
        )
        res_bwd = (warped_bwd - voxel_prev).abs().mean(dim=1, keepdim=True)

        residual_accum = torch.min(res_fwd, res_bwd)

        r_median = residual_accum.flatten(1).median(dim=1).values.view(B, 1, 1, 1)
        residual_norm = residual_accum / (r_median + 1e-6)

        residual_norm = torch.clamp(
            (residual_norm - 1.5) / 1.5,
            min=0.0,
            max=1.0,
        )
        residual_norm = residual_norm.pow(2.0)

        residual_full = F.interpolate(
            residual_norm,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        return residual_full

    @staticmethod
    def _invert_pose(pose: torch.Tensor) -> torch.Tensor:
        from src.losses.photometric_loss import invert_pose_6dof
        return invert_pose_6dof(pose)
