"""
SupervisedMotionSegmenter — fully supervised moving-object segmentation
from event voxel grids (+ optional inertial measurement unit data).

What it needs at TEST time
--------------------------
Only what a real event camera rig provides:
    * event voxel grids for the last T frames
    * (optional) the inertial measurement unit window for those frames
No ground-truth depth, pose or mask is used in the forward pass, so the
model is deployable — unlike GTMaskModel (train_gt_mask.py), which
needs ground-truth depth and pose as INPUTS and is therefore an
upper-bound / oracle experiment only.

What it learns from at TRAIN time
---------------------------------
Everything EVIMO2 provides, as training TARGETS (see loss.py):
    * main task      : ground-truth moving-object mask (speed-aware,
                       identical definition to src/utils/metrics.py)
    * helper task 1  : ground-truth metric depth   (optional, weight 0 = off)
    * helper task 2  : ground-truth relative camera motion between the
                       last two frames              (optional, weight 0 = off)
The helper tasks only shape the shared encoder; they never feed the
mask decoder directly, so turning them off is a clean ablation.

Why this design is stable (compared with WorldModelV2/V3)
---------------------------------------------------------
V2/V3 produce the mask from a photometric warp residual computed with
PREDICTED depth/pose, thresholded and upsampled from low resolution.
Any error in geometry shows up as mask noise, and geometry is never
directly supervised there. Here the mask decoder is a plain U-Net over
the event features with full-resolution skip connections, trained
directly on the ground-truth mask — no intermediate quantity can drift.

Architecture
------------
    voxels (B, T, C, H, W)
        -> early temporal fusion: reshape to (B, T*C, H, W)
        -> EventEncoder stages (reused from world_model, unchanged)
               stem   : full resolution      (c1)
               layer1 : 1/2                  (c1)
               layer2 : 1/4                  (c2)
               layer3 : 1/8                  (c3)
               layer4 : 1/16                 (c4)
        -> bottleneck residual blocks
           (+ feature-wise modulation from the inertial embedding)
        -> U-Net decoder with skip connections up to full resolution
        -> mask logits (B, 1, H, W)
    helper heads:
        depth : log-depth at 1/4 resolution, from the 1/4 decoder stage
        pose  : 9 numbers [tx, ty, tz, a1(3), a2(3)] (translation +
                first two columns of the rotation matrix), from the
                pooled bottleneck (+ inertial embedding)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.world_model.event_encoder import EventEncoder
from src.models.world_model.blocks import ConvBlock, ResidualBlock
from src.models.world_model.imu_encoder import IMUEncoder


IDENTITY_POSE_9D = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


class DecoderStage(nn.Module):
    """Upsample to the skip's resolution, concatenate, two convolutions."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels + skip_channels, out_channels, kernel_size=3),
            ConvBlock(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class SupervisedMotionSegmenter(nn.Module):
    def __init__(
        self,
        num_bins: int = 5,
        num_frames: int = 3,
        stage_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        decoder_channels: tuple[int, int, int, int] = (128, 64, 32, 32),
        use_imu: bool = True,
        imu_hidden: int = 64,
        imu_embedding: int = 128,
        predict_depth: bool = True,
        predict_pose: bool = True,
        mask_prior: float = 0.05,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.num_frames = num_frames
        self.use_imu = use_imu
        self.predict_depth = predict_depth
        self.predict_pose = predict_pose and num_frames >= 2

        c1, c2, c3, c4 = stage_channels
        d3, d2, d1, d0 = decoder_channels

        # ---- Encoder (reused, unchanged) ---------------------------------
        self.encoder = EventEncoder(
            input_channels=num_bins * num_frames,
            stage_channels=stage_channels,
        )

        # ---- Inertial measurement unit branch ----------------------------
        if use_imu:
            self.imu_encoder = IMUEncoder(
                hidden_channels=imu_hidden, embedding_dim=imu_embedding,
            )
            imu_feature_dim = num_frames * imu_embedding
            self.imu_projection = nn.Sequential(
                nn.Linear(imu_feature_dim, 256), nn.SiLU(),
            )
            # Feature-wise modulation of the bottleneck: x * (1 + scale) + shift.
            # Zero-initialised, so training starts from the events-only model
            # and the inertial branch can only add information.
            self.imu_modulation = nn.Linear(256, 2 * c4)
            nn.init.zeros_(self.imu_modulation.weight)
            nn.init.zeros_(self.imu_modulation.bias)
        else:
            self.imu_encoder = None

        # ---- Bottleneck ----------------------------------------------------
        self.bottleneck = nn.Sequential(ResidualBlock(c4, c4), ResidualBlock(c4, c4))

        # ---- Decoder -------------------------------------------------------
        self.up3 = DecoderStage(c4, c3, d3)   # 1/16 -> 1/8
        self.up2 = DecoderStage(d3, c2, d2)   # 1/8  -> 1/4
        self.up1 = DecoderStage(d2, c1, d1)   # 1/4  -> 1/2
        self.up0 = DecoderStage(d1, c1, d0)   # 1/2  -> full (stem skip)

        self.mask_out = nn.Conv2d(d0, 1, kernel_size=1)
        # Start the mask at the expected moving-pixel fraction instead of
        # 50%. Removes the large, noisy loss of the first few hundred
        # steps (same trick as focal-loss detectors).
        nn.init.zeros_(self.mask_out.weight)
        nn.init.constant_(self.mask_out.bias, math.log(mask_prior / (1.0 - mask_prior)))

        # ---- Helper heads --------------------------------------------------
        if self.predict_depth:
            self.depth_out = nn.Sequential(
                ConvBlock(d2, d2, kernel_size=3),
                nn.Conv2d(d2, 1, kernel_size=1),
            )
        if self.predict_pose:
            pose_in = c4 + (256 if use_imu else 0)
            self.pose_out = nn.Sequential(
                nn.Linear(pose_in, 256), nn.SiLU(), nn.Linear(256, 9),
            )
            # Start at "no motion" (identity rotation, zero translation).
            nn.init.zeros_(self.pose_out[-1].weight)
            with torch.no_grad():
                self.pose_out[-1].bias.copy_(torch.tensor(IDENTITY_POSE_9D))

    # ------------------------------------------------------------------
    def _encode_imu(self, imu_frames, batch_size: int) -> torch.Tensor:
        """imu_frames: list (length T) of VoxelFrameBatch on the model device."""
        if len(imu_frames) != self.num_frames:
            raise ValueError(
                f"Model built for {self.num_frames} frames, got {len(imu_frames)} "
                f"inertial windows."
            )
        per_frame = [self.imu_encoder(frame=f, batch_size=batch_size) for f in imu_frames]
        return self.imu_projection(torch.cat(per_frame, dim=1))  # (B, 256)

    def forward(self, voxels: torch.Tensor, imu_frames=None) -> dict:
        """
        voxels     : (B, T, C, H, W) event voxel grids, oldest frame first.
        imu_frames : list of T VoxelFrameBatch (required when use_imu=True).

        Returns dict:
            mask       (B, 1, H, W)   logits
            mask_probs (B, 1, H, W)   sigmoid(logits), detached
            depth_log  (B, 1, H/4, W/4) or None   natural log of metres
            pose       (B, 9) or None
        """
        if voxels.ndim != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got {tuple(voxels.shape)}")
        B, T, C, H, W = voxels.shape
        if T != self.num_frames or C != self.num_bins:
            raise ValueError(
                f"Model built for T={self.num_frames}, C={self.num_bins}; "
                f"got T={T}, C={C}."
            )

        x = voxels.reshape(B, T * C, H, W)

        enc = self.encoder
        s0 = enc.stem(x)          # full
        s1 = enc.layer1(s0)       # 1/2
        s2 = enc.layer2(s1)       # 1/4
        s3 = enc.layer3(s2)       # 1/8
        s4 = enc.layer4(s3)       # 1/16

        imu_feat = None
        if self.use_imu:
            if imu_frames is None:
                raise ValueError("use_imu=True but no inertial frames were passed.")
            imu_feat = self._encode_imu(imu_frames, B).to(s4.dtype)
            scale, shift = self.imu_modulation(imu_feat).chunk(2, dim=1)
            s4 = s4 * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        b = self.bottleneck(s4)

        u3 = self.up3(b, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s1)
        u0 = self.up0(u1, s0)

        mask_logits = self.mask_out(u0)

        out = {
            "mask": mask_logits,
            "mask_probs": torch.sigmoid(mask_logits).detach(),
            "depth_log": None,
            "pose": None,
        }

        if self.predict_depth:
            out["depth_log"] = self.depth_out(u2)

        if self.predict_pose:
            pooled = b.mean(dim=(2, 3))
            if imu_feat is not None:
                pooled = torch.cat([pooled, imu_feat], dim=1)
            out["pose"] = self.pose_out(pooled)

        return out
