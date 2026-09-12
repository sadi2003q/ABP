"""
GTMaskModel — predict the dynamic-object mask from GROUND-TRUTH depth
and GROUND-TRUTH pose only.

Why this exists
----------------
The existing WorldModelV2/V3 pipeline predicts depth AND pose with
learned heads, warps event features with them, turns the residual
into a mask, and compares that mask to GT. That conflates two very
different failure modes: (a) the geometry (depth/pose) is wrong, or
(b) the mask head can't turn a correct residual into a clean mask.
`evaluate_with_gt_depth.py`'s own diagnostics in this repo found that
substituting GT depth barely changes IoU while substituting GT pose
(when it was actually plumbed through) is what mattered, and that
with a good residual the mask head has plenty of capacity (>90% IoU
in --gt-mask-sanity tests). That strongly suggests the geometry
(depth+pose), not the mask head, is the bottleneck in the
self-supervised pipeline.

This model removes that confound entirely: depth and pose are taken
from EVIMO2 ground truth (metric depth map + motion-capture camera
pose), never predicted. The ONLY thing this network learns is
"given a geometrically-exact ego-motion-compensated warp residual
between two event frames, plus multi-scale spatial context from the
event stream, output the pixel-accurate moving-object mask."

Pipeline
--------
1. Encode event voxel grids at t-1 and t with a shared CNN encoder
   (multi-scale feature pyramid, reused unchanged from the existing
   world model so skip connections stay well-tested).
2. Build the EXACT relative camera transform T_(t <- t-1) from GT
   camera poses (see gt_pose_utils.py) — this is motion-capture
   ground truth, not learned, not approximate.
3. Warp the t-1 latent feature map into frame t using GT depth at
   t-1 and the GT relative pose, through the same differentiable
   pinhole renderer (LatentRenderer) the rest of the repo already
   validated.
4. Because the warp uses EXACT geometry, any large residual between
   the warped t-1 features and the real t features can only be
   caused by independent object motion (or occlusion/disocclusion,
   which the refinement head learns to suppress) — NOT by wrong
   depth or wrong pose. This residual is the strongest possible
   signal for "which pixels moved independently of the camera".
5. A U-Net-style refinement head (residual + multi-scale encoder
   skip connections from frame t) sharpens that residual into a
   full-resolution mask, directly supervised by GT dynamic mask
   (BCE + soft-Dice), with NO self-supervised photometric loss and
   NO gradient path into any depth/pose network (there isn't one).

This model is trained with a single, direct, fully-supervised
mask loss. It has no jointly-optimized geometry to fight with the
mask objective, so it isolates and maximizes the mask head's
achievable accuracy given ground-truth geometry -- the "ceiling"
the self-supervised model is trying to approach.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.world_model.event_encoder import EventEncoder
from src.models.world_model.latent_renderer import LatentRenderer
from src.models.gt_mask_model.gt_pose_utils import batch_gt_relative_pose_9d, gt_pose_validity
from src.models.gt_mask_model.gt_depth_utils import build_gt_depth_batch


class ResidualMaskHead(nn.Module):
    """
    U-Net-style refinement: turns a (B,1,h,w) warp-residual map into a
    full-resolution mask, using multi-scale event-encoder features
    from the TARGET frame as skip connections for sharp boundaries.

    Architecturally identical in spirit to MaskRefinementHead (v2/v3)
    -- deliberately, since that head already demonstrated >90% IoU
    capacity in the --gt-mask-sanity diagnostic. Kept as its own copy
    here (not imported) so this model has zero dependency on the
    predicted-geometry pipeline and can evolve independently.
    """

    def __init__(
        self,
        skip_channels: tuple[int, int, int] = (128, 64, 32),
        hidden: int = 32,
        extra_residual_channels: int = 0,
    ):
        super().__init__()
        s3, s2, s1 = skip_channels
        in_c = 1 + extra_residual_channels

        self.enc1 = self._dbl(in_c, hidden)
        self.enc2 = self._dbl(hidden + s1, hidden)
        self.enc3 = self._dbl(hidden + s2, hidden * 2)

        self.bottleneck = self._dbl(hidden * 2 + s3, hidden * 2)

        self.dec2 = self._dbl(hidden * 2 + hidden, hidden)
        self.dec1 = self._dbl(hidden + hidden, hidden)

        self.final = nn.Conv2d(hidden, 1, 1)

    @staticmethod
    def _dbl(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.GELU(),
        )

    def forward(self, residual: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        """
        residual : (B, 1 [+extra], H, W)  — full target-frame resolution
        skips    : [skip_l3, skip_l2, skip_l1] at H/8, H/4, H/2

        Returns mask logits (B, 1, H, W).
        """
        skip_l3, skip_l2, skip_l1 = skips

        x1 = self.enc1(residual)                                                    # H
        x2 = self.enc2(torch.cat([F.avg_pool2d(x1, 2), skip_l1], dim=1))             # H/2
        x3 = self.enc3(torch.cat([F.avg_pool2d(x2, 2), skip_l2], dim=1))             # H/4
        xb = self.bottleneck(torch.cat([F.avg_pool2d(x3, 2), skip_l3], dim=1))       # H/8

        d2 = self.dec2(torch.cat(
            [F.interpolate(F.interpolate(xb, scale_factor=2, mode="bilinear", align_corners=False),
                            size=x2.shape[-2:], mode="bilinear", align_corners=False), x2], dim=1
        ))
        d1 = self.dec1(torch.cat(
            [F.interpolate(d2, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1], dim=1
        ))
        return self.final(d1)


class GTMaskModel(nn.Module):
    """
    Predicts the dynamic mask from GT depth + GT pose + event frames.

    Forward requires ground truth depth (frame t-1 and, if
    `use_gt_depth_at_target=True`, frame t as well) and GT camera
    poses for both frames. There is no predicted geometry anywhere
    in this model.
    """

    def __init__(
        self,
        num_bins: int = 5,
        event_channels: int = 256,
        mask_extra_input: str = "photometric",
        # "photometric": also warp the raw voxel grid (like WorldModelV2/V3's
        #   residual, cheap and directly interpretable). "none": use only
        #   the warped-latent residual.
    ):
        super().__init__()
        assert mask_extra_input in ("photometric", "none")
        self.mask_extra_input = mask_extra_input
        self.num_bins = num_bins

        self.event_encoder = EventEncoder(input_channels=num_bins)
        self.renderer = LatentRenderer(rotation_type="6d")

        extra_ch = 1 if mask_extra_input == "photometric" else 0
        self.mask_head = ResidualMaskHead(
            skip_channels=(128, 64, 32),
            hidden=32,
            extra_residual_channels=extra_ch,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Sparse-mask-friendly init: most pixels are static, so bias the
        # final layer toward "not dynamic" at initialization.
        nn.init.constant_(self.mask_head.final.bias, -3.0)

    def forward(
        self,
        voxel_t0: torch.Tensor,
        voxel_t1: torch.Tensor,
        gt_depth_t0: torch.Tensor,
        camera_motion_t0,
        camera_motion_t1,
        gt_depth_t1: torch.Tensor | None = None,
        K: torch.Tensor | None = None,
        distortion: torch.Tensor | None = None,
        camera_intrinsics: list | None = None,
        camera_distortion: list | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        voxel_t0 : (B, C, H, W)   event voxel grid at SOURCE frame (t-1)
        voxel_t1 : (B, C, H, W)   event voxel grid at TARGET frame (t)
        gt_depth_t0 : (B, 1, H, W)  GT metric depth at the SOURCE frame,
            already resized to full (H, W) resolution (meters).
        camera_motion_t0 : list[CameraMotion], length B — GT pose @ t-1
        camera_motion_t1 : list[CameraMotion], length B — GT pose @ t
        gt_depth_t1 : optional (B, 1, H, W) GT depth at target frame.
            Not used geometrically (the warp only needs source depth),
            but exposed for callers that want it for diagnostics/loss.
        K, distortion : optional pre-built (B,3,3)/(B,4) camera tensors.
            If not given, built from camera_intrinsics/camera_distortion.
        camera_intrinsics, camera_distortion : list of per-sample numpy
            arrays (as stored on EVIMO2Batch), used to build K/distortion
            if not passed directly.

        Returns
        -------
        dict:
            mask        : (B, 1, H, W) logits
            mask_probs  : (B, 1, H, W) sigmoid(mask), detached
            residual    : (B, 1, H, W) the raw warp residual fed to the
                          mask head (before refinement), detached
            pose_gt     : (B, 9) the GT relative pose actually used
            warped_voxel: (B, C, H, W) t-1 voxel warped into t, detached
                          (diagnostic / visualization)
        """
        device = voxel_t0.device
        B, C, H, W = voxel_t0.shape

        if K is None:
            K = torch.stack([
                torch.as_tensor(k, device=device, dtype=torch.float32)
                for k in camera_intrinsics
            ])
        if distortion is None:
            distortion = torch.stack([
                torch.as_tensor(d, device=device, dtype=torch.float32)
                for d in camera_distortion
            ])

        # === 1. Encode both frames (shared weights) ===
        stacked = torch.stack([voxel_t0, voxel_t1], dim=1)  # (B, 2, C, H, W)
        pyramid = self.event_encoder(stacked)
        # pyramid[k] : (B, 2, Ck, Hk, Wk)
        feat_t0 = pyramid[-1][:, 0]  # (B, 256, H/16, W/16)
        feat_t1 = pyramid[-1][:, 1]

        skip_l1_t1 = pyramid[0][:, 1]  # H/2
        skip_l2_t1 = pyramid[1][:, 1]  # H/4
        skip_l3_t1 = pyramid[2][:, 1]  # H/8

        # === 2. Exact GT relative pose (t-1 -> t), metric, not learned ===
        pose_gt = batch_gt_relative_pose_9d(
            camera_motions_target=camera_motion_t1,
            camera_motions_source=camera_motion_t0,
            device=device,
        )  # (B, 9)

        # === 3. Warp latent features t-1 -> t using GT depth + GT pose ===
        Hf, Wf = feat_t0.shape[-2:]
        K_feat = self._scale_intrinsics(K, orig_hw=(H, W), new_hw=(Hf, Wf))

        depth_t0_feat = F.interpolate(gt_depth_t0, size=(Hf, Wf), mode="nearest")

        warped_feat = self.renderer(
            feature=feat_t0, depth=depth_t0_feat, pose=pose_gt, K=K_feat, distortion=distortion,
        )
        latent_residual = (warped_feat - feat_t1).abs().mean(dim=1, keepdim=True)  # (B,1,Hf,Wf)
        latent_residual_full = F.interpolate(
            latent_residual, size=(H, W), mode="bilinear", align_corners=False
        )

        residual_inputs = [latent_residual_full]
        warped_voxel_full = None
        if self.mask_extra_input == "photometric":
            warped_voxel = self.renderer(
                feature=voxel_t0, depth=gt_depth_t0, pose=pose_gt, K=K, distortion=distortion,
            )
            photometric_residual = (warped_voxel - voxel_t1).abs().mean(dim=1, keepdim=True)
            residual_inputs.append(photometric_residual)
            warped_voxel_full = warped_voxel

        residual_full = torch.cat(residual_inputs, dim=1)  # (B, 1[+1], H, W)

        # Normalize residual (per-sample) so its scale is comparable
        # across sequences/lighting before feeding the conv head.
        residual_norm = self._normalize_residual(residual_full)

        # === 4. Refine residual -> sharp mask ===
        mask_logits = self.mask_head(
            residual_norm, skips=[skip_l3_t1, skip_l2_t1, skip_l1_t1]
        )

        return {
            "mask": mask_logits,
            "mask_probs": torch.sigmoid(mask_logits).detach(),
            "residual": residual_full.detach(),
            "pose_gt": pose_gt.detach(),
            "warped_voxel": warped_voxel_full.detach() if warped_voxel_full is not None else None,
            "K": K,
            "distortion": distortion,
            # Kept WITH gradients (not detached) for contrastive/
            # feature-consistency losses that need to backprop into
            # the encoder itself, not just the mask head. Both at
            # (B, 256, H/16, W/16) -- the bottleneck feature pyramid
            # level used for the warp.
            "warped_feat": warped_feat,
            "feat_t1": feat_t1,
        }

    @staticmethod
    def _scale_intrinsics(K: torch.Tensor, orig_hw, new_hw) -> torch.Tensor:
        orig_h, orig_w = orig_hw
        new_h, new_w = new_hw
        sx = orig_w / new_w
        sy = orig_h / new_h
        K_new = K.clone()
        K_new[:, 0, 0] = K[:, 0, 0] / sx
        K_new[:, 1, 1] = K[:, 1, 1] / sy
        K_new[:, 0, 2] = K[:, 0, 2] / sx
        K_new[:, 1, 2] = K[:, 1, 2] / sy
        return K_new

    @staticmethod
    def _normalize_residual(residual: torch.Tensor) -> torch.Tensor:
        """Per-sample, per-channel min-max normalization to [0, 1]."""
        B, Ch = residual.shape[:2]
        flat = residual.reshape(B, Ch, -1)
        r_min = flat.min(dim=-1, keepdim=True).values.unsqueeze(-1)
        r_max = flat.max(dim=-1, keepdim=True).values.unsqueeze(-1)
        return (residual - r_min) / (r_max - r_min + 1e-6)