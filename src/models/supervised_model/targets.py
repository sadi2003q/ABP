"""
Build training targets for SupervisedMotionSegmenter from one
TemporalEVIMO2Batch (the raw batch, before voxelisation).

The moving-object mask uses EXACTLY the same definition as
src/utils/metrics.py (object speed > MOTION_THRESHOLD_SPEED), so the
thing trained on and the thing measured are the same.

Every target comes with a per-sample validity flag; nothing is
silently filled in for missing ground truth.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic
from src.models.gt_mask_model.gt_depth_utils import build_gt_depth_batch
from src.models.gt_mask_model.gt_pose_utils import batch_gt_relative_pose_9d
from src.models.supervised_model.model import IDENTITY_POSE_9D


def build_gt_mask(raw_masks, frame_motions, hw):
    """Returns (gt (B,1,H,W) float, valid (B,) bool)."""
    H, W = hw
    out, valid = [], []
    for raw, fm in zip(raw_masks, frame_motions):
        if raw is None:
            out.append(torch.zeros(H, W))
            valid.append(False)
            continue
        gt = evimo2_mask_to_binary_dynamic(raw, get_dynamic_object_ids(fm))
        while gt.ndim > 2:
            gt = gt.squeeze(0)
        gt = gt.float()
        if tuple(gt.shape) != (H, W):
            gt = F.interpolate(gt[None, None], size=(H, W), mode="nearest")[0, 0]
        out.append(gt)
        valid.append(True)
    return torch.stack(out).unsqueeze(1), torch.tensor(valid, dtype=torch.bool)


def build_gt_pose(camera_target, camera_source):
    """Relative camera motion source -> target as 9 numbers, plus validity."""
    valid = torch.tensor(
        [bool(ct.pose_available) and bool(cs.pose_available)
         for ct, cs in zip(camera_target, camera_source)],
        dtype=torch.bool,
    )
    B = len(camera_target)
    identity = torch.tensor(IDENTITY_POSE_9D).unsqueeze(0).repeat(B, 1)
    try:
        pose = batch_gt_relative_pose_9d(camera_target, camera_source, device=torch.device("cpu"))
    except Exception:
        return identity, torch.zeros(B, dtype=torch.bool)
    finite = torch.isfinite(pose).all(dim=1)
    valid = valid & finite
    pose = torch.where(valid[:, None], torch.nan_to_num(pose), identity)
    return pose, valid


def build_targets(raw_batch, mask_hw, depth_hw=None, need_pose=False) -> dict:
    """
    raw_batch : TemporalEVIMO2Batch (frames[-1] is the prediction frame)
    mask_hw   : (H, W) of the predicted mask
    depth_hw  : (h, w) of the predicted depth, or None to skip depth
    need_pose : build the relative camera motion between the last two frames

    All returned tensors are on CPU; the loss moves them to the device.
    """
    target = raw_batch.frames[-1]

    gt_mask, mask_valid = build_gt_mask(target.mask, target.frame_motion, mask_hw)
    targets = {
        "gt_mask": gt_mask,
        "mask_valid": mask_valid,
        # Raw masks + motion for SegmentationMetrics, which does its own
        # speed-aware binarisation (never pass it an already-binary mask
        # together with frame_motions=None — see trainer_gt_mask.py).
        "gt_mask_raw": list(target.mask),
        "frame_motions": list(target.frame_motion),
    }

    if depth_hw is not None:
        depth_list = list(target.depth) if target.depth is not None else [None] * len(target.mask)
        gt_depth, depth_valid = build_gt_depth_batch(depth_list, target_hw=depth_hw, device=torch.device("cpu"))
        targets["gt_depth"] = gt_depth / 1000 # EVIMO2 depth is in millimetres
        targets["depth_valid"] = depth_valid

    if need_pose and len(raw_batch.frames) >= 2:
        gt_pose, pose_valid = build_gt_pose(
            raw_batch.frames[-1].camera_motion, raw_batch.frames[-2].camera_motion,
        )
        targets["gt_pose"] = gt_pose
        targets["pose_valid"] = pose_valid

    return targets
