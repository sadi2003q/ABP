"""
Segmentation metrics for self-supervised dynamic object detection.

Computes the standard binary segmentation metrics:
  - IoU / Jaccard (the de facto EVIMO2 metric)
  - F1 / Dice
  - Precision
  - Recall
  - Dynamic ratio (mean of predicted mask — useful for detecting
    trivial-collapse to all-zeros or all-ones)

All metrics support a threshold sweep because self-supervised masks
are notoriously threshold-sensitive. The caller passes the predicted
LOGITS (or probabilities — both work, we apply sigmoid internally
if values look like logits); the metrics binarize at the given threshold.

Ground truth format
-------------------
EVIMO2 stores instance masks as uint16 with object_id * 1000
(e.g., 8000 = object 8, 12000 = object 12). NOT every object is
dynamic — some objects are static (just sitting in the scene).

To compute the GROUND TRUTH DYNAMIC MASK, we use the per-object
speed from `FrameMotion`:

    object_id -> speed
    8 -> 0.001  (static, < MOTION_THRESHOLD_SPEED)
    12 -> 0.150 (dynamic, > MOTION_THRESHOLD_SPEED)

A pixel is "dynamic" in the GT iff:
    (mask_value // 1000) -> object_id
    AND object_id is in the moving-objects set

Where the moving-objects set is:
    { object_id for object_id, speed in zip(frame_motion.object_ids,
                                            frame_motion.speed)
      if speed > MOTION_THRESHOLD_SPEED }

This matches the technique used in
tools/visualization/dataset_verification.py:render_motion_mask().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


# ==========================================================
# Configuration
# ==========================================================

# Speed threshold for classifying an object as "moving" / dynamic.
# Matches tools/visualization/dataset_verification.py
MOTION_THRESHOLD_SPEED = 0.05


# ==========================================================
# Helpers
# ==========================================================

def get_dynamic_object_ids(frame_motion) -> set[int]:
    """
    Return the set of object IDs that are actually moving
    (speed > MOTION_THRESHOLD_SPEED).

    Parameters
    ----------
    frame_motion : FrameMotion or None
        Per-frame motion data with .object_ids and .speed arrays.
        If None, returns an empty set (treat all objects as static).

    Returns
    -------
    set of int
        Object IDs that are moving.
    """
    if frame_motion is None:
        return set()

    object_ids = np.asarray(frame_motion.object_ids)
    speeds = np.asarray(frame_motion.speed)

    if len(object_ids) == 0:
        return set()

    # Boolean mask of moving objects
    moving_mask = speeds > MOTION_THRESHOLD_SPEED

    # Build the set of moving object IDs
    return set(int(oid) for oid in object_ids[moving_mask])


def evimo2_mask_to_binary_dynamic(
    mask,
    dynamic_object_ids: set[int] | None = None,
) -> torch.Tensor:
    """
    Convert an EVIMO2 instance mask to a binary "dynamic" mask.

    A pixel is "dynamic" iff:
        - The decoded object_id > 0 (it's an object, not background)
        - AND the object_id is in `dynamic_object_ids`
          (i.e., the object is actually moving)

    If `dynamic_object_ids` is None, this falls back to treating ALL
    object pixels as dynamic (the old behavior — kept for backward
    compat but not what you want for evaluation).

    Parameters
    ----------
    mask : torch.Tensor or np.ndarray
        Shape (B, H, W), (B, 1, H, W), (H, W).
        dtype: uint16 (raw EVIMO2) or any integer type.
    dynamic_object_ids : set of int, or None
        Set of object IDs that are actually moving.
        If None, all non-zero object pixels are considered dynamic
        (LEGACY BEHAVIOR — produces wrong GT for evaluation).

    Returns
    -------
    torch.Tensor
        Same shape (squeezed to (B, H, W) if input had channel dim),
        dtype=torch.bool. True where pixel is a moving object.
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    mask = mask.long()

    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    elif mask.ndim == 2:
        mask = mask.unsqueeze(0)

    # Decode object IDs (object_id = mask_value // 1000)
    object_id = mask // 1000  # (B, H, W)

    if dynamic_object_ids is None:
        # Legacy fallback: all non-zero objects are dynamic
        # (DO NOT use this for evaluation -- produces wrong GT)
        return object_id > 0

    if not dynamic_object_ids:
        # No objects moving -> all-zero GT
        return torch.zeros_like(object_id, dtype=torch.bool)

    # Build a boolean mask: pixel belongs to a moving object
    # We need to check if object_id[pixel] is in dynamic_object_ids.
    # Vectorized approach: stack all dynamic IDs and check membership.
    moving_ids = sorted(dynamic_object_ids)  # list[int]
    # Create a tensor of moving IDs and check element-wise
    moving_ids_tensor = torch.tensor(
        moving_ids, device=object_id.device, dtype=object_id.dtype,
    )
    # isin: returns True where object_id matches any of moving_ids
    return torch.isin(object_id, moving_ids_tensor)


# Legacy alias (kept for backward compat with code that may still call it)
def evimo2_mask_to_binary(mask) -> torch.Tensor:
    """Legacy: convert mask to binary treating ALL objects as dynamic.
    Prefer evimo2_mask_to_binary_dynamic() with dynamic_object_ids.
    """
    return evimo2_mask_to_binary_dynamic(mask, dynamic_object_ids=None)


# ==========================================================
# Segmentation Metrics
# ==========================================================

class SegmentationMetrics(nn.Module):
    """
    Binary segmentation metrics with threshold sweep.

    Supports speed-aware dynamic mask generation: pass a list of
    `frame_motion` objects (one per sample) and the metrics will
    compute the GT mask using only objects with speed > threshold.
    """

    def __init__(self, thresholds=None):
        super().__init__()
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        self.thresholds = list(thresholds)
        n = len(self.thresholds)
        self.register_buffer("tp", torch.zeros(n, dtype=torch.float64))
        self.register_buffer("fp", torch.zeros(n, dtype=torch.float64))
        self.register_buffer("fn", torch.zeros(n, dtype=torch.float64))
        self.register_buffer("tn", torch.zeros(n, dtype=torch.float64))
        self.register_buffer("pred_sum", torch.zeros((), dtype=torch.float64))
        self.register_buffer("gt_sum", torch.zeros((), dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    def reset(self):
        for buf in [self.tp, self.fp, self.fn, self.tn,
                    self.pred_sum, self.gt_sum, self.count]:
            buf.zero_()

    @torch.no_grad()
    def update(self, pred, gt, frame_motions=None):
        """
        Accumulate metrics over one batch.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted mask. Accepts either:
              - Probabilities in [0, 1] (from torch.sigmoid(logits))
              - Raw logits (we'll apply sigmoid internally)
            Shape: (B, 1, H, W) or (B, H, W).
        gt : torch.Tensor, np.ndarray, or list
            Ground-truth EVIMO2 mask (uint16 with object_id*1000),
            OR an already-binary mask (bool / 0/1).
            Shape: (B, 1, H, W), (B, H, W), (H, W), or list of these.
        frame_motions : list of FrameMotion, or None
            One FrameMotion per sample (length = batch size). Used to
            determine which object IDs are actually moving.
            If None, ALL non-zero object pixels are treated as dynamic
            (LEGACY BEHAVIOR — produces wrong GT for evaluation).
            Pass this to get the CORRECT speed-aware GT mask.
        """
        # Normalize pred shape to (B, H, W)
        if pred.ndim == 4:
            pred = pred.squeeze(1)
        elif pred.ndim == 2:
            pred = pred.unsqueeze(0)
        pred = pred.detach().float().cpu()

        # If pred values look like logits, apply sigmoid
        if pred.max().item() > 5 or pred.min().item() < -5:
            pred = torch.sigmoid(pred)

        B = pred.shape[0]

        # Normalize gt to list of (per-sample) arrays
        if isinstance(gt, (list, tuple)):
            gt_list = list(gt)
        else:
            # Single tensor -- split into per-sample slices
            if isinstance(gt, np.ndarray):
                gt = torch.from_numpy(gt)
            if gt.ndim == 4 and gt.shape[1] == 1:
                gt = gt.squeeze(1)
            elif gt.ndim == 2:
                gt = gt.unsqueeze(0)
            gt_list = [gt[i] for i in range(B)]

        # Normalize frame_motions to a list of length B
        if frame_motions is None:
            frame_motions = [None] * B
        elif len(frame_motions) != B:
            raise ValueError(
                f"frame_motions length {len(frame_motions)} != "
                f"batch size {B}"
            )

        # For each sample, convert GT mask to binary dynamic mask
        # using the per-sample frame_motion
        gt_binary_list = []
        for i in range(B):
            g = gt_list[i]
            if g is None:
                # Skip samples without GT
                gt_binary_list.append(None)
                continue

            # Get the dynamic object IDs for this sample
            dynamic_ids = get_dynamic_object_ids(frame_motions[i])

            # Convert to binary dynamic mask
            g_binary = evimo2_mask_to_binary_dynamic(g, dynamic_ids)
            gt_binary_list.append(g_binary)

        # Stack valid samples (some may be None)
        valid_pred_list = []
        valid_gt_list = []
        for i in range(B):
            if gt_binary_list[i] is not None:
                valid_pred_list.append(pred[i])
                valid_gt_list.append(gt_binary_list[i])

        if not valid_pred_list:
            return  # nothing to accumulate

        # Stack into batches (may have shape mismatches if masks vary)
        # We'll accumulate per-sample to handle this
        for p, g in zip(valid_pred_list, valid_gt_list):
            # Squeeze any leading batch dims from g -> (H, W)
            while g.ndim > 2:
                g = g.squeeze(0)

            # Resize gt to match pred if needed
            if g.shape != p.shape:
                # F.interpolate needs (N, C, H, W) input
                g = torch.nn.functional.interpolate(
                    g.float().unsqueeze(0).unsqueeze(0),
                    size=p.shape,
                    mode="nearest",
                ).squeeze().bool()

            # Accumulate dynamic ratio stats
            self.pred_sum += p.sum()
            self.gt_sum += g.float().sum()
            self.count += p.numel()

            # Per-threshold confusion matrix
            for i, thr in enumerate(self.thresholds):
                pred_bin = p > thr
                tp = (pred_bin & g).sum()
                fp = (pred_bin & ~g).sum()
                fn = (~pred_bin & g).sum()
                tn = (~pred_bin & ~g).sum()
                self.tp[i] += tp
                self.fp[i] += fp
                self.fn[i] += fn
                self.tn[i] += tn

    def compute(self):
        eps = 1e-7
        ious = (self.tp + eps) / (self.tp + self.fp + self.fn + eps)
        precisions = (self.tp + eps) / (self.tp + self.fp + eps)
        recalls = (self.tp + eps) / (self.tp + self.fn + eps)
        f1s = 2 * precisions * recalls / (precisions + recalls + eps)

        best_idx = int(ious.argmax().item())
        best_threshold = self.thresholds[best_idx]

        per_threshold = {}
        for i, thr in enumerate(self.thresholds):
            per_threshold[f"thr_{thr:.2f}"] = {
                "iou": ious[i].item(),
                "f1": f1s[i].item(),
                "precision": precisions[i].item(),
                "recall": recalls[i].item(),
                "tp": int(self.tp[i].item()),
                "fp": int(self.fp[i].item()),
                "fn": int(self.fn[i].item()),
                "tn": int(self.tn[i].item()),
            }

        if self.count > 0:
            pred_ratio = (self.pred_sum / self.count).item()
            gt_ratio = (self.gt_sum / self.count).item()
        else:
            pred_ratio = 0.0
            gt_ratio = 0.0

        return {
            "best_threshold": best_threshold,
            "best_iou": ious[best_idx].item(),
            "best_f1": f1s[best_idx].item(),
            "best_precision": precisions[best_idx].item(),
            "best_recall": recalls[best_idx].item(),
            "per_threshold": per_threshold,
            "pred_dynamic_ratio": pred_ratio,
            "gt_dynamic_ratio": gt_ratio,
        }


# ==========================================================
# Trivial Baselines
# ==========================================================

class TrivialBaselineMetrics(nn.Module):
    """Computes trivial baselines (all-zeros, all-ones, random)."""

    def __init__(self, thresholds=None):
        super().__init__()
        if thresholds is None:
            thresholds = [0.5]
        self.thresholds = thresholds

    @torch.no_grad()
    def compute_baselines(self, gt, frame_motions=None):
        """
        Compute baseline metrics for the given GT.

        Parameters
        ----------
        gt : same format as SegmentationMetrics.update
        frame_motions : list of FrameMotion, or None
            If provided, used to compute the speed-aware dynamic GT.
            If None, all non-zero objects are treated as dynamic.
        """
        # Normalize gt to list
        if isinstance(gt, (list, tuple)):
            gt_list = list(gt)
        else:
            if isinstance(gt, np.ndarray):
                gt = torch.from_numpy(gt)
            if gt.dtype != torch.bool:
                gt = gt.long()
            if gt.ndim == 4 and gt.shape[1] == 1:
                gt = gt.squeeze(1)
            elif gt.ndim == 2:
                gt = gt.unsqueeze(0)
            gt_list = [gt[i] for i in range(gt.shape[0])]

        B = len(gt_list)
        if frame_motions is None:
            frame_motions = [None] * B

        # Compute binary GT per sample
        gt_binary_list = []
        for i in range(B):
            g = gt_list[i]
            if g is None:
                continue
            dynamic_ids = get_dynamic_object_ids(frame_motions[i])
            g_binary = evimo2_mask_to_binary_dynamic(g, dynamic_ids)
            gt_binary_list.append(g_binary)

        if not gt_binary_list:
            return {
                "all_zeros": {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0},
                "all_ones": {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0},
                "random_uniform": {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0},
                "gt_dynamic_ratio": 0.0,
            }

        # Stack
        gt_binary = torch.stack(gt_binary_list).bool()

        eps = 1e-7
        tp_gt = gt_binary.sum().item()
        total = gt_binary.numel()
        tn_gt = total - tp_gt

        zeros_iou = 0.0
        zeros_f1 = 0.0
        zeros_precision = 0.0
        zeros_recall = 0.0

        ones_iou = tp_gt / (tp_gt + tn_gt + eps)
        ones_precision = tp_gt / (tp_gt + tn_gt + eps)
        ones_recall = 1.0
        ones_f1 = 2 * ones_precision * ones_recall / (ones_precision + ones_recall + eps)

        gt_ratio = tp_gt / total if total > 0 else 0
        rand_iou = gt_ratio * 0.5

        return {
            "all_zeros": {
                "iou": zeros_iou, "f1": zeros_f1,
                "precision": zeros_precision, "recall": zeros_recall,
            },
            "all_ones": {
                "iou": ones_iou, "f1": ones_f1,
                "precision": ones_precision, "recall": ones_recall,
            },
            "random_uniform": {
                "iou": rand_iou, "f1": 0.0,
                "precision": gt_ratio, "recall": 0.5,
            },
            "gt_dynamic_ratio": gt_ratio,
        }
