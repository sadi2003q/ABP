"""
Ground-truth depth batching utilities.

EVIMO2 provides per-frame metric depth (meters) as a (H, W) float32
array, or None for frames where depth wasn't captured/valid. This
module turns a list of such per-sample depth maps (as stored on
`EVIMO2Batch.depth`) into a single (B, 1, h, w) tensor resized to a
target resolution, plus a validity mask so callers can exclude
samples with missing GT depth instead of silently training on a
fabricated fallback value.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_gt_depth_batch(
    depth_list: list,
    target_hw: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """
    Parameters
    ----------
    depth_list : list of (np.ndarray | None), length B
        One GT metric depth map per sample (meters), or None.
    target_hw : (h, w)
        Resolution to resize depth to (nearest-neighbor — depth is a
        continuous quantity but nearest avoids blending foreground/
        background depth across object boundaries, which matters a
        lot right at the edges of moving objects).
    device, dtype : torch placement for the output tensor.

    Returns
    -------
    depth : (B, 1, h, w) tensor. Invalid samples are filled with the
        batch median depth (kept finite so downstream warps don't
        blow up), but MUST be excluded from loss/metrics using `valid`.
    valid : (B,) bool tensor — True where GT depth was actually present.
    """
    B = len(depth_list)
    h, w = target_hw

    per_sample = []
    valid = torch.zeros(B, dtype=torch.bool)

    for i, d in enumerate(depth_list):
        if d is None:
            per_sample.append(None)
            continue
        dt = torch.as_tensor(d, dtype=dtype, device=device)
        if dt.shape[-2:] != (h, w):
            dt = F.interpolate(
                dt.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest"
            ).squeeze(0).squeeze(0)
        per_sample.append(dt)
        valid[i] = True

    valid_vals = [d for d in per_sample if d is not None]
    fallback = (
        torch.stack(valid_vals).median(dim=0).values
        if valid_vals
        else torch.ones(h, w, device=device, dtype=dtype)
    )

    out = torch.zeros(B, 1, h, w, device=device, dtype=dtype)
    for i in range(B):
        out[i, 0] = per_sample[i] if per_sample[i] is not None else fallback

    return out, valid
