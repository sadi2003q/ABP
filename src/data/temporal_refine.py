"""
Temporal refinement of pseudo-labels (no ground truth involved).

A real moving object is detected in neighbouring frames at nearby
positions; spurious blobs from noise or imperfect motion compensation
tend to appear in one frame and vanish in the next. Two operations:

1. FILTER: keep a blob in frame t only if, in at least `min_support`
   of the frames t-window .. t+window (t excluded), some labelled pixel
   lies within `max_shift` pixels of it.
2. FILL (optional): if frame t has nothing at a place where BOTH frames
   t-1 and t+1 have a blob within `max_shift` pixels of each other, add
   the neighbours' blob pixels there (the object was missed in frame t).

Everything here uses only the pseudo-labels themselves.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _near(label: np.ndarray, max_shift: int) -> np.ndarray:
    """Pixels within `max_shift` of any labelled pixel (label itself included)."""
    if not label.any():
        return np.zeros(label.shape, bool)
    return ndimage.distance_transform_edt(~label) <= max_shift


def refine_sequence(
    labels: dict[int, np.ndarray],
    window: int = 1,
    min_support: int = 1,
    max_shift: int = 15,
    fill: bool = False,
) -> dict[int, np.ndarray]:
    """
    labels : {local_frame_index: bool (H, W)} for ONE sequence.
    Returns a new dict with the same keys.
    """
    near = {i: _near(lab, max_shift) for i, lab in labels.items()}
    out: dict[int, np.ndarray] = {}

    for i, lab in labels.items():
        neighbours = [j for j in range(i - window, i + window + 1) if j != i and j in labels]
        new = np.zeros(lab.shape, bool)

        # ---- 1. filter blobs without temporal support ----
        if lab.any():
            comp, n = ndimage.label(lab)
            for c in range(1, n + 1):
                blob = comp == c
                support = sum(bool((near[j] & blob).any()) for j in neighbours)
                if support >= min_support:
                    new |= blob

        # ---- 2. fill blobs missed in this frame ----
        if fill and (i - 1) in labels and (i + 1) in labels:
            prev, nxt = labels[i - 1], labels[i + 1]
            # neighbour blob pixels confirmed by the other neighbour
            candidate = (prev & near[i + 1]) | (nxt & near[i - 1])
            if candidate.any():
                comp, n = ndimage.label(candidate)
                already = _near(new, max_shift) if new.any() else np.zeros(lab.shape, bool)
                for c in range(1, n + 1):
                    blob = comp == c
                    if not (blob & already).any():   # only where this frame has nothing
                        new |= blob

        out[i] = new
    return out
