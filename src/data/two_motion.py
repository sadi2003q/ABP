"""
Two-motion background fit for STAGE B1 (events only).

Problem it solves
-----------------
The one-motion fit assumes that MOST events come from the background.
In many EVIMO2 recordings a large moving object produces most of the
events (e.g. scene15_00: 71 % of events in object frames), so the
"background" fit locks onto the object and the object is compensated
away (label recall ~0).

Idea
----
1. Fit the dominant motion (as before).
2. Fit a SECOND motion to the events the first motion explains worst.
3. Re-assign every event to the motion that makes it sharper, refit
   both motions (a simple, hard-assignment version of the
   expectation-maximisation segmentation of Stoffregen et al. 2019).
4. Decide which of the two motions is the BACKGROUND without any other
   sensor:
     - temporal rule: the camera's motion changes smoothly, so the
       background is the motion closest to the previous frame's
       background (similar in spirit to predicting the camera-caused
       motion and flagging what does not match it, Vasco et al. 2017);
     - start rule (no previous background yet, or the two motions are
       equally close): the background is the motion whose events are
       spread over more of the image.

Everything here uses only the events. The true masks are never used.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.data.event_flow import fit_background_flow, flow_field, _splat


def warp_events(xy, t, t0, t1, flow):
    """Move every event back to the window start t0 along `flow` (pixels over the window)."""
    # Step 1: image size
    H, W = flow.shape[:2]

    # Step 2: how far through the window each event happened (0 = start, 1 = end)
    alpha = np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)

    # Step 3: event positions
    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)

    # Step 4: flow at each event's pixel
    x_pixel = np.clip(np.round(x).astype(int), 0, W - 1)
    y_pixel = np.clip(np.round(y).astype(int), 0, H - 1)
    displacement = flow[y_pixel, x_pixel]

    # Step 5: warped positions
    x_warped = x - alpha * displacement[:, 0]
    y_warped = y - alpha * displacement[:, 1]
    return x_warped, y_warped


def per_event_sharpness(xy, t, t0, t1, flow, sigma=1.0):
    """
    For each event: how crowded its spot is in the image of warped events.
    Events that move with `flow` pile up on sharp edges -> high value.
    Events that move differently stay spread out -> low value.
    """
    # Step 1: image size
    H, W = flow.shape[:2]

    # Step 2: warp all events with this flow
    x_warped, y_warped = warp_events(xy, t, t0, t1, flow)

    # Step 3: image of warped events, slightly blurred
    image = ndimage.gaussian_filter(_splat(x_warped, y_warped, H, W), sigma)

    # Step 4: read the image at each event's warped position
    x_pixel = np.clip(np.round(x_warped).astype(int), 0, W - 1)
    y_pixel = np.clip(np.round(y_warped).astype(int), 0, H - 1)
    return image[y_pixel, x_pixel]


def fit_subset(xy, t, t0, t1, H, W, model, keep, init, max_events, seed):
    """Fit a flow to the chosen events only (keep = True/False per event)."""
    # Step 1: pick the events
    subset_xy = xy[keep]
    subset_t = t[keep]

    # Step 2: fit (the fitter itself refuses windows with too few events)
    theta, info = fit_background_flow(subset_xy, subset_t, t0, t1, H, W, model=model,
                                      init=init, max_events=max_events, seed=seed)
    return theta, info


def split_two_motions(xy, t, t0, t1, H, W, model, theta_dominant,
                      max_events=80_000, seed=0, iterations=2,
                      start_fraction=0.3, min_share=0.1, min_events=2000):
    """
    Find two motions in one window.

    Returns (theta_a, theta_b, in_a) or None when there is no clear second motion.
      theta_a, theta_b : the two flows' parameters
      in_a             : True for events that belong to motion a
    """
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    t = np.asarray(t, np.float64)
    n_events = len(t)
    if n_events < 2 * min_events:
        return None

    # Step 1: motion a = the dominant motion (already fitted)
    theta_a = np.asarray(theta_dominant, np.float64)
    sharp_a = per_event_sharpness(xy, t, t0, t1, flow_field(theta_a, H, W, model))

    # Step 2: start motion b from the events motion a explains worst (lowest 30 %)
    limit = np.quantile(sharp_a, start_fraction)
    in_b = sharp_a <= limit
    if in_b.sum() < min_events:
        return None
    theta_b, info_b = fit_subset(xy, t, t0, t1, H, W, model, in_b, None, max_events, seed + 1)
    if not info_b["fitted"]:
        return None

    # Step 3: alternate: give each event to the sharper motion, then refit both
    in_a = ~in_b
    for step in range(iterations):
        sharp_a = per_event_sharpness(xy, t, t0, t1, flow_field(theta_a, H, W, model))
        sharp_b = per_event_sharpness(xy, t, t0, t1, flow_field(theta_b, H, W, model))
        in_a = sharp_a >= sharp_b

        # stop when one motion keeps almost no events: there is only one motion
        share_b = 1.0 - in_a.mean()
        if share_b < min_share or share_b > 1.0 - min_share:
            return None
        if in_a.sum() < min_events or (~in_a).sum() < min_events:
            return None

        theta_a, info_a = fit_subset(xy, t, t0, t1, H, W, model, in_a, theta_a, max_events, seed + 2 + step)
        theta_b, info_b = fit_subset(xy, t, t0, t1, H, W, model, ~in_a, theta_b, max_events, seed + 10 + step)
        if not (info_a["fitted"] and info_b["fitted"]):
            return None

    return theta_a, theta_b, in_a


def coverage(xy, keep, H, W, cell=40, min_events=5):
    """Share of image cells (cell x cell pixels) that contain at least `min_events` of the chosen events."""
    # Step 1: chosen events
    chosen = np.asarray(xy, np.float64).reshape(-1, 2)[keep]

    # Step 2: which cell each event falls into
    columns = W // cell
    rows = H // cell
    cell_x = np.clip((chosen[:, 0] // cell).astype(int), 0, columns - 1)
    cell_y = np.clip((chosen[:, 1] // cell).astype(int), 0, rows - 1)

    # Step 3: count events per cell
    counts = np.bincount(cell_y * columns + cell_x, minlength=rows * columns)

    # Step 4: share of cells with enough events
    return float(np.mean(counts >= min_events))


def flow_distance(flow_1, flow_2, step=8):
    """Median distance (pixels) between two flow fields, on a coarse grid of pixels."""
    difference = flow_1[::step, ::step] - flow_2[::step, ::step]
    return float(np.median(np.linalg.norm(difference, axis=2)))


def choose_background(flow_a, flow_b, previous_background, cover_a, cover_b, margin=0.2):
    """
    Which motion is the background? Returns (0 for a / 1 for b, reason).

    temporal rule: closest to the previous frame's background flow,
                   when one is clearly closer (by more than `margin` pixels);
    start rule:    otherwise, the motion whose events cover more of the image.
    """
    # Step 1: temporal rule
    if previous_background is not None:
        distance_a = flow_distance(flow_a, previous_background)
        distance_b = flow_distance(flow_b, previous_background)
        if abs(distance_a - distance_b) >= margin:
            if distance_a < distance_b:
                return 0, "temporal"
            return 1, "temporal"

    # Step 2: start rule
    if cover_a >= cover_b:
        return 0, "coverage"
    return 1, "coverage"
