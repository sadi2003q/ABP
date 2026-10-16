"""
flow_labels.py — SMALL TEST of dense-flow pseudo-labels (stage B4).

Why
---
B1 finds moving objects by their blur, which needs fast relative motion.
DINOv2 appearance on event pictures splits by wall texture, not objects.
Methods that work on EVIMO2 without labels (Arja et al. 2024, 79 % IoU)
all use DENSE optical flow: how every pixel moves. This test checks that cue.

How (per labelled frame i)
--------------------------
1. Two event pictures: events of the `gap` frames BEFORE frame i, and of
   the `gap` frames FROM frame i on (e.g. gap 3 = about 50 ms each).
2. Dense flow between the two pictures with RAFT (torchvision, trained on
   synthetic pictures with known flow; it has never seen EVIMO2 or any of
   its masks).
3. Background motion: an affine motion fitted with RANSAC over image CELLS
   (16 x 16 pixels). Every cell with events counts ONCE, however many
   events it has, so a large textured moving object cannot outvote the
   background by producing more events (the failure of B1).
4. Moving pixels: event pixels whose flow differs from the background
   motion by more than a threshold. Scored side by side:
     fixed 1.0 / 1.5 / 2.0 / 3.0 px
     per-frame "otsu":  the threshold that best separates this frame's
                        differences into two groups (Otsu's method, as in
                        Un-EVIMO), never below 1 px
     per-frame "noise": 4 x the spread of the background cells' own
                        differences (how noisy the flow is in this frame),
                        never below 1 px
   Then close gaps, fill holes and drop small blobs. A frame with no blob
   left gets an EMPTY label ("no moving object").
   12 Oct small test: the planar background model (8 numbers, exact motion
   of a slanted flat wall) beat the affine one on scene14_01 (0.37 vs 0.20),
   so planar is now the default.
5. Epipolar check (added 12 Oct after the 8-sequence run): when the camera
   moves sideways through a 3D room, near and far surfaces move differently
   (parallax), so ONE plane cannot describe the background and walls get
   marked as moving (scene14_00: precision 0.09). Every STATIC point, at any
   depth, must still move along its "epipolar line", which depends only on
   the camera's own motion. The camera motion (essential matrix) is fitted
   with RANSAC to the cell flows using the camera calibration; a pixel is
   moving if it is far from its epipolar line. No depth is needed.
   "hybrid" = planar when most cells agree with one plane (>= 0.9), epipolar
   otherwise. FAILED on scene15_00: a big object lowers the agreement, so it
   switched to epipolar exactly when the camera barely moves (epipolar fit
   meaningless). "parallax" = the fix: remove the camera's fitted ROTATION
   from the agreeing cells' motion; what is left is caused by camera
   TRANSLATION (parallax). Epipolar only if that is larger than
   --parallax-px, planar otherwise (the usual check in visual SLAM before
   trusting an epipolar fit).
   Single-frame parallax is noisy (scene15_00 switched in 15/40 frames even
   at 6 px), but the kind of camera motion changes slowly. "smoothed" =
   decide with the MEDIAN parallax of the frames within +-30 frames (same
   sequence): epipolar (>= 2 px) if above --parallax-px, planar (noise
   rule) otherwise. 12 Oct small test: median parallax ~10 px when the
   camera moves through the room (scene14_00/01), ~3 px when it is almost
   still (scene15_00/03) -> --parallax-px 6 (chosen on training sequences
   only; scene13 untouched).
6. Labels of the two main rules are always saved, so a full run never has
   to be repeated: <out-dir>/planar_noise and <out-dir>/smoothed.

The true masks are used ONLY to score the labels.
Needs Internet ON in Kaggle the first time (downloads RAFT weights, ~20 MB).

Usage (small test)
------------------
python flow_labels.py \
    --dataset-root /kaggle/working/evimo_root \
    --sequence scene14_dyn_test_01_000000 scene15_dyn_test_00_000000 scene6_dyn_train_03_000000 \
    --frame-step 10 --out-dir /kaggle/working/flow_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))

import torch

from generate_pseudo_labels import Pooled, get_sample, drop_old_samples
from selfsup_signal_check import fill_blobs
from src.data.event_flow import _basis
from src.data.pseudo_labels import save_sequence_labels

THRESHOLDS = (1.0, 1.5, 2.0, 3.0)      # pixels of flow difference over the picture gap


# ----------------------------------------------------------------------
# 1. Event pictures
# ----------------------------------------------------------------------

def events_of_frames(ds, refs, start, stop, cache):
    """All events of frames start .. stop-1 (one sequence). None if outside the sequence."""
    # Step 1: inside the dataset?
    if start < 0 or stop > len(refs):
        return None

    # Step 2: all frames from the same sequence?
    sequence_id = refs[start].sequence_id
    for j in range(start, stop):
        if refs[j].sequence_id != sequence_id:
            return None

    # Step 3: join their events
    parts = []
    for j in range(start, stop):
        sample = get_sample(ds, j, cache)
        parts.append(np.asarray(sample.events_xy).reshape(-1, 2))
    return np.concatenate(parts, axis=0)


def event_picture(xy, H, W):
    """
    Events -> picture (H, W) with values 0..1, plus the raw count per pixel.
    Log scale so busy edges do not drown quiet ones; slight blur so RAFT sees shapes.
    """
    # Step 1: whole-pixel positions
    x = np.clip(np.round(xy[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(xy[:, 1]).astype(int), 0, H - 1)

    # Step 2: events per pixel
    count = np.zeros((H, W), np.float64)
    np.add.at(count, (y, x), 1.0)

    # Step 3: log scale + light blur, then 0..1
    picture = ndimage.gaussian_filter(np.log1p(count), 1.0)
    picture = picture / max(picture.max(), 1e-9)
    return picture.astype(np.float32), count


# ----------------------------------------------------------------------
# 1b. Published-style score (event pixels only)
# ----------------------------------------------------------------------

def event_counts(label, gt, event_pixels):
    """
    Overlap counted ONLY on pixels that have events in this frame.
    label, gt, event_pixels : (H, W) True/False maps
    Returns (tp, fp, fn), e.g. (950, 120, 300).
    """
    # Step 1: keep only event pixels
    label_on_events = label & event_pixels
    gt_on_events = gt & event_pixels

    # Step 2: count the three cases
    tp = int(np.sum(label_on_events & gt_on_events))      # moving, and labelled moving
    fp = int(np.sum(label_on_events & ~gt_on_events))     # labelled moving, but not moving
    fn = int(np.sum(~label_on_events & gt_on_events))     # moving, but missed
    return tp, fp, fn


class EventScore:
    """
    Published EVIMO2 style: only frames with a moving object, only event pixels.
    Keeps both the pooled IoU and the IoU of every frame (for the mean per frame).
    """

    def __init__(self):
        # Step 1: pooled counters
        self.tp = 0
        self.fp = 0
        self.fn = 0
        # Step 2: one IoU per scored frame
        self.frame_ious = []

    def add(self, tp, fp, fn):
        # Step 1: skip frames with nothing to compare
        # (e.g. the moving object made no events in this frame)
        union = tp + fp + fn
        if union == 0:
            return
        # Step 2: add to the pooled counters
        self.tp += tp
        self.fp += fp
        self.fn += fn
        # Step 3: this frame's own IoU, e.g. 950 / 1370 = 0.69
        self.frame_ious.append(tp / union)

    def pooled(self):
        union = self.tp + self.fp + self.fn
        if union == 0:
            return 0.0
        return self.tp / union

    def mean_per_frame(self):
        if len(self.frame_ious) == 0:
            return 0.0
        return float(np.mean(self.frame_ious))

    def frames(self):
        return len(self.frame_ious)


# ----------------------------------------------------------------------
# 2. Dense flow with RAFT
# ----------------------------------------------------------------------

def load_raft(device, size):
    """RAFT from torchvision with its pretrained weights (synthetic training data)."""
    from torchvision.models.optical_flow import raft_large, raft_small
    from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
    if size == "small":
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights)
    else:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights)
    model.eval()
    return model.to(device), weights.transforms()


@torch.no_grad()
def dense_flow(model, transforms, picture_1, picture_2, device):
    """Flow (H, W, 2) in pixels: where each pixel of picture_1 moved to in picture_2."""
    # Step 1: grey pictures -> 3 identical colour channels, shape (1, 3, H, W)
    image_1 = torch.from_numpy(picture_1)[None, None].repeat(1, 3, 1, 1)
    image_2 = torch.from_numpy(picture_2)[None, None].repeat(1, 3, 1, 1)

    # Step 2: the scaling RAFT was trained with
    image_1, image_2 = transforms(image_1, image_2)

    # Step 3: RAFT refines its estimate several times; keep the last one
    predictions = model(image_1.to(device), image_2.to(device))
    flow = predictions[-1][0]                     # (2, H, W)
    return flow.permute(1, 2, 0).cpu().numpy()


# ----------------------------------------------------------------------
# 3. Background motion: RANSAC over cells
# ----------------------------------------------------------------------

def cell_flows(flow, has_events, cell=16, min_pixels=8):
    """
    One flow value per cell (median over the cell's event pixels).
    Returns centre x, centre y, flow u, flow v for cells with enough event pixels.
    """
    H, W = has_events.shape
    xs, ys, us, vs = [], [], [], []
    for top in range(0, H - cell + 1, cell):
        for left in range(0, W - cell + 1, cell):
            # Step 1: event pixels in this cell
            inside = has_events[top:top + cell, left:left + cell]
            if inside.sum() < min_pixels:
                continue

            # Step 2: median flow over them (robust to a few bad pixels)
            block = flow[top:top + cell, left:left + cell][inside]
            xs.append(left + cell / 2.0)
            ys.append(top + cell / 2.0)
            us.append(float(np.median(block[:, 0])))
            vs.append(float(np.median(block[:, 1])))
    return np.array(xs), np.array(ys), np.array(us), np.array(vs)


def solve_motion(xs, ys, us, vs, H, W, model):
    """Least-squares motion parameters from cell flows."""
    Bu, Bv = _basis(xs, ys, H, W, model)
    matrix = np.concatenate([Bu, Bv], axis=0)
    target = np.concatenate([us, vs])
    theta, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return theta


def cell_residuals(theta, xs, ys, us, vs, H, W, model):
    """Distance (pixels) between each cell's flow and the motion model."""
    Bu, Bv = _basis(xs, ys, H, W, model)
    return np.hypot(us - Bu @ theta, vs - Bv @ theta)


def ransac_background(xs, ys, us, vs, H, W, model="affine", iterations=300,
                      inlier_px=1.0, seed=0):
    """
    The motion that agrees with the MOST CELLS (not the most events).
    Returns (theta, share of cells that agree) or (None, 0) if too few cells.
    """
    n_params = {"translation": 2, "affine": 6, "planar": 8}[model]
    n_needed = n_params // 2                     # each cell gives 2 equations
    n_cells = len(xs)
    if n_cells < max(10, 3 * n_needed):
        return None, 0.0

    rng = np.random.default_rng(seed)
    best_inliers = None
    for step in range(iterations):
        # Step 1: a random minimal set of cells
        pick = rng.choice(n_cells, size=n_needed, replace=False)
        theta = solve_motion(xs[pick], ys[pick], us[pick], vs[pick], H, W, model)

        # Step 2: how many cells agree with it?
        inliers = cell_residuals(theta, xs, ys, us, vs, H, W, model) < inlier_px
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    # Step 3: refit on all agreeing cells
    theta = solve_motion(xs[best_inliers], ys[best_inliers], us[best_inliers],
                         vs[best_inliers], H, W, model)
    return theta, float(best_inliers.mean())


def otsu_threshold(values, top=10.0, bins=256):
    """
    Otsu's method: the cut that best separates the values into a low group
    and a high group (largest difference between the two group averages,
    weighted by group sizes). Values are clipped to 0..top pixels.
    """
    # Step 1: histogram of the values
    values = np.clip(values, 0.0, top)
    counts, edges = np.histogram(values, bins=bins, range=(0.0, top))
    centres = (edges[:-1] + edges[1:]) / 2.0

    # Step 2: for every possible cut: size and average of the low group
    low_size = np.cumsum(counts).astype(np.float64)
    low_sum = np.cumsum(counts * centres)
    total_size = low_size[-1]
    total_sum = low_sum[-1]
    if total_size == 0:
        return top

    # Step 3: size and average of the high group
    high_size = total_size - low_size
    low_mean = low_sum / np.maximum(low_size, 1e-9)
    high_mean = (total_sum - low_sum) / np.maximum(high_size, 1e-9)

    # Step 4: between-group spread; best cut = largest spread
    spread = low_size * high_size * (low_mean - high_mean) ** 2
    return float(edges[int(np.argmax(spread)) + 1])


def to_normalised(points, K, dist):
    """Pixel positions (N, 2) -> undistorted normalised camera coordinates (N, 2)."""
    import cv2
    points = np.asarray(points, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(points, K, dist).reshape(-1, 2)


def fit_essential(xs, ys, us, vs, K, dist, inlier_px):
    """
    Camera motion (essential matrix) from the cell flows, with RANSAC.
    Returns (E 3x3, inlier mask per cell) or (None, None).
    """
    import cv2
    # Step 1: where each cell centre was, and where it moved to
    start = to_normalised(np.stack([xs, ys], 1), K, dist)
    end = to_normalised(np.stack([xs + us, ys + vs], 1), K, dist)

    # Step 2: RANSAC fit; the threshold is converted from pixels to normalised units
    focal = float(K[0, 0])
    E, mask = cv2.findEssentialMat(start, end, focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC,
                                   prob=0.999, threshold=inlier_px / focal)
    if E is None or E.shape[0] < 3:
        return None, None
    return E[:3, :3], mask.ravel().astype(bool)


def translation_parallax(E, start, end, inliers, focal):
    """
    How far (pixels) the agreeing cells moved BEYOND what the camera's
    rotation alone explains, i.e. motion caused by camera translation.
    Median over the agreeing cells. Small -> no parallax -> use one plane.
    """
    import cv2
    if inliers is None or inliers.sum() < 8:
        return 0.0

    # Step 1: camera rotation R (and translation direction) from the essential matrix
    _, R, t, _ = cv2.recoverPose(E, start[inliers], end[inliers], focal=1.0, pp=(0.0, 0.0))

    # Step 2: where each start point would land if the camera ONLY rotated
    ones = np.ones((int(inliers.sum()), 1))
    rays = np.hstack([start[inliers], ones]) @ R.T
    rotated = rays[:, :2] / rays[:, 2:3]

    # Step 3: what is left over is due to translation; median, in pixels
    left_over = np.linalg.norm(end[inliers] - rotated, axis=1)
    return float(np.median(left_over)) * focal


def epipolar_distance(E, start, end, focal):
    """
    Distance (pixels) of each moved point from its epipolar line
    (first-order "Sampson" distance). start, end: normalised (N, 2).
    """
    # Step 1: add a 1 to every point (homogeneous coordinates)
    ones = np.ones((len(start), 1))
    p1 = np.hstack([start, ones])
    p2 = np.hstack([end, ones])

    # Step 2: epipolar lines in both pictures
    line_2 = p1 @ E.T            # E p1
    line_1 = p2 @ E              # E^T p2

    # Step 3: how badly each pair breaks the rule p2^T E p1 = 0, scaled to a distance
    error = np.sum(p2 * line_2, axis=1)
    scale = line_2[:, 0] ** 2 + line_2[:, 1] ** 2 + line_1[:, 0] ** 2 + line_1[:, 1] ** 2
    distance = np.abs(error) / np.sqrt(np.maximum(scale, 1e-18))

    # Step 4: back to pixels
    return distance * focal


def model_flow(theta, H, W, model):
    """Motion model -> flow (H, W, 2) for every pixel."""
    v, u = np.mgrid[0:H, 0:W]
    Bu, Bv = _basis(u.ravel(), v.ravel(), H, W, model)
    return np.stack([Bu @ theta, Bv @ theta], 1).reshape(H, W, 2)


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Small test: dense-flow (RAFT) pseudo-labels")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--sensors", nargs="+", default=["right_camera"])
    p.add_argument("--subset", default="imo")
    p.add_argument("--split", default="train")
    p.add_argument("--sequence", nargs="+", required=True)
    p.add_argument("--frame-step", type=int, default=10, help="Label every N-th frame.")
    p.add_argument("--gap", type=int, default=3, help="Frames of events in each of the two pictures.")
    p.add_argument("--background-model", choices=["translation", "affine", "planar"], default="planar")
    p.add_argument("--inlier-px", type=float, default=1.0, help="RANSAC: cell agrees within this many pixels.")
    p.add_argument("--written", default="otsu",
                   help="Which rule's labels are WRITTEN: otsu, noise, or a fixed number such as 2.0. "
                        "Also: epipolar, hybrid, parallax. All rules are reported.")
    p.add_argument("--parallax-window", type=int, default=30,
                   help="smoothed rule: frames on each side whose parallax is combined (median).")
    p.add_argument("--parallax-px", type=float, default=6.0,
                   help="parallax rule: use the epipolar check only when the camera translation moves the "
                        "background by more than this many pixels (after removing rotation).")
    p.add_argument("--min-threshold", type=float, default=1.0,
                   help="Per-frame thresholds are never below this many pixels.")
    p.add_argument("--min-area", type=float, default=0.01, help="Smallest blob kept (fraction of image).")
    p.add_argument("--raft", choices=["large", "small"], default="large")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    H, W = args.height, args.width
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    from src.data.dataset import EVIMO2Dataset
    from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic

    # Step 1: data (no depth needed)
    ds = EVIMO2Dataset(dataset_root=args.dataset_root, sensors=tuple(args.sensors),
                       split=args.split, load_depth=False, load_mask=True,
                       subset=args.subset, sequence=tuple(args.sequence))
    refs = ds.index.references

    # Step 2: flow network
    raft, transforms = load_raft(device, args.raft)
    print(f"RAFT ({args.raft}) loaded", flush=True)

    scores = {}                  # (sequence, threshold) -> Pooled
    event_scores = {}            # (sequence, rule) -> EventScore (published style)
    labels_by_seq = {}
    agree_shares = {}            # sequence -> list of RANSAC agreement shares
    used_thresholds = {}         # sequence -> per-frame thresholds that were used
    parallax_values = {}         # sequence -> per-frame translation parallax (pixels)
    smoothing_store = []         # per frame: parallax + the two candidate labels (for the smoothed rule)
    epipolar_chosen = {}         # sequence -> frames where the parallax rule chose epipolar
    examples = []
    cache = {}
    n_done = 0
    t_start = time.time()
    dilate = np.ones((5, 5), bool)

    for i in range(len(ds) - 1):
        if int(refs[i].local_frame_index) % args.frame_step != 0:
            continue
        drop_old_samples(cache, keep_from=i - args.gap)

        # Step 3: events before and from frame i
        before = events_of_frames(ds, refs, i - args.gap, i, cache)
        after = events_of_frames(ds, refs, i, i + args.gap, cache)
        if before is None or after is None or len(before) < 500 or len(after) < 500:
            continue
        a = get_sample(ds, i, cache)
        sequence = a.sequence_name

        # Step 4: true mask (scoring only)
        gt = None
        if a.mask is not None:
            gt = evimo2_mask_to_binary_dynamic(a.mask, get_dynamic_object_ids(a.frame_motion)).numpy()
            while gt.ndim > 2:
                gt = gt[0]
            gt = gt.astype(bool)

        # Step 5: two pictures and the dense flow between them
        picture_1, count_1 = event_picture(before, H, W)
        picture_2, count_2 = event_picture(after, H, W)
        flow = dense_flow(raft, transforms, picture_1, picture_2, device)

        # Step 6: pixels where flow can be trusted (events nearby)
        has_events = ndimage.binary_dilation(count_1 > 0, structure=dilate)

        # Step 7: background motion agreed by most cells
        xs, ys, us, vs = cell_flows(flow, count_1 > 0)
        theta, agree = ransac_background(xs, ys, us, vs, H, W, model=args.background_model,
                                         inlier_px=args.inlier_px, seed=i)
        if theta is None:
            continue
        agree_shares.setdefault(sequence, []).append(agree)

        # Step 8: how different is each pixel's flow from the background?
        background = model_flow(theta, H, W, args.background_model)
        difference = np.linalg.norm(flow - background, axis=2)

        # Step 9: the per-frame thresholds
        #   otsu : best split of this frame's differences (event pixels only)
        otsu = max(args.min_threshold, otsu_threshold(difference[has_events]))
        #   noise: 4 x robust spread of the background cells' own differences
        residuals = cell_residuals(theta, xs, ys, us, vs, H, W, args.background_model)
        background_cells = residuals[residuals < args.inlier_px]
        spread = 1.4826 * float(np.median(background_cells)) if len(background_cells) else 1.0
        noise = max(args.min_threshold, 4.0 * spread)
        used_thresholds.setdefault(sequence, {"otsu": [], "noise": []})
        used_thresholds[sequence]["otsu"].append(otsu)
        used_thresholds[sequence]["noise"].append(noise)

        # Step 9b: epipolar check (handles parallax; needs no depth)
        K = np.asarray(a.camera_intrinsics, np.float64)
        dist = np.asarray(a.camera_distortion, np.float64)
        epipolar = None
        epipolar_noise = None
        E, cell_inliers = fit_essential(xs, ys, us, vs, K, dist, args.inlier_px)
        if E is not None:
            # distance of every event pixel from its epipolar line
            rows_px, cols_px = np.nonzero(has_events)
            start = to_normalised(np.stack([cols_px, rows_px], 1), K, dist)
            moved = np.stack([cols_px + flow[rows_px, cols_px, 0], rows_px + flow[rows_px, cols_px, 1]], 1)
            end = to_normalised(moved, K, dist)
            epipolar = np.zeros((H, W))
            epipolar[rows_px, cols_px] = epipolar_distance(E, start, end, float(K[0, 0]))

            # noise rule: 4 x robust spread of the agreeing cells' own distances
            cell_start = to_normalised(np.stack([xs, ys], 1), K, dist)
            cell_end = to_normalised(np.stack([xs + us, ys + vs], 1), K, dist)
            cell_distance = epipolar_distance(E, cell_start, cell_end, float(K[0, 0]))[cell_inliers]
            cell_spread = 1.4826 * float(np.median(cell_distance)) if len(cell_distance) else 1.0
            epipolar_noise = max(args.min_threshold, 4.0 * cell_spread)

            # parallax: motion left after removing the camera rotation (pixels)
            parallax = translation_parallax(E, cell_start, cell_end, cell_inliers, float(K[0, 0]))
        else:
            parallax = 0.0
        parallax_values.setdefault(sequence, []).append(parallax)

        # Step 10: one label per rule
        rules = {f"fixed {thr:.1f} px": (difference, thr) for thr in THRESHOLDS}
        rules["per-frame otsu"] = (difference, otsu)
        rules["per-frame noise"] = (difference, noise)
        if epipolar is not None:
            rules["epipolar 1.0 px"] = (epipolar, 1.0)
            rules["epipolar 2.0 px"] = (epipolar, 2.0)
            rules["epipolar noise"] = (epipolar, epipolar_noise)
            if agree >= 0.9:
                rules["hybrid noise"] = (difference, noise)
            else:
                rules["hybrid noise"] = (epipolar, epipolar_noise)
            # parallax rule: epipolar only when the camera translation causes real parallax
            if parallax > args.parallax_px:
                rules["parallax noise"] = (epipolar, epipolar_noise)
                rules["parallax 2.0 px"] = (epipolar, max(2.0, epipolar_noise))
                epipolar_chosen[sequence] = epipolar_chosen.get(sequence, 0) + 1
            else:
                rules["parallax noise"] = (difference, noise)
                rules["parallax 2.0 px"] = (difference, noise)
        else:
            rules["hybrid noise"] = (difference, noise)
            rules["parallax noise"] = (difference, noise)
            rules["parallax 2.0 px"] = (difference, noise)
        frame_labels = {}
        for name, (difference_map, threshold) in rules.items():
            moving = (difference_map > threshold) & has_events
            frame_labels[name] = fill_blobs(moving, close_radius=5, min_area_fraction=args.min_area)
        if gt is not None:
            for name, label in frame_labels.items():
                scores.setdefault((sequence, name), Pooled()).update(label, gt)
                scores.setdefault(("ALL", name), Pooled()).update(label, gt)

        # Step 10b: published-style score (frames with a moving object only)
        planar_counts = None
        epipolar_counts = None
        event_pixels = None
        if gt is not None and gt.any():
            # pixels with at least one event in frame i itself (no dilation)
            events_now = events_of_frames(ds, refs, i, i + 1, cache)
            _, count_now = event_picture(events_now, H, W)
            event_pixels = count_now > 0
            for name, label in frame_labels.items():
                counts = event_counts(label, gt, event_pixels)
                event_scores.setdefault((sequence, name), EventScore()).add(*counts)
                event_scores.setdefault(("ALL", name), EventScore()).add(*counts)

        # Step 11: keep the label of the --written rule
        if args.written in ("otsu", "noise"):
            chosen = frame_labels[f"per-frame {args.written}"]
        elif args.written in ("epipolar", "hybrid", "parallax"):
            chosen = frame_labels.get(f"{args.written} noise", frame_labels["per-frame noise"])
        else:
            threshold = float(args.written)
            chosen = fill_blobs((difference > threshold) & has_events, close_radius=5,
                                min_area_fraction=args.min_area)
        labels_by_seq.setdefault((a.sensor, sequence), {})[int(a.local_frame_index)] = chosen

        # Step 11b: keep what the smoothed rule needs (decided after the loop)
        epipolar_label = frame_labels.get("epipolar 2.0 px", frame_labels["per-frame noise"])
        if "epipolar 2.0 px" in frame_labels:
            # same as the "parallax 2.0 px" rule: threshold at least 2 px, or the noise level if higher
            epipolar_label = fill_blobs((epipolar > max(2.0, epipolar_noise)) & has_events,
                                        close_radius=5, min_area_fraction=args.min_area)
        # published-style counts of the two candidates (the smoothed rule picks one later)
        if event_pixels is not None:
            planar_counts = event_counts(frame_labels["per-frame noise"], gt, event_pixels)
            epipolar_counts = event_counts(epipolar_label, gt, event_pixels)
        smoothing_store.append({
            "sensor": a.sensor, "sequence": sequence, "frame": int(a.local_frame_index),
            "parallax": parallax, "planar": frame_labels["per-frame noise"],
            "epipolar": epipolar_label, "gt": gt,
            "planar_counts": planar_counts, "epipolar_counts": epipolar_counts,
        })

        # Step 12: frames for the check picture
        if gt is not None and gt.any() and len(examples) < 24:
            examples.append((sequence, int(a.local_frame_index), picture_1, gt,
                             np.linalg.norm(flow, axis=2), np.clip(difference, 0, 5), chosen))

        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done} frames ({(time.time() - t_start) / n_done:.1f} s/frame)", flush=True)

    # Step 13: smoothed rule: median parallax of the neighbouring frames decides
    smoothed_by_seq = {}
    planar_by_seq = {}
    smoothed_epipolar = {}
    for entry in smoothing_store:
        # Step 13a: parallax of all labelled frames within +-window in the same sequence
        nearby = [other["parallax"] for other in smoothing_store
                  if other["sequence"] == entry["sequence"]
                  and abs(other["frame"] - entry["frame"]) <= args.parallax_window]
        typical_parallax = float(np.median(nearby))

        # Step 13b: epipolar when the camera clearly moves through the scene, planar otherwise
        if typical_parallax > args.parallax_px:
            label = entry["epipolar"]
            chosen_counts = entry["epipolar_counts"]
            smoothed_epipolar[entry["sequence"]] = smoothed_epipolar.get(entry["sequence"], 0) + 1
        else:
            label = entry["planar"]
            chosen_counts = entry["planar_counts"]

        # Step 13b2: published-style score of the smoothed rule
        if chosen_counts is not None:
            event_scores.setdefault((entry["sequence"], "smoothed"), EventScore()).add(*chosen_counts)
            event_scores.setdefault(("ALL", "smoothed"), EventScore()).add(*chosen_counts)
        key = (entry["sensor"], entry["sequence"])
        smoothed_by_seq.setdefault(key, {})[entry["frame"]] = label
        planar_by_seq.setdefault(key, {})[entry["frame"]] = entry["planar"]

        # Step 13c: score it (true masks: report only)
        if entry["gt"] is not None:
            scores.setdefault((entry["sequence"], "smoothed"), Pooled()).update(label, entry["gt"])
            scores.setdefault(("ALL", "smoothed"), Pooled()).update(label, entry["gt"])

    # Step 13d: save labels: the --written rule, and always the two main rules
    for (sensor, sequence), labels in labels_by_seq.items():
        save_sequence_labels(out, sensor, sequence, labels)
    for (sensor, sequence), labels in planar_by_seq.items():
        save_sequence_labels(out / "planar_noise", sensor, sequence, labels)
    for (sensor, sequence), labels in smoothed_by_seq.items():
        save_sequence_labels(out / "smoothed", sensor, sequence, labels)

    # Step 14: result table
    print("\n" + "=" * 100)
    print(f"SMALL TEST: dense-flow labels against the TRUE masks (report only) | background model: {args.background_model}")
    print("Reference, B1 (one motion + dither) on the same sequences: 14_01 0.278 | 15_00 0.003 | 6_03 0.016")
    print("=" * 100)
    rule_names = [f"fixed {thr:.1f} px" for thr in THRESHOLDS] + ["per-frame otsu", "per-frame noise",
                                                                  "epipolar 1.0 px", "epipolar 2.0 px",
                                                                  "epipolar noise", "hybrid noise",
                                                                  "parallax noise", "parallax 2.0 px",
                                                                  "smoothed"]
    report = {}
    for sequence in list(args.sequence) + ["ALL"]:
        if sequence == "ALL":
            print("--- ALL sequences pooled ---")
        else:
            shares = agree_shares.get(sequence, [])
            median_share = float(np.median(shares)) if shares else float("nan")
            used = used_thresholds.get(sequence, {"otsu": [], "noise": []})
            otsu_median = float(np.median(used["otsu"])) if used["otsu"] else float("nan")
            noise_median = float(np.median(used["noise"])) if used["noise"] else float("nan")
            print(f"--- {sequence}   (cells agreeing with background: {median_share:.2f} | "
                  f"median per-frame threshold: otsu {otsu_median:.2f} px, noise {noise_median:.2f} px) ---")
            values = parallax_values.get(sequence, [])
            if values:
                print(f"    translation parallax: median {np.median(values):.2f} px | parallax rule chose "
                      f"epipolar in {epipolar_chosen.get(sequence, 0)}/{len(values)} frames | smoothed rule: "
                      f"{smoothed_epipolar.get(sequence, 0)}/{len(values)} frames")
        for name in rule_names:
            pooled = scores.get((sequence, name))
            if pooled is None:
                continue
            precision = pooled.tp / max(1, pooled.tp + pooled.fp)
            recall = pooled.tp / max(1, pooled.tp + pooled.fn)
            print(f"  {name:<17} IoU {pooled.iou():.3f}   precision {precision:.3f}   "
                  f"recall {recall:.3f}   false-blob frames {pooled.static_frames_with_fp}/{pooled.static_frames}")
            report[f"{sequence} | {name}"] = {"iou": pooled.iou(), "precision": precision, "recall": recall}

    # Step 14b: published-style table (event pixels, frames with a moving object)
    print("\n" + "=" * 100)
    print("PUBLISHED-STYLE SCORE: IoU on EVENT PIXELS of frame i, only frames with a moving object")
    print("('smoothed' is the FROZEN method; the other rules are for reference only, never for choosing)")
    print("=" * 100)
    for sequence in list(args.sequence) + ["ALL"]:
        print(f"--- {sequence} ---")
        for name in rule_names:
            score = event_scores.get((sequence, name))
            if score is None:
                continue
            print(f"  {name:<17} mean per frame {score.mean_per_frame():.3f}   "
                  f"pooled {score.pooled():.3f}   frames {score.frames()}")
            report[f"{sequence} | {name} | event"] = {
                "mean_per_frame": score.mean_per_frame(),
                "pooled": score.pooled(),
                "frames": score.frames(),
            }

    # Step 14c: average over sequences of the frozen method (how papers report a table average)
    per_sequence_means = []
    for sequence in args.sequence:
        score = event_scores.get((sequence, "smoothed"))
        if score is not None and score.frames() > 0:
            per_sequence_means.append(score.mean_per_frame())
    if per_sequence_means:
        average = float(np.mean(per_sequence_means))
        print(f"\nFROZEN method ('smoothed'), average over {len(per_sequence_means)} sequences "
              f"of the mean per-frame event-pixel IoU: {average:.3f}")
        report["average over sequences | smoothed | event"] = average

    with open(out / "flow_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Step 15: check picture
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picked = []
    for sequence in args.sequence:
        own = [e for e in examples if e[0] == sequence]
        picked += own[:: max(1, len(own) // 2)][:2]
    if picked:
        titles = ["events (before)", "true moving mask", "flow size (px)",
                  "difference from background (0-5 px)", f"label ({args.written})"]
        figure, axes = plt.subplots(len(picked), 5, figsize=(20, 3.2 * len(picked)), squeeze=False)
        for row, (sequence, frame_index, events_img, gt, flow_size, diff, label) in enumerate(picked):
            for column, picture in enumerate([events_img, gt, flow_size, diff, label]):
                axes[row, column].imshow(picture, cmap="gray" if column in (0, 1, 4) else "magma")
                axes[row, column].set_title(f"{titles[column]}\n{sequence[:12]} frame {frame_index}", fontsize=8)
                axes[row, column].axis("off")
        plt.tight_layout()
        plt.savefig(out / "flow_check.png", dpi=70)
        print(f"\nPicture saved: {out / 'flow_check.png'}")
    print(f"Labels saved (rule: {args.written}): {out}")
    print(f"Also saved: {out / 'planar_noise'} and {out / 'smoothed'}")


if __name__ == "__main__":
    main()
