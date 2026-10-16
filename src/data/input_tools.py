"""
input_tools.py — three small tools for the "why does the network not
generalise to unseen scenes?" diagnosis (Step 2, October 2026).

1. normalize_voxels(...)
   Rescales the event picture of every sample so its non-empty cells
   have mean 0 and spread 1. A fast camera produces many more events
   than a slow one; without this, the network sees very different
   number ranges in different scenes. Same rule as the public E-RAFT
   code (uzh-rpg/E-RAFT, utils/dsec_utils.py, VoxelGrid normalize=True).

2. random_flip_and_zoom(...)
   Data augmentation: every training sample is randomly mirrored
   left-right and zoomed in or out (with a small random shift). The
   same change is applied to the event picture AND to the label, so
   they stay aligned. Zooming changes the apparent object size, which
   matters because scene13 has tiny objects and scene15 large ones.
   (E-RAFT trains with horizontal flips and random crops.)

3. EventPixelIoU
   An extra score in the style of published EVIMO2 papers: the overlap
   is counted only on pixels that received events, and only in frames
   that contain a moving object. Our usual score counts every pixel of
   every frame, which is much harsher. Both are reported; neither is
   used for training.
"""

import torch
import torch.nn.functional as F


# ======================================================================
# 1. Voxel normalisation
# ======================================================================

def normalize_voxels(voxels):
    """
    voxels : tensor of shape (B, T, C, H, W)
             B = batch size (e.g. 4), T = frames (e.g. 3),
             C = time bins (e.g. 5), H x W = 480 x 640

    Returns a new tensor of the same shape. For every sample, the
    non-zero cells are shifted to mean 0 and scaled to spread 1.
    Cells without events stay exactly 0.
    """
    # Step 1: work on a copy, so the caller's tensor is not changed
    result = voxels.clone()

    # Step 2: how many samples are in the batch, e.g. 4
    batch_size = result.shape[0]

    # Step 3: normalise each sample on its own
    for sample_id in range(batch_size):

        # Step 3a: all cells of this sample, shape (T, C, H, W)
        one_sample = result[sample_id]

        # Step 3b: True where the cell has any event signal
        has_events = one_sample != 0

        # Step 3c: nothing to do if the sample has no events at all
        number_of_event_cells = int(has_events.sum())
        if number_of_event_cells == 0:
            continue

        # Step 3d: the values of the non-empty cells only
        event_values = one_sample[has_events]

        # Step 3e: their mean and spread, e.g. mean 0.02, std 0.9
        mean_value = event_values.mean()
        std_value = event_values.std()

        # Step 3f: shift (and scale, if the spread is not zero)
        if std_value > 0:
            new_values = (event_values - mean_value) / std_value
        else:
            new_values = event_values - mean_value

        # Step 3g: write the new values back into the same cells
        one_sample[has_events] = new_values

    return result


# ======================================================================
# 2. Random flip and zoom (training only)
# ======================================================================

def random_flip_and_zoom(voxels, label, weight, zoom_min=0.8, zoom_max=1.5):
    """
    voxels : (B, T, C, H, W) event pictures
    label  : (B, 1, H, W) moving-object label, 1 = moving, 0 = not
    weight : (B, 1, H, W) 1 = pixel counts in the loss, 0 = ignored,
             or None (then every pixel counts)
    zoom_min, zoom_max : zoom range. 1.5 = objects look 1.5x bigger,
                         0.8 = objects look smaller (empty border added)

    Returns (new_voxels, new_label, new_weight), all the same shapes.
    Border areas that come from outside the original picture get
    weight 0, so the loss ignores them.
    """
    # Step 1: read the sizes, e.g. B=4, T=3, C=5, H=480, W=640
    B, T, C, H, W = voxels.shape
    device = voxels.device

    # Step 2: if no weight was given, every pixel counts (all ones)
    if weight is None:
        weight = torch.ones(B, 1, H, W)

    # Step 3: move the label and weight to the same device as voxels
    label = label.to(device).float()
    weight = weight.to(device).float()

    # Step 4: build one "where to look" matrix per sample.
    # The matrix maps output positions to input positions:
    #     [[scale_x, 0,       shift_x],
    #      [0,       scale_y, shift_y]]
    # Positions run from -1 (left/top edge) to +1 (right/bottom edge).
    all_matrices = []
    for sample_id in range(B):

        # Step 4a: random zoom, e.g. 1.3 (objects 1.3x bigger)
        zoom = float(torch.empty(1).uniform_(zoom_min, zoom_max))

        # Step 4b: looking at a smaller area = zooming in
        scale = 1.0 / zoom

        # Step 4c: mirror left-right half of the time (-1 = mirror)
        if float(torch.rand(1)) < 0.5:
            mirror = -1.0
        else:
            mirror = 1.0

        # Step 4d: random shift, as large as the zoom allows
        # e.g. zoom 1.5 -> scale 0.667 -> shift up to +-0.333
        max_shift = abs(1.0 - scale)
        shift_x = float(torch.empty(1).uniform_(-max_shift, max_shift))
        shift_y = float(torch.empty(1).uniform_(-max_shift, max_shift))

        # Step 4e: put the numbers into the 2 x 3 matrix
        matrix = torch.tensor([[mirror * scale, 0.0, shift_x],
                               [0.0, scale, shift_y]])
        all_matrices.append(matrix)

    # Step 5: stack to shape (B, 2, 3) and move to the device
    matrices = torch.stack(all_matrices).to(device)

    # Step 6: turn the matrices into a sampling grid of shape (B, H, W, 2)
    grid = F.affine_grid(matrices, size=(B, 1, H, W), align_corners=False)

    # Step 7: resample the event pictures (smooth "bilinear" mode).
    # Frames and bins are folded into one axis first: (B, T*C, H, W)
    flat_voxels = voxels.reshape(B, T * C, H, W).float()
    new_flat = F.grid_sample(flat_voxels, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)
    new_voxels = new_flat.reshape(B, T, C, H, W)

    # Step 8: resample the label ("nearest" keeps it exactly 0 or 1)
    new_label = F.grid_sample(label, grid, mode="nearest",
                              padding_mode="zeros", align_corners=False)

    # Step 9: resample the weight; outside areas become 0 (ignored)
    new_weight = F.grid_sample(weight, grid, mode="nearest",
                               padding_mode="zeros", align_corners=False)

    return new_voxels, new_label, new_weight


# ======================================================================
# 3. Published-style score: IoU on event pixels only
# ======================================================================

class EventPixelIoU:
    """
    Counts the overlap only on pixels that received events, and only in
    frames that contain at least one moving pixel.

    Two numbers come out:
      pooled          : all such pixels of all frames added together
      mean_per_frame  : IoU of each frame, then the average over frames
    Threshold is fixed at 0.5 (no choice made with the evaluation masks).
    """

    def __init__(self, threshold=0.5):
        # Step 1: the decision threshold, e.g. 0.5
        self.threshold = threshold

        # Step 2: pooled counters (true positive, false positive, false negative)
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

        # Step 3: one IoU value per frame that was scored
        self.frame_ious = []

    def update(self, probs, true_mask, event_pixels):
        """
        probs        : (B, 1, H, W) predicted probability of "moving"
        true_mask    : (B, 1, H, W) 1 = moving, 0 = not (true masks)
        event_pixels : (B, 1, H, W) True where the frame has events
        """
        # Step 1: number of samples, e.g. 4
        batch_size = probs.shape[0]

        for sample_id in range(batch_size):

            # Step 2: this sample's maps, all on the CPU
            prob = probs[sample_id].detach().float().cpu()
            truth = true_mask[sample_id].detach().float().cpu() > 0.5
            events = event_pixels[sample_id].detach().cpu().bool()

            # Step 3: skip frames with no moving object at all
            if int(truth.sum()) == 0:
                continue

            # Step 4: predicted "moving" pixels
            predicted = prob >= self.threshold

            # Step 5: keep only pixels that have events
            predicted_on_events = predicted & events
            truth_on_events = truth & events

            # Step 6: count for this frame
            tp = float((predicted_on_events & truth_on_events).sum())
            fp = float((predicted_on_events & ~truth_on_events).sum())
            fn = float((~predicted_on_events & truth_on_events).sum())

            # Step 7: skip the frame if nothing is left to compare
            # (e.g. the moving object produced no events in this frame)
            union = tp + fp + fn
            if union == 0:
                continue

            # Step 8: add to the pooled counters and store the frame IoU
            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.frame_ious.append(tp / union)

    def compute(self):
        # Step 1: pooled IoU over all scored pixels
        pooled_union = self.tp + self.fp + self.fn
        if pooled_union > 0:
            pooled = self.tp / pooled_union
        else:
            pooled = 0.0

        # Step 2: average of the per-frame IoUs
        number_of_frames = len(self.frame_ious)
        if number_of_frames > 0:
            mean_per_frame = sum(self.frame_ious) / number_of_frames
        else:
            mean_per_frame = 0.0

        # Step 3: return everything in a dictionary
        return {
            "event_iou_pooled": pooled,
            "event_iou_mean_per_frame": mean_per_frame,
            "event_frames_scored": number_of_frames,
        }
