#!/usr/bin/env python3
"""
precompute_corrected_gt.py

Purpose
-------
Precomputes the SILHOUETTE-FILTERED dynamic-object-id set for every
frame in an EVIMO2 sequence, in true frame order (unaffected by
DataLoader shuffling), and saves it as a small cache file:

    <sequence_dir>/cache/corrected_dynamic_ids.npz

This cache lets trainer_v2.py look up "which object IDs are really,
visibly dynamic in frame N" by a simple dictionary lookup during
training, WITHOUT needing to know what the previous frame in the
current (shuffled) batch was — the previous-frame comparison is done
once, here, ahead of time, using the sequence's real frame order.

This does not modify dataset_mask.npz, frame_motion.npz, or any other
existing cache file — it only adds one new file.

Usage
-----
python precompute_corrected_gt.py \
    --sequence-dir /content/drive/MyDrive/single_seq_root/left_camera/imo/train/scene6_dyn_train_03_000000 \
    --self-iou-threshold 0.85

Run this once per sequence before training with --use-corrected-gt
(see the trainer_v2.py patch).
"""

import argparse
from pathlib import Path

import numpy as np

MOTION_THRESHOLD_SPEED = 0.05  # must match src/utils/metrics.py


def get_dynamic_object_ids(object_ids, speed_row):
    moving = speed_row > MOTION_THRESHOLD_SPEED
    return set(int(oid) for oid in object_ids[moving])


def silhouette_filter(object_ids, speed_row, current_mask, previous_mask,
                       self_iou_threshold):
    speed_ids = get_dynamic_object_ids(object_ids, speed_row)
    if not speed_ids or previous_mask is None:
        return speed_ids

    kept = set()
    for oid in speed_ids:
        cur = (current_mask // 1000) == oid
        prev = (previous_mask // 1000) == oid
        union = np.logical_or(cur, prev).sum()
        if union == 0:
            kept.add(oid)
            continue
        self_iou = np.logical_and(cur, prev).sum() / union
        if self_iou <= self_iou_threshold:
            kept.add(oid)
    return kept


def main():
    parser = argparse.ArgumentParser(
        description="Precompute silhouette-filtered dynamic-object IDs "
                     "for every frame in an EVIMO2 sequence."
    )
    parser.add_argument("--sequence-dir", type=str, required=True)
    parser.add_argument("--self-iou-threshold", type=float, default=0.85)
    args = parser.parse_args()

    sequence_dir = Path(args.sequence_dir)
    mask_path = sequence_dir / "dataset_mask.npz"
    motion_path = sequence_dir / "cache" / "frame_motion.npz"
    out_path = sequence_dir / "cache" / "corrected_dynamic_ids.npz"

    if not mask_path.exists():
        raise FileNotFoundError(f"Missing {mask_path}")
    if not motion_path.exists():
        raise FileNotFoundError(f"Missing {motion_path}")

    print("Loading:", mask_path)
    print("Loading:", motion_path)

    mask_npz = np.load(mask_path, allow_pickle=True)
    fm = np.load(motion_path, allow_pickle=True)

    object_ids = fm["object_ids"]
    speed = fm["speed"]          # (num_frames, num_objects)
    frame_ids = fm["frame_ids"]  # true frame_id per row

    mask_keys = mask_npz.files
    if len(mask_keys) != len(frame_ids):
        print(f"  WARNING: {len(mask_keys)} mask frames vs "
              f"{len(frame_ids)} frame_motion rows — mismatched counts, "
              f"proceeding by position, please double check inputs.")

    masks = [mask_npz[k] for k in mask_keys]

    # Output arrays: for each frame_id, a variable-length list of
    # corrected dynamic object ids. Stored as a ragged structure via
    # two parallel arrays (frame_ids, then a list of arrays) since
    # npz doesn't support ragged arrays directly -- we use
    # allow_pickle to store a plain Python dict of {frame_id: list}.
    corrected = {}
    original = {}

    prev_mask = None
    prev_frame_id = None

    n_changed = 0

    for i in range(len(masks)):
        fid = int(frame_ids[i])
        cur_mask = masks[i]

        orig_ids = get_dynamic_object_ids(object_ids, speed[i])

        prev_for_this = prev_mask if (prev_frame_id is not None and fid == prev_frame_id + 1) else None
        corr_ids = silhouette_filter(
            object_ids, speed[i], cur_mask, prev_for_this,
            self_iou_threshold=args.self_iou_threshold,
        )

        if corr_ids != orig_ids:
            n_changed += 1

        original[fid] = sorted(orig_ids)
        corrected[fid] = sorted(corr_ids)

        prev_mask = cur_mask
        prev_frame_id = fid

    np.savez(
        out_path,
        original_dynamic_ids=np.array(original, dtype=object),
        corrected_dynamic_ids=np.array(corrected, dtype=object),
        self_iou_threshold=args.self_iou_threshold,
        motion_threshold_speed=MOTION_THRESHOLD_SPEED,
    )

    print()
    print(f"Frames processed        : {len(masks)}")
    print(f"Frames where GT changed : {n_changed}")
    print(f"Saved corrected GT cache: {out_path}")
    print()
    print("Next step: pass --use-corrected-gt when running train_v2.py "
          "(requires the trainer_v2.py patch).")


if __name__ == "__main__":
    main()