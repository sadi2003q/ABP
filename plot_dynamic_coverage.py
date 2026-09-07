#!/usr/bin/env python3
"""
plot_dynamic_coverage.py

Purpose
-------
Plots what percentage of each frame is covered by the ground-truth
"dynamic object" mask, across every frame in a sequence.

    X axis: frame number
    Y axis: % of frame that is dynamic

Uses the exact same GT definition as trainer_v2.py / metrics.py
(MOTION_THRESHOLD_SPEED = 0.05 applied to 3D object speed) — nothing
here is a new metric, it's a plot of the same GT your model is
trained/evaluated against.

Usage
-----
python plot_dynamic_coverage.py \
    --sequence-dir /path/to/.../tabletop_2_flat_fb_000000 \
    --output ./dynamic_coverage.png

The sequence directory must contain:
    dataset_mask.npz
    cache/frame_motion.npz
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.metrics import (
    get_dynamic_object_ids,
    evimo2_mask_to_binary_dynamic,
)


def compute_dynamic_coverage(sequence_dir):
    """
    Returns
    -------
    frame_numbers : np.ndarray, shape (num_frames,)
    coverage_pct  : np.ndarray, shape (num_frames,)
        Percentage (0-100) of the frame covered by the dynamic mask.
    """
    sequence_dir = Path(sequence_dir)

    mask_path = sequence_dir / "dataset_mask.npz"
    motion_path = sequence_dir / "cache" / "frame_motion.npz"

    if not mask_path.exists():
        raise FileNotFoundError(f"Missing {mask_path}")
    if not motion_path.exists():
        raise FileNotFoundError(f"Missing {motion_path}")

    mask_npz = np.load(mask_path, allow_pickle=True)
    fm = np.load(motion_path, allow_pickle=True)

    frame_ids = fm["frame_ids"]
    object_ids = fm["object_ids"]
    speed = fm["speed"]  # (num_frames, num_objects)

    mask_keys = mask_npz.files
    if len(mask_keys) != len(frame_ids):
        print(f"WARNING: {len(mask_keys)} mask frames vs "
              f"{len(frame_ids)} frame_motion rows — using the "
              f"shorter of the two, please double check inputs.")

    n = min(len(mask_keys), len(frame_ids))

    frame_numbers = np.zeros(n, dtype=int)
    coverage_pct = np.zeros(n, dtype=float)

    for i in range(n):
        raw_mask = mask_npz[mask_keys[i]]

        # Build a minimal object holding just what get_dynamic_object_ids needs.
        class _FM:
            pass
        frame_motion = _FM()
        frame_motion.object_ids = object_ids
        frame_motion.speed = speed[i]

        dynamic_ids = get_dynamic_object_ids(frame_motion)
        dynamic_mask = evimo2_mask_to_binary_dynamic(raw_mask, dynamic_ids)

        total_px = dynamic_mask.numel()
        dynamic_px = int(dynamic_mask.sum())

        frame_numbers[i] = int(frame_ids[i])
        coverage_pct[i] = 100.0 * dynamic_px / total_px

    return frame_numbers, coverage_pct


def plot_coverage(frame_numbers, coverage_pct, output_path, title=None):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    ax.plot(frame_numbers, coverage_pct, marker="o", markersize=3,
            color="#1f77b4", linewidth=1.2)

    ax.set_xlabel("Frame number")
    ax.set_ylabel("Dynamic mask coverage (% of frame)")
    ax.set_title(title or "Dynamic mask coverage over the sequence")
    ax.set_ylim(0, max(5, coverage_pct.max() * 1.1))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ground-truth dynamic mask coverage (%) per "
                     "frame across an EVIMO2 sequence."
    )
    parser.add_argument("--sequence-dir", type=str, required=True,
                         help="Path to one EVIMO2 sequence folder "
                              "(containing dataset_mask.npz and cache/).")
    parser.add_argument("--output", type=str, default="./dynamic_coverage.png",
                         help="Output plot path (.png).")
    parser.add_argument("--csv", type=str, default=None,
                         help="Optional: also save the per-frame numbers "
                              "as a CSV at this path.")
    parser.add_argument("--title", type=str, default=None,
                         help="Optional custom plot title.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading sequence:", args.sequence_dir)
    frame_numbers, coverage_pct = compute_dynamic_coverage(args.sequence_dir)

    print(f"Frames processed: {len(frame_numbers)}")
    print(f"Coverage — min: {coverage_pct.min():.2f}%  "
          f"max: {coverage_pct.max():.2f}%  "
          f"mean: {coverage_pct.mean():.2f}%")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_coverage(frame_numbers, coverage_pct, output_path, title=args.title)
    print(f"Saved plot: {output_path}")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w") as f:
            f.write("frame_number,dynamic_coverage_pct\n")
            for fn, pct in zip(frame_numbers, coverage_pct):
                f.write(f"{fn},{pct:.4f}\n")
        print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()