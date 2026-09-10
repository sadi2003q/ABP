"""
Diagnostic: print per-frame GT dynamic-pixel count for one sequence.
Run this INSIDE your actual environment (Colab) against the real dataset:

    !python diag_gt_ratio.py \
        --dataset-root /content/drive/MyDrive/single_seq_root \
        --sensors left_camera --subset imo --split train \
        --sequence scene15_dyn_test_06_000000
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.data.dataset import EVIMO2Dataset
from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic

p = argparse.ArgumentParser()
p.add_argument("--dataset-root", required=True)
p.add_argument("--sensors", nargs="+", default=["left_camera"])
p.add_argument("--subset", default="imo")
p.add_argument("--split", default="train")
p.add_argument("--sequence", nargs="+", default=None)
args = p.parse_args()

ds = EVIMO2Dataset(
    dataset_root=args.dataset_root, sensors=tuple(args.sensors),
    split=args.split, load_depth=True, load_mask=True,
    subset=args.subset, sequence=tuple(args.sequence) if args.sequence else None,
)

print(f"Total frames: {len(ds)}")
n_empty, n_nonempty = 0, 0
for i in range(len(ds)):
    sample = ds[i]
    fm = sample.frame_motion
    ids = get_dynamic_object_ids(fm)
    speeds = dict(zip(np.asarray(fm.object_ids).tolist(), np.asarray(fm.speed).tolist()))
    if sample.mask is None:
        print(f"frame {i}: mask=None")
        continue
    gt = evimo2_mask_to_binary_dynamic(sample.mask, ids)
    n_dyn = int(gt.sum())
    tag = "DYNAMIC" if n_dyn > 0 else "empty"
    if n_dyn > 0: n_nonempty += 1
    else: n_empty += 1
    if i < 15 or n_dyn > 0:
        print(f"frame {i}: dynamic_ids={ids} speeds={speeds} dynamic_px={n_dyn} [{tag}]")

print(f"\nSummary: {n_nonempty} frames with dynamic pixels, {n_empty} frames empty (out of {len(ds)})")