"""
Diagnostic: run the REAL training pipeline (EVIMO2Dataset ->
TemporalEVIMO2Dataset -> collate -> transform -> _prepare_batch)
end-to-end and print gt_mask.sum() per sample per batch, so we can
see exactly where the dynamic-pixel signal is lost (if it is) between
diag_gt_ratio.py's per-frame numbers and what trainer_gt_mask.py
actually receives.

Usage (inside Colab, same args as train_gt_mask.py):

    !python diag_prepare_batch.py \
        --dataset-root /content/drive/MyDrive/single_seq_root \
        --sensors left_camera --subset imo --split train \
        --sequence scene15_dyn_test_06_000000 \
        --batch-size 4
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import EVIMO2Dataset
from src.data.temporal_dataset import TemporalEVIMO2Dataset
from src.data.collate import temporal_collate_fn
from src.data.transforms import Compose, ToTensor, NormalizeEventTime, NormalizeIMU, VoxelizeEvents
from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic
from trainer_gt_mask import TrainConfigGTMask, TrainerGTMask

p = argparse.ArgumentParser()
p.add_argument("--dataset-root", required=True)
p.add_argument("--sensors", nargs="+", default=["left_camera"])
p.add_argument("--subset", default="imo")
p.add_argument("--split", default="train")
p.add_argument("--sequence", nargs="+", default=None)
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--frame-gap", type=int, default=1)
args = p.parse_args()

ds = EVIMO2Dataset(
    dataset_root=args.dataset_root, sensors=tuple(args.sensors),
    split=args.split, load_depth=True, load_mask=True,
    subset=args.subset, sequence=tuple(args.sequence) if args.sequence else None,
)
tds = TemporalEVIMO2Dataset(ds, history_offsets=(-args.frame_gap, 0))
print(f"Dataset windows: {len(tds)}")

loader = DataLoader(
    tds, batch_size=args.batch_size, shuffle=False,  # shuffle OFF for deterministic inspection
    collate_fn=temporal_collate_fn, num_workers=0,
)

transform = Compose([
    ToTensor(), NormalizeEventTime(), NormalizeIMU(),
    VoxelizeEvents(num_bins=5),
])

trainer = object.__new__(TrainerGTMask)
trainer.cfg = TrainConfigGTMask(frame_gap=args.frame_gap)

total_windows = 0
total_dynamic_windows = 0
total_dynamic_px = 0
total_px = 0

for batch_idx, raw_batch in enumerate(loader):
    voxel_batch = transform(raw_batch)

    # Also independently recompute GT dynamic mask directly from
    # raw_batch (bypassing _prepare_batch) to cross-check.
    last = raw_batch.frames[-1]
    src_frame_ids = raw_batch.frames[0].local_frame_indices
    tgt_frame_ids = raw_batch.frames[-1].local_frame_indices

    for i in range(len(last.mask)):
        total_windows += 1
        if last.mask[i] is None:
            print(f"batch {batch_idx} sample {i}: src_frame={src_frame_ids[i]} tgt_frame={tgt_frame_ids[i]} mask=None")
            continue
        ids = get_dynamic_object_ids(last.frame_motion[i])
        gt = evimo2_mask_to_binary_dynamic(last.mask[i], ids)
        n_dyn = int(gt.sum())
        total_px += int(gt.numel())
        if n_dyn > 0:
            total_dynamic_windows += 1
            total_dynamic_px += n_dyn
        print(f"batch {batch_idx} sample {i}: src_frame={src_frame_ids[i]} tgt_frame={tgt_frame_ids[i]} "
              f"dynamic_ids={ids} dynamic_px={n_dyn} pose_avail(src,tgt)="
              f"({raw_batch.frames[0].camera_motion[i].pose_available},{raw_batch.frames[-1].camera_motion[i].pose_available}) "
              f"depth_none(src,tgt)=({raw_batch.frames[0].depth[i] is None},{raw_batch.frames[-1].depth[i] is None})")

    # Now run it through the ACTUAL _prepare_batch and report what survives
    batch = trainer._prepare_batch(raw_batch, voxel_batch)
    if batch is None:
        print(f"batch {batch_idx}: _prepare_batch returned None (ALL samples dropped)")
    else:
        gt_mask = batch["gt_mask"]
        per_sample_sum = gt_mask.flatten(1).sum(dim=1)
        print(f"batch {batch_idx}: _prepare_batch kept {gt_mask.shape[0]}/{len(last.mask)} samples, "
              f"gt_mask sums={per_sample_sum.tolist()}")

print(f"\n=== SUMMARY ===")
print(f"Total windows: {total_windows}")
print(f"Windows with dynamic pixels (raw, bypassing _prepare_batch): {total_dynamic_windows}")
print(f"Total dynamic px (raw): {total_dynamic_px} / {total_px} total px")