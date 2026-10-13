"""
self_train_labels.py — ONE round of self-training (relabelling).

Idea
----
A network trained on noisy automatic labels usually becomes cleaner than
the labels themselves: it learns what moving objects look like across
thousands of frames, so random label mistakes average out. Self-training
uses this: the trained network relabels the TRAINING sequences, and a new
network is trained on those cleaner labels. Published label-free motion
segmentation methods report their largest gain from this step
(e.g. Arja et al. 2024; Un-EVIMO, ECCV 2024, trains on its own pseudo-labels).

What this script does
---------------------
1. Loads a trained checkpoint (e.g. the last epoch of the first training run).
2. Predicts a moving-object probability for every frame of the TRAINING
   sequences you give (never the test sequences).
3. Writes new labels: probability > --threshold, small blobs removed.
   Same file format as flow_labels.py, so refine_pseudo_labels.py and
   train_selfsup.py work on them unchanged.
4. Reports label quality against the true masks (report only; the true
   masks never change the labels).

Usage
-----
python self_train_labels.py \
    --dataset-root /kaggle/working/evimo_root \
    --checkpoint /kaggle/working/runs/flow_split13/checkpoints/last.pth \
    --sequence scene14_dyn_test_00_000000 ... scene6_dyn_train_03_000000 \
    --out-dir /kaggle/working/self_train_round1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from generate_pseudo_labels import Pooled
from selfsup_signal_check import fill_blobs
from src.data.pseudo_labels import save_sequence_labels
from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic
from trainer_supervised import TrainConfigSupervised, TrainerSupervised, ARCHITECTURE_FIELDS


def true_mask_of(mask, frame_motion):
    """True moving-object mask as a boolean (H, W) array, or None (scoring only)."""
    if mask is None:
        return None
    gt = evimo2_mask_to_binary_dynamic(mask, get_dynamic_object_ids(frame_motion))
    gt = gt.cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)
    while gt.ndim > 2:
        gt = gt[0]
    return gt.astype(bool)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Self-training: relabel the training sequences with a trained network")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sequence", nargs="+", required=True, help="TRAINING sequences only.")
    p.add_argument("--sensors", nargs="+", default=["right_camera"])
    p.add_argument("--subset", default="imo")
    p.add_argument("--split", default="train")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability above which a pixel is 'moving'.")
    p.add_argument("--min-area", type=float, default=0.002, help="Smallest blob kept (fraction of image).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: configuration with the same network settings as the checkpoint
    cfg = TrainConfigSupervised(dataset_root=args.dataset_root, sensors=tuple(args.sensors),
                                subsets=(args.subset,), split=args.split, batch_size=args.batch_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint.get("config", {})
    for field in ARCHITECTURE_FIELDS:
        if field in saved:
            value = saved[field]
            setattr(cfg, field, tuple(value) if isinstance(value, list) else value)

    # Step 2: network + data of the training sequences
    trainer = TrainerSupervised(cfg, build_train=False)
    datasets = trainer._temporal_datasets(args.split, tuple(args.sequence))
    if not datasets:
        raise RuntimeError(f"No data for {args.sequence}")
    loader = trainer._loader(datasets, shuffle=False)

    # Step 3: load the trained weights (the smoothed "EMA" copy if it exists)
    if "ema_state_dict" in checkpoint:
        state = checkpoint["ema_state_dict"]["module"]
    else:
        state = checkpoint["model_state_dict"]
    trainer.model.load_state_dict(state)
    model = trainer.model
    model.eval()
    print(f"Checkpoint loaded: {args.checkpoint} (epoch {checkpoint.get('epoch', -1) + 1})", flush=True)

    # Step 4: predict every training frame
    labels_by_seq = {}
    scores = {}
    n_frames = 0
    for batch_number, raw_batch in enumerate(loader):
        outputs, _ = trainer._forward(model, raw_batch, train=False)
        probabilities = torch.sigmoid(outputs["mask"].float())
        # labels are stored at full camera resolution, e.g. 480 x 640
        if tuple(probabilities.shape[-2:]) != (args.height, args.width):
            probabilities = F.interpolate(probabilities, size=(args.height, args.width),
                                          mode="bilinear", align_corners=False)
        probabilities = probabilities.cpu().numpy()
        last = raw_batch.frames[-1]          # the frame the network predicts

        for k in range(probabilities.shape[0]):
            # Step 4a: new label = confident moving pixels, small blobs removed
            probability = probabilities[k]
            while probability.ndim > 2:
                probability = probability[0]
            label = fill_blobs(probability > args.threshold, close_radius=3,
                               min_area_fraction=args.min_area)

            sensor = last.sensors[k]
            sequence = last.sequence_names[k]
            frame_index = int(last.local_frame_indices[k])
            labels_by_seq.setdefault((sensor, sequence), {})[frame_index] = label

            # Step 4b: quality against the true mask (report only)
            gt = true_mask_of(last.mask[k], last.frame_motion[k])
            if gt is not None and gt.shape == label.shape:
                scores.setdefault(sequence, Pooled()).update(label, gt)
                scores.setdefault("ALL", Pooled()).update(label, gt)
            n_frames += 1

        if batch_number % 20 == 0:
            print(f"  {n_frames} frames relabelled", flush=True)

    # Step 5: save the new labels
    for (sensor, sequence), labels in labels_by_seq.items():
        save_sequence_labels(out, sensor, sequence, labels)

    # Step 6: report
    print("\n" + "=" * 90)
    print("SELF-TRAINING LABELS against the TRUE masks (report only)")
    print("=" * 90)
    for name in list(args.sequence) + ["ALL"]:
        pooled = scores.get(name)
        if pooled is None:
            continue
        precision = pooled.tp / max(1, pooled.tp + pooled.fp)
        recall = pooled.tp / max(1, pooled.tp + pooled.fn)
        print(f"  {name:<30} IoU {pooled.iou():.3f}   precision {precision:.3f}   recall {recall:.3f}   "
              f"false-blob frames {pooled.static_frames_with_fp}/{pooled.static_frames}")
    print(f"\nLabels saved: {out}")


if __name__ == "__main__":
    main()
