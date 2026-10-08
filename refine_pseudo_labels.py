"""
refine_pseudo_labels.py — temporal refinement of EXISTING pseudo-labels.

Reads a pseudo-label folder written by generate_pseudo_labels.py,
removes blobs that do not persist across neighbouring frames (and
optionally fills objects missed in single frames), and writes a new
pseudo-label folder. No regeneration needed: this takes about a minute
for a 400-frame sequence.

The refinement itself uses only the pseudo-labels. The true masks are
read solely to REPORT quality, before and after, for several settings.

Usage
-----
python refine_pseudo_labels.py \
    --pseudo-label-dir /kaggle/working/pseudo_labels_right_v2 \
    --out-dir /kaggle/working/pseudo_labels_right_v3 \
    --dataset-root /kaggle/input/datasets/makkisakib1/evimo2/single_seq_root \
    --sensors right_camera --subset imo --split train
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from generate_pseudo_labels import Pooled
from src.data.pseudo_labels import load_all_labels, save_sequence_labels
from src.data.temporal_refine import refine_sequence

logger = logging.getLogger("refine_pseudo_labels")

# (window, min_support, fill) settings reported side by side
GRID = [
    (1, 1, False),
    (1, 1, True),
    (1, 2, False),
    (2, 1, False),
    (2, 2, False),
    (2, 2, True),
]


def load_true_masks(args, wanted_keys):
    """{(sensor, sequence, local_frame_index): bool mask} for labelled frames that have a true mask."""
    from src.data.dataset import EVIMO2Dataset
    from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic

    ds = EVIMO2Dataset(dataset_root=args.dataset_root, sensors=tuple(args.sensors),
                       split=args.split, load_depth=False, load_mask=True,
                       subset=args.subset, sequence=args.sequence)
    gts = {}
    for i in range(len(ds)):
        s = ds[i]
        key = (s.sensor, s.sequence_name, int(s.local_frame_index))
        if key not in wanted_keys or s.mask is None:
            continue
        gt = evimo2_mask_to_binary_dynamic(s.mask, get_dynamic_object_ids(s.frame_motion)).numpy()
        while gt.ndim > 2:
            gt = gt[0]
        gts[key] = gt.astype(bool)
    return gts


def score(all_labels, gts) -> Pooled:
    pooled = Pooled()
    for (sensor, seq), labels in all_labels.items():
        for i, lab in labels.items():
            gt = gts.get((sensor, seq, i))
            if gt is not None:
                pooled.update(lab, gt)
    return pooled


def main():
    p = argparse.ArgumentParser(description="Temporal refinement of pseudo-labels")
    p.add_argument("--pseudo-label-dir", required=True, help="Input pseudo-label folder.")
    p.add_argument("--out-dir", required=True, help="Output folder for refined pseudo-labels.")
    p.add_argument("--dataset-root", required=True, help="Used only to report quality vs true masks.")
    p.add_argument("--sensors", nargs="+", default=["left_camera"])
    p.add_argument("--subset", default="imo")
    p.add_argument("--split", default="train")
    p.add_argument("--sequence", nargs="+", default=None)
    p.add_argument("--window", type=int, default=1, help="Neighbouring frames checked on each side.")
    p.add_argument("--min-support", type=int, default=1,
                   help="Neighbouring frames that must contain a nearby blob for a blob to be kept.")
    p.add_argument("--max-shift", type=int, default=15,
                   help="How far (pixels) an object may move between neighbouring frames.")
    p.add_argument("--fill", action="store_true", help="Also fill objects missed in single frames.")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "refine.log")], force=True)

    src_dir = Path(args.pseudo_label_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(
            f"Pseudo-label folder {src_dir} does not exist. If the Kaggle session restarted, "
            f"/kaggle/working was wiped: run generate_pseudo_labels.py again first.")
    found = sorted(str(f.relative_to(src_dir)) for f in src_dir.glob("*/*.npz"))
    if not found:
        raise RuntimeError(
            f"{src_dir} exists but contains no <sensor>/<sequence>.npz label files "
            f"(contents: {sorted(x.name for x in src_dir.iterdir())}).")
    all_labels = load_all_labels(src_dir)
    if args.sensors:
        all_labels = {k: v for k, v in all_labels.items() if k[0] in args.sensors}
    if args.sequence:
        all_labels = {k: v for k, v in all_labels.items() if k[1] in args.sequence}
    if not all_labels:
        raise RuntimeError(f"No pseudo-labels for sensors={args.sensors} / sequence={args.sequence} "
                           f"in {src_dir}. Label files present: {found}")
    n_frames = sum(len(v) for v in all_labels.values())
    logger.info(f"Loaded {n_frames} pseudo-labels from {len(all_labels)} sequence(s)")

    wanted = {(s, q, i) for (s, q), labs in all_labels.items() for i in labs}
    gts = load_true_masks(args, wanted)
    logger.info(f"True masks available for {len(gts)} labelled frames (report only)")

    rows = {}
    base = score(all_labels, gts)
    rows["input (no refinement)"] = base
    for (w, m, f) in GRID:
        refined = {k: refine_sequence(v, window=w, min_support=m, max_shift=args.max_shift, fill=f)
                   for k, v in all_labels.items()}
        rows[f"window={w} min_support={m} fill={'yes' if f else 'no'}"] = score(refined, gts)

    chosen = {k: refine_sequence(v, window=args.window, min_support=args.min_support,
                                 max_shift=args.max_shift, fill=args.fill)
              for k, v in all_labels.items()}
    for (sensor, seq), labels in chosen.items():
        save_sequence_labels(out, sensor, seq, labels)
    chosen_score = score(chosen, gts)

    logger.info("=" * 74)
    logger.info("Pseudo-label quality against the TRUE masks (report only)")
    logger.info(f"  {'setting':<38}{'IoU':>8}{'false-blob frames':>20}")
    for name, v in rows.items():
        logger.info(f"  {name:<38}{v.iou():>8.4f}{v.static_frames_with_fp:>12d}/{v.static_frames}")
    logger.info(f"  WRITTEN: window={args.window} min_support={args.min_support} "
                f"max_shift={args.max_shift} fill={'yes' if args.fill else 'no'} -> IoU {chosen_score.iou():.4f}")
    logger.info(f"  saved to {out}")
    logger.info("=" * 74)

    report = {
        "input_dir": str(args.pseudo_label_dir),
        "written": {"window": args.window, "min_support": args.min_support,
                    "max_shift": args.max_shift, "fill": args.fill, "iou": chosen_score.iou(),
                    "static_frames_with_false_blobs": chosen_score.static_frames_with_fp},
        "grid": {name: {"iou": v.iou(), "static_frames_with_false_blobs": v.static_frames_with_fp,
                        "static_frames": v.static_frames} for name, v in rows.items()},
    }
    with open(out / "pseudo_label_report.json", "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
