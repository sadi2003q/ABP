"""
train_selfsup.py — entry point for SELF-SUPERVISED training.

Same network and trainer as train_supervised.py, but always trains on
pseudo-labels (masks the computer makes itself from event motion, see
generate_pseudo_labels.py / refine_pseudo_labels.py). Human-made masks
are never used for training; they are read only to score the result.

Takes exactly the same options as train_supervised.py, except that
--label-source is fixed to "pseudo" and --pseudo-label-dir is required.

Example
-------
python train_selfsup.py \
    --dataset-root /kaggle/input/datasets/makkisakib1/evimo2/single_seq_root \
    --sensors right_camera --subsets imo --split train \
    --pseudo-label-dir /kaggle/working/pseudo_labels_right_v3 \
    --pseudo-ignore-band 20 --depth-weight 0 --pose-weight 0 \
    --overfit --epochs 8 \
    --save-dir /kaggle/working/runs/selfsup_right_v3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import train_supervised


def main():
    argv = sys.argv[1:]
    if "--label-source" in argv:
        i = argv.index("--label-source")
        if i + 1 < len(argv) and argv[i + 1] != "pseudo":
            sys.exit("train_selfsup.py always trains on pseudo-labels; "
                     "use train_supervised.py for ground-truth training.")
    else:
        argv = argv + ["--label-source", "pseudo"]
    if "--pseudo-label-dir" not in argv:
        sys.exit("train_selfsup.py needs --pseudo-label-dir (a folder made by "
                 "generate_pseudo_labels.py or refine_pseudo_labels.py).")
    sys.argv = [sys.argv[0]] + argv
    train_supervised.main()


if __name__ == "__main__":
    main()
