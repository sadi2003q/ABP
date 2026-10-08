"""
Train (or evaluate) the fully supervised, deployable moving-object
segmentation model (SupervisedMotionSegmenter).

Learns from: ground-truth moving-object mask (main target), plus
ground-truth depth and camera motion as helper targets.
Needs at test time: only event voxels (+ inertial data). No ground
truth is used in the forward pass.

Single sequence (check the pipeline; evaluates on the same windows)
------------------------------------------------------------------
python train_supervised.py \
    --dataset-root /content/drive/MyDrive/single_seq_root \
    --sensors left_camera --subsets imo --split train \
    --sequence scene15_dyn_test_06_000000 \
    --overfit --epochs 40 \
    --save-dir /content/drive/MyDrive/runs/supervised_single_seq

Full dataset (train on train split, select best on val split)
-------------------------------------------------------------
python train_supervised.py \
    --dataset-root /path/to/dataset_root \
    --sensors left_camera --subsets imo imo_II \
    --split train --val-split val \
    --epochs 60 --batch-size 8 --num-workers 4 --event-dropout 0.1 \
    --save-dir runs/supervised_full

Evaluate the best checkpoint on a held-out split at a fixed threshold
---------------------------------------------------------------------
python train_supervised.py --evaluate-only \
    --checkpoint runs/supervised_full/checkpoints/best.pth \
    --dataset-root /path/to/dataset_root \
    --sensors left_camera --subsets imo imo_II \
    --eval-split test --threshold 0.5 \
    --save-dir runs/supervised_full
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from trainer_supervised import TrainConfigSupervised, TrainerSupervised, evaluate_checkpoint


def main():
    p = argparse.ArgumentParser(description="Supervised moving-object segmentation (events + inertial)")

    g = p.add_argument_group("data")
    g.add_argument("--dataset-root", type=str, required=True)
    g.add_argument("--sensors", type=str, nargs="+", default=["left_camera"])
    g.add_argument("--subsets", "--subset", dest="subsets", type=str, nargs="+", default=["imo"],
                   help="One or more subset folders, e.g. imo imo_II.")
    g.add_argument("--split", type=str, default="train")
    g.add_argument("--sequence", type=str, nargs="+", default=None,
                   help="Restrict TRAINING to these sequence folder names.")
    g.add_argument("--val-split", type=str, default="val",
                   help="Validation split folder name, or 'none'.")
    g.add_argument("--val-sequence", type=str, nargs="+", default=None,
                   help="Restrict VALIDATION to these sequence folder names.")
    g.add_argument("--history-offsets", type=int, nargs="+", default=[-2, -1, 0])
    g.add_argument("--num-bins", type=int, default=5)
    g.add_argument("--batch-size", type=int, default=4)
    g.add_argument("--num-workers", type=int, default=2)
    g.add_argument("--no-balance", action="store_true",
                   help="Disable the guarantee that every batch contains a moving object.")
    g.add_argument("--min-dynamic-per-batch", type=int, default=1)
    g.add_argument("--event-dropout", type=float, default=0.0,
                   help="Fraction of pixels whose events are dropped during training "
                        "(augmentation). 0.1 recommended for full-dataset runs.")

    g = p.add_argument_group("labels")
    g.add_argument("--label-source", type=str, default="ground_truth", choices=["ground_truth", "pseudo"],
                   help="Train on true masks, or on pseudo-labels from generate_pseudo_labels.py. "
                        "Evaluation always uses the true masks.")
    g.add_argument("--pseudo-label-dir", type=str, default=None)

    g = p.add_argument_group("model")
    g.add_argument("--no-imu", action="store_true", help="Events only, no inertial input.")

    g = p.add_argument_group("loss")
    g.add_argument("--bce-weight", type=float, default=1.0)
    g.add_argument("--dice-weight", type=float, default=1.0)
    g.add_argument("--pos-weight", type=float, default=None,
                   help="Fixed positive-class weight; default is estimated from data.")
    g.add_argument("--max-pos-weight", type=float, default=10.0)
    g.add_argument("--depth-weight", type=float, default=0.1,
                   help="Helper depth supervision. 0 disables the depth head.")
    g.add_argument("--pose-weight", type=float, default=0.1,
                   help="Helper camera-motion supervision. 0 disables the pose head.")

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=60)
    g.add_argument("--learning-rate", type=float, default=2e-4)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--warmup-fraction", type=float, default=0.05)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--mixed-precision", type=str, default="auto",
                   choices=["auto", "bf16", "fp16", "none"])
    g.add_argument("--no-ema", action="store_true")
    g.add_argument("--ema-decay", type=float, default=0.999)

    g = p.add_argument_group("evaluation / logging")
    g.add_argument("--overfit", action="store_true",
                   help="Evaluate on the training windows (single-sequence check).")
    g.add_argument("--eval-every-n-epochs", type=int, default=1)
    g.add_argument("--select-metric", type=str, default="best_iou", choices=["best_iou", "iou_at_0.5"])
    g.add_argument("--early-stop-patience", type=int, default=0)
    g.add_argument("--checkpoint-every-n-epochs", type=int, default=10)
    g.add_argument("--log-every-n-steps", type=int, default=10)
    g.add_argument("--viz-every-n-steps", type=int, default=100)
    g.add_argument("--save-dir", type=str, default="runs/exp_supervised")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--resume-from", type=str, default=None)

    g = p.add_argument_group("evaluate a checkpoint only")
    g.add_argument("--evaluate-only", action="store_true")
    g.add_argument("--checkpoint", type=str, default=None)
    g.add_argument("--eval-split", type=str, default="test")
    g.add_argument("--eval-sequence", type=str, nargs="+", default=None)
    g.add_argument("--threshold", type=float, default=None,
                   help="Fixed decision threshold to report (choose it on validation).")

    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_name = "evaluate.log" if args.evaluate_only else "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(save_dir / log_name)],
        force=True,
    )

    cfg = TrainConfigSupervised(
        dataset_root=args.dataset_root,
        sensors=tuple(args.sensors),
        subsets=tuple(args.subsets),
        split=args.split,
        sequence=tuple(args.sequence) if args.sequence else None,
        val_split=None if args.val_split.lower() == "none" else args.val_split,
        val_sequence=tuple(args.val_sequence) if args.val_sequence else None,
        history_offsets=tuple(args.history_offsets),
        num_bins=args.num_bins,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance_dynamic_batches=not args.no_balance,
        min_dynamic_per_batch=args.min_dynamic_per_batch,
        event_dropout=args.event_dropout,
        label_source=args.label_source,
        pseudo_label_dir=args.pseudo_label_dir,
        use_imu=not args.no_imu,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=args.pos_weight,
        max_pos_weight=args.max_pos_weight,
        depth_weight=args.depth_weight,
        pose_weight=args.pose_weight,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
        grad_clip=args.grad_clip,
        mixed_precision=args.mixed_precision,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        overfit_mode=args.overfit,
        eval_every_n_epochs=args.eval_every_n_epochs,
        select_metric=args.select_metric,
        early_stop_patience=args.early_stop_patience,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        log_every_n_steps=args.log_every_n_steps,
        viz_every_n_steps=args.viz_every_n_steps,
        save_dir=args.save_dir,
        seed=args.seed,
        resume_from=args.resume_from,
    )

    if args.evaluate_only:
        if not args.checkpoint:
            p.error("--evaluate-only needs --checkpoint")
        evaluate_checkpoint(
            cfg, args.checkpoint, split=args.eval_split,
            sequence=tuple(args.eval_sequence) if args.eval_sequence else None,
            threshold=args.threshold,
        )
        return

    logging.info("=" * 70)
    logging.info("SUPERVISED MOVING-OBJECT SEGMENTATION")
    logging.info("=" * 70)
    TrainerSupervised(cfg).train()


if __name__ == "__main__":
    main()
