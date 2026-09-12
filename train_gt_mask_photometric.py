"""
Train GTMaskModel with a SELF-SUPERVISED PHOTOMETRIC/residual loss --
matching this project's actual V2/V3 recipe (residual pseudo-label +
sparsity/confidence regularization), using GROUND-TRUTH depth and
GROUND-TRUTH pose for the warp (no learned geometry). No real GT mask
is used in training -- only at evaluation time, to report IoU/F1.

Usage
-----
python train_gt_mask_photometric.py \
    --dataset-root /path/to/dataset_root \
    --sensors left_camera --subset imo \
    --split train --sequence scene15_dyn_test_06_000000 \
    --overfit \
    --batch-size 4 --epochs 100 \
    --balance-dynamic-batches \
    --eval-every-n-epochs 2 \
    --save-dir runs/exp_gt_mask_photometric
"""

import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from trainer_gt_mask_photometric import TrainConfigGTMaskPhotometric, TrainerGTMaskPhotometric


def main():
    p = argparse.ArgumentParser(description="Train GTMaskModel with self-supervised photometric/residual loss")
    p.add_argument("--dataset-root", type=str, default="/home/z/my-project/data/dataset_root")
    p.add_argument("--sensors", type=str, nargs="+", default=["left_camera"])
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--subset", type=str, default="imo")
    p.add_argument("--sequence", type=str, nargs="+", default=None)
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--frame-gap", type=int, default=1)
    p.add_argument("--num-bins", type=int, default=5)
    p.add_argument("--mask-extra-input", type=str, default="photometric", choices=["photometric", "none"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--mixed-precision", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--residual-weight", type=float, default=1.0,
                    help="Weight of the BCE-against-warp-residual pseudo-label term.")
    p.add_argument("--sparsity-weight", type=float, default=5.0)
    p.add_argument("--confidence-weight", type=float, default=1.0)
    p.add_argument("--target-dynamic-ratio", type=float, default=0.05,
                    help="Expected fraction of dynamic pixels (project convention: ~0.05 indoor).")
    p.add_argument("--save-dir", type=str, default="runs/exp_gt_mask_photometric")
    p.add_argument("--log-every-n-steps", type=int, default=10)
    p.add_argument("--viz-every-n-steps", type=int, default=50)
    p.add_argument("--eval-every-n-epochs", type=int, default=5)
    p.add_argument("--checkpoint-every-n-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--balance-dynamic-batches", action="store_true")
    p.add_argument("--min-dynamic-per-batch", type=int, default=1)
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(save_dir / "train.log")],
    )

    cfg = TrainConfigGTMaskPhotometric(
        dataset_root=args.dataset_root,
        sensors=tuple(args.sensors),
        split=args.split if not args.overfit else "train",
        subset=args.subset,
        sequence=tuple(args.sequence) if args.sequence else None,
        val_split=None if args.overfit else args.val_split,
        frame_gap=args.frame_gap,
        num_bins=args.num_bins,
        mask_extra_input=args.mask_extra_input,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mixed_precision=args.mixed_precision,
        grad_clip_max_norm=args.grad_clip,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        residual_weight=args.residual_weight,
        sparsity_weight=args.sparsity_weight,
        confidence_weight=args.confidence_weight,
        target_dynamic_ratio=args.target_dynamic_ratio,
        log_every_n_steps=args.log_every_n_steps,
        viz_every_n_steps=args.viz_every_n_steps,
        eval_every_n_epochs=args.eval_every_n_epochs,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        save_dir=args.save_dir,
        seed=args.seed,
        overfit_mode=args.overfit,
        resume_from=args.resume_from,
        balance_dynamic_batches=args.balance_dynamic_batches,
        min_dynamic_per_batch=args.min_dynamic_per_batch,
    )

    trainer = TrainerGTMaskPhotometric(cfg)
    trainer.train()


if __name__ == "__main__":
    main()