"""
Train the GT-geometry mask model: predicts the dynamic mask from
GROUND-TRUTH depth and GROUND-TRUTH pose only. No depth/pose network
is trained here -- this isolates and maximizes what the mask head can
achieve given perfect geometry, which is the ceiling the
self-supervised WorldModelV2/V3 pipeline is trying to approach.

Usage
-----
python train_gt_mask.py \
    --dataset-root /path/to/dataset_root \
    --sensors left_camera --subset imo \
    --batch-size 8 --epochs 50 --save-dir runs/exp_gt_mask
"""

import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from trainer_gt_mask import TrainConfigGTMask, TrainerGTMask


def main():
    p = argparse.ArgumentParser(description="Train GT-geometry mask model")
    p.add_argument("--dataset-root", type=str, default="/home/z/my-project/data/dataset_root")
    p.add_argument("--sensors", type=str, nargs="+", default=["left_camera"])
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--subset", type=str, default="imo",
                    help="Subset folder name under each sensor, e.g. "
                         "'imo', 'imo_II', 'sanity', 'sanity_II', 'sfm', 'sfm_II'.")
    p.add_argument("--sequence", type=str, nargs="+", default=None)
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--frame-gap", type=int, default=1,
                    help="Frame offset between source (t-1) and target (t) frame.")
    p.add_argument("--num-bins", type=int, default=5)
    p.add_argument("--mask-extra-input", type=str, default="photometric",
                    choices=["photometric", "none"],
                    help="Whether to also feed the raw-voxel photometric "
                         "residual to the mask head alongside the latent residual.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--mixed-precision", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--bce-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--pos-weight", type=float, default=None,
                    help="Manual BCE positive-class weight. If omitted, "
                         "auto-estimated from the training set's GT dynamic ratio.")
    p.add_argument("--no-auto-pos-weight", action="store_true")
    p.add_argument("--save-dir", type=str, default="runs/exp_gt_mask")
    p.add_argument("--log-every-n-steps", type=int, default=10)
    p.add_argument("--viz-every-n-steps", type=int, default=50)
    p.add_argument("--eval-every-n-epochs", type=int, default=5)
    p.add_argument("--checkpoint-every-n-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--balance-dynamic-batches", action="store_true",
                    help="Guarantee every training batch contains at least "
                         "--min-dynamic-per-batch windows with non-empty GT "
                         "dynamic mask. Recommended for short/sparse-motion "
                         "sequences.")
    p.add_argument("--min-dynamic-per-batch", type=int, default=1)
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(save_dir / "train.log"),
        ],
    )

    cfg = TrainConfigGTMask(
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
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=args.pos_weight,
        auto_pos_weight=not args.no_auto_pos_weight and args.pos_weight is None,
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

    trainer = TrainerGTMask(cfg)
    trainer.train()


if __name__ == "__main__":
    main()