"""Train v2 — direct residual mask model."""

import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from trainer_v2 import TrainConfigV2, TrainerV2

def main():
    p = argparse.ArgumentParser(description="Train v2 (direct residual mask)")
    p.add_argument("--dataset-root", type=str, default="/home/z/my-project/data/dataset_root")
    p.add_argument("--sensors", type=str, nargs="+", default=["left_camera"])
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--history-offsets", type=int, nargs="+", default=[-12, -8, -4, 0])
    p.add_argument("--num-bins", type=int, default=5)
    p.add_argument("--image-scale", type=float, default=1.0)
    p.add_argument("--memory-type", type=str, default="transformer",
                   choices=["transformer", "convlstm", "convgru"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--mixed-precision", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--photometric-loss-weight", type=float, default=10.0)
    p.add_argument("--depth-smoothness-weight", type=float, default=1.0)
    p.add_argument("--pose-temporal-weight", type=float, default=1.0)
    p.add_argument("--sparsity-loss-weight", type=float, default=5.0)
    p.add_argument("--depth-diversity-weight", type=float, default=5.0)
    p.add_argument("--save-dir", type=str, default="runs/exp_v2")
    p.add_argument("--log-every-n-steps", type=int, default=10)
    p.add_argument("--viz-every-n-steps", type=int, default=50)
    p.add_argument("--eval-every-n-epochs", type=int, default=5)
    p.add_argument("--checkpoint-every-n-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--gt-mask-sanity", action="store_true",
                   help="Train directly against EVIMO dynamic GT masks for debugging.")
    args = p.parse_args()

    # Create save directory BEFORE setting up logging
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(save_dir / "train.log"),
                  logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("torch").setLevel(logging.WARNING)

    cfg = TrainConfigV2(
        dataset_root=args.dataset_root, sensors=tuple(args.sensors),
        split=args.split,
        val_split=None if args.val_split.lower() == "none" else args.val_split,
        history_offsets=tuple(args.history_offsets),
        num_bins=args.num_bins, image_scale=args.image_scale,
        memory_type=args.memory_type,
        batch_size=args.batch_size, num_workers=args.num_workers,
        epochs=args.epochs, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mixed_precision=args.mixed_precision, grad_clip_max_norm=args.grad_clip,
        use_ema=not args.no_ema, ema_decay=args.ema_decay,
        photometric_loss_weight=args.photometric_loss_weight,
        depth_smoothness_weight=args.depth_smoothness_weight,
        pose_temporal_weight=args.pose_temporal_weight,
        sparsity_loss_weight=args.sparsity_loss_weight,
        depth_diversity_weight=args.depth_diversity_weight,
        save_dir=args.save_dir, log_every_n_steps=args.log_every_n_steps,
        viz_every_n_steps=args.viz_every_n_steps,
        eval_every_n_epochs=args.eval_every_n_epochs,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        seed=args.seed, overfit_mode=args.overfit, gt_mask_sanity=args.gt_mask_sanity,
    )

    if args.overfit:
        logging.info("=" * 60)
        logging.info("V2 OVERFIT MODE")
        logging.info("=" * 60)

    if args.gt_mask_sanity:
        logging.info("=" * 60)
        logging.info("GT MASK SANITY TEST")
        logging.info("=" * 60)
        cfg.val_split = None
        cfg.eval_every_n_epochs = 1
        # For the overfit diagnostic, remove the artificial depth-diversity pressure.
        cfg.depth_diversity_weight = 0.0

    trainer = TrainerV2(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
