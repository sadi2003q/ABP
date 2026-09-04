"""
CLI entry point for training the event-camera world model.

Optimized for RTX 4070 12GB.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trainer import TrainConfig, Trainer


def setup_logging(save_dir: Path, log_level: str = "INFO"):
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "train.log"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("torch").setLevel(logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train event-camera self-supervised world model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--dataset-root", type=str,
                   default="/home/z/my-project/data/dataset_root")
    p.add_argument("--sensors", type=str, nargs="+", default=["left_camera"])
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--history-offsets", type=int, nargs="+",
                   default=[-3, -2, -1, 0])
    p.add_argument("--num-bins", type=int, default=5)
    p.add_argument("--image-scale", type=float, default=1.0)

    p.add_argument("--event-channels", type=int, default=256)
    p.add_argument("--imu-hidden", type=int, default=64)
    p.add_argument("--imu-embedding", type=int, default=128)
    p.add_argument("--decoder-channels", type=int, default=16)
    p.add_argument("--memory-type", type=str, default="transformer",
                   choices=["transformer", "convlstm", "convgru"],
                   help="Temporal memory backend")

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--mixed-precision", type=str, default="bf16",
                   choices=["bf16", "fp16", "none"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--residual-warmup-fraction", type=float, default=0.1)
    p.add_argument("--photometric-loss-weight", type=float, default=10.0,
                   help="Weight for photometric event reconstruction loss "
                        "(the core depth+pose supervision). Default 10.0 "
                        "because it must dominate the latent loss.")
    p.add_argument("--latent-warmup-fraction", type=float, default=0.2,
                   help="Fraction of training before latent loss starts (0-1)")
    p.add_argument("--residual-phase-start", type=float, default=0.4,
                   help="Fraction of training before residual+mask losses start (0-1)")

    p.add_argument("--save-dir", type=str, default="runs/exp001")
    p.add_argument("--log-every-n-steps", type=int, default=10,
                   help="Scalar logging interval (steps)")
    p.add_argument("--viz-every-n-steps", type=int, default=50,
                   help="Mask visualization logging interval. 0 = disabled")
    p.add_argument("--viz-max-samples", type=int, default=4)
    p.add_argument("--eval-every-n-epochs", type=int, default=5)
    p.add_argument("--checkpoint-every-n-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--overfit", action="store_true")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(Path(args.save_dir), args.log_level)
    log = logging.getLogger("train")
    log.info(f"Args: {vars(args)}")

    cfg = TrainConfig(
        dataset_root=args.dataset_root,
        sensors=tuple(args.sensors),
        split=args.split,
        val_split=(None if args.val_split.lower() == "none" else args.val_split),
        history_offsets=tuple(args.history_offsets),
        num_bins=args.num_bins,
        image_scale=args.image_scale,
        event_channels=args.event_channels,
        imu_hidden=args.imu_hidden,
        imu_embedding=args.imu_embedding,
        decoder_channels=args.decoder_channels,
        memory_type=args.memory_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        scheduler_eta_min=args.eta_min,
        mixed_precision=args.mixed_precision,
        grad_clip_max_norm=args.grad_clip,
        use_gradient_checkpointing=args.gradient_checkpointing,
        use_ema=(not args.no_ema),
        ema_decay=args.ema_decay,
        residual_warmup_fraction=args.residual_warmup_fraction,
        photometric_loss_weight=args.photometric_loss_weight,
        latent_warmup_fraction=args.latent_warmup_fraction,
        residual_phase_start=args.residual_phase_start,
        log_every_n_steps=args.log_every_n_steps,
        viz_every_n_steps=args.viz_every_n_steps,
        viz_max_samples=args.viz_max_samples,
        eval_every_n_epochs=args.eval_every_n_epochs,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        save_dir=args.save_dir,
        seed=args.seed,
        overfit_mode=args.overfit,
        resume_from=args.resume_from,
    )

    if args.overfit:
        log.info("=" * 60)
        log.info("OVERFIT MODE: single sequence, no val")
        log.info("=" * 60)
        cfg.val_split = None
        cfg.eval_every_n_epochs = 1
        if cfg.epochs > 20:
            log.info(f"Overfit mode: reducing epochs {cfg.epochs} -> 20")
            # cfg.epochs = 20

    if cfg.mixed_precision != "none":
        log.info(f"Mixed precision: {cfg.mixed_precision}")
    if cfg.use_ema:
        log.info(f"EMA: ON (decay={cfg.ema_decay})")
    log.info(f"Residual warmup fraction: {cfg.residual_warmup_fraction}")

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
