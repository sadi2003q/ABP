"""
Single-batch smoke test for the trainer.

Runs ONE forward + backward pass through the actual trainer on real
EVIMO2 data, logs to TensorBoard, and exits.

Usage:
    python test_trainer_one_batch.py
    python test_trainer_one_batch.py --image-scale 0.5 --mixed-precision bf16
"""

import sys
import time
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from trainer import TrainConfig, Trainer, scale_voxel_batch
from src.utils.metrics import SegmentationMetrics, TrivialBaselineMetrics

logger = logging.getLogger("test_one_batch")


def setup_logging(log_level="INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("torch").setLevel(logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-batch smoke test for the trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-root", type=str,
                   default="/home/z/my-project/data/dataset_root")
    p.add_argument("--save-dir", type=str,
                   default="/home/z/my-project/work/runs/test_one_batch")
    p.add_argument("--image-scale", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-bins", type=int, default=5)
    p.add_argument("--mixed-precision", type=str, default="none",
                   choices=["bf16", "fp16", "none"],
                   help="Use bf16 on GPU, none on CPU")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    cfg = TrainConfig(
        dataset_root=args.dataset_root,
        sensors=("left_camera",),
        split="train",
        val_split=None,
        history_offsets=(-3, -2, -1, 0),
        num_bins=args.num_bins,
        image_scale=args.image_scale,
        batch_size=args.batch_size,
        num_workers=0,
        epochs=1,
        learning_rate=1e-4,
        weight_decay=1e-5,
        warmup_epochs=0,
        scheduler_eta_min=1e-6,
        mixed_precision=args.mixed_precision,
        grad_clip_max_norm=1.0,
        use_gradient_checkpointing=False,
        use_ema=(not args.no_ema),
        ema_decay=0.999,
        residual_warmup_fraction=1.0,
        log_every_n_steps=1,
        viz_every_n_steps=1,
        viz_max_samples=4,
        eval_every_n_epochs=999,
        checkpoint_every_n_epochs=999,
        save_dir=args.save_dir,
        seed=42,
        overfit_mode=True,
    )

    logger.info("=" * 70)
    logger.info("SINGLE-BATCH TRAINER SMOKE TEST")
    logger.info("=" * 70)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device_str}")
    logger.info(f"Image scale: {cfg.image_scale}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"Mixed precision: {cfg.mixed_precision}")

    t0 = time.time()
    trainer = Trainer(cfg)
    logger.info(f"Trainer built in {time.time()-t0:.1f}s")
    logger.info(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")

    train_iter = iter(trainer.train_loader)
    raw_batch = next(train_iter)
    voxel_batch = trainer.transform(raw_batch)

    voxels = torch.stack(
        [f.voxel_grid for f in voxel_batch.frames], dim=1
    ).to(trainer.device)
    voxel_batch = voxel_batch.to(trainer.device)

    if cfg.image_scale < 1.0:
        voxels = scale_voxel_batch(voxels, cfg.image_scale)

    logger.info(f"Voxels shape: {tuple(voxels.shape)}")

    gt_masks = raw_batch.frames[-1].mask
    n_gt = sum(1 for m in gt_masks if m is not None)
    logger.info(f"GT masks available: {n_gt} / {len(gt_masks)}")
    if n_gt > 0:
        first_gt = next(m for m in gt_masks if m is not None)
        first_gt = np.asarray(first_gt)
        ids = np.unique(first_gt // 1000)
        logger.info(f"  Object IDs: {ids[ids > 0]}")

    # ============================================================
    # FORWARD + BACKWARD
    # ============================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("RUNNING FORWARD + BACKWARD")
    logger.info("=" * 70)

    trainer.loss_fn.residual_loss_weight = 1.0

    t0 = time.time()
    loss_output, total_loss = trainer._forward_backward(voxels, voxel_batch)
    fb_time = time.time() - t0
    logger.info(f"Forward + backward time: {fb_time:.2f}s")

    trainer.optimizer.step()
    if trainer.ema is not None:
        trainer.ema.update(trainer.model)

    # ============================================================
    # Inspect outputs
    # ============================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("OUTPUT INSPECTION")
    logger.info("=" * 70)

    # mask_probs is sigmoid(logits), computed inside DynamicResidualLoss
    mask_probs = loss_output["mask_probs"]
    if mask_probs is None:
        # Fallback: re-run forward to compute it
        trainer.model.eval()
        with torch.no_grad():
            outputs = trainer.model(voxels, voxel_batch)
        mask_probs = torch.sigmoid(outputs["mask"])
    else:
        outputs = None  # don't need to re-run

    logger.info(f"Predicted mask (sigmoid'd) shape: {tuple(mask_probs.shape)}")
    logger.info(f"  min={mask_probs.min():.4f} max={mask_probs.max():.4f} mean={mask_probs.mean():.4f}")
    if torch.isnan(mask_probs).any():
        logger.error("  [FAIL] Mask contains NaN!")
    elif torch.isinf(mask_probs).any():
        logger.error("  [FAIL] Mask contains Inf!")
    else:
        logger.info("  [OK] Mask is finite")

    logger.info("")
    logger.info("Loss breakdown:")
    loss_keys = ["prediction_loss", "rendering_loss", "agreement_loss",
                 "depth_smoothness_loss", "pose_temporal_loss",
                 "depth_temporal_loss", "residual_loss",
                 "mask_sparsity_loss", "mask_confidence_loss",
                 "dynamic_ratio", "grad_norm",
                 "photometric_loss"]
    for k in loss_keys:
        if k in loss_output:
            v = loss_output[k]
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                logger.info(f"  {k:30s} = {v.item():.6f}")
            else:
                logger.info(f"  {k:30s} = {v}")
    logger.info(f"  {'TOTAL LOSS':30s} = {total_loss.item():.6f}")

    n_with_grad = sum(1 for p in trainer.model.parameters()
                     if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in trainer.model.parameters())
    nan_grads = sum(1 for _, p in trainer.model.named_parameters()
                   if p.grad is not None and torch.isnan(p.grad).any())
    inf_grads = sum(1 for _, p in trainer.model.named_parameters()
                   if p.grad is not None and torch.isinf(p.grad).any())
    logger.info(f"  Params with grad: {n_with_grad} / {n_total}")
    logger.info(f"  NaN grads: {nan_grads}, Inf grads: {inf_grads}")

    # ============================================================
    # TensorBoard logging
    # ============================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("TENSORBOARD LOGGING (1 step)")
    logger.info("=" * 70)

    global_step = 0
    trainer.writer.add_scalar("test/total_loss", total_loss.item(), global_step)
    for k in loss_keys:
        if k in loss_output:
            v = loss_output[k]
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                trainer.writer.add_scalar(f"test/{k}", v.item(), global_step)

    trainer._log_mask_visualization(global_step, loss_output, raw_batch)
    logger.info(f"Mask visualization logged to {trainer.save_dir}/tb")

    trainer.writer.add_histogram(
        "test/mask_probs_histogram",
        mask_probs.detach().cpu(),
        global_step,
    )

    # ============================================================
    # Metrics
    # ============================================================
    if n_gt > 0:
        logger.info("")
        logger.info("=" * 70)
        logger.info("METRICS ON THIS BATCH (speed-aware GT)")
        logger.info("=" * 70)
        metrics = SegmentationMetrics(thresholds=[0.3, 0.4, 0.5, 0.6, 0.7])

        # Extract frame_motion for speed-aware dynamic mask generation
        last_frame = raw_batch.frames[-1]
        frame_motions = last_frame.frame_motion  # list[FrameMotion]

        # Show per-sample object speed summary (for verification)
        from src.utils.metrics import get_dynamic_object_ids
        for i, fm in enumerate(frame_motions):
            if fm is None:
                continue
            object_ids = list(fm.object_ids) if hasattr(fm.object_ids, '__iter__') else []
            if object_ids:
                dynamic_ids = get_dynamic_object_ids(fm)
                speeds = list(fm.speed) if hasattr(fm.speed, '__iter__') else []
                logger.info(
                    f"  Sample {i}: {len(object_ids)} objects, "
                    f"{len(dynamic_ids)} moving (threshold=0.05 m/s)"
                )
                for oid, sp in zip(object_ids, speeds):
                    marker = "MOVING" if sp > 0.05 else "static"
                    logger.info(f"    obj {oid}: speed={sp:.4f} m/s [{marker}]")

        valid = [(p, g, fm) for p, g, fm in zip(mask_probs, gt_masks, frame_motions)
                 if g is not None]
        if valid:
            valid_preds = torch.stack([p for p, _, _ in valid])
            valid_gts = [g for _, g, _ in valid]
            valid_fms = [fm for _, _, fm in valid]

            if valid_preds.shape[-2:] != valid_gts[0].shape:
                import torch.nn.functional as F
                valid_preds = F.interpolate(
                    valid_preds, size=valid_gts[0].shape,
                    mode="bilinear", align_corners=False,
                )

            # Pass frame_motions so metrics uses SPEED-AWARE GT
            metrics.update(valid_preds, valid_gts, frame_motions=valid_fms)
        results = metrics.compute()

        logger.info(f"Best IoU:  {results['best_iou']:.4f} (threshold={results['best_threshold']})")
        logger.info(f"Best F1:   {results['best_f1']:.4f}")
        logger.info(f"Best Prec: {results['best_precision']:.4f}")
        logger.info(f"Best Rec:  {results['best_recall']:.4f}")
        logger.info(f"Pred dynamic ratio: {results['pred_dynamic_ratio']:.4f}")
        logger.info(f"GT dynamic ratio:   {results['gt_dynamic_ratio']:.4f}")

        bl = TrivialBaselineMetrics().compute_baselines(valid_gts, frame_motions=valid_fms)
        logger.info("")
        logger.info("Trivial baselines (for reference, also speed-aware):")
        logger.info(f"  all_zeros:   IoU={bl['all_zeros']['iou']:.4f}")
        logger.info(f"  all_ones:    IoU={bl['all_ones']['iou']:.4f}")
        logger.info(f"  GT dynamic ratio (chance IoU ~ this): {bl['gt_dynamic_ratio']:.4f}")

        trainer.writer.add_scalar("test_batch/best_iou", results["best_iou"], 0)
        trainer.writer.add_scalar("test_batch/best_f1", results["best_f1"], 0)

    trainer.writer.close()

    # ============================================================
    # Summary
    # ============================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    checks = {
        "trainer_built": True,
        "forward_backward_succeeded": bool(torch.isfinite(total_loss).item()),
        "mask_is_finite": bool((not torch.isnan(mask_probs).any()) and (not torch.isinf(mask_probs).any())),
        "gradients_are_finite": (nan_grads == 0 and inf_grads == 0),
        "all_params_have_grad": (n_with_grad == n_total),
        "tensorboard_logged": (trainer.save_dir / "tb").exists(),
    }
    for k, v in checks.items():
        marker = "PASS" if v else "FAIL"
        logger.info(f"  [{marker}] {k}")

    if all(checks.values()):
        logger.info("")
        logger.info("=" * 70)
        logger.info("SUCCESS: SINGLE-BATCH TRAINER TEST PASSED")
        logger.info("=" * 70)
        logger.info(f"TensorBoard logs at: {trainer.save_dir}/tb")
        logger.info(f"To view: tensorboard --logdir {trainer.save_dir}/tb --port 6006")
        return 0
    else:
        logger.error("")
        logger.error("FAILED: SINGLE-BATCH TRAINER TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
