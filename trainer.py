"""
Trainer for the self-supervised event-camera world model.

Optimized for a single RTX 4070 (12GB VRAM).

Features
--------
- Mixed precision (bf16 by default — more stable than fp16, no
  GradScaler needed, supported on Ampere+)
- Gradient clipping (max_norm=1.0) — MANDATORY due to axis-angle
  rotation instability
- EMA of model weights (decay 0.999) — stabilizes self-supervised
  training, used for validation
- Per-loss TensorBoard logging (every component logged separately)
- Mask visualization logging (pred / pseudo-label / GT / overlay)
- Loss warmup: residual_loss_weight ramps 0 -> 1.0 over first 10%
  of training (residuals are noise at step 0)
- Gradient norm logging
- Checkpointing every N epochs + best_model by val IoU
- Configurable image downscale for fast iteration
- Configurable gradient checkpointing
- Optional overfit mode (single sequence, no val) for debugging

Memory budget (RTX 4070 12GB, bf16, B=2, T=4):
- Full res (480x640):   ~5GB activations + 180MB params/optimizer
                        = ~5.5GB. Fits with margin.
- Half res (240x320):   ~1.5GB activations. B=4-6 easy.

Important note about mixed precision:
  The mask head outputs LOGITS (no sigmoid). The DynamicResidualLoss
  uses binary_cross_entropy_with_logits, which is autocast-safe.
  When we need probabilities (for metrics, visualization, IoU), we
  explicitly call torch.sigmoid(logits).
"""

from __future__ import annotations

import os
import time
import math
import copy
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.world_model.model import WorldModel
from src.losses.total_loss import TotalLoss
from src.utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


# ==========================================================
# Config
# ==========================================================

@dataclass
class TrainConfig:
    """Configuration for the trainer."""

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    dataset_root: str = "/home/z/my-project/data/dataset_root"
    sensors: tuple = ("left_camera",)
    split: str = "train"
    val_split: str = "val"

    history_offsets: tuple = (-3, -2, -1, 0)
    num_bins: int = 5
    image_scale: float = 1.0

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    event_channels: int = 256
    imu_hidden: int = 64
    imu_embedding: int = 128
    decoder_channels: int = 16
    memory_type: str = "transformer"  # 'transformer', 'convlstm', 'convgru'

    # --------------------------------------------------
    # Optimization
    # --------------------------------------------------
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True

    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 0
    scheduler_eta_min: float = 1e-6

    mixed_precision: str = "bf16"  # 'bf16', 'fp16', or 'none'
    grad_clip_max_norm: float = 1.0
    use_gradient_checkpointing: bool = False

    # --------------------------------------------------
    # EMA
    # --------------------------------------------------
    use_ema: bool = True
    ema_decay: float = 0.999

    # --------------------------------------------------
    # Loss warmup
    # --------------------------------------------------
    residual_warmup_fraction: float = 0.1
    photometric_loss_weight: float = 10.0

    # --------------------------------------------------
    # Loss scheduling — phase in losses gradually
    # --------------------------------------------------
    # Phase 1 (0 to latent_start): ONLY photometric + depth_smoothness
    #   → depth+pose learn without identity shortcut competition
    # Phase 2 (latent_start to residual_start): + latent consistency
    #   → transition starts learning (now that depth/pose work)
    # Phase 3 (residual_start to end): + residual + mask regularization
    #   → mask learning starts (now that renderer is converged)
    #
    # Default: latent starts at 20%, residual at 40%
    latent_warmup_fraction: float = 0.2
    residual_phase_start: float = 0.4

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    log_every_n_steps: int = 10
    viz_every_n_steps: int = 50
    viz_max_samples: int = 4
    eval_every_n_epochs: int = 5
    checkpoint_every_n_epochs: int = 5
    save_dir: str = "/home/z/my-project/work/runs/exp001"

    seed: int = 42
    overfit_mode: bool = False
    resume_from: str | None = None


# ==========================================================
# Helpers
# ==========================================================

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_scheduler(optimizer, cfg, steps_per_epoch):
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return cfg.scheduler_eta_min / cfg.learning_rate + \
               (1 - cfg.scheduler_eta_min / cfg.learning_rate) * \
               0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    p.detach(), alpha=1.0 - self.decay
                )

    def apply_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])


def scale_voxel_batch(voxels: torch.Tensor, scale: float) -> torch.Tensor:
    if scale >= 1.0:
        return voxels
    B, T, C, H, W = voxels.shape
    new_h, new_w = int(H * scale), int(W * scale)
    return F.interpolate(
        voxels.view(B * T, C, H, W),
        size=(new_h, new_w),
        mode="area",
    ).view(B, T, C, new_h, new_w)


# ==========================================================
# Trainer
# ==========================================================

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"VRAM: {mem_gb:.1f} GB")

        set_seed(cfg.seed)

        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "tb"))

        # Build model
        self.model = WorldModel(
            num_bins=cfg.num_bins,
            event_channels=cfg.event_channels,
            imu_hidden=cfg.imu_hidden,
            imu_embedding=cfg.imu_embedding,
            decoder_channels=cfg.decoder_channels,
            memory_type=cfg.memory_type,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

        # Loss
        self.loss_fn = TotalLoss(
            photometric_loss_weight=cfg.photometric_loss_weight,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = None

        # EMA
        self.ema = EMA(self.model, decay=cfg.ema_decay) if cfg.use_ema else None

        # AMP setup
        if cfg.mixed_precision == "bf16":
            self.amp_dtype = torch.bfloat16
            self.use_scaler = False
        elif cfg.mixed_precision == "fp16":
            self.amp_dtype = torch.float16
            self.use_scaler = True
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.amp_dtype = None
            self.use_scaler = False

        # Data
        self._build_dataloaders()

        # Resume
        self.start_epoch = 0
        self.best_iou = 0.0
        if cfg.resume_from is not None:
            self._load_checkpoint(cfg.resume_from)

    def _build_dataloaders(self):
        from src.data.dataset import EVIMO2Dataset
        from src.data.temporal_dataset import TemporalEVIMO2Dataset
        from src.data.collate import temporal_collate_fn
        from src.data.transforms import (
            Compose, ToTensor, NormalizeEventTime, NormalizeIMU, VoxelizeEvents,
        )

        logger.info(f"Building train dataset from {self.cfg.dataset_root}")
        train_ds = EVIMO2Dataset(
            dataset_root=self.cfg.dataset_root,
            sensors=self.cfg.sensors,
            split=self.cfg.split,
            load_depth=True,
            load_mask=True,
        )
        train_tds = TemporalEVIMO2Dataset(
            train_ds, history_offsets=self.cfg.history_offsets,
        )
        logger.info(f"Train: {len(train_tds)} temporal windows")

        self.train_loader = DataLoader(
            train_tds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=temporal_collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=True,
        )

        self.val_loader = None
        if not self.cfg.overfit_mode and self.cfg.val_split:
            try:
                val_ds = EVIMO2Dataset(
                    dataset_root=self.cfg.dataset_root,
                    sensors=self.cfg.sensors,
                    split=self.cfg.val_split,
                    load_depth=True,
                    load_mask=True,
                )
                val_tds = TemporalEVIMO2Dataset(
                    val_ds, history_offsets=self.cfg.history_offsets,
                )
                logger.info(f"Val: {len(val_tds)} temporal windows")
                self.val_loader = DataLoader(
                    val_tds,
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    collate_fn=temporal_collate_fn,
                    num_workers=self.cfg.num_workers,
                    pin_memory=self.cfg.pin_memory,
                    drop_last=False,
                )
            except Exception as e:
                logger.warning(f"Could not build val loader: {e}")
                self.val_loader = None

        self.transform = Compose([
            ToTensor(),
            NormalizeEventTime(),
            NormalizeIMU(),
            VoxelizeEvents(num_bins=self.cfg.num_bins),
        ])

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    def train(self):
        cfg = self.cfg
        steps_per_epoch = len(self.train_loader)
        self.scheduler = get_lr_scheduler(
            self.optimizer, cfg, steps_per_epoch,
        )
        total_steps = steps_per_epoch * cfg.epochs
        logger.info(f"Steps per epoch: {steps_per_epoch}, total: {total_steps}")

        global_step = self.start_epoch * steps_per_epoch

        for epoch in range(self.start_epoch, cfg.epochs):
            self.model.train()
            epoch_start = time.time()

            for batch_idx, raw_batch in enumerate(self.train_loader):
                voxel_batch = self.transform(raw_batch)
                voxels = torch.stack(
                    [f.voxel_grid for f in voxel_batch.frames], dim=1
                ).to(self.device, non_blocking=True)
                voxel_batch = voxel_batch.to(self.device)

                if cfg.image_scale < 1.0:
                    voxels = scale_voxel_batch(voxels, cfg.image_scale)

                # ------------------------------------------------
                # LOSS SCHEDULING: phase in losses gradually
                # ------------------------------------------------
                progress = global_step / total_steps if total_steps > 0 else 1.0

                # Phase 1 (0 → latent_warmup_fraction): photometric only
                #   depth+pose learn WITHOUT latent shortcut competition
                latent_weight = min(
                    1.0,
                    max(0.0, (progress - cfg.latent_warmup_fraction) / 0.1)
                ) if progress > cfg.latent_warmup_fraction else 0.0

                # Phase 2 (latent_warmup → residual_phase_start):
                #   latent consistency joins
                # Phase 3 (residual_phase_start → end):
                #   residual + mask regularization join
                residual_weight = min(
                    1.0,
                    max(0.0, (progress - cfg.residual_phase_start) / 0.1)
                ) if progress > cfg.residual_phase_start else 0.0

                # Apply weights
                self.loss_fn.prediction_loss_weight = float(latent_weight)
                self.loss_fn.rendering_loss_weight = float(latent_weight)
                self.loss_fn.agreement_loss_weight = float(latent_weight)
                self.loss_fn.residual_loss_weight = float(residual_weight)
                self.loss_fn.sparsity_loss_weight = float(5.0 * residual_weight)
                self.loss_fn.confidence_loss_weight = float(0.1 * residual_weight)
                self.loss_fn.depth_temporal_weight = float(latent_weight)
                self.loss_fn.pose_temporal_weight = float(latent_weight)
                # CRITICAL: depth_smoothness must also be scheduled.
                # In P1 (photo only), depth_smoothness=0 so depth is FREE
                # to develop spatial structure. If smoothness is active in P1,
                # it pushes depth toward constant (killing the gradient).
                self.loss_fn.depth_smoothness_weight = float(latent_weight)

                loss_output, total_loss = self._forward_backward(
                    voxels, voxel_batch
                )

                # Store schedule info for logging
                loss_output["schedule_latent"] = float(latent_weight)
                loss_output["schedule_residual"] = float(residual_weight)
                loss_output["schedule_progress"] = float(progress)

                self.optimizer.step()
                self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)

                if global_step % cfg.log_every_n_steps == 0:
                    self._log_step(
                        epoch, batch_idx, global_step,
                        total_loss, loss_output, residual_weight,
                    )

                if cfg.viz_every_n_steps > 0 and \
                        global_step % cfg.viz_every_n_steps == 0:
                    self._log_mask_visualization(
                        global_step, loss_output, raw_batch,
                    )

                global_step += 1

            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{cfg.epochs} done in {epoch_time:.1f}s, "
                f"last_loss={total_loss.item():.4f}, "
                f"lr={self.scheduler.get_last_lr()[0]:.6f}"
            )

            val_iou = 0.0
            if (epoch + 1) % cfg.eval_every_n_epochs == 0 and \
                    self.val_loader is not None:
                val_iou = self.validate(epoch)
                if val_iou > self.best_iou:
                    self.best_iou = val_iou
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(f"New best IoU: {val_iou:.4f}")
            elif cfg.overfit_mode and (epoch + 1) % cfg.eval_every_n_epochs == 0:
                val_iou = self._evaluate_on_loader(self.train_loader, "overfit_eval")
                logger.info(f"[Overfit] Train IoU after epoch {epoch+1}: {val_iou:.4f}")

            if (epoch + 1) % cfg.checkpoint_every_n_epochs == 0:
                self._save_checkpoint(epoch, is_best=False)

        self._save_checkpoint(cfg.epochs - 1, is_best=False)
        self.writer.close()
        logger.info(f"Training complete. Best val IoU: {self.best_iou:.4f}")

    def _forward_backward(self, voxels, voxel_batch):
        cfg = self.cfg
        self.optimizer.zero_grad(set_to_none=True)

        if self.amp_dtype is not None:
            # Use autocast for forward. The DynamicResidualLoss uses
            # binary_cross_entropy_with_logits which is autocast-safe.
            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                outputs = self.model(voxels, voxel_batch)
                loss_output = self.loss_fn(
                    outputs=outputs,
                    inputs={"voxel_grid": voxels},
                )
                total_loss = loss_output["loss"]
        else:
            outputs = self.model(voxels, voxel_batch)
            loss_output = self.loss_fn(
                outputs=outputs,
                inputs={"voxel_grid": voxels},
            )
            total_loss = loss_output["loss"]

        if not torch.isfinite(total_loss):
            logger.error(f"Non-finite loss: {total_loss.item()}. Skipping batch.")
            return loss_output, total_loss.detach()

        if self.use_scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip_max_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip_max_norm,
            )

        loss_output["grad_norm"] = grad_norm.detach()
        return loss_output, total_loss

    def _log_step(self, epoch, batch_idx, global_step,
                  total_loss, loss_output, warmup_frac):
        self.writer.add_scalar("train/total_loss", total_loss.item(), global_step)
        for key in ["prediction_loss", "rendering_loss", "agreement_loss",
                    "depth_smoothness_loss", "pose_temporal_loss",
                    "depth_temporal_loss", "residual_loss", "dynamic_mask_loss",
                    "photometric_loss"]:
            if key in loss_output:
                v = loss_output[key]
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    self.writer.add_scalar(f"train/{key}", v.item(), global_step)
        if "grad_norm" in loss_output:
            self.writer.add_scalar("train/grad_norm",
                                    loss_output["grad_norm"].item(), global_step)
        self.writer.add_scalar("train/residual_warmup_factor", warmup_frac, global_step)
        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], global_step)
        if "dynamic_ratio" in loss_output:
            self.writer.add_scalar("train/dynamic_ratio",
                                    loss_output["dynamic_ratio"].item(), global_step)
        if "pseudo_label" in loss_output:
            self.writer.add_scalar("train/pseudo_label_mean",
                                    loss_output["pseudo_label"].mean().item(), global_step)
        # Log noise detection status
        if "is_noise" in loss_output:
            self.writer.add_scalar("train/residual_is_noise",
                                    float(loss_output["is_noise"]), global_step)
        if "pseudo_mean" in loss_output:
            self.writer.add_scalar("train/pseudo_label_mean_raw",
                                    loss_output["pseudo_mean"], global_step)
        # Log schedule
        if "schedule_latent" in loss_output:
            self.writer.add_scalar("train/schedule_latent_weight",
                                    loss_output["schedule_latent"], global_step)
        if "schedule_residual" in loss_output:
            self.writer.add_scalar("train/schedule_residual_weight",
                                    loss_output["schedule_residual"], global_step)

        if batch_idx == 0 or batch_idx % (self.cfg.log_every_n_steps * 5) == 0:
            is_noise_str = "NOISE" if loss_output.get("is_noise", False) else "ok"
            lat_w = loss_output.get("schedule_latent", 0)
            res_w = loss_output.get("schedule_residual", 0)
            if res_w > 0:
                phase = "P3:full"
            elif lat_w > 0:
                phase = "P2:latent"
            else:
                phase = "P1:photo"
            logger.info(
                f"E{epoch+1} B{batch_idx:4d} | "
                f"loss={total_loss.item():.4f} | "
                f"photo={loss_output.get('photometric_loss', torch.tensor(0)).item():.4f} | "
                f"res={loss_output.get('residual_loss', torch.tensor(0)).item():.4f} | "
                f"latent={loss_output.get('latent_loss', torch.tensor(0)).item():.4f} | "
                f"dm={loss_output.get('dynamic_mask_loss', torch.tensor(0)).item():.4f} | "
                f"dr={loss_output.get('dynamic_ratio', torch.tensor(0)).item():.3f} | "
                f"gn={loss_output.get('grad_norm', torch.tensor(0)).item():.2f} | "
                f"phase={phase}({lat_w:.1f},{res_w:.1f}) | "
                f"residual={is_noise_str}"
            )

    # --------------------------------------------------
    # Mask visualization
    # --------------------------------------------------

    @torch.no_grad()
    def _log_mask_visualization(self, global_step, loss_output, raw_batch):
        """Log mask visualization to TensorBoard.

        Uses the sigmoid'd mask_probs from loss_output (which is
        what DynamicResidualLoss computes internally for logging).

        GT dynamic mask is computed using SPEED-AWARE selection:
        only objects with speed > MOTION_THRESHOLD_SPEED are marked
        as dynamic in the GT. This matches the technique used in
        tools/visualization/dataset_verification.py:render_motion_mask().
        """
        from src.utils.metrics import (
            get_dynamic_object_ids,
            evimo2_mask_to_binary_dynamic,
        )

        cfg = self.cfg

        if "mask_probs" not in loss_output or loss_output["mask_probs"] is None:
            return

        # mask_probs: (B, 1, H, W) in [0, 1] -- sigmoid of logits
        mask_probs = loss_output["mask_probs"].float().cpu()
        max_n = min(cfg.viz_max_samples, mask_probs.shape[0])

        # Get pseudo_label from loss_output too (for visualization)
        pseudo_label = None
        if "pseudo_label" in loss_output:
            pseudo_label = loss_output["pseudo_label"].float().cpu()

        # Extract the LAST frame's GT mask AND frame_motion
        last_frame = raw_batch.frames[-1]
        gt_masks_raw = last_frame.mask
        frame_motions = last_frame.frame_motion  # list[FrameMotion]

        for i in range(max_n):
            pred = mask_probs[i, 0]  # (H, W) in [0, 1]

            if pseudo_label is not None and i < pseudo_label.shape[0]:
                pseudo = pseudo_label[i, 0]
            else:
                pseudo = torch.zeros_like(pred)

            gt_raw = gt_masks_raw[i] if i < len(gt_masks_raw) else None
            frame_motion_i = frame_motions[i] if i < len(frame_motions) else None

            if gt_raw is not None:
                # SPEED-AWARE dynamic mask generation
                # Get the set of object IDs that are actually moving
                dynamic_ids = get_dynamic_object_ids(frame_motion_i)

                # Brief log on first sample (helpful for verification)
                if i == 0:
                    n_total = len(frame_motion_i.object_ids) if frame_motion_i is not None else 0
                    n_moving = len(dynamic_ids)
                    logger.info(
                        f"[viz] sample 0: {n_moving}/{n_total} objects moving "
                        f"(threshold={0.05} m/s)"
                    )

                # Convert GT mask to binary dynamic mask.
                # evimo2_mask_to_binary_dynamic returns (B, H, W) or (1, H, W).
                # For per-sample viz, we want (H, W) matching pred.
                gt_binary = evimo2_mask_to_binary_dynamic(
                    gt_raw, dynamic_ids,
                )
                # Squeeze any leading batch dim -> (H, W)
                while gt_binary.ndim > 2:
                    gt_binary = gt_binary.squeeze(0)

                if gt_binary.shape != pred.shape:
                    # F.interpolate needs (N, C, H, W) input
                    gt_binary = F.interpolate(
                        gt_binary.float().unsqueeze(0).unsqueeze(0),
                        size=pred.shape,
                        mode="nearest",
                    ).squeeze().bool()
            else:
                gt_binary = torch.zeros_like(pred, dtype=torch.bool)

            # Scene image (use mask_probs as a stand-in for spatial structure)
            scene = pred.clone()

            self.writer.add_image(
                f"viz/sample_{i}/1_scene_mask",
                scene.unsqueeze(0),
                global_step, dataformats="CHW",
            )
            self.writer.add_image(
                f"viz/sample_{i}/2_pred_mask",
                pred.unsqueeze(0),
                global_step, dataformats="CHW",
            )
            self.writer.add_image(
                f"viz/sample_{i}/3_pseudo_label",
                pseudo.unsqueeze(0),
                global_step, dataformats="CHW",
            )
            self.writer.add_image(
                f"viz/sample_{i}/4_gt_dynamic_mask",
                gt_binary.float().unsqueeze(0),
                global_step, dataformats="CHW",
            )

            # Overlay: R=pred, G=GT, B=intersection
            overlay = torch.zeros(3, pred.shape[0], pred.shape[1])
            overlay[0] = pred
            overlay[1] = gt_binary.float()
            overlay[2] = (pred > 0.5) & gt_binary
            self.writer.add_image(
                f"viz_overlay/sample_{i}",
                overlay, global_step, dataformats="CHW",
            )

        n_gt = sum(1 for m in gt_masks_raw[:max_n] if m is not None)
        self.writer.add_scalar("viz/n_gt_available", n_gt, global_step)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        if self.val_loader is None:
            return 0.0
        return self._evaluate_on_loader(self.val_loader, f"val/epoch_{epoch+1}")

    @torch.no_grad()
    def _evaluate_on_loader(self, loader, tag: str) -> float:
        cfg = self.cfg
        self.model.eval()

        if self.ema is not None:
            original_state = {
                name: p.data.clone()
                for name, p in self.model.named_parameters()
            }
            self.ema.apply_to(self.model)

        metrics = SegmentationMetrics(
            thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]
        )

        n_samples = 0
        for raw_batch in loader:
            voxel_batch = self.transform(raw_batch)
            voxels = torch.stack(
                [f.voxel_grid for f in voxel_batch.frames], dim=1
            ).to(self.device)
            voxel_batch = voxel_batch.to(self.device)

            if cfg.image_scale < 1.0:
                voxels = scale_voxel_batch(voxels, cfg.image_scale)

            if self.amp_dtype is not None:
                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    outputs = self.model(voxels, voxel_batch)
            else:
                outputs = self.model(voxels, voxel_batch)

            # mask is LOGITS -> apply sigmoid for metrics
            pred_probs = torch.sigmoid(outputs["mask"])

            # Extract BOTH the GT mask AND the per-sample frame_motion
            # (needed for speed-aware dynamic mask generation)
            last_frame = raw_batch.frames[-1]
            gt_masks = last_frame.mask
            frame_motions = last_frame.frame_motion  # list[FrameMotion]

            valid = [(p, g, fm)
                     for p, g, fm in zip(pred_probs, gt_masks, frame_motions)
                     if g is not None]
            if not valid:
                continue

            valid_preds = torch.stack([p for p, _, _ in valid])
            valid_gts = [g for _, g, _ in valid]
            valid_fms = [fm for _, _, fm in valid]

            if valid_preds.shape[-2:] != valid_gts[0].shape:
                valid_preds = F.interpolate(
                    valid_preds,
                    size=valid_gts[0].shape,
                    mode="bilinear",
                    align_corners=False,
                )

            # Pass frame_motions so the metrics use SPEED-AWARE dynamic
            # mask generation (only objects with speed > threshold are
            # considered "dynamic" in the GT)
            metrics.update(valid_preds, valid_gts, frame_motions=valid_fms)
            n_samples += len(valid)

        results = metrics.compute()

        self.writer.add_scalar(f"{tag}/best_iou", results["best_iou"], 0)
        self.writer.add_scalar(f"{tag}/best_f1", results["best_f1"], 0)
        self.writer.add_scalar(f"{tag}/best_precision", results["best_precision"], 0)
        self.writer.add_scalar(f"{tag}/best_recall", results["best_recall"], 0)
        self.writer.add_scalar(f"{tag}/best_threshold", results["best_threshold"], 0)
        self.writer.add_scalar(f"{tag}/pred_dynamic_ratio",
                               results["pred_dynamic_ratio"], 0)
        self.writer.add_scalar(f"{tag}/gt_dynamic_ratio",
                               results["gt_dynamic_ratio"], 0)

        for thr_key, thr_metrics in results["per_threshold"].items():
            self.writer.add_scalar(
                f"{tag}_per_threshold/{thr_key}_iou",
                thr_metrics["iou"], 0
            )

        logger.info(
            f"[{tag}] samples={n_samples} | "
            f"best_IoU={results['best_iou']:.4f} (thr={results['best_threshold']:.2f}) | "
            f"F1={results['best_f1']:.4f} | "
            f"P={results['best_precision']:.4f} R={results['best_recall']:.4f} | "
            f"pred_dr={results['pred_dynamic_ratio']:.4f} gt_dr={results['gt_dynamic_ratio']:.4f}"
        )

        if self.ema is not None:
            for name, p in self.model.named_parameters():
                if name in original_state:
                    p.data.copy_(original_state[name])

        return results["best_iou"]

    # --------------------------------------------------
    # Checkpointing
    # --------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_iou": self.best_iou,
            "config": self.cfg.__dict__,
        }
        if self.ema is not None:
            state["ema_state_dict"] = {
                k: v.cpu() for k, v in self.ema.shadow.items()
            }

        path = self.save_dir / "checkpoints" / f"epoch_{epoch+1:03d}.pth"
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.save_dir / "checkpoints" / "best_model.pth"
            torch.save(state, best_path)
            logger.info(f"Saved BEST model: {best_path}")

    def _load_checkpoint(self, path: str):
        logger.info(f"Loading checkpoint: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        self.start_epoch = state.get("epoch", 0)
        self.best_iou = state.get("best_iou", 0.0)

        if self.ema is not None and state.get("ema_state_dict"):
            for name, tensor in state["ema_state_dict"].items():
                if name in self.ema.shadow:
                    self.ema.shadow[name] = tensor.to(self.device)

        logger.info(f"Resumed from epoch {self.start_epoch}, best_iou={self.best_iou:.4f}")
