"""
Trainer v2 — for the direct residual mask model.

Simpler than v1 trainer:
- No loss scheduling (only 4 losses, no identity shortcut possible)
- No latent warmup (no latent loss)
- No residual warmup (mask IS the residual)
- Direct gradient flow from photometric to depth+pose+mask
"""

from __future__ import annotations

import os, time, math, logging
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.world_model_v2 import WorldModelV2
from src.losses.total_loss_v2 import TotalLossV2
from src.utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrainConfigV2:
    dataset_root: str = "/home/z/my-project/data/dataset_root"
    sensors: tuple = ("left_camera",)
    split: str = "train"
    val_split: str = "val"
    history_offsets: tuple = (-12, -8, -4, 0)
    num_bins: int = 5
    image_scale: float = 1.0
    event_channels: int = 256
    imu_hidden: int = 64
    imu_embedding: int = 128
    memory_type: str = "transformer"
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_eta_min: float = 1e-6
    mixed_precision: str = "bf16"
    grad_clip_max_norm: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    photometric_loss_weight: float = 10.0
    depth_smoothness_weight: float = 0.1  # low — don't push depth toward constant
    pose_temporal_weight: float = 1.0
    sparsity_loss_weight: float = 5.0
    depth_diversity_weight: float = 5.0
    residual_mask_weight: float = 0.5
    log_every_n_steps: int = 10
    viz_every_n_steps: int = 50
    viz_max_samples: int = 4
    eval_every_n_epochs: int = 5
    checkpoint_every_n_epochs: int = 5
    save_dir: str = "runs/exp_v2"
    seed: int = 42
    overfit_mode: bool = False
    gt_mask_sanity: bool = False
    resume_from: str | None = None


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])


class TrainerV2:
    def __init__(self, cfg: TrainConfigV2):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        import random, numpy as np
        random.seed(cfg.seed); np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)

        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "tb"))

        self.model = WorldModelV2(
            num_bins=cfg.num_bins,
            event_channels=cfg.event_channels,
            imu_hidden=cfg.imu_hidden,
            imu_embedding=cfg.imu_embedding,
            memory_type=cfg.memory_type,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model v2 parameters: {n_params:,} ({n_params/1e6:.2f}M)")

        self.loss_fn = TotalLossV2(
            photometric_loss_weight=cfg.photometric_loss_weight,
            depth_smoothness_weight=cfg.depth_smoothness_weight,
            pose_temporal_weight=cfg.pose_temporal_weight,
            sparsity_loss_weight=cfg.sparsity_loss_weight,
            depth_diversity_weight=cfg.depth_diversity_weight,
            residual_mask_weight=cfg.residual_mask_weight,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )

        # Build dataloaders before constructing the LR scheduler so the
        # scheduler uses the REAL number of optimizer steps.
        self._build_dataloaders()
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * cfg.epochs

        def lr_lambda(step):
            progress = step / max(1, total_steps)
            return cfg.scheduler_eta_min / cfg.learning_rate + \
                   (1 - cfg.scheduler_eta_min / cfg.learning_rate) * \
                   0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        self.ema = EMA(self.model, cfg.ema_decay) if cfg.use_ema else None
        self.amp_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else None

    def _build_dataloaders(self):
        from src.data.dataset import EVIMO2Dataset
        from src.data.temporal_dataset import TemporalEVIMO2Dataset
        from src.data.collate import temporal_collate_fn
        from src.data.transforms import Compose, ToTensor, NormalizeEventTime, NormalizeIMU, VoxelizeEvents

        ds = EVIMO2Dataset(
            dataset_root=self.cfg.dataset_root,
            sensors=self.cfg.sensors, split=self.cfg.split,
            load_depth=True, load_mask=True,
        )
        tds = TemporalEVIMO2Dataset(ds, history_offsets=self.cfg.history_offsets)
        logger.info(f"Train: {len(tds)} windows")
        self.train_loader = DataLoader(
            tds, batch_size=self.cfg.batch_size, shuffle=True,
            collate_fn=temporal_collate_fn,
            num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, drop_last=True,
        )
        self.val_loader = None
        if not self.cfg.overfit_mode and self.cfg.val_split:
            try:
                vds = EVIMO2Dataset(
                    dataset_root=self.cfg.dataset_root,
                    sensors=self.cfg.sensors, split=self.cfg.val_split,
                    load_depth=True, load_mask=True,
                )
                vtds = TemporalEVIMO2Dataset(vds, history_offsets=self.cfg.history_offsets)
                self.val_loader = DataLoader(
                    vtds, batch_size=self.cfg.batch_size, shuffle=False,
                    collate_fn=temporal_collate_fn,
                    num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
                )
            except Exception as e:
                logger.warning(f"No val loader: {e}")

        self.transform = Compose([
            ToTensor(), NormalizeEventTime(), NormalizeIMU(),
            VoxelizeEvents(num_bins=self.cfg.num_bins),
        ])

    def _build_gt_dynamic_mask(self, raw_batch, target_hw):
        """Build speed-aware binary EVIMO dynamic GT masks for valid samples."""
        from src.utils.metrics import (
            get_dynamic_object_ids,
            evimo2_mask_to_binary_dynamic,
        )

        last_frame = raw_batch.frames[-1]
        gt_masks = last_frame.mask
        frame_motions = last_frame.frame_motion

        gt_batch = []
        valid_indices = []

        for i, (gt_raw, frame_motion) in enumerate(zip(gt_masks, frame_motions)):
            if gt_raw is None:
                continue

            dynamic_ids = get_dynamic_object_ids(frame_motion)
            gt = evimo2_mask_to_binary_dynamic(gt_raw, dynamic_ids)

            while gt.ndim > 2:
                gt = gt.squeeze(0)

            gt = gt.float()

            if tuple(gt.shape[-2:]) != tuple(target_hw):
                gt = F.interpolate(
                    gt.unsqueeze(0).unsqueeze(0),
                    size=target_hw,
                    mode="nearest",
                ).squeeze(0).squeeze(0)

            gt_batch.append(gt)
            valid_indices.append(i)

        if not gt_batch:
            return None, []

        gt_batch = torch.stack(gt_batch, dim=0).unsqueeze(1).to(self.device)
        return gt_batch, valid_indices

    def train(self):
        cfg = self.cfg
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * cfg.epochs
        logger.info(f"Steps/epoch: {steps_per_epoch}, total: {total_steps}")
        global_step = 0

        for epoch in range(cfg.epochs):
            self.model.train()
            t0 = time.time()
            for batch_idx, raw_batch in enumerate(self.train_loader):
                voxel_batch = self.transform(raw_batch)
                voxels = torch.stack([f.voxel_grid for f in voxel_batch.frames], dim=1).to(self.device)
                voxel_batch = voxel_batch.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                if self.cfg.gt_mask_sanity:
                    # GT-mask sanity test: directly supervise the predicted mask
                    # with the speed-aware EVIMO ground-truth dynamic mask.
                    outputs = self.model(voxels, voxel_batch)
                    gt_mask, valid_indices = self._build_gt_dynamic_mask(
                        raw_batch, outputs["mask"].shape[-2:]
                    )

                    if gt_mask is None:
                        logger.warning(
                            "Skipping batch %d: no GT masks available for this batch.",
                            batch_idx,
                        )
                        continue

                    pred_mask = outputs["mask"][valid_indices]
                    total_loss = F.binary_cross_entropy_with_logits(
                        pred_mask, gt_mask
                    )
                    dynamic_ratio = torch.sigmoid(pred_mask).mean().detach()
                    loss_output = {
                        "loss": total_loss,
                        "gt_mask_loss": total_loss.detach(),
                        "dynamic_ratio": dynamic_ratio,
                    }
                elif self.amp_dtype:
                    with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                        outputs = self.model(voxels, voxel_batch)
                        loss_output = self.loss_fn(
                            outputs, inputs={"voxel_grid": voxels}
                        )
                        total_loss = loss_output["loss"]
                else:
                    outputs = self.model(voxels, voxel_batch)
                    loss_output = self.loss_fn(
                        outputs, inputs={"voxel_grid": voxels}
                    )
                    total_loss = loss_output["loss"]

                if not torch.isfinite(total_loss):
                    logger.error(f"Non-finite loss: {total_loss.item()}")
                    continue

                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip_max_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                if self.ema: self.ema.update(self.model)

                if global_step % cfg.log_every_n_steps == 0:
                    self._log_step(epoch, batch_idx, global_step, total_loss, loss_output, grad_norm)

                if cfg.viz_every_n_steps > 0 and global_step % cfg.viz_every_n_steps == 0:
                    self._log_viz(global_step, outputs, loss_output, raw_batch)

                global_step += 1

            dt = time.time() - t0
            logger.info(f"Epoch {epoch+1}/{cfg.epochs} done in {dt:.1f}s, loss={total_loss.item():.4f}")

            if (epoch + 1) % cfg.eval_every_n_epochs == 0:
                if self.val_loader:
                    iou = self._evaluate(self.val_loader, f"val/epoch_{epoch+1}")
                elif cfg.overfit_mode:
                    iou = self._evaluate(self.train_loader, "overfit_eval")
                logger.info(f"Epoch {epoch+1} IoU: {iou:.4f}")

            if (epoch + 1) % cfg.checkpoint_every_n_epochs == 0:
                self._save(epoch)

        self._save(cfg.epochs - 1)
        self.writer.close()

    def _log_step(self, epoch, batch_idx, gs, total_loss, lo, gn):
        self.writer.add_scalar("train/total_loss", total_loss.item(), gs)
        for k in ["photometric_loss", "depth_smoothness_loss", "pose_temporal_loss",
                   "sparsity_loss", "dynamic_ratio", "depth_diversity",
                   "depth_diversity_loss"]:
            if k in lo:
                v = lo[k]
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    self.writer.add_scalar(f"train/{k}", v.item(), gs)
        self.writer.add_scalar("train/grad_norm", gn.item(), gs)
        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], gs)

        if batch_idx == 0 or batch_idx % (self.cfg.log_every_n_steps * 5) == 0:
            dd = lo.get("depth_diversity", torch.tensor(0))
            if isinstance(dd, torch.Tensor):
                dd_val = dd.item()
            else:
                dd_val = float(dd)
            if self.cfg.gt_mask_sanity:
                logger.info(
                    f"E{epoch+1} B{batch_idx:4d} | "
                    f"loss={total_loss.item():.4f} | "
                    f"gt_mask={lo.get('gt_mask_loss', torch.tensor(0.0)).item():.4f} | "
                    f"dr={lo.get('dynamic_ratio', torch.tensor(0.0)).item():.3f} | "
                    f"gn={gn.item():.2f}"
                )
            else:
                logger.info(
                    f"E{epoch+1} B{batch_idx:4d} | "
                    f"loss={total_loss.item():.4f} | "
                    f"photo={lo.get('photometric_loss', torch.tensor(0)).item():.4f} | "
                    f"ds={lo.get('depth_smoothness_loss', torch.tensor(0)).item():.4f} | "
                    f"dvar={dd_val:.3f} | "
                    f"dr={lo.get('dynamic_ratio', torch.tensor(0)).item():.3f} | "
                    f"gn={gn.item():.2f}"
                )

    @torch.no_grad()
    def _log_viz(self, gs, outputs, lo, raw_batch):
        from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic
        mp = lo.get("mask_probs")
        if mp is not None:
            mask_probs = mp
        else:
            mask_probs = torch.sigmoid(outputs["mask"])
        mask_probs = mask_probs.float().cpu()
        max_n = min(self.cfg.viz_max_samples, mask_probs.shape[0])
        gt_masks = raw_batch.frames[-1].mask
        fms = raw_batch.frames[-1].frame_motion
        residual = outputs.get("residual")
        if residual is not None:
            residual = residual.float().cpu()

        for i in range(max_n):
            pred = mask_probs[i, 0]
            pseudo = residual[i, 0] if residual is not None and i < residual.shape[0] else torch.zeros_like(pred)
            gt_raw = gt_masks[i] if i < len(gt_masks) else None
            fm = fms[i] if i < len(fms) else None
            if gt_raw is not None:
                ids = get_dynamic_object_ids(fm)
                from src.utils.metrics import evimo2_mask_to_binary_dynamic
                gt = evimo2_mask_to_binary_dynamic(gt_raw, ids)
                while gt.ndim > 2: gt = gt.squeeze(0)
                if gt.shape != pred.shape:
                    gt = F.interpolate(gt.float().unsqueeze(0).unsqueeze(0), size=pred.shape, mode="nearest").squeeze().bool()
            else:
                gt = torch.zeros_like(pred, dtype=torch.bool)

            self.writer.add_image(f"viz/s{i}/1_residual", pseudo.unsqueeze(0), gs, dataformats="CHW")
            self.writer.add_image(f"viz/s{i}/2_pred_mask", pred.unsqueeze(0), gs, dataformats="CHW")
            self.writer.add_image(f"viz/s{i}/3_gt_mask", gt.float().unsqueeze(0), gs, dataformats="CHW")
            ov = torch.zeros(3, pred.shape[0], pred.shape[1])
            ov[0] = pred; ov[1] = gt.float(); ov[2] = (pred > 0.5) & gt
            self.writer.add_image(f"viz_overlay/s{i}", ov, gs, dataformats="CHW")

    @torch.no_grad()
    def _evaluate(self, loader, tag):
        self.model.eval()
        if self.ema:
            orig = {n: p.data.clone() for n, p in self.model.named_parameters()}
            self.ema.apply_to(self.model)
        metrics = SegmentationMetrics([0.3, 0.4, 0.5, 0.6, 0.7])
        n = 0
        for raw in loader:
            vb = self.transform(raw)
            vox = torch.stack([f.voxel_grid for f in vb.frames], dim=1).to(self.device)
            vb = vb.to(self.device)
            out = self.model(vox, vb)
            probs = torch.sigmoid(out["mask"])
            gts = raw.frames[-1].mask
            fms = raw.frames[-1].frame_motion
            valid = [(p, g, fm) for p, g, fm in zip(probs, gts, fms) if g is not None]
            if not valid: continue
            vp = torch.stack([p for p, _, _ in valid])
            vg = [g for _, g, _ in valid]
            vf = [fm for _, _, fm in valid]
            if vp.shape[-2:] != vg[0].shape:
                vp = F.interpolate(vp, size=vg[0].shape, mode="bilinear", align_corners=False)
            metrics.update(vp, vg, frame_motions=vf)
            n += len(valid)
        r = metrics.compute()
        self.writer.add_scalar(f"{tag}/best_iou", r["best_iou"], 0)
        self.writer.add_scalar(f"{tag}/best_f1", r["best_f1"], 0)
        logger.info(f"[{tag}] n={n} IoU={r['best_iou']:.4f} F1={r['best_f1']:.4f} dr={r['pred_dynamic_ratio']:.4f}")
        if self.ema:
            for n2, p in self.model.named_parameters():
                if n2 in orig: p.data.copy_(orig[n2])
        return r["best_iou"]

    def _save(self, epoch):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg.__dict__,
        }, self.save_dir / "checkpoints" / f"epoch_{epoch+1:03d}.pth")
        logger.info(f"Saved checkpoint: epoch_{epoch+1:03d}.pth")
