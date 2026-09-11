"""
Trainer for GTMaskModel — mask prediction from GROUND-TRUTH depth and
GROUND-TRUTH pose only (no learned geometry anywhere in the loop).

Deliberately does NOT reuse TrainerV2/TrainerV3: those trainers are
built around the self-supervised photometric pipeline (predicted
depth/pose, EMA over a jointly-optimized geometry network, loss
warmup schedules for an ill-posed objective). None of that applies
here -- this is a plain supervised-segmentation training loop. Reusing
their machinery would mean carrying dead config knobs (photometric
weight, depth smoothness, pose temporal consistency, etc.) that don't
apply to a model with no depth/pose network.

Data requirement
----------------
This model only needs a PAIR of consecutive frames (t-1, t), not the
longer history window WorldModelV2/V3 use for their ConvGRU/
transformer temporal fusion (which don't exist here since there's no
learned depth to fuse over time). We reuse TemporalEVIMO2Dataset with
history_offsets=(-N, 0) for a configurable frame gap N (default 1 =
adjacent frames), which keeps the exact same dataset/cache/collate
code path as the rest of the repo (so results are directly
comparable) while trimming the window to what this model actually
needs.
"""

from __future__ import annotations

import os, time, math, logging
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.gt_mask_model.model import GTMaskModel
from src.models.gt_mask_model.loss import GTMaskLoss
from src.models.gt_mask_model.gt_depth_utils import build_gt_depth_batch
from src.utils.metrics import (
    SegmentationMetrics,
    get_dynamic_object_ids,
    evimo2_mask_to_binary_dynamic,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfigGTMask:
    dataset_root: str = "/home/z/my-project/data/dataset_root"
    sensors: tuple = ("left_camera",)
    split: str = "train"
    subset: str = "imo"
    sequence: tuple | None = None
    val_split: str = "val"
    frame_gap: int = 1
    """Frame offset between the source (t-1) and target (t) frame,
    in dataset frame units. 1 = adjacent frames."""
    num_bins: int = 5
    event_channels: int = 256
    mask_extra_input: str = "photometric"  # "photometric" | "none"
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
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    pos_weight: float | None = None
    """Optional BCE positive-class weight to counter class imbalance
    (dynamic pixels are usually a small minority). If None, computed
    automatically from the observed GT dynamic ratio at startup
    when `auto_pos_weight=True`."""
    auto_pos_weight: bool = True
    log_every_n_steps: int = 10
    viz_every_n_steps: int = 50
    viz_max_samples: int = 4
    eval_every_n_epochs: int = 5
    checkpoint_every_n_epochs: int = 5
    save_dir: str = "runs/exp_gt_mask"
    seed: int = 42
    overfit_mode: bool = False
    resume_from: str | None = None
    balance_dynamic_batches: bool = False
    """If True, use DynamicBalancedBatchSampler so every training
    batch contains at least `min_dynamic_per_batch` windows whose
    GT mask has at least one dynamic pixel. Recommended for short/
    sparse-motion sequences where dynamic frames are a small,
    contiguous minority (plain shuffling can leave many consecutive
    batches with zero positive examples, making training curves
    hard to read and slowing convergence)."""
    min_dynamic_per_batch: int = 1


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

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}


class TrainerGTMask:
    def __init__(self, cfg: TrainConfigGTMask):
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

        self.model = GTMaskModel(
            num_bins=cfg.num_bins,
            event_channels=cfg.event_channels,
            mask_extra_input=cfg.mask_extra_input,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

        self._build_dataloaders()

        pos_weight = cfg.pos_weight
        if pos_weight is None and cfg.auto_pos_weight:
            pos_weight = self._estimate_pos_weight()
            logger.info(f"Auto pos_weight (1/gt_dynamic_ratio - 1): {pos_weight:.2f}")

        self.loss_fn = GTMaskLoss(
            bce_weight=cfg.bce_weight, dice_weight=cfg.dice_weight, pos_weight=pos_weight,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )

        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * cfg.epochs

        def lr_lambda(step):
            progress = step / max(1, total_steps)
            return cfg.scheduler_eta_min / cfg.learning_rate + \
                   (1 - cfg.scheduler_eta_min / cfg.learning_rate) * \
                   0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.ema = EMA(self.model, cfg.ema_decay) if cfg.use_ema else None
        self.amp_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else (
            torch.float16 if cfg.mixed_precision == "fp16" else None
        )

        self.start_epoch = 0
        if cfg.resume_from:
            self._resume(cfg.resume_from)

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    def _build_dataloaders(self):
        from src.data.dataset import EVIMO2Dataset
        from src.data.temporal_dataset import TemporalEVIMO2Dataset
        from src.data.collate import temporal_collate_fn
        from src.data.transforms import Compose, ToTensor, NormalizeEventTime, NormalizeIMU, VoxelizeEvents

        history_offsets = (-self.cfg.frame_gap, 0)

        ds = EVIMO2Dataset(
            dataset_root=self.cfg.dataset_root,
            sensors=self.cfg.sensors, split=self.cfg.split,
            load_depth=True, load_mask=True,
            subset=self.cfg.subset,
            sequence=self.cfg.sequence,
        )
        tds = TemporalEVIMO2Dataset(ds, history_offsets=history_offsets)
        logger.info(f"Train: {len(tds)} windows (frame_gap={self.cfg.frame_gap})")

        if self.cfg.balance_dynamic_batches:
            from src.data.dynamic_balanced_sampler import DynamicBalancedBatchSampler
            self.train_batch_sampler = DynamicBalancedBatchSampler(
                tds, batch_size=self.cfg.batch_size,
                min_dynamic_per_batch=self.cfg.min_dynamic_per_batch,
                drop_last=True, seed=self.cfg.seed,
            )
            self.train_loader = DataLoader(
                tds, batch_sampler=self.train_batch_sampler,
                collate_fn=temporal_collate_fn,
                num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
            )
        else:
            self.train_batch_sampler = None
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
                    subset=self.cfg.subset,
                )
                vtds = TemporalEVIMO2Dataset(vds, history_offsets=history_offsets)
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

    @torch.no_grad()
    def _estimate_pos_weight(self, max_batches: int = 20) -> float:
        """Peek at up to `max_batches` training batches to estimate the
        GT dynamic-pixel ratio, and derive a BCE pos_weight from it so
        rare dynamic pixels aren't drowned out by the static majority."""
        total_pos, total_pix = 0, 0
        for i, raw_batch in enumerate(self.train_loader):
            if i >= max_batches:
                break
            last = raw_batch.frames[-1]
            for gt_raw, fm in zip(last.mask, last.frame_motion):
                if gt_raw is None:
                    continue
                ids = get_dynamic_object_ids(fm)
                gt = evimo2_mask_to_binary_dynamic(gt_raw, ids)
                total_pos += int(gt.sum().item())
                total_pix += int(gt.numel())
        if total_pix == 0 or total_pos == 0:
            return 1.0
        ratio = total_pos / total_pix
        return float(min(50.0, max(1.0, (1.0 - ratio) / ratio)))

    # ------------------------------------------------------------
    # GT construction per batch
    # ------------------------------------------------------------

    def _prepare_batch(self, raw_batch, voxel_batch):
        """
        Build everything the model forward pass and loss need from one
        two-frame TemporalEVIMO2Batch:
          - voxel grids at t-1 (source) and t (target)
          - GT depth at t-1, resized to full image resolution
          - GT camera poses (raw CameraMotion) at t-1 and t
          - GT dynamic mask at t (target)
        Samples missing GT depth, GT pose, or GT mask at either frame
        are dropped from the batch (returns None if nothing is valid).
        """
        frame_src_raw = raw_batch.frames[0]   # t-1
        frame_tgt_raw = raw_batch.frames[-1]  # t
        frame_src_vox = voxel_batch.frames[0]
        frame_tgt_vox = voxel_batch.frames[-1]

        voxel_src = frame_src_vox.voxel_grid  # (B, C, H, W)
        voxel_tgt = frame_tgt_vox.voxel_grid
        H, W = voxel_src.shape[-2:]

        B = voxel_src.shape[0]

        cam_src = frame_src_raw.camera_motion
        cam_tgt = frame_tgt_raw.camera_motion

        pose_valid = torch.tensor(
            [bool(cs.pose_available) and bool(ct.pose_available)
             for cs, ct in zip(cam_src, cam_tgt)],
            dtype=torch.bool,
        )

        gt_depth_src, depth_valid = build_gt_depth_batch(
            frame_src_raw.depth, target_hw=(H, W), device=voxel_src.device,
        )

        gt_masks_raw = frame_tgt_raw.mask
        frame_motions = frame_tgt_raw.frame_motion
        mask_valid = torch.tensor([m is not None for m in gt_masks_raw], dtype=torch.bool)

        valid = pose_valid & depth_valid & mask_valid
        valid_idx = valid.nonzero(as_tuple=True)[0].tolist()
        if not valid_idx:
            return None

        gt_mask_list = []
        for i in valid_idx:
            ids = get_dynamic_object_ids(frame_motions[i])
            gt = evimo2_mask_to_binary_dynamic(gt_masks_raw[i], ids)
            while gt.ndim > 2:
                gt = gt.squeeze(0)
            gt = gt.float()
            if tuple(gt.shape[-2:]) != (H, W):
                gt = F.interpolate(
                    gt.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest",
                ).squeeze(0).squeeze(0)
            gt_mask_list.append(gt)
        gt_mask = torch.stack(gt_mask_list, dim=0).unsqueeze(1).to(voxel_src.device)

        return {
            "voxel_src": voxel_src[valid_idx],
            "voxel_tgt": voxel_tgt[valid_idx],
            "gt_depth_src": gt_depth_src[valid_idx],
            "camera_motion_src": [cam_src[i] for i in valid_idx],
            "camera_motion_tgt": [cam_tgt[i] for i in valid_idx],
            "camera_intrinsics": [frame_tgt_raw.camera_intrinsics[i] for i in valid_idx],
            "camera_distortion": [frame_tgt_raw.camera_distortion[i] for i in valid_idx],
            "gt_mask": gt_mask,
            # Raw (un-binarized) EVIMO2 masks + their FrameMotion, kept
            # around so _evaluate() can hand them to SegmentationMetrics
            # directly -- that class does its OWN speed-thresholded
            # binarization internally and returns an all-zero mask if
            # you pass frame_motions=None (see get_dynamic_object_ids),
            # so passing an already-binarized mask through it a SECOND
            # time silently zeroes out real dynamic pixels. gt_mask
            # above is for the loss function; gt_mask_raw+frame_motions
            # are for metrics.
            "gt_mask_raw": [gt_masks_raw[i] for i in valid_idx],
            "frame_motions": [frame_motions[i] for i in valid_idx],
        }

    # ------------------------------------------------------------
    # Train / eval loops
    # ------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * cfg.epochs
        logger.info(f"Steps/epoch: {steps_per_epoch}, total: {total_steps}")
        global_step = self.start_epoch * steps_per_epoch

        for epoch in range(self.start_epoch, cfg.epochs):
            if self.train_batch_sampler is not None:
                self.train_batch_sampler.set_epoch(epoch)
            self.model.train()
            t0 = time.time()
            last_loss = None

            for batch_idx, raw_batch in enumerate(self.train_loader):
                voxel_batch = self.transform(raw_batch)
                voxel_batch = voxel_batch.to(self.device)

                batch = self._prepare_batch(raw_batch, voxel_batch)
                if batch is None:
                    continue

                self.optimizer.zero_grad(set_to_none=True)

                if self.amp_dtype:
                    with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                        outputs = self.model(
                            voxel_t0=batch["voxel_src"], voxel_t1=batch["voxel_tgt"],
                            gt_depth_t0=batch["gt_depth_src"],
                            camera_motion_t0=batch["camera_motion_src"],
                            camera_motion_t1=batch["camera_motion_tgt"],
                            camera_intrinsics=batch["camera_intrinsics"],
                            camera_distortion=batch["camera_distortion"],
                        )
                        loss_output = self.loss_fn(outputs["mask"], batch["gt_mask"])
                        total_loss = loss_output["loss"]
                else:
                    outputs = self.model(
                        voxel_t0=batch["voxel_src"], voxel_t1=batch["voxel_tgt"],
                        gt_depth_t0=batch["gt_depth_src"],
                        camera_motion_t0=batch["camera_motion_src"],
                        camera_motion_t1=batch["camera_motion_tgt"],
                        camera_intrinsics=batch["camera_intrinsics"],
                        camera_distortion=batch["camera_distortion"],
                    )
                    loss_output = self.loss_fn(outputs["mask"], batch["gt_mask"])
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
                if self.ema:
                    self.ema.update(self.model)

                last_loss = total_loss

                if global_step % cfg.log_every_n_steps == 0:
                    self._log_step(epoch, batch_idx, global_step, total_loss, loss_output, grad_norm)

                if cfg.viz_every_n_steps > 0 and global_step % cfg.viz_every_n_steps == 0:
                    self._log_viz(global_step, outputs, batch)

                global_step += 1

            dt = time.time() - t0
            loss_str = f"{last_loss.item():.4f}" if last_loss is not None else "n/a"
            logger.info(f"Epoch {epoch+1}/{cfg.epochs} done in {dt:.1f}s, loss={loss_str}")

            if (epoch + 1) % cfg.eval_every_n_epochs == 0:
                if self.val_loader:
                    iou = self._evaluate(self.val_loader, f"val/epoch_{epoch+1}")
                    logger.info(f"Epoch {epoch+1} val IoU: {iou:.4f}")
                elif cfg.overfit_mode:
                    iou = self._evaluate(self.train_loader, "overfit_eval")
                    logger.info(f"Epoch {epoch+1} train (overfit) IoU: {iou:.4f}")

            if (epoch + 1) % cfg.checkpoint_every_n_epochs == 0:
                self._save(epoch)

        self._save(cfg.epochs - 1)
        self.writer.close()

    def _log_step(self, epoch, batch_idx, gs, total_loss, lo, gn):
        self.writer.add_scalar("train/total_loss", total_loss.item(), gs)
        for k in ["bce_loss", "dice_loss", "pred_dynamic_ratio", "gt_dynamic_ratio"]:
            if k in lo and isinstance(lo[k], torch.Tensor):
                self.writer.add_scalar(f"train/{k}", lo[k].item(), gs)
        self.writer.add_scalar("train/grad_norm", gn.item(), gs)
        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], gs)

        if batch_idx == 0 or batch_idx % (self.cfg.log_every_n_steps * 5) == 0:
            logger.info(
                f"E{epoch+1} B{batch_idx:4d} | "
                f"loss={total_loss.item():.4f} | "
                f"bce={lo.get('bce_loss', torch.tensor(0.0)).item():.4f} | "
                f"dice={lo.get('dice_loss', torch.tensor(0.0)).item():.4f} | "
                f"pred_dr={lo.get('pred_dynamic_ratio', torch.tensor(0.0)).item():.3f} | "
                f"gt_dr={lo.get('gt_dynamic_ratio', torch.tensor(0.0)).item():.3f} | "
                f"gn={gn.item():.2f}"
            )

    @torch.no_grad()
    def _log_viz(self, gs, outputs, batch):
        mask_probs = outputs["mask_probs"].float().cpu()
        gt = batch["gt_mask"].float().cpu()
        residual = outputs.get("residual")
        max_n = min(self.cfg.viz_max_samples, mask_probs.shape[0])
        for i in range(max_n):
            self.writer.add_image(f"mask/pred_{i}", mask_probs[i], gs)
            self.writer.add_image(f"mask/gt_{i}", gt[i], gs)
            if residual is not None and i < residual.shape[0]:
                self.writer.add_image(f"mask/residual_{i}", residual[i, :1].float().cpu(), gs)

    @torch.no_grad()
    def _evaluate(self, loader, tag: str) -> float:
        if self.ema:
            backup = {n: p.detach().clone() for n, p in self.model.named_parameters()}
            self.ema.apply_to(self.model)

        self.model.eval()
        metrics = SegmentationMetrics([0.3, 0.4, 0.5, 0.6, 0.7])

        for raw_batch in loader:
            voxel_batch = self.transform(raw_batch).to(self.device)
            batch = self._prepare_batch(raw_batch, voxel_batch)
            if batch is None:
                continue
            outputs = self.model(
                voxel_t0=batch["voxel_src"], voxel_t1=batch["voxel_tgt"],
                gt_depth_t0=batch["gt_depth_src"],
                camera_motion_t0=batch["camera_motion_src"],
                camera_motion_t1=batch["camera_motion_tgt"],
                camera_intrinsics=batch["camera_intrinsics"],
                camera_distortion=batch["camera_distortion"],
            )
            probs = torch.sigmoid(outputs["mask"])
            # Pass the RAW (un-binarized) EVIMO2 mask + real per-sample
            # FrameMotion so SegmentationMetrics does its own correct
            # speed-thresholded binarization. Passing frame_motions=None
            # here (as this used to do) makes get_dynamic_object_ids()
            # return an empty set for every sample, which makes
            # evimo2_mask_to_binary_dynamic() short-circuit to an
            # all-zero mask regardless of the real mask content --
            # silently reporting gt_dr=0 even when real dynamic pixels
            # exist. See _prepare_batch's gt_mask_raw for details.
            metrics.update(probs, batch["gt_mask_raw"], frame_motions=batch["frame_motions"])

        r = metrics.compute()
        self.writer.add_scalar(f"{tag}/iou", r["best_iou"], 0)
        self.writer.add_scalar(f"{tag}/f1", r["best_f1"], 0)
        logger.info(
            f"[{tag}] IoU={r['best_iou']:.4f} F1={r['best_f1']:.4f} "
            f"thr={r['best_threshold']:.2f} pred_dr={r['pred_dynamic_ratio']:.4f} "
            f"gt_dr={r['gt_dynamic_ratio']:.4f}"
        )

        self.model.train()
        if self.ema:
            for n, p in self.model.named_parameters():
                if n in backup:
                    p.data.copy_(backup[n])

        return r["best_iou"]

    # ------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------

    def _save(self, epoch):
        path = self.save_dir / "checkpoints" / f"epoch_{epoch+1:03d}.pth"
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.cfg.__dict__,
        }
        if self.ema:
            ckpt["ema_state_dict"] = self.ema.state_dict()
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint: {path}")

    def _resume(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.ema and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        self.start_epoch = ckpt.get("epoch", -1) + 1
        logger.info(f"Resumed from {path} at epoch {self.start_epoch}")