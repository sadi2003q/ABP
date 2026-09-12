"""
Trainer for SELF-SUPERVISED mask-consistency training of GTMaskModel.

Difference from trainer_gt_mask.py (TrainerGTMask)
----------------------------------------------------
TrainerGTMask trains with GTMaskLoss: direct supervision against the
real EVIMO2 GT dynamic mask (BCE+Dice vs ground truth labels). That
is standard supervised learning, not self-supervised, even though
depth/pose inputs are ground truth.

TrainerGTMaskConsistency (this file) trains with MaskConsistencyLoss
(consistency_loss.py): the ONLY training signal is agreement between
the model's own predictions at two consecutive frame pairs, warped
into alignment using GT depth + GT pose (still ground truth -- this
stage deliberately keeps depth/pose fixed to isolate "does a
consistency loss alone work" from "do predicted depth/pose also
work", per the incremental plan). The real GT mask is loaded ONLY
for evaluation (to report IoU/F1 against ground truth so you can
judge how good the self-supervised result is) -- it is never part of
the loss or backward pass.

Data requirement
-----------------
Needs a 3-FRAME window (t-2, t-1, t), not the 2-frame window
TrainerGTMask uses, because computing a consistency signal requires
two separate forward passes: (t-2 -> t-1) and (t-1 -> t).
"""

from __future__ import annotations

import time, math, logging
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.gt_mask_model.model import GTMaskModel
from src.models.gt_mask_model.consistency_loss import MaskConsistencyLoss
from src.models.gt_mask_model.gt_depth_utils import build_gt_depth_batch
from src.models.gt_mask_model.gt_pose_utils import batch_gt_relative_pose_9d
from src.utils.metrics import SegmentationMetrics, get_dynamic_object_ids, evimo2_mask_to_binary_dynamic

logger = logging.getLogger(__name__)


@dataclass
class TrainConfigGTMaskConsistency:
    dataset_root: str = "/home/z/my-project/data/dataset_root"
    sensors: tuple = ("left_camera",)
    split: str = "train"
    subset: str = "imo"
    sequence: tuple | None = None
    val_split: str = "val"
    frame_gap: int = 1
    num_bins: int = 5
    event_channels: int = 256
    mask_extra_input: str = "photometric"
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
    collapse_penalty_weight: float = 0.5
    target_dynamic_ratio: float = 0.02
    log_every_n_steps: int = 10
    viz_every_n_steps: int = 50
    viz_max_samples: int = 4
    eval_every_n_epochs: int = 2
    checkpoint_every_n_epochs: int = 5
    save_dir: str = "runs/exp_gt_mask_consistency"
    seed: int = 42
    overfit_mode: bool = False
    resume_from: str | None = None
    balance_dynamic_batches: bool = False
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


class TrainerGTMaskConsistency:
    def __init__(self, cfg: TrainConfigGTMaskConsistency):
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

        self.loss_fn = MaskConsistencyLoss(
            bce_weight=cfg.bce_weight, dice_weight=cfg.dice_weight,
            collapse_penalty_weight=cfg.collapse_penalty_weight,
            target_dynamic_ratio=cfg.target_dynamic_ratio,
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
    # Data (3-frame windows: t-2, t-1, t)
    # ------------------------------------------------------------

    def _build_dataloaders(self):
        from src.data.dataset import EVIMO2Dataset
        from src.data.temporal_dataset import TemporalEVIMO2Dataset
        from src.data.collate import temporal_collate_fn
        from src.data.transforms import Compose, ToTensor, NormalizeEventTime, NormalizeIMU, VoxelizeEvents

        history_offsets = (-2 * self.cfg.frame_gap, -self.cfg.frame_gap, 0)

        ds = EVIMO2Dataset(
            dataset_root=self.cfg.dataset_root,
            sensors=self.cfg.sensors, split=self.cfg.split,
            load_depth=True, load_mask=True,
            subset=self.cfg.subset,
            sequence=self.cfg.sequence,
        )
        tds = TemporalEVIMO2Dataset(ds, history_offsets=history_offsets)
        logger.info(f"Train: {len(tds)} windows (frame_gap={self.cfg.frame_gap}, 3-frame)")

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

    # ------------------------------------------------------------
    # Batch preparation for a 3-frame window
    # ------------------------------------------------------------

    def _prepare_batch(self, raw_batch, voxel_batch):
        """
        Builds everything needed for TWO forward passes:
          pair A: (t-2 -> t-1)
          pair B: (t-1 -> t)
        plus GT depth/pose for both pairs (still ground truth at this
        stage), and the REAL GT mask at frame t ONLY for evaluation
        (never used in the loss).
        Drops samples missing GT depth/pose at any of the 3 frames,
        or GT mask at frame t (needed for eval bookkeeping only).
        """
        f_tm2 = raw_batch.frames[0]   # t-2
        f_tm1_raw, f_tm1_vox = raw_batch.frames[1], voxel_batch.frames[1]  # t-1
        f_t_raw, f_t_vox = raw_batch.frames[2], voxel_batch.frames[2]      # t
        f_tm2_vox = voxel_batch.frames[0]

        voxel_tm2 = f_tm2_vox.voxel_grid
        voxel_tm1 = f_tm1_vox.voxel_grid
        voxel_t = f_t_vox.voxel_grid
        H, W = voxel_tm2.shape[-2:]
        B = voxel_tm2.shape[0]

        cam_tm2 = f_tm2.camera_motion
        cam_tm1 = f_tm1_raw.camera_motion
        cam_t = f_t_raw.camera_motion

        pose_valid = torch.tensor([
            bool(a.pose_available) and bool(b.pose_available) and bool(c.pose_available)
            for a, b, c in zip(cam_tm2, cam_tm1, cam_t)
        ], dtype=torch.bool)

        gt_depth_tm2, depth_valid_a = build_gt_depth_batch(f_tm2.depth, target_hw=(H, W), device=voxel_tm2.device)
        gt_depth_tm1, depth_valid_b = build_gt_depth_batch(f_tm1_raw.depth, target_hw=(H, W), device=voxel_tm2.device)

        gt_masks_t_raw = f_t_raw.mask
        frame_motions_t = f_t_raw.frame_motion
        mask_valid = torch.tensor([m is not None for m in gt_masks_t_raw], dtype=torch.bool)

        valid = pose_valid & depth_valid_a & depth_valid_b & mask_valid
        valid_idx = valid.nonzero(as_tuple=True)[0].tolist()
        if not valid_idx:
            return None

        return {
            "voxel_tm2": voxel_tm2[valid_idx],
            "voxel_tm1": voxel_tm1[valid_idx],
            "voxel_t": voxel_t[valid_idx],
            "gt_depth_tm2": gt_depth_tm2[valid_idx],
            "gt_depth_tm1": gt_depth_tm1[valid_idx],
            "camera_motion_tm2": [cam_tm2[i] for i in valid_idx],
            "camera_motion_tm1": [cam_tm1[i] for i in valid_idx],
            "camera_motion_t": [cam_t[i] for i in valid_idx],
            "camera_intrinsics": [f_t_raw.camera_intrinsics[i] for i in valid_idx],
            "camera_distortion": [f_t_raw.camera_distortion[i] for i in valid_idx],
            # eval-only, never used in the loss:
            "gt_mask_raw": [gt_masks_t_raw[i] for i in valid_idx],
            "frame_motions": [frame_motions_t[i] for i in valid_idx],
        }

    def _forward_two_pairs(self, batch):
        """Runs GTMaskModel on (t-2->t-1) and (t-1->t), then warps
        pred_mask(t-1) into t's view using GT depth(t-1)+GT pose(t-1->t)
        via the model's own renderer -- the SAME warp mechanism used
        internally, applied here to the predicted mask instead of
        features."""
        out_a = self.model(
            voxel_t0=batch["voxel_tm2"], voxel_t1=batch["voxel_tm1"],
            gt_depth_t0=batch["gt_depth_tm2"],
            camera_motion_t0=batch["camera_motion_tm2"],
            camera_motion_t1=batch["camera_motion_tm1"],
            camera_intrinsics=batch["camera_intrinsics"],
            camera_distortion=batch["camera_distortion"],
        )
        out_b = self.model(
            voxel_t0=batch["voxel_tm1"], voxel_t1=batch["voxel_t"],
            gt_depth_t0=batch["gt_depth_tm1"],
            camera_motion_t0=batch["camera_motion_tm1"],
            camera_motion_t1=batch["camera_motion_t"],
            camera_intrinsics=batch["camera_intrinsics"],
            camera_distortion=batch["camera_distortion"],
        )

        pred_mask_tm1_probs = torch.sigmoid(out_a["mask"])  # keep graph; consistency_loss detaches its copy

        pose_tm1_to_t = batch_gt_relative_pose_9d(
            camera_motions_target=batch["camera_motion_t"],
            camera_motions_source=batch["camera_motion_tm1"],
            device=pred_mask_tm1_probs.device,
        )
        warped_pred_mask = self.model.renderer(
            feature=pred_mask_tm1_probs, depth=batch["gt_depth_tm1"],
            pose=pose_tm1_to_t, K=out_b["K"], distortion=out_b["distortion"],
        )

        return out_a, out_b, warped_pred_mask

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
                voxel_batch = self.transform(raw_batch).to(self.device)
                batch = self._prepare_batch(raw_batch, voxel_batch)
                if batch is None:
                    continue

                self.optimizer.zero_grad(set_to_none=True)

                if self.amp_dtype:
                    with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                        out_a, out_b, warped_pred_mask = self._forward_two_pairs(batch)
                        loss_output = self.loss_fn(out_a["mask"], warped_pred_mask, out_b["mask"])
                        total_loss = loss_output["loss"]
                else:
                    out_a, out_b, warped_pred_mask = self._forward_two_pairs(batch)
                    loss_output = self.loss_fn(out_a["mask"], warped_pred_mask, out_b["mask"])
                    total_loss = loss_output["loss"]

                if not torch.isfinite(total_loss):
                    logger.error(f"Non-finite loss: {total_loss.item()}")
                    continue

                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_max_norm)
                self.optimizer.step()
                self.scheduler.step()
                if self.ema:
                    self.ema.update(self.model)

                last_loss = total_loss

                if global_step % cfg.log_every_n_steps == 0:
                    self._log_step(epoch, batch_idx, global_step, total_loss, loss_output, grad_norm)

                global_step += 1

            dt = time.time() - t0
            loss_str = f"{last_loss.item():.4f}" if last_loss is not None else "n/a"
            logger.info(f"Epoch {epoch+1}/{cfg.epochs} done in {dt:.1f}s, loss={loss_str}")

            if (epoch + 1) % cfg.eval_every_n_epochs == 0:
                if self.val_loader:
                    iou = self._evaluate(self.val_loader, f"val/epoch_{epoch+1}")
                    logger.info(f"Epoch {epoch+1} val IoU (vs REAL GT, eval-only): {iou:.4f}")
                elif cfg.overfit_mode:
                    iou = self._evaluate(self.train_loader, "overfit_eval")
                    logger.info(f"Epoch {epoch+1} train (overfit) IoU (vs REAL GT, eval-only): {iou:.4f}")

            if (epoch + 1) % cfg.checkpoint_every_n_epochs == 0:
                self._save(epoch)

        self._save(cfg.epochs - 1)
        self.writer.close()

    def _log_step(self, epoch, batch_idx, gs, total_loss, lo, gn):
        self.writer.add_scalar("train/total_loss", total_loss.item(), gs)
        for k in ["consistency_bce", "consistency_dice", "collapse_penalty",
                  "pred_dynamic_ratio", "warped_target_ratio"]:
            if k in lo and isinstance(lo[k], torch.Tensor):
                self.writer.add_scalar(f"train/{k}", lo[k].item(), gs)
        self.writer.add_scalar("train/grad_norm", gn.item(), gs)
        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], gs)

        if batch_idx == 0 or batch_idx % (self.cfg.log_every_n_steps * 5) == 0:
            logger.info(
                f"E{epoch+1} B{batch_idx:4d} | "
                f"loss={total_loss.item():.4f} | "
                f"cons_bce={lo.get('consistency_bce', torch.tensor(0.0)).item():.4f} | "
                f"cons_dice={lo.get('consistency_dice', torch.tensor(0.0)).item():.4f} | "
                f"collapse_pen={lo.get('collapse_penalty', torch.tensor(0.0)).item():.4f} | "
                f"pred_dr={lo.get('pred_dynamic_ratio', torch.tensor(0.0)).item():.3f} | "
                f"warped_tgt_dr={lo.get('warped_target_ratio', torch.tensor(0.0)).item():.3f} | "
                f"gn={gn.item():.2f}"
            )

    @torch.no_grad()
    def _evaluate(self, loader, tag: str) -> float:
        """
        EVAL-ONLY: measures the self-supervised model's predicted mask
        (at frame t, from the t-1->t pair) against the REAL GT mask.
        This comparison is NEVER used for training/backward -- it's
        purely how you and your supervisor judge how good the
        self-supervised result turned out to be.
        """
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
            out_b = self.model(
                voxel_t0=batch["voxel_tm1"], voxel_t1=batch["voxel_t"],
                gt_depth_t0=batch["gt_depth_tm1"],
                camera_motion_t0=batch["camera_motion_tm1"],
                camera_motion_t1=batch["camera_motion_t"],
                camera_intrinsics=batch["camera_intrinsics"],
                camera_distortion=batch["camera_distortion"],
            )
            probs = torch.sigmoid(out_b["mask"])
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