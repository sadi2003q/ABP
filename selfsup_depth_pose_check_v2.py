"""
selfsup_depth_pose_check_v2.py

Self-supervised depth + pose from event voxels (photometric loss only),
checked against GT depth/pose for EVALUATION ONLY.

What changed vs v1 (and why -- v1 log showed gn=0.00 from epoch 5 on, frozen
eval metrics, and loss that only wobbled per-batch):

  1. Depth decoder with a BOUNDED, non-dead output
     (disparity = min + (max-min)*sigmoid, depth = 1/disparity, LeakyReLU body,
     initialised at ~1 m). Depth can never hit 0 / inf, so the warp cannot
     blow up, and there is no ReLU that can die.
  2. Pose net that keeps SPATIAL information (concat features of both frames
     -> convs -> per-location 6-vector -> spatial mean), output scaled by 0.01
     so training starts near identity motion (Monodepth2-style). v1 global-
     mean-pooled the feature difference, which throws the motion cue away.
  3. Robust Rodrigues (safe at rotation = 0) shared by model and eval.
  4. Gaussian blur of the event voxels BEFORE the photometric loss. Sparse
     event voxels give a loss where "warp everything out of the image" is a
     trivial minimum (warped = 0 -> loss = mean|target|, independent of the
     parameters -> zero gradient, batch-dependent value). Blurring widens the
     basin so a correct warp is rewarded.
  5. Edge-aware depth smoothness term (small weight).
  6. LR warm-up + cosine.
  7. Diagnostics: first-batch gradient/baseline check, depth/pose stats in the
     log, zero-gradient warning, and a stronger eval (per-pixel valid mask,
     constant-depth baseline, sign / transpose-convention checks, no-rotation
     baseline) so you can tell "learned" from "matches a trivial baseline".

The project's EventEncoder and PhotometricLoss are reused unchanged. DepthHead,
PoseHead and LatentRenderer are no longer used here (their output activations
were the prime suspects for the dead gradients; swap them back in once this
version trains, to compare).

Usage
-----
python selfsup_depth_pose_check_v2.py \
    --dataset-root /path/to/dataset_root \
    --sensors left_camera --subset imo \
    --split train --sequence scene15_dyn_test_06_000000 \
    --overfit --batch-size 4 --epochs 100 --num-workers 2 \
    --save-dir runs/exp_selfsup_depth_pose_v2
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.models.world_model.event_encoder import EventEncoder
from src.losses.photometric_loss import PhotometricLoss
from src.models.gt_mask_model.gt_pose_utils import (
    batch_gt_relative_pose_9d,
    quaternion_to_matrix,
)
from src.models.gt_mask_model.gt_depth_utils import build_gt_depth_batch

logger = logging.getLogger(__name__)


# ======================================================================
# Geometry / image helpers
# ======================================================================

def axis_angle_to_matrix(r: torch.Tensor) -> torch.Tensor:
    """(B,3) axis-angle -> (B,3,3) rotation via Rodrigues; safe at r = 0."""
    theta2 = (r * r).sum(dim=1, keepdim=True)                 # (B,1)
    theta = torch.sqrt(theta2 + 1e-12)
    a = torch.sin(theta) / theta                              # sin(t)/t
    b_full = (1.0 - torch.cos(theta)) / (theta2 + 1e-12)      # (1-cos t)/t^2
    b = torch.where(theta2 < 1e-6, 0.5 - theta2 / 24.0, b_full)

    zero = torch.zeros_like(r[:, 0])
    K = torch.stack(
        [zero, -r[:, 2], r[:, 1],
         r[:, 2], zero, -r[:, 0],
         -r[:, 1], r[:, 0], zero], dim=1,
    ).view(-1, 3, 3)
    I = torch.eye(3, device=r.device, dtype=r.dtype).unsqueeze(0)
    return I + a.unsqueeze(-1) * K + b.unsqueeze(-1) * torch.bmm(K, K)


def so3_angle_deg(R: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (degrees) of a batch of rotation matrices."""
    trace = R.diagonal(dim1=1, dim2=2).sum(dim=1)
    return torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)) * 180.0 / math.pi


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over the last two dims of any (..., H, W) tensor."""
    if sigma <= 0:
        return x
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    shape = x.shape
    y = x.reshape(-1, 1, shape[-2], shape[-1])
    y = F.pad(y, (radius, radius, radius, radius), mode="reflect")
    y = F.conv2d(y, k.view(1, 1, 1, -1))
    y = F.conv2d(y, k.view(1, 1, -1, 1))
    return y.reshape(shape)


def edge_aware_smoothness(depths: torch.Tensor, voxel: torch.Tensor) -> torch.Tensor:
    """
    depths: (B,T,1,h,w) ; voxel: (B,T,C,H,W).
    Smoothness of mean-normalised disparity, down-weighted at event edges.
    """
    B, T, _, h, w = depths.shape
    disp = (1.0 / depths).reshape(B * T, 1, h, w)
    disp = disp / (disp.mean(dim=(2, 3), keepdim=True) + 1e-7)

    guide = voxel.abs().sum(dim=2).reshape(B * T, 1, voxel.shape[-2], voxel.shape[-1])
    guide = F.adaptive_avg_pool2d(guide, (h, w))
    guide = guide / (guide.mean(dim=(2, 3), keepdim=True) + 1e-7)

    dx = (disp[..., :, 1:] - disp[..., :, :-1]).abs()
    dy = (disp[..., 1:, :] - disp[..., :-1, :]).abs()
    gx = (guide[..., :, 1:] - guide[..., :, :-1]).abs()
    gy = (guide[..., 1:, :] - guide[..., :-1, :]).abs()
    return (dx * torch.exp(-gx)).mean() + (dy * torch.exp(-gy)).mean()


# ======================================================================
# Model
# ======================================================================

class DepthDecoder(nn.Module):
    """
    (B,T,C,h,w) features -> (B,T,1,h,w) depth in [min_depth, max_depth].
    Sigmoid-bounded disparity, LeakyReLU body (nothing that can die),
    output bias set so the initial depth is ~init_depth everywhere.
    """

    def __init__(self, in_ch=256, hidden=128, min_depth=0.1, max_depth=10.0, init_depth=1.0):
        super().__init__()
        self.min_disp = 1.0 / max_depth
        self.max_disp = 1.0 / min_depth
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
        )
        self.out = nn.Conv2d(hidden // 2, 1, 3, padding=1)

        p = (1.0 / init_depth - self.min_disp) / (self.max_disp - self.min_disp)
        assert 0.0 < p < 1.0, "init_depth must lie inside [min_depth, max_depth]"
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            self.out.bias.fill_(math.log(p / (1.0 - p)))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, T, C, h, w = feat.shape
        x = self.body(feat.reshape(B * T, C, h, w))
        disp = self.min_disp + (self.max_disp - self.min_disp) * torch.sigmoid(self.out(x))
        return (1.0 / disp).view(B, T, 1, h, w)


class SpatialPoseNet(nn.Module):
    """
    Relative pose t-1 -> t from the two frames' feature maps.
    Keeps spatial structure (per-location 6-vector, then spatial mean) and is
    scaled by 0.01 so the initial motion is ~identity.
    Output: (B,6) = [tx,ty,tz, rx,ry,rz] (axis-angle).
    """

    def __init__(self, in_ch=256, hidden=256, out_scale=0.01):
        super().__init__()
        self.out_scale = out_scale
        self.net = nn.Sequential(
            nn.Conv2d(2 * in_ch, hidden, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 6, 1),
        )

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f0, f1], dim=1)
        return self.out_scale * self.net(x).mean(dim=[2, 3])


class SelfSupDepthPoseModel(nn.Module):
    """No ground truth anywhere in the forward pass."""

    def __init__(self, num_bins=5, min_depth=0.1, max_depth=10.0, init_depth=1.0):
        super().__init__()
        self.event_encoder = EventEncoder(input_channels=num_bins)
        self.depth_decoder = DepthDecoder(256, 128, min_depth, max_depth, init_depth)
        self.pose_net = SpatialPoseNet(256)

    def forward(self, voxel_t0: torch.Tensor, voxel_t1: torch.Tensor):
        stacked = torch.stack([voxel_t0, voxel_t1], dim=1)        # (B,2,C,H,W)
        feat = self.event_encoder(stacked)[-1]                    # (B,2,256,h,w)

        depths = self.depth_decoder(feat)                         # (B,2,1,h,w)
        pose_6dof = self.pose_net(feat[:, 0], feat[:, 1])         # (B,6)

        R = axis_angle_to_matrix(pose_6dof[:, 3:])
        pose_9d = torch.cat([pose_6dof[:, :3], R[:, :, 0], R[:, :, 1]], dim=1)  # (B,9)

        return {
            "depths": depths,
            "pose_6dof": pose_6dof,
            "pose_9d": pose_9d,
            "voxel_stacked": stacked,
            "depth_t0_pred": depths[:, 0],
        }


# ======================================================================
# Config / Trainer
# ======================================================================

@dataclass
class TrainConfig:
    dataset_root: str = "/home/z/my-project/data/dataset_root"
    sensors: tuple = ("left_camera",)
    split: str = "train"
    subset: str = "imo"
    sequence: tuple | None = None
    val_split: str | None = "val"
    frame_gap: int = 1
    num_bins: int = 5
    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_max_norm: float = 1.0
    warmup_steps: int = 100
    min_depth: float = 0.1
    max_depth: float = 10.0
    init_depth: float = 1.0
    blur_sigma: float = 1.5          # 0 disables the blur on the loss input
    smooth_weight: float = 1e-3      # 0 disables the smoothness term
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 2
    checkpoint_every_n_epochs: int = 5
    save_dir: str = "runs/exp_selfsup_depth_pose_v2"
    seed: int = 42
    overfit_mode: bool = False


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)

        self.model = SelfSupDepthPoseModel(
            num_bins=cfg.num_bins, min_depth=cfg.min_depth,
            max_depth=cfg.max_depth, init_depth=cfg.init_depth,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

        self._build_dataloaders()
        self.photometric_loss = PhotometricLoss().to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
        total_steps = len(self.train_loader) * cfg.epochs
        warmup = max(1, cfg.warmup_steps)

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

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
            load_depth=True, load_mask=True,  # GT loaded for EVAL ONLY
            subset=self.cfg.subset,
            sequence=self.cfg.sequence,
        )
        tds = TemporalEVIMO2Dataset(ds, history_offsets=history_offsets)
        logger.info(f"Train: {len(tds)} windows (frame_gap={self.cfg.frame_gap})")

        self.train_loader = DataLoader(
            tds, batch_size=self.cfg.batch_size, shuffle=True,
            collate_fn=temporal_collate_fn,
            num_workers=self.cfg.num_workers, drop_last=True,
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
                    collate_fn=temporal_collate_fn, num_workers=self.cfg.num_workers,
                )
            except Exception as e:
                logger.warning(f"No val loader: {e}")

        self.transform = Compose([
            ToTensor(), NormalizeEventTime(), NormalizeIMU(),
            VoxelizeEvents(num_bins=self.cfg.num_bins),
        ])

    # ------------------------------------------------------------
    def _prepare_batch(self, raw_batch, voxel_batch, with_gt: bool = False):
        """
        Training needs only voxels + intrinsics/distortion. GT depth/pose are
        built only when with_gt=True (evaluation), which also saves time per
        training step.
        """
        frame_src_raw, frame_tgt_raw = raw_batch.frames[0], raw_batch.frames[-1]
        voxel_src = voxel_batch.frames[0].voxel_grid
        voxel_tgt = voxel_batch.frames[-1].voxel_grid

        batch = {
            "voxel_src": voxel_src,
            "voxel_tgt": voxel_tgt,
            "camera_intrinsics": frame_tgt_raw.camera_intrinsics,
            "camera_distortion": frame_tgt_raw.camera_distortion,
        }
        if with_gt:
            H, W = voxel_src.shape[-2:]
            gt_depth_src, depth_valid = build_gt_depth_batch(
                frame_src_raw.depth, target_hw=(H, W), device=voxel_src.device,
            )
            cam_src = frame_src_raw.camera_motion
            cam_tgt = frame_tgt_raw.camera_motion
            batch.update({
                "gt_depth_src": gt_depth_src,
                "depth_valid": depth_valid,
                "camera_motion_src": cam_src,
                "camera_motion_tgt": cam_tgt,
                "pose_valid": torch.tensor(
                    [bool(a.pose_available) and bool(b.pose_available) for a, b in zip(cam_src, cam_tgt)],
                    dtype=torch.bool,
                ),
            })
        return batch

    def _intrinsics(self, batch):
        K = torch.stack([
            torch.as_tensor(k, device=self.device, dtype=torch.float32)
            for k in batch["camera_intrinsics"]
        ])
        distortion = torch.stack([
            torch.as_tensor(d, device=self.device, dtype=torch.float32)
            for d in batch["camera_distortion"]
        ])
        return K, distortion

    # ------------------------------------------------------------
    def _compute_loss(self, outputs, K, distortion, override_pose=None, override_depth=None):
        """
        PhotometricLoss reads poses[:, t] as the transform (t-1)->t for
        t = 1..T-1, so with T=2 the pose must sit at INDEX 1 (index 0 unused).
        The blurred voxel is used ONLY as the loss input; the model itself
        sees the sharp voxel.
        """
        cfg = self.cfg
        depths = outputs["depths"] if override_depth is None else override_depth
        pose_9d = outputs["pose_9d"] if override_pose is None else override_pose

        poses_seq = torch.zeros(pose_9d.shape[0], 2, 9, device=pose_9d.device, dtype=pose_9d.dtype)
        poses_seq[:, 1] = pose_9d

        vox = gaussian_blur(outputs["voxel_stacked"], cfg.blur_sigma)
        loss_dict = self.photometric_loss(
            voxel_grid=vox, depths=depths, poses=poses_seq,
            K=K, distortion=distortion, mask_probs=None,
        )
        photo = loss_dict["loss"]

        if cfg.smooth_weight > 0:
            smooth = edge_aware_smoothness(depths, outputs["voxel_stacked"])
        else:
            smooth = torch.zeros((), device=photo.device, dtype=photo.dtype)
        return photo + cfg.smooth_weight * smooth, photo, smooth

    def _module_grad_norms(self):
        out = {}
        for name in ("event_encoder", "depth_decoder", "pose_net"):
            sq = 0.0
            for p in getattr(self.model, name).parameters():
                if p.grad is not None:
                    sq += p.grad.detach().float().pow(2).sum().item()
            out[name] = math.sqrt(sq)
        return out

    # ------------------------------------------------------------
    def _diagnose_first_batch(self):
        """
        One-off sanity check before training:
          * loss at model init vs loss with identity pose + constant depth
            (a model that cannot beat the identity baseline has learned nothing)
          * per-module gradient norms (all zero => graph is disconnected or
            the warp leaves the image)
        """
        self.model.train()
        raw_batch = next(iter(self.train_loader))
        voxel_batch = self.transform(raw_batch).to(self.device)
        batch = self._prepare_batch(raw_batch, voxel_batch)
        K, distortion = self._intrinsics(batch)

        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batch["voxel_src"], batch["voxel_tgt"])
        total, photo, smooth = self._compute_loss(outputs, K, distortion)

        B = outputs["pose_9d"].shape[0]
        ident = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float32, device=self.device).expand(B, 9)
        with torch.no_grad():
            _, base_photo, _ = self._compute_loss(
                outputs, K, distortion, override_pose=ident,
                override_depth=torch.ones_like(outputs["depths"]),
            )

        total.backward()
        norms = self._module_grad_norms()
        self.optimizer.zero_grad(set_to_none=True)

        logger.info(
            f"[diag] init photo={photo.item():.4f} smooth={smooth.item():.4f} | "
            f"identity/const-depth baseline photo={base_photo.item():.4f}"
        )
        logger.info("[diag] grad norms: " + ", ".join(f"{k}={v:.3e}" for k, v in norms.items()))
        if all(v == 0.0 for v in norms.values()):
            logger.warning("[diag] ALL gradients are zero at init: the loss is not connected to the "
                           "parameters (check the warp / valid-mask inside PhotometricLoss).")

    # ------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        self._diagnose_first_batch()
        global_step = 0

        for epoch in range(cfg.epochs):
            self.model.train()
            t0 = time.time()
            last_loss = None
            zero_grad_steps = 0

            for batch_idx, raw_batch in enumerate(self.train_loader):
                voxel_batch = self.transform(raw_batch).to(self.device)
                batch = self._prepare_batch(raw_batch, voxel_batch)
                K, distortion = self._intrinsics(batch)

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(batch["voxel_src"], batch["voxel_tgt"])
                total_loss, photo, smooth = self._compute_loss(outputs, K, distortion)

                if not torch.isfinite(total_loss):
                    logger.error(f"Non-finite loss: {total_loss.item()}")
                    continue

                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_max_norm)
                if grad_norm.item() < 1e-8:
                    zero_grad_steps += 1
                self.optimizer.step()
                self.scheduler.step()
                last_loss = total_loss

                if global_step % cfg.log_every_n_steps == 0:
                    d = outputs["depths"].detach()
                    p = outputs["pose_6dof"].detach()
                    logger.info(
                        f"E{epoch + 1} B{batch_idx:4d} | loss={total_loss.item():.4f} "
                        f"(photo={photo.item():.4f} smooth={smooth.item():.4f}) | "
                        f"gn={grad_norm.item():.2e} | lr={self.scheduler.get_last_lr()[0]:.2e} | "
                        f"depth[min/mean/max]={d.min().item():.2f}/{d.mean().item():.2f}/{d.max().item():.2f} | "
                        f"|t|={p[:, :3].norm(dim=1).mean().item():.4f} "
                        f"|r|={p[:, 3:].norm(dim=1).mean().item() * 180 / math.pi:.2f}deg"
                    )
                global_step += 1

            dt = time.time() - t0
            loss_str = f"{last_loss.item():.4f}" if last_loss is not None else "n/a"
            logger.info(f"Epoch {epoch + 1}/{cfg.epochs} done in {dt:.1f}s, loss={loss_str}")
            if zero_grad_steps:
                logger.warning(f"Epoch {epoch + 1}: {zero_grad_steps} steps had ~zero gradient")

            if (epoch + 1) % cfg.eval_every_n_epochs == 0:
                if self.val_loader:
                    self._evaluate(self.val_loader, f"val@epoch{epoch + 1}")
                elif cfg.overfit_mode:
                    self._evaluate(self.train_loader, f"overfit@epoch{epoch + 1}")

            if (epoch + 1) % cfg.checkpoint_every_n_epochs == 0:
                self._save(epoch)

        self._save(cfg.epochs - 1)

    # ------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, loader, tag: str):
        """
        Compares PREDICTED depth/pose with REAL GT (diagnostic only), against
        trivial baselines so "learned" can be told from "matches a constant":
          depth : per-sample median-scaled AbsRel on valid GT pixels, plus the
                  AbsRel of a constant-depth prediction (same scaling).
          trans : direction error in degrees (also sign-flipped: ~180 - err).
          rot   : geodesic error vs R_tgt^T R_src, vs its transpose (convention
                  check), and vs predicting no rotation at all.
        """
        self.model.eval()
        m = {k: [] for k in ("absrel", "absrel_const", "pred_cv",
                             "t_err", "t_err_flip", "r_err", "r_err_T", "r_zero")}

        for raw_batch in loader:
            voxel_batch = self.transform(raw_batch).to(self.device)
            batch = self._prepare_batch(raw_batch, voxel_batch, with_gt=True)

            valid_idx = (batch["depth_valid"] & batch["pose_valid"]).nonzero(as_tuple=True)[0].tolist()
            if not valid_idx:
                continue

            outputs = self.model(batch["voxel_src"], batch["voxel_tgt"])

            # ---- depth ----
            pred = outputs["depth_t0_pred"][valid_idx]
            gt = batch["gt_depth_src"][valid_idx]
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            for i in range(pred.shape[0]):
                mask = torch.isfinite(gt[i]) & (gt[i] > 1e-3)
                if mask.sum() < 100:
                    continue
                p, g = pred[i][mask], gt[i][mask]
                s = g.median() / p.median().clamp_min(1e-6)
                m["absrel"].append(((p * s - g).abs() / g).mean().item())
                m["absrel_const"].append(((g.median() - g).abs() / g).mean().item())
                m["pred_cv"].append((p.std() / p.mean().clamp_min(1e-6)).item())

            # ---- pose ----
            pred_pose = outputs["pose_6dof"][valid_idx]
            cam_src = [batch["camera_motion_src"][i] for i in valid_idx]
            cam_tgt = [batch["camera_motion_tgt"][i] for i in valid_idx]

            gt_pose_9d = batch_gt_relative_pose_9d(cam_tgt, cam_src, device=self.device)
            pt = F.normalize(pred_pose[:, :3], dim=1, eps=1e-8)
            gt_t = F.normalize(gt_pose_9d[:, :3], dim=1, eps=1e-8)
            t_err = torch.acos((pt * gt_t).sum(dim=1).clamp(-1, 1)) * 180.0 / math.pi
            m["t_err"] += t_err.tolist()
            m["t_err_flip"] += (180.0 - t_err).tolist()

            R_pred = axis_angle_to_matrix(pred_pose[:, 3:])
            q_tgt = torch.as_tensor(np.stack([c.quaternion for c in cam_tgt]), device=self.device, dtype=torch.float32)
            q_src = torch.as_tensor(np.stack([c.quaternion for c in cam_src]), device=self.device, dtype=torch.float32)
            R_gt = torch.bmm(quaternion_to_matrix(q_tgt).transpose(1, 2), quaternion_to_matrix(q_src))

            m["r_err"] += so3_angle_deg(torch.bmm(R_pred.transpose(1, 2), R_gt)).tolist()
            m["r_err_T"] += so3_angle_deg(torch.bmm(R_pred.transpose(1, 2), R_gt.transpose(1, 2))).tolist()
            m["r_zero"] += so3_angle_deg(R_gt).tolist()

        if m["absrel"] and m["t_err"]:
            avg = {k: float(np.mean(v)) if v else float("nan") for k, v in m.items()}
            logger.info(
                f"[{tag}] depth_absrel={avg['absrel']:.4f} (const-depth baseline={avg['absrel_const']:.4f}, "
                f"pred_cv={avg['pred_cv']:.3f}) | trans_err={avg['t_err']:.1f}deg "
                f"(sign-flipped={avg['t_err_flip']:.1f}) | rot_err={avg['r_err']:.2f}deg "
                f"(transposed-GT={avg['r_err_T']:.2f}, no-rotation baseline={avg['r_zero']:.2f})"
            )
        else:
            logger.info(f"[{tag}] no valid samples with GT depth+pose")

        self.model.train()

    # ------------------------------------------------------------
    def _save(self, epoch):
        path = self.save_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pth"
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict()}, path)
        logger.info(f"Saved checkpoint: {path}")


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(description="Self-supervised depth+pose prediction, checked against GT.")
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--sensors", type=str, nargs="+", default=["left_camera"])
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--subset", type=str, default="imo")
    p.add_argument("--sequence", type=str, nargs="+", default=None)
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--frame-gap", type=int, default=1)
    p.add_argument("--num-bins", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-depth", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--init-depth", type=float, default=1.0)
    p.add_argument("--blur-sigma", type=float, default=1.5, help="Gaussian sigma (px) on voxels fed to the loss; 0 = off")
    p.add_argument("--smooth-weight", type=float, default=1e-3, help="edge-aware depth smoothness weight; 0 = off")
    p.add_argument("--log-every-n-steps", type=int, default=10)
    p.add_argument("--eval-every-n-epochs", type=int, default=2)
    p.add_argument("--checkpoint-every-n-epochs", type=int, default=5)
    p.add_argument("--save-dir", type=str, default="runs/exp_selfsup_depth_pose_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overfit", action="store_true")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(save_dir / "train.log")],
    )

    cfg = TrainConfig(
        dataset_root=args.dataset_root,
        sensors=tuple(args.sensors),
        split=args.split if not args.overfit else "train",
        subset=args.subset,
        sequence=tuple(args.sequence) if args.sequence else None,
        val_split=None if args.overfit else args.val_split,
        frame_gap=args.frame_gap,
        num_bins=args.num_bins,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_max_norm=args.grad_clip,
        warmup_steps=args.warmup_steps,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        init_depth=args.init_depth,
        blur_sigma=args.blur_sigma,
        smooth_weight=args.smooth_weight,
        log_every_n_steps=args.log_every_n_steps,
        eval_every_n_epochs=args.eval_every_n_epochs,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        save_dir=args.save_dir,
        seed=args.seed,
        overfit_mode=args.overfit,
    )

    Trainer(cfg).train()


if __name__ == "__main__":
    main()