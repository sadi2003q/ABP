"""
selfsup_depth_pose_check.py

A SINGLE, SELF-CONTAINED script that:
  1. Predicts DEPTH and POSE from event data (no ground truth used as
     input -- this is the genuinely self-supervised model, unlike
     GTMaskModel which took GT depth/pose as fixed inputs).
  2. Trains them with PHOTOMETRIC LOSS (self-supervised: the loss
     compares a warped voxel frame against the real next frame --
     never touches GT depth/pose during training).
  3. Lets you CHECK the predictions against real GT depth/pose
     (loaded only for evaluation/plotting, never for training) so you
     can see how close the learned geometry gets to ground truth.

This reuses your project's OWN architecture pieces exactly as they
already exist and are validated elsewhere in the repo:
  - EventEncoder       (src/models/world_model/event_encoder.py)
  - DepthHead          (src/models/world_model/depth_head.py)
  - PoseHead           (src/models/world_model/pose_head.py)
  - LatentRenderer      (src/models/world_model/latent_renderer.py)
  - PhotometricLoss     (src/losses/photometric_loss.py)

Difference from PoseHead's normal usage: PoseHead is designed to take
an IMU embedding. To keep this a SINGLE runnable file without wiring
the full IMU-sample-reconstruction pipeline, pose here is predicted
directly from a pooled EVENT feature embedding instead of IMU -- a
reasonable, common substitution (event-only ego-motion estimation),
and easy to swap for the real IMUEncoder later if you want to compare.

Usage
-----
python selfsup_depth_pose_check.py \
    --dataset-root /path/to/dataset_root \
    --sensors left_camera --subset imo \
    --split train --sequence scene15_dyn_test_06_000000 \
    --overfit \
    --batch-size 4 --epochs 100 \
    --eval-every-n-epochs 2 \
    --save-dir runs/exp_selfsup_depth_pose
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.models.world_model.event_encoder import EventEncoder
from src.models.world_model.depth_head import DepthHead
from src.models.world_model.pose_head import PoseHead
from src.models.world_model.latent_renderer import LatentRenderer
from src.losses.photometric_loss import PhotometricLoss
from src.models.gt_mask_model.gt_pose_utils import (
    batch_gt_relative_pose_9d,
    quaternion_to_matrix,
)
from src.models.gt_mask_model.gt_depth_utils import build_gt_depth_batch

logger = logging.getLogger(__name__)


# ======================================================================
# Model: predicts DEPTH and POSE, no ground truth as input
# ======================================================================

class SelfSupDepthPoseModel(nn.Module):
    """
    Predicts depth (from event features) and pose (from a pooled event
    embedding, standing in for IMU) -- both LEARNED, no ground truth
    used anywhere in the forward pass. This is the genuinely self-
    supervised counterpart to GTMaskModel (which took GT depth/pose as
    fixed inputs instead of predicting them).

    Pose output format: PhotometricLoss (src/losses/photometric_loss.py)
    hardcodes `self.renderer = LatentRenderer()`, whose DEFAULT
    rotation_type is '6d' (9-DoF: [tx,ty,tz, a1x,a1y,a1z, a2x,a2y,a2z]),
    NOT PoseHead's native 6-DoF axis-angle output. Feeding a 6-vector
    into 6d-rotation code doesn't error -- pose[:, 6:9] on a 6-column
    tensor silently returns an EMPTY (B,0) slice, which then breaks
    downstream in a much more confusing way (NaN/shape errors far from
    the real cause). So this model converts PoseHead's 6-DoF axis-angle
    output into the 9-DoF 6d representation PhotometricLoss's renderer
    actually expects, via the same axis-angle -> rotation-matrix ->
    first-two-columns conversion GTMaskModel's gt_pose_utils.py uses
    for its (already-orthonormal) GT rotation matrices.
    """

    def __init__(self, num_bins: int = 5, pose_embedding_dim: int = 128):
        super().__init__()
        self.event_encoder = EventEncoder(input_channels=num_bins)
        self.depth_head = DepthHead(input_channels=256, hidden_channels=128)
        self.pose_head = PoseHead(input_dim=pose_embedding_dim, hidden_dim=128)
        self.pose_pool_proj = nn.Linear(256, pose_embedding_dim)
        # Kept for reference/inspection only -- PhotometricLoss uses its
        # OWN internal renderer (default rotation_type='6d'), this one
        # is not called in the training path.
        self.renderer = LatentRenderer(rotation_type="axis_angle")

    @staticmethod
    def _axis_angle_to_6d_pose(pose_6dof: torch.Tensor) -> torch.Tensor:
        """
        Converts PoseHead's native [tx,ty,tz,rx,ry,rz] (axis-angle)
        output into the [tx,ty,tz, a1x,a1y,a1z, a2x,a2y,a2z] 9-DoF
        format LatentRenderer(rotation_type='6d') (PhotometricLoss's
        internal renderer) actually expects, so the model's pose
        output is directly compatible with the project's photometric
        loss without silently misinterpreting values.
        """
        t = pose_6dof[:, :3]
        r = pose_6dof[:, 3:]
        theta = r.norm(dim=1, keepdim=True)
        axis = r / (theta + 1e-8)
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        cos_t, sin_t = torch.cos(theta[:, 0]), torch.sin(theta[:, 0])

        B = pose_6dof.shape[0]
        R = torch.zeros(B, 3, 3, device=pose_6dof.device, dtype=pose_6dof.dtype)
        R[:, 0, 0] = cos_t + x*x*(1-cos_t); R[:, 0, 1] = x*y*(1-cos_t) - z*sin_t; R[:, 0, 2] = x*z*(1-cos_t) + y*sin_t
        R[:, 1, 0] = y*x*(1-cos_t) + z*sin_t; R[:, 1, 1] = cos_t + y*y*(1-cos_t); R[:, 1, 2] = y*z*(1-cos_t) - x*sin_t
        R[:, 2, 0] = z*x*(1-cos_t) - y*sin_t; R[:, 2, 1] = z*y*(1-cos_t) + x*sin_t; R[:, 2, 2] = cos_t + z*z*(1-cos_t)

        a1 = R[:, :, 0]
        a2 = R[:, :, 1]
        return torch.cat([t, a1, a2], dim=1)  # (B, 9)

    def forward(self, voxel_t0: torch.Tensor, voxel_t1: torch.Tensor):
        """
        voxel_t0, voxel_t1 : (B, C, H, W) -- source (t-1) and target (t)

        Returns dict with predicted depth (t-1), predicted pose (t-1
        -> t) in BOTH the native 6-DoF axis-angle form ("pose_6dof",
        used for GT comparison in _evaluate) and the 9-DoF 6d form
        ("pose_9d", used for PhotometricLoss during training), plus
        the stacked (B,T=2,...) voxel tensor PhotometricLoss expects.
        """
        B, C, H, W = voxel_t0.shape
        stacked = torch.stack([voxel_t0, voxel_t1], dim=1)  # (B, 2, C, H, W)

        pyramid = self.event_encoder(stacked)
        feat_l4 = pyramid[-1]  # (B, 2, 256, H/16, W/16)

        # --- Depth: from EACH frame's own features (matches DepthHead's
        # documented (B,T,C,H,W) -> (B,T,1,H,W) contract). ---
        depths = self.depth_head(feat_l4)  # (B, 2, 1, H/16, W/16)

        # --- Pose: pool frame t-1 and t features into one embedding,
        # standing in for the IMU embedding PoseHead normally takes.
        # Using both frames (not just one) gives the pose head signal
        # about what actually changed between them. ---
        pooled_t0 = feat_l4[:, 0].mean(dim=[-2, -1])  # (B, 256)
        pooled_t1 = feat_l4[:, 1].mean(dim=[-2, -1])  # (B, 256)
        pose_embedding = self.pose_pool_proj(pooled_t1 - pooled_t0)  # (B, pose_embedding_dim)
        pose_6dof = self.pose_head(pose_embedding.unsqueeze(1))  # (B, 1, 6)
        pose_6dof = pose_6dof.squeeze(1)  # (B, 6) -- axis-angle [tx,ty,tz,rx,ry,rz]
        pose_9d = self._axis_angle_to_6d_pose(pose_6dof)  # (B, 9) -- for PhotometricLoss

        return {
            "depths": depths,          # (B, 2, 1, H/16, W/16)
            "pose_6dof": pose_6dof,     # (B, 6) axis-angle -- for GT comparison
            "pose_9d": pose_9d,         # (B, 9) 6d rep -- for PhotometricLoss
            "voxel_stacked": stacked,   # (B, 2, C, H, W) -- for PhotometricLoss
            "depth_t0_pred": depths[:, 0],  # (B, 1, H/16, W/16) convenience
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
    val_split: str = "val"
    frame_gap: int = 1
    num_bins: int = 5
    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_max_norm: float = 1.0
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 2
    checkpoint_every_n_epochs: int = 5
    save_dir: str = "runs/exp_selfsup_depth_pose"
    seed: int = 42
    overfit_mode: bool = False


class Trainer:
    def __init__(self, cfg: TrainConfig):
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

        self.model = SelfSupDepthPoseModel(num_bins=cfg.num_bins).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

        self._build_dataloaders()

        self.photometric_loss = PhotometricLoss().to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * cfg.epochs

        def lr_lambda(step):
            progress = step / max(1, total_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

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
            load_depth=True, load_mask=True,  # depth/mask loaded for EVAL ONLY
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
    def _prepare_batch(self, raw_batch, voxel_batch):
        """
        Only needs voxel grids + camera_intrinsics/distortion for
        TRAINING (photometric loss). GT depth/pose are ALSO loaded
        here, but ONLY used in _evaluate() -- never passed into the
        model or the loss during training.
        """
        frame_src_raw, frame_tgt_raw = raw_batch.frames[0], raw_batch.frames[-1]
        frame_src_vox, frame_tgt_vox = voxel_batch.frames[0], voxel_batch.frames[-1]

        voxel_src = frame_src_vox.voxel_grid
        voxel_tgt = frame_tgt_vox.voxel_grid
        H, W = voxel_src.shape[-2:]

        gt_depth_src, depth_valid = build_gt_depth_batch(
            frame_src_raw.depth, target_hw=(H, W), device=voxel_src.device,
        )
        cam_src = frame_src_raw.camera_motion
        cam_tgt = frame_tgt_raw.camera_motion
        pose_valid = torch.tensor(
            [bool(a.pose_available) and bool(b.pose_available) for a, b in zip(cam_src, cam_tgt)],
            dtype=torch.bool,
        )

        return {
            "voxel_src": voxel_src,
            "voxel_tgt": voxel_tgt,
            "camera_intrinsics": frame_tgt_raw.camera_intrinsics,
            "camera_distortion": frame_tgt_raw.camera_distortion,
            # EVAL-ONLY fields below (never used in training loss):
            "gt_depth_src": gt_depth_src,
            "depth_valid": depth_valid,
            "camera_motion_src": cam_src,
            "camera_motion_tgt": cam_tgt,
            "pose_valid": pose_valid,
        }

    # ------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        steps_per_epoch = len(self.train_loader)
        global_step = 0

        for epoch in range(cfg.epochs):
            self.model.train()
            t0 = time.time()
            last_loss = None

            for batch_idx, raw_batch in enumerate(self.train_loader):
                voxel_batch = self.transform(raw_batch).to(self.device)
                batch = self._prepare_batch(raw_batch, voxel_batch)

                K = torch.stack([
                    torch.as_tensor(k, device=self.device, dtype=torch.float32)
                    for k in batch["camera_intrinsics"]
                ])
                distortion = torch.stack([
                    torch.as_tensor(d, device=self.device, dtype=torch.float32)
                    for d in batch["camera_distortion"]
                ])

                self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(batch["voxel_src"], batch["voxel_tgt"])
                # PhotometricLoss's internal loop is `for t in range(1, T):
                # pose_t = poses[:, t]` -- i.e. poses[:, t] is the relative
                # transform taking frame (t-1) into frame t's coordinate
                # frame. With exactly 2 frames (T=2), only t=1 is ever used,
                # so poses needs a value at INDEX 1 specifically (index 0 is
                # never read by the loss but must exist for the tensor shape
                # to match). Pad accordingly rather than passing (B,1,9).
                pose_9d = outputs["pose_9d"]
                poses_seq = torch.zeros(pose_9d.shape[0], 2, 9, device=pose_9d.device, dtype=pose_9d.dtype)
                poses_seq[:, 1] = pose_9d

                loss_dict = self.photometric_loss(
                    voxel_grid=outputs["voxel_stacked"],
                    depths=outputs["depths"],
                    poses=poses_seq,
                    K=K, distortion=distortion,
                    mask_probs=None,
                )
                total_loss = loss_dict["loss"]

                if not torch.isfinite(total_loss):
                    logger.error(f"Non-finite loss: {total_loss.item()}")
                    continue

                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_max_norm)
                self.optimizer.step()
                self.scheduler.step()
                last_loss = total_loss

                if global_step % cfg.log_every_n_steps == 0:
                    logger.info(
                        f"E{epoch+1} B{batch_idx:4d} | loss={total_loss.item():.4f} | "
                        f"gn={grad_norm.item():.2f} | lr={self.scheduler.get_last_lr()[0]:.2e}"
                    )
                global_step += 1

            dt = time.time() - t0
            loss_str = f"{last_loss.item():.4f}" if last_loss is not None else "n/a"
            logger.info(f"Epoch {epoch+1}/{cfg.epochs} done in {dt:.1f}s, loss={loss_str}")

            if (epoch + 1) % cfg.eval_every_n_epochs == 0:
                if self.val_loader:
                    self._evaluate(self.val_loader, f"val@epoch{epoch+1}")
                elif cfg.overfit_mode:
                    self._evaluate(self.train_loader, f"overfit@epoch{epoch+1}")

            if (epoch + 1) % cfg.checkpoint_every_n_epochs == 0:
                self._save(epoch)

        self._save(cfg.epochs - 1)

    # ------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, loader, tag: str):
        """
        Compares the model's PREDICTED depth/pose against REAL GT
        depth/pose. Never used for training -- purely diagnostic, to
        answer "how close is the self-supervised prediction to
        ground truth".
        """
        self.model.eval()
        depth_errors, trans_errors, rot_errors = [], [], []

        for raw_batch in loader:
            voxel_batch = self.transform(raw_batch).to(self.device)
            batch = self._prepare_batch(raw_batch, voxel_batch)

            valid_idx = (batch["depth_valid"] & batch["pose_valid"]).nonzero(as_tuple=True)[0].tolist()
            if not valid_idx:
                continue

            outputs = self.model(batch["voxel_src"], batch["voxel_tgt"])

            # --- Depth comparison (resize predicted depth to GT resolution) ---
            pred_depth = outputs["depth_t0_pred"][valid_idx]  # (b,1,Hf,Wf)
            gt_depth = batch["gt_depth_src"][valid_idx]         # (b,1,H,W)
            pred_depth_up = F.interpolate(pred_depth, size=gt_depth.shape[-2:], mode="bilinear", align_corners=False)

            # Scale-invariant comparison: self-supervised monocular depth is
            # only defined up to an unknown global scale, so align medians
            # before computing error (standard practice, e.g. Zhou et al.).
            scale = gt_depth.median() / pred_depth_up.median().clamp_min(1e-6)
            pred_depth_scaled = pred_depth_up * scale
            depth_err = (pred_depth_scaled - gt_depth).abs().mean() / gt_depth.mean().clamp_min(1e-6)
            depth_errors.append(depth_err.item())

            # --- Pose comparison ---
            pred_pose = outputs["pose_6dof"][valid_idx]  # (b,6) axis-angle -- for direct comparison against GT axis-angle-style rotation matrix below
            pred_t = pred_pose[:, :3]

            cam_src = [batch["camera_motion_src"][i] for i in valid_idx]
            cam_tgt = [batch["camera_motion_tgt"][i] for i in valid_idx]
            gt_pose_9d = batch_gt_relative_pose_9d(cam_tgt, cam_src, device=self.device)
            gt_t = gt_pose_9d[:, :3]

            # Translation direction error (scale-free, since predicted
            # translation from a monocular-style setup has no metric
            # scale either) -- cosine similarity in degrees.
            pred_t_norm = F.normalize(pred_t, dim=1, eps=1e-6)
            gt_t_norm = F.normalize(gt_t, dim=1, eps=1e-6)
            cos_sim = (pred_t_norm * gt_t_norm).sum(dim=1).clamp(-1, 1)
            trans_angle_err = torch.acos(cos_sim) * 180.0 / math.pi
            trans_errors.append(trans_angle_err.mean().item())

            # Rotation error: compare predicted axis-angle rotation matrix
            # to GT rotation matrix via geodesic angle.
            theta = pred_pose[:, 3:].norm(dim=1, keepdim=True)
            axis = pred_pose[:, 3:] / (theta + 1e-8)
            x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
            cos_t, sin_t = torch.cos(theta[:, 0]), torch.sin(theta[:, 0])
            R_pred = torch.zeros(len(valid_idx), 3, 3, device=self.device)
            R_pred[:, 0, 0] = cos_t + x*x*(1-cos_t); R_pred[:, 0, 1] = x*y*(1-cos_t) - z*sin_t; R_pred[:, 0, 2] = x*z*(1-cos_t) + y*sin_t
            R_pred[:, 1, 0] = y*x*(1-cos_t) + z*sin_t; R_pred[:, 1, 1] = cos_t + y*y*(1-cos_t); R_pred[:, 1, 2] = y*z*(1-cos_t) - x*sin_t
            R_pred[:, 2, 0] = z*x*(1-cos_t) - y*sin_t; R_pred[:, 2, 1] = z*y*(1-cos_t) + x*sin_t; R_pred[:, 2, 2] = cos_t + z*z*(1-cos_t)

            q_tgt = torch.as_tensor(
                np.stack([c.quaternion for c in cam_tgt]), device=self.device, dtype=torch.float32,
            )
            q_src = torch.as_tensor(
                np.stack([c.quaternion for c in cam_src]), device=self.device, dtype=torch.float32,
            )
            R_tgt = quaternion_to_matrix(q_tgt)
            R_src = quaternion_to_matrix(q_src)
            R_gt_rel = torch.bmm(R_tgt.transpose(1, 2), R_src)

            R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt_rel)
            trace = R_diff.diagonal(dim1=1, dim2=2).sum(dim=1)
            rot_angle_err = torch.acos(((trace - 1.0) / 2.0).clamp(-1, 1)) * 180.0 / math.pi
            rot_errors.append(rot_angle_err.mean().item())

        if depth_errors:
            logger.info(
                f"[{tag}] depth_rel_err={sum(depth_errors)/len(depth_errors):.4f} "
                f"trans_angle_err_deg={sum(trans_errors)/len(trans_errors):.2f} "
                f"rot_angle_err_deg={sum(rot_errors)/len(rot_errors):.2f}"
            )
        else:
            logger.info(f"[{tag}] no valid samples with GT depth+pose")

        self.model.train()

    # ------------------------------------------------------------
    def _save(self, epoch):
        path = self.save_dir / "checkpoints" / f"epoch_{epoch+1:03d}.pth"
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
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every-n-steps", type=int, default=10)
    p.add_argument("--eval-every-n-epochs", type=int, default=2)
    p.add_argument("--checkpoint-every-n-epochs", type=int, default=5)
    p.add_argument("--save-dir", type=str, default="runs/exp_selfsup_depth_pose")
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
        log_every_n_steps=args.log_every_n_steps,
        eval_every_n_epochs=args.eval_every_n_epochs,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        save_dir=args.save_dir,
        seed=args.seed,
        overfit_mode=args.overfit,
    )

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()