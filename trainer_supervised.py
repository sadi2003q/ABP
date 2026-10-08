"""
Trainer for SupervisedMotionSegmenter — the fully supervised,
deployable moving-object segmentation model.

Stability measures (each one addresses something seen in earlier runs)
----------------------------------------------------------------------
1. Exponential moving average of the weights WITH a warm-up ramp
   (decay_t = min(decay, (1+t)/(10+t))). The old trainers initialised
   the average at the random weights with a fixed 0.999 decay, so for
   short runs the averaged model stayed ~90% random and evaluation
   showed IoU near zero. With the ramp the average tracks the live
   model early on and smooths it later — this is what removes the
   epoch-to-epoch IoU swings (0.60 -> 0.19 -> 0.58 in the v2 runs).
2. Learning-rate warm-up, then cosine decay to a small floor (not to
   zero), so the last epochs still learn.
3. Mask output initialised at the expected moving-pixel fraction, so
   the first steps don't produce huge, noisy gradients.
4. Batch-global Dice + class-weighted binary cross-entropy, and every
   training batch guaranteed to contain a moving object (balanced
   sampler), so no step is wasted on all-static batches.
5. Mixed precision chosen per GPU: bfloat16 on Ampere or newer,
   float16 with loss scaling on older cards such as the Tesla T4
   (which has no native bfloat16).
6. best.pth saved whenever the evaluation IoU improves, so a late dip
   never costs you the best model; last.pth every epoch for resuming.
7. Evaluation never silently disappears: if the validation split is
   missing it says so and falls back to the training windows, clearly
   labelled.

Evaluation protocol
-------------------
Uses the repo's SegmentationMetrics (pooled over all pixels, speed-aware
ground truth), so numbers are directly comparable to every earlier
trainer. Both "best IoU over thresholds 0.3-0.7" (the repo convention)
and "IoU at the fixed threshold 0.5" are logged. For a paper, choose
the threshold on validation and report test at that fixed threshold
(train_supervised.py --evaluate-only --threshold ...).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.supervised_model import SupervisedMotionSegmenter, SupervisedLoss, build_targets
from src.utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)

# Fields that change the network's shape; on evaluation/resume they are
# always taken from the checkpoint, never from the command line.
ARCHITECTURE_FIELDS = ("num_bins", "history_offsets", "use_imu", "depth_weight", "pose_weight")


@dataclass
class TrainConfigSupervised:
    # ---- data ----
    dataset_root: str = "/content/drive/MyDrive/single_seq_root"
    sensors: tuple = ("left_camera",)
    subsets: tuple = ("imo",)
    split: str = "train"
    sequence: tuple | None = None
    val_split: str | None = "val"
    val_sequence: tuple | None = None
    history_offsets: tuple = (-2, -1, 0)
    num_bins: int = 5
    batch_size: int = 4
    num_workers: int = 2
    pin_memory: bool = True
    balance_dynamic_batches: bool = True
    min_dynamic_per_batch: int = 1
    event_dropout: float = 0.0
    # ---- labels ----
    label_source: str = "ground_truth"   # ground_truth | pseudo
    pseudo_label_dir: str | None = None
    pseudo_ignore_band: int = 0          # pixels around each pseudo-labelled blob left out of the loss
    # ---- model ----
    use_imu: bool = True
    # ---- loss ----
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    pos_weight: float | None = None
    auto_pos_weight: bool = True
    max_pos_weight: float = 10.0
    depth_weight: float = 0.1
    pose_weight: float = 0.1
    translation_scale: float = 100.0
    rotation_scale: float = 10.0
    # ---- optimisation ----
    epochs: int = 60
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.02
    grad_clip: float = 1.0
    mixed_precision: str = "auto"  # auto | bf16 | fp16 | none
    use_ema: bool = True
    ema_decay: float = 0.999
    # ---- evaluation / logging ----
    eval_every_n_epochs: int = 1
    eval_thresholds: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    select_metric: str = "best_iou"  # best_iou | iou_at_0.5
    early_stop_patience: int = 0     # evaluations without improvement; 0 = off
    checkpoint_every_n_epochs: int = 10
    log_every_n_steps: int = 10
    viz_every_n_steps: int = 100
    viz_max_samples: int = 4
    save_dir: str = "runs/exp_supervised"
    seed: int = 42
    overfit_mode: bool = False
    resume_from: str | None = None


# ======================================================================
# Exponential moving average with warm-up ramp
# ======================================================================

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        ema_state = self.module.state_dict()
        for name, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                ema_state[name].mul_(d).add_(value.detach(), alpha=1.0 - d)
            else:
                ema_state[name].copy_(value)

    def state_dict(self):
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd["module"])
        self.updates = sd.get("updates", 0)


# ======================================================================
# Trainer
# ======================================================================

class TrainerSupervised:
    def __init__(
        self,
        cfg: TrainConfigSupervised,
        build_train: bool = True,
        train_loader=None,
        eval_loader=None,
        transform=None,
        eval_tag: str | None = None,
    ):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        import random
        import numpy as np
        random.seed(cfg.seed); np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)

        self.save_dir = Path(cfg.save_dir)
        (self.save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "tb"))

        self._setup_precision()

        # ---- model ----
        self.model = SupervisedMotionSegmenter(
            num_bins=cfg.num_bins,
            num_frames=len(cfg.history_offsets),
            use_imu=cfg.use_imu,
            predict_depth=cfg.depth_weight > 0,
            predict_pose=cfg.pose_weight > 0,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
        logger.info(
            f"Inputs: {len(cfg.history_offsets)} frames {tuple(cfg.history_offsets)} x {cfg.num_bins} bins"
            f" | inertial: {'on' if cfg.use_imu else 'off'}"
            f" | helper depth: {'on' if cfg.depth_weight > 0 else 'off'}"
            f" | helper pose: {'on' if cfg.pose_weight > 0 else 'off'}"
        )

        # ---- labels ----
        self.pseudo_store = None
        if cfg.label_source == "pseudo":
            from src.data.pseudo_labels import PseudoLabelStore
            if not cfg.pseudo_label_dir:
                raise ValueError("--label-source pseudo needs --pseudo-label-dir")
            self.pseudo_store = PseudoLabelStore(cfg.pseudo_label_dir)
            logger.info("TRAINING LABELS: PSEUDO-LABELS (true masks are used for evaluation only)")
            logger.info(f"Ignore band around pseudo-labelled blobs: {cfg.pseudo_ignore_band} px"
                        + (" (off)" if cfg.pseudo_ignore_band <= 0 else ""))
        else:
            logger.info("TRAINING LABELS: ground-truth masks")

        # ---- data ----
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.eval_tag = eval_tag or "eval"
        self.train_sampler = None
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._default_transform()
        if build_train and self.train_loader is None:
            self._build_data()

        # ---- loss ----
        pos_weight = cfg.pos_weight
        if build_train and pos_weight is None and cfg.auto_pos_weight:
            pos_weight = self._estimate_pos_weight()
        self.pos_weight = pos_weight
        self.loss_fn = SupervisedLoss(
            bce_weight=cfg.bce_weight,
            dice_weight=cfg.dice_weight,
            pos_weight=pos_weight,
            depth_weight=cfg.depth_weight,
            pose_weight=cfg.pose_weight,
            translation_scale=cfg.translation_scale,
            rotation_scale=cfg.rotation_scale,
        )

        # ---- optimisation ----
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
        steps_per_epoch = len(self.train_loader) if self.train_loader is not None else 1
        self.total_steps = max(1, steps_per_epoch * cfg.epochs)
        warmup = max(1, int(cfg.warmup_fraction * self.total_steps))

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, self.total_steps - warmup))
            return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_mode == "fp16")
        self.ema = ModelEMA(self.model, cfg.ema_decay) if cfg.use_ema else None

        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = -1.0
        self.best_epoch = -1
        self.evals_without_improvement = 0
        if cfg.resume_from:
            self._resume(cfg.resume_from)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_precision(self):
        mode = self.cfg.mixed_precision
        if self.device.type != "cuda":
            mode = "none"
        elif mode == "auto":
            major, _ = torch.cuda.get_device_capability(0)
            mode = "bf16" if major >= 8 else "fp16"
        self.amp_mode = mode
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mode)
        logger.info(f"Mixed precision: {mode}")

    def _autocast(self):
        if self.amp_dtype is None:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _default_transform(self):
        from src.data.transforms import Compose, ToTensor, NormalizeEventTime, NormalizeIMU, VoxelizeEvents
        return Compose([
            ToTensor(), NormalizeEventTime(), NormalizeIMU(),
            VoxelizeEvents(num_bins=self.cfg.num_bins),
        ])

    def _temporal_datasets(self, split: str, sequence) -> list:
        from src.data.dataset import EVIMO2Dataset
        from src.data.temporal_dataset import TemporalEVIMO2Dataset

        out = []
        for subset in self.cfg.subsets:
            try:
                ds = EVIMO2Dataset(
                    dataset_root=self.cfg.dataset_root,
                    sensors=self.cfg.sensors, split=split,
                    load_depth=self.cfg.depth_weight > 0, load_mask=True,
                    subset=subset, sequence=sequence,
                )
            except ValueError as e:
                logger.info(f"[{subset}/{split}] skipped: {e}")
                continue
            if len(ds) == 0:
                continue
            tds = TemporalEVIMO2Dataset(ds, history_offsets=self.cfg.history_offsets)
            if len(tds) == 0:
                logger.warning(f"[{subset}/{split}] sequences too short for offsets {self.cfg.history_offsets}")
                continue
            logger.info(f"[{subset}/{split}] {len(tds)} windows")
            out.append(tds)
        return out

    def _loader(self, datasets, shuffle: bool, batch_sampler=None, drop_last=False):
        from src.data.collate import temporal_collate_fn
        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        kwargs = dict(
            collate_fn=temporal_collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory and self.device.type == "cuda",
            persistent_workers=self.cfg.num_workers > 0,
        )
        if batch_sampler is not None:
            return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=shuffle,
                          drop_last=drop_last, **kwargs)

    def _build_data(self):
        cfg = self.cfg
        train_sets = self._temporal_datasets(cfg.split, cfg.sequence)
        if not train_sets:
            raise RuntimeError(
                f"No training data found under {cfg.dataset_root} for sensors={cfg.sensors}, "
                f"subsets={cfg.subsets}, split={cfg.split!r}, sequence={cfg.sequence}. "
                f"Expected layout: <root>/<sensor>/<subset>/<split>/<sequence>/"
            )
        n_train = sum(len(d) for d in train_sets)
        logger.info(f"Training windows: {n_train}")

        if cfg.balance_dynamic_batches:
            from src.data.concat_balanced_sampler import ConcatBalancedBatchSampler
            is_dynamic = None
            if self.pseudo_store is not None:
                # Balance on PSEUDO-labels; the true masks must not steer training.
                from src.data.pseudo_labels import target_frame_keys
                keys = [k for tds in train_sets for k in target_frame_keys(tds)]
                is_dynamic = [self.pseudo_store.has_positive(*k) for k in keys]
                n_labelled = sum(self.pseudo_store.get(*k) is not None for k in keys[:: max(1, len(keys) // 200)])
                logger.info(f"Pseudo-labels cover ~{100 * n_labelled / max(1, len(keys[:: max(1, len(keys) // 200)])):.0f}% "
                            f"of training windows (sampled check).")
            self.train_sampler = ConcatBalancedBatchSampler(
                train_sets, batch_size=cfg.batch_size,
                min_dynamic_per_batch=cfg.min_dynamic_per_batch,
                drop_last=True, seed=cfg.seed, is_dynamic=is_dynamic,
            )
            self.train_loader = self._loader(train_sets, shuffle=False, batch_sampler=self.train_sampler)
        else:
            self.train_loader = self._loader(train_sets, shuffle=True,
                                             drop_last=n_train >= cfg.batch_size)

        # ---- evaluation data ----
        if cfg.overfit_mode:
            self.eval_loader = self._loader(train_sets, shuffle=False)
            self.eval_tag = "train_overfit"
            logger.info("Evaluation set: the TRAINING windows (overfit mode).")
            return

        val_sets = self._temporal_datasets(cfg.val_split, cfg.val_sequence) if cfg.val_split else []
        if val_sets:
            self.eval_loader = self._loader(val_sets, shuffle=False)
            self.eval_tag = f"{cfg.val_split}"
            logger.info(f"Evaluation set: split {cfg.val_split!r}, {sum(len(d) for d in val_sets)} windows.")
        else:
            logger.warning("=" * 70)
            logger.warning(
                f"No validation data for split {cfg.val_split!r}. Falling back to "
                f"evaluating on the TRAINING windows — these IoU numbers measure "
                f"fitting, not generalisation. Pass --overfit to make this explicit, "
                f"or point --val-split at a split that exists."
            )
            logger.warning("=" * 70)
            self.eval_loader = self._loader(train_sets, shuffle=False)
            self.eval_tag = "train_fallback"

    @torch.no_grad()
    def _estimate_pos_weight(self, max_batches: int = 20) -> float:
        pos, total = 0.0, 0.0
        for i, raw in enumerate(self.train_loader):
            if i >= max_batches:
                break
            last = raw.frames[-1]
            if self.pseudo_store is not None:
                for k in range(len(last.sequence_names)):
                    lab = self.pseudo_store.get(last.sensors[k], last.sequence_names[k],
                                                int(last.local_frame_indices[k]))
                    if lab is not None:
                        pos += float(lab.sum()); total += float(lab.size)
                continue
            from src.models.supervised_model.targets import build_gt_mask
            # Native resolution is fine for a ratio.
            for m, fm in zip(last.mask, last.frame_motion):
                if m is None:
                    continue
                gm, _ = build_gt_mask([m], [fm], hw=tuple(m.shape[-2:]))
                pos += float(gm.sum()); total += float(gm.numel())
        if total == 0 or pos == 0:
            logger.info("Positive-class weight: 1.00 (no moving pixels seen while estimating)")
            return 1.0
        r = pos / total
        w = float(min(self.cfg.max_pos_weight, max(1.0, (1 - r) / r)))
        logger.info(f"Moving-pixel fraction in training batches: {r:.4f} -> positive-class weight {w:.2f}")
        return w

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward(self, model, raw_batch, train: bool):
        voxel_batch = self.transform(raw_batch).to(self.device)
        voxels = torch.stack([f.voxel_grid for f in voxel_batch.frames], dim=1).float()
        if train and self.cfg.event_dropout > 0:
            B, _, _, H, W = voxels.shape
            keep = (torch.rand(B, 1, 1, H, W, device=voxels.device) >= self.cfg.event_dropout)
            voxels = voxels * keep
        imu_frames = voxel_batch.frames if self.cfg.use_imu else None
        with self._autocast():
            outputs = model(voxels, imu_frames)
        return outputs, voxels

    def _targets(self, raw_batch, outputs):
        depth_hw = tuple(outputs["depth_log"].shape[-2:]) if outputs.get("depth_log") is not None else None
        return build_targets(
            raw_batch,
            mask_hw=tuple(outputs["mask"].shape[-2:]),
            depth_hw=depth_hw,
            need_pose=outputs.get("pose") is not None,
            pseudo_store=self.pseudo_store,
            pseudo_ignore_band=self.cfg.pseudo_ignore_band,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        steps_per_epoch = len(self.train_loader)
        logger.info(f"Steps per epoch: {steps_per_epoch}, total steps: {self.total_steps}")
        bad_steps_in_a_row = 0

        for epoch in range(self.start_epoch, cfg.epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self.model.train()
            t0 = time.time()
            running, n_running = 0.0, 0

            for batch_idx, raw_batch in enumerate(self.train_loader):
                outputs, voxels = self._forward(self.model, raw_batch, train=True)
                targets = self._targets(raw_batch, outputs)
                if not targets["mask_valid"].any():
                    continue

                loss_out = self.loss_fn(outputs, targets)
                loss = loss_out["loss"]

                if not torch.isfinite(loss):
                    bad_steps_in_a_row += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    logger.error(f"Non-finite loss at step {self.global_step}; skipped.")
                    if bad_steps_in_a_row >= 20:
                        raise RuntimeError("20 non-finite losses in a row — stopping. "
                                           "Try --mixed-precision none or a lower --learning-rate.")
                    continue
                bad_steps_in_a_row = 0

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)

                running += loss.item(); n_running += 1

                if self.global_step % cfg.log_every_n_steps == 0:
                    self._log_step(epoch, batch_idx, steps_per_epoch, loss_out, grad_norm)
                if cfg.viz_every_n_steps > 0 and self.global_step % cfg.viz_every_n_steps == 0:
                    self._log_images(self.global_step, outputs, targets, voxels)
                self.global_step += 1

            mean_loss = running / max(1, n_running)
            logger.info(
                f"Epoch {epoch + 1}/{cfg.epochs} done in {time.time() - t0:.1f}s | "
                f"mean loss {mean_loss:.4f} | lr {self.scheduler.get_last_lr()[0]:.2e}"
            )
            self.writer.add_scalar("epoch/mean_train_loss", mean_loss, epoch + 1)

            stop = False
            if self.eval_loader is not None and (epoch + 1) % cfg.eval_every_n_epochs == 0:
                stop = self._evaluate_and_track(epoch)

            self._save("last.pth", epoch)
            if cfg.checkpoint_every_n_epochs > 0 and (epoch + 1) % cfg.checkpoint_every_n_epochs == 0:
                self._save(f"epoch_{epoch + 1:03d}.pth", epoch)
            if stop:
                logger.info(f"Early stopping: no improvement in {cfg.early_stop_patience} evaluations.")
                break

        self._final_report()
        self.writer.close()

    def _evaluate_and_track(self, epoch) -> bool:
        model = self.ema.module if self.ema is not None else self.model
        r = self.evaluate(self.eval_loader, model=model)
        metric = r["iou_at_0.5"] if self.cfg.select_metric == "iou_at_0.5" else r["best_iou"]

        improved = metric > self.best_metric
        if improved:
            self.best_metric, self.best_epoch = metric, epoch + 1
            self.evals_without_improvement = 0
            self._save("best.pth", epoch)
        else:
            self.evals_without_improvement += 1

        e = epoch + 1
        tag = self.eval_tag
        self.writer.add_scalar(f"{tag}/iou_best_threshold", r["best_iou"], e)
        self.writer.add_scalar(f"{tag}/iou_at_0.5", r["iou_at_0.5"], e)
        self.writer.add_scalar(f"{tag}/f1", r["best_f1"], e)
        self.writer.add_scalar(f"{tag}/precision", r["best_precision"], e)
        self.writer.add_scalar(f"{tag}/recall", r["best_recall"], e)
        self.writer.add_scalar(f"{tag}/best_threshold", r["best_threshold"], e)
        self.writer.add_scalar(f"{tag}/pred_dynamic_ratio", r["pred_dynamic_ratio"], e)
        self.writer.add_scalar(f"{tag}/gt_dynamic_ratio", r["gt_dynamic_ratio"], e)
        self.writer.add_scalar(f"{tag}/best_so_far", self.best_metric, e)

        logger.info(
            f"[{tag} | epoch {e}] IoU={r['best_iou']:.4f} (thr {r['best_threshold']:.2f}) | "
            f"IoU@0.5={r['iou_at_0.5']:.4f} | F1={r['best_f1']:.4f} | "
            f"P={r['best_precision']:.3f} R={r['best_recall']:.3f} | "
            f"pred_dr={r['pred_dynamic_ratio']:.4f} gt_dr={r['gt_dynamic_ratio']:.4f} | "
            f"best {self.best_metric:.4f} @ epoch {self.best_epoch}"
            + ("  << NEW BEST" if improved else "")
        )
        if r["gt_dynamic_ratio"] == 0:
            logger.warning("Evaluation set has NO moving pixels — IoU is meaningless here.")

        p = self.cfg.early_stop_patience
        return p > 0 and self.evals_without_improvement >= p

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader, model=None, per_sequence: bool = False) -> dict:
        model = model if model is not None else (self.ema.module if self.ema else self.model)
        was_training = model.training
        model.eval()

        thresholds = sorted(set(self.cfg.eval_thresholds) | {0.5})
        overall = SegmentationMetrics(thresholds)
        by_sequence: dict[str, SegmentationMetrics] = {}

        for raw_batch in loader:
            outputs, _ = self._forward(model, raw_batch, train=False)
            probs = torch.sigmoid(outputs["mask"].float())
            last = raw_batch.frames[-1]
            masks, motions = list(last.mask), list(last.frame_motion)
            overall.update(probs, masks, frame_motions=motions)

            if per_sequence:
                names = list(getattr(last, "sequence_names", ["all"] * len(masks)))
                for name in set(names):
                    idx = [i for i, n in enumerate(names) if n == name]
                    m = by_sequence.setdefault(name, SegmentationMetrics(thresholds))
                    m.update(probs[idx], [masks[i] for i in idx], frame_motions=[motions[i] for i in idx])

        r = overall.compute()
        r["iou_at_0.5"] = r["per_threshold"]["thr_0.50"]["iou"]
        if per_sequence:
            r["per_sequence"] = {}
            for name, m in sorted(by_sequence.items()):
                s = m.compute()
                r["per_sequence"][name] = {
                    "best_iou": s["best_iou"], "best_threshold": s["best_threshold"],
                    "iou_at_0.5": s["per_threshold"]["thr_0.50"]["iou"],
                    "gt_dynamic_ratio": s["gt_dynamic_ratio"],
                }
        if was_training:
            model.train()
        return r

    def _final_report(self):
        best_path = self.save_dir / "checkpoints" / "best.pth"
        if self.eval_loader is None or not best_path.exists():
            return
        ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
        model = SupervisedMotionSegmenter(
            num_bins=self.cfg.num_bins, num_frames=len(self.cfg.history_offsets),
            use_imu=self.cfg.use_imu, predict_depth=self.cfg.depth_weight > 0,
            predict_pose=self.cfg.pose_weight > 0,
        ).to(self.device)
        model.load_state_dict(ckpt["ema_state_dict"]["module"] if "ema_state_dict" in ckpt
                              else ckpt["model_state_dict"])
        r = self.evaluate(self.eval_loader, model=model, per_sequence=True)
        report = {
            "evaluated_on": self.eval_tag,
            "best_epoch": self.best_epoch,
            "best_iou": r["best_iou"], "best_threshold": r["best_threshold"],
            "iou_at_0.5": r["iou_at_0.5"], "f1": r["best_f1"],
            "precision": r["best_precision"], "recall": r["best_recall"],
            "per_threshold": r["per_threshold"],
            "per_sequence": r.get("per_sequence", {}),
        }
        with open(self.save_dir / "final_metrics.json", "w") as f:
            json.dump(report, f, indent=2)

        logger.info("=" * 70)
        logger.info(f"FINAL (best checkpoint, epoch {self.best_epoch}) on {self.eval_tag}")
        logger.info(f"  IoU at best threshold ({r['best_threshold']:.2f}) : {r['best_iou']:.4f}")
        logger.info(f"  IoU at threshold 0.50        : {r['iou_at_0.5']:.4f}")
        logger.info(f"  F1                           : {r['best_f1']:.4f}")
        for name, s in report["per_sequence"].items():
            logger.info(f"  {name:40s} IoU={s['best_iou']:.4f}  IoU@0.5={s['iou_at_0.5']:.4f}")
        logger.info(f"  saved: {self.save_dir / 'final_metrics.json'}")
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_step(self, epoch, batch_idx, steps_per_epoch, lo, grad_norm):
        gs = self.global_step
        for k in ("loss", "mask_loss", "bce_loss", "dice_loss", "depth_loss", "pose_loss",
                  "pred_dynamic_ratio", "gt_dynamic_ratio", "ignored_fraction"):
            v = lo.get(k)
            if isinstance(v, torch.Tensor):
                self.writer.add_scalar(f"train/{k}", v.item(), gs)
        self.writer.add_scalar("train/grad_norm", float(grad_norm), gs)
        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], gs)
        logger.info(
            f"E{epoch + 1} B{batch_idx:4d}/{steps_per_epoch} | loss={lo['loss'].item():.4f} | "
            f"bce={lo['bce_loss'].item():.4f} dice={lo['dice_loss'].item():.4f} | "
            f"depth={lo['depth_loss'].item():.4f} pose={lo['pose_loss'].item():.4f} | "
            f"pred_dr={lo['pred_dynamic_ratio'].item():.3f} gt_dr={lo['gt_dynamic_ratio'].item():.3f} | "
            f"gn={float(grad_norm):.2f}"
        )

    @torch.no_grad()
    def _log_images(self, gs, outputs, targets, voxels):
        probs = outputs["mask_probs"].float().cpu()
        gt = targets["gt_mask"].float()
        events = voxels[:, -1].abs().sum(dim=1, keepdim=True).float().cpu()
        n = min(self.cfg.viz_max_samples, probs.shape[0])
        for i in range(n):
            ev = events[i] / (events[i].max() + 1e-6)
            overlay = torch.zeros(3, *probs.shape[-2:])
            overlay[0] = probs[i, 0]                       # red   = prediction
            overlay[1] = gt[i, 0]                          # green = ground truth
            overlay[2] = ((probs[i, 0] > 0.5) & (gt[i, 0] > 0.5)).float()
            self.writer.add_image(f"sample_{i}/1_events", ev, gs)
            self.writer.add_image(f"sample_{i}/2_prediction", probs[i], gs)
            self.writer.add_image(f"sample_{i}/3_ground_truth", gt[i], gs)
            self.writer.add_image(f"sample_{i}/4_overlay_red_pred_green_truth", overlay, gs)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save(self, name: str, epoch: int):
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "pos_weight": self.pos_weight,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": asdict(self.cfg),
        }
        if self.ema is not None:
            ckpt["ema_state_dict"] = self.ema.state_dict()
        torch.save(ckpt, self.save_dir / "checkpoints" / name)

    def _resume(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_metric = ckpt.get("best_metric", -1.0)
        self.best_epoch = ckpt.get("best_epoch", -1)
        logger.info(f"Resumed from {path} at epoch {self.start_epoch + 1} "
                    f"(best so far {self.best_metric:.4f} @ epoch {self.best_epoch})")


# ======================================================================
# Stand-alone evaluation of a saved checkpoint (e.g. on the test split)
# ======================================================================

def evaluate_checkpoint(cfg: TrainConfigSupervised, checkpoint: str, split: str,
                        sequence=None, threshold: float | None = None) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt.get("config", {})
    for field in ARCHITECTURE_FIELDS:
        if field in saved:
            value = saved[field]
            setattr(cfg, field, tuple(value) if isinstance(value, list) else value)
    if threshold is not None:
        cfg.eval_thresholds = tuple(sorted(set(cfg.eval_thresholds) | {threshold}))

    trainer = TrainerSupervised(cfg, build_train=False)
    sets = trainer._temporal_datasets(split, sequence)
    if not sets:
        raise RuntimeError(f"No data for split {split!r} under {cfg.dataset_root}")
    loader = trainer._loader(sets, shuffle=False)

    state = ckpt["ema_state_dict"]["module"] if "ema_state_dict" in ckpt else ckpt["model_state_dict"]
    trainer.model.load_state_dict(state)
    r = trainer.evaluate(loader, model=trainer.model, per_sequence=True)

    logger.info("=" * 70)
    logger.info(f"Checkpoint : {checkpoint} (epoch {ckpt.get('epoch', -1) + 1})")
    logger.info(f"Split      : {split}  ({sum(len(s) for s in sets)} windows)")
    logger.info(f"{'threshold':>10s} {'IoU':>8s} {'F1':>8s} {'precision':>10s} {'recall':>8s}")
    for key, v in r["per_threshold"].items():
        logger.info(f"{key[4:]:>10s} {v['iou']:8.4f} {v['f1']:8.4f} {v['precision']:10.4f} {v['recall']:8.4f}")
    if threshold is not None:
        fixed = r["per_threshold"][f"thr_{threshold:.2f}"]
        logger.info(f"REPORTED (fixed threshold {threshold:.2f}): IoU={fixed['iou']:.4f} F1={fixed['f1']:.4f}")
    for name, s in r.get("per_sequence", {}).items():
        logger.info(f"  {name:40s} IoU={s['best_iou']:.4f}  IoU@0.5={s['iou_at_0.5']:.4f}")
    logger.info("=" * 70)

    out_path = Path(cfg.save_dir) / f"eval_{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in r.items()}, f, indent=2, default=float)
    logger.info(f"Saved {out_path}")
    return r
