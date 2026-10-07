"""
End-to-end test of the supervised pipeline on SYNTHETIC batches
(no dataset needed). Run before a real training job:

    python tests/test_supervised_model.py

Checks
------
1. Model output shapes, with and without inertial input / helper heads.
2. Target construction: speed-aware mask, missing masks, missing depth,
   missing pose.
3. Loss is finite and gradients reach every trained parameter.
4. The full trainer (balanced data, exponential moving average,
   scheduler, checkpointing, final report) drives IoU above 0.90 on a
   learnable synthetic task: a moving object vs. a static object, both
   producing events.
5. best.pth / last.pth / final_metrics.json are written and resume works.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.supervised_model import SupervisedMotionSegmenter, SupervisedLoss, build_targets
from trainer_supervised import TrainConfigSupervised, TrainerSupervised

H, W, C, T = 48, 64, 5, 3
DYNAMIC_ID, STATIC_ID = 3, 7


# ---------------------------------------------------------------------
# Synthetic EVIMO2-like batches
# ---------------------------------------------------------------------

class FakeVoxelFrame:
    def __init__(self, voxel, n_imu, B, rng):
        self.voxel_grid = voxel
        self.imu_timestamps = torch.linspace(0, 1, n_imu).repeat(B)
        self.imu_angular_velocity = torch.tensor(rng.normal(size=(n_imu * B, 3)), dtype=torch.float32)
        self.imu_linear_acceleration = torch.tensor(rng.normal(size=(n_imu * B, 3)), dtype=torch.float32)
        self.imu_sample_indices = torch.arange(B).repeat_interleave(n_imu)

    def to(self, device):
        for k in ("voxel_grid", "imu_timestamps", "imu_angular_velocity",
                  "imu_linear_acceleration", "imu_sample_indices"):
            setattr(self, k, getattr(self, k).to(device))
        return self


class FakeVoxelBatch:
    def __init__(self, frames):
        self.frames = frames

    def to(self, device):
        for f in self.frames:
            f.to(device)
        return self


def make_batch(rng, B=4, with_object_prob=0.75, drop_mask=False, drop_depth=False, drop_pose=False):
    """Moving square (time-shifted events) + static square (constant events) + noise."""
    voxels = np.zeros((T, B, C, H, W), np.float32)
    masks, motions, depths, cams_t, cams_s = [], [], [], [], []
    for b in range(B):
        has_dynamic = rng.random() < with_object_prob
        sx, sy = rng.integers(2, W - 14), rng.integers(2, H - 14)       # static object
        dx, dy = rng.integers(2, W - 24), rng.integers(2, H - 14)       # dynamic object
        mask = np.zeros((H, W), np.uint16)
        mask[sy:sy + 10, sx:sx + 10] = STATIC_ID * 1000
        for t in range(T):
            voxels[t, b] += rng.random((C, H, W)) < 0.03                  # background noise
            voxels[t, b, :, sy:sy + 10, sx:sx + 10] += 0.8                # static: same every bin
            if has_dynamic:
                for c in range(C):                                        # dynamic: moves across bins
                    x0 = dx + 2 * (t * C + c) // C
                    voxels[t, b, c, dy:dy + 10, x0:x0 + 10] += 1.0
        if has_dynamic:
            x_last = dx + 2 * ((T - 1) * C + C - 1) // C
            mask[dy:dy + 10, x_last:x_last + 10] = DYNAMIC_ID * 1000
        masks.append(None if (drop_mask and b == 0) else mask)
        motions.append(SimpleNamespace(
            object_ids=np.array([DYNAMIC_ID, STATIC_ID]),
            speed=np.array([0.3 if has_dynamic else 0.0, 0.0]),
        ))
        depths.append(None if (drop_depth and b == 0) else (1.0 + rng.random((H, W))).astype(np.float32))
        q = np.array([0, 0, np.sin(0.01), np.cos(0.01)])
        cams_t.append(SimpleNamespace(translation=np.array([0.01, 0, 0]), quaternion=q,
                                      pose_available=not (drop_pose and b == 0)))
        cams_s.append(SimpleNamespace(translation=np.zeros(3), quaternion=np.array([0, 0, 0, 1.0]),
                                      pose_available=True))

    raw_frames = []
    for t in range(T):
        last = t == T - 1
        raw_frames.append(SimpleNamespace(
            mask=masks if last else [None] * B,
            frame_motion=motions if last else [None] * B,
            depth=depths if last else [None] * B,
            camera_motion=cams_t if last else cams_s,
            sequence_names=[f"synthetic_seq_{b % 2}" for b in range(B)],
        ))
    raw = SimpleNamespace(frames=raw_frames)
    vox_frames = [FakeVoxelFrame(torch.from_numpy(voxels[t].copy()), 8, B, rng) for t in range(T)]
    return raw, FakeVoxelBatch(vox_frames)


class FakeLoader:
    """Iterable of raw batches; transform() looks up the matching voxel batch."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __iter__(self):
        return iter([raw for raw, _ in self.pairs])

    def __len__(self):
        return len(self.pairs)


def make_transform(loaders):
    lookup = {}
    for loader in loaders:
        for raw, vox in loader.pairs:
            lookup[id(raw)] = vox
    return lambda raw: lookup[id(raw)]


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_shapes():
    rng = np.random.default_rng(0)
    raw, vox = make_batch(rng)
    voxels = torch.stack([f.voxel_grid for f in vox.frames], dim=1)
    for use_imu, depth, pose in [(True, True, True), (False, False, False), (True, False, True)]:
        m = SupervisedMotionSegmenter(num_bins=C, num_frames=T, use_imu=use_imu,
                                      predict_depth=depth, predict_pose=pose)
        out = m(voxels, vox.frames if use_imu else None)
        assert out["mask"].shape == (4, 1, H, W)
        assert (out["depth_log"] is not None) == depth
        assert (out["pose"] is not None) == pose
        if depth:
            assert out["depth_log"].shape[-2:] == (H // 4, W // 4)
        if pose:
            assert out["pose"].shape == (4, 9)
    # odd size must still work (interpolation to skip size)
    m = SupervisedMotionSegmenter(num_bins=C, num_frames=T, use_imu=False)
    assert m(torch.zeros(1, T, C, 45, 61))["mask"].shape == (1, 1, 45, 61)
    print("  shapes ................................ ok")


def test_targets_and_loss():
    rng = np.random.default_rng(1)
    raw, vox = make_batch(rng, with_object_prob=1.0, drop_mask=True, drop_depth=True, drop_pose=True)
    t = build_targets(raw, mask_hw=(H, W), depth_hw=(H // 4, W // 4), need_pose=True)
    assert t["mask_valid"].tolist() == [False, True, True, True]
    assert t["depth_valid"].tolist() == [False, True, True, True]
    assert t["pose_valid"].tolist() == [False, True, True, True]
    assert t["gt_mask"][1].sum() == 100            # only the moving 10x10 square
    # T_target_source = inv(T_world_target) @ T_world_source, source at the
    # origin -> translation = -R_target^T @ t_target
    a = 0.02  # quaternion z = sin(0.01) -> rotation of 0.02 rad about z
    R = torch.tensor([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=torch.float32)
    expected_t = -(R.T @ torch.tensor([0.01, 0.0, 0.0]))
    assert torch.allclose(t["gt_pose"][1, :3], expected_t, atol=1e-6), t["gt_pose"][1, :3]
    assert torch.allclose(t["gt_pose"][1, 3:6], R.T[:, 0], atol=1e-6)

    m = SupervisedMotionSegmenter(num_bins=C, num_frames=T)
    voxels = torch.stack([f.voxel_grid for f in vox.frames], dim=1)
    out = m(voxels, vox.frames)
    lo = SupervisedLoss(pos_weight=5.0)(out, t)
    assert torch.isfinite(lo["loss"])
    lo["loss"].backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient for: {missing[:5]}"
    print("  targets, loss, gradients .............. ok")


def test_training_reaches_90(tmp: Path):
    rng = np.random.default_rng(2)
    train = FakeLoader([make_batch(rng) for _ in range(12)])
    val = FakeLoader([make_batch(rng) for _ in range(4)])
    transform = make_transform([train, val])

    cfg = TrainConfigSupervised(
        dataset_root="unused", history_offsets=(-2, -1, 0), num_bins=C,
        epochs=12, learning_rate=1e-3, warmup_fraction=0.05,
        eval_every_n_epochs=2, checkpoint_every_n_epochs=6,
        log_every_n_steps=1000, viz_every_n_steps=20,
        save_dir=str(tmp / "run"), num_workers=0, mixed_precision="none",
    )
    trainer = TrainerSupervised(cfg, train_loader=train, eval_loader=val,
                                transform=transform, eval_tag="synthetic_val")
    trainer.train()

    ck = tmp / "run" / "checkpoints"
    assert (ck / "best.pth").exists() and (ck / "last.pth").exists() and (ck / "epoch_006.pth").exists()
    report = json.loads((tmp / "run" / "final_metrics.json").read_text())
    print(f"  synthetic held-out IoU = {report['best_iou']:.4f} "
          f"(IoU@0.5 = {report['iou_at_0.5']:.4f}, best epoch {report['best_epoch']})")
    assert set(report["per_sequence"]) == {"synthetic_seq_0", "synthetic_seq_1"}
    assert report["best_iou"] > 0.90, "training did not reach 0.90 IoU on the synthetic task"

    # resume: continues from last.pth with best metric restored
    cfg.epochs, cfg.resume_from = 13, str(ck / "last.pth")
    t2 = TrainerSupervised(cfg, train_loader=train, eval_loader=val,
                           transform=transform, eval_tag="synthetic_val")
    assert t2.start_epoch == 12 and t2.best_metric == trainer.best_metric
    t2.train()
    print("  training > 0.90, checkpoints, resume .. ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp())
    try:
        print("Supervised pipeline self-test")
        test_shapes()
        test_targets_and_loss()
        test_training_reaches_90(tmp)
        print("ALL TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
