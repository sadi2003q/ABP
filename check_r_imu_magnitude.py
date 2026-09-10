"""
Check the actual magnitude of R_imu (the integrated-gyro rotation
prior computed inside WorldModelV3) across the sequence's frames.

This answers: is the near-identical v2/v3 training trajectory because
this sequence genuinely has almost no real rotation for the IMU
anchor to constrain (expected, and would mean the fix needs a more
dynamic sequence to show its effect) -- or because R_imu is being
computed incorrectly and is ~identity even when it shouldn't be
(a real bug)?

Run from /content/ABP:
    python check_r_imu_magnitude.py \
        --dataset-root /content/drive/MyDrive/single_seq_root \
        --sensors left_camera --subset imo --split train \
        --sequence scene15_dyn_test_06_000000 \
        --batch-size 4 --overfit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from trainer_v3 import TrainerV3, TrainConfigV3
from src.models.world_model_v2.imu_integration import integrate_frame_rotation


def rotation_angle_degrees(R: torch.Tensor) -> torch.Tensor:
    """
    Geodesic rotation angle of R (radians -> degrees), via
    theta = arccos((trace(R) - 1) / 2).

    R: (..., 3, 3)
    Returns: (...,) angle in degrees.
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    return theta * (180.0 / torch.pi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--sensors", type=str, nargs="+", default=["left_camera"])
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--subset", type=str, default="imo")
    p.add_argument("--sequence", type=str, nargs="+", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--history-offsets", type=int, nargs="+", default=[-12, -8, -4, 0])
    p.add_argument("--num-bins", type=int, default=5)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--max-batches", type=int, default=20,
                    help="How many batches to scan (each has T frames).")
    args = p.parse_args()

    cfg = TrainConfigV3(
        dataset_root=args.dataset_root,
        sensors=tuple(args.sensors),
        split=args.split if not args.overfit else "train",
        val_split=None,
        subset=args.subset,
        sequence=tuple(args.sequence) if args.sequence else None,
        history_offsets=tuple(args.history_offsets),
        num_bins=args.num_bins,
        batch_size=args.batch_size,
        num_workers=0,
        use_ema=False,
        overfit_mode=args.overfit,
        mixed_precision="none",
    )
    trainer = TrainerV3(cfg)

    loader = trainer.train_loader
    device = trainer.device

    all_angles = []
    n_zero_imu_frames = 0
    n_total_frames = 0

    for batch_idx, raw_batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        voxel_batch = trainer.transform(raw_batch)
        voxel_batch = voxel_batch.to(device)

        B = len(voxel_batch.frames[0].metadata.camera_intrinsics)

        for t, frame in enumerate(voxel_batch.frames):
            R_imu = integrate_frame_rotation(
                imu_timestamps=frame.metadata.imu_timestamps,
                imu_angular_velocity=frame.metadata.imu_angular_velocity,
                imu_sample_indices=frame.metadata.imu_sample_indices,
                batch_size=B,
                device=device,
                dtype=torch.float32,
            )
            angles = rotation_angle_degrees(R_imu)  # (B,)
            all_angles.extend(angles.tolist())
            n_total_frames += B

            # Count identity fallbacks (< 2 IMU samples in that frame)
            for b in range(B):
                mask = frame.metadata.imu_sample_indices == b
                n = int((torch.as_tensor(frame.metadata.imu_sample_indices) == b).sum()) \
                    if not torch.is_tensor(frame.metadata.imu_sample_indices) else int(mask.sum())
                if n < 2:
                    n_zero_imu_frames += 1

    all_angles_t = torch.tensor(all_angles)
    print("=" * 70)
    print("R_imu ROTATION ANGLE MAGNITUDE (degrees) ACROSS SCANNED BATCHES")
    print("=" * 70)
    print(f"Total (frame, batch-item) samples scanned: {n_total_frames}")
    print(f"Frames with <2 IMU samples (identity fallback): {n_zero_imu_frames}")
    print()
    print(f"min    : {all_angles_t.min().item():.6f} deg")
    print(f"max    : {all_angles_t.max().item():.6f} deg")
    print(f"mean   : {all_angles_t.mean().item():.6f} deg")
    print(f"median : {all_angles_t.median().item():.6f} deg")
    print(f"std    : {all_angles_t.std().item():.6f} deg")
    print()
    print("Interpretation:")
    print(" - If mean/median angle is << 1 degree per frame, this sequence")
    print("   genuinely has almost no real rotation for R_imu to anchor --")
    print("   the v3 fix would need a more dynamic sequence to show an effect.")
    print(" - If angles are consistently ~0.000000 (exactly zero, not just")
    print("   small), that suggests a bug (e.g. gyro reading as all-zero,")
    print("   or timestamps/sample_indices misaligned) rather than a")
    print("   genuinely stationary camera -- compare against the earlier")
    print("   check_imu_density.py gyro std/min/max on this same sequence.")


if __name__ == "__main__":
    main()