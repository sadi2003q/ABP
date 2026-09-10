"""
Run this in Colab from the ABP repo root (same environment you used for
train_v2.py / evaluate_with_gt_depth.py).

Checks:
  1. Raw IMU sample count vs. frame count for the sequence.
  2. IMU samples per frame (via the cached imu_index) — flags frames
     with zero or very few IMU samples.
  3. Basic sanity on the gyro/accel magnitudes (all-zero would mean
     the IMU stream is a stub, not real data).
"""

import sys
from pathlib import Path

"""
Full Dataset location

/run/media/adnan-abdullah-sadi/Extreme SSD/EVIMO2_official/left_camera/imo/train/scene6_dyn_train_03_000000
"""

sys.path.insert(0, "/home/adnan-abdullah-sadi/Coding/ABP")  # adjust if your repo path differs

import numpy as np


from src.data.evimo2.parser import EVIMO2Parser
from src.preprocessing.imu_index import load_imu_index

# ---- EDIT THESE TWO LINES ----
DATASET_ROOT = "/run/media/adnan-abdullah-sadi/Extreme SSD/EVIMO2_official/"
SEQUENCE_DIR = (
    Path(DATASET_ROOT)
    / "left_camera" / "imo" / "train" / "scene6_dyn_train_03_000000"
)
# -------------------------------

print("=" * 70)
print("IMU DENSITY CHECK")
print("=" * 70)

parser = EVIMO2Parser(SEQUENCE_DIR)
n_frames = len(parser.frames)
print(f"Frames in sequence: {n_frames}")

imu = parser.imu  # dict: sensor_name -> {"timestamps", "gyro", "acceleration"}
if not imu:
    print("!! parser.imu is EMPTY — no IMU sensors were parsed at all.")
else:
    for sensor_name, data in imu.items():
        n_imu = len(data["timestamps"])
        print(f"\nSensor: {sensor_name}")
        print(f"  IMU samples total   : {n_imu}")
        print(f"  IMU samples / frame  (avg): {n_imu / max(n_frames, 1):.2f}")

        gyro = data["gyro"]
        accel = data["acceleration"]
        print(f"  gyro   min/max/mean/std : "
              f"{gyro.min():.4f} / {gyro.max():.4f} / "
              f"{gyro.mean():.4f} / {gyro.std():.4f}")
        print(f"  accel  min/max/mean/std : "
              f"{accel.min():.4f} / {accel.max():.4f} / "
              f"{accel.mean():.4f} / {accel.std():.4f}")

        if np.allclose(gyro, 0) and np.allclose(accel, 0):
            print("  !! WARNING: gyro and accel are ALL ZERO — IMU stream "
                  "looks like a stub, not real sensor data.")

# ---- Per-frame IMU coverage via the cache (matches what the Dataset uses) ----
print("\n" + "-" * 70)
print("Per-frame IMU coverage (from cache/imu_index.npz)")
print("-" * 70)

try:
    imu_idx = load_imu_index(SEQUENCE_DIR)
    counts = imu_idx.imu_end.astype(np.int64) - imu_idx.imu_start.astype(np.int64)
    zero_frames = int((counts == 0).sum())
    print(f"Frames with ZERO imu samples : {zero_frames} / {len(counts)}")
    print(f"Min / median / max samples per frame : "
          f"{counts.min()} / {int(np.median(counts))} / {counts.max()}")
    if zero_frames > 0:
        bad = np.where(counts == 0)[0][:10]
        print(f"First zero-IMU frame indices: {bad.tolist()}")
except FileNotFoundError:
    print("No cache/imu_index.npz found for this sequence — "
          "the Dataset would build it (or crash) on first load. "
          "Run the dataset once (e.g. via train_v2.py) or the "
          "preprocessing tool to generate it first.")

print("\nDone.")