"""
Voxelized batches produced by the VoxelizeEvents transform.

These classes represent the final input fed into the neural network.

Raw event tensors are replaced by voxel grids while all remaining
metadata is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.sample import EVIMO2Batch


# ==========================================================
# One temporal frame
# ==========================================================

@dataclass(slots=True)
class VoxelFrameBatch:
    """
    One temporal frame after voxelization.

    Contains both the voxelized events and the normalized IMU data.
    """

    # ------------------------------------------------------
    # Events
    # ------------------------------------------------------

    voxel_grid: torch.Tensor

    # ------------------------------------------------------
    # Camera calibration
    # ------------------------------------------------------

    camera_intrinsics: torch.Tensor
    """
    (B, 3, 3)
    """

    camera_distortion: torch.Tensor
    """
    (B, 4)
    """

    # ------------------------------------------------------
    # IMU
    # ------------------------------------------------------

    imu_timestamps: torch.Tensor

    imu_angular_velocity: torch.Tensor

    imu_linear_acceleration: torch.Tensor

    imu_sample_indices: torch.Tensor

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    metadata: EVIMO2Batch

    # ------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------

    def to(self, device: torch.device) -> "VoxelFrameBatch":
        return VoxelFrameBatch(
            voxel_grid=self.voxel_grid.to(device),

            camera_intrinsics=self.camera_intrinsics.to(device),
            camera_distortion=self.camera_distortion.to(device),

            imu_timestamps=self.imu_timestamps.to(device),
            imu_angular_velocity=self.imu_angular_velocity.to(device),
            imu_linear_acceleration=self.imu_linear_acceleration.to(device),
            imu_sample_indices=self.imu_sample_indices.to(device),

            metadata=self.metadata,
        )


# ==========================================================
# Whole temporal sample
# ==========================================================

@dataclass(slots=True)
class VoxelTemporalBatch:
    """
    Output of the VoxelizeEvents transform.

    Every temporal position contains one voxelized frame.
    """

    frames: list[VoxelFrameBatch]

    history_offsets: tuple[int, ...]

    # ------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------

    def to(self, device: torch.device) -> "VoxelTemporalBatch":
        return VoxelTemporalBatch(
            frames=[
                frame.to(device)
                for frame in self.frames
            ],
            history_offsets=self.history_offsets,
        )