from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.sample import (
    EVIMO2Batch,
    TemporalEVIMO2Batch,
)


from src.data.voxelizer import Voxelizer
from src.data.voxel_sample import (
    VoxelFrameBatch,
    VoxelTemporalBatch,
)


# ==========================================================
# Tensor dataclasses
# ==========================================================



@dataclass(slots=True)
class TensorFrameBatch:
    """
    Tensor version of one EVIMO2Batch.

    Only tensors consumed by the model are converted.
    """

    #
    # Events
    #

    events_xy: torch.Tensor

    events_t: torch.Tensor

    events_p: torch.Tensor

    event_sample_indices: torch.Tensor

    #
    # IMU
    #

    imu_timestamps: torch.Tensor

    imu_angular_velocity: torch.Tensor

    imu_linear_acceleration: torch.Tensor

    imu_sample_indices: torch.Tensor

    #
    # Everything else
    #

    metadata: EVIMO2Batch

@dataclass(slots=True)
class TensorTemporalBatch:
    """
    Tensor version of TemporalEVIMO2Batch.
    """

    frames: list[TensorFrameBatch]

    history_offsets: tuple[int, ...]




# ==========================================================
# Compose
# ==========================================================

class Compose:

    def __init__(self, transforms):

        self.transforms = list(transforms)

    def __call__(self, batch):

        for transform in self.transforms:

            batch = transform(batch)

        return batch



# ==========================================================
# ToTensor
# ==========================================================

class ToTensor:
    """
    Convert only model inputs to torch tensors.

    Metadata, camera motion, object motion, RGB, depth and masks
    remain inside the original EVIMO2Batch.
    """

    def __call__(
        self,
        batch: TemporalEVIMO2Batch,
    ) -> TensorTemporalBatch:

        tensor_frames = []

        for frame in batch.frames:

            tensor_frames.append(

                TensorFrameBatch(

                    #
                    # Events
                    #

                    events_xy=torch.from_numpy(
                        frame.events_xy
                    ).long(),

                    events_t=torch.from_numpy(
                        frame.events_t
                    ).float(),

                    events_p=torch.from_numpy(
                        frame.events_p
                    ).float(),

                    event_sample_indices=torch.from_numpy(
                        frame.event_sample_indices
                    ).long(),

                    #
                    # IMU
                    #

                    imu_timestamps=torch.from_numpy(
                        frame.imu_timestamps
                    ).float(),

                    imu_angular_velocity=torch.from_numpy(
                        frame.imu_angular_velocity
                    ).float(),

                    imu_linear_acceleration=torch.from_numpy(
                        frame.imu_linear_acceleration
                    ).float(),

                    imu_sample_indices=torch.from_numpy(
                        frame.imu_sample_indices
                    ).long(),

                    #
                    # Metadata
                    #

                    metadata=frame,
                )
            )

        return TensorTemporalBatch(

            frames=tensor_frames,

            history_offsets=batch.history_offsets,
        )


# ==========================================================
# Normalize event timestamps
# ==========================================================

class NormalizeEventTime:
    """
    Normalize event timestamps independently for every sample.

    Each sample is normalized to

        first event -> 0
        last event  -> 1

    using event_sample_indices.

    This ensures every frame's event packet has its own normalized
    temporal axis regardless of batching.
    """

    def __call__(
        self,
        batch: TensorTemporalBatch,
    ) -> TensorTemporalBatch:

        for frame in batch.frames:

            sample_ids = torch.unique(
                frame.event_sample_indices
            )

            for sample_id in sample_ids:

                mask = (
                    frame.event_sample_indices
                    == sample_id
                )

                t = frame.events_t[mask]

                #
                # No events
                #
                if t.numel() == 0:
                    continue

                #
                # Single event
                #
                if t.numel() == 1:

                    frame.events_t[mask] = 0.0

                    continue

                t0 = t[0]
                t1 = t[-1]

                dt = t1 - t0

                if dt > 0:

                    frame.events_t[mask] = (
                        t - t0
                    ) / dt

                else:

                    frame.events_t[mask] = 0.0

        return batch


# ==========================================================
# Normalize IMU
# ==========================================================

class NormalizeIMU:
    """
    Normalize IMU timestamps independently for every sample.

    Each sample is normalized to

        first imu -> 0
        last imu  -> 1
    """

    def __call__(
        self,
        batch: TensorTemporalBatch,
    ) -> TensorTemporalBatch:

        for frame in batch.frames:

            sample_ids = torch.unique(
                frame.imu_sample_indices
            )

            for sample_id in sample_ids:

                mask = (
                    frame.imu_sample_indices
                    == sample_id
                )

                t = frame.imu_timestamps[mask]

                if t.numel() == 0:
                    continue

                if t.numel() == 1:

                    frame.imu_timestamps[mask] = 0.0

                    continue

                t0 = t[0]
                t1 = t[-1]

                dt = t1 - t0

                if dt > 0:

                    frame.imu_timestamps[mask] = (
                        t - t0
                    ) / dt

                else:

                    frame.imu_timestamps[mask] = 0.0

        return batch


# ==========================================================
# Voxelization
# ==========================================================


class VoxelizeEvents:
    """
    Convert normalized event packets into voxel grids.

    One voxel grid is produced for every sample inside every temporal
    frame.

    Output shape
    ------------
    (batch_size,
     num_bins,
     H,
     W)
    """

    def __init__(
        self,
        num_bins: int = 5,
        height: int = 480,
        width: int = 640,
    ):

        self.voxelizer = Voxelizer(
            height=height,
            width=width,
            num_bins=num_bins,
        )

    def __call__(
        self,
        batch: TensorTemporalBatch,
    ) -> VoxelTemporalBatch:

        voxel_frames = []

        #
        # Every temporal frame
        #
        for frame in batch.frames:

            metadata = frame.metadata

            batch_size = len(metadata.sequence_names)

            voxel_list = []

            #
            # Build one voxel grid per sample
            #
            for sample_id in range(batch_size):

                mask = (
                    frame.event_sample_indices
                    == sample_id
                )

                voxel = self.voxelizer(

                    frame.events_xy[mask],

                    frame.events_t[mask],

                    frame.events_p[mask],
                )

                voxel_list.append(voxel)

            voxel_frames.append(

                VoxelFrameBatch(

                    voxel_grid=torch.stack(
                        voxel_list,
                        dim=0,
                    ),

                    camera_intrinsics=torch.stack(
                        [
                            torch.from_numpy(K).float()
                            for K in metadata.camera_intrinsics
                        ],
                        dim=0,
                    ),

                    camera_distortion=torch.stack(
                        [
                            torch.from_numpy(D).float()
                            for D in metadata.camera_distortion
                        ],
                        dim=0,
                    ),

                    imu_timestamps=frame.imu_timestamps,
                    imu_angular_velocity=frame.imu_angular_velocity,
                    imu_linear_acceleration=frame.imu_linear_acceleration,
                    imu_sample_indices=frame.imu_sample_indices,

                    metadata=metadata,
                )
            )

        return VoxelTemporalBatch(

            frames=voxel_frames,

            history_offsets=batch.history_offsets,
        )
