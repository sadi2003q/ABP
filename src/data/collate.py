"""
Collate function for TemporalEVIMO2Dataset.

Unlike the frame dataset, every dataset item already contains
multiple temporally ordered EVIMO2Sample objects.

This collate function preserves that temporal ordering while
concatenating events independently for every timestep.
"""

from __future__ import annotations

import numpy as np

from src.data.temporal_sample import (
    TemporalEVIMO2Sample,
)

from src.data.sample import (
    EVIMO2Batch,
    TemporalEVIMO2Batch,
)

def concatenate_events(samples):
    """
    Concatenate events from multiple EVIMO2Sample objects.

    Parameters
    ----------
    samples
        list[EVIMO2Sample]

    Returns
    -------
    tuple
        (
            events_xy,
            events_t,
            events_p,
            event_sample_indices,
        )
    """

    xy = []
    t = []
    p = []
    sample_index = []

    for sample_id, sample in enumerate(samples):

        n = len(sample.events_t)

        xy.append(sample.events_xy)
        t.append(sample.events_t)
        p.append(sample.events_p)

        sample_index.append(
            np.full(
                n,
                sample_id,
                dtype=np.int32,
            )
        )

    return (

        np.concatenate(xy, axis=0),

        np.concatenate(t, axis=0),

        np.concatenate(p, axis=0),

        np.concatenate(sample_index, axis=0),
    )


def concatenate_imu(samples):
    """
    Concatenate IMU measurements from multiple EVIMO2Sample objects.

    Returns
    -------
    (
        timestamps,
        angular_velocity,
        linear_acceleration,
        sample_indices,
    )
    """

    timestamps = []
    gyro = []
    acc = []
    sample_indices = []

    for sample_id, sample in enumerate(samples):

        n = len(sample.imu.timestamps)

        timestamps.append(sample.imu.timestamps)
        gyro.append(sample.imu.angular_velocity)
        acc.append(sample.imu.linear_acceleration)

        sample_indices.append(
            np.full(
                n,
                sample_id,
                dtype=np.int32,
            )
        )

    return (
        np.concatenate(timestamps, axis=0),
        np.concatenate(gyro, axis=0),
        np.concatenate(acc, axis=0),
        np.concatenate(sample_indices, axis=0),
    )


def temporal_collate_fn(
    batch: list[TemporalEVIMO2Sample],
) -> TemporalEVIMO2Batch:
    """
    Collate TemporalEVIMO2Samples.

    Parameters
    ----------
    batch
        List of TemporalEVIMO2Sample objects.

    Returns
    -------
    TemporalEVIMO2Batch

    Notes
    -----
    The output preserves temporal ordering.

    For every timestep, one EVIMO2Batch is created.

    Example
    -------

    history_offsets = (-3,-2,-1,0)

    output.frames[0] -> batch for t-3

    output.frames[1] -> batch for t-2

    output.frames[2] -> batch for t-1

    output.frames[3] -> batch for t
    """

    temporal_size = len(batch[0].frames)

    frame_batches: list[EVIMO2Batch] = []

    for t_index in range(temporal_size):

        #
        # Samples belonging to this timestep
        #
        timestep_samples = [
            sample.frames[t_index]
            for sample in batch
        ]

        #
        # Concatenate events
        #
        events_xy, events_t, events_p, event_sample_indices = (
            concatenate_events(
                timestep_samples
            )
        )

        #
        # Concatenate IMU
        #
        
        imu_timestamps, imu_angular_velocity, imu_linear_acceleration, imu_sample_indices = (
            concatenate_imu(
                timestep_samples
            )
        )

        #
        # Stack dense tensors
        # #
        # print("\nCurrent timestep")

        # for s in timestep_samples:
        #     print(
        #         s.sensor,
        #         None if s.depth is None else s.depth.shape,
        #         None if s.mask is None else s.mask.shape,
        #         None if s.rgb is None else s.rgb.shape,
        #     )

        #
        # --------------------------------------------------
        # Dense tensors
        # --------------------------------------------------
        #

        depth = [s.depth for s in timestep_samples]

        mask = [s.mask for s in timestep_samples]

        rgb = [s.rgb for s in timestep_samples]

        camera_intrinsics = [
            s.camera_intrinsics
            for s in timestep_samples
        ]


        camera_distortion = [
            s.camera_distortion
            for s in timestep_samples
        ]

        #
        # Build one frame batch
        #
        frame_batch = EVIMO2Batch(

            sequence_names=[
                s.sequence_name
                for s in timestep_samples
            ],

            sensors=[
                s.sensor
                for s in timestep_samples
            ],

            local_frame_indices=np.asarray(
                [s.local_frame_index for s in timestep_samples],
                dtype=np.int32,
            ),

            frame_ids=np.asarray(
                [s.frame_id for s in timestep_samples],
                dtype=np.int32,
            ),

            timestamps=np.asarray(
                [s.timestamp for s in timestep_samples],
                dtype=np.float64,
            ),

            events_xy=events_xy,

            events_t=events_t,

            events_p=events_p,

            event_sample_indices=event_sample_indices,

            camera_motion=[
                s.camera_motion
                for s in timestep_samples
            ],

            frame_motion=[
                s.frame_motion
                for s in timestep_samples
            ],

            camera_intrinsics=camera_intrinsics,

            camera_distortion=camera_distortion,

            imu_timestamps=imu_timestamps,

            imu_angular_velocity=imu_angular_velocity,

            imu_linear_acceleration=imu_linear_acceleration,

            imu_sample_indices=imu_sample_indices,

            depth=depth,

            mask=mask,

            rgb=rgb,
        )

        frame_batches.append(frame_batch)

    return TemporalEVIMO2Batch(

        frames=frame_batches,

        history_offsets=batch[0].history_offsets,
 
    )