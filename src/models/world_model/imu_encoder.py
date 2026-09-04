from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock1D


class IMUEncoder(nn.Module):
    """
    Encode one IMU sequence into a compact motion embedding.

    The input is the flattened representation produced by the dataloader.
    Internally, the encoder reconstructs each sample's IMU sequence using
    imu_sample_indices.

    Output
    ------
    (batch_size, embedding_dim)
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        embedding_dim: int = 128,
    ):
        super().__init__()

        #
        # Input feature:
        #
        # [timestamp,
        #  gyro_x, gyro_y, gyro_z,
        #  accel_x, accel_y, accel_z]
        #

        self.input_projection = nn.Linear(
            7,
            hidden_channels,
        )

        self.temporal_encoder = nn.Sequential(

            ConvBlock1D(
                hidden_channels,
                hidden_channels,
            ),

            ConvBlock1D(
                hidden_channels,
                hidden_channels,
            ),

            ConvBlock1D(
                hidden_channels,
                hidden_channels,
            ),
        )

        self.output_projection = nn.Linear(
            hidden_channels,
            embedding_dim,
        )

    def forward(
        self,
        frame,
        batch_size: int,
    ) -> torch.Tensor:

        embeddings = []

        for sample_index in range(batch_size):

            #
            # Recover one IMU sequence
            #

            mask = (
                frame.imu_sample_indices
                == sample_index
            )

            timestamps = frame.imu_timestamps[mask].unsqueeze(1)

            gyro = frame.imu_angular_velocity[mask]

            accel = frame.imu_linear_acceleration[mask]

            #
            # (L,7)
            #

            sequence = torch.cat(

                (
                    timestamps,
                    gyro,
                    accel,
                ),

                dim=1,
            )

            #
            # Handle empty sequences
            #

            if sequence.shape[0] == 0:

                embeddings.append(

                    torch.zeros(
                        self.output_projection.out_features,
                        device=sequence.device,
                    )
                )

                continue

            #
            # Project features
            #
            # (L,C)
            #

            sequence = self.input_projection(
                sequence
            )

            #
            # Conv1D expects
            # (N,C,L)
            #

            sequence = sequence.transpose(
                0,
                1,
            ).unsqueeze(0)

            #
            # Temporal CNN
            #

            sequence = self.temporal_encoder(
                sequence
            )

            #
            # Global Average Pool
            #

            sequence = sequence.mean(
                dim=-1
            )

            #
            # (1,C)
            #

            embedding = self.output_projection(
                sequence.squeeze(0)
            )

            embeddings.append(
                embedding
            )

        return torch.stack(
            embeddings,
            dim=0,
        )