from __future__ import annotations

import torch
import torch.nn as nn


class MotionFusion(nn.Module):
    """
    Fuse event features with IMU motion embedding.

    Event feature:
        (B,T,C,H,W)

    IMU embedding:
        (B,T,D)

    Output:
        (B,T,C,H,W)
    """

    def __init__(
        self,
        event_channels: int = 256,
        imu_dim: int = 128,
    ):
        super().__init__()

        #
        # Convert IMU embedding into event feature channels
        #

        self.imu_projection = nn.Linear(
            imu_dim,
            event_channels,
        )

        #
        # Feature refinement
        #

        self.refinement = nn.Sequential(

            nn.Conv2d(
                event_channels,
                event_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=32,
                num_channels=event_channels,
            ),

            nn.SiLU(inplace=True),
        )


    def forward(
        self,
        event_feature: torch.Tensor,
        imu_embedding: torch.Tensor,
    ) -> torch.Tensor:

        """
        Parameters
        ----------
        event_feature:
            (B,T,C,H,W)

        imu_embedding:
            (B,T,D)

        Returns
        -------
        fused:
            (B,T,C,H,W)
        """

        B,T,C,H,W = event_feature.shape


        #
        # Project IMU
        #

        motion = self.imu_projection(
            imu_embedding
        )

        #
        # (B,T,C)
        #

        motion = motion.unsqueeze(-1).unsqueeze(-1)

        #
        # (B,T,C,1,1)
        #

        motion = motion.expand(
            -1,
            -1,
            -1,
            H,
            W,
        )


        #
        # Fuse
        #

        fused = event_feature + motion


        #
        # Conv2D expects (B*T,C,H,W)
        #

        fused = fused.reshape(
            B*T,
            C,
            H,
            W,
        )

        fused = self.refinement(
            fused
        )

        fused = fused.reshape(
            B,
            T,
            C,
            H,
            W,
        )


        return fused