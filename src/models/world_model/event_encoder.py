"""
Spatial encoder for event voxel grids.

Input
-----
(B, T, C, H, W)

Output
------
[
    l1 : (B,T,C1,H/2,W/2),
    l2 : (B,T,C2,H/4,W/4),
    l3 : (B,T,C3,H/8,W/8),
    l4 : (B,T,C4,H/16,W/16),
]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock, ResidualBlock


class EventEncoder(nn.Module):
    """
    ResNet-18 style encoder for event voxel grids.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        stage_channels: tuple[int, int, int, int] = (
            32,
            64,
            128,
            256,
        ),
    ):
        super().__init__()

        c1, c2, c3, c4 = stage_channels

        ####################################################################
        # Stem
        ####################################################################

        self.stem = nn.Sequential(
            ConvBlock(
                input_channels,
                c1,
                kernel_size=3,
            ),
            ConvBlock(
                c1,
                c1,
                kernel_size=3,
            ),
        )

        ####################################################################
        # Stages
        ####################################################################

        self.layer1 = nn.Sequential(
            ResidualBlock(
                c1,
                c1,
                stride=2,
            ),
            ResidualBlock(
                c1,
                c1,
            ),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(
                c1,
                c2,
                stride=2,
            ),
            ResidualBlock(
                c2,
                c2,
            ),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(
                c2,
                c3,
                stride=2,
            ),
            ResidualBlock(
                c3,
                c3,
            ),
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(
                c3,
                c4,
                stride=2,
            ),
            ResidualBlock(
                c4,
                c4,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:

        if x.ndim != 5:
            raise ValueError(
                f"Expected (B,T,C,H,W), got {tuple(x.shape)}"
            )

        batch_size, seq_len, channels, height, width = x.shape

        ####################################################################
        # Merge batch and time
        ####################################################################

        x = x.reshape(
            batch_size * seq_len,
            channels,
            height,
            width,
        )

        ####################################################################
        # Encoder
        ####################################################################

        x = self.stem(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        ####################################################################
        # Restore temporal dimension
        ####################################################################

        features = []

        for feature in (l1, l2, l3, l4):

            _, c, h, w = feature.shape

            feature = feature.reshape(
                batch_size,
                seq_len,
                c,
                h,
                w,
            )

            features.append(feature)

        return features