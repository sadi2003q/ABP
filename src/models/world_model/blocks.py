"""
Reusable neural network building blocks.

These blocks are shared across the entire world model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _group_norm(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """
    Creates a valid GroupNorm layer.

    The number of groups is automatically reduced if the number
    of channels is too small or not divisible by the requested
    number of groups.
    """
    groups = min(num_groups, num_channels)

    while num_channels % groups != 0:
        groups -= 1

    return nn.GroupNorm(groups, num_channels)


class ConvBlock1D(nn.Module):
    """
    1D Convolution -> GroupNorm -> SiLU

    Input
    -----
    (B, C, L)

    Output
    ------
    (B, C_out, L_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(

            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),

            _group_norm(out_channels),

            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)

class ConvBlock(nn.Module):
    """
    Conv -> GroupNorm -> SiLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Standard ResNet residual block.

    Supports:
        - Channel change
        - Spatial downsampling
        - Identity shortcut
        - Projection shortcut
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            _group_norm(out_channels),
        )

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _group_norm(out_channels),
            )

        else:

            self.shortcut = nn.Identity()

        self.activation = nn.SiLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + identity

        return self.activation(out)




class UpsampleBlock(nn.Module):
    """
    Nearest-neighbor upsampling followed by convolution.

        H x W
            ↓
        2H x 2W
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="nearest",
        )

        self.conv = ConvBlock(
            in_channels,
            out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        return x