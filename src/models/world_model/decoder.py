"""
World decoder with U-Net skip connections.

Progressively upsamples the latent world representation to image
resolution, fusing high-resolution encoder features at each stage
via U-Net skip connections.

Input
-----
world_feature : (B, 256, H/16, W/16)

skips : list of 3 tensors (l3_skip, l2_skip, l1_skip) OR None
    - l3_skip : (B, 128, H/8,  W/8)   from event_encoder pyramid level 3
    - l2_skip : (B, 64,  H/4,  W/4)   from event_encoder pyramid level 2
    - l1_skip : (B, 32,  H/2,  W/2)   from event_encoder pyramid level 1

    If skips is None, zeros are used in place of the skip features
    (backward-compat behavior; produces no U-Net benefit but does
    not crash).

Output
------
(B, output_channels, H, W)

Why U-Net skips?
----------------
Without skips, the decoder receives only the H/16 world_feature
and has no access to fine spatial detail. The mask it produces is
effectively a smoothed-up version of the residual pseudo-label --
object boundaries are blurry.

With U-Net skips, the decoder can fuse:
  - coarse motion signal (from world_feature at H/16)
  - fine spatial detail (from encoder features at H/8, H/4, H/2)

producing sharp masks at object boundaries.

Channel bookkeeping
-------------------
Stage 1: up(x, 256->128) at H/8  + concat l3 (128)  -> conv (256->128)
Stage 2: up(x, 128->64)  at H/4  + concat l2 (64)   -> conv (128->64)
Stage 3: up(x, 64->32)   at H/2  + concat l1 (32)   -> conv (64->32)
Stage 4: up(x, 32->16)   at H    + no skip           -> conv (32->16)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """
    Single U-Net decoder block.

    1. Upsample input by 2x (bilinear)
    2. Concat skip (from encoder at matching resolution)
    3. Conv 3x3 -> GroupNorm -> GELU
    4. Conv 3x3 -> GroupNorm -> GELU

    If skip is None at forward time, zeros are substituted (so the
    block can run in a non-U-Net configuration for backward compat).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        groups1 = min(32, out_channels) if out_channels > 0 else 1
        while groups1 > 1 and out_channels % groups1 != 0:
            groups1 -= 1
        self.norm1 = nn.GroupNorm(groups1, out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        groups2 = min(32, out_channels) if out_channels > 0 else 1
        while groups2 > 1 and out_channels % groups2 != 0:
            groups2 -= 1
        self.norm2 = nn.GroupNorm(groups2, out_channels)
        self.act2 = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.up(x)

        if self.skip_channels > 0:
            if skip is None:
                skip = torch.zeros(
                    x.shape[0],
                    self.skip_channels,
                    x.shape[2],
                    x.shape[3],
                    device=x.device,
                    dtype=x.dtype,
                )
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(
                    skip,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        return x


class WorldDecoder(nn.Module):
    """
    U-Net decoder with 4 upsampling stages and 3 skip connections.
    """

    def __init__(
        self,
        input_channels: int = 256,
        output_channels: int = 16,
        skip_channels: tuple[int, int, int] = (128, 64, 32),
    ):
        super().__init__()

        s1, s2, s3 = skip_channels

        self.decoder = nn.ModuleList([

            DecoderBlock(
                in_channels=input_channels,
                skip_channels=s1,
                out_channels=128,
            ),

            DecoderBlock(
                in_channels=128,
                skip_channels=s2,
                out_channels=64,
            ),

            DecoderBlock(
                in_channels=64,
                skip_channels=s3,
                out_channels=32,
            ),

            DecoderBlock(
                in_channels=32,
                skip_channels=0,
                out_channels=output_channels,
            ),
        ])

    def forward(
        self,
        x: torch.Tensor,
        skips: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if skips is None:
            skips = [None, None, None]
        while len(skips) < 3:
            skips.append(None)

        for i, block in enumerate(self.decoder):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)

        return x
