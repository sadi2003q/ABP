"""
Residual latent alignment module.

The latent renderer performs geometric warping using the predicted
depth and camera pose. Due to depth errors, pose errors, interpolation,
and occlusions, the rendered features are only approximately aligned.

This module learns a small residual correction before temporal fusion.

Input
-----
(B,T,C,H,W)

Output
------
(B,T,C,H,W)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Alignment(nn.Module):

    def __init__(
        self,
        channels: int = 256,
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(
                channels,
            ),

            nn.ReLU(
                inplace=True,
            ),

            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(
                channels,
            ),
        )

        self.activation = nn.ReLU(
            inplace=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Shape:
                (B,T,C,H,W)

        Returns
        -------
        Tensor
            Shape:
                (B,T,C,H,W)
        """

        #
        # Conv3D expects
        #
        # (B,C,T,H,W)
        #

        residual = x.permute(
            0,
            2,
            1,
            3,
            4,
        )

        correction = self.block(
            residual
        )

        aligned = residual + correction

        aligned = self.activation(
            aligned
        )

        #
        # Back to
        # (B,T,C,H,W)
        #

        aligned = aligned.permute(
            0,
            2,
            1,
            3,
            4,
        )

        return aligned