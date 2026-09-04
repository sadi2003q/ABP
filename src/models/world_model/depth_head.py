"""
Depth prediction head for self-supervised world model.

Predicts a dense depth representation from latent features.

The depth is not supervised using dataset ground truth.
It is only used for differentiable geometric warping.

Input:
    Temporal feature sequence

        (B,T,C,H,W)

Output:
    Dense depth sequence

        (B,T,1,H,W)

Each temporal feature is decoded independently using the
same convolutional decoder.

Only the last depth map is required by the renderer,
while the complete sequence is used for temporal
consistency losses.

Depth is constrained to be positive.
"""


from __future__ import annotations


import torch
import torch.nn as nn

import torch.nn.functional as F



class DepthHead(nn.Module):
    """
    Predict dense depth from latent feature representation.
    """


    def __init__(
        self,
        input_channels: int = 256,
        hidden_channels: int = 128,
    ):
        super().__init__()


        self.decoder = nn.Sequential(

            nn.Conv2d(
                input_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),

            nn.GroupNorm(
                8,
                hidden_channels,
            ),

            nn.SiLU(),



            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),

            nn.GroupNorm(
                8,
                hidden_channels,
            ),

            nn.SiLU(),



            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=3,
                padding=1,
            ),

        )


    def forward(
        self,
        features,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        feature:

            Latent feature map

            (B,C,H,W)


        Returns
        -------

        depth:

            Positive depth map

            (B,1,H,W)

        """


        if features.ndim != 5:

            raise ValueError(
                "Expected feature shape "
                "(B,T,C,H,W)"
            )

        B, T, C, H, W = features.shape

        features = features.reshape(

            B * T,

            C,

            H,

            W,

        )
        depth = self.decoder(
            features
        )

        # Use 1 + softplus(x) so depth starts at 1.0 (not 0.69).
        # This prevents zero/near-zero depths that cause numerical
        # issues in the renderer, and gives a better init for the
        # photometric loss to work with.
        depth = 1.0 + F.softplus(depth)


        depth = depth.reshape(

            B,

            T,

            1,

            H,

            W,

        )


        return depth