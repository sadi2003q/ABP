"""
Dynamic object segmentation head.

Converts latent world residuals into
dynamic probability maps.

Input:
    residual:
        (B,C,H,W)

Output:
    dynamic probability:
        (B,1,H,W)

No direct supervision is assumed.
The head learns from self-supervised
world prediction residuals.
"""


from __future__ import annotations


import torch
import torch.nn as nn



class DynamicHead(nn.Module):
    """
    Predict dynamic regions from latent residuals.
    """

    def __init__(
        self,
        input_channels: int = 256,
        hidden_channels: int = 128,
    ):
        super().__init__()


        self.network = nn.Sequential(

            nn.Conv2d(
                input_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                32,
                hidden_channels,
            ),

            nn.SiLU(inplace=True),



            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                32,
                hidden_channels // 2,
            ),

            nn.SiLU(inplace=True),



            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
            ),

        )


    def forward(
        self,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        residual:

            latent prediction error

            (B,C,H,W)


        Returns
        -------

        dynamic probability:

            (B,1,H,W)

        """


        logits = self.network(
            residual
        )


        probability = torch.sigmoid(
            logits
        )


        return probability