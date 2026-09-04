"""
Latent residual computation.

Computes the difference between:

    observed latent state
    predicted latent state

The residual represents motion that cannot be
explained by the learned static world model.

Input:
    predicted_state:
        (B,C,H,W)

    observed_state:
        (B,C,H,W)

Output:
    residual:
        (B,C,H,W)
"""


from __future__ import annotations

import torch
import torch.nn as nn



class LatentResidual(nn.Module):
    """
    Compute latent world prediction residual.

    Version 1:
        Absolute feature difference.

    Future:
        - cosine residual
        - temporal residual
        - multi-scale residual
    """


    def __init__(
        self,
    ):
        super().__init__()



    def forward(
        self,
        predicted_state: torch.Tensor,
        observed_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predicted_state:

            predicted latent state

            (B,C,H,W)


        observed_state:

            actual latent state

            (B,C,H,W)


        Returns
        -------

        residual:

            (B,C,H,W)

        """


        if predicted_state.shape != observed_state.shape:

            raise ValueError(
                "Predicted and observed states "
                "must have identical shapes."
            )


        residual = torch.abs(
            observed_state -
            predicted_state
        )


        return residual