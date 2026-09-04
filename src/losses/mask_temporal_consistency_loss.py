"""
Temporal consistency loss for dynamic masks.

Encourages predicted dynamic masks to evolve smoothly over time.

Instead of penalizing the masks themselves, this loss penalizes the
temporal acceleration of the mask sequence.

For a temporal sequence

    M0 M1 M2 ... MT

the first derivative is

    ΔM_i = M_i - M_{i-1}

and the second derivative is

    Δ²M_i = ΔM_i - ΔM_{i-1}

The loss minimizes

    mean(|Δ²M|)

which discourages mask flickering while allowing objects to move
naturally.

Input
-----

masks

    Shape

        (B,T,1,H,W)

Returns
-------

dict

    loss
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskTemporalConsistencyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        if masks.ndim != 5:
            raise ValueError(
                "Expected mask sequence "
                "(B,T,1,H,W)"
            )

        if masks.size(1) < 3:
            raise ValueError(
                "Mask temporal consistency requires "
                "at least three temporal frames."
            )

        #
        # First temporal derivative
        #
        # (B,T-1,1,H,W)
        #

        first_derivative = torch.diff(
            masks,
            dim=1,
        )

        #
        # Second temporal derivative
        #
        # (B,T-2,1,H,W)
        #

        second_derivative = torch.diff(
            first_derivative,
            dim=1,
        )

        loss = second_derivative.abs().mean()

        return {

            "loss": loss,

        }