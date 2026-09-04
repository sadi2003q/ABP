"""
Temporal consistency loss for depth prediction.

The model predicts a depth map for every temporal feature.

Unlike spatial smoothness, this loss regularizes the evolution of depth
predictions across time.

Instead of penalizing

    D_t - D_{t-1}

(which incorrectly assumes depth should remain nearly constant),

we penalize the temporal second derivative

    D_t - 2 D_{t-1} + D_{t-2}

which discourages abrupt temporal oscillations while allowing smooth
depth changes caused by camera motion and object motion.

Input
-----
depths

    Predicted depth sequence

        (B,T,1,H,W)

Output
------
loss

    Scalar temporal consistency loss.


Example
-------

depth_losses = depth_temporal_loss(
    outputs["depths"]
)

loss = (
    ...
    + depth_losses["loss"]
)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthTemporalConsistencyLoss(nn.Module):
    """
    Penalize the second temporal derivative of depth.
    """

    def __init__(
        self,
        beta: float = 1.0,
    ):
        super().__init__()

        self.beta = beta

    def forward(
        self,
        depths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        depths

            Predicted depth sequence

            Shape

                (B,T,1,H,W)

        Returns
        -------
        dict
        """

        if depths.ndim != 5:
            raise ValueError(
                "Expected depth tensor "
                "(B,T,1,H,W)"
            )

        if depths.shape[1] < 3:
            raise ValueError(
                "Temporal consistency requires "
                "at least three frames."
            )

        #
        # Compute temporal second derivative
        #
        # For every valid triplet:
        #
        #   D_t - 2 D_{t-1} + D_{t-2}
        #

        second_difference = (

            depths[:, 2:, ...]

            - 2.0 * depths[:, 1:-1, ...]

            + depths[:, :-2, ...]

        )

        loss = F.smooth_l1_loss(

            second_difference,

            torch.zeros_like(second_difference),

            beta=self.beta,

        )

        return {

            "loss": loss,

        }