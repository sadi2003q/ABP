"""
Latent consistency losses.

This module provides the primary self-supervised supervision for the
latent world model.

The world model predicts the latent representation of the current frame
from the previous world state using the estimated camera motion.

Three complementary objectives are optimized.

1. Prediction Consistency

        previous latent
                │
        World Transition
                │
                ▼
      predicted current latent
                │
                ▼
 compare against encoded current latent

2. Rendering Consistency

        predicted current latent
                │
      Differentiable Renderer
                │
                ▼
      rendered current latent
                │
                ▼
 compare against encoded current latent

3. Prediction-Rendering Agreement

      predicted current latent
                ↕
      rendered current latent

The first two losses provide the primary supervision while the agreement
loss acts as a lightweight regularizer.

Returns
-------
losses["loss"]
    Total latent consistency loss.

    
Parameters
----------
predicted_state

    Current latent state predicted by the
    WorldTransition module from the previous
    latent state and estimated camera motion.

    Shape:
        (B,C,H,W)

rendered_state

    Current latent state obtained after
    differentiably rendering the predicted
    latent using the predicted depth and pose.

    Shape:
        (B,C,H,W)


target_state

    Current latent representation produced
    by the EventEncoder → MotionFusion →
    TemporalEncoder pipeline.

    This latent serves as the self-supervised
    training target.

    Shape:
        (B,C,H,W)

Returns
-------
dict

    loss

    prediction_loss

    rendering_loss

    agreement_loss


Example usage:
outputs = model(
    voxel_batch,
    batch,
)

#
# Target latent from the Temporal Encoder.
#
target_state = outputs["temporal_features"][:, -1]

latent_losses = latent_consistency_loss(

    predicted_state=outputs["predicted_state"],

    rendered_state=outputs["rendered_state"],

    target_state=target_state,

)

loss = latent_losses["loss"]

"""



from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentConsistencyLoss(nn.Module):
    """
    Self-supervised latent consistency loss.

    Returns

        Prediction Loss
        Rendering Loss
        Agreement Loss
        Total Loss
    """

    def __init__(
        self,
        prediction_weight: float = 1.0,
        rendering_weight: float = 1.0,
        agreement_weight: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()

        self.prediction_weight = prediction_weight
        self.rendering_weight = rendering_weight
        self.agreement_weight = agreement_weight
        self.beta = beta

    def forward(
        self,
        predicted_state: torch.Tensor,
        rendered_state: torch.Tensor,
        target_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        predicted_state

            Output of WorldTransition

            Shape:
                (B,C,H,W)

        rendered_state

            Output of LatentRenderer

            Shape:
                (B,C,H,W)

        target_state

            Latent representation encoded from the
            current event frame.

            Shape:
                (B,C,H,W)

        Returns
        -------
        dict
        """

        if predicted_state.shape != target_state.shape:
            raise ValueError(
                "predicted_state and target_state "
                "must have identical shapes."
            )

        if rendered_state.shape != target_state.shape:
            raise ValueError(
                "rendered_state and target_state "
                "must have identical shapes."
            )

        #
        # World dynamics supervision
        #

        prediction_loss = F.smooth_l1_loss(
            predicted_state,
            target_state,
            beta=self.beta,
        )

        #
        # Geometry supervision
        #

        rendering_loss = F.smooth_l1_loss(
            rendered_state,
            target_state,
            beta=self.beta,
        )

        #
        # Small regularizer
        #

        agreement_loss = F.smooth_l1_loss(
            predicted_state,
            rendered_state,
            beta=self.beta,
        )

        total_loss = (

            self.prediction_weight
            * prediction_loss

            +

            self.rendering_weight
            * rendering_loss

            +

            self.agreement_weight
            * agreement_loss
        )

        return {

            "loss": total_loss,

            "prediction_loss": prediction_loss,

            "rendering_loss": rendering_loss,

            "agreement_loss": agreement_loss,

        }