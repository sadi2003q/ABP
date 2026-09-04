"""
World transition module.

Predicts the next latent world state from:

    previous latent state
    +
    camera motion embedding

The model learns:

    z_hat(t) = f(z(t-1), u(t))

Input:
    previous_state:
        (B,C,H,W)

    motion_embedding:
        (B,D)

Output:
    predicted_state:
        (B,C,H,W)
"""


from __future__ import annotations


import torch
import torch.nn as nn



class WorldTransition(nn.Module):
    """
    Latent world dynamics model.

    Version 1:
        Motion conditioned residual CNN.

    Future:
        - Transformer dynamics
        - Neural ODE
        - State space models
    """


    def __init__(
        self,
        state_channels: int = 256,
        motion_dim: int = 128,
        hidden_channels: int | None = None,
    ):
        super().__init__()


        if hidden_channels is None:
            hidden_channels = state_channels


        #
        # Project IMU motion into latent space
        #

        self.motion_projection = nn.Linear(
            motion_dim,
            state_channels,
        )


        #
        # Residual transition network
        #

        self.transition = nn.Sequential(

            nn.Conv2d(
                state_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=32,
                num_channels=hidden_channels,
            ),

            nn.SiLU(inplace=True),


            nn.Conv2d(
                hidden_channels,
                state_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=32,
                num_channels=state_channels,
            ),
        )



        #
        # Residual scaling
        #
        # Helps stable training
        #

        self.activation = nn.SiLU()



    def forward(
        self,
        previous_state: torch.Tensor,
        motion_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        previous_state:

            (B,C,H,W)


        motion_embedding:

            (B,D)


        Returns
        -------

        predicted_state:

            (B,C,H,W)

        """


        if previous_state.ndim != 4:
            raise ValueError(
                "previous_state must be "
                "(B,C,H,W)"
            )


        if motion_embedding.ndim != 2:
            raise ValueError(
                "motion_embedding must be "
                "(B,D)"
            )


        B,C,H,W = previous_state.shape


        #
        # Motion conditioning
        #

        motion = self.motion_projection(
            motion_embedding
        )


        motion = motion.unsqueeze(
            -1
        ).unsqueeze(
            -1
        )


        motion = motion.expand(
            -1,
            -1,
            H,
            W,
        )


        #
        # Condition state
        #

        conditioned = (
            previous_state
            +
            motion
        )


        #
        # Predict residual change
        #

        delta = self.transition(
            conditioned
        )


        #
        # Residual dynamics
        #

        predicted_state = (
            previous_state
            +
            delta
        )


        predicted_state = self.activation(
            predicted_state
        )


        return predicted_state