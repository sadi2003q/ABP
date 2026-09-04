"""
Pose prediction head for self-supervised world model.

Predicts relative camera motion from IMU features.

Input:
    IMU embedding:
        (B, T, motion_dim)

Output:
    6DoF motion vector:
        (B,6)

Representation:

    [tx, ty, tz, rx, ry, rz]

where:
    translation = first 3 values
    rotation vector = last 3 values

No ground truth pose is used.
The pose is learned through geometric consistency losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn



class PoseHead(nn.Module):
    """
    Predict relative camera motion from IMU embedding.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.LayerNorm(
                hidden_dim,
            ),

            nn.SiLU(),


            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.LayerNorm(
                hidden_dim,
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                6,
            ),

        )

        # --------------------------------------------------
        # Initialize the final linear's bias with random noise
        # (std=0.1) so pose starts NON-ZERO.
        #
        # This is CRITICAL: with zero pose (identity transform),
        # the projection u = fx * (ray_x * depth) / (ray_z * depth)
        # = fx * ray_x / ray_z — the depth CANCELS OUT.
        # So depth has ZERO gradient when pose is identity.
        # By initializing pose with random values, we break
        # this symmetry and allow depth to start learning.
        #
        # std=0.1 (not 0.01) because the gradient w.r.t. depth is
        # proportional to pose — larger initial pose = larger
        # initial depth gradient = faster convergence.
        # --------------------------------------------------
        final_linear = self.network[-1]
        nn.init.normal_(final_linear.bias, mean=0.0, std=0.1)


    def forward(
        self,
        motion_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        motion_embeddings

            Shape:
                (B,T,D)

        Returns
        -------
        poses

            Shape:
                (B,T,6)
        """

        if motion_embeddings.ndim != 3:
            raise ValueError(
                "Expected motion embeddings "
                "shape (B,T,D)"
            )

        B, T, D = motion_embeddings.shape

        x = motion_embeddings.reshape(
            B * T,
            D,
        )

        poses = self.network(
            x
        )

        poses = poses.reshape(
            B,
            T,
            6,
        )

        return poses