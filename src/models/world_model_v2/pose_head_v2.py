"""
Pose Head v2 — Fuses IMU embeddings WITH event features.

CRITICAL IMPROVEMENT over v1:
The v1 pose head only took IMU data as input. This meant:
  - Pose couldn't "see" the scene (no visual information)
  - Photometric loss gradient reached pose only through the IMU
    encoder (very indirect path)
  - Depth and pose were completely disconnected

In SfMLearner/Monodepth2, the pose network takes BOTH visual
features AND the motion source. This is essential because:
  1. Visual features contain ego-motion information (optical flow
     patterns, scene structure)
  2. The photometric loss gradient flows DIRECTLY through the
     event encoder to the pose head
  3. Depth and pose share the encoder, creating a coupled
     optimization where both must agree on the scene geometry

Architecture:
  event_features (B,T,C,H,W) → global avg pool → (B,T,C)
  imu_embedding (B,T,D) ──────────────────────────┘
                    ↓ concat
              (B, T, C+D)
                    ↓ MLP
              (B, T, 6)  ← pose [tx,ty,tz,rx,ry,rz]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PoseHeadV2(nn.Module):
    """Predict pose from FUSED event features + IMU embeddings.

    Parameters
    ----------
    event_channels : int
        Channel dimension of event features (256).
    imu_embedding_dim : int
        Dimension of IMU embedding (128).
    hidden_dim : int
        Hidden layer dimension.
    """

    def __init__(
        self,
        event_channels: int = 256,
        imu_embedding_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()

        input_dim = event_channels + imu_embedding_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, 9),  # 3 translation + 6 rotation (6D rep)
        )

        # Non-zero pose init (break depth-pose symmetry)
        final_linear = self.network[-1]
        nn.init.zeros_(final_linear.bias)
        with torch.no_grad():
            final_linear.bias[3:6] = torch.tensor(
                [1.0, 0.0, 0.0], device=final_linear.bias.device
            )
            final_linear.bias[6:9] = torch.tensor(
                [0.0, 1.0, 0.0], device=final_linear.bias.device
            )

    def forward(
        self,
        event_features: torch.Tensor,
        imu_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        event_features : (B, T, C, H, W)
            Spatial features from the event encoder.
        imu_embeddings : (B, T, D)
            IMU motion embeddings.

        Returns
        -------
        poses : (B, T, 6)
            Per-frame 6-DoF pose [tx, ty, tz, rx, ry, rz].
        """
        B, T, C, H, W = event_features.shape

        # Global average pool event features → (B, T, C)
        event_pooled = event_features.mean(dim=[-2, -1])

        # Concatenate with IMU embeddings
        x = torch.cat([event_pooled, imu_embeddings], dim=-1)  # (B, T, C+D)

        # Reshape for MLP
        x = x.reshape(B * T, -1)

        poses = self.network(x)
        poses = poses.reshape(B, T, 9)  # 3 translation + 6 rotation (6D rep)

        return poses
