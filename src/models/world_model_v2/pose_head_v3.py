"""
Pose Head v3 — architectural IMU rotation injection.

Diagnosis that motivated this (see evaluate_with_gt_depth.py results
on scene15_dyn_test_06_000000, run 2026-09-08):

    predicted depth, predicted pose   IoU 0.0331
    GT depth,        predicted pose   IoU 0.0300   (no gain -> depth is
                                                      NOT the bottleneck)
    predicted depth, GT pose          IoU 0.0000   (catastrophic)
    GT depth,        GT pose          IoU 0.0000   (catastrophic)

Injecting real GT pose while depth is still in the model's own
unconstrained scale makes things *worse*, not better. This shows
depth and pose in v2 are jointly self-consistent (satisfy the
photometric loss together) but NOT anchored to real geometry —
the classic monocular depth/pose scale-and-shape ambiguity: photometric
reconstruction error is invariant to a whole family of (wrong) joint
solutions, and nothing in v2 breaks that symmetry for rotation.

v3 fix (rotation only — translation/depth scale is a separate,
follow-up problem, see imu_integration.py's module docstring and the
project's diagnostic plan):

    R_imu      = exact integration of raw gyro samples (imu_integration.py)
    R_residual = small learned correction, network output
    R_final    = R_imu @ R_residual

Because R_imu comes from a real physical sensor and is NOT a function
of the photometric loss, the network can no longer minimize
photometric error by inventing an arbitrary rotation — it can only
apply a *correction* on top of a real measurement. This directly
targets the diagnosed failure mode.

Translation is left as a free network output for now (same as v2);
metric scale for translation/depth is a known separate gap (the GT
pose *scaled* variants in the same diagnostic still underperformed
prediction, confirming scale, not just rotation, remains open) and
is intentionally deferred rather than bolted on with an unvalidated
accelerometer double-integration, which is a much higher-risk
addition (drift compounds quadratically) — see project discussion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.world_model_v2.imu_integration import rotation_matrix_to_6d


class PoseHeadV3(nn.Module):
    """
    Predict pose from fused event features + IMU embedding, with the
    rotation component anchored to an integrated-gyro prior.

    Parameters
    ----------
    event_channels : int
        Channel dimension of event features (256).
    imu_embedding_dim : int
        Dimension of the learned IMU embedding (128) — still used as
        auxiliary context (e.g. helps translation, and lets the
        residual rotation correction see recent acceleration/gyro
        shape, not just the integrated summary).
    hidden_dim : int
        Hidden layer dimension.
    residual_init_scale : float
        Initializes the final layer so the network starts by
        predicting a near-identity residual rotation and near-zero
        translation, i.e. R_final ~= R_imu at init.
    """

    def __init__(
        self,
        event_channels: int = 256,
        imu_embedding_dim: int = 128,
        hidden_dim: int = 256,
        residual_init_scale: float = 0.01,
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

            # 3 translation + 6 residual-rotation (6D rep)
            nn.Linear(hidden_dim, 9),
        )

        #
        # Init: near-zero translation, near-identity residual rotation.
        # Identity in 6D rep is a1=(1,0,0), a2=(0,1,0).
        #
        final_linear = self.network[-1]
        nn.init.zeros_(final_linear.bias)
        nn.init.normal_(
            final_linear.weight,
            mean=0.0,
            std=residual_init_scale,
        )
        with torch.no_grad():
            final_linear.bias[3:6] = torch.tensor([1.0, 0.0, 0.0])
            final_linear.bias[6:9] = torch.tensor([0.0, 1.0, 0.0])

    @staticmethod
    def _residual_6d_to_matrix(pose_6d: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt 6D rep -> rotation matrix (same convention as
        LatentRenderer.pose_to_matrix / invert_pose_6dof elsewhere in
        the codebase, kept identical on purpose so downstream warping
        code needs zero changes)."""
        import torch.nn.functional as F

        a1 = pose_6d[..., 0:3]
        a2 = pose_6d[..., 3:6]

        b1 = F.normalize(a1, p=2, dim=-1, eps=1e-6)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, p=2, dim=-1, eps=1e-6)
        b3 = torch.cross(b1, b2, dim=-1)

        R = torch.stack([b1, b2, b3], dim=-1)  # columns = b1,b2,b3 -> (...,3,3)
        return R

    def forward(
        self,
        event_features: torch.Tensor,
        imu_embeddings: torch.Tensor,
        R_imu: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        event_features : (B, T, C, H, W)
        imu_embeddings : (B, T, D)
        R_imu : (B, T, 3, 3)
            Integrated-gyro rotation prior, per frame (identity where
            unavailable — see imu_integration.integrate_frame_rotation).

        Returns
        -------
        poses : (B, T, 9)
            [tx, ty, tz, a1(3), a2(3)] — same 9-DoF convention as v2,
            so LatentRenderer / invert_pose_6dof need no changes.
            The rotation encoded by (a1, a2) here is R_imu @ R_residual,
            not a free network output.
        """
        B, T, C, H, W = event_features.shape

        event_pooled = event_features.mean(dim=[-2, -1])  # (B,T,C)
        x = torch.cat([event_pooled, imu_embeddings], dim=-1)
        x = x.reshape(B * T, -1)

        raw = self.network(x)  # (B*T, 9)
        raw = raw.reshape(B, T, 9)

        translation = raw[..., 0:3]
        residual_6d = raw[..., 3:9]

        R_residual = self._residual_6d_to_matrix(residual_6d)  # (B,T,3,3)

        R_final = torch.matmul(R_imu, R_residual)  # (B,T,3,3)

        rotation_6d = rotation_matrix_to_6d(R_final)  # (B,T,6)

        poses = torch.cat([translation, rotation_6d], dim=-1)  # (B,T,9)

        return poses
