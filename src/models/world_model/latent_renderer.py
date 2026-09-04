"""
Differentiable latent feature renderer.

Warp latent feature maps using predicted depth and camera motion.

Inputs
------
feature:
    (B,C,H,W)

depth:
    (B,1,H,W)

pose:
    (B,6)

    [tx, ty, tz, rx, ry, rz]

K:
    (B,3,3)

distortion:
    (B,4)

    [k1, k2, p1, p2]

Output
------
(B,C,H,W)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentRenderer(nn.Module):

    def __init__(self, rotation_type: str = "6d"):
        """
        Parameters
        ----------
        rotation_type : str
            'axis_angle' (6-DoF pose: [tx,ty,tz,rx,ry,rz])
            '6d' (9-DoF pose: [tx,ty,tz, a1x,a1y,a1z, a2x,a2y,a2z])
            '6d' uses Zhou et al. CVPR'19 6D rotation rep which
            is continuous everywhere (no singularity at theta=0).
        """
        super().__init__()
        self.rotation_type = rotation_type

    # ==========================================================
    # Pose
    # ==========================================================

    def pose_to_matrix(
        self,
        pose: torch.Tensor,
    ) -> torch.Tensor:

        B = pose.shape[0]

        t = pose[:, :3]

        if self.rotation_type == "6d":
            # 6D rotation representation (Zhou et al. CVPR'19)
            # pose = [tx, ty, tz, a1x, a1y, a1z, a2x, a2y, a2z]
            a1 = pose[:, 3:6]   # (B, 3)
            a2 = pose[:, 6:9]   # (B, 3)

            # Gram-Schmidt orthonormalization
            b1 = F.normalize(a1, p=2, dim=1, eps=1e-6)
            b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
            b2 = F.normalize(b2, p=2, dim=1, eps=1e-6)
            b3 = torch.cross(b1, b2, dim=1)

            R = torch.stack([b1, b2, b3], dim=2)  # (B, 3, 3)
        else:
            # Axis-angle (legacy, has singularity at theta=0)
            r = pose[:, 3:]
            theta = torch.norm(r, dim=1, keepdim=True)
            axis = r / (theta + 1e-8)
            x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
            cos = torch.cos(theta)
            sin = torch.sin(theta)

            R = torch.zeros(B, 3, 3, device=pose.device, dtype=pose.dtype)
            R[:, 0, 0] = cos[:, 0] + x * x * (1 - cos[:, 0])
            R[:, 0, 1] = x * y * (1 - cos[:, 0]) - z * sin[:, 0]
            R[:, 0, 2] = x * z * (1 - cos[:, 0]) + y * sin[:, 0]
            R[:, 1, 0] = y * x * (1 - cos[:, 0]) + z * sin[:, 0]
            R[:, 1, 1] = cos[:, 0] + y * y * (1 - cos[:, 0])
            R[:, 1, 2] = y * z * (1 - cos[:, 0]) - x * sin[:, 0]
            R[:, 2, 0] = z * x * (1 - cos[:, 0]) - y * sin[:, 0]
            R[:, 2, 1] = z * y * (1 - cos[:, 0]) + x * sin[:, 0]
            R[:, 2, 2] = cos[:, 0] + z * z * (1 - cos[:, 0])

        T = torch.eye(4, device=pose.device, dtype=pose.dtype).repeat(B, 1, 1)

        T[:, :3, :3] = R
        T[:, :3, 3] = t

        return T

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(
        self,
        feature: torch.Tensor,
        depth: torch.Tensor,
        pose: torch.Tensor,
        K: torch.Tensor,
        distortion: torch.Tensor,
    ) -> torch.Tensor:

        B, C, H, W = feature.shape

        device = feature.device
        dtype = feature.dtype

        # ------------------------------------------------------
        # Pose
        # ------------------------------------------------------

        T = self.pose_to_matrix(
            pose
        )

        # ------------------------------------------------------
        # Pixel grid
        # ------------------------------------------------------

        y, x = torch.meshgrid(
            torch.arange(
                H,
                device=device,
                dtype=dtype,
            ),
            torch.arange(
                W,
                device=device,
                dtype=dtype,
            ),
            indexing="ij",
        )

        ones = torch.ones_like(x)

        pixels = torch.stack(
            [
                x,
                y,
                ones,
            ],
            dim=0,
        )

        pixels = pixels.reshape(
            3,
            -1,
        )

        pixels = pixels.unsqueeze(0).repeat(
            B,
            1,
            1,
        )

        # ------------------------------------------------------
        # Backproject
        # ------------------------------------------------------

        cam = torch.linalg.solve(
            K,
            pixels,
        )

        cam = cam * depth.reshape(
            B,
            1,
            -1,
        )

        # ------------------------------------------------------
        # Transform
        # ------------------------------------------------------

        cam_h = torch.cat(
            [
                cam,
                torch.ones(
                    B,
                    1,
                    cam.shape[-1],
                    device=device,
                    dtype=dtype,
                ),
            ],
            dim=1,
        )

        warped = torch.bmm(
            T,
            cam_h,
        )

        warped = warped[:, :3]

        # ------------------------------------------------------
        # Normalized image coordinates
        # ------------------------------------------------------

        X = warped[:, 0]
        Y = warped[:, 1]
        Z = torch.clamp(
            warped[:, 2],
            min=1e-4,
        )

        x0 = X / Z
        y0 = Y / Z

        # ------------------------------------------------------
        # EVIMO2 distortion
        # ------------------------------------------------------

        k1 = distortion[:, 0].view(B, 1)
        k2 = distortion[:, 1].view(B, 1)
        p1 = distortion[:, 2].view(B, 1)
        p2 = distortion[:, 3].view(B, 1)

        r2 = x0.square() + y0.square()
        r4 = r2.square()

        radial = (
            1.0
            + k1 * r2
            + k2 * r4
        )

        x = (
            x0 * radial
            + 2.0 * p1 * x0 * y0
            + p2 * (r2 + 2.0 * x0.square())
        )

        y = (
            y0 * radial
            + p1 * (r2 + 2.0 * y0.square())
            + 2.0 * p2 * x0 * y0
        )


        # ------------------------------------------------------
        # Intrinsics
        # ------------------------------------------------------

        fx = K[:, 0, 0].view(B, 1)
        fy = K[:, 1, 1].view(B, 1)

        cx = K[:, 0, 2].view(B, 1)
        cy = K[:, 1, 2].view(B, 1)

        u = fx * x + cx
        v = fy * y + cy

        # ------------------------------------------------------
        # grid_sample coordinates
        # ------------------------------------------------------

        u = 2.0 * u / (W - 1) - 1.0
        v = 2.0 * v / (H - 1) - 1.0

        grid = torch.stack(
            [
                u,
                v,
            ],
            dim=-1,
        )

        grid = grid.reshape(
            B,
            H,
            W,
            2,
        )

        # ------------------------------------------------------
        # Warp
        # ------------------------------------------------------

        warped_feature = F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return warped_feature