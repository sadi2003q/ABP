"""
Event voxelization.

Converts one packet of normalized events into a voxel grid using
bilinear temporal interpolation.

Input
-----
events_xy
    (N,2) int64

events_t
    (N,) float32

    Normalized to [0,1].

events_p
    (N,) float32

    0 = negative
    1 = positive

Output
------
(num_bins, H, W)

Reference
---------
Zhu et al.
"Unsupervised Event-based Learning of Optical Flow, Depth,
and Egomotion"
CVPR 2019
"""

from __future__ import annotations

import torch


class Voxelizer:
    """
    Bilinear event voxelizer.

    The voxelizer is completely stateless after construction and may
    be reused for every sample.
    """

    def __init__(
        self,
        height: int,
        width: int,
        num_bins: int,
    ):

        self.height = height
        self.width = width
        self.num_bins = num_bins

    def __call__(
        self,
        events_xy: torch.Tensor,
        events_t: torch.Tensor,
        events_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        events_xy
            (N,2)

        events_t
            (N,)

            Normalized timestamps in [0,1].

        events_p
            (N,)

            0/1 polarity.

        Returns
        -------
        voxel_grid

            Shape
            -----
            (num_bins,H,W)
        """

        device = events_xy.device

        voxel = torch.zeros(
            (
                self.num_bins,
                self.height,
                self.width,
            ),
            dtype=torch.float32,
            device=device,
        )

        #
        # Empty packet
        #
        if events_t.numel() == 0:
            return voxel

        #
        # Coordinates
        #
        x = events_xy[:, 0].long()
        y = events_xy[:, 1].long()
        assert (
            (x >= 0).all()
            and (x < self.width).all()
        )

        assert (
            (y >= 0).all()
            and (y < self.height).all()
        )
        #
        # Convert polarity
        #
        polarity = events_p * 2.0 - 1.0

        #
        # Scale timestamps
        #
        t_scaled = events_t * (self.num_bins - 1)

        t0 = torch.floor(t_scaled).long()
        t1 = t0 + 1

        w1 = t_scaled - t0.float()
        w0 = 1.0 - w1

        #
        # Lower bin
        #
        voxel.index_put_(
            (
                t0,
                y,
                x,
            ),
            polarity * w0,
            accumulate=True,
        )

        #
        # Upper bin
        #
        valid = t1 < self.num_bins

        if valid.any():

            voxel.index_put_(
                (
                    t1[valid],
                    y[valid],
                    x[valid],
                ),
                polarity[valid] * w1[valid],
                accumulate=True,
            )

        return voxel