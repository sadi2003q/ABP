"""
Mask Refinement Head (v2) — simplified.

Takes the photometric residual and refines it into a mask using
a small number of conv layers + encoder skip connections.

The residual already tells us WHERE motion is. The refinement
just sharpens boundaries and removes noise using the encoder's
spatial features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskRefinementHead(nn.Module):
    """Small U-Net that refines the residual into a sharp mask.

    Architecture:
        residual (B,1,H,W) → conv → down → concat(l1) → conv → down → concat(l2) → conv
        → up → concat → conv → up → concat → conv → 1x1 → logits

    Parameters
    ----------
    skip_channels : (l3_ch, l2_ch, l1_ch) = (128, 64, 32)
    hidden : internal channels
    """

    def __init__(
        self,
        residual_channels: int = 1,
        skip_channels: tuple = (128, 64, 32),
        hidden: int = 32,
        out_channels: int = 1,
    ):
        super().__init__()
        s3, s2, s1 = skip_channels

        # Encoder
        self.enc1 = self._dbl(residual_channels, hidden)       # H → H
        self.enc2 = self._dbl(hidden + s1, hidden)              # H/2 → H/2
        self.enc3 = self._dbl(hidden + s2, hidden * 2)          # H/4 → H/4

        # Decoder
        self.dec2 = self._dbl(hidden * 2 + hidden, hidden)      # H/2
        self.dec1 = self._dbl(hidden + hidden, hidden)          # H

        self.final = nn.Conv2d(hidden, out_channels, 1)

    @staticmethod
    def _dbl(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.GELU(),
        )

    def forward(self, residual, skips):
        """
        Parameters
        ----------
        residual : (B, 1, H, W)
        skips : [skip_l3, skip_l2, skip_l1] at H/8, H/4, H/2

        Returns
        -------
        (B, 1, H, W) — mask logits
        """
        skip_l3, skip_l2, skip_l1 = skips

        # Encoder
        x1 = self.enc1(residual)           # (B, h, H, W)
        x2 = self.enc2(torch.cat([F.avg_pool2d(x1, 2), skip_l1], dim=1))  # (B, h, H/2, W/2)
        x3 = self.enc3(torch.cat([F.avg_pool2d(x2, 2), skip_l2], dim=1))  # (B, 2h, H/4, W/4)

        # Decoder
        d2 = self.dec2(torch.cat([F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False), x2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), x1], dim=1))

        return self.final(d1)
