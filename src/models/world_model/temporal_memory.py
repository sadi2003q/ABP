"""
Temporal memory module with configurable architecture.

Supports three temporal aggregation backends:
  - 'transformer': self-attention (default, current)
  - 'convlstm':    ConvLSTM recurrent
  - 'convgru':     ConvGRU recurrent (lighter than ConvLSTM)

For T=4 (our sequence length), all three perform similarly.
The choice mainly affects:
  - Memory: transformer O(T²), ConvLSTM/GRU O(T)
  - Gradient flow: transformer direct, ConvLSTM/GRU via BPTT
  - Parameters: transformer ~530K, ConvLSTM ~400K, ConvGRU ~300K

NOTE: The temporal_memory is in the MASK prediction path, NOT the
depth/pose path. Changing it will NOT affect depth/pose convergence.
It may affect mask quality (how well temporal context is aggregated
for mask prediction).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ==========================================================
# ConvLSTM Cell
# ==========================================================

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.hidden_channels = hidden_channels

    def forward(self, x, hidden=None):
        B, C, H, W = x.shape
        if hidden is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = hidden

        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, (h_new, c_new)


# ==========================================================
# ConvGRU Cell
# ==========================================================

class ConvGRUCell(nn.Module):
    """Convolutional GRU cell (lighter than LSTM)."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        # Update gate
        self.conv_z = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        # Reset gate
        self.conv_r = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        # Candidate
        self.conv_h = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        self.hidden_channels = hidden_channels

    def forward(self, x, hidden=None):
        B, C, H, W = x.shape
        if hidden is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h = hidden

        cat = torch.cat([x, h], dim=1)

        z = torch.sigmoid(self.conv_z(cat))
        r = torch.sigmoid(self.conv_r(cat))

        cat_r = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(cat_r))

        h_new = (1 - z) * h + z * h_tilde

        return h_new, h_new


# ==========================================================
# Temporal Memory (unified interface)
# ==========================================================

class TemporalMemory(nn.Module):
    """
    Temporal aggregation with configurable backend.

    Parameters
    ----------
    channels : int
        Feature dimension (C).
    memory_type : str
        'transformer', 'convlstm', or 'convgru'
    num_layers : int
        Number of layers (transformer only).
    num_heads : int
        Number of attention heads (transformer only).
    """

    def __init__(
        self,
        channels: int = 256,
        memory_type: str = "transformer",
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.memory_type = memory_type

        if memory_type == "transformer":
            self.positional_encoding = nn.Parameter(
                torch.randn(1, max_seq_len, channels) * 0.02
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=num_heads,
                dim_feedforward=int(channels * mlp_ratio),
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers,
            )
            self.max_seq_len = max_seq_len

        elif memory_type == "convlstm":
            self.cell = ConvLSTMCell(channels, channels, kernel_size=3)

        elif memory_type == "convgru":
            self.cell = ConvGRUCell(channels, channels, kernel_size=3)

        else:
            raise ValueError(
                f"Unknown memory_type: {memory_type}. "
                f"Use 'transformer', 'convlstm', or 'convgru'."
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, T, C, H, W)

        Returns
        -------
        (B, C, H, W) — reference frame (t = T-1) representation
        """
        B, T, C, H, W = features.shape

        if self.memory_type == "transformer":
            if T > self.max_seq_len:
                raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}")

            x = features.permute(0, 3, 4, 1, 2)  # (B, H, W, T, C)
            x = x.reshape(B * H * W, T, C)

            pe = self.positional_encoding[:, :T, :]
            x = x + pe

            x = self.encoder(x)
            world = x[:, -1]  # reference frame

            world = world.reshape(B, H, W, C)
            world = world.permute(0, 3, 1, 2)

        elif self.memory_type in ("convlstm", "convgru"):
            # Process sequentially
            hidden = None
            for t in range(T):
                x_t = features[:, t]  # (B, C, H, W)
                out = self.cell(x_t, hidden)

                if self.memory_type == "convlstm":
                    # ConvLSTMCell returns (h_new, (h_new, c_new))
                    world = out[0]
                    hidden = out[1]
                else:
                    # ConvGRUCell returns (h_new, h_new)
                    world = out[0]
                    hidden = out[1]

        return world
