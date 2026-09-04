"""
Temporal encoder for the world model.

The public interface is the TemporalEncoder class.

Version 1 uses a single-layer ConvLSTM internally.
The encoder preserves the temporal dimension and returns the hidden
representation for every timestep.

Input:
    (B, T, C, H, W)

Output:
    (B, T, HIDDEN, H, W)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell.

    Reference:
        Xingjian et al.,
        "Convolutional LSTM Network:
        A Machine Learning Approach for
        Precipitation Nowcasting"
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.hidden_channels = hidden_channels

        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        h_prev, c_prev = state

        combined = torch.cat([x, h_prev], dim=1)

        gates = self.gates(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c

    def init_hidden(
        self,
        batch_size: int,
        spatial_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        h, w = spatial_size

        hidden = torch.zeros(
            batch_size,
            self.hidden_channels,
            h,
            w,
            device=device,
            dtype=dtype,
        )

        cell = torch.zeros_like(hidden)

        return hidden, cell


class TemporalEncoder(nn.Module):
    """
    Temporal feature encoder.

    Version 1:
        Single-layer ConvLSTM

    Future versions:
        - Transformer
        - Mamba
        - SSM
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        encoder_type: str = "convlstm",
    ) -> None:
        super().__init__()

        if encoder_type.lower() != "convlstm":
            raise NotImplementedError(
                f"Temporal encoder '{encoder_type}' "
                "is not implemented."
            )

        self.cell = ConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor of shape

            (B, T, C, H, W)

        hidden_state:
            Optional initial state.

        Returns
        -------
        Tensor

            (B, T, hidden_channels, H, W)
        """

        if x.ndim != 5:
            raise ValueError(
                f"Expected input shape (B,T,C,H,W), "
                f"got {tuple(x.shape)}"
            )

        batch_size, seq_len, _, height, width = x.shape

        if hidden_state is None:
            hidden_state = self.cell.init_hidden(
                batch_size=batch_size,
                spatial_size=(height, width),
                device=x.device,
                dtype=x.dtype,
            )

        h, c = hidden_state

        outputs = []

        for t in range(seq_len):
            h, c = self.cell(
                x[:, t],
                (h, c),
            )

            outputs.append(h)

        return torch.stack(outputs, dim=1)