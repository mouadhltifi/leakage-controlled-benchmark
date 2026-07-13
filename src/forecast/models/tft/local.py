"""LSTM local processor (Lim et al. 2021, §4.3).

A single-layer (default) LSTM over the VSN-selected past-observed
sequence. Hidden and cell states are initialised from two of the static
contexts (``c_h`` and ``c_c`` from :class:`StaticCovariateEncoders`),
so the LSTM has access to per-sample static information from step 0.

Notes on dropout
----------------

``torch.nn.LSTM``'s built-in ``dropout`` parameter is **inter-layer
only** and does nothing when ``num_layers == 1``. We therefore apply an
explicit ``nn.Dropout`` on the output tensor to respect the global
``dropout`` config across the full TFT body (the architecture spec).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LocalProcessor(nn.Module):
    """LSTM local processor over the past-observed sequence.

    Parameters
    ----------
    hidden_dim
        Hidden width ``H``. The LSTM's input, hidden, and output dims
        are all ``H`` (matching the VSN output width).
    n_lstm_layers
        Number of stacked LSTM layers. Default 1 (paper / spec).
    dropout
        Global dropout probability. Applied on the LSTM output tensor;
        when ``n_lstm_layers > 1`` it is additionally passed to
        ``nn.LSTM`` so inter-layer dropout kicks in.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_lstm_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {hidden_dim}")
        if n_lstm_layers < 1:
            raise ValueError(
                f"n_lstm_layers must be >= 1; got {n_lstm_layers}"
            )

        self.hidden_dim = hidden_dim
        self.n_lstm_layers = n_lstm_layers

        # torch's LSTM inter-layer dropout is only active when num_layers >= 2.
        inter_layer_dropout = dropout if n_lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=inter_layer_dropout,
        )
        # Explicit output dropout — LSTM's built-in does nothing at
        # num_layers=1 (our default).
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        c_h: Tensor,
        c_c: Tensor,
    ) -> Tensor:
        """Run the LSTM over the past sequence.

        Parameters
        ----------
        x
            Input tensor of shape ``(B, L, H)``.
        c_h
            Static-derived hidden initialiser of shape ``(B, H)``.
            Repeated across LSTM layers for a stacked configuration.
        c_c
            Static-derived cell initialiser of shape ``(B, H)``.

        Returns
        -------
        Tensor
            LSTM output of shape ``(B, L, H)``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"LSTM input must be (B, L, H); got {tuple(x.shape)}"
            )
        if c_h.shape != c_c.shape:
            raise ValueError(
                f"c_h shape {tuple(c_h.shape)} != c_c shape "
                f"{tuple(c_c.shape)}"
            )
        if c_h.dim() != 2 or c_h.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"c_h must be (B, H={self.hidden_dim}); got "
                f"{tuple(c_h.shape)}"
            )

        # PyTorch expects (num_layers, B, H) for initial states.
        # Repeat the single static context across layers so every layer
        # shares the same initialisation (the classic approach in TFT
        # reference implementations).
        h0 = c_h.unsqueeze(0).expand(self.n_lstm_layers, -1, -1).contiguous()
        c0 = c_c.unsqueeze(0).expand(self.n_lstm_layers, -1, -1).contiguous()

        out, _ = self.lstm(x, (h0, c0))
        return self.output_dropout(out)


__all__ = ["LocalProcessor"]
