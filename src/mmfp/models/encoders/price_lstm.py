r"""LSTM encoder for sequential price features.

Inputs are ``(B, L, F_price)`` tensors produced by
:class:`~mmfp.datasets.multimodal.MultiModalDataset` when
``cfg.data.lookback > 1``. Output is the last-layer final hidden state
``h_n[-1]`` of a stacked LSTM, layer-normalised to :math:`H =
\mathrm{cfg.model.hidden\_dim}`.

Ported from v1 ``src/models/encoders.py::PriceEncoder``. Changes:

* ``num_layers`` comes from ``cfg.model.price_lstm_layers`` (configurable
  rather than hardcoded at 2).
* Dropout comes from ``cfg.model.dropout`` (no hardcoded 0.2).
* Type hints + docstrings cleaned up.

The final ``nn.LayerNorm(H)`` matches v1's behaviour and matches the
universal encoder contract documented in :mod:`mmfp.models.encoders.base`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mmfp.config.schema import ExperimentConfig

from .base import EncoderBase


class PriceLSTMEncoder(EncoderBase):
    """Stacked LSTM over price/technical-indicator sequences.

    Parameters
    ----------
    cfg
        Full experiment config. Only ``cfg.model.hidden_dim``,
        ``cfg.model.dropout``, and ``cfg.model.price_lstm_layers`` are
        consulted.
    input_dim
        Per-step feature dimension (``F_price``). Discovered by
        :class:`~mmfp.data.assemble.FeatureSchema`.

    Notes
    -----
    PyTorch documents LSTM dropout as applied to the output of each
    intermediate layer (not the last), so the effective dropout is zero
    when ``num_layers == 1``. We pass ``dropout=0`` in that case to
    suppress PyTorch's UserWarning.
    """

    def __init__(self, cfg: ExperimentConfig, input_dim: int) -> None:
        super().__init__(output_dim=cfg.model.hidden_dim)

        if input_dim < 1:
            raise ValueError(
                f"PriceLSTMEncoder requires input_dim >= 1; got {input_dim}"
            )

        num_layers = cfg.model.price_lstm_layers
        dropout = cfg.model.dropout

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.output_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a ``(B, L, F_price)`` batch to ``(B, H)``.

        Parameters
        ----------
        x
            Shape ``(B, L, F_price)``.

        Returns
        -------
        Tensor
            Shape ``(B, H)``. The last-layer final hidden state,
            layer-normalised.
        """
        if x.dim() != 3:
            raise ValueError(
                f"PriceLSTMEncoder expects (B, L, F_price); got shape {tuple(x.shape)}"
            )
        _, (h_n, _) = self.lstm(x)
        # ``h_n`` is (num_layers, B, H); take the final layer.
        return self.norm(h_n[-1])


__all__ = ["PriceLSTMEncoder"]
