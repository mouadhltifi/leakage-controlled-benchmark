r"""Feedforward price encoder for ``lookback=1`` configurations.

When ``cfg.data.lookback == 1`` the multimodal dataset emits a flat
``(B, F_price)`` tensor (no temporal dimension). The feedforward encoder
projects that to :math:`H = \mathrm{cfg.model.hidden\_dim}` using a
single ``Linear + ReLU + Dropout + LayerNorm`` block.

This class is new relative to v1 — v1 implicitly reused
``MacroEncoder`` for the ``lookback=1`` price case. Making the intent
explicit (audit B.5) means the selection logic lives in
:class:`~mmfp.models.predictor.Predictor` rather than being spread across
encoder aliases.
"""

from __future__ import annotations

from torch import Tensor, nn

from mmfp.config.schema import ExperimentConfig

from .base import EncoderBase


class PriceFFEncoder(EncoderBase):
    """Dense feedforward projection for flat price feature vectors.

    Parameters
    ----------
    cfg
        Full experiment config. Only ``cfg.model.hidden_dim`` and
        ``cfg.model.dropout`` are consulted.
    input_dim
        Number of price feature columns (``F_price``). Discovered by
        :class:`~mmfp.data.assemble.FeatureSchema`.
    """

    def __init__(self, cfg: ExperimentConfig, input_dim: int) -> None:
        super().__init__(output_dim=cfg.model.hidden_dim)

        if input_dim < 1:
            raise ValueError(
                f"PriceFFEncoder requires input_dim >= 1; got {input_dim}"
            )

        self.projection = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(cfg.model.dropout),
        )
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``(B, F_price)`` to ``(B, H)``."""
        if x.dim() != 2:
            raise ValueError(
                f"PriceFFEncoder expects (B, F_price); got shape {tuple(x.shape)}"
            )
        return self.norm(self.projection(x))


__all__ = ["PriceFFEncoder"]
