"""Generic tabular feedforward encoder.

Replaces v1's four aliased classes (``TextEncoder``, ``NewsEncoder``,
``SocialEncoder``, ``MacroEncoder``) with a single implementation
parameterised by input dimension and depth (audit B.5). The only
conceptually distinct tabular operation across those four v1 classes was
the *input dimension*; the rest (Linear + ReLU + Dropout + LayerNorm)
was copied verbatim.

Used for:

* macro features (`F_macro` floats)
* news 11-dim statistical features
* news embeddings post-PCA (e.g. 32-dim)
* social features (per-stock StockTwits stats)

Depth is configurable via ``cfg.model.tabular_hidden_layers`` (default 1
= single linear + non-linearity, matching v1). Deeper tabular stacks are
supported for the high-dim encoder ablation so a 768-dim FinBERT
embedding can be reduced through multiple bottlenecks rather than a
single wide projection.
"""

from __future__ import annotations

from torch import Tensor, nn

from mmfp.config.schema import ExperimentConfig

from .base import EncoderBase


class TabularFFEncoder(EncoderBase):
    """Dense tabular encoder ``(B, F_mod) -> (B, H)``.

    Parameters
    ----------
    cfg
        Full experiment config. ``cfg.model.hidden_dim``,
        ``cfg.model.dropout``, and ``cfg.model.tabular_hidden_layers``
        are consulted.
    input_dim
        Number of feature columns for this modality. Discovered by
        :class:`~mmfp.data.assemble.FeatureSchema`.

    Notes
    -----
    Architecture:

    * **Depth 1** (``tabular_hidden_layers=1``): ``Linear(F_in, H) ->
      ReLU -> Dropout -> LayerNorm(H)``. Matches v1's four aliased
      classes exactly (up to the dropout value coming from config).
    * **Depth > 1**: adds ``hidden_layers - 1`` additional blocks of
      ``Linear(H, H) -> ReLU -> Dropout`` between the input projection
      and the final ``LayerNorm``. Every linear except the first has
      ``in_features = out_features = H``.
    """

    def __init__(self, cfg: ExperimentConfig, input_dim: int) -> None:
        super().__init__(output_dim=cfg.model.hidden_dim)

        if input_dim < 1:
            raise ValueError(
                f"TabularFFEncoder requires input_dim >= 1; got {input_dim}"
            )

        hidden_layers = cfg.model.tabular_hidden_layers
        dropout = cfg.model.dropout

        layers: list[nn.Module] = [
            nn.Linear(input_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(hidden_layers - 1):
            layers.extend([
                nn.Linear(self.output_dim, self.output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.projection = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``(B, F_mod)`` to ``(B, H)``."""
        if x.dim() != 2:
            raise ValueError(
                f"TabularFFEncoder expects (B, F_mod); got shape {tuple(x.shape)}"
            )
        return self.norm(self.projection(x))


__all__ = ["TabularFFEncoder"]
