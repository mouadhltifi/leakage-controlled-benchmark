"""Encoders: one class per conceptually distinct computation.

Milestone 5 deliverable. Resolves audit finding B.5 (aliased encoders).

Every encoder inherits from :class:`EncoderBase` and honours its
contract: output dim equal to ``cfg.model.hidden_dim``, dropout read
from ``cfg.model.dropout``, ``nn.LayerNorm`` as the final op.

Public classes
--------------

* :class:`PriceLSTMEncoder` — sequential price path (lookback > 1).
* :class:`PriceFFEncoder` — flat price path (lookback == 1).
* :class:`TabularFFEncoder` — unifies v1's ``TextEncoder``,
  ``NewsEncoder``, ``SocialEncoder`` and ``MacroEncoder``.
* :class:`GraphGATEncoder` — graph attention network with optional
  static+dynamic averaging head.
"""

from .base import EncoderBase
from .graph_gat import GraphGATEncoder
from .price_ff import PriceFFEncoder
from .price_lstm import PriceLSTMEncoder
from .tabular_ff import TabularFFEncoder

__all__ = [
    "EncoderBase",
    "GraphGATEncoder",
    "PriceFFEncoder",
    "PriceLSTMEncoder",
    "TabularFFEncoder",
]
