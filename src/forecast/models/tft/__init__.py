"""v3 TFT forecasting body (Milestone 2).

Component modules (the architecture spec):

* :mod:`forecast.models.tft.grn`        — :class:`GRN` (ELU + GLU + LN + skip).
* :mod:`forecast.models.tft.vsn`        — :class:`VariableSelectionNetwork`.
* :mod:`forecast.models.tft.static_enc` — :class:`StaticCovariateEncoders`.
* :mod:`forecast.models.tft.local`      — :class:`LocalProcessor` (LSTM).
* :mod:`forecast.models.tft.attention`  — :class:`InterpretableMultiHeadAttention`.
* :mod:`forecast.models.tft.body`       — :class:`TFTBody` orchestrator.
"""

from __future__ import annotations

from forecast.models.tft.attention import InterpretableMultiHeadAttention
from forecast.models.tft.body import TFTBody
from forecast.models.tft.grn import GRN
from forecast.models.tft.local import LocalProcessor
from forecast.models.tft.static_enc import StaticCovariateEncoders
from forecast.models.tft.vsn import VariableSelectionNetwork

__all__ = [
    "GRN",
    "InterpretableMultiHeadAttention",
    "LocalProcessor",
    "StaticCovariateEncoders",
    "TFTBody",
    "VariableSelectionNetwork",
]
