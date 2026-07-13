"""v3 forecasting model package (Milestones 2 + 3).

Modules shipped in M2:

* :mod:`forecast.models.projections` — :class:`ModalityProjections`
  (light Linear + LN per modality stream, the architecture spec).
* :mod:`forecast.models.tft` — full TFT sub-stack (GRN, VSN, static
  encoders, local LSTM, interpretable MHA, body orchestration;
  the architecture spec).
* :mod:`forecast.models.heads` — :class:`QuantileHead` +
  ``derive_direction`` / ``derive_volatility`` helpers
  (the architecture spec).
* :mod:`forecast.models.losses` — :class:`QuantileLoss` +
  :class:`ForecastLoss` composite (the architecture spec).

Added in M3:

* :mod:`forecast.models.predictor` — :class:`ForecastPredictor`
  assembling ``ModalityProjections → TFTBody → QuantileHead`` from a
  :class:`V3ExperimentConfig` (the architecture spec).
* :func:`forecast.models.losses.build_loss` — factory that returns
  the loss module matching ``cfg.forecast.architecture``.
"""

from __future__ import annotations

from forecast.models.heads import (
    QuantileHead,
    derive_direction,
    derive_volatility,
)
from forecast.models.losses import ForecastLoss, QuantileLoss, build_loss
from forecast.models.predictor import ForecastPredictor
from forecast.models.projections import ModalityProjections

__all__ = [
    "ForecastLoss",
    "ForecastPredictor",
    "ModalityProjections",
    "QuantileHead",
    "QuantileLoss",
    "build_loss",
    "derive_direction",
    "derive_volatility",
]
