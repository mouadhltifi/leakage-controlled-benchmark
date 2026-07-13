"""v3 training integration (M4).

Exports:

* :class:`~forecast.training.trainer.ForecastTrainer` — v3 subclass of
  :class:`mmfp.training.trainer.Trainer` that bridges the quantile-loss
  signature, extends the move-to-device gate to all batch tensors, and
  reports quantile-specific validation metrics (pinball, coverage,
  interval width) alongside MCC / R² / RMSE.

v2's callbacks (:class:`EarlyStopping`, :class:`BestModelCheckpoint`,
:class:`LRLogger`) are reused verbatim via ``mmfp.training.callbacks``.
"""

from __future__ import annotations

from forecast.training.trainer import ForecastTrainer

__all__ = ["ForecastTrainer"]
