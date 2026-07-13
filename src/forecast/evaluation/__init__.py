"""PLACEHOLDER (M5): v3 evaluation metrics (quantile-specific additions).

M5 will reuse :mod:`mmfp.evaluation` as a base and add two v3-native
functions to :mod:`mmfp.evaluation.metrics`:

* ``quantile_coverage(q_pred, y_true, lo_idx, hi_idx)`` — empirical
  fraction of samples inside the predicted 80% band.
* ``interval_width(q_pred, lo_idx, hi_idx)`` — mean width normalised
  by ``std(y_true)``.

Existing metrics (MCC, F1, accuracy, R², RMSE, Sharpe, volatility
RMSE/R²) are reused verbatim via derivations on the quantile output
(the architecture spec). No v3-specific evaluation code lives here yet.
"""

from __future__ import annotations

__all__: list[str] = []
