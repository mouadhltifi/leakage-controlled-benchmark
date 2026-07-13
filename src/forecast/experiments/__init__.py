"""v3 experiment runner, result schema, and sweep wrapper (M5).

Public surface
--------------

* :func:`run_one_forecast_experiment` -- atomic unit, produces a
  :class:`ForecastResultRecord`.
* :class:`ForecastResultRecord` -- v2's record + two quantile
  calibration columns.
* :func:`run_forecast_sweep` -- thin fork of
  :func:`mmfp.experiments.sweep.run_sweep` wired to the v3 runner.
* :func:`append_result`, :func:`load_results`, :func:`config_hash`,
  :func:`record_fingerprint`, :func:`fingerprint_from_row` -- re-exports
  of the v2 helpers so v3 callers don't need to reach across packages.
"""

from __future__ import annotations

from forecast.experiments.result_schema import (
    ForecastResultRecord,
    ResultRecord,
    append_result,
    config_hash,
    fingerprint_from_row,
    load_results,
    record_fingerprint,
)
from forecast.experiments.runner import (
    evaluate_on_test_forecast,
    run_one_forecast_experiment,
)
from forecast.experiments.sweep import run_forecast_sweep

__all__ = [
    "ForecastResultRecord",
    "ResultRecord",
    "append_result",
    "config_hash",
    "evaluate_on_test_forecast",
    "fingerprint_from_row",
    "load_results",
    "record_fingerprint",
    "run_forecast_sweep",
    "run_one_forecast_experiment",
]
