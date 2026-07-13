"""v3 configuration package: ``ForecastConfig`` + extended experiment schema.

Usage::

    from forecast.config.load import load_config
    cfg = load_config("forecast/configs/experiments/A7_price_only_tft.toml")
    cfg.forecast.architecture  # -> "tft"

v2's pydantic sub-configs (``DataConfig``, ``NewsConfig``, ...) are imported
verbatim from :mod:`mmfp.config.schema`. v3 adds:

* :class:`~forecast.config.schema.ForecastConfig` — TFT-specific knobs.
* :class:`~forecast.config.schema.V3ExperimentConfig` — root config composed
  of v2's ``ExperimentConfig`` plus ``forecast``.
* Cross-field rules in :mod:`forecast.config.validate`.
* Canonical defaults in :mod:`forecast.config.defaults` and the paired
  ``forecast/configs/defaults.toml``.

Precedence for :func:`forecast.config.load.load_config`:

1. ``DEFAULT_CONFIG`` in :mod:`forecast.config.defaults`
2. Experiment TOML file
3. ``--set key.path=value`` CLI overrides
"""

from forecast.config.schema import ForecastConfig, V3ExperimentConfig
from forecast.config.validate import (
    ConfigValidationError,
    validate_forecast_config,
)

__all__ = [
    "ConfigValidationError",
    "ForecastConfig",
    "V3ExperimentConfig",
    "validate_forecast_config",
]
