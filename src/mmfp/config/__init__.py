"""Configuration package: pydantic schema, defaults, loader, validators.

The single source of truth for every experimental knob.

Usage::

    from mmfp.config.load import load_config
    cfg = load_config("mmfp/configs/experiments/A7_price_only.toml",
                      overrides=["training.batch_size=128"])

Precedence order (lowest to highest):

1. ``DEFAULT_CONFIG`` in :mod:`mmfp.config.defaults`
2. TOML file passed to :func:`mmfp.config.load.load_config`
3. CLI ``--set key=value`` dotted-path overrides
"""

from mmfp.config.schema import (
    DataConfig,
    ExperimentConfig,
    FusionConfig,
    GraphConfig,
    HeadConfig,
    IntradayConfig,
    LoggingConfig,
    MacroConfig,
    ModelConfig,
    NewsConfig,
    PriceConfig,
    SocialConfig,
    StandardizationConfig,
    TrainingConfig,
)

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "FusionConfig",
    "GraphConfig",
    "HeadConfig",
    "IntradayConfig",
    "LoggingConfig",
    "MacroConfig",
    "ModelConfig",
    "NewsConfig",
    "PriceConfig",
    "SocialConfig",
    "StandardizationConfig",
    "TrainingConfig",
]
