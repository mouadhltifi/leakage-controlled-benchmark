"""Tests for :mod:`mmfp.config.schema` and the cross-field validator.

Covers:

* Happy-path construction of every model with defaults.
* Constraint violations on single fields (range, literals, etc).
* All cross-field validation rules with at least one failing case.
* Round-trip TOML -> model -> TOML via :func:`dump_config`.
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any

import pytest
import tomllib
from pydantic import ValidationError

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.load import dump_config, load_toml
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
from mmfp.config.validate import (
    ConfigValidationError,
    validate_experiment_config,
)


# ---------------------------------------------------------------------------
# Happy paths: each sub-config loads with defaults.
# ---------------------------------------------------------------------------


def test_data_config_defaults() -> None:
    cfg = DataConfig()
    assert cfg.universe == "sp500_sector_55"
    assert cfg.fold_idx == 0
    assert cfg.lookback == 1
    assert cfg.deadzone == 0.005


def test_price_config_default() -> None:
    assert PriceConfig().enabled is True


def test_news_config_default_disabled() -> None:
    cfg = NewsConfig()
    assert cfg.enabled is False
    assert cfg.encoder == "finbert_11dim"
    assert cfg.pca_dims == 32
    assert cfg.pca_train_end is None  # "" -> None via validator


def test_news_pca_train_end_empty_becomes_none() -> None:
    cfg = NewsConfig(pca_train_end="")
    assert cfg.pca_train_end is None


def test_news_pca_train_end_iso_date() -> None:
    cfg = NewsConfig(pca_train_end="2019-06-30")
    assert cfg.pca_train_end == "2019-06-30"


def test_news_pca_train_end_invalid_raises() -> None:
    with pytest.raises(ValidationError):
        NewsConfig(pca_train_end="06-30-2019")


def test_social_config_default_disabled() -> None:
    cfg = SocialConfig()
    assert cfg.enabled is False
    assert cfg.aggregation_window == 3


def test_macro_config_default_disabled() -> None:
    assert MacroConfig().enabled is False


def test_graph_config_default_disabled() -> None:
    cfg = GraphConfig()
    assert cfg.enabled is False
    assert cfg.source == "static_gics"
    assert cfg.dynamic_threshold == 0.3


def test_standardization_default_all_on() -> None:
    cfg = StandardizationConfig()
    assert cfg.price is True
    assert cfg.macro is True
    assert cfg.news is True
    assert cfg.social is True
    assert cfg.graph_node is True
    assert cfg.fit_scope == "train_fold_only"


def test_model_config_defaults() -> None:
    cfg = ModelConfig()
    assert cfg.hidden_dim == 64
    assert cfg.dropout == 0.2
    assert cfg.num_heads == 4
    assert cfg.price_encoder == "lstm"


def test_fusion_config_defaults() -> None:
    cfg = FusionConfig()
    assert cfg.strategy == "concat"
    assert cfg.primary_modality == "price"


def test_head_config_defaults() -> None:
    cfg = HeadConfig()
    assert cfg.architecture == "parallel_multitask"
    assert cfg.targets == ["direction", "return"]
    assert cfg.mtl_alpha == 0.5


def test_training_config_defaults() -> None:
    cfg = TrainingConfig()
    assert cfg.batch_size == 64
    assert cfg.patience == 30
    assert cfg.grad_clip == 1.0


def test_intraday_config_default_disabled() -> None:
    cfg = IntradayConfig()
    assert cfg.enabled is False
    assert cfg.frequency == "daily"
    assert cfg.source == "none"


def test_logging_config_defaults() -> None:
    cfg = LoggingConfig()
    assert cfg.level == "INFO"
    assert cfg.per_epoch_log_every == 10


def test_experiment_config_happy_path(
    minimal_cfg_dict: dict[str, Any],
) -> None:
    cfg = ExperimentConfig.model_validate(minimal_cfg_dict)
    assert cfg.name == "default"
    assert cfg.seed == 42
    assert cfg.device == "auto"
    assert cfg.active_modalities() == ["price"]


def test_active_modalities_tracks_enabled_flags(
    minimal_cfg_dict: dict[str, Any],
) -> None:
    d = copy.deepcopy(minimal_cfg_dict)
    d["news"]["enabled"] = True
    d["social"]["enabled"] = True
    cfg = ExperimentConfig.model_validate(d)
    assert cfg.active_modalities() == ["price", "news", "social"]


# ---------------------------------------------------------------------------
# Constraint violations on individual fields.
# ---------------------------------------------------------------------------


def test_fold_idx_out_of_range() -> None:
    with pytest.raises(ValidationError):
        DataConfig(fold_idx=5)


def test_lookback_out_of_range() -> None:
    with pytest.raises(ValidationError):
        DataConfig(lookback=0)
    with pytest.raises(ValidationError):
        DataConfig(lookback=100)


def test_deadzone_negative() -> None:
    with pytest.raises(ValidationError):
        DataConfig(deadzone=-0.001)


def test_dropout_out_of_range() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(dropout=1.0)
    with pytest.raises(ValidationError):
        ModelConfig(dropout=-0.1)


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(batch_size=0)


def test_invalid_encoder_literal() -> None:
    with pytest.raises(ValidationError):
        NewsConfig(encoder="bogus")  # type: ignore[arg-type]


def test_extra_field_forbidden(minimal_cfg_dict: dict[str, Any]) -> None:
    d = copy.deepcopy(minimal_cfg_dict)
    d["bogus_field"] = "nope"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(d)


def test_bad_iso_date_rejected() -> None:
    with pytest.raises(ValidationError):
        DataConfig(start_date="2015/01/01")


def test_head_targets_empty_rejected() -> None:
    with pytest.raises(ValidationError):
        HeadConfig(targets=[])


def test_head_targets_duplicate_rejected() -> None:
    with pytest.raises(ValidationError):
        HeadConfig(targets=["direction", "direction"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Cross-field validation rules. The spec's Section 3.1 lists 7 rules; the
# implementation adds two safety rules (single-task + lookback-LSTM warning).
# Each error path is exercised here.
# ---------------------------------------------------------------------------


def _cfg_from(overrides: dict[str, Any]) -> ExperimentConfig:
    d = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in overrides.items():
        path = key.split(".")
        cursor: Any = d
        for p in path[:-1]:
            cursor = cursor[p]
        cursor[path[-1]] = value
    return ExperimentConfig.model_validate(d)


def test_rule_gated_cross_attention_requires_multiple_modalities() -> None:
    # Default has only price -> gated cross-attention should fail.
    cfg = _cfg_from({"fusion.strategy": "gated_cross_attention"})
    with pytest.raises(ConfigValidationError, match=">=2 active"):
        validate_experiment_config(cfg)


def test_rule_gated_cross_attention_passes_with_two_modalities() -> None:
    cfg = _cfg_from(
        {
            "fusion.strategy": "gated_cross_attention",
            "news.enabled": True,
            "news.encoder": "finbert_cls_768",  # pca_dims allowed
            "fusion.primary_modality": "price",
        }
    )
    validate_experiment_config(cfg)


def test_rule_gated_cross_attention_primary_modality_must_be_active() -> None:
    cfg = _cfg_from(
        {
            "fusion.strategy": "gated_cross_attention",
            "news.enabled": True,
            "news.encoder": "finbert_cls_768",
            "fusion.primary_modality": "social",  # not active
        }
    )
    with pytest.raises(ConfigValidationError, match="primary_modality"):
        validate_experiment_config(cfg)


def test_rule_finbert_11dim_forbids_pca() -> None:
    cfg = _cfg_from(
        {
            "news.enabled": True,
            "news.encoder": "finbert_11dim",
            "news.pca_dims": 32,
        }
    )
    with pytest.raises(ConfigValidationError, match="pca_dims=None"):
        validate_experiment_config(cfg)


def test_rule_finbert_11dim_with_none_pca_passes() -> None:
    cfg = _cfg_from(
        {
            "news.enabled": True,
            "news.encoder": "finbert_11dim",
            "news.pca_dims": None,
        }
    )
    validate_experiment_config(cfg)


def test_rule_spherical_mean_11dim_incompatible() -> None:
    # 11-dim + spherical is forbidden. Also 11-dim requires pca=None.
    cfg = _cfg_from(
        {
            "news.enabled": True,
            "news.encoder": "finbert_11dim",
            "news.pca_dims": None,
            "news.aggregation": "spherical_mean",
        }
    )
    with pytest.raises(ConfigValidationError, match="spherical"):
        validate_experiment_config(cfg)


def test_rule_single_task_requires_one_target() -> None:
    cfg = _cfg_from(
        {
            "head.architecture": "single_task",
            "head.targets": ["direction", "return"],
        }
    )
    with pytest.raises(ConfigValidationError, match="exactly one target"):
        validate_experiment_config(cfg)


def test_rule_single_task_with_one_target_passes() -> None:
    cfg = _cfg_from(
        {
            "head.architecture": "single_task",
            "head.targets": ["direction"],
        }
    )
    validate_experiment_config(cfg)


def test_rule_intraday_not_implemented() -> None:
    cfg = _cfg_from({"intraday.enabled": True})
    with pytest.raises(NotImplementedError):
        validate_experiment_config(cfg)


def test_rule_lookback_1_lstm_warns() -> None:
    cfg = _cfg_from(
        {
            "data.lookback": 1,
            "model.price_encoder": "lstm",
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_experiment_config(cfg)
    assert any(
        "lookback=1" in str(w.message) for w in caught
    ), "expected LSTM lookback=1 UserWarning"


def test_rule_lookback_1_ff_does_not_warn() -> None:
    cfg = _cfg_from(
        {
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_experiment_config(cfg)
    assert not any(
        "lookback=1" in str(w.message) for w in caught
    )


def test_rule_sequential_cascade_needs_direction() -> None:
    cfg = _cfg_from(
        {
            "head.architecture": "sequential_cascade",
            "head.targets": ["return"],
        }
    )
    with pytest.raises(ConfigValidationError, match="direction"):
        validate_experiment_config(cfg)


def test_rule_sequential_cascade_reg_target_must_be_active() -> None:
    cfg = _cfg_from(
        {
            "head.architecture": "sequential_cascade",
            "head.targets": ["direction", "return"],
            "head.cascade_reg_target": "volatility",
        }
    )
    with pytest.raises(ConfigValidationError, match="cascade_reg_target"):
        validate_experiment_config(cfg)


# ---------------------------------------------------------------------------
# Round-trip: TOML -> model -> TOML.
# ---------------------------------------------------------------------------


def test_toml_round_trip_preserves_content(
    defaults_toml_path: Path, tmp_path: Path
) -> None:
    """Loading defaults.toml, building the model, and dumping back to TOML
    should yield identical content after a reload.
    """
    original = load_toml(defaults_toml_path)
    cfg = ExperimentConfig.model_validate(original)

    out = tmp_path / "round_trip.toml"
    dump_config(cfg, out)

    reloaded_raw = load_toml(out)
    reloaded_cfg = ExperimentConfig.model_validate(reloaded_raw)

    # Compare the pydantic model dumps (invariant under field order).
    assert cfg.model_dump() == reloaded_cfg.model_dump()


def test_defaults_toml_matches_default_config_dict(
    defaults_toml_path: Path,
) -> None:
    """The in-Python ``DEFAULT_CONFIG`` must agree with ``defaults.toml``.

    They are two representations of the same canonical defaults and must
    stay in sync.
    """
    with defaults_toml_path.open("rb") as fh:
        toml_dict = tomllib.load(fh)

    cfg_from_toml = ExperimentConfig.model_validate(toml_dict).model_dump()
    cfg_from_py = ExperimentConfig.model_validate(DEFAULT_CONFIG).model_dump()
    assert cfg_from_toml == cfg_from_py
