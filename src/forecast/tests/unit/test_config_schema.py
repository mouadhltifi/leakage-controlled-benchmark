"""Unit tests for :mod:`forecast.config.schema`, ``defaults``, ``load`` and
``validate`` (Milestone 1 deliverable).

Coverage targets:

* Every :class:`~forecast.config.schema.ForecastConfig` field: default,
  bounds, and the ``extra="forbid"`` policy.
* The ``quantiles`` custom validator: sorted, median present, (0, 1) bounds.
* :class:`~forecast.config.schema.V3ExperimentConfig` construction and the
  ``forecast`` sub-config wiring.
* Round-trip TOML -> model -> TOML identity.
* Each of the five v3 validation rules (soft and hard).
* Happy-path load of both canonical TOML files
  (``defaults.toml`` and ``A7_price_only_tft.toml``).
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any

import pytest
import tomllib
from pydantic import ValidationError

from forecast.config.defaults import DEFAULT_CONFIG
from forecast.config.load import dump_config, load_config, load_toml
from forecast.config.schema import ForecastConfig, V3ExperimentConfig
from forecast.config.validate import (
    ConfigValidationError,
    validate_forecast_config,
)


# ---------------------------------------------------------------------------
# Helpers: build a V3ExperimentConfig from DEFAULT_CONFIG with dotted-path
# overrides (mirrors v2's test helper in ``mmfp/tests/unit/test_config_schema``).
# ---------------------------------------------------------------------------


def _cfg_from(overrides: dict[str, Any]) -> V3ExperimentConfig:
    """Build a :class:`V3ExperimentConfig` from defaults plus dotted overrides."""
    d = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in overrides.items():
        path = key.split(".")
        cursor: Any = d
        for p in path[:-1]:
            cursor = cursor[p]
        cursor[path[-1]] = value
    return V3ExperimentConfig.model_validate(d)


# ---------------------------------------------------------------------------
# 1. ForecastConfig defaults.
# ---------------------------------------------------------------------------


def test_forecast_config_default_architecture() -> None:
    cfg = ForecastConfig()
    assert cfg.architecture == "tft"


def test_forecast_config_default_hidden_dim() -> None:
    assert ForecastConfig().hidden_dim == 64


def test_forecast_config_default_n_heads() -> None:
    assert ForecastConfig().n_heads == 1


def test_forecast_config_default_n_lstm_layers() -> None:
    assert ForecastConfig().n_lstm_layers == 1


def test_forecast_config_default_dropout() -> None:
    assert ForecastConfig().dropout == pytest.approx(0.3)


def test_forecast_config_default_lookback() -> None:
    assert ForecastConfig().lookback == 60


def test_forecast_config_default_quantiles() -> None:
    cfg = ForecastConfig()
    assert cfg.quantiles == (0.1, 0.25, 0.5, 0.75, 0.9)


def test_forecast_config_default_aux_weights_are_zero() -> None:
    cfg = ForecastConfig()
    assert cfg.direction_aux_weight == 0.0
    assert cfg.volatility_aux_weight == 0.0


def test_forecast_config_default_graph_precompute_true() -> None:
    assert ForecastConfig().graph_precompute is True


def test_forecast_config_default_static_cardinalities() -> None:
    cfg = ForecastConfig()
    assert cfg.n_tickers == 55
    assert cfg.n_sectors == 11


def test_forecast_config_default_warmup_epochs() -> None:
    assert ForecastConfig().warmup_epochs == 5


# ---------------------------------------------------------------------------
# 2. ForecastConfig field bounds.
# ---------------------------------------------------------------------------


def test_hidden_dim_below_minimum_rejected() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(hidden_dim=8)


def test_hidden_dim_above_maximum_rejected() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(hidden_dim=1024)


def test_n_heads_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(n_heads=0)


def test_n_heads_above_maximum_rejected() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(n_heads=16)


def test_dropout_at_one_rejected() -> None:
    # lt=1.0, so 1.0 must raise.
    with pytest.raises(ValidationError):
        ForecastConfig(dropout=1.0)


def test_dropout_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(dropout=-0.1)


def test_lookback_below_minimum_rejected() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(lookback=0)


def test_lookback_above_maximum_rejected() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(lookback=500)


def test_direction_aux_weight_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(direction_aux_weight=-0.1)


def test_warmup_epochs_above_maximum_rejected() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(warmup_epochs=50)


# ---------------------------------------------------------------------------
# 3. extra="forbid" policy.
# ---------------------------------------------------------------------------


def test_forecast_config_forbids_extra_field() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(hiddem_dim=64)  # type: ignore[call-arg]


def test_v3_experiment_config_forbids_extra_field(
    minimal_cfg_dict: dict[str, Any],
) -> None:
    d = copy.deepcopy(minimal_cfg_dict)
    d["bogus_new_section"] = "nope"
    with pytest.raises(ValidationError):
        V3ExperimentConfig.model_validate(d)


def test_forecast_sub_config_forbids_extra_field(
    minimal_cfg_dict: dict[str, Any],
) -> None:
    d = copy.deepcopy(minimal_cfg_dict)
    d["forecast"]["typo_field"] = 42
    with pytest.raises(ValidationError):
        V3ExperimentConfig.model_validate(d)


# ---------------------------------------------------------------------------
# 4. Quantiles validator.
# ---------------------------------------------------------------------------


def test_quantiles_must_be_strictly_increasing() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        ForecastConfig(quantiles=(0.1, 0.5, 0.25, 0.9))


def test_quantiles_equal_adjacent_values_rejected() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        ForecastConfig(quantiles=(0.1, 0.5, 0.5, 0.9))


def test_quantiles_must_include_median() -> None:
    with pytest.raises(ValidationError, match="median"):
        ForecastConfig(quantiles=(0.1, 0.25, 0.75, 0.9))


def test_quantiles_must_be_in_open_unit_interval_low() -> None:
    with pytest.raises(ValidationError, match=r"\(0, 1\)"):
        ForecastConfig(quantiles=(0.0, 0.5, 0.9))


def test_quantiles_must_be_in_open_unit_interval_high() -> None:
    with pytest.raises(ValidationError, match=r"\(0, 1\)"):
        ForecastConfig(quantiles=(0.1, 0.5, 1.0))


def test_quantiles_minimum_length_enforced() -> None:
    # A single quantile leaves no band; validator requires at least 2.
    with pytest.raises(ValidationError, match="at least 2"):
        ForecastConfig(quantiles=(0.5,))


def test_quantiles_from_list_is_coerced_to_tuple() -> None:
    """TOML deserialises array literals as lists; validator must accept."""
    cfg = ForecastConfig(quantiles=[0.1, 0.5, 0.9])  # type: ignore[arg-type]
    assert isinstance(cfg.quantiles, tuple)
    assert cfg.quantiles == (0.1, 0.5, 0.9)


def test_custom_three_quantile_set_accepted() -> None:
    cfg = ForecastConfig(quantiles=(0.05, 0.5, 0.95))
    assert cfg.quantiles == (0.05, 0.5, 0.95)


# ---------------------------------------------------------------------------
# 5. V3ExperimentConfig construction.
# ---------------------------------------------------------------------------


def test_v3_experiment_config_defaults_construct() -> None:
    """Top-level construction with a minimal dict (name + seed) works."""
    cfg = V3ExperimentConfig(name="unit_test", seed=0)
    assert cfg.name == "unit_test"
    assert cfg.seed == 0
    # The nested ForecastConfig defaults come from ForecastConfig's own defaults.
    assert cfg.forecast.architecture == "tft"
    assert cfg.forecast.hidden_dim == 64


def test_v3_experiment_config_happy_path(
    minimal_cfg_dict: dict[str, Any],
) -> None:
    cfg = V3ExperimentConfig.model_validate(minimal_cfg_dict)
    assert cfg.name == "default_v3"
    assert cfg.seed == 42
    assert cfg.forecast.lookback == 60
    assert cfg.data.lookback == 60
    assert cfg.training.optimizer == "adamw"
    # Active modalities behaviour inherited from v2 ExperimentConfig.
    assert cfg.active_modalities() == ["price"]


def test_v3_default_matches_forecast_defaults(
    minimal_cfg_dict: dict[str, Any],
) -> None:
    """DEFAULT_CONFIG['forecast'] must produce the same ForecastConfig as
    constructing ForecastConfig() with no args."""
    from_default_cfg = V3ExperimentConfig.model_validate(
        minimal_cfg_dict
    ).forecast
    native_default = ForecastConfig()
    assert from_default_cfg.model_dump() == native_default.model_dump()


# ---------------------------------------------------------------------------
# 6. TOML round-trip + DEFAULT_CONFIG parity.
# ---------------------------------------------------------------------------


def test_defaults_toml_matches_default_config_dict(
    defaults_toml_path: Path,
) -> None:
    """In-Python DEFAULT_CONFIG must agree with defaults.toml field-for-field.

    The two are independent representations of the same canonical defaults
    and are kept in sync manually; this test catches drift.
    """
    with defaults_toml_path.open("rb") as fh:
        toml_dict = tomllib.load(fh)

    cfg_from_toml = V3ExperimentConfig.model_validate(toml_dict).model_dump()
    cfg_from_py = V3ExperimentConfig.model_validate(
        DEFAULT_CONFIG
    ).model_dump()
    assert cfg_from_toml == cfg_from_py


def test_toml_round_trip_preserves_content(
    defaults_toml_path: Path, tmp_path: Path
) -> None:
    """Loading defaults.toml, building the model, and dumping back to TOML
    should reload to an identical config."""
    original = load_toml(defaults_toml_path)
    cfg = V3ExperimentConfig.model_validate(original)

    out = tmp_path / "v3_round_trip.toml"
    dump_config(cfg, out)

    reloaded_raw = load_toml(out)
    reloaded_cfg = V3ExperimentConfig.model_validate(reloaded_raw)

    # Compare pydantic dumps (invariant under TOML ordering).
    assert cfg.model_dump() == reloaded_cfg.model_dump()


def test_load_config_defaults_only() -> None:
    """Calling load_config with no TOML yields the DEFAULT_CONFIG instance."""
    cfg = load_config()
    assert cfg.name == "default_v3"
    assert cfg.forecast.architecture == "tft"


def test_load_config_with_cli_override_hidden_dim() -> None:
    """CLI override should reach ForecastConfig fields via dotted path."""
    cfg = load_config(overrides=["forecast.hidden_dim=128"])
    assert cfg.forecast.hidden_dim == 128


def test_load_config_with_cli_override_quantiles() -> None:
    """CLI override should reach the custom list-typed quantiles field."""
    cfg = load_config(overrides=["forecast.quantiles=[0.05, 0.5, 0.95]"])
    assert cfg.forecast.quantiles == (0.05, 0.5, 0.95)


# ---------------------------------------------------------------------------
# 7. Validation rule 1: lookback coherence warning.
# ---------------------------------------------------------------------------


def test_rule_lookback_below_30_emits_warning() -> None:
    cfg = _cfg_from({"data.lookback": 20, "forecast.lookback": 20})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_forecast_config(cfg)
    msgs = [str(w.message) for w in caught]
    assert any(
        "lookback" in m and "30" in m for m in msgs
    ), f"expected a lookback>=30 UserWarning; got {msgs}"


def test_rule_lookback_at_30_does_not_warn() -> None:
    cfg = _cfg_from({"data.lookback": 30, "forecast.lookback": 30})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_forecast_config(cfg)
    msgs = [str(w.message) for w in caught]
    assert not any(
        "data.lookback >= 30" in m for m in msgs
    ), f"lookback=30 should not trigger warning; got {msgs}"


# ---------------------------------------------------------------------------
# 8. Validation rule 2: news encoder scope lock (Qwen3 + PCA128).
# ---------------------------------------------------------------------------


def test_rule_news_encoder_must_be_qwen3() -> None:
    cfg = _cfg_from(
        {
            "news.enabled": True,
            "news.encoder": "finbert_cls_768",
            "news.pca_dims": 128,
        }
    )
    with pytest.raises(ConfigValidationError, match="qwen3_embedding"):
        validate_forecast_config(cfg)


def test_rule_news_pca_dims_must_be_128() -> None:
    cfg = _cfg_from(
        {
            "news.enabled": True,
            "news.encoder": "qwen3_embedding",
            "news.pca_dims": 32,
        }
    )
    with pytest.raises(ConfigValidationError, match="pca_dims=128"):
        validate_forecast_config(cfg)


def test_rule_news_disabled_skips_encoder_check() -> None:
    """When news.enabled is False, the encoder/pca combo is irrelevant."""
    cfg = _cfg_from(
        {
            "news.enabled": False,
            "news.encoder": "finbert_11dim",
            "news.pca_dims": None,
        }
    )
    # Should not raise — news rule only fires when news.enabled is True.
    validate_forecast_config(cfg)


def test_rule_news_enabled_with_qwen3_and_128_passes() -> None:
    cfg = _cfg_from(
        {
            "news.enabled": True,
            "news.encoder": "qwen3_embedding",
            "news.pca_dims": 128,
        }
    )
    validate_forecast_config(cfg)


# ---------------------------------------------------------------------------
# 9. Validation rule 3: dynamic graph + refresh-divides-lookback.
# ---------------------------------------------------------------------------


def test_rule_dynamic_graph_refresh_must_divide_lookback() -> None:
    # 60 % 7 == 4, so this should fail.
    cfg = _cfg_from(
        {
            "graph.enabled": True,
            "graph.source": "dynamic_corr",
            "graph.dynamic_refresh_every": 7,
            "forecast.graph_precompute": True,
            "forecast.lookback": 60,
            "data.lookback": 60,
        }
    )
    with pytest.raises(
        ConfigValidationError, match="dynamic_refresh_every"
    ):
        validate_forecast_config(cfg)


def test_rule_dynamic_graph_refresh_that_divides_passes() -> None:
    # 60 % 20 == 0, passes.
    cfg = _cfg_from(
        {
            "graph.enabled": True,
            "graph.source": "dynamic_corr",
            "graph.dynamic_refresh_every": 20,
            "forecast.graph_precompute": True,
            "forecast.lookback": 60,
            "data.lookback": 60,
        }
    )
    validate_forecast_config(cfg)


def test_rule_static_graph_skips_divisibility_check() -> None:
    """Static-GICS graphs have no refresh cadence; rule shouldn't fire."""
    cfg = _cfg_from(
        {
            "graph.enabled": True,
            "graph.source": "static_gics",
            "graph.dynamic_refresh_every": 7,  # would fail under dynamic
            "forecast.graph_precompute": True,
            "forecast.lookback": 60,
        }
    )
    validate_forecast_config(cfg)


# ---------------------------------------------------------------------------
# 10. Validation rule 4: Adam + TFT warning.
# ---------------------------------------------------------------------------


def test_rule_adam_with_tft_emits_warning() -> None:
    cfg = _cfg_from({"training.optimizer": "adam"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_forecast_config(cfg)
    msgs = [str(w.message) for w in caught]
    assert any(
        "AdamW is recommended" in m for m in msgs
    ), f"expected AdamW recommendation warning; got {msgs}"


def test_rule_adamw_with_tft_does_not_warn() -> None:
    cfg = _cfg_from({"training.optimizer": "adamw"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_forecast_config(cfg)
    msgs = [str(w.message) for w in caught]
    assert not any(
        "AdamW is recommended" in m for m in msgs
    )


# ---------------------------------------------------------------------------
# 11. Validation rule 5: hidden_dim divisible by n_heads.
# ---------------------------------------------------------------------------


def test_rule_hidden_dim_not_divisible_by_n_heads_raises() -> None:
    # 64 % 5 != 0 -> fails.
    cfg = _cfg_from({"forecast.hidden_dim": 64, "forecast.n_heads": 5})
    with pytest.raises(ConfigValidationError, match="divisible"):
        validate_forecast_config(cfg)


def test_rule_hidden_dim_divisible_by_n_heads_passes() -> None:
    cfg = _cfg_from({"forecast.hidden_dim": 64, "forecast.n_heads": 4})
    validate_forecast_config(cfg)


# ---------------------------------------------------------------------------
# 12. Canonical TOML files load cleanly.
# ---------------------------------------------------------------------------


def test_defaults_toml_loads_and_validates(
    defaults_toml_path: Path,
) -> None:
    cfg = load_config(defaults_toml_path)
    assert cfg.name == "default_v3"
    assert cfg.forecast.architecture == "tft"
    assert cfg.data.lookback == 60
    assert cfg.training.optimizer == "adamw"


def test_a7_toml_loads_and_validates(a7_toml_path: Path) -> None:
    cfg = load_config(a7_toml_path)
    assert cfg.name == "v3_A7_price_only"
    assert cfg.forecast.architecture == "tft"
    assert cfg.price.enabled is True
    assert cfg.news.enabled is False
    assert cfg.social.enabled is False
    assert cfg.macro.enabled is False
    assert cfg.graph.enabled is False
    assert cfg.active_modalities() == ["price"]
