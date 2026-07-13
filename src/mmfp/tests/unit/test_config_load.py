"""Tests for :func:`mmfp.config.load.load_config`.

Covers:

* Defaults-only load returns a valid config.
* TOML file overrides defaults.
* ``--set`` CLI overrides beat the TOML file.
* Precedence order: defaults < TOML < CLI.
* CLI scalar parsing (bool, int, float, string, null).
* Invalid override strings raise ``ValueError``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mmfp.config.load import (
    _apply_cli_overrides,
    _parse_scalar,
    load_config,
)


def test_load_defaults_only_no_path(defaults_toml_path: Path) -> None:
    # Pass defaults.toml explicitly; this exercises the default-merge path.
    cfg = load_config(defaults_toml_path)
    assert cfg.name == "default"
    assert cfg.seed == 42
    assert cfg.training.batch_size == 64


def test_load_no_toml_uses_in_python_defaults() -> None:
    cfg = load_config(experiment_toml=None)
    assert cfg.name == "default"
    assert cfg.seed == 42


def test_toml_overrides_defaults(tmp_path: Path) -> None:
    exp = tmp_path / "exp.toml"
    exp.write_text(
        """
name = "exp_foo"
seed = 99

[training]
batch_size = 128
"""
    )
    cfg = load_config(exp)
    assert cfg.name == "exp_foo"
    assert cfg.seed == 99
    assert cfg.training.batch_size == 128
    # Unspecified fields retain defaults.
    assert cfg.training.patience == 30


def test_cli_overrides_beat_toml(tmp_path: Path) -> None:
    exp = tmp_path / "exp.toml"
    exp.write_text(
        """
name = "exp_foo"

[training]
batch_size = 128
"""
    )
    cfg = load_config(exp, overrides=["training.batch_size=256"])
    assert cfg.training.batch_size == 256
    assert cfg.name == "exp_foo"


def test_cli_overrides_without_toml() -> None:
    cfg = load_config(
        experiment_toml=None,
        overrides=[
            "name=cli_only",
            "seed=123",
            "training.batch_size=32",
        ],
    )
    assert cfg.name == "cli_only"
    assert cfg.seed == 123
    assert cfg.training.batch_size == 32


def test_precedence_defaults_toml_cli(tmp_path: Path) -> None:
    """Defaults < TOML < CLI — verified field-by-field."""
    exp = tmp_path / "exp.toml"
    exp.write_text(
        """
name = "toml_name"

[training]
batch_size = 128
patience = 50
"""
    )
    cfg = load_config(
        exp,
        overrides=["training.batch_size=256"],
    )
    # CLI beats TOML.
    assert cfg.training.batch_size == 256
    # TOML beats defaults.
    assert cfg.training.patience == 50
    assert cfg.name == "toml_name"
    # Defaults preserved for untouched fields.
    assert cfg.training.grad_clip == 1.0


def test_parse_scalar_bool_int_float_string() -> None:
    assert _parse_scalar("true") is True
    assert _parse_scalar("false") is False
    assert _parse_scalar("42") == 42
    assert _parse_scalar("3.14") == pytest.approx(3.14)
    # Bareword string that is not a TOML scalar.
    assert _parse_scalar("hello") == "hello"
    # Quoted string stays a string.
    assert _parse_scalar('"42"') == "42"


def test_parse_scalar_null_synonyms() -> None:
    assert _parse_scalar("none") is None
    assert _parse_scalar("null") is None
    assert _parse_scalar("None") is None


def test_invalid_override_missing_equals() -> None:
    with pytest.raises(ValueError, match="key=value"):
        _apply_cli_overrides({}, ["training.batch_size"])


def test_invalid_override_empty_path() -> None:
    with pytest.raises(ValueError, match="dotted path"):
        _apply_cli_overrides({}, [".foo=1"])


def test_cli_override_deep_path(tmp_path: Path) -> None:
    cfg = load_config(
        experiment_toml=None,
        overrides=[
            "head.architecture=single_task",
            "head.targets=[\"direction\"]",
        ],
    )
    assert cfg.head.architecture == "single_task"
    assert cfg.head.targets == ["direction"]


def test_invalid_config_raises_on_load() -> None:
    with pytest.raises(Exception):
        load_config(
            experiment_toml=None,
            overrides=["training.batch_size=-1"],
        )


def test_validation_rule_runs_on_load() -> None:
    """Cross-field rules run by default during load."""
    from mmfp.config.validate import ConfigValidationError

    with pytest.raises(ConfigValidationError):
        load_config(
            experiment_toml=None,
            overrides=[
                "fusion.strategy=gated_cross_attention",
            ],
        )


def test_validation_can_be_skipped() -> None:
    """Passing validate=False yields the model without cross-field rules."""
    cfg = load_config(
        experiment_toml=None,
        overrides=["fusion.strategy=gated_cross_attention"],
        validate=False,
    )
    assert cfg.fusion.strategy == "gated_cross_attention"
