"""v3 configuration loader — mirrors v2's layered defaults + TOML + CLI.

Reuses v2's merging helpers (:func:`mmfp.config.load._deep_merge`,
:func:`mmfp.config.load._apply_cli_overrides`, :func:`mmfp.config.load.load_toml`,
:func:`mmfp.config.load._normalise_for_toml`) so behaviour is identical to
v2; only the output type differs — v3 produces
:class:`~forecast.config.schema.V3ExperimentConfig`.

Precedence (lowest → highest):

1. :data:`forecast.config.defaults.DEFAULT_CONFIG`
2. Experiment TOML file passed to :func:`load_config`
3. ``--set key.path=value`` CLI overrides
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

# Reuse v2's private helpers (underscored) — they are stable within the
# v2 package and v3 explicitly depends on them per the architecture-spec
# reuse map (§7).
from mmfp.config.load import (
    _apply_cli_overrides,
    _deep_merge,
    _normalise_for_toml,
    load_toml,
)

from forecast.config.defaults import DEFAULT_CONFIG
from forecast.config.schema import V3ExperimentConfig
from forecast.config.validate import validate_forecast_config


def load_config(
    experiment_toml: str | Path | None = None,
    overrides: Iterable[str] | None = None,
    *,
    validate: bool = True,
) -> V3ExperimentConfig:
    """Load a :class:`V3ExperimentConfig` with layered defaults.

    Parameters
    ----------
    experiment_toml
        Optional path to an experiment-specific TOML. When ``None``, only
        the v3 defaults (plus any CLI overrides) are used.
    overrides
        Optional iterable of ``"dotted.path=value"`` CLI overrides; values
        are parsed as TOML scalars and applied last.
    validate
        If ``True`` (default), run the cross-field validator. Tests may
        pass ``False`` to inspect partial configs.

    Returns
    -------
    V3ExperimentConfig
        The fully validated v3 experiment config.
    """
    merged = copy.deepcopy(DEFAULT_CONFIG)
    if experiment_toml is not None:
        toml_data = load_toml(experiment_toml)
        merged = _deep_merge(merged, toml_data)
    if overrides is not None:
        merged = _apply_cli_overrides(merged, overrides)

    cfg = V3ExperimentConfig.model_validate(merged)
    if validate:
        validate_forecast_config(cfg)
    return cfg


def dump_config(cfg: V3ExperimentConfig, path: str | Path) -> None:
    """Dump a :class:`V3ExperimentConfig` to a TOML file.

    Supports the round-trip test ``TOML → model → TOML``. ``None`` values
    are handled by v2's :func:`mmfp.config.load._normalise_for_toml` helper
    (retains empty string for ``news.pca_train_end`` which the schema
    validator re-coerces on load; drops other ``None`` fields and lets
    the pydantic defaults reassert on reload).
    """
    import tomli_w

    path = Path(path)
    data = cfg.model_dump(mode="python")
    data = _normalise_for_toml(data, path=())
    with path.open("wb") as fh:
        tomli_w.dump(data, fh)


__all__ = ["load_config", "load_toml", "dump_config"]
