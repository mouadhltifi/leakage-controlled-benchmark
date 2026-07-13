"""Configuration loader with defaults + TOML + CLI override layering.

Precedence (lowest to highest):

1. ``DEFAULT_CONFIG`` from :mod:`mmfp.config.defaults`
2. Experiment TOML file passed to :func:`load_config`
3. CLI ``--set key=value`` dotted-path overrides

All three layers are merged into a single dict and handed to
:class:`~mmfp.config.schema.ExperimentConfig` for validation. Cross-field
rules are applied afterwards by
:func:`mmfp.config.validate.validate_experiment_config`.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any, Iterable

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.config.validate import validate_experiment_config


def _deep_merge(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Scalar values in ``override`` replace the corresponding values in
    ``base``. Dict values are merged recursively. Lists are replaced
    wholesale (TOML semantics).
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_scalar(value: str) -> Any:
    """Parse a CLI-provided value as JSON-ish scalar.

    Attempts TOML scalar parsing; falls back to the raw string. Supports
    booleans (``true``/``false``), null (``none``/``null``), integers,
    floats, lists via JSON-like syntax, and bare strings.

    TOML semantics are honoured: ``"42"`` (quoted) is a string,
    ``42`` (unquoted) is an int.
    """
    # TOML requires a key=value line; use a temporary parse.
    tiny = f"x = {value}"
    try:
        parsed = tomllib.loads(tiny)
        return parsed["x"]
    except tomllib.TOMLDecodeError:
        pass

    # Handle bareword nulls not covered by TOML.
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None

    # Treat as bare string if TOML failed.
    return value


def _apply_cli_overrides(
    cfg: dict[str, Any], overrides: Iterable[str]
) -> dict[str, Any]:
    """Apply ``--set key.path=value`` overrides to a config dict.

    Each override is a single ``key.subkey=value`` string. Values are
    parsed as TOML scalars.

    Parameters
    ----------
    cfg
        Current merged config (``DEFAULT_CONFIG`` already applied + any
        TOML file).
    overrides
        Iterable of ``"dotted.path=value"`` strings.

    Returns
    -------
    dict
        A deep copy of ``cfg`` with overrides applied in order.
    """
    out = copy.deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"CLI override {override!r} must be in 'key=value' form"
            )
        key, _, raw_value = override.partition("=")
        path = key.strip().split(".")
        if not path or any(not p for p in path):
            raise ValueError(
                f"CLI override key {key!r} is not a valid dotted path"
            )
        parsed_value = _parse_scalar(raw_value.strip())

        # Walk the nested dict, creating intermediate dicts where needed.
        cursor = out
        for segment in path[:-1]:
            if segment not in cursor or not isinstance(
                cursor[segment], dict
            ):
                cursor[segment] = {}
            cursor = cursor[segment]
        cursor[path[-1]] = parsed_value
    return out


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML file from disk.

    Parameters
    ----------
    path
        Filesystem path to a TOML file.

    Returns
    -------
    dict
        Parsed TOML as a nested dict.
    """
    path = Path(path)
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(
    experiment_toml: str | Path | None = None,
    overrides: Iterable[str] | None = None,
    *,
    validate: bool = True,
) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` with layered defaults.

    Parameters
    ----------
    experiment_toml
        Optional path to an experiment-specific TOML. If ``None``, only
        the defaults (plus any CLI overrides) are used.
    overrides
        Optional iterable of ``"dotted.path=value"`` strings. Applied
        after TOML, so they always win.
    validate
        Whether to run the cross-field validator. Tests may wish to
        skip this to inspect partial configs. Defaults to ``True``.

    Returns
    -------
    ExperimentConfig
        Validated experiment config.
    """
    merged = copy.deepcopy(DEFAULT_CONFIG)
    if experiment_toml is not None:
        toml_data = load_toml(experiment_toml)
        merged = _deep_merge(merged, toml_data)
    if overrides is not None:
        merged = _apply_cli_overrides(merged, overrides)

    cfg = ExperimentConfig.model_validate(merged)
    if validate:
        validate_experiment_config(cfg)
    return cfg


#: Fields whose ``None`` value must be serialised as an empty TOML string
#: (because the schema's validator normalises ``""`` back to ``None`` on load).
#: All other ``None`` fields are dropped from the TOML output; they'll default
#: back to ``None`` via the pydantic schema on reload.
_NONE_AS_EMPTY_STRING: set[tuple[str, ...]] = {
    ("news", "pca_train_end"),
}


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Dump an :class:`ExperimentConfig` to a TOML file.

    Supports the round-trip test (TOML -> model -> TOML). Because TOML
    has no null sentinel, ``None`` values are handled by:

    * fields listed in :data:`_NONE_AS_EMPTY_STRING` — serialised as
      empty string (the schema's field validator restores ``None``);
    * all other ``None`` fields — dropped from the TOML output, allowing
      the pydantic default ``None`` to reassert on reload.
    """
    import tomli_w

    path = Path(path)
    data = cfg.model_dump(mode="python")
    data = _normalise_for_toml(data, path=())
    with path.open("wb") as fh:
        tomli_w.dump(data, fh)


def _normalise_for_toml(obj: Any, path: tuple[str, ...]) -> Any:
    """Recursively strip ``None`` values for TOML-safe output.

    Drops ``None``-valued keys unless the dotted path is listed in
    :data:`_NONE_AS_EMPTY_STRING`, in which case the empty string is
    substituted so the schema's validator can round-trip it.
    """
    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for k, v in obj.items():
            sub_path = path + (k,)
            if v is None:
                if sub_path in _NONE_AS_EMPTY_STRING:
                    result[k] = ""
                # Otherwise skip: defaults reassert on reload.
                continue
            result[k] = _normalise_for_toml(v, sub_path)
        return result
    if isinstance(obj, list):
        return [_normalise_for_toml(x, path) for x in obj]
    return obj


__all__ = ["load_config", "load_toml", "dump_config"]
