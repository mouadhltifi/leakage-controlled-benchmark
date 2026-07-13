"""Cross-field validation rules for :class:`ExperimentConfig`.

Rules per spec Section 3.1 and Section 4.

Rules enforced (error unless noted):

1. Gated cross-attention requires >=2 active modalities AND
   ``fusion.primary_modality`` in active modalities.
2. ``finbert_11dim`` encoder + non-None ``pca_dims`` is inconsistent.
3. Spherical-mean aggregation is undefined for the 11-dim statistical path.
4. ``single_task`` architecture requires exactly one target.
5. Volatility-only target makes ``class_weights`` moot (warning, not error).
6. Dynamic graph sources require the snapshots file (delegated to
   data-assembly time; here we only verify the config is internally
   consistent).
7. ``lookback == 1`` + LSTM price encoder is allowed but warned.
8. ``intraday.enabled == True`` raises ``NotImplementedError`` in v2.
9. ``head.mtl_beta`` is only meaningful when all three targets are
   active.
10. Sequential cascade requires both a classification and a regression
    target to be active.
"""

from __future__ import annotations

import logging
import warnings

from mmfp.config.schema import ExperimentConfig

log = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when cross-field validation fails."""


def _active_modalities(cfg: ExperimentConfig) -> list[str]:
    return cfg.active_modalities()


def _rule_gated_cross_attention(cfg: ExperimentConfig) -> None:
    if cfg.fusion.strategy != "gated_cross_attention":
        return
    active = _active_modalities(cfg)
    if len(active) < 2:
        raise ConfigValidationError(
            "fusion.strategy=gated_cross_attention requires >=2 active "
            f"modalities, got {active}"
        )
    if cfg.fusion.primary_modality not in active:
        raise ConfigValidationError(
            f"fusion.primary_modality={cfg.fusion.primary_modality!r} "
            f"must be one of the active modalities {active}"
        )


def _rule_finbert_11dim_no_pca(cfg: ExperimentConfig) -> None:
    if not cfg.news.enabled:
        return
    if cfg.news.encoder == "finbert_11dim" and cfg.news.pca_dims is not None:
        raise ConfigValidationError(
            "news.encoder='finbert_11dim' expects news.pca_dims=None "
            f"(the 11-dim statistical path has no PCA), got pca_dims={cfg.news.pca_dims}"
        )


def _rule_spherical_requires_vector_embeddings(cfg: ExperimentConfig) -> None:
    if not cfg.news.enabled:
        return
    if (
        cfg.news.aggregation == "spherical_mean"
        and cfg.news.encoder == "finbert_11dim"
    ):
        raise ConfigValidationError(
            "news.aggregation='spherical_mean' is only defined for vector "
            "embedding encoders; not compatible with 'finbert_11dim'"
        )


def _rule_single_task_one_target(cfg: ExperimentConfig) -> None:
    if cfg.head.architecture != "single_task":
        return
    if len(cfg.head.targets) != 1:
        raise ConfigValidationError(
            "head.architecture='single_task' requires exactly one target, "
            f"got targets={cfg.head.targets}"
        )


def _rule_volatility_only_warn_class_weights(cfg: ExperimentConfig) -> None:
    if cfg.head.targets == ["volatility"]:
        if cfg.training.class_weights != "none":
            log.warning(
                "head.targets=['volatility']: training.class_weights=%r will "
                "be ignored (no classification target present)",
                cfg.training.class_weights,
            )


def _rule_lstm_lookback_1(cfg: ExperimentConfig) -> None:
    if cfg.data.lookback == 1 and cfg.model.price_encoder == "lstm":
        warnings.warn(
            "data.lookback=1 with model.price_encoder='lstm': the LSTM will "
            "see a single timestep. Consider price_encoder='feedforward'.",
            UserWarning,
            stacklevel=2,
        )


def _rule_intraday_not_implemented(cfg: ExperimentConfig) -> None:
    if cfg.intraday.enabled:
        raise NotImplementedError(
            "intraday.enabled=true is out of scope for v2. The schema is "
            "provided for forward-compatibility only; the data loader is "
            "not implemented. Set intraday.enabled=false."
        )


def _rule_mtl_beta_requires_three_targets(cfg: ExperimentConfig) -> None:
    if cfg.head.mtl_beta is None:
        return
    if (
        cfg.head.architecture != "parallel_multitask"
        or len(cfg.head.targets) < 3
    ):
        log.warning(
            "head.mtl_beta=%s but head has <3 active targets (targets=%s, "
            "architecture=%s); beta will be ignored.",
            cfg.head.mtl_beta,
            cfg.head.targets,
            cfg.head.architecture,
        )


def _rule_sequential_cascade_targets(cfg: ExperimentConfig) -> None:
    if cfg.head.architecture != "sequential_cascade":
        return
    if "direction" not in cfg.head.targets:
        raise ConfigValidationError(
            "head.architecture='sequential_cascade' requires a 'direction' "
            f"target; got targets={cfg.head.targets}"
        )
    reg_target = cfg.head.cascade_reg_target
    if reg_target not in cfg.head.targets:
        raise ConfigValidationError(
            "head.architecture='sequential_cascade' requires "
            f"head.cascade_reg_target={reg_target!r} to be in head.targets, "
            f"got targets={cfg.head.targets}"
        )


# All rules registered here. Order matters only for error legibility when
# multiple would fail; each rule is independent.
_RULES: list = [
    _rule_gated_cross_attention,
    _rule_finbert_11dim_no_pca,
    _rule_spherical_requires_vector_embeddings,
    _rule_single_task_one_target,
    _rule_volatility_only_warn_class_weights,
    _rule_lstm_lookback_1,
    _rule_intraday_not_implemented,
    _rule_mtl_beta_requires_three_targets,
    _rule_sequential_cascade_targets,
]


def validate_experiment_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Run all cross-field validation rules.

    Returns the config unchanged on success to support a fluent-style
    ``validate_experiment_config(load_config(...))`` call site. Raises
    :class:`ConfigValidationError` (or ``NotImplementedError`` for
    intraday) on failure.
    """
    for rule in _RULES:
        rule(cfg)
    return cfg


__all__ = [
    "ConfigValidationError",
    "validate_experiment_config",
]
