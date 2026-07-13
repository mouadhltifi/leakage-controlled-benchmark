"""Pydantic v2 schema for ``ExperimentConfig`` and its nested models.

Every experimental knob is declared here with a type, a default, and
constraints. Cross-field validation lives in :mod:`mmfp.config.validate`
to keep single-field and cross-field concerns separate.

See the design spec for the authoritative
specification and section 4 for the canonical TOML layout.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Base model: shared configuration across all sub-configs.
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Base for all config models.

    * ``extra='forbid'`` — unknown keys raise rather than being silently
      dropped (catches typos and drifting schemas).
    * ``validate_assignment=True`` — mutation after construction is
      re-validated. Keeps invariants even when callers edit cfg in place.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# ---------------------------------------------------------------------------
# Per-modality configs.
# ---------------------------------------------------------------------------


class DataConfig(_StrictModel):
    """Universe, date range, fold index, lookback window.

    ``fold_idx`` selects one of five expanding-window folds (0..4).
    ``lookback`` is the number of trading days fed to sequence encoders;
    ``lookback=1`` implies a feedforward price encoder.
    """

    universe: Literal["sp500_sector_55"] = "sp500_sector_55"
    start_date: str = "2015-01-01"
    end_date: str = "2023-12-31"
    fold_idx: int = Field(default=0, ge=0, le=4)
    lookback: int = Field(default=1, ge=1, le=60)
    deadzone: float = Field(default=0.005, ge=0.0)
    warmup_days: int = Field(default=252, ge=0)

    @field_validator("start_date", "end_date")
    @classmethod
    def _valid_iso_date(cls, v: str) -> str:
        import datetime as _dt

        try:
            _dt.date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(
                f"Expected ISO date YYYY-MM-DD, got {v!r}"
            ) from exc
        return v


class PriceConfig(_StrictModel):
    """Price modality flag.

    Price is always-on in the platform; this config exists so the
    ``ExperimentConfig`` shape is uniform across modalities and a future
    experiment can in principle ablate it. The field is retained as a
    boolean for symmetry with :class:`NewsConfig` etc.
    """

    enabled: bool = True


class NewsConfig(_StrictModel):
    """News modality: encoder, aggregation, PCA, empty-day policy."""

    enabled: bool = False
    encoder: Literal[
        "finbert_11dim",
        "finbert_cls_768",
        "qwen3_embedding",
        "deberta_v3_financial",
        "bge_base",
    ] = "finbert_11dim"
    aggregation: Literal[
        "arithmetic_mean",
        "spherical_mean",
        "attention_weighted",
        "recency_weighted",
        "max_sentiment",
    ] = "arithmetic_mean"
    pca_dims: int | None = Field(default=32, ge=1)
    pca_train_end: str | None = None
    log1p_volume: bool = True
    dispersion_feature: bool = False
    empty_day_policy: Literal[
        "zero_fill_has_news_flag",
        "sentinel_vector",
    ] = "zero_fill_has_news_flag"
    lag_days: int = Field(default=1, ge=0)

    @field_validator("pca_train_end")
    @classmethod
    def _normalize_pca_train_end(cls, v: str | None) -> str | None:
        """Normalise empty string to ``None`` (TOML lacks a true null)."""
        if v is None or v == "":
            return None
        import datetime as _dt

        try:
            _dt.date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(
                f"pca_train_end must be ISO date or empty, got {v!r}"
            ) from exc
        return v


class SocialConfig(_StrictModel):
    """Social media modality (StockTwits)."""

    enabled: bool = False
    aggregation_window: int = Field(default=3, ge=1)
    log1p_volume: bool = True
    lag_days: int = Field(default=1, ge=0)


class MacroConfig(_StrictModel):
    """Macro modality (FRED indicators + FOMC windows)."""

    enabled: bool = False
    fomc_window: int = Field(default=1, ge=0)


class GraphConfig(_StrictModel):
    """Graph modality: GICS static, dynamic correlation, or both."""

    enabled: bool = False
    source: Literal[
        "static_gics",
        "dynamic_corr",
        "static_plus_dynamic",
    ] = "static_gics"
    dynamic_window: int = Field(default=20, ge=1)
    dynamic_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    dynamic_refresh_every: int = Field(default=20, ge=1)


class StandardizationConfig(_StrictModel):
    """Input standardisation per modality. Always on."""

    price: bool = True
    macro: bool = True
    news: bool = True
    social: bool = True
    graph_node: bool = True
    fit_scope: Literal["train_fold_only"] = "train_fold_only"


class ModelConfig(_StrictModel):
    """Model architectural knobs shared across encoders.

    ``dropout`` is the single dropout rate used everywhere in the model.
    ``num_heads`` is the single head count used by attention-based fusion.
    """

    hidden_dim: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    num_heads: int = Field(default=4, ge=1)
    price_encoder: Literal["lstm", "feedforward"] = "lstm"
    price_lstm_layers: int = Field(default=2, ge=1)
    tabular_hidden_layers: int = Field(default=1, ge=1)
    graph_gat_heads: int = Field(default=4, ge=1)
    graph_gat_layers: int = Field(default=2, ge=1)


class FusionConfig(_StrictModel):
    """Fusion strategy selection."""

    strategy: Literal[
        "concat",
        "gated_cross_attention",
        "multihead_attention",
    ] = "concat"
    primary_modality: str = "price"


class HeadConfig(_StrictModel):
    """Prediction head architecture, targets, and multi-task weights."""

    architecture: Literal[
        "parallel_multitask",
        "sequential_cascade",
        "single_task",
    ] = "parallel_multitask"
    targets: list[Literal["direction", "return", "volatility"]] = Field(
        default_factory=lambda: ["direction", "return"],
    )
    mtl_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    mtl_beta: float | None = Field(default=None)
    cascade_reg_target: Literal["return", "volatility"] = "return"
    detach_cascade: bool = False

    @field_validator("targets")
    @classmethod
    def _nonempty_unique_targets(
        cls, v: list[str]
    ) -> list[Literal["direction", "return", "volatility"]]:
        if len(v) == 0:
            raise ValueError("head.targets must contain at least one target")
        if len(set(v)) != len(v):
            raise ValueError(
                f"head.targets contains duplicates: {v}"
            )
        return v  # type: ignore[return-value]

    @field_validator("mtl_beta")
    @classmethod
    def _beta_range(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"mtl_beta must be in [0, 1], got {v}")
        return v


class TrainingConfig(_StrictModel):
    """Optimisation, scheduling, and class-balancing knobs."""

    batch_size: int = Field(default=64, ge=1)
    max_epochs: int = Field(default=150, ge=1)
    min_epochs: int = Field(default=0, ge=0)
    patience: int = Field(default=30, ge=1)
    grad_clip: float = Field(default=1.0, gt=0.0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    optimizer: Literal["adam", "adamw"] = "adam"
    scheduler: Literal[
        "cosine_warm_restarts",
        "reduce_on_plateau",
        "none",
    ] = "cosine_warm_restarts"
    class_weights: Literal["inverse_frequency", "balanced", "none"] = (
        "inverse_frequency"
    )
    val_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)

    @field_validator("max_epochs")
    @classmethod
    def _max_ge_min(cls, v: int, info) -> int:
        # min_epochs has already been validated when we reach this.
        min_epochs = info.data.get("min_epochs", 0)
        if v < min_epochs:
            raise ValueError(
                f"max_epochs ({v}) must be >= min_epochs ({min_epochs})"
            )
        return v


class IntradayConfig(_StrictModel):
    """Intraday modality stub — platform-level only; no loader in v2.

    The schema exists so ``ExperimentConfig`` has a stable shape when
    intraday data becomes available. See spec Section 9 non-goal #3.
    """

    enabled: bool = False
    frequency: Literal["daily", "15min", "5min", "1min"] = "daily"
    source: str = "none"


class LoggingConfig(_StrictModel):
    """Structured logging knobs."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    per_epoch_log_every: int = Field(default=10, ge=1)


# ---------------------------------------------------------------------------
# Top-level ExperimentConfig.
# ---------------------------------------------------------------------------


class ExperimentConfig(_StrictModel):
    """Root config: identification, device, seeds, and all modality subconfigs.

    Cross-field validation is performed by
    :func:`mmfp.config.validate.validate_experiment_config` which should be
    called immediately after loading. It is not embedded as a model-level
    validator because the rules depend on combinations of sibling subconfigs
    and are clearer as a separate, explicitly-invoked pass.
    """

    name: str
    seed: int
    device: Literal["mps", "cuda", "cpu", "auto"] = "auto"

    data: DataConfig = Field(default_factory=DataConfig)
    price: PriceConfig = Field(default_factory=PriceConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    social: SocialConfig = Field(default_factory=SocialConfig)
    macro: MacroConfig = Field(default_factory=MacroConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    standardization: StandardizationConfig = Field(
        default_factory=StandardizationConfig
    )
    model: ModelConfig = Field(default_factory=ModelConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    head: HeadConfig = Field(default_factory=HeadConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    intraday: IntradayConfig = Field(default_factory=IntradayConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def active_modalities(self) -> list[str]:
        """Return names of currently active modalities (for fusion wiring).

        Always includes ``price``; includes other modalities iff their
        subconfig ``enabled`` flag is ``True``.
        """
        mods = ["price"] if self.price.enabled else []
        if self.news.enabled:
            mods.append("news")
        if self.social.enabled:
            mods.append("social")
        if self.macro.enabled:
            mods.append("macro")
        if self.graph.enabled:
            mods.append("graph")
        return mods
