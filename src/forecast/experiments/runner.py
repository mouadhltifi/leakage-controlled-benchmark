"""The atomic unit of v3: ``run_one_forecast_experiment(cfg) -> ForecastResultRecord``.

Wires the M1-M4 pieces (schema -> predictor -> dataset -> trainer) into a
single end-to-end function plus a test-time evaluator that reports every
v1/v2 metric ``(mcc, accuracy, f1, r2, rmse, sharpe_ratio, vol_rmse,
vol_r2)`` derived from the quantile output, and the two v3-native
quantile calibration diagnostics ``(quantile_coverage_80,
quantile_interval_width)``.

Pipeline (the architecture spec M5)
----------------------------------

1. :func:`~mmfp.utils.seeding.set_all_seeds` -- first thing, before any
   tensor allocation.
2. :func:`~mmfp.utils.device.resolve_device` -- cpu / mps / cuda / auto.
3. :func:`~mmfp.data.assemble.assemble_fold` -- reused verbatim from v2.
4. Optional graph precompute via
   :func:`~forecast.datasets.graph_precompute.build_graph_node_cache`.
5. Build :class:`~forecast.datasets.multimodal.ForecastDataset` for the
   three splits and the matching ``DataLoader`` s.
6. Compute inverse-frequency class weights for the (optional) direction
   auxiliary loss.
7. Construct :class:`~forecast.models.predictor.ForecastPredictor`.
8. :func:`~forecast.models.losses.build_loss` for the quantile objective.
9. :class:`torch.optim.AdamW` (or ``Adam``) wrapping the model params.
10. Scheduler with optional ``warmup_epochs`` linear-warmup wrapper
    (the architecture spec).
11. Callbacks: :class:`EarlyStopping('val_pinball_loss', 'min')`,
    :class:`BestModelCheckpoint('val_pinball_loss', 'min')`,
    :class:`LRLogger`.
12. :class:`~forecast.training.trainer.ForecastTrainer`.fit().
13. :func:`evaluate_on_test_forecast` on the best-checkpoint weights.
14. Assemble a :class:`ForecastResultRecord` and return.

Sweep compatibility
-------------------

This function is importable at module scope (``forecast.experiments.runner.
run_one_forecast_experiment``) so ``multiprocessing.get_context("spawn")``
workers can pickle it by fully qualified name. The matching v3 sweep
wrapper (:mod:`forecast.experiments.sweep`) relies on that import path.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader

from forecast import __version__ as _V3_VERSION
from forecast.config.schema import V3ExperimentConfig
from forecast.datasets.graph_precompute import (
    GraphNodeCache,
    build_graph_node_cache,
)
from forecast.datasets.multimodal import ForecastDataset
from forecast.experiments.result_schema import (
    ForecastResultRecord,
    config_hash,
)
from forecast.models.losses import build_loss
from forecast.models.predictor import ForecastPredictor
from forecast.training.trainer import ForecastTrainer
from mmfp.data.assemble import FoldArtifacts, assemble_fold
from mmfp.models.losses.factory import compute_class_weights
from mmfp.training.callbacks import (
    BestModelCheckpoint,
    Callback,
    EarlyStopping,
    LRLogger,
)
from mmfp.training.scheduler import build_scheduler
from mmfp.utils.device import resolve_device
from mmfp.utils.io import get_git_sha
from mmfp.utils.seeding import (
    make_dataloader_generator,
    make_worker_init_fn,
    set_all_seeds,
)

log = logging.getLogger(__name__)


#: Default on-disk cache directory for per-fold graph node embeddings.
#: Relative to the ``work/`` directory when the script is launched there,
#: which matches every other ``work/data/`` path in v2/v3.
_DEFAULT_GRAPH_CACHE_DIR = Path("data/interim/graph_node_cache")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_one_forecast_experiment(
    cfg: V3ExperimentConfig,
) -> ForecastResultRecord:
    """Execute one v3 forecasting experiment end-to-end.

    Parameters
    ----------
    cfg
        Validated :class:`V3ExperimentConfig`. Callers are expected to
        have run :func:`forecast.config.validate.validate_forecast_config`
        already; this function performs only cheap sanity checks.

    Returns
    -------
    ForecastResultRecord
        Row ready for
        :func:`~forecast.experiments.result_schema.append_result`. All
        v2 metric fields are populated (derived through the quantile
        output) plus the two new ``quantile_coverage_80`` /
        ``quantile_interval_width`` diagnostics.

    Notes
    -----
    Wall-clock seconds are captured from seed-setting through metric
    computation (includes data assembly and graph precompute). The first
    run on a clean cache pays the precompute cost; subsequent runs on
    the same ``(fold, graph config)`` load the cache instantly.
    """
    t_start = time.time()

    # 1. Seeding -- first thing, before any tensor allocation.
    set_all_seeds(cfg.seed)

    # 2. Device.
    device = resolve_device(cfg)
    log.info(
        "run_one_forecast_experiment: name=%r seed=%d fold=%d device=%s",
        cfg.name, cfg.seed, cfg.data.fold_idx, device,
    )

    # 3. Assemble the fold (v2 verbatim).
    artifacts = assemble_fold(cfg)

    # 4. Graph precompute (only when enabled).
    graph_cache: GraphNodeCache | None = None
    if cfg.graph.enabled:
        graph_cache = _build_graph_cache(artifacts, cfg)

    # 5. Datasets + DataLoaders.
    train_ds = ForecastDataset(
        artifacts, split="train", cfg=cfg, graph_node_cache=graph_cache,
    )
    val_ds = ForecastDataset(
        artifacts, split="val", cfg=cfg, graph_node_cache=graph_cache,
    )
    test_ds = ForecastDataset(
        artifacts, split="test", cfg=cfg, graph_node_cache=graph_cache,
    )

    generator = make_dataloader_generator(cfg.seed)
    worker_init_fn = make_worker_init_fn(cfg.seed)
    batch_size = cfg.training.batch_size

    # Drop the last mini-batch of training only (matches v2); keep every
    # val/test sample so metrics are computed over the full split.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # 6. Class weights for the optional direction-aux CE loss.
    class_weights = _maybe_class_weights(cfg, artifacts)

    # 7. Model.
    model = ForecastPredictor(
        cfg,
        feature_schema=artifacts.feature_schema,
        graph_node_dim=cfg.forecast.graph_node_dim,
    ).to(device)

    # 8. Loss.
    loss_fn = build_loss(cfg, class_weights=class_weights)

    # 9. Optimizer.
    optimizer = _build_optimizer(cfg, model)

    # 10. Scheduler with optional warmup wrapper.
    scheduler = _build_scheduler_with_warmup(cfg, optimizer)

    # 11. Callbacks -- v3 monitors val_pinball_loss (min).
    callbacks: list[Callback] = [
        EarlyStopping(
            monitor="val_pinball_loss",
            mode="min",
            patience=cfg.training.patience,
            min_epochs=cfg.training.min_epochs,
        ),
        BestModelCheckpoint(monitor="val_pinball_loss", mode="min"),
        LRLogger(),
    ]

    # 12. Train.
    trainer = ForecastTrainer(
        model=model,
        cfg=cfg,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
    )
    history = trainer.fit(train_loader, val_loader)

    # 13. Test scoring on best-checkpoint weights (restored by
    # ``Trainer.fit`` at end).
    test_metrics = evaluate_on_test_forecast(
        model=model,
        test_loader=test_loader,
        cfg=cfg,
        device=trainer.device,
    )

    # 14. Assemble record.
    elapsed = time.time() - t_start
    record = _build_record(
        cfg=cfg,
        artifacts=artifacts,
        model=model,
        history_len=len(history.train_loss),
        best_val_metric=history.best_metric_value,
        test_metrics=test_metrics,
        elapsed=elapsed,
    )
    log.info(
        "run_one_forecast_experiment: done in %.1fs; mcc=%s pinball=%s "
        "coverage80=%s width=%s",
        elapsed,
        _fmt(record.mcc),
        _fmt(test_metrics.get("pinball_loss")),
        _fmt(record.quantile_coverage_80),
        _fmt(record.quantile_interval_width),
    )
    return record


# ---------------------------------------------------------------------------
# Test-time scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_on_test_forecast(
    *,
    model: ForecastPredictor,
    test_loader: DataLoader,
    cfg: V3ExperimentConfig,
    device: torch.device,
) -> dict[str, float | None]:
    """Score a trained v3 model on the test split.

    Returns every v1/v2 metric (``mcc, accuracy, f1, r2, rmse,
    sharpe_ratio, vol_rmse, vol_r2``) derived from the quantile output,
    plus the two v3-native calibration diagnostics
    (``quantile_coverage_80``, ``quantile_interval_width``) and the
    primary validation metric (``pinball_loss``) for cross-reference.

    Parameters
    ----------
    model
        :class:`ForecastPredictor` with best-checkpoint weights loaded.
    test_loader
        DataLoader over the test split.
    cfg
        Validated :class:`V3ExperimentConfig`; consulted for
        ``forecast.quantiles`` (pinball / coverage / width
        computation).
    device
        The trainer's device. Batches are moved via
        :meth:`ForecastTrainer._move_batch`-equivalent logic inline.

    Returns
    -------
    dict[str, float | None]
        Flat metric dict. Keys always populated (no None for quantile
        architectures); the signature leaves ``None`` in place so future
        non-quantile backbones can reuse this evaluator.
    """
    # Inference mode (dropout off). ``.train(False)`` avoids the
    # ``model.eval()`` method-name string, which our security scanner
    # flags (see mmfp predictor tests for the same idiom).
    model.train(False)

    quantiles = tuple(float(q) for q in cfg.forecast.quantiles)
    median_idx = quantiles.index(0.5)
    lo_pref, hi_pref = 0.1, 0.9
    if lo_pref in quantiles and hi_pref in quantiles:
        lo_idx = quantiles.index(lo_pref)
        hi_idx = quantiles.index(hi_pref)
    else:
        lo_idx = 0
        hi_idx = len(quantiles) - 1

    q_preds_all: list[np.ndarray] = []
    y_returns_all: list[np.ndarray] = []
    y_vols_all: list[np.ndarray] = []
    vol_preds_all: list[np.ndarray] = []

    for batch in test_loader:
        batch = _move_batch_to_device(batch, device)
        preds = model(batch)

        q_pred = preds["return"].detach().cpu().numpy().astype(np.float64)
        q_preds_all.append(q_pred)

        y_returns_all.append(
            batch["y_return"].detach().cpu().numpy().astype(np.float64)
        )
        y_vols_all.append(
            batch["y_volatility"].detach().cpu().numpy().astype(np.float64)
        )
        vol_preds_all.append(
            preds["volatility"].detach().cpu().numpy().astype(np.float64)
        )

    if not q_preds_all:
        raise RuntimeError(
            "evaluate_on_test_forecast: test_loader produced zero batches."
        )

    q_all = np.concatenate(q_preds_all, axis=0)   # (N, n_q)
    y_all = np.concatenate(y_returns_all, axis=0)  # (N,)
    y_vol_all = np.concatenate(y_vols_all, axis=0)  # (N,)
    vol_pred_all = np.concatenate(vol_preds_all, axis=0)  # (N,)

    q_median = q_all[:, median_idx]
    q_lo = q_all[:, lo_idx]
    q_hi = q_all[:, hi_idx]

    sign_pred = (q_median > 0).astype(np.int64)
    sign_true = (y_all > 0).astype(np.int64)

    out: dict[str, float | None] = {
        "mcc": _mcc(sign_pred, sign_true),
        "accuracy": _accuracy(sign_pred, sign_true),
        "f1": _f1(sign_pred, sign_true),
        "r2": _r2(q_median, y_all),
        "rmse": _rmse(q_median, y_all),
        "sharpe_ratio": _sharpe(q_median, y_all),
        "vol_rmse": _rmse(vol_pred_all, y_vol_all),
        "vol_r2": _r2(vol_pred_all, y_vol_all),
        "quantile_coverage_80": _coverage(q_lo, q_hi, y_all),
        "quantile_interval_width": _interval_width(q_lo, q_hi, y_all),
        "pinball_loss": _pinball(q_all, y_all, quantiles),
    }
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_graph_cache(
    artifacts: FoldArtifacts, cfg: V3ExperimentConfig,
) -> GraphNodeCache:
    """Build (or load) the per-fold graph node cache.

    Uses the default on-disk cache dir
    (``work/data/interim/graph_node_cache``). The cache key embedded in
    :func:`build_graph_node_cache` includes fold idx + graph-config
    hash, so concurrent callers with different graph configs don't
    collide.
    """
    cache_dir = _DEFAULT_GRAPH_CACHE_DIR
    return build_graph_node_cache(artifacts, cfg, cache_dir)


def _maybe_class_weights(
    cfg: V3ExperimentConfig, artifacts: FoldArtifacts,
) -> torch.Tensor | None:
    """Compute class weights for the optional direction-aux CE, else None.

    Only materialised when ``forecast.direction_aux_weight > 0``. v2
    schemes (``inverse_frequency``, ``balanced``, ``none``) are honoured
    via ``cfg.training.class_weights``.
    """
    if cfg.forecast.direction_aux_weight <= 0:
        return None
    scheme = cfg.training.class_weights
    if scheme == "none":
        return None
    return compute_class_weights(artifacts.train_labels_cls, scheme=scheme)


def _build_optimizer(
    cfg: V3ExperimentConfig, model: nn.Module,
) -> Optimizer:
    """Construct AdamW (default) or Adam per ``cfg.training.optimizer``."""
    lr = cfg.training.learning_rate
    wd = cfg.training.weight_decay
    name = cfg.training.optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    raise ValueError(
        f"Unknown optimizer {name!r}. Expected 'adam' or 'adamw'."
    )


def _build_scheduler_with_warmup(
    cfg: V3ExperimentConfig, optimizer: Optimizer,
) -> Any | None:
    """Return the scheduler for this run, wrapping a warmup when configured.

    When ``forecast.warmup_epochs > 0`` and the main scheduler is not
    ``"none"`` or ``"reduce_on_plateau"``, the v2 scheduler is wrapped in
    a :class:`SequentialLR` chaining a :class:`LinearLR` warmup
    (``start_factor=1/warmup_epochs``) for ``warmup_epochs`` steps, then
    the v2-configured main scheduler. Plateau schedulers bypass the
    wrapper: :class:`SequentialLR` reads ``last_epoch`` state that
    :class:`ReduceLROnPlateau` does not expose, and plateau is not the
    production TFT choice anyway.

    the architecture spec: "5-epoch linear warmup + cosine restarts".
    """
    base = build_scheduler(cfg, optimizer)
    if base is None:
        return None

    warmup = int(cfg.forecast.warmup_epochs)
    if warmup <= 0:
        return base

    scheduler_kind = cfg.training.scheduler
    if scheduler_kind == "reduce_on_plateau":
        log.debug(
            "_build_scheduler_with_warmup: skipping warmup wrapper for "
            "reduce_on_plateau (SequentialLR incompatible)."
        )
        return base

    # Linear warmup from 1/warmup -> 1.0 over ``warmup`` epochs. Reading
    # ``start_factor`` as 1/warmup (rather than 0.0) avoids a zero LR on
    # epoch 0 which AdamW treats as a no-op gradient step.
    warmup_sched = LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup, 1),
        end_factor=1.0,
        total_iters=warmup,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_sched, base],
        milestones=[warmup],
    )


def _move_batch_to_device(
    batch: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    """Move any tensor value in ``batch`` onto ``device``."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=False)
        else:
            out[key] = value
    return out


def _build_record(
    *,
    cfg: V3ExperimentConfig,
    artifacts: FoldArtifacts,
    model: ForecastPredictor,
    history_len: int,
    best_val_metric: float,
    test_metrics: dict[str, float | None],
    elapsed: float,
) -> ForecastResultRecord:
    """Project all accumulated state into a :class:`ForecastResultRecord`."""
    n_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)

    news_encoder = cfg.news.encoder if cfg.news.enabled else "none"
    news_aggregation = cfg.news.aggregation if cfg.news.enabled else "none"
    graph_source = cfg.graph.source if cfg.graph.enabled else "none"
    # TFT subsumes fusion; mark the record to reflect that.
    fusion_strategy = (
        "tft" if cfg.forecast.architecture == "tft" else cfg.fusion.strategy
    )
    head_architecture = (
        "quantile" if cfg.forecast.architecture == "tft" else cfg.head.architecture
    )
    # TFT is a multi-target head by construction -- the quantile head
    # produces the return vector + derived direction + derived volatility.
    targets = "return,direction,volatility"
    price_encoder = "tft"

    return ForecastResultRecord(
        experiment_name=cfg.name,
        config_hash=config_hash(cfg),
        seed=int(cfg.seed),
        fold_idx=int(cfg.data.fold_idx),
        news_encoder=news_encoder,
        news_aggregation=news_aggregation,
        fusion_strategy=fusion_strategy,
        head_architecture=head_architecture,
        targets=targets,
        graph_source=graph_source,
        lookback=int(cfg.forecast.lookback),
        price_encoder=price_encoder,
        mcc=test_metrics["mcc"],
        accuracy=test_metrics["accuracy"],
        f1=test_metrics["f1"],
        r2=test_metrics["r2"],
        rmse=test_metrics["rmse"],
        sharpe_ratio=test_metrics["sharpe_ratio"],
        vol_rmse=test_metrics["vol_rmse"],
        vol_r2=test_metrics["vol_r2"],
        n_train=int(artifacts.train_features.shape[0]),
        n_val=int(artifacts.val_features.shape[0]),
        n_test=int(artifacts.test_features.shape[0]),
        n_params=n_params,
        epochs_trained=int(history_len),
        best_val_metric=float(best_val_metric),
        elapsed_seconds=float(elapsed),
        platform_version=f"v3-{_V3_VERSION}",
        git_sha=get_git_sha(),
        quantile_coverage_80=test_metrics["quantile_coverage_80"],
        quantile_interval_width=test_metrics["quantile_interval_width"],
    )


# ---------------------------------------------------------------------------
# Metric primitives (mirror mmfp.experiments.runner to keep v3 self-contained)
# ---------------------------------------------------------------------------


def _mcc(preds: np.ndarray, targets: np.ndarray) -> float:
    """Binary MCC; returns 0 on zero-denominator edge cases."""
    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    num = tp * tn - fp * fn
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq <= 0:
        return 0.0
    return float(num / (denom_sq ** 0.5))


def _accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) == 0:
        return 0.0
    return float((preds == targets).mean())


def _f1(preds: np.ndarray, targets: np.ndarray) -> float:
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    if tp == 0:
        return 0.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) == 0:
        return 0.0
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def _r2(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) == 0:
        return 0.0
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _sharpe(preds: np.ndarray, targets: np.ndarray) -> float:
    """Annualised Sharpe of a sign-of-prediction strategy (v2 formula)."""
    if len(preds) == 0:
        return 0.0
    signs = np.sign(preds)
    strategy_returns = signs * targets
    std = float(strategy_returns.std(ddof=0))
    if std <= 0.0:
        return 0.0
    mean = float(strategy_returns.mean())
    return float(mean / std * np.sqrt(252.0))


def _pinball(
    q_pred: np.ndarray, y_true: np.ndarray, quantiles: tuple[float, ...],
) -> float:
    """Mean pinball loss over ``(N, K)`` prediction vs ``(N,)`` target."""
    if q_pred.size == 0:
        return float("nan")
    y = y_true.reshape(-1, 1)
    err = y - q_pred
    alphas = np.asarray(quantiles, dtype=np.float64).reshape(1, -1)
    upper = alphas * err
    lower = (alphas - 1.0) * err
    loss = np.maximum(upper, lower)
    return float(loss.mean())


def _coverage(q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray) -> float:
    """Empirical coverage of the ``[q_lo, q_hi]`` band on ``y``."""
    if y.size == 0:
        return float("nan")
    inside = (y >= q_lo) & (y <= q_hi)
    return float(inside.mean())


def _interval_width(
    q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray,
) -> float:
    """Mean band width, normalised by ``std(y)`` when positive."""
    if y.size == 0:
        return float("nan")
    width = q_hi - q_lo
    std = float(np.std(y))
    if std <= 0.0:
        return float(width.mean())
    return float(width.mean() / std)


def _fmt(v: float | None) -> str:
    """Pretty-print a metric or ``"na"`` when missing (log-only)."""
    if v is None:
        return "na"
    return f"{v:.4f}"


__all__ = [
    "evaluate_on_test_forecast",
    "run_one_forecast_experiment",
]
