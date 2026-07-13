"""The atomic unit of the platform: ``run_one_experiment(cfg) -> ResultRecord``.

This module wires M1-M6 together in a single, deterministic,
test-covered function. Sweeps are thin wrappers over it (see
:mod:`mmfp.experiments.sweep`).

Pipeline (spec Section 3.12)
----------------------------

1. Seed every RNG (:func:`~mmfp.utils.seeding.set_all_seeds`).
2. Resolve the compute device (:func:`~mmfp.utils.device.resolve_device`).
3. Assemble the fold (:func:`~mmfp.data.assemble.assemble_fold`).
4. Build train/val/test :class:`~mmfp.datasets.multimodal.MultiModalDataset`
   and their DataLoaders, with seeded generator and worker init.
5. Compute classification class weights from the train split labels.
6. Construct :class:`~mmfp.models.predictor.Predictor`,
   :func:`~mmfp.models.losses.factory.build_loss`, optimizer,
   scheduler, and callbacks.
7. Instantiate :class:`~mmfp.training.trainer.Trainer` and fit.
8. Score on the test loader via :func:`evaluate_on_test`.
9. Assemble a :class:`~mmfp.experiments.result_schema.ResultRecord` and
   return.

All heavy lifting happens downstream; this module is plumbing plus a
strictly-typed record construction.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmfp import __version__ as _PLATFORM_VERSION
from mmfp.config.schema import ExperimentConfig
from mmfp.data.assemble import FoldArtifacts, assemble_fold
from mmfp.datasets.collate import make_collate_fn
from mmfp.datasets.multimodal import MultiModalDataset
from mmfp.experiments.result_schema import ResultRecord, config_hash
from mmfp.models.losses.factory import build_loss, compute_class_weights
from mmfp.models.predictor import Predictor
from mmfp.training.callbacks import (
    BestModelCheckpoint,
    Callback,
    EarlyStopping,
    LRLogger,
)
from mmfp.training.scheduler import build_scheduler
from mmfp.training.trainer import Trainer
from mmfp.utils.device import resolve_device
from mmfp.utils.io import get_git_sha
from mmfp.utils.seeding import (
    make_dataloader_generator,
    make_worker_init_fn,
    set_all_seeds,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_one_experiment(cfg: ExperimentConfig) -> ResultRecord:
    """Execute one experiment end-to-end and return a :class:`ResultRecord`.

    Parameters
    ----------
    cfg
        Validated :class:`ExperimentConfig`. Callers are expected to have
        run :func:`mmfp.config.validate.validate_experiment_config`
        already; this function only performs cheap sanity checks.

    Returns
    -------
    ResultRecord
        Row ready for :func:`~mmfp.experiments.result_schema.append_result`.

    Notes
    -----
    Wall-clock seconds are captured from seed-setting through metric
    computation. That includes data assembly time; sweep callers may
    see longer elapsed on the first run due to parquet reads.
    """
    t_start = time.time()

    # 1. Seeding — must be the first thing, before any Tensor allocation.
    set_all_seeds(cfg.seed)

    # 2. Device.
    device = resolve_device(cfg)
    log.info(
        "run_one_experiment: name=%r seed=%d fold=%d device=%s",
        cfg.name, cfg.seed, cfg.data.fold_idx, device,
    )

    # 3. Assemble the fold.
    artifacts = assemble_fold(cfg)

    # 4. Datasets + DataLoaders.
    train_ds = MultiModalDataset(artifacts, split="train", cfg=cfg)
    val_ds = MultiModalDataset(artifacts, split="val", cfg=cfg)
    test_ds = MultiModalDataset(artifacts, split="test", cfg=cfg)

    collate_fn = make_collate_fn(cfg)
    generator = make_dataloader_generator(cfg.seed)
    worker_init_fn = make_worker_init_fn(cfg.seed)
    batch_size = cfg.training.batch_size

    # Drop the last mini-batch of training only — keeps batch-norm-like
    # semantics clean and matches v1 runner conventions. Val/test keep
    # every sample so metrics are computed over the full split.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=0,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 5. Class weights for direction targets only.
    class_weights = _maybe_class_weights(cfg, artifacts)

    # 6. Model.
    model = Predictor(cfg, artifacts.feature_schema)

    # 7. Loss.
    loss_fn = build_loss(cfg, class_weights=class_weights)

    # 8. Optimizer.
    optimizer = _build_optimizer(cfg, model)

    # 9. Scheduler.
    scheduler = build_scheduler(cfg, optimizer)

    # 10. Callbacks. Monitor follows Trainer.primary_metric() so the
    # stopping target matches the metric the trainer already reports.
    primary_metric, mode = _resolve_primary_metric(cfg)
    callbacks: list[Callback] = [
        EarlyStopping(
            monitor=primary_metric,
            mode=mode,
            patience=cfg.training.patience,
            min_epochs=cfg.training.min_epochs,
        ),
        BestModelCheckpoint(monitor=primary_metric, mode=mode),
        LRLogger(),
    ]

    # 11. Train.
    trainer = Trainer(
        model=model,
        cfg=cfg,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
    )
    history = trainer.fit(train_loader, val_loader)

    # 12. Test scoring on the best-checkpoint weights (restored by
    # Trainer.fit at end).
    test_metrics = evaluate_on_test(
        model=model, test_loader=test_loader, cfg=cfg, device=trainer.device,
    )

    # 13. Record.
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
        "run_one_experiment: done in %.1fs; mcc=%s r2=%s vol_rmse=%s",
        elapsed,
        _fmt(record.mcc),
        _fmt(record.r2),
        _fmt(record.vol_rmse),
    )
    return record


# ---------------------------------------------------------------------------
# Test scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_on_test(
    *,
    model: Predictor,
    test_loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, float | None]:
    """Score a trained model on the test split for every active target.

    Returns
    -------
    dict[str, float | None]
        Keys ``{"mcc", "accuracy", "f1", "r2", "rmse", "sharpe_ratio",
        "vol_rmse", "vol_r2"}``. Each value is a float if the
        corresponding target is active in ``cfg.head.targets``, else
        ``None``.
    """
    model.eval()

    targets_active = set(cfg.head.targets)
    do_direction = "direction" in targets_active
    do_return = "return" in targets_active
    do_volatility = "volatility" in targets_active

    cls_preds: list[np.ndarray] = []
    cls_targets: list[np.ndarray] = []
    reg_preds: list[np.ndarray] = []
    reg_targets: list[np.ndarray] = []
    vol_preds: list[np.ndarray] = []
    vol_targets: list[np.ndarray] = []

    for batch in test_loader:
        batch = _move_batch_to_device(batch, device)
        preds = model(batch)

        if do_direction:
            logits = preds["direction"].detach().cpu().numpy()
            cls_preds.append(np.asarray(logits.argmax(axis=-1), dtype=np.int64))
            cls_targets.append(
                batch["cls_target"].detach().cpu().numpy().astype(np.int64)
            )
        if do_return:
            reg_preds.append(
                preds["return"].squeeze(-1).detach().cpu().numpy().astype(np.float64)
            )
            reg_targets.append(
                batch["reg_target"].detach().cpu().numpy().astype(np.float64)
            )
        if do_volatility:
            vol_preds.append(
                preds["volatility"].squeeze(-1).detach().cpu().numpy().astype(np.float64)
            )
            vol_targets.append(
                batch["vol_target"].detach().cpu().numpy().astype(np.float64)
            )

    out: dict[str, float | None] = {
        "mcc": None,
        "accuracy": None,
        "f1": None,
        "r2": None,
        "rmse": None,
        "sharpe_ratio": None,
        "vol_rmse": None,
        "vol_r2": None,
    }

    if do_direction and cls_preds:
        p = np.concatenate(cls_preds)
        t = np.concatenate(cls_targets)
        out["mcc"] = _mcc(p, t)
        out["accuracy"] = _accuracy(p, t)
        out["f1"] = _f1(p, t)

    if do_return and reg_preds:
        p = np.concatenate(reg_preds)
        t = np.concatenate(reg_targets)
        out["r2"] = _r2(p, t)
        out["rmse"] = _rmse(p, t)
        out["sharpe_ratio"] = _sharpe(p, t)

    if do_volatility and vol_preds:
        p = np.concatenate(vol_preds)
        t = np.concatenate(vol_targets)
        out["vol_rmse"] = _rmse(p, t)
        out["vol_r2"] = _r2(p, t)

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _maybe_class_weights(
    cfg: ExperimentConfig, artifacts: FoldArtifacts,
) -> torch.Tensor | None:
    """Compute class weights if ``direction`` is active, else ``None``.

    Scheme follows ``cfg.training.class_weights``:

    * ``"inverse_frequency"`` / ``"balanced"`` - see
      :func:`compute_class_weights`.
    * ``"none"`` - returns ``None`` (loss falls back to unweighted CE).
    """
    if "direction" not in cfg.head.targets:
        return None
    scheme = cfg.training.class_weights
    if scheme == "none":
        return None
    return compute_class_weights(artifacts.train_labels_cls, scheme=scheme)


def _build_optimizer(cfg: ExperimentConfig, model: Predictor) -> torch.optim.Optimizer:
    """Construct Adam or AdamW per ``cfg.training.optimizer``."""
    lr = cfg.training.learning_rate
    wd = cfg.training.weight_decay
    name = cfg.training.optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(
        f"Unknown optimizer {name!r}. Expected 'adam' or 'adamw'."
    )


def _resolve_primary_metric(cfg: ExperimentConfig) -> tuple[str, str]:
    """Mirror :func:`mmfp.training.trainer._resolve_primary_metric` minimally."""
    targets = list(cfg.head.targets)
    if "direction" in targets:
        return "val_mcc", "max"
    if "volatility" in targets:
        return "val_rmse", "min"
    return "val_r2", "max"


def _move_batch_to_device(
    batch: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    """Move any ``Tensor`` values in the batch onto ``device``."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=False)
        else:
            out[key] = value
    return out


def _build_record(
    *,
    cfg: ExperimentConfig,
    artifacts: FoldArtifacts,
    model: Predictor,
    history_len: int,
    best_val_metric: float,
    test_metrics: dict[str, float | None],
    elapsed: float,
) -> ResultRecord:
    """Project all accumulated state into a :class:`ResultRecord`."""
    n_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)

    # Axis fingerprint: cover news/graph fields regardless of whether
    # they are enabled so downstream filters can compare enabled-vs-
    # disabled rows cleanly.
    news_encoder = cfg.news.encoder if cfg.news.enabled else "none"
    news_aggregation = cfg.news.aggregation if cfg.news.enabled else "none"
    graph_source = cfg.graph.source if cfg.graph.enabled else "none"

    return ResultRecord(
        experiment_name=cfg.name,
        config_hash=config_hash(cfg),
        seed=int(cfg.seed),
        fold_idx=int(cfg.data.fold_idx),
        news_encoder=news_encoder,
        news_aggregation=news_aggregation,
        fusion_strategy=cfg.fusion.strategy,
        head_architecture=cfg.head.architecture,
        targets=",".join(cfg.head.targets),
        graph_source=graph_source,
        lookback=int(cfg.data.lookback),
        price_encoder=cfg.model.price_encoder,
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
        platform_version=_PLATFORM_VERSION,
        git_sha=get_git_sha(),
    )


# ---------------------------------------------------------------------------
# Metric primitives (inline to avoid circular imports with
# ``mmfp.training.trainer`` and keep this module self-contained).
# ---------------------------------------------------------------------------


def _mcc(preds: np.ndarray, targets: np.ndarray) -> float:
    """Matthews Correlation Coefficient for binary classification."""
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
    """Annualised Sharpe of a sign-of-prediction strategy on realised returns.

    Ported from v1 ``src/evaluation/metrics.py::financial_metrics``.
    Trading rule: go long when ``pred > 0``, short when ``pred < 0``,
    flat when ``pred == 0``. Returns zero when the strategy has zero
    variance (e.g. all-flat prediction).
    """
    if len(preds) == 0:
        return 0.0
    signs = np.sign(preds)
    strategy_returns = signs * targets
    std = float(strategy_returns.std(ddof=0))
    if std <= 0.0:
        return 0.0
    mean = float(strategy_returns.mean())
    return float(mean / std * np.sqrt(252.0))


def _fmt(v: float | None) -> str:
    """Pretty-print a metric or ``"na"`` when missing (log-message use only)."""
    if v is None:
        return "na"
    return f"{v:.4f}"


__all__ = ["evaluate_on_test", "run_one_experiment"]
