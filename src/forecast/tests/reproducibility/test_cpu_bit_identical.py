"""CPU bit-identical reproducibility test for the M4 training loop.

Two successive runs of the 5-epoch integration test with ``seed=42``
must produce bit-identical val metric histories. Mirrors v2's M6
discipline. Known-non-deterministic MPS / CUDA ops are avoided by
forcing ``device="cpu"``.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader

from forecast.config.schema import V3ExperimentConfig
from forecast.datasets.multimodal import ForecastDataset
from forecast.models.losses import build_loss
from forecast.models.predictor import ForecastPredictor
from forecast.training.trainer import ForecastTrainer
from mmfp.utils.seeding import (
    make_dataloader_generator,
    make_worker_init_fn,
    set_all_seeds,
)


def _build_cfg(base: dict) -> V3ExperimentConfig:
    cfg_dict = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    cfg_dict["forecast"]["lookback"] = 20
    cfg_dict["data"]["lookback"] = 20
    cfg_dict["forecast"]["hidden_dim"] = 16
    cfg_dict["training"]["batch_size"] = 16
    cfg_dict["training"]["max_epochs"] = 3
    cfg_dict["training"]["min_epochs"] = 0
    cfg_dict["training"]["patience"] = 1000
    cfg_dict["training"]["learning_rate"] = 3e-3
    cfg_dict["training"]["scheduler"] = "none"
    cfg_dict["head"]["targets"] = ["return"]
    cfg_dict["device"] = "cpu"
    return V3ExperimentConfig.model_validate(cfg_dict)


def _single_run(synthetic_artifacts_factory, minimal_cfg_dict) -> dict[str, list[float]]:
    set_all_seeds(42)
    arts = synthetic_artifacts_factory(
        n_tickers=3, n_days=160, lookback=20, seed=42,
    )
    cfg = _build_cfg(minimal_cfg_dict)

    train_ds = ForecastDataset(arts, "train", cfg)
    val_ds = ForecastDataset(arts, "val", cfg)

    gen = make_dataloader_generator(cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=gen,
        worker_init_fn=make_worker_init_fn(cfg.seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = ForecastPredictor(cfg, feature_schema=arts.feature_schema)
    loss_fn = build_loss(cfg, class_weights=None)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    trainer = ForecastTrainer(
        model=model,
        cfg=cfg,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=None,
        callbacks=[],
    )
    hist = trainer.fit(train_loader, val_loader)
    return {
        "train_loss": list(hist.train_loss),
        "val_loss": list(hist.val_loss),
        **{k: list(v) for k, v in hist.val_metrics.items()},
    }


@pytest.mark.slow
def test_cpu_bit_identical(synthetic_artifacts_factory, minimal_cfg_dict):
    """Two seed=42 runs produce bit-identical histories on CPU."""
    run_a = _single_run(synthetic_artifacts_factory, minimal_cfg_dict)
    run_b = _single_run(synthetic_artifacts_factory, minimal_cfg_dict)

    assert run_a.keys() == run_b.keys()
    for key in run_a:
        series_a = run_a[key]
        series_b = run_b[key]
        assert len(series_a) == len(series_b), (
            f"series length mismatch for {key}"
        )
        for i, (a, b) in enumerate(zip(series_a, series_b)):
            # NaN handling: both must be NaN or neither.
            if math.isnan(a) or math.isnan(b):
                assert math.isnan(a) and math.isnan(b), (
                    f"{key}[{i}]: one run NaN ({a}) the other not ({b})"
                )
            else:
                assert a == b, (
                    f"{key}[{i}]: {a} != {b}  (not bit-identical)"
                )
