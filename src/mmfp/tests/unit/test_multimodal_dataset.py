"""Unit tests for :class:`mmfp.datasets.multimodal.MultiModalDataset`.

Every test either builds a synthetic :class:`FoldArtifacts` in memory
(for hermetic invariants) or calls :func:`assemble_fold` against cached
parquets (for integration behaviour). Tests that require real data
skip gracefully when caches are absent.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.data.assemble import (
    FeatureSchema,
    FoldArtifacts,
    assemble_fold,
)
from mmfp.data.paths import FEATURES_DIR
from mmfp.data.universe import ALL_TICKERS
from mmfp.datasets.multimodal import MultiModalDataset


_PRICE_PARQUET = FEATURES_DIR / "price_features.parquet"

_HAS_PRICE_CACHE = _PRICE_PARQUET.exists()


# ---------------------------------------------------------------------------
# Synthetic artifact builder — no disk dependency.
# ---------------------------------------------------------------------------


def _build_synthetic_artifacts(
    *,
    lookback: int,
    n_train: int = 30,
    n_val: int = 10,
    n_test: int = 10,
    include_graph: bool = False,
    include_dynamic: bool = False,
    include_macro: bool = False,
) -> FoldArtifacts:
    """Construct an in-memory :class:`FoldArtifacts` for unit-testing.

    Uses a small fake ticker universe of 4 symbols drawn from
    :data:`ALL_TICKERS` to keep graph tensor shapes manageable while
    still exercising the real stock-idx remapping code path.
    """
    from mmfp.features.scalers import FittedScaler
    import pandas as pd

    n_stocks = len(ALL_TICKERS)
    f_price = 10
    f_macro = 7

    def _gen_rows(n: int, start_date: str) -> tuple[np.ndarray, ...]:
        dates = (
            pd.bdate_range(start_date, periods=n).strftime("%Y-%m-%d").to_numpy().astype("<U10")
        )
        features_price = np.random.default_rng(42).standard_normal(
            (n, f_price)
        ).astype(np.float32)
        features_macro = np.random.default_rng(43).standard_normal(
            (n, f_macro)
        ).astype(np.float32)
        cls = np.random.default_rng(44).integers(0, 2, size=n).astype(np.int64)
        reg = np.random.default_rng(45).standard_normal(n).astype(np.float32)
        vol = np.abs(reg)
        # Cycle a few stocks rather than just one.
        stock_idx = (np.arange(n) % 4).astype(np.int64)
        return (
            features_price,
            features_macro,
            cls,
            reg,
            vol,
            dates,
            stock_idx,
        )

    (
        tr_p, tr_m, tr_cls, tr_reg, tr_vol, tr_d, tr_s,
    ) = _gen_rows(n_train, "2016-01-04")
    (
        va_p, va_m, va_cls, va_reg, va_vol, va_d, va_s,
    ) = _gen_rows(n_val, "2019-01-02")
    (
        te_p, te_m, te_cls, te_reg, te_vol, te_d, te_s,
    ) = _gen_rows(n_test, "2019-07-01")

    if include_macro:
        train_features = np.concatenate([tr_p, tr_m], axis=1)
        val_features = np.concatenate([va_p, va_m], axis=1)
        test_features = np.concatenate([te_p, te_m], axis=1)
        schema = FeatureSchema(
            price=slice(0, f_price),
            macro=slice(f_price, f_price + f_macro),
        )
    else:
        train_features = tr_p
        val_features = va_p
        test_features = te_p
        schema = FeatureSchema(price=slice(0, f_price))

    # Dummy fitted scaler (train-only, same shape contract).
    scaler = FittedScaler()
    scaler.fit(pd.DataFrame(tr_p, columns=[f"price_{i}" for i in range(f_price)]))
    scalers = {"price": scaler}
    if include_macro:
        macro_scaler = FittedScaler()
        macro_scaler.fit(
            pd.DataFrame(tr_m, columns=[f"macro_{i}" for i in range(f_macro)])
        )
        scalers["macro"] = macro_scaler

    # Build the per-(ticker, date) lookup with enough history to honour
    # ``lookback`` — include a 50-day pre-train warm-up per ticker.
    price_by_day: dict[tuple[str, str], np.ndarray] = {}
    warmup_dates = pd.bdate_range("2015-10-01", periods=80).strftime("%Y-%m-%d")
    for si in range(min(4, n_stocks)):
        ticker = ALL_TICKERS[si]
        for d in warmup_dates:
            price_by_day[(ticker, d)] = np.random.default_rng(si * 100).standard_normal(
                f_price,
            ).astype(np.float32)

    # Also populate entries for every sample present in train/val/test.
    for split_p, split_d, split_s in (
        (tr_p, tr_d, tr_s),
        (va_p, va_d, va_s),
        (te_p, te_d, te_s),
    ):
        for i in range(len(split_d)):
            t = ALL_TICKERS[int(split_s[i])]
            d = str(split_d[i])
            price_by_day[(t, d)] = split_p[i]

    # Static adjacency if requested.
    static_adj: np.ndarray | None = None
    dyn: dict[str, np.ndarray] | None = None
    if include_graph:
        static_adj = np.zeros((n_stocks, n_stocks), dtype=np.float32)
        # Dense sample edges for first 4 nodes.
        for i in range(4):
            for j in range(4):
                if i != j:
                    static_adj[i, j] = 1.0
    if include_dynamic:
        d0 = str(tr_d[0])
        dyn = {d0: static_adj if static_adj is not None else np.ones((n_stocks, n_stocks), dtype=np.float32)}

    art = FoldArtifacts(
        train_features=train_features,
        train_labels_cls=tr_cls,
        train_labels_reg=tr_reg,
        train_labels_vol=tr_vol,
        train_dates=tr_d,
        train_stock_idx=tr_s,
        val_features=val_features,
        val_labels_cls=va_cls,
        val_labels_reg=va_reg,
        val_labels_vol=va_vol,
        val_dates=va_d,
        val_stock_idx=va_s,
        test_features=test_features,
        test_labels_cls=te_cls,
        test_labels_reg=te_reg,
        test_labels_vol=te_vol,
        test_dates=te_d,
        test_stock_idx=te_s,
        feature_schema=schema,
        scalers=scalers,
        static_adj=static_adj,
        dynamic_snapshots=dyn,
        tickers=list(ALL_TICKERS),
        price_features_by_stock_day=price_by_day,
        fold_idx=0,
        train_start="2016-01-04",
        train_end=str(tr_d[-1]),
        val_end=str(va_d[-1]),
        test_end=str(te_d[-1]),
    )
    return art


# ---------------------------------------------------------------------------
# Helpers: build a config with a specific modality/target combo.
# ---------------------------------------------------------------------------


def _cfg_from_base(**overrides) -> ExperimentConfig:
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "test"
    for path, value in overrides.items():
        parts = path.split(".")
        cursor = d
        for p in parts[:-1]:
            cursor = cursor[p]
        cursor[parts[-1]] = value
    return ExperimentConfig.model_validate(d)


# ---------------------------------------------------------------------------
# Hermetic tests (synthetic artifacts).
# ---------------------------------------------------------------------------


def test_dataset_length_matches_features_count() -> None:
    art = _build_synthetic_artifacts(lookback=1)
    cfg = _cfg_from_base(**{"data.lookback": 1, "model.price_encoder": "feedforward"})
    ds = MultiModalDataset(art, "train", cfg)
    assert len(ds) == art.train_features.shape[0]


def test_dataset_lookback_1_price_shape() -> None:
    art = _build_synthetic_artifacts(lookback=1)
    cfg = _cfg_from_base(**{"data.lookback": 1, "model.price_encoder": "feedforward"})
    ds = MultiModalDataset(art, "train", cfg)
    sample = ds[0]
    assert sample["price"].shape == (10,)
    assert sample["price"].dtype == torch.float32


def test_dataset_lookback_20_price_shape() -> None:
    art = _build_synthetic_artifacts(lookback=20)
    cfg = _cfg_from_base(**{"data.lookback": 20})
    ds = MultiModalDataset(art, "train", cfg)
    sample = ds[0]
    assert sample["price"].shape == (20, 10)
    assert sample["price"].dtype == torch.float32


def test_dataset_cls_target_never_minus_one() -> None:
    """Deadzone samples must be filtered upstream by assemble_fold."""
    art = _build_synthetic_artifacts(lookback=1)
    # Inject a -1 label; the dataset must raise (never silently emit).
    art.train_labels_cls[5] = -1
    cfg = _cfg_from_base(**{"data.lookback": 1, "model.price_encoder": "feedforward"})
    ds = MultiModalDataset(art, "train", cfg)
    with pytest.raises(RuntimeError, match="deadzone"):
        _ = ds[5]


def test_dataset_target_tensors_dtypes() -> None:
    art = _build_synthetic_artifacts(lookback=1)
    cfg = _cfg_from_base(**{
        "data.lookback": 1,
        "model.price_encoder": "feedforward",
        "head.targets": ["direction", "return", "volatility"],
    })
    ds = MultiModalDataset(art, "train", cfg)
    sample = ds[0]
    assert sample["cls_target"].dtype == torch.long
    assert sample["reg_target"].dtype == torch.float32
    assert sample["vol_target"].dtype == torch.float32


def test_dataset_targets_absent_when_not_in_cfg() -> None:
    art = _build_synthetic_artifacts(lookback=1)
    cfg = _cfg_from_base(**{
        "data.lookback": 1,
        "model.price_encoder": "feedforward",
        "head.targets": ["volatility"],
        "head.architecture": "single_task",
    })
    ds = MultiModalDataset(art, "train", cfg)
    sample = ds[0]
    assert "cls_target" not in sample
    assert "reg_target" not in sample
    assert "vol_target" in sample


def test_dataset_macro_slice_tensor_present() -> None:
    art = _build_synthetic_artifacts(lookback=1, include_macro=True)
    cfg = _cfg_from_base(**{
        "data.lookback": 1,
        "model.price_encoder": "feedforward",
        "macro.enabled": True,
    })
    ds = MultiModalDataset(art, "train", cfg)
    sample = ds[0]
    assert "macro" in sample
    assert sample["macro"].shape == (7,)


def test_dataset_graph_edge_index_static_returned() -> None:
    """``graph.source='static_gics'`` returns ``edge_index`` + ``graph_features``."""
    art = _build_synthetic_artifacts(lookback=1, include_graph=True)
    cfg = _cfg_from_base(**{
        "data.lookback": 1,
        "model.price_encoder": "feedforward",
        "graph.enabled": True,
        "graph.source": "static_gics",
    })
    ds = MultiModalDataset(art, "train", cfg)
    sample = ds[0]
    assert "graph_features" in sample
    assert "edge_index" in sample
    assert sample["graph_features"].shape == (len(ALL_TICKERS), 10)
    assert sample["edge_index"].dim() == 2
    assert sample["edge_index"].size(0) == 2
    assert sample["stock_idx"].dtype == torch.long


def test_dataset_graph_static_plus_dynamic_emits_both_edges() -> None:
    art = _build_synthetic_artifacts(
        lookback=1, include_graph=True, include_dynamic=True,
    )
    cfg = _cfg_from_base(**{
        "data.lookback": 1,
        "model.price_encoder": "feedforward",
        "graph.enabled": True,
        "graph.source": "static_plus_dynamic",
    })
    ds = MultiModalDataset(art, "train", cfg)
    sample = ds[0]
    assert "edge_index_static" in sample
    assert "edge_index_dynamic" in sample
    assert "edge_index" not in sample


def test_dataset_getitem_out_of_range_raises() -> None:
    art = _build_synthetic_artifacts(lookback=1)
    cfg = _cfg_from_base(**{"data.lookback": 1, "model.price_encoder": "feedforward"})
    ds = MultiModalDataset(art, "train", cfg)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_dataset_invalid_split_raises() -> None:
    art = _build_synthetic_artifacts(lookback=1)
    cfg = _cfg_from_base(**{"data.lookback": 1, "model.price_encoder": "feedforward"})
    with pytest.raises(ValueError, match="split must be"):
        MultiModalDataset(art, "bogus", cfg)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration tests against real parquets.
# ---------------------------------------------------------------------------


pytestmark_integration = pytest.mark.skipif(
    not _HAS_PRICE_CACHE,
    reason=f"price parquet missing at {_PRICE_PARQUET}",
)


@pytestmark_integration
def test_dataset_integration_a7_price_only() -> None:
    """Full pipeline: assemble_fold ⇒ MultiModalDataset ⇒ sample."""
    cfg = _cfg_from_base(**{
        "data.fold_idx": 0,
        "data.lookback": 1,
        "model.price_encoder": "feedforward",
    })
    art = assemble_fold(cfg)
    train_ds = MultiModalDataset(art, "train", cfg)
    val_ds = MultiModalDataset(art, "val", cfg)
    test_ds = MultiModalDataset(art, "test", cfg)

    # Non-empty.
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

    # One sample has expected keys.
    s = train_ds[0]
    assert set(s.keys()) == {"price", "stock_idx", "cls_target", "reg_target"}
    assert s["price"].shape == (art.feature_schema.total_dim(),)
    assert s["cls_target"].item() in (0, 1)


@pytestmark_integration
def test_dataset_integration_lookback_20_no_early_samples() -> None:
    """Real data: lookback=20 drops first 19 trading days per ticker."""
    cfg = _cfg_from_base(**{
        "data.fold_idx": 0,
        "data.lookback": 20,
    })
    art = assemble_fold(cfg)
    ds = MultiModalDataset(art, "train", cfg)

    # Pick samples for ticker 'AAPL' and verify none fall within its
    # first 19 trading days in the aligned frame.
    aapl_idx = ALL_TICKERS.index("AAPL")
    aapl_mask = art.train_stock_idx == aapl_idx
    aapl_dates = sorted(set(art.train_dates[aapl_mask].tolist()))
    assert len(aapl_dates) > 0

    # Full history for AAPL (from price_by_day) includes warm-up rows.
    all_aapl_dates = sorted(
        d for (t, d) in art.price_features_by_stock_day if t == "AAPL"
    )
    # The earliest training date for AAPL should be at least position
    # 19 within the full history.
    earliest_train = aapl_dates[0]
    pos = all_aapl_dates.index(earliest_train)
    assert pos >= 19, (
        f"AAPL earliest train date {earliest_train} is at position {pos} in "
        "the full history; lookback=20 needs >= 19 prior days."
    )

    # Sample from the dataset.
    sample = ds[0]
    assert sample["price"].shape == (20, art.feature_schema.total_dim())
