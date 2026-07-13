"""Unit tests for :class:`forecast.datasets.multimodal.ForecastDataset`.

Covers shape contracts, key emission per active-modality set, lookback
filtering, deadzone propagation, and the graph-sequence alignment.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from forecast.config.schema import V3ExperimentConfig
from forecast.datasets.graph_precompute import GraphNodeCache
from forecast.datasets.multimodal import ForecastDataset


def _config_for(
    minimal_cfg_dict: dict,
    *,
    news: bool = False,
    macro: bool = False,
    social: bool = False,
    graph: bool = False,
    lookback: int = 60,
    deadzone: float = 0.0,
) -> V3ExperimentConfig:
    """Build a validated config for a given modality set.

    The dataset reads only ``cfg.forecast.lookback``; we leave
    ``cfg.data.lookback`` untouched so values above the v2 DataConfig
    cap (60) pass schema validation when only the forecast lookback
    needs to be stretched.
    """
    cfg_dict = dict(minimal_cfg_dict)
    cfg_dict = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_dict.items()}
    cfg_dict["forecast"]["lookback"] = lookback
    # Keep data.lookback within [1, 60] (v2 DataConfig cap).
    cfg_dict["data"]["lookback"] = min(lookback, 60)
    cfg_dict["data"]["deadzone"] = deadzone
    cfg_dict["news"]["enabled"] = news
    cfg_dict["macro"]["enabled"] = macro
    cfg_dict["social"]["enabled"] = social
    cfg_dict["graph"]["enabled"] = graph
    # Reduce head targets so v3 doesn't force direction filtering in
    # downstream assemble calls (not used here, but keeps semantics clean).
    cfg_dict["head"]["targets"] = ["return"]
    return V3ExperimentConfig.model_validate(cfg_dict)


# ---------------------------------------------------------------------------
# Length + shapes
# ---------------------------------------------------------------------------


def test_length_matches_split_features(synthetic_artifacts_factory, minimal_cfg_dict):
    arts = synthetic_artifacts_factory(n_tickers=3, n_days=200, lookback=60)
    cfg = _config_for(minimal_cfg_dict, lookback=60)

    ds = ForecastDataset(arts, "train", cfg)
    assert len(ds) == arts.train_features.shape[0]

    ds_val = ForecastDataset(arts, "val", cfg)
    assert len(ds_val) == arts.val_features.shape[0]

    ds_test = ForecastDataset(arts, "test", cfg)
    assert len(ds_test) == arts.test_features.shape[0]


def test_price_only_keys(synthetic_artifacts_factory, minimal_cfg_dict):
    arts = synthetic_artifacts_factory(n_tickers=3, n_days=200, lookback=60)
    cfg = _config_for(minimal_cfg_dict, lookback=60)

    ds = ForecastDataset(arts, "train", cfg)
    sample = ds[0]
    expected_keys = {
        "price_seq", "static_categorical",
        "y_return", "y_direction", "y_volatility",
    }
    assert set(sample.keys()) == expected_keys


def test_all_modalities_keys_minus_graph(synthetic_artifacts_factory, minimal_cfg_dict):
    arts = synthetic_artifacts_factory(
        n_tickers=3, n_days=200, lookback=60,
        include_news=True, include_macro=True, include_social=True,
    )
    cfg = _config_for(
        minimal_cfg_dict, news=True, macro=True, social=True, lookback=60,
    )

    ds = ForecastDataset(arts, "train", cfg)
    sample = ds[0]
    for key in ("price_seq", "news_seq", "macro_seq", "social_seq"):
        assert key in sample, f"expected key {key!r} in sample"


def test_sequence_shapes(synthetic_artifacts_factory, minimal_cfg_dict):
    f_news = 129
    arts = synthetic_artifacts_factory(
        n_tickers=3, n_days=200, lookback=60,
        include_news=True, f_news=f_news,
        include_macro=True,
    )
    cfg = _config_for(minimal_cfg_dict, news=True, macro=True, lookback=60)

    ds = ForecastDataset(arts, "train", cfg)
    sample = ds[0]
    assert sample["price_seq"].shape == (60, 13)
    assert sample["news_seq"].shape == (60, f_news)
    assert sample["macro_seq"].shape == (60, 9)


def test_static_categorical_shape_and_dtype(
    synthetic_artifacts_factory, minimal_cfg_dict,
):
    arts = synthetic_artifacts_factory(n_tickers=3, n_days=200, lookback=60)
    cfg = _config_for(minimal_cfg_dict, lookback=60)
    ds = ForecastDataset(arts, "train", cfg)
    sample = ds[0]
    assert sample["static_categorical"].shape == (2,)
    assert sample["static_categorical"].dtype == torch.int64


def test_target_dtypes(synthetic_artifacts_factory, minimal_cfg_dict):
    arts = synthetic_artifacts_factory(n_tickers=3, n_days=200, lookback=60)
    cfg = _config_for(minimal_cfg_dict, lookback=60)
    ds = ForecastDataset(arts, "train", cfg)
    sample = ds[0]
    assert sample["y_return"].dtype == torch.float32
    assert sample["y_direction"].dtype == torch.int64
    assert sample["y_volatility"].dtype == torch.float32
    # And they're scalars (0-dim).
    assert sample["y_return"].dim() == 0
    assert sample["y_direction"].dim() == 0
    assert sample["y_volatility"].dim() == 0


# ---------------------------------------------------------------------------
# Lookback
# ---------------------------------------------------------------------------


def test_lookback_filtering_excludes_earliest(
    synthetic_artifacts_factory, minimal_cfg_dict,
):
    """The synthetic factory excludes the first (lookback-1) days, so
    the train split should start at day ``lookback - 1`` of the grid
    (and the dataset must accept every sample without raising)."""
    lookback = 60
    arts = synthetic_artifacts_factory(
        n_tickers=3, n_days=200, lookback=lookback,
    )
    cfg = _config_for(minimal_cfg_dict, lookback=lookback)
    ds = ForecastDataset(arts, "train", cfg)
    # Every __getitem__ should succeed — no sample should trigger the
    # "insufficient history" RuntimeError.
    for idx in range(len(ds)):
        _ = ds[idx]


def test_lookback_short_raises_when_insufficient_history(
    synthetic_artifacts_factory, minimal_cfg_dict,
):
    """If the fixture emits samples without enough prior days for the
    declared lookback, the dataset's `_resolve_window_dates` must
    raise."""
    # n_days=50, lookback=60 — by construction factory filters these,
    # producing zero samples. So we set lookback *after* artifacts are
    # built to simulate mismatch.
    arts = synthetic_artifacts_factory(
        n_tickers=3, n_days=100, lookback=10,
    )
    # Now override the dataset lookback to 120, which exceeds the
    # number of available prior days for any sample.
    cfg = _config_for(minimal_cfg_dict, lookback=120)
    ds = ForecastDataset(arts, "train", cfg)
    with pytest.raises(RuntimeError, match="lookback"):
        _ = ds[0]


# ---------------------------------------------------------------------------
# Deadzone
# ---------------------------------------------------------------------------


def test_deadzone_flag_stored_in_labels(
    synthetic_artifacts_factory, minimal_cfg_dict,
):
    """Samples whose |return| <= deadzone should have y_direction == -1.

    The dataset itself does not filter deadzone rows — assembly does
    that when direction is in targets. Since our synthetic factory
    preserves ``-1`` labels when deadzone > 0, verify the dataset
    propagates them into ``y_direction``.
    """
    arts = synthetic_artifacts_factory(
        n_tickers=3, n_days=200, lookback=60, deadzone=0.005, seed=1,
    )
    cfg = _config_for(minimal_cfg_dict, lookback=60, deadzone=0.005)
    ds = ForecastDataset(arts, "train", cfg)
    # Just check that y_direction preserves the int label faithfully.
    for idx in range(min(20, len(ds))):
        sample = ds[idx]
        stored = int(arts.train_labels_cls[idx])
        assert int(sample["y_direction"]) == stored


# ---------------------------------------------------------------------------
# Graph sequence alignment
# ---------------------------------------------------------------------------


def test_graph_sequence_alignment(synthetic_artifacts_factory, minimal_cfg_dict):
    """``graph_seq[t]`` must match ``cache.get(date_{T - L + 1 + t})[ticker_id]``."""
    lookback = 60
    n_tickers = 3
    arts = synthetic_artifacts_factory(
        n_tickers=n_tickers, n_days=200, lookback=lookback,
        include_graph_adj=True,
    )
    cfg = _config_for(minimal_cfg_dict, graph=True, lookback=lookback)
    # Hand-roll a GraphNodeCache with one distinguishing entry per
    # (date, ticker): value = sin(ticker_id * seed + days_since_start).
    dates = sorted({d for _, d in arts.price_features_by_stock_day.keys()})
    graph_dim = cfg.forecast.graph_node_dim
    data = {}
    for i, date in enumerate(dates):
        mat = np.zeros((n_tickers, graph_dim), dtype=np.float32)
        for ticker_id in range(n_tickers):
            mat[ticker_id, :] = np.arange(graph_dim, dtype=np.float32) + (
                ticker_id * 1000 + i
            )
        data[date] = mat
    cache = GraphNodeCache(data=data, dim=graph_dim, n_tickers=n_tickers)

    ds = ForecastDataset(arts, "train", cfg, graph_node_cache=cache)
    sample = ds[0]
    graph_seq = sample["graph_seq"].numpy()
    assert graph_seq.shape == (lookback, graph_dim)

    # Reconstruct the expected window dates.
    ticker_id = int(sample["static_categorical"][0])
    ticker = arts.tickers[ticker_id]
    end_date = str(arts.train_dates[0])
    sorted_dates = sorted(
        d for (t, d) in arts.price_features_by_stock_day.keys() if t == ticker
    )
    end_idx = sorted_dates.index(end_date)
    window = sorted_dates[end_idx - lookback + 1 : end_idx + 1]
    for t_idx, date in enumerate(window):
        expected = cache.data[date][ticker_id]
        np.testing.assert_allclose(graph_seq[t_idx], expected, atol=1e-6)


def test_graph_disabled_has_no_graph_seq(
    synthetic_artifacts_factory, minimal_cfg_dict,
):
    arts = synthetic_artifacts_factory(n_tickers=3, n_days=200, lookback=60)
    cfg = _config_for(minimal_cfg_dict, lookback=60)
    ds = ForecastDataset(arts, "train", cfg)
    sample = ds[0]
    assert "graph_seq" not in sample


def test_graph_enabled_requires_cache(
    synthetic_artifacts_factory, minimal_cfg_dict,
):
    arts = synthetic_artifacts_factory(
        n_tickers=3, n_days=200, lookback=60, include_graph_adj=True,
    )
    cfg = _config_for(minimal_cfg_dict, graph=True, lookback=60)
    with pytest.raises(ValueError, match="graph_node_cache"):
        ForecastDataset(arts, "train", cfg, graph_node_cache=None)
