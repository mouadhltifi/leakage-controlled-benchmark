"""Unit tests for :mod:`forecast.datasets.graph_precompute`."""

from __future__ import annotations

import numpy as np
import pytest

from forecast.config.schema import V3ExperimentConfig
from forecast.datasets.graph_precompute import (
    GraphNodeCache,
    build_graph_node_cache,
)


def _cfg_with_graph(
    minimal_cfg_dict: dict, *, source: str = "static_gics"
) -> V3ExperimentConfig:
    cfg_dict = {
        k: (dict(v) if isinstance(v, dict) else v)
        for k, v in minimal_cfg_dict.items()
    }
    cfg_dict["graph"]["enabled"] = True
    cfg_dict["graph"]["source"] = source
    cfg_dict["forecast"]["graph_node_dim"] = 64
    cfg_dict["model"]["hidden_dim"] = 64
    return V3ExperimentConfig.model_validate(cfg_dict)


# ---------------------------------------------------------------------------
# Build end-to-end
# ---------------------------------------------------------------------------


def test_build_static_returns_cache_with_all_dates(
    synthetic_artifacts_factory, minimal_cfg_dict, tmp_path,
):
    arts = synthetic_artifacts_factory(
        n_tickers=3, n_days=80, lookback=20, include_graph_adj=True,
    )
    cfg = _cfg_with_graph(minimal_cfg_dict)
    cache = build_graph_node_cache(arts, cfg, cache_dir=tmp_path)
    assert isinstance(cache, GraphNodeCache)
    unique_dates = {d for _, d in arts.price_features_by_stock_day.keys()}
    assert set(cache.data.keys()) == unique_dates
    # Each matrix is (n_tickers, dim).
    for mat in cache.data.values():
        assert mat.shape == (3, 64)
        assert mat.dtype == np.float32


def test_get_sequence_shape(synthetic_artifacts_factory, minimal_cfg_dict, tmp_path):
    arts = synthetic_artifacts_factory(
        n_tickers=3, n_days=80, lookback=20, include_graph_adj=True,
    )
    cfg = _cfg_with_graph(minimal_cfg_dict)
    cache = build_graph_node_cache(arts, cfg, cache_dir=tmp_path)
    dates = sorted(cache.data.keys())
    seq = cache.get_sequence(1, dates[:10])
    assert seq.shape == (10, 64)


def test_get_sequence_missing_date_returns_zero(
    synthetic_artifacts_factory, minimal_cfg_dict, tmp_path,
):
    arts = synthetic_artifacts_factory(
        n_tickers=3, n_days=80, lookback=20, include_graph_adj=True,
    )
    cfg = _cfg_with_graph(minimal_cfg_dict)
    cache = build_graph_node_cache(arts, cfg, cache_dir=tmp_path)
    # Ask for a date that was never produced.
    seq = cache.get_sequence(0, ["1900-01-01", "1900-01-02"])
    assert seq.shape == (2, 64)
    np.testing.assert_allclose(seq, np.zeros((2, 64), dtype=np.float32))


def test_get_sequence_bad_ticker_raises():
    cache = GraphNodeCache(data={}, dim=64, n_tickers=3)
    with pytest.raises(IndexError):
        cache.get_sequence(99, ["2016-01-04"])


# ---------------------------------------------------------------------------
# Determinism + cache reuse
# ---------------------------------------------------------------------------


def test_deterministic_same_config_same_artifacts(
    synthetic_artifacts_factory, minimal_cfg_dict, tmp_path,
):
    """Two successive calls with the same config + artifacts produce
    identical cached tensors (disk cache returns the same data).
    """
    arts = synthetic_artifacts_factory(
        n_tickers=3, n_days=80, lookback=20, include_graph_adj=True, seed=7,
    )
    cfg = _cfg_with_graph(minimal_cfg_dict)

    cache_a = build_graph_node_cache(arts, cfg, cache_dir=tmp_path)
    # Second call reads the on-disk cache.
    cache_b = build_graph_node_cache(arts, cfg, cache_dir=tmp_path)

    assert set(cache_a.data.keys()) == set(cache_b.data.keys())
    for date in cache_a.data:
        np.testing.assert_array_equal(cache_a.data[date], cache_b.data[date])


def test_graph_disabled_raises(minimal_cfg, synthetic_artifacts, tmp_path):
    """Calling with cfg.graph.enabled=False is a caller bug."""
    # Minimal cfg has graph disabled by default; artifacts have no adjacency.
    with pytest.raises(ValueError, match="graph.enabled"):
        build_graph_node_cache(synthetic_artifacts, minimal_cfg, cache_dir=tmp_path)
