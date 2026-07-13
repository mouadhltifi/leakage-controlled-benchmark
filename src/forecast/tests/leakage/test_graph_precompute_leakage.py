"""CRITICAL leakage tests for :mod:`forecast.datasets.graph_precompute`.

The graph-precompute step is the one new leakage surface introduced by
v3 M4. These tests assert:

1. For a dynamic-correlation source, the cache for date ``d`` uses only
   adjacency snapshots dated ``<= d``.
2. No future snapshot influences an earlier-date embedding. Concretely:
   inserting a blatantly-different snapshot at a FUTURE date must not
   change the embedding at any past date.
3. The cache is deterministic w.r.t. the train/val/test split
   boundaries — calling ``build_graph_node_cache`` with a fold whose
   test-split data has been perturbed must not change the cache entries
   for dates in the train split.

These are all consequences of the ``pick_snapshot_for_date`` contract
(v2), which selects the most-recent snapshot with
``snapshot_date <= target_date``. The tests exercise that contract at
the v3 cache boundary — the surface v3 adds.
"""

from __future__ import annotations

import numpy as np
import pytest

from forecast.config.schema import V3ExperimentConfig
from forecast.datasets.graph_precompute import build_graph_node_cache


def _cfg(minimal_cfg_dict: dict, *, source: str) -> V3ExperimentConfig:
    cfg_dict = {
        k: (dict(v) if isinstance(v, dict) else v)
        for k, v in minimal_cfg_dict.items()
    }
    cfg_dict["graph"]["enabled"] = True
    cfg_dict["graph"]["source"] = source
    cfg_dict["forecast"]["graph_node_dim"] = 64
    cfg_dict["model"]["hidden_dim"] = 64
    # Align refresh_every with lookback so the cross-field validator passes.
    cfg_dict["graph"]["dynamic_refresh_every"] = 20
    cfg_dict["forecast"]["lookback"] = 60
    cfg_dict["data"]["lookback"] = 60
    return V3ExperimentConfig.model_validate(cfg_dict)


def _make_dynamic_snapshots(
    n_tickers: int,
    dates: list[str],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Build one adjacency per date. Each snapshot has a distinguishing
    sparsity pattern so that swapping them produces different GAT
    outputs."""
    rng = np.random.default_rng(seed)
    snaps: dict[str, np.ndarray] = {}
    for i, date in enumerate(dates):
        adj = (rng.standard_normal((n_tickers, n_tickers)) > (0.5 - 0.01 * i)).astype(
            np.float32
        )
        # Ensure self-loops so GAT doesn't produce degenerate zero outputs.
        np.fill_diagonal(adj, 1.0)
        snaps[date] = adj
    return snaps


def test_dynamic_cache_uses_only_past_snapshots(
    synthetic_artifacts_factory, minimal_cfg_dict, tmp_path,
):
    """Embed the fold under a specific snapshot set; then change the
    LATEST snapshot and rerun. Past-date entries in the cache must be
    unchanged."""
    n_tickers = 3
    arts = synthetic_artifacts_factory(
        n_tickers=n_tickers, n_days=80, lookback=20, include_graph_adj=True,
    )
    all_dates = sorted(
        {d for _, d in arts.price_features_by_stock_day.keys()}
    )
    # Place dynamic snapshots every 10 days starting at the first date.
    snapshot_dates = all_dates[::10]
    snaps = _make_dynamic_snapshots(n_tickers, snapshot_dates, seed=0)
    arts.dynamic_snapshots = snaps

    cfg = _cfg(minimal_cfg_dict, source="dynamic_corr")

    cache1 = build_graph_node_cache(arts, cfg, cache_dir=tmp_path / "c1")

    # Mutate the LAST snapshot dramatically — should not affect entries
    # for dates < last snapshot date.
    last_date = snapshot_dates[-1]
    original_last = snaps[last_date].copy()
    snaps[last_date] = np.ones((n_tickers, n_tickers), dtype=np.float32)

    cache2 = build_graph_node_cache(arts, cfg, cache_dir=tmp_path / "c2")

    # Every date strictly before ``last_date`` must have identical embedding.
    for date in cache1.data:
        if date < last_date:
            np.testing.assert_array_equal(
                cache1.data[date],
                cache2.data[date],
                err_msg=(
                    f"Graph embedding at date {date} changed after mutating "
                    f"a LATER snapshot at {last_date} — leakage."
                ),
            )

    # And a positive control: at least one date on or after last_date
    # should differ (unless random adjacency collides pathologically).
    differing = any(
        not np.array_equal(cache1.data[d], cache2.data[d])
        for d in cache1.data
        if d >= last_date
    )
    assert differing, (
        "Mutating the last snapshot should have changed some on-or-after "
        "date embedding. If this fails the positive control is broken."
    )

    # Restore for hygiene.
    snaps[last_date] = original_last


def test_test_split_features_do_not_influence_train_cache(
    synthetic_artifacts_factory, minimal_cfg_dict, tmp_path,
):
    """Mutating the price features for a TEST-split ``(ticker, date)``
    entry must not change the cache embedding for any TRAIN-split date.

    This rules out the subtle failure mode where node features from
    future dates leak into past embeddings via the precompute step.
    """
    n_tickers = 3
    arts = synthetic_artifacts_factory(
        n_tickers=n_tickers, n_days=120, lookback=20, include_graph_adj=True,
    )
    cfg = _cfg(minimal_cfg_dict, source="static_gics")

    cache_clean = build_graph_node_cache(arts, cfg, cache_dir=tmp_path / "clean")

    # Pick a test-split (ticker, date) and overwrite its features.
    te_idx = 0
    te_ticker = arts.tickers[int(arts.test_stock_idx[te_idx])]
    te_date = str(arts.test_dates[te_idx])
    arts.price_features_by_stock_day[(te_ticker, te_date)] = np.full(
        arts.price_features_by_stock_day[(te_ticker, te_date)].shape,
        999.0,
        dtype=np.float32,
    )

    cache_mut = build_graph_node_cache(
        arts, cfg, cache_dir=tmp_path / "mutated",
    )

    # All TRAIN-split dates (strictly before the earliest test date)
    # must be identical between the two caches.
    earliest_test = min(str(d) for d in arts.test_dates)
    for date in cache_clean.data:
        if date < earliest_test:
            np.testing.assert_array_equal(
                cache_clean.data[date],
                cache_mut.data[date],
                err_msg=(
                    f"Train-era date {date} changed after mutating the "
                    f"test-era features at {te_date} — leakage."
                ),
            )


def test_static_source_cache_is_snapshot_date_independent(
    synthetic_artifacts_factory, minimal_cfg_dict, tmp_path,
):
    """A static-GICS cache is trivially leakage-free because the
    adjacency never changes. Confirm the precompute path respects this
    by producing identical cache under two different (fake) dynamic
    snapshot orderings."""
    arts = synthetic_artifacts_factory(
        n_tickers=3, n_days=80, lookback=20, include_graph_adj=True,
    )
    cfg = _cfg(minimal_cfg_dict, source="static_gics")

    cache_a = build_graph_node_cache(arts, cfg, cache_dir=tmp_path / "a")
    # Attach irrelevant dynamic snapshots that a buggy implementation
    # might accidentally use.
    all_dates = sorted(
        {d for _, d in arts.price_features_by_stock_day.keys()}
    )
    arts.dynamic_snapshots = _make_dynamic_snapshots(3, all_dates[::10], seed=42)

    cache_b = build_graph_node_cache(arts, cfg, cache_dir=tmp_path / "b")

    for date in cache_a.data:
        np.testing.assert_array_equal(
            cache_a.data[date],
            cache_b.data[date],
            err_msg=(
                f"Static-GICS cache changed at {date} when irrelevant "
                f"dynamic snapshots were attached — possible leakage path."
            ),
        )
