"""Unit tests for :func:`mmfp.datasets.collate.make_collate_fn`.

The collate function has two non-trivial responsibilities:

1. Stack flat-modality tensors (``price``, ``macro``, ``news``,
   ``social``, target tensors) with :func:`torch.stack`.
2. Handle graph batches PyG-style: concat node features on the node
   axis, offset per-sample edge_indices by ``batch_pos * N_nodes``.
"""

from __future__ import annotations

import copy

import pytest
import torch

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.data.universe import N_STOCKS
from mmfp.datasets.collate import make_collate_fn


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


def _sample_non_graph(price_dim: int = 10, macro_dim: int | None = None) -> dict:
    out: dict[str, torch.Tensor] = {
        "price": torch.randn(price_dim),
        "stock_idx": torch.tensor(0, dtype=torch.long),
        "cls_target": torch.tensor(1, dtype=torch.long),
        "reg_target": torch.tensor(0.01, dtype=torch.float32),
    }
    if macro_dim is not None:
        out["macro"] = torch.randn(macro_dim)
    return out


def _sample_graph(price_dim: int = 10, n_edges: int = 8) -> dict:
    n = N_STOCKS
    return {
        "price": torch.randn(price_dim),
        "stock_idx": torch.tensor(0, dtype=torch.long),
        "cls_target": torch.tensor(1, dtype=torch.long),
        "reg_target": torch.tensor(0.01, dtype=torch.float32),
        "graph_features": torch.randn(n, price_dim),
        # Deterministic edge_index: self-loops for nodes 0..n_edges-1 (so
        # we can verify the offsetting without worrying about symmetry).
        "edge_index": torch.stack(
            [torch.arange(n_edges, dtype=torch.long)] * 2, dim=0,
        ),
    }


# ---------------------------------------------------------------------------
# Non-graph collate
# ---------------------------------------------------------------------------


def test_collate_non_graph_stacks_flat_tensors() -> None:
    cfg = _cfg_from_base(**{"data.lookback": 1, "model.price_encoder": "feedforward"})
    collate = make_collate_fn(cfg)
    batch = [_sample_non_graph() for _ in range(4)]
    out = collate(batch)

    assert out["price"].shape == (4, 10)
    assert out["cls_target"].shape == (4,)
    assert out["reg_target"].shape == (4,)
    assert out["stock_idx"].shape == (4,)
    # No graph keys present.
    assert "graph_features" not in out
    assert "edge_index" not in out


def test_collate_with_macro_modality() -> None:
    cfg = _cfg_from_base(**{
        "data.lookback": 1,
        "model.price_encoder": "feedforward",
        "macro.enabled": True,
    })
    collate = make_collate_fn(cfg)
    batch = [_sample_non_graph(macro_dim=7) for _ in range(3)]
    out = collate(batch)
    assert out["macro"].shape == (3, 7)


def test_collate_empty_batch_raises() -> None:
    cfg = _cfg_from_base(**{"data.lookback": 1, "model.price_encoder": "feedforward"})
    collate = make_collate_fn(cfg)
    with pytest.raises(ValueError, match="Empty batch"):
        collate([])


def test_collate_lookback_sequence_stacks_correctly() -> None:
    """Sequence price tensors ``(L, F)`` stack to ``(B, L, F)``."""
    cfg = _cfg_from_base(**{"data.lookback": 20})
    collate = make_collate_fn(cfg)
    batch = [
        {
            "price": torch.randn(20, 10),
            "stock_idx": torch.tensor(0, dtype=torch.long),
            "cls_target": torch.tensor(0, dtype=torch.long),
            "reg_target": torch.tensor(0.0, dtype=torch.float32),
        }
        for _ in range(5)
    ]
    out = collate(batch)
    assert out["price"].shape == (5, 20, 10)


# ---------------------------------------------------------------------------
# Graph collate
# ---------------------------------------------------------------------------


def test_collate_graph_concatenates_node_features() -> None:
    cfg = _cfg_from_base(**{
        "data.lookback": 1,
        "model.price_encoder": "feedforward",
        "graph.enabled": True,
        "graph.source": "static_gics",
    })
    collate = make_collate_fn(cfg)
    batch = [_sample_graph() for _ in range(4)]
    out = collate(batch)

    # Node features concatenate: (B*N, F).
    assert out["graph_features"].shape == (4 * N_STOCKS, 10)


def test_collate_graph_edge_index_offsets_are_correct() -> None:
    """Batched edge_index has offsets of ``i * N_STOCKS`` per sample.

    Uses self-loops on nodes 0..7 for each sample so the expected
    batched edge_index is ``concat([i*N + [0..7]] for i in range(B))``.
    """
    cfg = _cfg_from_base(**{
        "data.lookback": 1,
        "model.price_encoder": "feedforward",
        "graph.enabled": True,
        "graph.source": "static_gics",
    })
    collate = make_collate_fn(cfg)
    batch = [_sample_graph(n_edges=8) for _ in range(4)]
    out = collate(batch)

    expected_rows = torch.cat(
        [torch.arange(8) + i * N_STOCKS for i in range(4)]
    )
    assert out["edge_index"].shape == (2, 4 * 8)
    assert torch.equal(out["edge_index"][0], expected_rows)
    assert torch.equal(out["edge_index"][1], expected_rows)


def test_collate_graph_stock_idx_offsets_match() -> None:
    """``graph_stock_idx`` = ``stock_idx + batch_pos * N_STOCKS``."""
    cfg = _cfg_from_base(**{
        "data.lookback": 1,
        "model.price_encoder": "feedforward",
        "graph.enabled": True,
        "graph.source": "static_gics",
    })
    collate = make_collate_fn(cfg)
    batch = []
    for i, local_idx in enumerate([3, 5, 7, 2]):
        s = _sample_graph()
        s["stock_idx"] = torch.tensor(local_idx, dtype=torch.long)
        batch.append(s)
    out = collate(batch)

    expected = torch.tensor(
        [3, 5 + N_STOCKS, 7 + 2 * N_STOCKS, 2 + 3 * N_STOCKS], dtype=torch.long
    )
    assert torch.equal(out["graph_stock_idx"], expected)


def test_collate_static_plus_dynamic_offsets_both_edges() -> None:
    cfg = _cfg_from_base(**{
        "data.lookback": 1,
        "model.price_encoder": "feedforward",
        "graph.enabled": True,
        "graph.source": "static_plus_dynamic",
    })
    collate = make_collate_fn(cfg)

    batch: list[dict[str, torch.Tensor]] = []
    for _ in range(3):
        sample = _sample_graph()
        # Replace edge_index key with both static + dynamic.
        sample.pop("edge_index")
        sample["edge_index_static"] = torch.stack(
            [torch.arange(4, dtype=torch.long)] * 2, dim=0,
        )
        sample["edge_index_dynamic"] = torch.stack(
            [torch.arange(6, dtype=torch.long)] * 2, dim=0,
        )
        batch.append(sample)
    out = collate(batch)

    assert out["edge_index_static"].shape == (2, 3 * 4)
    assert out["edge_index_dynamic"].shape == (2, 3 * 6)


def test_collate_edge_index_bad_shape_raises() -> None:
    cfg = _cfg_from_base(**{
        "data.lookback": 1,
        "model.price_encoder": "feedforward",
        "graph.enabled": True,
        "graph.source": "static_gics",
    })
    collate = make_collate_fn(cfg)
    batch = [_sample_graph() for _ in range(2)]
    # Corrupt the first sample's edge_index to wrong shape.
    batch[0]["edge_index"] = torch.zeros(3, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="edge_index must be shape"):
        collate(batch)
