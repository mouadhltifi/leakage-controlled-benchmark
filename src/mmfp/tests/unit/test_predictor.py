"""Unit tests for :class:`mmfp.models.predictor.Predictor`.

Covers the six diagonal axis combinations from spec Section 3.10. Each
combination constructs a :class:`Predictor`, forwards a synthetic
collated batch and asserts output shapes. Gradient-flow assertions
live in :mod:`mmfp.tests.integration.test_predictor_gradient_flow` (a
separate file to keep per-test time budget small).

Simulating encoder-agnostic input dims
--------------------------------------

The predictor is encoder-family agnostic — it consumes per-modality
flat tensors whose width is declared by
:class:`~mmfp.data.assemble.FeatureSchema`. We simulate the news
encoder choice by varying the news feature dim:

* FinBERT 11-dim stats → ``news_dim = 11``
* FinBERT 768-CLS with PCA(32) → ``news_dim = 32``
* Qwen3 raw embedding (simulated) → ``news_dim = 1024``
* DeBERTa (simulated) → ``news_dim = 768``
* Spherical mean (simulated) → ``news_dim = 768 + 1`` (dispersion)
* Attention-weighted (simulated) → ``news_dim = 768``

The feature schema layer carries those widths so the encoder picks them
up without caring about the upstream embedding model.
"""

from __future__ import annotations

import copy

import pytest
import torch

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.config.validate import validate_experiment_config
from mmfp.data.assemble import FeatureSchema
from mmfp.data.universe import N_STOCKS
from mmfp.models import Predictor


def _cfg(**overrides) -> ExperimentConfig:
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "test-predictor"
    for dotted, value in overrides.items():
        cursor = d
        parts = dotted.split(".")
        for p in parts[:-1]:
            cursor = cursor[p]
        cursor[parts[-1]] = value
    cfg = ExperimentConfig.model_validate(d)
    return validate_experiment_config(cfg)


def _schema_from_dims(
    *,
    price_dim: int,
    macro_dim: int | None = None,
    news_dim: int | None = None,
    social_dim: int | None = None,
) -> FeatureSchema:
    """Construct a FeatureSchema with the requested per-modality widths."""
    offset = 0
    kwargs: dict[str, slice | None] = {
        "price": slice(offset, offset + price_dim),
        "macro": None,
        "news": None,
        "social": None,
    }
    offset += price_dim
    if macro_dim is not None:
        kwargs["macro"] = slice(offset, offset + macro_dim)
        offset += macro_dim
    if news_dim is not None:
        kwargs["news"] = slice(offset, offset + news_dim)
        offset += news_dim
    if social_dim is not None:
        kwargs["social"] = slice(offset, offset + social_dim)
        offset += social_dim
    return FeatureSchema(**kwargs)  # type: ignore[arg-type]


def _make_batch(
    cfg: ExperimentConfig,
    schema: FeatureSchema,
    *,
    batch_size: int = 4,
) -> dict[str, torch.Tensor]:
    """Construct a synthetic collated batch matching the config."""
    batch: dict[str, torch.Tensor] = {}

    # Price: flat or sequential depending on lookback/encoder.
    price_dim = schema.range_for("price").stop - schema.range_for("price").start
    if cfg.model.price_encoder == "lstm" and cfg.data.lookback > 1:
        batch["price"] = torch.randn(batch_size, cfg.data.lookback, price_dim)
    else:
        batch["price"] = torch.randn(batch_size, price_dim)

    # Tabular modalities.
    if cfg.macro.enabled:
        w = schema.range_for("macro").stop - schema.range_for("macro").start
        batch["macro"] = torch.randn(batch_size, w)
    if cfg.news.enabled:
        w = schema.range_for("news").stop - schema.range_for("news").start
        batch["news"] = torch.randn(batch_size, w)
    if cfg.social.enabled:
        w = schema.range_for("social").stop - schema.range_for("social").start
        batch["social"] = torch.randn(batch_size, w)

    # Graph tensors.
    if cfg.graph.enabled:
        batch["graph_features"] = torch.randn(batch_size * N_STOCKS, price_dim)
        n = batch_size * N_STOCKS
        ei = torch.randint(0, n, (2, 2 * n), dtype=torch.long)
        if cfg.graph.source == "static_plus_dynamic":
            batch["edge_index_static"] = ei.clone()
            batch["edge_index_dynamic"] = torch.randint(0, n, (2, n), dtype=torch.long)
        else:
            batch["edge_index"] = ei
        batch["graph_stock_idx"] = torch.arange(
            batch_size, dtype=torch.long
        ) * N_STOCKS
        batch["_batch_size"] = torch.as_tensor(batch_size, dtype=torch.long)

    return batch


# ---------------------------------------------------------------------------
# The six axis-combination tests.
# ---------------------------------------------------------------------------


class TestPredictorAxisCombos:
    """Construct + forward each of the six diagonal spec combinations."""

    def test_combo1_parallel_finbert11_static_graph_direction_return_lstm(self) -> None:
        cfg = _cfg(**{
            "data.lookback": 20,
            "model.price_encoder": "lstm",
            "news.enabled": True,
            "news.encoder": "finbert_11dim",
            "news.pca_dims": None,
            "graph.enabled": True,
            "graph.source": "static_gics",
            "head.architecture": "parallel_multitask",
            "head.targets": ["direction", "return"],
        })
        schema = _schema_from_dims(price_dim=10, news_dim=11)
        predictor = Predictor(cfg, schema)
        batch = _make_batch(cfg, schema)
        out = predictor(batch)
        assert set(out.keys()) == {"direction", "return"}
        assert out["direction"].shape == (4, 2)
        assert out["return"].shape == (4, 1)

    def test_combo2_sequential_finbert768cls_dynamic_graph_volatility_ff(self) -> None:
        cfg = _cfg(**{
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
            "news.enabled": True,
            "news.encoder": "finbert_cls_768",
            "news.pca_dims": 32,
            "graph.enabled": True,
            "graph.source": "dynamic_corr",
            "head.architecture": "sequential_cascade",
            "head.targets": ["direction", "volatility"],
            "head.cascade_reg_target": "volatility",
        })
        schema = _schema_from_dims(price_dim=10, news_dim=32)
        predictor = Predictor(cfg, schema)
        batch = _make_batch(cfg, schema)
        out = predictor(batch)
        assert set(out.keys()) == {"direction", "volatility"}
        assert out["direction"].shape == (4, 2)
        assert out["volatility"].shape == (4, 1)

    def test_combo3_single_task_qwen3_no_graph_direction_lstm(self) -> None:
        # Qwen3 raw: simulate with tabular 1024 input.
        cfg = _cfg(**{
            "data.lookback": 20,
            "model.price_encoder": "lstm",
            "news.enabled": True,
            "news.encoder": "qwen3_embedding",
            "news.pca_dims": 32,  # actual study default; simulating raw 1024 below
            "graph.enabled": False,
            "head.architecture": "single_task",
            "head.targets": ["direction"],
        })
        # Simulate Qwen3 raw (pre-PCA) by using a wide news dim.
        schema = _schema_from_dims(price_dim=10, news_dim=1024)
        predictor = Predictor(cfg, schema)
        batch = _make_batch(cfg, schema)
        out = predictor(batch)
        assert set(out.keys()) == {"direction"}
        assert out["direction"].shape == (4, 2)

    def test_combo4_parallel_deberta_static_plus_dynamic_all_three_targets_lstm(self) -> None:
        # DeBERTa: 768-dim; static + dynamic graph; parallel all-three targets.
        cfg = _cfg(**{
            "data.lookback": 20,
            "model.price_encoder": "lstm",
            "news.enabled": True,
            "news.encoder": "deberta_v3_financial",
            "news.pca_dims": 32,
            "graph.enabled": True,
            "graph.source": "static_plus_dynamic",
            "head.architecture": "parallel_multitask",
            "head.targets": ["direction", "return", "volatility"],
        })
        schema = _schema_from_dims(price_dim=10, news_dim=768)
        predictor = Predictor(cfg, schema)
        batch = _make_batch(cfg, schema)
        out = predictor(batch)
        assert set(out.keys()) == {"direction", "return", "volatility"}
        assert out["direction"].shape == (4, 2)
        assert out["return"].shape == (4, 1)
        assert out["volatility"].shape == (4, 1)

    def test_combo5_parallel_spherical_mean_no_graph_direction_ff(self) -> None:
        # Spherical mean: simulate 768 + dispersion feature (769 total).
        # finbert_cls_768 is incompatible with spherical_mean, so use
        # qwen3_embedding which accepts spherical_mean.
        cfg = _cfg(**{
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
            "news.enabled": True,
            "news.encoder": "qwen3_embedding",
            "news.pca_dims": 32,
            "news.aggregation": "spherical_mean",
            "news.dispersion_feature": True,
            "graph.enabled": False,
            "head.architecture": "parallel_multitask",
            "head.targets": ["direction"],
        })
        schema = _schema_from_dims(price_dim=10, news_dim=769)  # 768 + dispersion
        predictor = Predictor(cfg, schema)
        batch = _make_batch(cfg, schema)
        out = predictor(batch)
        assert set(out.keys()) == {"direction"}
        assert out["direction"].shape == (4, 2)

    def test_combo6_parallel_attention_weighted_static_graph_return_lstm(self) -> None:
        cfg = _cfg(**{
            "data.lookback": 20,
            "model.price_encoder": "lstm",
            "news.enabled": True,
            "news.encoder": "bge_base",
            "news.pca_dims": 32,
            "news.aggregation": "attention_weighted",
            "graph.enabled": True,
            "graph.source": "static_gics",
            "head.architecture": "single_task",
            "head.targets": ["return"],
        })
        schema = _schema_from_dims(price_dim=10, news_dim=768)
        predictor = Predictor(cfg, schema)
        batch = _make_batch(cfg, schema)
        out = predictor(batch)
        assert set(out.keys()) == {"return"}
        assert out["return"].shape == (4, 1)


# ---------------------------------------------------------------------------
# Additional focused tests
# ---------------------------------------------------------------------------


class TestPredictorAssembly:
    def test_price_only_ff(self) -> None:
        cfg = _cfg(**{
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        schema = _schema_from_dims(price_dim=10)
        predictor = Predictor(cfg, schema)
        assert list(predictor.encoders.keys()) == ["price"]

    def test_price_only_lstm(self) -> None:
        cfg = _cfg(**{
            "data.lookback": 20,
            "model.price_encoder": "lstm",
        })
        schema = _schema_from_dims(price_dim=10)
        predictor = Predictor(cfg, schema)
        assert list(predictor.encoders.keys()) == ["price"]
        batch = _make_batch(cfg, schema)
        out = predictor(batch)
        assert out["direction"].shape == (4, 2)

    def test_ff_when_lookback_1_and_encoder_is_ff(self) -> None:
        cfg = _cfg(**{
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        schema = _schema_from_dims(price_dim=10)
        predictor = Predictor(cfg, schema)
        from mmfp.models.encoders import PriceFFEncoder

        assert isinstance(predictor.encoders["price"], PriceFFEncoder)

    def test_lstm_when_lookback_gt_1_and_encoder_is_lstm(self) -> None:
        cfg = _cfg(**{
            "data.lookback": 20,
            "model.price_encoder": "lstm",
        })
        schema = _schema_from_dims(price_dim=10)
        predictor = Predictor(cfg, schema)
        from mmfp.models.encoders import PriceLSTMEncoder

        assert isinstance(predictor.encoders["price"], PriceLSTMEncoder)

    def test_ff_falls_back_when_lookback_1_but_lstm_requested(self) -> None:
        # User asks for LSTM but lookback=1: predictor should fall back to FF
        # per spec rule.
        import warnings

        with warnings.catch_warnings():
            # Validator emits a UserWarning for lookback=1 + lstm; swallow it.
            warnings.simplefilter("ignore")
            cfg = _cfg(**{
                "data.lookback": 1,
                "model.price_encoder": "lstm",
            })
        schema = _schema_from_dims(price_dim=10)
        predictor = Predictor(cfg, schema)
        from mmfp.models.encoders import PriceFFEncoder

        assert isinstance(predictor.encoders["price"], PriceFFEncoder)

    def test_all_modalities_enabled(self) -> None:
        cfg = _cfg(**{
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
            "macro.enabled": True,
            "news.enabled": True,
            "news.encoder": "finbert_11dim",
            "news.pca_dims": None,
            "social.enabled": True,
            "graph.enabled": True,
            "graph.source": "static_gics",
        })
        schema = _schema_from_dims(
            price_dim=10, macro_dim=7, news_dim=11, social_dim=6,
        )
        predictor = Predictor(cfg, schema)
        assert list(predictor.encoders.keys()) == [
            "price", "macro", "news", "social", "graph",
        ]

    def test_batch_without_bookkeeping_tensor(self) -> None:
        """Non-graph batches don't include ``_batch_size``."""
        cfg = _cfg(**{
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        schema = _schema_from_dims(price_dim=10)
        predictor = Predictor(cfg, schema)
        batch = {"price": torch.randn(4, 10)}  # bare minimum
        out = predictor(batch)
        assert out["direction"].shape == (4, 2)

    def test_graph_missing_edge_index_raises(self) -> None:
        cfg = _cfg(**{
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
            "graph.enabled": True,
            "graph.source": "static_gics",
        })
        schema = _schema_from_dims(price_dim=10)
        predictor = Predictor(cfg, schema)
        batch = {
            "price": torch.randn(4, 10),
            "graph_features": torch.randn(4 * N_STOCKS, 10),
            # Missing edge_index
            "graph_stock_idx": torch.arange(4) * N_STOCKS,
            "_batch_size": torch.as_tensor(4),
        }
        with pytest.raises(KeyError, match="edge_index"):
            predictor(batch)

    def test_graph_static_plus_dynamic_missing_edges_raises(self) -> None:
        cfg = _cfg(**{
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
            "graph.enabled": True,
            "graph.source": "static_plus_dynamic",
        })
        schema = _schema_from_dims(price_dim=10)
        predictor = Predictor(cfg, schema)
        batch = {
            "price": torch.randn(4, 10),
            "graph_features": torch.randn(4 * N_STOCKS, 10),
            "graph_stock_idx": torch.arange(4) * N_STOCKS,
            "_batch_size": torch.as_tensor(4),
            # Only static, missing dynamic
            "edge_index_static": torch.randint(0, 4 * N_STOCKS, (2, 40)),
        }
        with pytest.raises(KeyError, match="edge_index_dynamic"):
            predictor(batch)

    def test_sequential_cascade_full_assembly(self) -> None:
        cfg = _cfg(**{
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
            "head.architecture": "sequential_cascade",
            "head.targets": ["direction", "return"],
            "head.cascade_reg_target": "return",
        })
        schema = _schema_from_dims(price_dim=10)
        predictor = Predictor(cfg, schema)
        batch = {"price": torch.randn(4, 10)}
        out = predictor(batch)
        assert "direction" in out and "return" in out
