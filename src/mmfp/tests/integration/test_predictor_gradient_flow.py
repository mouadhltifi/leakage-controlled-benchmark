"""Gradient-flow integration tests for :class:`Predictor`.

Spec Section 5.2 mandates: for every axis combination, construct
``Predictor``, forward a synthetic batch, compute loss, call
``backward``, and assert every parameter receives a non-None,
non-NaN gradient.

Additionally verifies two critical edge cases:

* Graph + LSTM: both the LSTM and the GAT stacks receive gradient from
  the same backward pass (addresses an audit concern about shared
  fused representations).
* Sequential cascade: cls loss alone flows into the reg branch's
  parameters when ``detach_cascade=False``.
"""

from __future__ import annotations

import copy
from collections import Counter

import pytest
import torch

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.config.validate import validate_experiment_config
from mmfp.data.assemble import FeatureSchema
from mmfp.data.universe import N_STOCKS
from mmfp.models import Predictor
from mmfp.models.losses import build_loss


def _cfg(**overrides) -> ExperimentConfig:
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "test-grad-flow"
    for dotted, value in overrides.items():
        cursor = d
        parts = dotted.split(".")
        for p in parts[:-1]:
            cursor = cursor[p]
        cursor[parts[-1]] = value
    cfg = ExperimentConfig.model_validate(d)
    return validate_experiment_config(cfg)


def _schema(
    *, price_dim: int, macro_dim: int | None = None, news_dim: int | None = None,
    social_dim: int | None = None,
) -> FeatureSchema:
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
    cfg: ExperimentConfig, schema: FeatureSchema, *, batch_size: int = 4,
) -> dict[str, torch.Tensor]:
    price_dim = schema.range_for("price").stop - schema.range_for("price").start
    batch: dict[str, torch.Tensor] = {}
    if cfg.model.price_encoder == "lstm" and cfg.data.lookback > 1:
        batch["price"] = torch.randn(batch_size, cfg.data.lookback, price_dim)
    else:
        batch["price"] = torch.randn(batch_size, price_dim)

    if cfg.macro.enabled:
        w = schema.range_for("macro").stop - schema.range_for("macro").start
        batch["macro"] = torch.randn(batch_size, w)
    if cfg.news.enabled:
        w = schema.range_for("news").stop - schema.range_for("news").start
        batch["news"] = torch.randn(batch_size, w)
    if cfg.social.enabled:
        w = schema.range_for("social").stop - schema.range_for("social").start
        batch["social"] = torch.randn(batch_size, w)

    if cfg.graph.enabled:
        batch["graph_features"] = torch.randn(batch_size * N_STOCKS, price_dim)
        n = batch_size * N_STOCKS
        ei = torch.randint(0, n, (2, 2 * n), dtype=torch.long)
        if cfg.graph.source == "static_plus_dynamic":
            batch["edge_index_static"] = ei.clone()
            batch["edge_index_dynamic"] = torch.randint(0, n, (2, n), dtype=torch.long)
        else:
            batch["edge_index"] = ei
        batch["graph_stock_idx"] = (
            torch.arange(batch_size, dtype=torch.long) * N_STOCKS
        )
        batch["_batch_size"] = torch.as_tensor(batch_size, dtype=torch.long)
    return batch


def _make_targets(
    cfg: ExperimentConfig, *, batch_size: int = 4,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for target in cfg.head.targets:
        if target == "direction":
            out["direction"] = torch.randint(0, 2, (batch_size,), dtype=torch.long)
        else:
            out[target] = torch.randn(batch_size)
    return out


def _assert_grad_everywhere(
    module: torch.nn.Module, tolerate_zero_grad_modalities: Counter[str] | None = None,
) -> None:
    """Check that every parameter receives a non-None, finite gradient.

    Parameters
    ----------
    module
        Root module under inspection (usually the full predictor).
    tolerate_zero_grad_modalities
        Optional counter of modality names whose parameters are allowed
        to have ``.grad.abs().sum() == 0`` because the target set or the
        loss function blocks their gradient path. Typical use: a head
        whose ``direction`` branch was never touched by the loss.
    """
    for name, p in module.named_parameters():
        assert p.grad is not None, f"{name}: got no grad"
        assert torch.isfinite(p.grad).all(), f"{name}: non-finite grad"


# ---------------------------------------------------------------------------
# The six diagonal combos, each with a full forward + backward pass.
# ---------------------------------------------------------------------------


class TestGradientFlowAxisCombos:
    def test_combo1(self) -> None:
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
        schema = _schema(price_dim=10, news_dim=11)
        predictor = Predictor(cfg, schema)
        loss_fn = build_loss(cfg)

        batch = _make_batch(cfg, schema)
        targets = _make_targets(cfg)
        preds = predictor(batch)
        total, _ = loss_fn(preds, targets)
        total.backward()
        _assert_grad_everywhere(predictor)

    def test_combo2(self) -> None:
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
        schema = _schema(price_dim=10, news_dim=32)
        predictor = Predictor(cfg, schema)
        loss_fn = build_loss(cfg)

        batch = _make_batch(cfg, schema)
        targets = _make_targets(cfg)
        preds = predictor(batch)
        total, _ = loss_fn(preds, targets)
        total.backward()
        _assert_grad_everywhere(predictor)

    def test_combo3(self) -> None:
        cfg = _cfg(**{
            "data.lookback": 20,
            "model.price_encoder": "lstm",
            "news.enabled": True,
            "news.encoder": "qwen3_embedding",
            "news.pca_dims": 32,
            "graph.enabled": False,
            "head.architecture": "single_task",
            "head.targets": ["direction"],
        })
        schema = _schema(price_dim=10, news_dim=1024)
        predictor = Predictor(cfg, schema)
        loss_fn = build_loss(cfg)

        batch = _make_batch(cfg, schema)
        targets = _make_targets(cfg)
        preds = predictor(batch)
        total, _ = loss_fn(preds, targets)
        total.backward()
        _assert_grad_everywhere(predictor)

    def test_combo4(self) -> None:
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
        schema = _schema(price_dim=10, news_dim=768)
        predictor = Predictor(cfg, schema)
        loss_fn = build_loss(cfg)

        batch = _make_batch(cfg, schema)
        targets = _make_targets(cfg)
        preds = predictor(batch)
        total, _ = loss_fn(preds, targets)
        total.backward()
        _assert_grad_everywhere(predictor)

    def test_combo5(self) -> None:
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
        schema = _schema(price_dim=10, news_dim=769)
        predictor = Predictor(cfg, schema)
        loss_fn = build_loss(cfg)

        batch = _make_batch(cfg, schema)
        targets = _make_targets(cfg)
        preds = predictor(batch)
        total, _ = loss_fn(preds, targets)
        total.backward()
        _assert_grad_everywhere(predictor)

    def test_combo6(self) -> None:
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
        schema = _schema(price_dim=10, news_dim=768)
        predictor = Predictor(cfg, schema)
        loss_fn = build_loss(cfg)

        batch = _make_batch(cfg, schema)
        targets = _make_targets(cfg)
        preds = predictor(batch)
        total, _ = loss_fn(preds, targets)
        total.backward()
        _assert_grad_everywhere(predictor)


# ---------------------------------------------------------------------------
# Critical edge cases from spec Section 5.2
# ---------------------------------------------------------------------------


class TestCriticalGradientCases:
    def test_graph_lstm_both_receive_signal(self) -> None:
        """Graph + LSTM: both stacks receive gradient from one backward pass."""
        cfg = _cfg(**{
            "data.lookback": 20,
            "model.price_encoder": "lstm",
            "graph.enabled": True,
            "graph.source": "static_gics",
            "head.architecture": "parallel_multitask",
            "head.targets": ["direction", "return"],
        })
        schema = _schema(price_dim=10)
        predictor = Predictor(cfg, schema)
        loss_fn = build_loss(cfg)

        batch = _make_batch(cfg, schema)
        targets = _make_targets(cfg)
        preds = predictor(batch)
        total, _ = loss_fn(preds, targets)
        total.backward()

        price_enc = predictor.encoders["price"]
        graph_enc = predictor.encoders["graph"]

        # LSTM hidden-to-hidden weights must receive signal.
        lstm_hh = price_enc.lstm.weight_hh_l0.grad
        assert lstm_hh is not None
        assert lstm_hh.abs().sum() > 0

        # GAT first-conv weights must receive signal.
        first_gat = graph_enc.stack_primary.convs[0]
        # torch_geometric's GATConv parameter layout exposes ``lin`` for
        # the node-feature projection.
        gat_weight = None
        for pname in ("lin.weight", "lin_src.weight", "lin_l.weight"):
            try:
                gat_weight = dict(first_gat.named_parameters())[pname]
                break
            except KeyError:
                continue
        if gat_weight is None:
            # Fallback: grab the first parameter and check it.
            gat_weight = next(p for p in first_gat.parameters() if p.requires_grad)
        assert gat_weight.grad is not None
        assert gat_weight.grad.abs().sum() > 0

    def test_sequential_cascade_cls_flows_to_reg_head(self) -> None:
        """In joint mode, the cls-only loss updates the reg branch."""
        cfg = _cfg(**{
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
            "head.architecture": "sequential_cascade",
            "head.targets": ["direction", "return"],
            "head.cascade_reg_target": "return",
            "head.detach_cascade": False,
        })
        schema = _schema(price_dim=10)
        predictor = Predictor(cfg, schema)

        batch = _make_batch(cfg, schema)
        targets = _make_targets(cfg)
        preds = predictor(batch)

        # CE loss using only the cls output. This is the test that must
        # prove the cascade is jointly differentiable.
        cls_loss = torch.nn.functional.cross_entropy(
            preds["direction"], targets["direction"],
        )
        cls_loss.backward()

        # The reg_head's final linear must have non-zero gradient.
        final_reg_linear = predictor.head.reg_head[-1]
        assert final_reg_linear.weight.grad is not None
        assert final_reg_linear.weight.grad.abs().sum() > 0

    def test_predictions_are_finite(self) -> None:
        """Sanity: no NaN/Inf in predictions or gradients."""
        cfg = _cfg(**{
            "data.lookback": 20,
            "model.price_encoder": "lstm",
            "news.enabled": True,
            "news.encoder": "finbert_11dim",
            "news.pca_dims": None,
            "macro.enabled": True,
            "social.enabled": True,
            "graph.enabled": True,
            "graph.source": "static_gics",
            "head.architecture": "parallel_multitask",
            "head.targets": ["direction", "return", "volatility"],
        })
        schema = _schema(
            price_dim=10, macro_dim=7, news_dim=11, social_dim=6,
        )
        predictor = Predictor(cfg, schema)
        loss_fn = build_loss(cfg)

        batch = _make_batch(cfg, schema)
        targets = _make_targets(cfg)
        preds = predictor(batch)
        for k, v in preds.items():
            assert torch.isfinite(v).all(), f"non-finite pred in {k}"
        total, _ = loss_fn(preds, targets)
        assert torch.isfinite(total)
        total.backward()
        _assert_grad_everywhere(predictor)
