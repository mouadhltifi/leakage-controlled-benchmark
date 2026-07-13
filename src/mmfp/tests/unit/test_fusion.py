"""Unit tests for :mod:`mmfp.models.fusion`.

Covers the contract for each fusion strategy:

* Shape: output is ``(B, H)`` for any active modality count.
* Single-modality pass-through: the sole encoding is returned unchanged
  regardless of strategy.
* Gated cross-attention missing-primary-modality error.
* Config knobs (dropout, num_heads) are actually honoured.
* Gradient flows to every parameter.
"""

from __future__ import annotations

import copy

import pytest
import torch

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.models.fusion import (
    ConcatFusion,
    FusionStrategy,
    GatedCrossAttentionFusion,
    MultiheadAttentionFusion,
    build_fusion,
)


def _cfg(**overrides) -> ExperimentConfig:
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "test-fusion"
    for dotted, value in overrides.items():
        cursor = d
        parts = dotted.split(".")
        for p in parts[:-1]:
            cursor = cursor[p]
        cursor[parts[-1]] = value
    return ExperimentConfig.model_validate(d)


def _dummy_encodings(n_mod: int, hidden: int, batch: int = 4) -> dict[str, torch.Tensor]:
    """Generate ``n_mod`` random ``(B, H)`` encodings with canonical keys."""
    names = ["price", "macro", "news", "social", "graph"][:n_mod]
    return {name: torch.randn(batch, hidden, requires_grad=True) for name in names}


# ---------------------------------------------------------------------------
# ConcatFusion
# ---------------------------------------------------------------------------


class TestConcatFusion:
    def test_shape_three_modalities(self) -> None:
        cfg = _cfg()
        fusion = ConcatFusion(cfg, n_modalities=3)
        encs = _dummy_encodings(3, cfg.model.hidden_dim)
        out = fusion(encs)
        assert out.shape == (4, cfg.model.hidden_dim)

    def test_shape_five_modalities(self) -> None:
        cfg = _cfg()
        fusion = ConcatFusion(cfg, n_modalities=5)
        encs = _dummy_encodings(5, cfg.model.hidden_dim)
        out = fusion(encs)
        assert out.shape == (4, cfg.model.hidden_dim)

    def test_single_modality_passthrough(self) -> None:
        cfg = _cfg()
        fusion = ConcatFusion(cfg, n_modalities=1)
        enc = torch.randn(3, cfg.model.hidden_dim)
        out = fusion({"price": enc})
        assert torch.equal(out, enc)

    def test_empty_encodings_raises(self) -> None:
        cfg = _cfg()
        fusion = ConcatFusion(cfg, n_modalities=3)
        with pytest.raises(ValueError, match="empty encodings"):
            fusion({})

    def test_invalid_n_modalities_raises(self) -> None:
        cfg = _cfg()
        with pytest.raises(ValueError, match="n_modalities"):
            ConcatFusion(cfg, n_modalities=0)

    def test_gradient_flow(self) -> None:
        cfg = _cfg()
        fusion = ConcatFusion(cfg, n_modalities=3)
        encs = _dummy_encodings(3, cfg.model.hidden_dim)
        out = fusion(encs)
        out.sum().backward()
        for name, p in fusion.named_parameters():
            assert p.grad is not None, f"{name} no grad"

    def test_dropout_from_config(self) -> None:
        # Different dropouts → different parameter counts? No — same count,
        # but the Dropout layer's ``p`` attribute should match cfg.
        cfg = _cfg(**{"model.dropout": 0.42})
        fusion = ConcatFusion(cfg, n_modalities=3)
        # projection = Linear + ReLU + Dropout(p)
        dropout_layer = next(m for m in fusion.projection if isinstance(m, torch.nn.Dropout))
        assert dropout_layer.p == 0.42


# ---------------------------------------------------------------------------
# GatedCrossAttentionFusion
# ---------------------------------------------------------------------------


class TestGatedCrossAttentionFusion:
    def test_shape(self) -> None:
        cfg = _cfg(**{"fusion.strategy": "gated_cross_attention"})
        fusion = GatedCrossAttentionFusion(cfg)
        encs = _dummy_encodings(3, cfg.model.hidden_dim)
        out = fusion(encs)
        assert out.shape == (4, cfg.model.hidden_dim)

    def test_single_modality_passthrough(self) -> None:
        cfg = _cfg(**{"fusion.strategy": "gated_cross_attention"})
        fusion = GatedCrossAttentionFusion(cfg)
        enc = torch.randn(3, cfg.model.hidden_dim)
        out = fusion({"price": enc})
        assert torch.equal(out, enc)

    def test_missing_primary_modality_raises(self) -> None:
        cfg = _cfg(**{
            "fusion.strategy": "gated_cross_attention",
            "fusion.primary_modality": "price",
        })
        fusion = GatedCrossAttentionFusion(cfg)
        encs = {
            "macro": torch.randn(3, cfg.model.hidden_dim),
            "news": torch.randn(3, cfg.model.hidden_dim),
        }
        with pytest.raises(ValueError, match="primary modality"):
            fusion(encs)

    def test_primary_modality_custom(self) -> None:
        cfg = _cfg(**{
            "fusion.strategy": "gated_cross_attention",
            "fusion.primary_modality": "news",
            "news.enabled": True,
            "macro.enabled": True,
        })
        fusion = GatedCrossAttentionFusion(cfg)
        encs = _dummy_encodings(3, cfg.model.hidden_dim)  # price, macro, news
        out = fusion(encs)
        assert out.shape == (4, cfg.model.hidden_dim)

    def test_gradient_flow(self) -> None:
        cfg = _cfg(**{"fusion.strategy": "gated_cross_attention"})
        fusion = GatedCrossAttentionFusion(cfg)
        encs = _dummy_encodings(3, cfg.model.hidden_dim)
        out = fusion(encs)
        out.sum().backward()
        for name, p in fusion.named_parameters():
            assert p.grad is not None, f"{name} no grad"

    def test_num_heads_from_config(self) -> None:
        cfg = _cfg(**{
            "fusion.strategy": "gated_cross_attention",
            "model.num_heads": 8,
        })
        fusion = GatedCrossAttentionFusion(cfg)
        assert fusion.mha.num_heads == 8

    def test_dropout_from_config(self) -> None:
        cfg = _cfg(**{
            "fusion.strategy": "gated_cross_attention",
            "model.dropout": 0.42,
        })
        fusion = GatedCrossAttentionFusion(cfg)
        assert fusion.dropout.p == 0.42
        assert fusion.mha.dropout == 0.42

    def test_hidden_dim_not_divisible_raises(self) -> None:
        cfg = _cfg(**{
            "fusion.strategy": "gated_cross_attention",
            "model.hidden_dim": 65,
        })
        with pytest.raises(ValueError, match="must be divisible"):
            GatedCrossAttentionFusion(cfg)


# ---------------------------------------------------------------------------
# MultiheadAttentionFusion
# ---------------------------------------------------------------------------


class TestMultiheadAttentionFusion:
    def test_shape(self) -> None:
        cfg = _cfg(**{"fusion.strategy": "multihead_attention"})
        fusion = MultiheadAttentionFusion(cfg)
        encs = _dummy_encodings(3, cfg.model.hidden_dim)
        out = fusion(encs)
        assert out.shape == (4, cfg.model.hidden_dim)

    def test_single_modality_passthrough(self) -> None:
        cfg = _cfg(**{"fusion.strategy": "multihead_attention"})
        fusion = MultiheadAttentionFusion(cfg)
        enc = torch.randn(3, cfg.model.hidden_dim)
        out = fusion({"price": enc})
        assert torch.equal(out, enc)

    def test_gradient_flow(self) -> None:
        cfg = _cfg(**{"fusion.strategy": "multihead_attention"})
        fusion = MultiheadAttentionFusion(cfg)
        encs = _dummy_encodings(3, cfg.model.hidden_dim)
        out = fusion(encs)
        out.sum().backward()
        for name, p in fusion.named_parameters():
            assert p.grad is not None, f"{name} no grad"

    def test_num_heads_from_config(self) -> None:
        cfg = _cfg(**{
            "fusion.strategy": "multihead_attention",
            "model.num_heads": 2,
        })
        fusion = MultiheadAttentionFusion(cfg)
        assert fusion.mha.num_heads == 2


# ---------------------------------------------------------------------------
# build_fusion factory
# ---------------------------------------------------------------------------


class TestBuildFusion:
    def test_concat(self) -> None:
        cfg = _cfg(**{"fusion.strategy": "concat"})
        fusion = build_fusion(cfg, n_modalities=2)
        assert isinstance(fusion, ConcatFusion)

    def test_gated(self) -> None:
        cfg = _cfg(**{"fusion.strategy": "gated_cross_attention"})
        fusion = build_fusion(cfg, n_modalities=2)
        assert isinstance(fusion, GatedCrossAttentionFusion)

    def test_mha(self) -> None:
        cfg = _cfg(**{"fusion.strategy": "multihead_attention"})
        fusion = build_fusion(cfg, n_modalities=2)
        assert isinstance(fusion, MultiheadAttentionFusion)

    def test_returns_fusion_strategy_subclass(self) -> None:
        cfg = _cfg()
        fusion = build_fusion(cfg, n_modalities=3)
        assert isinstance(fusion, FusionStrategy)
