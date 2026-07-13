"""Unit tests for :mod:`mmfp.models.encoders`.

Covers the contract for each encoder:

* Shape: output is ``(B, H)`` (or ``(N, H)`` for graph) for randomized
  inputs.
* Gradient flow: ``.grad`` reaches every parameter after a backward
  pass from the encoder's output.
* Numerical: zero-variance features don't produce NaN.
* Validation: invalid shapes and dimensions raise ``ValueError``.

Every test builds an :class:`ExperimentConfig` from the canonical
defaults and mutates the relevant fields. No disk I/O; encoders need
only the config + an input dim.
"""

from __future__ import annotations

import copy

import pytest
import torch

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.data.universe import N_STOCKS
from mmfp.models.encoders import (
    EncoderBase,
    GraphGATEncoder,
    PriceFFEncoder,
    PriceLSTMEncoder,
    TabularFFEncoder,
)


def _cfg(**overrides) -> ExperimentConfig:
    """Build a validated experiment config from defaults + dotted overrides."""
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "test-encoder"
    for dotted, value in overrides.items():
        cursor = d
        parts = dotted.split(".")
        for p in parts[:-1]:
            cursor = cursor[p]
        cursor[parts[-1]] = value
    return ExperimentConfig.model_validate(d)


def _assert_grad_reaches_all_params(module: torch.nn.Module) -> None:
    """Every leaf parameter has a finite, non-zero gradient."""
    for name, p in module.named_parameters():
        assert p.grad is not None, f"param {name} got no grad"
        assert torch.isfinite(p.grad).all(), f"param {name} has non-finite grad"


# ---------------------------------------------------------------------------
# PriceLSTMEncoder
# ---------------------------------------------------------------------------


class TestPriceLSTMEncoder:
    def test_output_shape(self) -> None:
        cfg = _cfg()
        enc = PriceLSTMEncoder(cfg, input_dim=10)
        x = torch.randn(8, 20, 10)
        out = enc(x)
        assert out.shape == (8, cfg.model.hidden_dim)
        assert enc.output_dim == cfg.model.hidden_dim

    def test_output_shape_custom_hidden(self) -> None:
        cfg = _cfg(**{"model.hidden_dim": 128})
        enc = PriceLSTMEncoder(cfg, input_dim=5)
        out = enc(torch.randn(3, 10, 5))
        assert out.shape == (3, 128)

    def test_layers_config_honoured(self) -> None:
        cfg = _cfg(**{"model.price_lstm_layers": 3})
        enc = PriceLSTMEncoder(cfg, input_dim=4)
        assert enc.lstm.num_layers == 3

    def test_gradient_flow(self) -> None:
        cfg = _cfg()
        enc = PriceLSTMEncoder(cfg, input_dim=6)
        x = torch.randn(4, 5, 6, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        _assert_grad_reaches_all_params(enc)

    def test_zero_variance_input_no_nan(self) -> None:
        cfg = _cfg()
        enc = PriceLSTMEncoder(cfg, input_dim=3)
        x = torch.zeros(2, 4, 3)
        out = enc(x)
        assert torch.isfinite(out).all()

    def test_wrong_shape_raises(self) -> None:
        cfg = _cfg()
        enc = PriceLSTMEncoder(cfg, input_dim=3)
        with pytest.raises(ValueError, match="PriceLSTMEncoder expects"):
            enc(torch.randn(2, 3))  # missing time dim

    def test_invalid_input_dim_raises(self) -> None:
        cfg = _cfg()
        with pytest.raises(ValueError, match="input_dim"):
            PriceLSTMEncoder(cfg, input_dim=0)

    def test_dropout_read_from_config(self) -> None:
        cfg = _cfg(**{"model.dropout": 0.5})
        enc = PriceLSTMEncoder(cfg, input_dim=4)
        # multi-layer LSTM exposes dropout directly.
        assert enc.lstm.dropout == 0.5


# ---------------------------------------------------------------------------
# PriceFFEncoder
# ---------------------------------------------------------------------------


class TestPriceFFEncoder:
    def test_output_shape(self) -> None:
        cfg = _cfg()
        enc = PriceFFEncoder(cfg, input_dim=11)
        out = enc(torch.randn(5, 11))
        assert out.shape == (5, cfg.model.hidden_dim)

    def test_gradient_flow(self) -> None:
        cfg = _cfg()
        enc = PriceFFEncoder(cfg, input_dim=7)
        out = enc(torch.randn(3, 7, requires_grad=True))
        out.sum().backward()
        _assert_grad_reaches_all_params(enc)

    def test_zero_variance_input_no_nan(self) -> None:
        cfg = _cfg()
        enc = PriceFFEncoder(cfg, input_dim=5)
        out = enc(torch.zeros(2, 5))
        assert torch.isfinite(out).all()

    def test_wrong_shape_raises(self) -> None:
        cfg = _cfg()
        enc = PriceFFEncoder(cfg, input_dim=3)
        with pytest.raises(ValueError, match="PriceFFEncoder expects"):
            enc(torch.randn(2, 3, 3))

    def test_invalid_input_dim_raises(self) -> None:
        cfg = _cfg()
        with pytest.raises(ValueError, match="input_dim"):
            PriceFFEncoder(cfg, input_dim=0)


# ---------------------------------------------------------------------------
# TabularFFEncoder
# ---------------------------------------------------------------------------


class TestTabularFFEncoder:
    def test_output_shape_default_depth(self) -> None:
        cfg = _cfg()
        enc = TabularFFEncoder(cfg, input_dim=11)
        out = enc(torch.randn(4, 11))
        assert out.shape == (4, cfg.model.hidden_dim)

    def test_output_shape_deep(self) -> None:
        cfg = _cfg(**{"model.tabular_hidden_layers": 3})
        enc = TabularFFEncoder(cfg, input_dim=8)
        out = enc(torch.randn(4, 8))
        assert out.shape == (4, cfg.model.hidden_dim)

    def test_deeper_encoder_has_more_params(self) -> None:
        shallow = TabularFFEncoder(_cfg(), input_dim=16)
        deeper = TabularFFEncoder(
            _cfg(**{"model.tabular_hidden_layers": 3}), input_dim=16,
        )
        n_shallow = sum(p.numel() for p in shallow.parameters())
        n_deeper = sum(p.numel() for p in deeper.parameters())
        assert n_deeper > n_shallow

    def test_gradient_flow(self) -> None:
        cfg = _cfg()
        enc = TabularFFEncoder(cfg, input_dim=9)
        out = enc(torch.randn(3, 9, requires_grad=True))
        out.sum().backward()
        _assert_grad_reaches_all_params(enc)

    def test_zero_variance_input_no_nan(self) -> None:
        cfg = _cfg()
        enc = TabularFFEncoder(cfg, input_dim=4)
        out = enc(torch.zeros(2, 4))
        assert torch.isfinite(out).all()

    def test_wrong_shape_raises(self) -> None:
        cfg = _cfg()
        enc = TabularFFEncoder(cfg, input_dim=3)
        with pytest.raises(ValueError, match="TabularFFEncoder expects"):
            enc(torch.randn(2, 3, 3))


# ---------------------------------------------------------------------------
# GraphGATEncoder
# ---------------------------------------------------------------------------


class TestGraphGATEncoder:
    @staticmethod
    def _edge_index(n_nodes: int, n_edges: int) -> torch.Tensor:
        return torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long)

    def test_output_shape_static(self) -> None:
        cfg = _cfg(**{"graph.enabled": True, "graph.source": "static_gics"})
        enc = GraphGATEncoder(cfg, input_dim=10)
        n = 20
        out = enc(torch.randn(n, 10), self._edge_index(n, 60))
        assert out.shape == (n, cfg.model.hidden_dim)

    def test_output_shape_dynamic(self) -> None:
        cfg = _cfg(**{"graph.enabled": True, "graph.source": "dynamic_corr"})
        enc = GraphGATEncoder(cfg, input_dim=8)
        n = 15
        out = enc(torch.randn(n, 8), self._edge_index(n, 40))
        assert out.shape == (n, cfg.model.hidden_dim)

    def test_output_shape_static_plus_dynamic(self) -> None:
        cfg = _cfg(**{"graph.enabled": True, "graph.source": "static_plus_dynamic"})
        enc = GraphGATEncoder(cfg, input_dim=7)
        n = 12
        x = torch.randn(n, 7)
        out = enc(
            x,
            edge_index_static=self._edge_index(n, 30),
            edge_index_dynamic=self._edge_index(n, 25),
        )
        assert out.shape == (n, cfg.model.hidden_dim)

    def test_gradient_flow(self) -> None:
        cfg = _cfg(**{"graph.enabled": True, "graph.source": "static_gics"})
        enc = GraphGATEncoder(cfg, input_dim=6)
        n = 10
        x = torch.randn(n, 6, requires_grad=True)
        ei = self._edge_index(n, 30)
        out = enc(x, ei)
        out.sum().backward()
        _assert_grad_reaches_all_params(enc)

    def test_gradient_flow_static_plus_dynamic_reaches_both_stacks(self) -> None:
        cfg = _cfg(**{"graph.enabled": True, "graph.source": "static_plus_dynamic"})
        enc = GraphGATEncoder(cfg, input_dim=6)
        n = 10
        x = torch.randn(n, 6)
        out = enc(
            x,
            edge_index_static=self._edge_index(n, 30),
            edge_index_dynamic=self._edge_index(n, 20),
        )
        out.sum().backward()
        # Both sub-stacks must receive gradient.
        for sub in (enc.stack_primary, enc.stack_secondary):
            for name, p in sub.named_parameters():
                assert p.grad is not None, f"{name} got no grad"

    def test_batched_nodes(self) -> None:
        cfg = _cfg(**{"graph.enabled": True, "graph.source": "static_gics"})
        enc = GraphGATEncoder(cfg, input_dim=10)
        B = 4
        x = torch.randn(B * N_STOCKS, 10)
        ei = self._edge_index(B * N_STOCKS, 500)
        out = enc(x, ei)
        assert out.shape == (B * N_STOCKS, cfg.model.hidden_dim)

    def test_missing_edge_index_static_plus_dynamic_raises(self) -> None:
        cfg = _cfg(**{"graph.enabled": True, "graph.source": "static_plus_dynamic"})
        enc = GraphGATEncoder(cfg, input_dim=5)
        x = torch.randn(4, 5)
        with pytest.raises(ValueError, match="edge_index_static"):
            enc(x, edge_index=self._edge_index(4, 4))

    def test_missing_edge_index_raises(self) -> None:
        cfg = _cfg(**{"graph.enabled": True, "graph.source": "static_gics"})
        enc = GraphGATEncoder(cfg, input_dim=5)
        x = torch.randn(4, 5)
        with pytest.raises(ValueError, match="edge_index is required"):
            enc(x)

    def test_wrong_shape_raises(self) -> None:
        cfg = _cfg(**{"graph.enabled": True, "graph.source": "static_gics"})
        enc = GraphGATEncoder(cfg, input_dim=5)
        with pytest.raises(ValueError, match="GraphGATEncoder expects"):
            enc(torch.randn(2, 3, 5), self._edge_index(2, 4))

    def test_hidden_not_divisible_by_heads_raises(self) -> None:
        cfg = _cfg(**{
            "graph.enabled": True,
            "graph.source": "static_gics",
            "model.hidden_dim": 65,  # not divisible by 4 (default heads)
        })
        with pytest.raises(ValueError, match="must be divisible by"):
            GraphGATEncoder(cfg, input_dim=5)

    def test_configurable_layers_and_heads(self) -> None:
        cfg = _cfg(**{
            "graph.enabled": True,
            "graph.source": "static_gics",
            "model.graph_gat_heads": 2,
            "model.graph_gat_layers": 3,
        })
        enc = GraphGATEncoder(cfg, input_dim=6)
        # Access the underlying stack to check layer count.
        assert len(enc.stack_primary.convs) == 3

    def test_output_dim_attribute(self) -> None:
        cfg = _cfg(**{"graph.enabled": True, "graph.source": "static_gics"})
        enc = GraphGATEncoder(cfg, input_dim=5)
        assert enc.output_dim == cfg.model.hidden_dim


# ---------------------------------------------------------------------------
# EncoderBase contract
# ---------------------------------------------------------------------------


class TestEncoderBase:
    def test_base_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            EncoderBase(output_dim=16)  # type: ignore[abstract]

    def test_invalid_output_dim_raises(self) -> None:
        # Workaround: build a minimal concrete subclass just to reach the guard.
        class _Concrete(EncoderBase):
            def forward(self, x):  # pragma: no cover - never called
                return x

        with pytest.raises(ValueError, match="output_dim"):
            _Concrete(output_dim=0)
