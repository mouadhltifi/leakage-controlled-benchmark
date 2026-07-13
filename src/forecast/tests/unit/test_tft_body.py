"""Unit tests for :class:`forecast.models.tft.body.TFTBody`.

Full-pipeline sanity: forward pass, shape contract on every returned
tensor, parameter-count budget, gradient flow, NaN-safety on degenerate
inputs (zero / ±10 sigma), interpretability-signal contracts (VSN
weights sum to 1, causal attention weights), and reproducibility.
"""

from __future__ import annotations

import torch

from forecast.config.defaults import DEFAULT_CONFIG
from forecast.config.schema import ForecastConfig
from forecast.models.tft.body import TFTBody
from mmfp.utils.seeding import set_all_seeds


def _default_cfg(**overrides) -> ForecastConfig:
    base = dict(DEFAULT_CONFIG["forecast"])
    base.update(overrides)
    return ForecastConfig(**base)


def _make_past_inputs(
    names: list[str], B: int, L: int, H: int
) -> dict[str, torch.Tensor]:
    """Helper: build a {name: (B, L, H)} batch."""
    return {name: torch.randn(B, L, H) for name in names}


def _static_cat(B: int) -> torch.Tensor:
    """Helper: return a (B, 2) long tensor with valid ticker/sector ids."""
    return torch.stack(
        [
            torch.randint(0, 55, (B,), dtype=torch.long),
            torch.randint(0, 11, (B,), dtype=torch.long),
        ],
        dim=1,
    )


# ---------------------------------------------------------------------------
# Forward pass + shape contract.
# ---------------------------------------------------------------------------


def test_tft_body_forward_all_output_shapes() -> None:
    cfg = _default_cfg()
    body = TFTBody(cfg, n_past_modalities=1, static_cardinalities=[55, 11])
    body.train(False)
    B, L, H = 4, cfg.lookback, cfg.hidden_dim
    past = _make_past_inputs(["price"], B, L, H)
    static_cat = _static_cat(B)
    out = body(past, static_cat)

    assert out["decoder_out"].shape == (B, L, H)
    assert out["last_hidden"].shape == (B, H)
    assert out["vsn_weights_past"].shape == (B, L, 1)
    assert out["attention_weights"].shape == (B, L, L)


def test_tft_body_forward_multiple_modalities() -> None:
    cfg = _default_cfg()
    body = TFTBody(cfg, n_past_modalities=5, static_cardinalities=[55, 11])
    body.train(False)
    B, L, H = 2, cfg.lookback, cfg.hidden_dim
    past = _make_past_inputs(
        ["price", "news", "macro", "social", "graph"], B, L, H
    )
    static_cat = _static_cat(B)
    out = body(past, static_cat)

    assert out["decoder_out"].shape == (B, L, H)
    assert out["vsn_weights_past"].shape == (B, L, 5)


# ---------------------------------------------------------------------------
# Param-count budget.
# ---------------------------------------------------------------------------


def test_tft_body_param_count_budget_single_modality() -> None:
    """1-modality body should fit in [100k, 700k] (spec's verification
    command range)."""
    cfg = _default_cfg()
    body = TFTBody(cfg, n_past_modalities=1, static_cardinalities=[55, 11])
    n = sum(p.numel() for p in body.parameters())
    assert 100_000 <= n <= 700_000, f"1-mod body params = {n}"


def test_tft_body_param_count_budget_five_modalities() -> None:
    """5-modality body: the brief and spec target range.

    The TFT body alone (no ModalityProjections, no QuantileHead) is
    expected to be ~250–300 k at hidden_dim=64 with 5 past modalities.
    The predictor (M3) adds ~15 k projection + quantile head weights.
    We use a liberal [100k, 700k] range here — tight bound on the
    full predictor is enforced in M3's test_predictor.py.
    """
    cfg = _default_cfg()
    body = TFTBody(cfg, n_past_modalities=5, static_cardinalities=[55, 11])
    n = sum(p.numel() for p in body.parameters())
    assert 100_000 <= n <= 700_000, f"5-mod body params = {n}"


# ---------------------------------------------------------------------------
# Gradient flow.
# ---------------------------------------------------------------------------


def test_tft_body_gradient_flow_to_every_parameter() -> None:
    cfg = _default_cfg(dropout=0.0)
    body = TFTBody(cfg, n_past_modalities=2, static_cardinalities=[55, 11])
    B, L, H = 2, cfg.lookback, cfg.hidden_dim
    past = _make_past_inputs(["price", "news"], B, L, H)
    for t in past.values():
        t.requires_grad_(True)
    static_cat = _static_cat(B)

    out = body(past, static_cat)
    loss = out["decoder_out"].pow(2).mean()
    loss.backward()

    missing = []
    zero = []
    for name, p in body.named_parameters():
        if p.grad is None:
            missing.append(name)
        elif p.grad.abs().sum().item() == 0.0:
            zero.append(name)
    assert not missing, f"params without gradient: {missing}"
    assert not zero, f"params with zero gradient: {zero}"


# ---------------------------------------------------------------------------
# Interpretability signals.
# ---------------------------------------------------------------------------


def test_tft_body_vsn_weights_sum_to_one() -> None:
    cfg = _default_cfg(dropout=0.0)
    body = TFTBody(cfg, n_past_modalities=5, static_cardinalities=[55, 11])
    body.train(False)
    B, L, H = 2, cfg.lookback, cfg.hidden_dim
    past = _make_past_inputs(
        ["price", "news", "macro", "social", "graph"], B, L, H
    )
    static_cat = _static_cat(B)
    out = body(past, static_cat)
    w = out["vsn_weights_past"]
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_tft_body_attention_weights_causal() -> None:
    """Attention weights above the diagonal must be 0 (causal mask)."""
    cfg = _default_cfg(dropout=0.0)
    body = TFTBody(cfg, n_past_modalities=1, static_cardinalities=[55, 11])
    body.train(False)
    B, L, H = 2, cfg.lookback, cfg.hidden_dim
    past = _make_past_inputs(["price"], B, L, H)
    static_cat = _static_cat(B)
    out = body(past, static_cat)
    w = out["attention_weights"]
    # Upper triangle (j > i) must be 0.
    for b in range(B):
        upper_tri = torch.triu(w[b], diagonal=1)
        assert upper_tri.abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# NaN-safety on extreme inputs.
# ---------------------------------------------------------------------------


def test_tft_body_no_nan_on_zero_input() -> None:
    cfg = _default_cfg(dropout=0.0)
    body = TFTBody(cfg, n_past_modalities=3, static_cardinalities=[55, 11])
    body.train(False)
    B, L, H = 2, cfg.lookback, cfg.hidden_dim
    past = {name: torch.zeros(B, L, H) for name in ["price", "news", "macro"]}
    static_cat = _static_cat(B)
    out = body(past, static_cat)
    for k, v in out.items():
        assert not torch.isnan(v).any(), f"NaN in {k}"


def test_tft_body_no_nan_on_extreme_input() -> None:
    """±10σ inputs should not overflow or produce NaN."""
    cfg = _default_cfg(dropout=0.0)
    body = TFTBody(cfg, n_past_modalities=2, static_cardinalities=[55, 11])
    body.train(False)
    B, L, H = 2, cfg.lookback, cfg.hidden_dim
    past = {
        "price": torch.randn(B, L, H) * 10.0,
        "news": torch.randn(B, L, H) * 10.0,
    }
    static_cat = _static_cat(B)
    out = body(past, static_cat)
    for k, v in out.items():
        assert not torch.isnan(v).any(), f"NaN in {k}"
        assert not torch.isinf(v).any(), f"Inf in {k}"


# ---------------------------------------------------------------------------
# Reproducibility.
# ---------------------------------------------------------------------------


def test_tft_body_reproducible() -> None:
    cfg = _default_cfg(dropout=0.0)

    set_all_seeds(777)
    body1 = TFTBody(cfg, n_past_modalities=1, static_cardinalities=[55, 11])
    body1.train(False)
    past = {"price": torch.randn(2, cfg.lookback, cfg.hidden_dim)}
    static_cat = _static_cat(2)
    out1 = body1(past, static_cat)

    set_all_seeds(777)
    body2 = TFTBody(cfg, n_past_modalities=1, static_cardinalities=[55, 11])
    body2.train(False)
    _ = torch.randn(2, cfg.lookback, cfg.hidden_dim)  # consume RNG
    _ = _static_cat(2)  # consume RNG
    out2 = body2(past, static_cat)

    for key in out1:
        assert torch.allclose(out1[key], out2[key]), f"{key} differs"
