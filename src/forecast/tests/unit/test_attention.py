"""Unit tests for :class:`forecast.models.tft.attention.InterpretableMultiHeadAttention`.

Covers:

* Output + attention-weight shapes.
* Causal mask: future timesteps attend to nothing (weights = 0).
* Attention weights sum to 1 along the key axis.
* n_heads=1 and n_heads=4 both work.
* Constructor validates ``hidden_dim % n_heads != 0``.
* Implementation does NOT use ``torch.nn.functional.scaled_dot_product_attention``
  (determinism requirement; source-grep check).
* Reproducibility under ``set_all_seeds``.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest
import torch

from forecast.models.tft import attention as attention_module
from forecast.models.tft.attention import InterpretableMultiHeadAttention
from mmfp.utils.seeding import set_all_seeds


# ---------------------------------------------------------------------------
# Shape contract.
# ---------------------------------------------------------------------------


def test_attention_shape_contract() -> None:
    attn = InterpretableMultiHeadAttention(hidden_dim=16, n_heads=1, dropout=0.0)
    x = torch.randn(3, 8, 16)
    out, weights = attn(x)
    assert out.shape == (3, 8, 16)
    assert weights.shape == (3, 8, 8)


def test_attention_multi_head_shape_contract() -> None:
    attn = InterpretableMultiHeadAttention(hidden_dim=16, n_heads=4, dropout=0.0)
    x = torch.randn(2, 6, 16)
    out, weights = attn(x)
    assert out.shape == (2, 6, 16)
    assert weights.shape == (2, 6, 6)


# ---------------------------------------------------------------------------
# Causal mask.
# ---------------------------------------------------------------------------


def test_causal_mask_zeros_out_future_positions() -> None:
    """Attention weights to ``j > i`` must be 0 under a causal mask."""
    attn = InterpretableMultiHeadAttention(hidden_dim=16, n_heads=1, dropout=0.0)
    attn.train(False)
    x = torch.randn(2, 10, 16)
    mask = InterpretableMultiHeadAttention.build_causal_mask(10)
    _, weights = attn(x, causal_mask=mask)

    # Upper triangle (strictly above diagonal) must be 0.
    B, L, _ = weights.shape
    for b in range(B):
        for i in range(L):
            for j in range(i + 1, L):
                assert weights[b, i, j].item() == 0.0, (
                    f"leak at (b={b}, i={i}, j={j}): {weights[b, i, j].item()}"
                )


def test_causal_mask_shape_mismatch_raises() -> None:
    attn = InterpretableMultiHeadAttention(hidden_dim=8, n_heads=1, dropout=0.0)
    x = torch.randn(1, 5, 8)
    bad_mask = torch.zeros((4, 4), dtype=torch.bool)  # wrong size
    with pytest.raises(ValueError, match="causal_mask shape"):
        attn(x, causal_mask=bad_mask)


# ---------------------------------------------------------------------------
# Softmax normalisation.
# ---------------------------------------------------------------------------


def test_attention_weights_sum_to_one_bidirectional() -> None:
    """Without a mask, attention weights sum to 1 along the last axis."""
    attn = InterpretableMultiHeadAttention(hidden_dim=16, n_heads=1, dropout=0.0)
    attn.train(False)
    x = torch.randn(3, 8, 16)
    _, weights = attn(x)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_attention_weights_sum_to_one_under_causal_mask() -> None:
    """Under a causal mask, each row still sums to 1 (the softmax
    renormalises over the unmasked positions)."""
    attn = InterpretableMultiHeadAttention(hidden_dim=16, n_heads=1, dropout=0.0)
    attn.train(False)
    L = 10
    x = torch.randn(2, L, 16)
    mask = InterpretableMultiHeadAttention.build_causal_mask(L)
    _, weights = attn(x, causal_mask=mask)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# Constructor validation.
# ---------------------------------------------------------------------------


def test_attention_hidden_dim_not_divisible_by_heads_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        InterpretableMultiHeadAttention(hidden_dim=15, n_heads=4)


def test_attention_zero_heads_raises() -> None:
    with pytest.raises(ValueError, match="n_heads"):
        InterpretableMultiHeadAttention(hidden_dim=16, n_heads=0)


# ---------------------------------------------------------------------------
# Determinism requirement: no scaled_dot_product_attention.
# ---------------------------------------------------------------------------


def test_no_sdpa_usage_in_attention_source() -> None:
    """Source file must not CALL the fused SDPA kernel.

    SDPA is fused and not bit-identical; the spec determinism
    requirement pins explicit ``matmul + softmax`` instead. We allow
    the name to appear in comments / docstrings (for context) but not
    as an actual call — check by stripping all block-comments and
    docstrings before searching for the call signature.
    """
    import ast

    source_path = pathlib.Path(inspect.getsourcefile(attention_module))
    text = source_path.read_text()
    tree = ast.parse(text)

    # Walk AST and check no Call node resolves to scaled_dot_product_attention.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Handle `X.scaled_dot_product_attention(...)` attribute access.
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "scaled_dot_product_attention"
            ):
                raise AssertionError(
                    f"SDPA call detected at line {node.lineno}: "
                    f"attention.py must use matmul+softmax for determinism."
                )
            # Handle bare `scaled_dot_product_attention(...)`.
            if (
                isinstance(func, ast.Name)
                and func.id == "scaled_dot_product_attention"
            ):
                raise AssertionError(
                    f"SDPA call detected at line {node.lineno}: "
                    f"attention.py must use matmul+softmax for determinism."
                )


# ---------------------------------------------------------------------------
# Gradient flow.
# ---------------------------------------------------------------------------


def test_attention_gradient_flow() -> None:
    attn = InterpretableMultiHeadAttention(hidden_dim=16, n_heads=1, dropout=0.0)
    x = torch.randn(2, 6, 16, requires_grad=True)
    out, _ = attn(x)
    out.pow(2).mean().backward()
    for name, p in attn.named_parameters():
        assert p.grad is not None, f"{name} no grad"
        assert p.grad.abs().sum().item() > 0, f"{name} zero grad"


# ---------------------------------------------------------------------------
# Reproducibility.
# ---------------------------------------------------------------------------


def test_attention_reproducible() -> None:
    set_all_seeds(3)
    a1 = InterpretableMultiHeadAttention(hidden_dim=16, n_heads=2, dropout=0.0)
    a1.train(False)
    x = torch.randn(2, 5, 16)
    out1, w1 = a1(x)

    set_all_seeds(3)
    a2 = InterpretableMultiHeadAttention(hidden_dim=16, n_heads=2, dropout=0.0)
    a2.train(False)
    _ = torch.randn(2, 5, 16)
    out2, w2 = a2(x)

    assert torch.allclose(out1, out2)
    assert torch.allclose(w1, w2)
