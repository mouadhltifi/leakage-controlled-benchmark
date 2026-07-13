"""TFT body — orchestrates all sub-modules (Lim et al. 2021, §4).

Wiring (per the architecture spec):

    projected past  ─┐
                     ▼
             ┌──►  Past-input VSN (static ctx c_s)
             │             │
             │             ▼
             │     LSTM local processor (init from c_h, c_c)
             │             │
             │             ▼
             │     Static enrichment GRN (context c_e)
             │             │
             │             ▼
             │     Interpretable MHA (causal)
             │             │
             │             ▼
             │     GateAddNorm post-attention
             │             │
             │             ▼
             │     Position-wise feed-forward (GRN, no context)
             │             │
             │             ▼
             │     GateAddNorm post-FFN   ──►  decoder_out

All LayerNorms and gated skips follow the reference implementation's
``GateAddNorm`` pattern: a small GLU applied to the new tensor followed
by LayerNorm over the sum with the skip.

Outputs
-------

The body returns a dict with four tensors:

* ``'decoder_out'``       — full sequence output ``(B, L, H)``.
* ``'last_hidden'``       — ``decoder_out[:, -1, :]``, shape ``(B, H)``.
* ``'vsn_weights_past'``  — per-timestep modality weights ``(B, L, n_past_modalities)``.
* ``'attention_weights'`` — mean-across-heads attention ``(B, L, L)``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from forecast.config.schema import ForecastConfig
from forecast.models.tft.attention import InterpretableMultiHeadAttention
from forecast.models.tft.grn import GRN, _GatedLinearUnit
from forecast.models.tft.local import LocalProcessor
from forecast.models.tft.static_enc import StaticCovariateEncoders
from forecast.models.tft.vsn import VariableSelectionNetwork


class _GateAddNorm(nn.Module):
    """Gated skip connection + LayerNorm. Used after attention and FFN.

    ``GateAddNorm(x, skip) = LayerNorm( skip + GLU(x) )``

    ``x`` is typically the newly-produced tensor (e.g. attention output
    or feed-forward output); ``skip`` is the residual from before the
    sub-block. Both must share the same shape.
    """

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.glu = _GatedLinearUnit(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return self.layer_norm(skip + self.glu(self.dropout(x)))


class TFTBody(nn.Module):
    """Temporal Fusion Transformer body.

    Consumes projected past-observed modality tensors (already width
    ``H``) and static categorical features, and produces the full
    sequence output plus interpretability signals.

    Parameters
    ----------
    cfg
        v3 :class:`~forecast.config.schema.ForecastConfig`. Consulted
        for ``hidden_dim``, ``n_heads``, ``n_lstm_layers``, ``dropout``,
        ``n_tickers``, ``n_sectors``.
    n_past_modalities
        Number of active past-observed modality streams that the VSN
        will select over. Must match the number of tensors passed to
        ``past_projected`` in :meth:`forward`.
    static_cardinalities
        Per-static-categorical cardinalities. Default from spec:
        ``[55, 11]`` for (ticker, sector).

    Attributes
    ----------
    hidden_dim, lookback, dropout, n_heads
        Stored for downstream reference.
    """

    def __init__(
        self,
        cfg: ForecastConfig,
        n_past_modalities: int,
        static_cardinalities: list[int],
    ) -> None:
        super().__init__()
        if n_past_modalities < 1:
            raise ValueError(
                f"n_past_modalities must be >= 1; got {n_past_modalities}"
            )

        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.lookback = cfg.lookback
        self.dropout_p = cfg.dropout
        self.n_heads = cfg.n_heads
        self.n_past_modalities = n_past_modalities

        H = cfg.hidden_dim

        # 1. Static covariate encoders: produce {c_s, c_c, c_h, c_e}.
        self.static_encoders = StaticCovariateEncoders(
            static_cardinalities=static_cardinalities,
            hidden_dim=H,
            dropout=cfg.dropout,
        )

        # 2. Past-input VSN: selects across the projected modalities.
        self.past_vsn = VariableSelectionNetwork(
            n_variables=n_past_modalities,
            input_dim=H,
            hidden_dim=H,
            context_dim=H,  # c_s from static encoders
            dropout=cfg.dropout,
        )

        # 3. LSTM local processor.
        self.local_processor = LocalProcessor(
            hidden_dim=H,
            n_lstm_layers=cfg.n_lstm_layers,
            dropout=cfg.dropout,
        )

        # 4. Gated skip over the LSTM (paper's "locality enhancement").
        self.post_lstm_gate = _GateAddNorm(hidden_dim=H, dropout=cfg.dropout)

        # 5. Static enrichment GRN: per-timestep, conditioned on c_e.
        self.enrichment_grn = GRN(
            input_dim=H,
            hidden_dim=H,
            output_dim=H,
            context_dim=H,
            dropout=cfg.dropout,
        )

        # 6. Interpretable multi-head attention.
        self.attention = InterpretableMultiHeadAttention(
            hidden_dim=H,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )

        # 7. Gated skip post-attention.
        self.post_attn_gate = _GateAddNorm(hidden_dim=H, dropout=cfg.dropout)

        # 8. Position-wise feed-forward (GRN, no context).
        self.position_ff = GRN(
            input_dim=H,
            hidden_dim=H,
            output_dim=H,
            context_dim=None,
            dropout=cfg.dropout,
        )

        # 9. Gated skip post-FFN.
        self.post_ff_gate = _GateAddNorm(hidden_dim=H, dropout=cfg.dropout)

        # Cached causal mask tensor (on cpu; moved to device lazily).
        self.register_buffer(
            "_causal_mask",
            InterpretableMultiHeadAttention.build_causal_mask(cfg.lookback),
            persistent=False,
        )

    def _get_causal_mask(self, L: int, device: torch.device) -> Tensor:
        """Return a causal mask of shape ``(L, L)`` on the given device.

        If the cached mask matches the requested ``L`` we reuse it; else
        we rebuild on the fly. Both paths are deterministic.
        """
        buf = getattr(self, "_causal_mask", None)
        if buf is not None and buf.shape == (L, L):
            return buf.to(device)
        return InterpretableMultiHeadAttention.build_causal_mask(L, device=device)

    def forward(
        self,
        past_projected: dict[str, Tensor],
        static_categorical: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Parameters
        ----------
        past_projected
            Dict from modality name to tensor of shape ``(B, L, H)``.
            The number of keys must equal ``n_past_modalities``.
            Iteration order is the sorted key order for determinism.
        static_categorical
            Integer tensor of shape ``(B, n_static)``.

        Returns
        -------
        dict
            Keys ``'decoder_out'``, ``'last_hidden'``,
            ``'vsn_weights_past'``, ``'attention_weights'``.
        """
        if len(past_projected) != self.n_past_modalities:
            raise ValueError(
                f"past_projected has {len(past_projected)} modalities; "
                f"expected {self.n_past_modalities}."
            )

        # 1. Static contexts.
        static_ctxs = self.static_encoders(static_categorical)
        c_s = static_ctxs["c_s"]  # (B, H)
        c_h = static_ctxs["c_h"]
        c_c = static_ctxs["c_c"]
        c_e = static_ctxs["c_e"]

        # 2. Past VSN. The VSN wants a per-timestep context:
        # broadcast c_s from (B, H) to (B, L, H) so every timestep
        # receives the same static selection context.
        # Use a dict for stable key ordering.
        selected_past, vsn_weights = self.past_vsn(
            past_projected,
            context=c_s.unsqueeze(1).expand(-1, self.lookback, -1),
        )
        # selected_past: (B, L, H); vsn_weights: (B, L, n_past_modalities)

        # Keep the pre-LSTM tensor as the post-LSTM skip residual.
        pre_lstm = selected_past

        # 3. LSTM local processing.
        lstm_out = self.local_processor(selected_past, c_h=c_h, c_c=c_c)

        # 4. Gated skip over LSTM output: GateAddNorm(lstm_out, skip=pre_lstm).
        # This lets the network bypass the LSTM if useful.
        post_lstm = self.post_lstm_gate(lstm_out, pre_lstm)

        # 5. Static enrichment: per-timestep GRN with context c_e.
        # GRN expects c to broadcast; unsqueeze to (B, 1, H).
        enriched = self.enrichment_grn(post_lstm, c_e.unsqueeze(1))

        # 6. Interpretable MHA (causal self-attention).
        causal_mask = self._get_causal_mask(self.lookback, enriched.device)
        attn_out, attn_weights = self.attention(enriched, causal_mask=causal_mask)

        # 7. Gated skip post-attention: skip = enriched.
        post_attn = self.post_attn_gate(attn_out, enriched)

        # 8. Position-wise feed-forward (GRN).
        ff_out = self.position_ff(post_attn)

        # 9. Gated skip post-FFN: final skip goes all the way back to
        # the post-LSTM output (the reference implementation uses the
        # LSTM output as the final residual anchor so attention + FFN
        # are a "refinement" on top of the local representation).
        decoder_out = self.post_ff_gate(ff_out, post_lstm)

        return {
            "decoder_out": decoder_out,
            "last_hidden": decoder_out[:, -1, :],
            "vsn_weights_past": vsn_weights,
            "attention_weights": attn_weights,
        }


__all__ = ["TFTBody"]
