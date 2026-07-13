"""Quantile head + direction / volatility derivations (see the architecture spec).

:class:`QuantileHead` is the output projection: a single
``Linear(H, n_quantiles)`` that consumes the TFT body's last-timestep
hidden state and produces the predicted quantile vector for the return
target.

Two helper functions derive *direction* and *volatility* from the
quantile vector so that downstream evaluation can report v1/v2-style
metrics (MCC, F1, volatility R²) from a single quantile-trained model:

* :func:`derive_direction` — turns ``q_median`` into a 2-class
  logit pair ``[-q_median, q_median]`` whose argmax is ``sign(q_median)``
  and whose softmax is a smooth probability useful for the CE auxiliary.

* :func:`derive_volatility` — returns a Gaussian-implied standard
  deviation from the quantile spread (``q_hi - q_lo``) at a given
  coverage level. Clamps the spread to a tiny positive before dividing
  so untrained networks (with potentially unsorted quantiles) do not
  crash the forward pass.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class QuantileHead(nn.Module):
    """Linear projection from hidden_dim to n_quantiles.

    Parameters
    ----------
    hidden_dim
        Input width ``H`` (== TFT body hidden_dim).
    quantiles
        Strictly-increasing tuple of quantiles in (0, 1). Stored for
        reference; the head itself just outputs ``len(quantiles)`` units.
    """

    def __init__(
        self,
        hidden_dim: int,
        quantiles: tuple[float, ...],
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {hidden_dim}")
        if len(quantiles) < 2:
            raise ValueError(
                f"quantiles must have at least 2 values; got {quantiles}"
            )

        self.hidden_dim = hidden_dim
        self.quantiles = tuple(float(q) for q in quantiles)
        self.n_quantiles = len(quantiles)
        self.fc = nn.Linear(hidden_dim, self.n_quantiles)

    def forward(self, h: Tensor) -> Tensor:
        """Project hidden state to quantile vector.

        Parameters
        ----------
        h
            Tensor of shape ``(B, H)``.

        Returns
        -------
        Tensor
            Tensor of shape ``(B, n_quantiles)``.
        """
        if h.dim() != 2 or h.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"QuantileHead expects (B, H={self.hidden_dim}); got "
                f"{tuple(h.shape)}"
            )
        return self.fc(h)


def derive_direction(q: Tensor, median_idx: int) -> Tensor:
    """Produce 2-class direction logits from the median quantile.

    The logits are ``[-q_median, q_median]``, which is strictly
    monotone: larger ``q_median`` → larger up-class logit. This is the
    smoothest derivation that preserves gradient flow to the quantile
    head and recovers ``sign(q_median)`` via ``argmax``.

    Parameters
    ----------
    q
        Tensor of shape ``(B, n_quantiles)``.
    median_idx
        Index of the median (0.5) quantile along the last axis.

    Returns
    -------
    Tensor
        Logits of shape ``(B, 2)``. Column 0 = down, column 1 = up.
    """
    if q.dim() != 2:
        raise ValueError(
            f"derive_direction expects (B, n_q); got {tuple(q.shape)}"
        )
    if not 0 <= median_idx < q.shape[-1]:
        raise ValueError(
            f"median_idx {median_idx} out of range for last-dim "
            f"{q.shape[-1]}"
        )
    q_med = q[:, median_idx]
    return torch.stack([-q_med, q_med], dim=-1)


def derive_volatility(
    q: Tensor,
    lo_idx: int,
    hi_idx: int,
    coverage: float = 0.8,
) -> Tensor:
    """Gaussian-implied sigma from a lower / upper quantile pair.

    Computes ``sigma = (q_hi - q_lo) / (2 * z)`` where ``z`` is the
    standard-Normal quantile at ``(1 + coverage) / 2``. At
    ``coverage=0.8`` → ``z ≈ 1.2816``, divisor ≈ ``2.5631``.

    The spread is clamped to a tiny positive value (``1e-6``) before
    division so an untrained network whose quantile output may be
    unsorted does not crash or produce ``-inf``.

    Parameters
    ----------
    q
        Tensor of shape ``(B, n_quantiles)``.
    lo_idx, hi_idx
        Indices of the lower and upper quantile for the band. E.g. 0
        and 4 for a 5-quantile ``(0.1, ..., 0.9)`` set.
    coverage
        Intended coverage of the band ``[q_lo, q_hi]``. Must be in
        (0, 1) and consistent with the (lo_idx, hi_idx) pair — e.g.
        ``0.8`` for ``(0.1, 0.9)``.

    Returns
    -------
    Tensor
        Tensor of shape ``(B,)`` with positive Gaussian-implied sigma.
    """
    if q.dim() != 2:
        raise ValueError(
            f"derive_volatility expects (B, n_q); got {tuple(q.shape)}"
        )
    if lo_idx == hi_idx:
        raise ValueError(
            f"lo_idx ({lo_idx}) and hi_idx ({hi_idx}) must differ."
        )
    if not 0 < coverage < 1:
        raise ValueError(f"coverage must lie in (0, 1); got {coverage}")

    spread = q[:, hi_idx] - q[:, lo_idx]
    # Clamp to prevent negatives / zeros from unsorted quantile outputs.
    spread = spread.clamp(min=1e-6)
    # Inverse standard-Normal CDF at (1 + coverage) / 2.
    # erfinv(2p - 1) * sqrt(2) = inv Normal CDF at p.
    # At p = (1 + 0.8) / 2 = 0.9, returns ~1.2816.
    z = math.sqrt(2.0) * _erfinv((1.0 + coverage) - 1.0)
    return spread / (2.0 * z)


def _erfinv(x: float) -> float:
    """Inverse error function evaluated at a Python float.

    Uses :func:`math.erfc` plus a Newton step. Acceptable accuracy for
    the handful of coverage levels we use; exact enough to keep
    derive_volatility deterministic across devices.
    """
    # Delegate to torch for accuracy; we only evaluate at a constant
    # during construction, not per-batch.
    return float(torch.erfinv(torch.tensor(x, dtype=torch.float64)).item())


__all__ = ["QuantileHead", "derive_direction", "derive_volatility"]
