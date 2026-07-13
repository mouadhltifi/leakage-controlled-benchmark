"""Explicit fit/transform scaler for per-fold standardization.

Resolves an audit-critical finding that earlier standardization was either
absent or silently pooling train+val rows (look-ahead leakage). The
:class:`FittedScaler` enforces:

1. **Fit once.** A second call to :meth:`FittedScaler.fit` raises.
   Pools cannot accidentally grow.
2. **Transform requires fit.** :meth:`FittedScaler.transform` before
   :meth:`FittedScaler.fit` raises.
3. **Train-only stats.** The scaler is always fit on a *train* block;
   val/test rows are transformed with train parameters. The leakage
   test (``mmfp/tests/leakage/test_scaler_fit_on_train_only.py``)
   enforces this behavior.
4. **Typed column handling.**

   * ``log1p_cols`` receive ``np.log1p`` before standardization.
     Negative inputs raise a ``ValueError`` (``log1p`` is defined for
     ``x >= -1`` but finance count features are ``>= 0``).
   * ``passthrough_cols`` skip both transformations entirely; they
     pass through unchanged. Intended for binary indicators such as
     ``has_news``, ``has_social``, ``is_fomc_window``.
   * All other columns receive a plain z-score (mean 0, std 1 on the
     training set).

The scaler is state-serializable (via ``__getstate__`` / ``__setstate__``)
for per-fold artifact caching.

See spec Section 3.3 "Standardization contract" for the authoritative
contract and defaults per modality.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Floor for the learnt per-column std. Guards against divide-by-zero on
#: constant train columns (e.g., a flag that is always 0 in a fold).
_STD_EPSILON: float = 1e-8


class FittedScaler:
    """Thin scaler with explicit fit/transform discipline.

    Parameters
    ----------
    log1p_cols
        Column names receiving ``np.log1p`` before z-score. Empty by
        default.
    passthrough_cols
        Column names skipping all transformations. Empty by default.

    Attributes
    ----------
    is_fitted : bool
        ``True`` after a successful :meth:`fit`.
    feature_names_in_ : list[str] | None
        Column names seen at fit time. ``None`` before fit.
    feature_names_out_ : list[str]
        Output column names; equal to ``feature_names_in_`` (the scaler
        preserves column order and does not add or drop columns).
    log1p_cols : tuple[str, ...]
        Immutable copy of the constructor-supplied list.
    passthrough_cols : tuple[str, ...]
        Immutable copy of the constructor-supplied list.
    mean_ : numpy.ndarray | None
        Per-column fitted mean (``0.0`` for passthrough columns).
    scale_ : numpy.ndarray | None
        Per-column fitted std (``1.0`` for passthrough columns;
        clamped to ``max(std, _STD_EPSILON)`` otherwise).

    Notes
    -----
    Both ``mean_`` and ``scale_`` are computed *after* the ``log1p``
    step for columns in :attr:`log1p_cols`. This matches the standard
    "log then normalize" pipeline for count features.
    """

    def __init__(
        self,
        log1p_cols: Iterable[str] = (),
        passthrough_cols: Iterable[str] = (),
    ) -> None:
        log1p = tuple(log1p_cols)
        passthrough = tuple(passthrough_cols)

        overlap = set(log1p) & set(passthrough)
        if overlap:
            raise ValueError(
                "log1p_cols and passthrough_cols must be disjoint; "
                f"overlap: {sorted(overlap)}"
            )

        self.log1p_cols: tuple[str, ...] = log1p
        self.passthrough_cols: tuple[str, ...] = passthrough

        self.is_fitted: bool = False
        self.feature_names_in_: list[str] | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API: fit / transform / fit_transform.
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "FittedScaler":
        """Learn the transformation parameters from train rows only.

        Parameters
        ----------
        X
            Training DataFrame. Column set defines :attr:`feature_names_in_`
            and the output column order.

        Returns
        -------
        FittedScaler
            ``self`` for method chaining.

        Raises
        ------
        RuntimeError
            If the scaler is already fitted. Construct a new instance to
            refit (this is intentional — silent refits were the v1 bug).
        ValueError
            If ``X`` is empty, not a DataFrame, or misses declared
            ``log1p_cols`` / ``passthrough_cols``.
        """
        if self.is_fitted:
            raise RuntimeError(
                "FittedScaler.fit called twice. Create a new instance "
                "to refit - silent refits are the exact leakage bug "
                "this class prevents."
            )
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                f"FittedScaler.fit requires a DataFrame, got {type(X).__name__}"
            )
        if X.shape[0] == 0:
            raise ValueError("FittedScaler.fit called with empty DataFrame")

        self._validate_column_declarations(X)

        columns = list(X.columns)
        n_cols = len(columns)
        col_to_idx = {c: i for i, c in enumerate(columns)}

        arr = X.to_numpy(dtype=np.float64, copy=True)

        # Apply log1p to declared columns (in-place on the working copy).
        for col in self.log1p_cols:
            j = col_to_idx[col]
            col_vals = arr[:, j]
            if np.any(col_vals < 0):
                neg_count = int(np.sum(col_vals < 0))
                raise ValueError(
                    f"log1p_cols column {col!r} has {neg_count} "
                    "negative value(s). log1p is finite only for "
                    "x >= -1 and the convention for count features is "
                    "x >= 0. Check upstream feature computation."
                )
            arr[:, j] = np.log1p(col_vals)

        # Initialize mean=0, scale=1 for passthrough columns; that makes
        # the transform an identity on them.
        mean = np.zeros(n_cols, dtype=np.float64)
        scale = np.ones(n_cols, dtype=np.float64)

        passthrough_set = set(self.passthrough_cols)
        for j, col in enumerate(columns):
            if col in passthrough_set:
                continue
            col_vals = arr[:, j]
            m = float(np.nanmean(col_vals))
            s = float(np.nanstd(col_vals))  # population std (ddof=0)
            mean[j] = m
            scale[j] = max(s, _STD_EPSILON)

        self.feature_names_in_ = columns
        self.mean_ = mean
        self.scale_ = scale
        self.is_fitted = True

        log.debug(
            "FittedScaler.fit on %d rows / %d cols (%d log1p, %d passthrough)",
            X.shape[0],
            n_cols,
            len(self.log1p_cols),
            len(self.passthrough_cols),
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted transformation to ``X``.

        Parameters
        ----------
        X
            Any DataFrame with the same columns as the fit data (order
            is normalised; extras or missing columns raise).

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_features)`` float32 array. Column order matches
            :attr:`feature_names_out_`.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If ``X`` misses any column seen at fit time or has extra
            columns (extras are flagged to catch schema drift).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "FittedScaler.transform called before fit. "
                "Call fit(train_df) first - never transform val/test "
                "without a prior fit on train-only rows."
            )
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                f"FittedScaler.transform requires a DataFrame, "
                f"got {type(X).__name__}"
            )

        assert self.feature_names_in_ is not None  # noqa: S101 — for type-narrowing
        assert self.mean_ is not None
        assert self.scale_ is not None

        expected_cols = self.feature_names_in_
        seen_cols = list(X.columns)

        missing = [c for c in expected_cols if c not in seen_cols]
        extras = [c for c in seen_cols if c not in expected_cols]
        if missing or extras:
            raise ValueError(
                "FittedScaler.transform column mismatch. "
                f"Missing: {missing}. Extras: {extras}. "
                f"Fit columns: {expected_cols}."
            )

        # Reorder and copy so callers can't accidentally mutate X.
        arr = X[expected_cols].to_numpy(dtype=np.float64, copy=True)

        # Apply log1p to declared columns.
        col_to_idx = {c: i for i, c in enumerate(expected_cols)}
        for col in self.log1p_cols:
            j = col_to_idx[col]
            col_vals = arr[:, j]
            if np.any(col_vals < 0):
                neg_count = int(np.sum(col_vals < 0))
                raise ValueError(
                    f"log1p_cols column {col!r} has {neg_count} "
                    "negative value(s) in transform input."
                )
            arr[:, j] = np.log1p(col_vals)

        # Z-score using train stats. For passthrough columns mean=0,
        # scale=1, so the subtract/divide is a no-op.
        arr = (arr - self.mean_) / self.scale_
        return arr.astype(np.float32, copy=False)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit on ``X_train`` and return the train-block transform.

        Convenience for the common pattern. Equivalent to
        ``self.fit(X_train).transform(X_train)``.
        """
        return self.fit(X_train).transform(X_train)

    # ------------------------------------------------------------------
    # Properties.
    # ------------------------------------------------------------------

    @property
    def feature_names_out_(self) -> list[str]:
        """Output feature names (same order and names as fit input).

        Raises
        ------
        RuntimeError
            If accessed before :meth:`fit`.
        """
        if not self.is_fitted:
            raise RuntimeError("feature_names_out_ requires a fitted scaler")
        assert self.feature_names_in_ is not None  # noqa: S101
        return list(self.feature_names_in_)

    # ------------------------------------------------------------------
    # Private helpers.
    # ------------------------------------------------------------------

    def _validate_column_declarations(self, X: pd.DataFrame) -> None:
        """Ensure declared ``log1p_cols`` / ``passthrough_cols`` exist in ``X``."""
        cols = set(X.columns)
        missing_log = [c for c in self.log1p_cols if c not in cols]
        missing_pass = [c for c in self.passthrough_cols if c not in cols]
        if missing_log:
            raise ValueError(
                f"log1p_cols not present in fit data: {missing_log}"
            )
        if missing_pass:
            raise ValueError(
                f"passthrough_cols not present in fit data: {missing_pass}"
            )

    # ------------------------------------------------------------------
    # Serialization support: explicit state for round-trippable caching.
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        return {
            "log1p_cols": self.log1p_cols,
            "passthrough_cols": self.passthrough_cols,
            "is_fitted": self.is_fitted,
            "feature_names_in_": self.feature_names_in_,
            "mean_": self.mean_,
            "scale_": self.scale_,
        }

    def __setstate__(self, state: dict) -> None:
        self.log1p_cols = state["log1p_cols"]
        self.passthrough_cols = state["passthrough_cols"]
        self.is_fitted = state["is_fitted"]
        self.feature_names_in_ = state["feature_names_in_"]
        self.mean_ = state["mean_"]
        self.scale_ = state["scale_"]


__all__ = ["FittedScaler"]
