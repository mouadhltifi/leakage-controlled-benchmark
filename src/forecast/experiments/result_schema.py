"""v3 result-record schema — v2's :class:`ResultRecord` + two quantile fields.

Milestone 5 extends v2's :class:`mmfp.experiments.result_schema.ResultRecord`
with the two v3-native calibration diagnostics from the architecture spec:

* ``quantile_coverage_80`` — fraction of test samples falling in the
  ``[q_0.1, q_0.9]`` band. Target ≈ 0.8 for a well-calibrated model;
  catastrophic miscalibration (< 0.5 or > 0.95) is flagged in the run log.
* ``quantile_interval_width`` — mean band width ``q_0.9 − q_0.1``,
  normalised by ``std(y)``. Detects quantile collapse (width ≈ 0) early.

Implementation uses dataclass inheritance so v2's CSV writers
(:func:`append_result`, :func:`load_results`) and the fingerprint-based
resume machinery (:func:`record_fingerprint`, :func:`fingerprint_from_row`)
work unchanged on v3 records. All the v2 helpers are re-exported here so
v3 callers import from a single location:

    from forecast.experiments.result_schema import (
        ForecastResultRecord, append_result, config_hash,
        record_fingerprint, fingerprint_from_row, load_results,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

# Re-exports: v3 callers stick to ``forecast.experiments.result_schema`` so
# they don't have to reach across to mmfp. Behaviour is identical because
# the CSV schema is driven by ``dataclasses.fields(record)`` — adding two
# fields on the v3 subclass extends the CSV header and nothing else.
from mmfp.experiments.result_schema import (
    ResultRecord,
    append_result,
    config_hash,
    fingerprint_from_row,
    load_results,
    record_fingerprint,
)


@dataclass
class ForecastResultRecord(ResultRecord):
    """v3 per-experiment result row.

    Inherits every field of :class:`ResultRecord` and appends two
    v3-native quantile-calibration diagnostics (see the architecture spec).

    Because :class:`ResultRecord` uses the ``@dataclass`` decorator and
    this subclass also uses ``@dataclass``, :func:`dataclasses.fields`
    returns the full ordered field list (parent first, child last). The
    CSV header produced by :meth:`to_row` therefore looks identical to
    the v2 header up to the two trailing columns, so v2-consumer code
    that reads only known columns continues to work and new code can
    access the two new columns by name.

    Attributes
    ----------
    quantile_coverage_80
        Fraction of test samples with ``q_0.1 <= y_true <= q_0.9`` on
        the test split. ``None`` if not measured (reserved for
        non-quantile future architectures).
    quantile_interval_width
        Mean of ``q_0.9 - q_0.1`` on the test split, normalised by
        ``std(y_true)`` (if positive; raw mean otherwise). ``None`` if
        not measured.
    """

    quantile_coverage_80: float | None = None
    quantile_interval_width: float | None = None


__all__ = [
    "ForecastResultRecord",
    "ResultRecord",
    "append_result",
    "config_hash",
    "fingerprint_from_row",
    "load_results",
    "record_fingerprint",
]
