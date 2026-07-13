"""v3 forecasting package — proper TFT-based architecture for daily return forecasting.

This package replaces v2's model stack (the classifier model stack) with a Temporal
Fusion Transformer (TFT) forecasting body while reusing v2's data, feature,
trainer, experiment-runner and utility modules verbatim.

Scope is locked in the v3 scope doc; architecture is specified in
the architecture spec.

Milestone 1 (this release): scaffolding + ``ForecastConfig`` schema + cross-
field validators. No model code yet — see Section 9 of the architecture spec
for the implementation order.
"""

from __future__ import annotations

__version__ = "3.0.0-dev"

__all__ = ["__version__"]
