"""mmfp — Multimodal Financial Prediction platform (v2).

The classifier-shaped experimental platform for the controlled multi-source
ablation: leveraging heterogeneous data sources for next-day financial market
direction prediction.

This package provides:

* config-first experiment definitions (pydantic v2, TOML-backed);
* seeded, deterministic training;
* modular encoders, fusions, heads and losses;
* clean separation between feature engineering, datasets, and model code;
* a single ``run_one_experiment`` entry point writing ``ResultRecord`` rows.
"""

__version__ = "2.0.0-dev"
