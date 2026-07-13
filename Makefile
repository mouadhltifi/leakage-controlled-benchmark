# Reproducibility artifact — convenience targets.
#
# All targets run from the artifact root with no manual PYTHONPATH (the analysis
# and figure scripts are pure pandas/matplotlib over the committed results/; the
# test target sets PYTHONPATH=src because the packages live under src/).
#
# Override the interpreter if needed:  make reproduce PYTHON=python3.11
PYTHON ?= python

.DEFAULT_GOAL := help

.PHONY: help reproduce smoke test figures clean-figures

help:  ## Show this help.
	@echo "Targets:"
	@echo "  make reproduce   Regenerate every headline number + all figures from committed CSVs"
	@echo "  make smoke       Fast subset: the RQ1 null table + macro re-run analysis (seconds)"
	@echo "  make test        Run both pytest suites (PYTHONPATH=src)"
	@echo "  make figures     Regenerate the paper figures into figures/"
	@echo "  make clean-figures  Remove generated figures/*.pdf"

reproduce:  ## Full headline reproduction from committed results (no GPU, no download).
	$(PYTHON) scripts/analysis/analyze_ablation.py
	$(PYTHON) scripts/analysis/analyze_macrolag.py
	$(PYTHON) scripts/analysis/analyze_volatility.py
	$(PYTHON) scripts/figures/make_paper_figures.py

smoke:  ## Fast sanity subset.
	$(PYTHON) scripts/analysis/analyze_ablation.py --section rq1
	$(PYTHON) scripts/analysis/analyze_macrolag.py

figures:  ## Regenerate the paper figures (vector PDFs into figures/).
	$(PYTHON) scripts/figures/make_paper_figures.py

test:  ## Run both test suites.
	PYTHONPATH=src $(PYTHON) -m pytest src/mmfp/tests
	PYTHONPATH=src $(PYTHON) -m pytest src/forecast/tests

clean-figures:  ## Remove generated figures.
	rm -f figures/*.pdf
