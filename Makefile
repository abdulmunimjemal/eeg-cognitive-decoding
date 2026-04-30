# ---- eeg-cognitive ---------------------------------------------------
# One-command reproducibility.

PY ?= python3
PIP ?= pip
NODE ?= node

.PHONY: help install experiments figures notebooks slides brief all clean

help:
	@echo "  make install       install package + dependencies"
	@echo "  make experiments   run all four (dataset, model) experiments"
	@echo "  make figures       regenerate every figure"
	@echo "  make notebooks     execute every notebook end-to-end"
	@echo "  make slides        rebuild the 12-slide deck (requires Node + pptxgenjs)"
	@echo "  make brief         rebuild the 24-page teaching brief PDF"
	@echo "  make all           experiments + figures + notebooks + slides + brief"

install:
	$(PIP) install --break-system-packages -e .[dev]

experiments:
	$(PY) scripts/run_experiment.py motor_imagery     fbcsp
	$(PY) scripts/run_experiment.py motor_imagery     eegnet
	$(PY) scripts/run_experiment.py mental_arithmetic fbcsp
	$(PY) scripts/run_experiment.py mental_arithmetic eegnet

figures:
	$(PY) scripts/generate_figures.py
	$(PY) scripts/generate_styled_figures.py

notebooks:
	@for nb in notebooks/0*.ipynb; do \
	  echo "executing $$nb"; \
	  $(PY) -m jupyter nbconvert --to notebook --execute --inplace \
	    --ExecutePreprocessor.timeout=120 "$$nb"; \
	done

slides:
	cd slides && $(NODE) build_slides.js

brief:
	$(PY) brief/build_brief.py

all: experiments figures notebooks slides brief

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .ipynb_checkpoints */.ipynb_checkpoints
