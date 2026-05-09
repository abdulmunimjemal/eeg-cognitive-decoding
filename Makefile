# ---- eeg-cognitive ---------------------------------------------------
# One-command reproducibility.

PY ?= python3
PIP ?= pip
NODE ?= node

.PHONY: help install experiments figures notebooks all clean

help:
	@echo "  make install       install package + dependencies"
	@echo "  make experiments   run all (dataset, model) experiments"
	@echo "  make figures       regenerate figures"
	@echo "  make notebooks     execute notebooks end-to-end"
	@echo "  make all           experiments + figures + notebooks"
	@echo "  make clean         remove generated caches"

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

all: experiments figures notebooks

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .ipynb_checkpoints */.ipynb_checkpoints
