CONFIG ?= configs/train.yaml
PYTHON ?= python

.PHONY: install data train test lint format serve drift pdf ci all

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m payguard.pipeline --config $(CONFIG) --stage data

train:
	$(PYTHON) -m payguard.pipeline --config $(CONFIG) --stage all

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

format:
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

serve:
	$(PYTHON) -m uvicorn payguard.serving.app:app --host 0.0.0.0 --port 8000

drift:
	$(PYTHON) -m payguard.pipeline --config $(CONFIG) --stage drift

pdf:
	$(PYTHON) scripts/render_explainer_pdf.py

ci:
	$(PYTHON) -m payguard.pipeline --config configs/ci.yaml

all: lint test train
