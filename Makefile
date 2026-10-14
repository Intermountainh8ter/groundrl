PY ?= python
PIP ?= pip

.PHONY: install install-train test lint cov data audit baselines report dry-run diagrams docker clean all

all: lint test

install:
	$(PIP) install -e ".[dev]"

install-train:
	$(PIP) install -e ".[train,dev]"

test:
	pytest -q

lint:
	ruff check .

cov:
	pytest -q --cov --cov-report=term-missing

data:
	groundrl build-tasks
	groundrl check-data

audit:
	$(PY) scripts/reward_audit.py --split test
	$(PY) scripts/reward_audit.py --split val

baselines:
	groundrl baseline-eval --split test

report:
	$(PY) scripts/make_report.py

dry-run:
	$(PY) train/sft.py --config train/configs/qwen3-0.6b-24gb.json --dry-run
	$(PY) train/grpo.py --config train/configs/qwen3-0.6b-24gb.json --dry-run
	$(PY) train/grpo.py --config train/configs/qwen3-1.7b-12gb.json --dry-run

diagrams:
	bash scripts/render_diagrams.sh

docker:
	docker build -t groundrl .

clean:
	rm -rf .pytest_cache .ruff_cache .coverage build dist src/*.egg-info outputs
