# Quality-gate entrypoints. Python prefers the local venv when present.
VENV_PY := $(wildcard .venv/bin/python)$(wildcard .venv/Scripts/python.exe)
PYTHON ?= $(if $(VENV_PY),$(VENV_PY),python)

.PHONY: eval eval-retrieval test lint migrate

# Phase 5 quality gate: exits non-zero when faithfulness/hit@5 drop below thresholds.
eval:
	$(PYTHON) -m eval.run_eval

# Retrieval-only pass (no LLM calls).
eval-retrieval:
	$(PYTHON) -m eval.run_eval --skip-generation

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

migrate:
	$(PYTHON) -m alembic upgrade head
