# AI Research Landscape Agent V2 — the one-command loop.
# `make help` lists every target. Later-phase targets are stubbed: running one
# tells you which phase implements it instead of failing mysteriously, and
# `make -n <target>` always prints the intended command.

SHELL := /bin/sh
UV  := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
PY  := .venv/bin/python
API_PORT ?= 8000

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

.PHONY: bootstrap
bootstrap: ## Create .venv and install the pinned Python environment (+ web deps when web/ exists)
	$(UV) venv --python 3.11 .venv
	$(UV) pip install --python .venv/bin/python -r api/requirements.txt
	@if [ -f web/package.json ]; then cd web && npm install; else echo "web/: scaffolded in Phase 9 — skipping npm install"; fi

.PHONY: install
install: bootstrap ## Alias for bootstrap

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

.PHONY: api
api: ## Serve the API on http://127.0.0.1:$(API_PORT) (auto-reload)
	cd api && ../$(PY) -m uvicorn main:app --host 127.0.0.1 --port $(API_PORT) --reload

.PHONY: web
web: ## Serve the UI on http://localhost:3000 (Phase 9)
	@if [ -f web/package.json ]; then cd web && npm run dev; else echo "web/ arrives in Phase 9 (plan: Task 9.1). Nothing to serve yet."; fi

.PHONY: dev
dev: ## Run API + UI together
	@$(MAKE) api & $(MAKE) web; wait

# ---------------------------------------------------------------------------
# Test / quality
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Offline unit + integration suite (<15s, no network, no model downloads)
	$(PY) -m pytest api/tests -q

.PHONY: test-live
test-live: ## Opt-in live smoke tests (network + real keys)
	RUN_LIVE_TESTS=1 $(PY) -m pytest api/tests -m live

.PHONY: lint
lint: ## ruff (api + scripts); next lint joins in Phase 9
	$(PY) -m ruff check api scripts

.PHONY: typecheck
typecheck: ## mypy (api); tsc --noEmit joins in Phase 9
	$(PY) -m mypy api

.PHONY: types
types: ## Regenerate web/lib/api-types.ts from the OpenAPI schema (Phase 8); fails on drift
	@if [ -f scripts/generate_types.py ]; then $(PY) scripts/generate_types.py --check; else echo "types: implemented in Phase 8 (Task 8.5)"; fi

.PHONY: audit
audit: ## pip-audit + npm audit (supply-chain gate)
	@if $(PY) -m pip_audit --version >/dev/null 2>&1; then $(PY) -m pip_audit -r api/requirements.txt; else echo "pip-audit not installed; run: $(UV) pip install --python .venv/bin/python pip-audit"; fi
	@if [ -f web/package.json ]; then cd web && npm audit; fi

# ---------------------------------------------------------------------------
# Verification / evaluation
# ---------------------------------------------------------------------------

.PHONY: verify-llm
verify-llm: ## Probe the LLM catalogue and run ONE real completion against LLM_MODEL
	$(PY) scripts/verify_llm.py

.PHONY: probe-sources
probe-sources: ## Live-probe arXiv / S2 / OpenAlex / full-text behaviour (Phase 2)
	@if [ -f scripts/probe_sources.py ]; then $(PY) scripts/probe_sources.py; else echo "probe-sources: implemented in Phase 2 (Task 2.1)"; fi

.PHONY: eval
eval: ## Gold-set metrics + regression gates (Phase 10)
	@if [ -f evals/run.py ]; then $(PY) -m evals.run; else echo "eval: implemented in Phase 10 (Task 10.2)"; fi

.PHONY: demo
demo: ## Scripted end-to-end demo run (Phase 13)
	@if [ -f scripts/demo.py ]; then $(PY) scripts/demo.py; else echo "demo: implemented in Phase 13"; fi

# ---------------------------------------------------------------------------
# Operations (data lifecycle, §13.4–13.7)
# ---------------------------------------------------------------------------

.PHONY: doctor
doctor: ## Mark interrupted 'running' landscapes failed; flag stale fixtures (Phase 8/13)
	@if [ -f scripts/doctor.py ]; then $(PY) scripts/doctor.py; else echo "doctor: implemented in Phase 8 (§13.5 recovery) "; fi

.PHONY: cost
cost: ## Spend by day/stage/model and per-landscape average from llm_calls (Phase 8)
	@if [ -f scripts/cost.py ]; then $(PY) scripts/cost.py; else echo "cost: implemented in Phase 8 (Task 8.6)"; fi

.PHONY: prune
prune: ## Drop expired caches + old run events, then VACUUM (Phase 8)
	@if [ -f scripts/prune.py ]; then $(PY) scripts/prune.py $(PRUNE_ARGS); else echo "prune: implemented in Phase 8 (§13.6)"; fi

.PHONY: backup
backup: ## Timestamped copy of data/landscapes.db (Phase 13)
	@if [ -f scripts/backup.py ]; then $(PY) scripts/backup.py; else echo "backup: implemented in Phase 13"; fi

.PHONY: refresh-fixtures
refresh-fixtures: ## Re-capture upstream fixtures, honouring rate limits (Phase 2)
	@if [ -f scripts/refresh_fixtures.py ]; then $(PY) scripts/refresh_fixtures.py; else echo "refresh-fixtures: implemented in Phase 2 (§11.3)"; fi

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

.PHONY: db-reset
db-reset: ## Delete the local database (landscapes, caches, extractions — everything)
	rm -f data/landscapes.db data/landscapes.db-wal data/landscapes.db-shm
	@echo "data/landscapes.db removed; it is recreated on next run"

.PHONY: clean
clean: ## Remove caches and build artefacts (keeps .venv and data/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true

.PHONY: help
help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
