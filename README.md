# AI Research Landscape Agent — V2

Turn a plain-English topic into a **reading map** of the arXiv literature:
retrieval → citation enrichment → reranking → structured extraction → layout &
clustering → relationship graph + synthesis, streamed stage-by-stage to a local
web UI.

The full implementation plan — including the evidence base, the interface
contracts every module must satisfy, and the anti-pattern ledger — is in
[`research-landscape-agent.md`](research-landscape-agent.md). Read it before
changing anything.

## Quickstart

```bash
cp .env.example .env        # fill in NVIDIA_API_KEY (see .env.example)
make bootstrap              # uv venv + pinned deps into .venv/
make verify-llm             # prove the configured model with a REAL completion
make test                   # offline unit/integration suite (<15 s, no network)
make api                    # serve the API on http://127.0.0.1:8000 (/health)
make web                    # serve the UI on http://localhost:3000   (Phase 9)
```

`make help` lists every target. Later-phase targets (`eval`, `doctor`, `cost`,
`prune`, `backup`, `demo`, `probe-sources`, `types`) are stubbed until their
phase lands — running one tells you so instead of failing mysteriously.

## Layout

```
api/        FastAPI service + pipeline (config, models, store, llm/, sources/, pipeline/)
scripts/    operator entry points (verify_llm.py now; more per phase)
evals/      gold sets and the metrics runner (Phase 10)
web/        Next.js reading-map UI (Phase 9)
docs/       architecture notes, ADR log (decisions.md), measured perf (perf.md)
data/       gitignored runtime state: landscapes.db, caches, model weights
```

## Ground rules (short version)

- Tests assert contracts, evals assert quality. The offline suite never touches
  the network or a model; `make test-live` opts in.
- No LLM model slug is ever guessed — `make verify-llm` probes the provider
  catalogue and runs one real completion before a slug is trusted.
- `ARXIV_DELAY_SECONDS` can never go below 3.0 (arXiv Terms of Use); the config
  layer refuses it.
- Secrets live only in `.env`; `Settings.describe()` is the only config object
  that may be logged, and it redacts keys.
