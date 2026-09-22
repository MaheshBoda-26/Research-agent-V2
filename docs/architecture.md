# Architecture (placeholder — Phase 1)

The normative architecture spec lives in `research-landscape-agent.md`:

- §3.1 repository layout
- §3.2 data model (schema v2: V1's 12 tables + citations / source_cache /
  llm_calls, plus the D1/D2/D3/D8/D9 columns)
- §3.3 pipeline contract: fixed stage order
  `retrieval → enrichment → rerank → extraction → layout → synthesis`,
  the `StageEvent` shape, and the per-stage failure semantics table
- §3.4 wire protocol (POST + SSE, replay via `runs`)

This file will grow the pipeline diagram and stage-by-stage failure semantics
as the corresponding phases land. It is deliberately a pointer, not a copy —
one source of truth.
