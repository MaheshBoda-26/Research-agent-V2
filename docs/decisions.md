# Decisions log

One entry per decision that looked reversible but wasn't (or vice versa).
Newest first.

## 2026-09-22 — LLM provider/model verification

- `make verify-llm` was run live against `https://integrate.api.nvidia.com/v1`
  (NVIDIA NIM, key from `.env`, never committed).
- Catalogue evidence: `GET /models` returned **82 models** (sample: yi-large,
  fuyu-8b, jamba-1.5-large, gemma-3-12b-it, nemotron family, gpt-oss-20b, …).
- Preferred candidates `nvidia/llama-3.1-nemotron-70b-instruct`,
  `nvidia/llama-3.1-nemotron-ultra-253b-v1`, `mistralai/mistral-large-2-instruct`
  and `mistralai/mixtral-8x22b-v0.1` are **listed** in the catalogue but return
  HTTP 404 ("Function … not found for account") on this NIM account — listed
  does not mean available. The verifier now cascades through candidates.
- **Chosen `LLM_MODEL=openai/gpt-oss-20b`** — first candidate that passed a live
  structured-output probe (`nvext.guided_json`, strict schema
  `{city, population_millions}`; validated output `{"city":"Paris, France",
  "population_millions":2.1}`, prompt=116, completion≈70 tokens).
- Consequence: all LLM calls use `extra_body={"nvext": {"guided_json": …}}`
  (V1-proven path) with schema validation + bounded repair as backstop; the
  `response_format` capability-discovery fallback remains for other providers.

## Cross-encoder API (Phase 4, Task 4.0) — recorded 2026-09-22

`pip freeze` pins `sentence-transformers==3.3.1`. `inspect.signature` on the
installed distribution prints:

- `CrossEncoder.__init__(self, model_name: str, num_labels: int = None, max_length: int = None, device: str | None = None, ...)`
- `CrossEncoder.predict(self, sentences, batch_size: int = 32, show_progress_bar: bool | None = None, ...)`
- `CrossEncoder.rank(self, query: str, documents: list[str], top_k: int | None = None, batch_size: int = 32, ...)`

**Consequence:** the installed 3.3.1 API still spells the truncation length
`max_length` — the v6 rename to `max_seq_length` has *not* landed in our pin.
`CrossEncoderReranker` therefore passes `max_length=` and, if the pin is ever
bumped, that kwarg is the thing to update (§2.5).

