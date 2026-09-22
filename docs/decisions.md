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
