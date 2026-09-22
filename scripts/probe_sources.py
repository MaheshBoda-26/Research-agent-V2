#!/usr/bin/env python
"""Source health probe — Phase 3, Task 3.5.

One real call against each third-party source so a future incident is
diagnosed with a single command instead of a bisect:

    make probe-sources

Prints a four-line table (arXiv, Semantic Scholar, OpenAlex, arXiv HTML) with
status and latency. Exits 0 even when a source is down — this is a *probe*,
not a gate. Without ``S2_API_KEY`` the S2 line is expected to read 429.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

import httpx  # noqa: E402
from config import load_settings  # noqa: E402

TIMEOUT = 15.0


def probe(name: str, method: str, url: str, **kwargs: object) -> None:
    start = time.monotonic()
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.request(method, url, **kwargs)  # type: ignore[arg-type]
        elapsed = (time.monotonic() - start) * 1000
        print(f"{name:10s} {response.status_code}  {elapsed:7.0f} ms  {url}")
    except httpx.HTTPError as exc:
        elapsed = (time.monotonic() - start) * 1000
        print(f"{name:10s} ERR  {elapsed:7.0f} ms  {url}  ({exc})")


def main() -> int:
    settings = load_settings(require_llm=False)
    s2_headers = {"x-api-key": settings.s2_api_key} if settings.s2_api_key else {}

    probe("arxiv", "GET", "https://export.arxiv.org/api/query",
          params={"search_query": 'all:"retrieval-augmented generation"', "max_results": "1"})
    probe("s2", "GET", "https://api.semanticscholar.org/graph/v1/paper/arXiv:2005.11401",
          params={"fields": "title,citationCount"}, headers=s2_headers)
    mailto = {"mailto": settings.openalex_mailto} if settings.openalex_mailto else {}
    probe("openalex", "GET", "https://api.openalex.org/works",
          params={"search": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                  "per-page": "1", "select": "id,title,cited_by_count", **mailto})
    probe("arxiv-html", "GET", "https://arxiv.org/html/2005.11401v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
