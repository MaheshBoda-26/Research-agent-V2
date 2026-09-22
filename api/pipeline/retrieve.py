"""Stage 1 — retrieval (wiring).

The implementation lives in :mod:`sources.arxiv_client` (plan A.5) so the
transport, cache, and query-tier logic are importable without the pipeline
package. This module is the stage-shaped façade: the orchestrator (Phase 8)
imports ``fetch_candidates`` from here and wraps it in ``asyncio.to_thread``.
"""

from __future__ import annotations

from sources.arxiv_client import (
    ProgressCallback,
    QueryResolver,
    RetrievalError,
    RetrievalOffline,
    RetrievalThrottled,
    cache_key,
    candidate_queries,
    dedupe_by_paper_id,
    fetch_candidates,
    make_client,
    normalize_paper_id,
    read_cache,
    resolve_query,
    result_to_paper,
    to_paper,
    version_suffix,
    write_cache,
)

__all__ = [
    "ProgressCallback",
    "QueryResolver",
    "RetrievalError",
    "RetrievalOffline",
    "RetrievalThrottled",
    "cache_key",
    "candidate_queries",
    "dedupe_by_paper_id",
    "fetch_candidates",
    "make_client",
    "normalize_paper_id",
    "read_cache",
    "resolve_query",
    "result_to_paper",
    "to_paper",
    "version_suffix",
    "write_cache",
]
