"""Semantic Scholar Graph API — best-effort citation enrichment (Phase 3).

Plan §2.2's design consequence, restated: S2 is **best-effort, cached,
key-optional**. Unauthenticated access shares a 1000 RPS pool with the entire
internet and the batch endpoint 429s without a key, so every call here is
paced to 1 RPS behind a module-level lock, every failure degrades to ``None``
or ``[]`` rather than raising, and responses are cached for 7 days when a
connection is supplied.

Every function takes an optional ``httpx.Client`` so tests can inject an
``httpx.MockTransport`` and never touch the network.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from config import Settings
from store import get_cached, put_cached

logger = logging.getLogger(__name__)

_BASE = "https://api.semanticscholar.org"

#: Documented S2 limit with an API key; we honour it even unauthenticated,
#: since being polite costs nothing at 60 papers.
_MIN_INTERVAL_SECONDS: float = 1.0

#: Responses are cached for 7 days: citation counts move slowly.
_CACHE_TTL_SECONDS: int = 7 * 24 * 3600

_PAPER_FIELDS = "title,year,citationCount,referenceCount,externalIds,venue"
_REF_FIELDS = "title,year,externalIds"

_pacing_lock = threading.Lock()
_last_request_at = 0.0


def _pace() -> None:
    """Hold the module-level lock until at least 1s separates requests."""
    global _last_request_at
    with _pacing_lock:
        wait = _MIN_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _request(
    settings: Settings,
    path: str,
    *,
    client: httpx.Client | None = None,
) -> httpx.Response | None:
    """One paced GET. Retries a 429 honouring Retry-After, then gives up."""
    headers = {"x-api-key": settings.s2_api_key} if settings.s2_api_key else {}
    owned = client is None
    if owned:
        client = httpx.Client(base_url=_BASE, timeout=15.0)
    try:
        assert client is not None
        _pace()
        response = client.get(path, headers=headers)
        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", "2") or "2")
            wait = max(retry_after, 2.0)
            logger.warning("S2 throttled (429); retrying once after %.1fs", wait)
            time.sleep(wait)
            _pace()
            response = client.get(path, headers=headers)
        return response
    except httpx.HTTPError as exc:
        logger.warning("S2 request failed: %s", exc)
        return None
    finally:
        if owned:
            client.close()  # type: ignore[union-attr]


def fetch_paper(
    arxiv_id: str,
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    conn: Any | None = None,
) -> dict[str, Any] | None:
    """Paper metadata by arXiv id, or ``None`` on any failure (never raises)."""
    key = f"s2:paper:arXiv:{arxiv_id}"
    if conn is not None:
        cached = get_cached(conn, key, "semanticscholar")
        if cached is not None:
            return cached
    response = _request(
        settings,
        f"/graph/v1/paper/arXiv:{arxiv_id}?fields={_PAPER_FIELDS}",
        client=client,
    )
    if response is None or response.status_code != 200:
        logger.warning("S2 paper lookup for %s degraded: %s", arxiv_id,
                       response.status_code if response is not None else "unreachable")
        return None
    payload: dict[str, Any] = response.json()
    if conn is not None:
        put_cached(conn, key, "semanticscholar", payload, _CACHE_TTL_SECONDS)
    return payload


def fetch_references(
    arxiv_id: str,
    settings: Settings,
    limit: int = 100,
    *,
    client: httpx.Client | None = None,
    conn: Any | None = None,
) -> list[dict[str, Any]]:
    """Outgoing references, or ``[]`` on any failure (never raises)."""
    key = f"s2:refs:arXiv:{arxiv_id}:{limit}"
    if conn is not None:
        cached = get_cached(conn, key, "semanticscholar")
        if cached is not None:
            data = cached.get("data")
            if isinstance(data, list):
                return data  # type: ignore[no-any-return]
    response = _request(
        settings,
        f"/graph/v1/paper/arXiv:{arxiv_id}/references?fields={_REF_FIELDS}&limit={limit}",
        client=client,
    )
    if response is None or response.status_code != 200:
        return []
    payload: dict[str, Any] = response.json()
    if conn is not None:
        put_cached(conn, key, "semanticscholar", payload, _CACHE_TTL_SECONDS)
    return list(payload.get("data") or [])


def fetch_citations(
    arxiv_id: str,
    settings: Settings,
    limit: int = 100,
    *,
    client: httpx.Client | None = None,
    conn: Any | None = None,
) -> list[dict[str, Any]]:
    """Incoming citations, or ``[]`` on any failure (never raises)."""
    key = f"s2:cites:arXiv:{arxiv_id}:{limit}"
    if conn is not None:
        cached = get_cached(conn, key, "semanticscholar")
        if cached is not None:
            data = cached.get("data")
            if isinstance(data, list):
                return data  # type: ignore[no-any-return]
    response = _request(
        settings,
        f"/graph/v1/paper/arXiv:{arxiv_id}/citations?fields={_REF_FIELDS}&limit={limit}",
        client=client,
    )
    if response is None or response.status_code != 200:
        return []
    payload: dict[str, Any] = response.json()
    if conn is not None:
        put_cached(conn, key, "semanticscholar", payload, _CACHE_TTL_SECONDS)
    return list(payload.get("data") or [])


__all__ = ["fetch_citations", "fetch_paper", "fetch_references"]
