"""Offline stand-ins for the arXiv API (Phase 2 tests).

arXiv has been returning 429/503 under its documented rate limit since early
2026, so no test may depend on it. These fakes let the retrieval tests exercise
the *real* ``arxiv.Client`` code path — URL formatting, pagination, retry
logic, and Atom parsing — by swapping only the HTTP session.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from arxiv import Result

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "arxiv_response.xml"


def fixture_bytes() -> bytes:
    return FIXTURE_PATH.read_bytes()


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code
        self.text = content.decode("utf-8", errors="replace")


class FakeSession:
    """Mimics ``requests.Session`` just enough for ``arxiv.Client``.

    Records every URL requested so tests can assert on pagination, query
    construction, and retry behaviour without touching the network.
    """

    def __init__(self, content: bytes | None = None, status_code: int = 200) -> None:
        self.content = content if content is not None else fixture_bytes()
        self.status_code = status_code
        self.calls: list[str] = []
        self.headers: list[dict | None] = []

    def get(self, url: str, headers: dict | None = None, **_: object) -> FakeResponse:
        self.calls.append(url)
        self.headers.append(headers)
        return FakeResponse(self.content, self.status_code)


def fake_client(content: bytes | None = None, status_code: int = 200):
    """An ``arxiv.Client`` with a fake session and no throttling delays."""
    import arxiv

    client = arxiv.Client(page_size=100, delay_seconds=0.0, num_retries=0)
    client._session = FakeSession(content, status_code)  # type: ignore[attr-defined]  # test-only session swap
    return client


def empty_feed_bytes() -> bytes:
    """A valid Atom feed with zero entries — the zero-result case."""
    return (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b'<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" '
        b'xmlns="http://www.w3.org/2005/Atom">'
        b"<id>https://arxiv.org/api/empty</id><title>arXiv Query</title>"
        b"<updated>2026-01-01T00:00:00Z</updated>"
        b"<opensearch:totalResults>0</opensearch:totalResults>"
        b"</feed>"
    )


def mk_result(
    entry_id: str = "http://arxiv.org/abs/2107.05580v1",
    *,
    updated: datetime | None = None,
    title: str = "A title",
    summary: str = "An abstract.",
    categories: list[str] | None = None,
    primary_category: str = "cs.CL",
    doi: str = "",
    journal_ref: str = "",
) -> Result:
    """Build an ``arxiv.Result`` without touching the network."""
    moment = updated or datetime(2021, 7, 12, tzinfo=timezone.utc)
    return Result(
        entry_id=entry_id,
        updated=moment,
        published=moment,
        title=title,
        authors=[],
        summary=summary,
        comment="",
        journal_ref=journal_ref,
        doi=doi,
        primary_category=primary_category,
        categories=categories or ["cs.CL"],
        links=[],
    )
