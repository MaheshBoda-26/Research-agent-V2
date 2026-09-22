"""Tests for the OpenAlex source — Phase 3, Task 3.3.

Identity matching is the subject: a 0.95-similar title resolves, a 0.6-similar
decoy must not, and a DOI-shaped lookup must never be constructed (the §2.3
verified trap). All through ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from conftest import make_paper

from sources import openalex

FIXTURES = Path(__file__).parent / "fixtures"


def _transport(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=openalex._BASE)


def _fixture_payload() -> dict:
    return json.loads((FIXTURES / "openalex_search.json").read_text())


def test_resolve_work_accepts_a_similar_title_and_close_year(settings, tmp_path):
    from dataclasses import replace as _replace

    settings = _replace(settings, openalex_mailto="test@example.com")
    paper = make_paper(title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                       published="2020-08-01T00:00:00Z")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_fixture_payload())

    work = openalex.resolve_work(paper, settings, client=_transport(handler))
    assert work is not None
    assert work["id"] == "https://openalex.org/W2963403868"
    assert work["cited_by_count"] == 19463
    # Search-by-title, polite pool; never a DOI lookup.
    url = str(requests[0].url)
    assert "search=" in url
    assert "doi:10.48550" not in url
    assert "mailto=test%40example.com" in url or "mailto=test@example.com" in url


def test_resolve_work_rejects_a_dissimilar_decoy(settings):
    paper = make_paper(title="Bayesian Optimization of Neural Architectures",
                       published="2023-03-01T00:00:00Z")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fixture_payload())  # RAG paper, wrong title

    assert openalex.resolve_work(paper, settings, client=_transport(handler)) is None


def test_resolve_work_rejects_a_far_off_year(settings):
    paper = make_paper(title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                       published="2012-08-01T00:00:00Z")  # fixture year is 2020

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fixture_payload())

    assert openalex.resolve_work(paper, settings, client=_transport(handler)) is None


def test_title_similarity_floor():
    fixture_title = _fixture_payload()["results"][0]["title"]
    assert openalex.title_similarity(fixture_title, fixture_title) >= 0.9
    # Case/punctuation-insensitive normalisation still matches.
    assert openalex.title_similarity(fixture_title.upper(), fixture_title) >= 0.9
    assert openalex.title_similarity("Bayesian Optimization of Neural Architectures", fixture_title) < 0.6


def test_resolve_work_returns_none_on_transport_failure(settings):
    paper = make_paper(title="Anything")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("blackholed")

    assert openalex.resolve_work(paper, settings, client=_transport(handler)) is None


def test_fetch_citations_by_work_id(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["filter"] == "cites:W2963403868"
        return httpx.Response(200, json={"results": [{"id": "https://openalex.org/W1"},
                                                     {"id": "https://openalex.org/W2"}]})

    ids = openalex.fetch_citations_by_work_id(
        "https://openalex.org/W2963403868", settings, client=_transport(handler)
    )
    assert ids == ["https://openalex.org/W1", "https://openalex.org/W2"]


def test_fetch_citations_by_work_id_degrades_to_empty(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate"})

    assert openalex.fetch_citations_by_work_id("W1", settings, client=_transport(handler)) == []


def test_resolve_work_is_cached(tmp_path, settings, conn):
    paper = make_paper(title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                       published="2020-08-01T00:00:00Z")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_fixture_payload())

    client = _transport(handler)
    assert openalex.resolve_work(paper, settings, client=client, conn=conn) is not None
    assert openalex.resolve_work(paper, settings, client=client, conn=conn) is not None
    assert len(requests) == 1  # second call came from source_cache
