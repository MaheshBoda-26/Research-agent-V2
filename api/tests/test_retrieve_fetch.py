"""Retrieval tests that run entirely offline (Phase 2, Task 2.2).

These drive the real ``arxiv.Client`` — URL formatting, pagination, retry
logic, and Atom parsing — with only the HTTP session swapped for a fake. The
response body is a real arXiv response captured once and committed as a
fixture.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from arxiv import _feed
from fake_arxiv import fake_client, fixture_bytes

from config import Settings
from models import Paper
from pipeline.retrieve import (
    RetrievalError,
    RetrievalThrottled,
    fetch_candidates,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _settings(tmp_path, **overrides) -> Settings:
    base = replace(
        Settings.from_env(),
        db_path=tmp_path / "test.db",
        retrieval_cache_dir=tmp_path / "cache",
        retrieval_max_results=5,
        arxiv_offline=False,
    )
    return replace(base, **overrides)  # type: ignore[type-var]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return _settings(tmp_path)


@pytest.fixture
def fixture_results():
    """Real ``arxiv.Result`` objects parsed from the committed capture."""
    return list(_feed.parse(fixture_bytes()).results)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_fetch_candidates_parses_the_fixture(tmp_path):
    settings = _settings(tmp_path)
    papers = fetch_candidates(
        "retrieval-augmented generation", settings, client=fake_client(), use_cache=False
    )
    assert len(papers) == 5
    assert all(isinstance(p, Paper) for p in papers)
    # Version-stripped ids, canonical abs links.
    for paper in papers:
        assert paper.abs_url == f"https://arxiv.org/abs/{paper.paper_id}"


def test_search_query_is_the_heuristic_tier_without_a_completer(tmp_path):
    settings = _settings(tmp_path)
    session_client = fake_client()
    fetch_candidates(
        "retrieval-augmented generation", settings, client=session_client, use_cache=False
    )
    url = session_client._session.calls[0]  # type: ignore[attr-defined]
    assert "sortBy=relevance" in url
    assert "retrieval-augmented" in url  # the quoted topic phrase, URL-encoded


def test_tier_recording_reports_the_winning_tier(tmp_path):
    settings = _settings(tmp_path)
    events: list[tuple[str, dict]] = []
    fetch_candidates(
        "retrieval-augmented generation",
        settings,
        client=fake_client(),
        use_cache=False,
        progress=events.append,
    )
    tiers = [e for e in events if e.get("event") == "query_tier"]
    assert tiers and tiers[-1]["tier"] == "heuristic"
    assert 'all:"retrieval-augmented generation"' in tiers[-1]["search_query"]


def test_zero_result_llm_query_falls_back_to_the_plain_tier(tmp_path):
    """Task 2.4's addition: an LLM query matching nothing must not end the run."""
    from fake_arxiv import empty_feed_bytes

    from prompts.query import QUERY_SYSTEM_PROMPT

    settings = _settings(tmp_path)
    events: list[tuple[str, dict]] = []

    class Completer:
        def complete_json(self, *, system, user, schema, **kwargs):
            assert system == QUERY_SYSTEM_PROMPT
            assert "graph neural networks" in user

            class Plan:
                search_query = 'all:"graph neural networks" AND cat:cs.LG'
                rationale = "canonical GNN phrasing"

            return Plan

    # Empty feed: every tier matches nothing for its first request, so all three
    # tiers are tried and zero papers come back, with no tier reported as won.
    papers = fetch_candidates(
        "graph neural networks",
        settings,
        client=fake_client(content=empty_feed_bytes()),
        use_cache=False,
        completer=Completer(),
        progress=events.append,
    )
    assert papers == []
    assert [e for e in events if e.get("event") == "query_tier"] == []

    # Now the real fixture: the first (llm) tier matches, and the tier is "llm".
    events.clear()
    papers = fetch_candidates(
        "graph neural networks",
        settings,
        client=fake_client(),
        use_cache=False,
        completer=Completer(),
        progress=events.append,
    )
    assert len(papers) == 5
    tiers = [e for e in events if e.get("event") == "query_tier"]
    assert tiers[-1]["tier"] == "llm"


def test_malformed_llm_output_falls_back_to_the_heuristic(tmp_path):
    settings = _settings(tmp_path)
    events: list[tuple[str, dict]] = []

    class Completer:
        def complete_json(self, *, system, user, schema, **kwargs):
            class Plan:
                search_query = "retrieval augmented generation"  # no field prefixes
                rationale = ""

            return Plan

    fetch_candidates(
        "retrieval-augmented generation",
        settings,
        client=fake_client(),
        use_cache=False,
        completer=Completer(),
        progress=events.append,
    )
    tiers = [e for e in events if e.get("event") == "query_tier"]
    assert tiers[-1]["tier"] == "heuristic"



def test_pagination_requests_multiple_pages(tmp_path):
    """The fixture holds 5 entries; requesting more forces a second page."""
    settings = _settings(tmp_path, retrieval_max_results=10)
    session_client = fake_client()
    papers = fetch_candidates(
        "retrieval-augmented generation", settings, client=session_client, use_cache=False
    )
    calls = session_client._session.calls  # type: ignore[attr-defined]
    assert len(papers) == 5
    assert len(calls) >= 2
    assert "start=0" in calls[0]
    assert "start=5" in calls[1]


def test_throttled_status_code_raises_typed_error(tmp_path):
    settings = _settings(tmp_path)
    with pytest.raises(RetrievalThrottled):
        fetch_candidates(
            "a topic", settings, client=fake_client(status_code=429), use_cache=False
        )


def test_generic_http_error_maps_to_retrieval_error(tmp_path):
    settings = _settings(tmp_path)
    with pytest.raises(RetrievalError):
        fetch_candidates("a topic", settings, client=fake_client(status_code=500), use_cache=False)


def test_blank_topic_is_an_input_error(tmp_path):
    settings = _settings(tmp_path)
    with pytest.raises(RetrievalError, match="topic is required"):
        fetch_candidates("   ", settings, client=fake_client(), use_cache=False)


def test_one_client_is_threaded_through_every_attempt(tmp_path):
    """The unquoted retry must reuse the same client (ToU delay memory)."""
    from fake_arxiv import empty_feed_bytes

    from sources.arxiv_client import make_client

    settings = _settings(tmp_path)
    session_client = fake_client(content=empty_feed_bytes())
    built = make_client(settings)
    assert built is not None  # sanity: factory honours settings
    fetch_candidates("a topic", settings, client=session_client, use_cache=False)
    # One session object served both tier attempts: no client was rebuilt.
    assert len(session_client._session.calls) >= 2  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Live smoke test — deselected by default (opt in with -m live)
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_fetch_returns_a_real_corpus(tmp_path):
    live = _settings(tmp_path, arxiv_offline=False, retrieval_max_results=40)
    papers = fetch_candidates("retrieval-augmented generation", live)
    assert len(papers) >= 30
    assert all(isinstance(p, Paper) and p.abstract for p in papers)
    assert len({p.paper_id for p in papers}) == len(papers)


def test_raising_completer_degrades_to_the_heuristic(tmp_path):
    settings = _settings(tmp_path)
    events: list[tuple[str, dict]] = []

    class Completer:
        def complete_json(self, *, system, user, schema, **kwargs):
            raise RuntimeError("model had a bad day")

    fetch_candidates(
        "retrieval-augmented generation",
        settings,
        client=fake_client(),
        use_cache=False,
        completer=Completer(),
        progress=events.append,
    )
    tiers = [e for e in events if e.get("event") == "query_tier"]
    assert tiers[-1]["tier"] == "heuristic"
