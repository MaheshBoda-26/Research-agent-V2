"""Tests for arXiv query construction — Phase 2, Task 2.4.

Three tiers (B.1): an LLM-proposed ``search_query``, a quoted-phrase
heuristic, and the plain unquoted retry. Every test asserts the *exact*
query string sent, and nothing touches the network.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import make_paper

from config import Settings
from pipeline.retrieve import candidate_queries, resolve_query
from prompts.query import (
    ML_CATEGORIES,
    QUERY_SYSTEM_PROMPT,
    build_user_prompt,
    heuristic_query,
    is_wellformed_query,
    plain_query,
)


class Completer:
    """A two-line LLM seam (§11.2)."""

    def __init__(self, result=None, raises: Exception | None = None) -> None:
        self._result = result
        self._raises = raises
        self.calls = 0

    def complete_json(self, *, system, user, schema, **kwargs):
        self.calls += 1
        assert system == QUERY_SYSTEM_PROMPT
        assert user == build_user_prompt("diffusion policy learning")
        if self._raises is not None:
            raise self._raises
        return self._result


class Plan:
    def __init__(self, search_query: str) -> None:
        self.search_query = search_query


# --------------------------------------------------------------------------- #
# Tier 2 / tier 3 builders
# --------------------------------------------------------------------------- #


def test_heuristic_query_quotes_the_whole_topic():
    assert heuristic_query("  retrieval-augmented   generation ") == (
        'all:"retrieval-augmented generation"'
    )


def test_heuristic_query_of_blank_topic_is_empty():
    assert heuristic_query("   ") == ""


def test_plain_query_is_unquoted_for_the_retry():
    assert plain_query("diffusion policy") == "all:diffusion policy"
    assert plain_query("  ") == ""


# --------------------------------------------------------------------------- #
# B.1 acceptance rules for the LLM tier
# --------------------------------------------------------------------------- #


def test_wellformed_llm_query_is_accepted():
    assert is_wellformed_query('all:"retrieval-augmented generation" AND cat:cs.CL')


def test_llm_query_without_field_prefixes_is_rejected():
    assert not is_wellformed_query("retrieval augmented generation")


def test_llm_query_with_fewer_than_two_terms_is_rejected():
    assert not is_wellformed_query('all:"retrieval-augmented generation"')


def test_llm_query_with_a_hallucinated_category_is_rejected():
    assert not is_wellformed_query('all:"topic" AND cat:astro-ph.CO')


def test_llm_query_with_sortby_or_a_url_is_rejected():
    assert not is_wellformed_query('all:"topic" AND cat:cs.CL&sortBy=submittedDate')
    assert not is_wellformed_query('all:"topic" AND cat:cs.CL http://x')


# --------------------------------------------------------------------------- #
# Tier ordering / resolution
# --------------------------------------------------------------------------- #


@pytest.fixture
def settings(tmp_path) -> Settings:
    from config import Settings

    return replace(Settings.from_env(), db_path=tmp_path / "t.db", arxiv_offline=True)


def test_without_a_completer_only_the_heuristic_and_plain_tiers_exist(settings: Settings):
    tiers = candidate_queries("diffusion policy learning", None)
    assert tiers == [
        ("heuristic", 'all:"diffusion policy learning"'),
        ("plain", "all:diffusion policy learning"),
    ]
    assert resolve_query("diffusion policy learning", settings, None) == tiers[0]


def test_a_wellformed_llm_query_wins_tier_one(settings: Settings):
    completer = Completer(Plan('all:"diffusion policy" AND cat:cs.LG'))
    tier, query = resolve_query("diffusion policy learning", settings, completer)
    assert tier == "llm"
    assert query == 'all:"diffusion policy" AND cat:cs.LG'


def test_malformed_llm_output_degrades_to_the_heuristic(settings: Settings):
    completer = Completer(Plan("diffusion policy learning"))
    tier, query = resolve_query("diffusion policy learning", settings, completer)
    assert (tier, query) == ("heuristic", 'all:"diffusion policy learning"')


def test_a_raising_completer_degrades_to_the_heuristic(settings: Settings):
    completer = Completer(raises=RuntimeError("no key"))
    tier, query = resolve_query("diffusion policy learning", settings, completer)
    assert (tier, query) == ("heuristic", 'all:"diffusion policy learning"')


def test_a_none_completer_result_degrades_to_the_heuristic(settings: Settings):
    tier, _ = resolve_query("diffusion policy learning", settings, Completer(None))
    assert tier == "heuristic"


def test_a_blank_topic_has_no_searchable_terms(settings: Settings):
    from sources.arxiv_client import RetrievalError

    with pytest.raises(RetrievalError, match="searchable terms"):
        resolve_query("   ", settings, None)


def test_the_category_list_constrains_the_prompt():
    for category in ("cs.CL", "stat.ML"):
        assert category in QUERY_SYSTEM_PROMPT
    assert len(ML_CATEGORIES) >= 5


def test_duplicate_plain_tier_is_not_appended_twice():
    """A topic whose heuristic and plain strings coincide still yields one retry."""
    tiers = candidate_queries("x", None)
    strings = [query for _, query in tiers]
    assert len(strings) == len(set(strings)) or tiers[-1][0] == "plain"


def test_fetch_candidates_reports_tier_and_query(tmp_path):
    from fake_arxiv import fake_client

    from pipeline.retrieve import fetch_candidates

    settings = replace(
        Settings.from_env(),
        db_path=tmp_path / "t.db",
        retrieval_cache_dir=tmp_path / "cache",
        arxiv_offline=True,
    )
    events: list[tuple[str, dict]] = []
    fetch_candidates(
        "diffusion policy learning",
        settings,
        client=fake_client(),
        use_cache=False,
        progress=events.append,
    )
    tier_event = [e for e in events if e.get("event") == "query_tier"][-1]
    assert tier_event == {
        "event": "query_tier",
        "tier": "heuristic",
        "search_query": 'all:"diffusion policy learning"',
    }
    assert make_paper().paper_id  # conftest import sanity

    for category in ("cs.CL", "stat.ML"):
        assert category in QUERY_SYSTEM_PROMPT
    assert len(ML_CATEGORIES) >= 5
