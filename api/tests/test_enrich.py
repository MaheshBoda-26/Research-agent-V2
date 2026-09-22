"""Tests for the enrichment stage — Phase 3, Task 3.4.

The subject is degradation: both sources failing must return the papers
unchanged with ``citation_count IS NULL``, a partial mock must count only the
resolved papers, and intra-corpus citation links must only be written when
both endpoints are in the corpus.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import make_paper

from config import Settings
from pipeline.enrich import enrich_papers


@pytest.fixture
def settings(tmp_path) -> Settings:
    return replace(
        Settings.from_env(),
        db_path=tmp_path / "t.db",
        arxiv_offline=True,
        s2_api_key="",
        openalex_mailto="",
    )


class FailingS2:
    def fetch_paper(self, arxiv_id, settings, **kwargs):
        raise ConnectionError("blackholed")

    def fetch_references(self, arxiv_id, settings, **kwargs):
        raise ConnectionError("blackholed")


class FailingOpenAlex:
    def resolve_work(self, paper, settings, **kwargs):
        raise ConnectionError("blackholed")


class NoneS2:
    def fetch_paper(self, arxiv_id, settings, **kwargs):
        return None

    def fetch_references(self, arxiv_id, settings, **kwargs):
        return []


class NoneOpenAlex:
    def resolve_work(self, paper, settings, **kwargs):
        return None


class GoodS2:
    def __init__(self, counts: dict[str, int], refs: dict[str, list[str]] | None = None):
        self.counts = counts
        self.refs = refs or {}

    def fetch_paper(self, arxiv_id, settings, **kwargs):
        if arxiv_id not in self.counts:
            return None
        return {
            "paperId": f"s2-{arxiv_id}",
            "externalIds": {"ArXiv": arxiv_id},
            "citationCount": self.counts[arxiv_id],
        }

    def fetch_references(self, arxiv_id, settings, **kwargs):
        return [{"citedPaper": {"externalIds": {"ArXiv": cited}}}
                for cited in self.refs.get(arxiv_id, [])]


class GoodOpenAlex:
    def __init__(self, counts: dict[str, int]):
        self.counts = counts

    def resolve_work(self, paper, settings, **kwargs):
        if paper.paper_id not in self.counts:
            return None
        return {
            "id": f"https://openalex.org/W_{paper.paper_id}",
            "title": paper.title,
            "cited_by_count": self.counts[paper.paper_id],
            "publication_year": 2021,
        }


async def test_both_sources_failing_leaves_papers_unchanged(settings):
    papers = [make_paper(paper_id="a"), make_paper(paper_id="b")]
    result, triples = await enrich_papers(
        papers, settings, s2=FailingS2(), openalex=FailingOpenAlex()
    )
    assert [p.paper_id for p in result] == ["a", "b"]
    assert all(p.citation_count is None for p in result)
    assert all(p.citation_source == "" for p in result)
    assert triples == []


async def test_null_returning_sources_also_leave_papers_unchanged(settings):
    papers = [make_paper(paper_id="a")]
    result, triples = await enrich_papers(
        papers, settings, s2=NoneS2(), openalex=NoneOpenAlex()
    )
    assert result[0].citation_count is None
    assert triples == []


async def test_s2_count_wins_and_ids_are_recorded(settings):
    papers = [make_paper(paper_id="a")]
    result, _ = await enrich_papers(
        papers, settings, s2=GoodS2(counts={"a": 42}), openalex=FailingOpenAlex()
    )
    assert result[0].citation_count == 42
    assert result[0].citation_source == "semanticscholar"
    assert result[0].s2_paper_id == "s2-a"


async def test_openalex_is_the_fallback_when_s2_misses(settings):
    papers = [make_paper(paper_id="a", published="2021-01-01T00:00:00Z")]
    result, _ = await enrich_papers(
        papers, settings, s2=NoneS2(), openalex=GoodOpenAlex(counts={"a": 7})
    )
    assert result[0].citation_count == 7
    assert result[0].citation_source == "openalex"
    assert result[0].openalex_id == "https://openalex.org/W_a"


async def test_partial_mock_counts_only_resolved_papers(settings):
    papers = [make_paper(paper_id="a"), make_paper(paper_id="b")]
    result, _ = await enrich_papers(
        papers, settings, s2=GoodS2(counts={"a": 3}), openalex=FailingOpenAlex()
    )
    by_id = {p.paper_id: p for p in result}
    assert by_id["a"].citation_count == 3
    assert by_id["b"].citation_count is None


async def test_intra_corpus_links_require_both_ends(settings):
    papers = [make_paper(paper_id="a"), make_paper(paper_id="b"), make_paper(paper_id="outside")]
    s2 = GoodS2(
        counts={"a": 1, "b": 1, "outside": 1},
        refs={"a": ["b", "outside", "absent-from-corpus"]},
    )
    result, triples = await enrich_papers(papers, settings, s2=s2)
    assert ("a", "b", "semanticscholar") in triples
    assert ("a", "outside", "semanticscholar") in triples  # outside is in the corpus list
    assert all(dst != "absent-from-corpus" for _, dst, _ in triples)
    assert all(src != "absent-from-corpus" for src, _, _ in triples)


async def test_progress_reports_done_and_total(settings):
    papers = [make_paper(paper_id=f"p{i}") for i in range(3)]
    events: list[dict] = []
    await enrich_papers(
        papers, settings, s2=NoneS2(), openalex=NoneOpenAlex(), progress=events.append
    )
    assert [e["done"] for e in events] == [1, 2, 3]
    assert all(e["total"] == 3 for e in events)


async def test_empty_input_is_a_no_op(settings):
    assert await enrich_papers([], settings) == ([], [])
