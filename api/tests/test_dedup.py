"""Dedup and identifier-normalisation tests — Phase 2, Task 2.5.

Ported from V1's ``test_dedup.py`` (the plan says the cases are reusable
as-is), plus the old-style id form the plan asks to add. All offline: the
captured Atom response in ``tests/fixtures/`` is parsed with the same
``arxiv._feed`` parser that ``arxiv.Client`` uses internally.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from arxiv import _feed
from conftest import make_paper
from fake_arxiv import fixture_bytes, mk_result

from config import Settings
from pipeline.retrieve import (
    dedupe_by_paper_id,
    normalize_paper_id,
    result_to_paper,
    version_suffix,
)


@pytest.fixture
def fixture_results():
    """Real ``arxiv.Result`` objects parsed from the committed capture."""
    return list(_feed.parse(fixture_bytes()).results)


@pytest.fixture
def settings(tmp_path) -> Settings:
    from dataclasses import replace

    from config import Settings

    return replace(Settings.from_env(), db_path=tmp_path / "t.db", arxiv_offline=True)


# --------------------------------------------------------------------------- #
# Identifier handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("short_id", "expected"),
    [
        ("2411.18583v1", "2411.18583"),
        ("2411.18583v12", "2411.18583"),
        ("2411.18583", "2411.18583"),
        ("quant-ph/0201082v1", "quant-ph/0201082"),
        ("quant-ph/0201082v3", "quant-ph/0201082"),
        ("hep-th/9901001", "hep-th/9901001"),
        ("", ""),
    ],
)
def test_normalize_paper_id(short_id, expected):
    assert normalize_paper_id(short_id) == expected


@pytest.mark.parametrize(
    ("short_id", "expected"),
    [("2411.18583v1", "1"), ("2411.18583v10", "10"), ("2411.18583", ""), ("", "")],
)
def test_version_suffix(short_id, expected):
    assert version_suffix(short_id) == expected


def test_version_is_not_confused_by_a_v_inside_the_id():
    """`v` appears in old-style archive names; only a trailing marker counts."""
    assert normalize_paper_id("solv-int/9701001v2") == "solv-int/9701001"
    assert version_suffix("solv-int/9701001v2") == "2"


def test_old_style_id_form_round_trips():
    """Task 2.5's addition: ``quant-ph/0201082v3`` must survive conversion."""
    result = mk_result(entry_id="http://arxiv.org/abs/quant-ph/0201082v3")
    paper = result_to_paper(result)
    assert paper.paper_id == "quant-ph/0201082"
    assert paper.version == "3"
    assert paper.abs_url == "https://arxiv.org/abs/quant-ph/0201082"


# --------------------------------------------------------------------------- #
# Dedup
# --------------------------------------------------------------------------- #


def test_two_versions_of_one_paper_collapse_to_the_newest():
    result = dedupe_by_paper_id(
        [
            make_paper(paper_id="2107.05580", version="1", title="old"),
            make_paper(paper_id="2107.05580", version="3", title="new"),
        ]
    )
    assert len(result) == 1
    assert result[0].version == "3" and result[0].title == "new"


def test_dedup_keeps_the_higher_version_regardless_of_arrival_order():
    result = dedupe_by_paper_id(
        [
            make_paper(paper_id="a", version="10", title="v10"),
            make_paper(paper_id="a", version="9", title="v9"),
        ]
    )
    assert result[0].title == "v10"


def test_dedup_merges_the_longer_abstract():
    """Task 2.5: the duplicate keeps the higher version AND the longer abstract."""
    result = dedupe_by_paper_id(
        [
            make_paper(paper_id="a", version="1", abstract="long abstract " * 20),
            make_paper(paper_id="a", version="3", abstract="short"),
        ]
    )
    assert len(result) == 1
    assert result[0].version == "3"
    assert "long abstract" in result[0].abstract


def test_dedup_merges_the_longer_abstract_when_the_newer_arrives_second():
    result = dedupe_by_paper_id(
        [
            make_paper(paper_id="a", version="3", abstract="short"),
            make_paper(paper_id="a", version="1", abstract="long abstract " * 20),
        ]
    )
    assert len(result) == 1
    assert result[0].version == "3"  # version still wins for metadata
    assert "long abstract" in result[0].abstract


def test_dedup_preserves_relevance_order_of_first_appearance():
    result = dedupe_by_paper_id(
        [
            make_paper(paper_id="b", version="1"),
            make_paper(paper_id="a", version="1"),
            make_paper(paper_id="b", version="2"),
        ]
    )
    assert [p.paper_id for p in result] == ["b", "a"]


def test_dedup_skips_papers_without_an_id():
    assert dedupe_by_paper_id([make_paper(paper_id=""), make_paper(paper_id="a")]) == [
        make_paper(paper_id="a")
    ]


def test_dedup_of_an_unversioned_duplicate_keeps_the_first():
    result = dedupe_by_paper_id(
        [make_paper(paper_id="a", title="first"), make_paper(paper_id="a", title="second")]
    )
    assert len(result) == 1 and result[0].title == "first"

# --------------------------------------------------------------------------- #
# Result conversion, against the real captured response
# --------------------------------------------------------------------------- #


def test_fixture_parses_into_expected_number_of_results(fixture_results):
    """Five entries, deliberately fewer than the requested max.

    The fake session always serves the same page, so a fixture smaller than the
    requested limit is what forces ``Client`` to paginate and makes the
    cross-page version collapsing testable.
    """
    assert len(fixture_results) == 5


def test_every_fixture_result_converts_to_a_usable_paper(fixture_results):
    papers = [result_to_paper(r) for r in fixture_results]
    for paper in papers:
        assert paper.paper_id
        assert paper.title and paper.abstract
        assert paper.abs_url == f"https://arxiv.org/abs/{paper.paper_id}"
        assert paper.primary_category
        assert paper.authors


def test_converted_ids_never_carry_a_version_suffix(fixture_results):
    versioned = 0
    for result in fixture_results:
        paper = result_to_paper(result)
        if version_suffix(result.get_short_id()):
            versioned += 1
        # ``v(\d+)$`` must not fire on an id whose last character merely
        # happens to be a digit, the common case for new-style ids.
        assert version_suffix(paper.paper_id) == ""
    # Guard the guard: if no fixture entry were versioned, this test would be
    # vacuously green and would stop protecting the version-stripping rule.
    assert versioned > 0


def test_converted_papers_dedupe_cleanly(fixture_results):
    papers = [result_to_paper(r) for r in fixture_results]
    deduped = dedupe_by_paper_id(papers)
    assert len(deduped) == len({p.paper_id for p in papers})


def test_published_is_an_iso_string_not_a_datetime(fixture_results):
    paper = result_to_paper(fixture_results[0])
    assert isinstance(paper.published, str)
    assert paper.published.startswith("20")
    datetime.fromisoformat(paper.published.replace("Z", "+00:00"))


def test_abs_link_falls_back_when_abs_url_is_empty():
    """Contract test from Task 2.1: the canonical link is derived, not stored."""
    paper = make_paper(abs_url="")
    assert paper.abs_link == f"https://arxiv.org/abs/{paper.paper_id}"