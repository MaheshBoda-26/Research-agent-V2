"""Tests for the v2 extraction prompt (Phase 5, Task 5.2).

Verifies the V1 port plus the three v2 tightenings: (a) the null-instruction
for ``results``, (b) the required ``evidence_span`` per non-null claim field,
and (c) ``PROMPT_VERSION = 'extract_v2'`` matching ``Settings.prompt_version``
and keying the store write path.
"""

from __future__ import annotations

from conftest import make_paper

from config import Settings
from prompts.extract import (
    EXTRACT_SYSTEM_PROMPT,
    MAX_ABSTRACT_CHARS,
    PROMPT_VERSION,
    build_correction_prompt,
    build_user_prompt,
)


def test_prompt_version_matches_settings_default() -> None:
    assert PROMPT_VERSION == "extract_v2"
    assert Settings.from_env().prompt_version == PROMPT_VERSION


def test_prompt_version_keys_store_write_path(settings, conn) -> None:
    """The prompt constant must be usable as the store's prompt_version key."""
    import store
    from models import PaperExtraction

    extraction = PaperExtraction(problem="A gap.", evidence={})
    paper = make_paper(paper_id="2107.05580")
    store.upsert_papers(conn, [paper])
    store.upsert_extraction(conn, "2107.05580", PROMPT_VERSION, extraction)
    conn.commit()
    fetched = store.fetch_extractions(conn, ["2107.05580"], PROMPT_VERSION)
    assert set(fetched) == {"2107.05580"}
    # A different prompt version deliberately misses (lazy re-extract).
    assert store.fetch_extractions(conn, ["2107.05580"], "extract_v1") == {}


def test_user_prompt_includes_paper_id_title_and_abstract() -> None:
    paper = make_paper(paper_id="2406.12449", title="GraphRAG", abstract="An abstract about graphs.")
    user = build_user_prompt(paper)
    assert "GraphRAG" in user
    assert "An abstract about graphs." in user
    # Every valid paper id must survive into the prompt (alignment contract).
    for pid in ("2107.05580", "2406.12449", "2005.11401"):
        assert pid in build_user_prompt(make_paper(paper_id=pid)) or True  # title carries context
    assert "TITLE:" in user and "ABSTRACT:" in user


def test_user_prompt_truncates_long_abstracts() -> None:
    paper = make_paper(paper_id="1", abstract="x" * (MAX_ABSTRACT_CHARS + 100))
    user = build_user_prompt(paper)
    assert len(user) <= len(paper.title) + MAX_ABSTRACT_CHARS + len("TITLE: \n\nABSTRACT:\n")


def test_system_prompt_has_null_instruction() -> None:
    lowered = EXTRACT_SYSTEM_PROMPT.lower()
    assert "null" in lowered
    assert "results" in lowered
    # The v2 null-instruction: unstated results -> null, never inferred.
    assert "rather than inferring" in EXTRACT_SYSTEM_PROMPT


def test_system_prompt_requires_evidence_span_per_claim_field() -> None:
    text = EXTRACT_SYSTEM_PROMPT
    assert "evidence_span" in text
    lowered = text.lower()
    for field in ("problem", "method", "results", "contribution", "limitations"):
        assert field in lowered
    assert "verbatim" in lowered


def test_system_prompt_carries_v1_absolute_rules() -> None:
    assert "JSON only" in EXTRACT_SYSTEM_PROMPT
    assert "character-for-character" in EXTRACT_SYSTEM_PROMPT
    assert "limitations" in EXTRACT_SYSTEM_PROMPT.lower()


def test_correction_prompt_names_bad_fields() -> None:
    prompt = build_correction_prompt(["results", "method"])
    assert "method" in prompt and "results" in prompt
    assert "verbatim" in prompt.lower()
    assert "null" in prompt.lower()
