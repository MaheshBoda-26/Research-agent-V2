"""Tests for api/models.py — the data contract (plan Appendix A.2).

The load-bearing test here is groundedness: evidence spans must be verbatim
substrings of the text the model was shown (§12.2 layer 3; §11.4 invariants).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import (
    STAGE_ORDER,
    Claims,
    ClusterLabel,
    EdgeTyping,
    JudgeBatch,
    Narrative,
    Paper,
    PaperExtraction,
    QueryPlan,
    ReadingPath,
    StageEvent,
    TopicRequest,
    evidence_violations,
)

ABSTRACT = (
    "We study retrieval-augmented generation for domain QA. "
    "Our method couples a dense retriever with a frozen reader. "
    "On three benchmarks we improve exact match by 4.2 points over DPR baselines. "
    "The approach does not address multi-hop questions."
)


def _grounded_extraction() -> PaperExtraction:
    return PaperExtraction(
        problem="domain QA needs retrieval",
        method="dense retriever + frozen reader",
        results="+4.2 EM over DPR",
        contribution="a coupled architecture",
        limitations="no multi-hop",
        novelty="substantial",
        evidence={
            "problem": "retrieval-augmented generation for domain QA",
            "method": "couples a dense retriever with a frozen reader",
            "results": "improve exact match by 4.2 points over DPR baselines",
            "contribution": "Our method couples a dense retriever",
            "limitations": "does not address multi-hop questions",
        },
    )


def test_evidence_spans_must_be_verbatim_substrings() -> None:
    extraction = _grounded_extraction()
    assert evidence_violations(extraction, ABSTRACT) == []


def test_fabricated_span_is_flagged() -> None:
    extraction = _grounded_extraction()
    extraction.evidence["results"] = "improve exact match by 42 points"  # not in the abstract
    assert evidence_violations(extraction, ABSTRACT) == ["results"]


def test_claim_without_span_is_flagged() -> None:
    extraction = _grounded_extraction()
    del extraction.evidence["problem"]  # claim present, span absent
    violations = evidence_violations(extraction, ABSTRACT)
    assert violations == ["problem"]


def test_span_for_absent_claim_is_flagged() -> None:
    extraction = _grounded_extraction()
    extraction.results = None  # claim dropped, but its span lingers
    assert evidence_violations(extraction, ABSTRACT) == ["results"]


def test_failed_extraction_carries_an_error() -> None:
    failed = PaperExtraction(status="failed", error="timeout after 60s")
    assert failed.status == "failed"
    assert failed.error
    assert evidence_violations(failed, ABSTRACT) == []


def test_null_evidence_entries_are_dropped_not_fatal() -> None:
    extraction = PaperExtraction.model_validate(
        {"problem": "p", "evidence": {"problem": None, "results": None}}
    )
    assert extraction.evidence == {}


# --- Strict LLM schemas (B.9 rule 3) ---------------------------------------- #


@pytest.mark.parametrize(
    "schema",
    [QueryPlan, JudgeBatch, ClusterLabel, EdgeTyping, Narrative, Claims, ReadingPath],
)
def test_llm_schemas_are_strict(schema) -> None:
    """The two conditions provider-side `strict: true` needs (§11.6)."""
    json_schema = schema.model_json_schema()
    assert json_schema.get("additionalProperties") is False
    required = set(json_schema.get("required", []))
    assert required == set(json_schema.get("properties", {}))


def test_llm_schemas_reject_undeclared_fields() -> None:
    with pytest.raises(ValidationError):
        Narrative.model_validate({"title": "t", "summary": "s", "surprise": "hallucinated"})


def test_llm_schemas_require_every_field() -> None:
    with pytest.raises(ValidationError):
        QueryPlan.model_validate({"search_query": "all:rag"})  # missing rationale


def test_judge_score_bounds() -> None:
    batch = JudgeBatch.model_validate({"scores": [{"paper_id": "2107.05580", "relevance_0_10": 9.5}]})
    assert batch.scores[0].paper_id == "2107.05580"
    with pytest.raises(ValidationError):
        JudgeBatch.model_validate({"scores": [{"paper_id": "x", "relevance_0_10": 11.0}]})


# --- Pipeline models ---------------------------------------------------------- #


def test_stage_order_is_fixed() -> None:
    assert STAGE_ORDER == ("retrieval", "enrichment", "rerank", "extraction", "layout", "synthesis")


def test_stage_event_defaults() -> None:
    event = StageEvent(run_id="r1", stage="retrieval", status="running")
    assert event.degraded is False
    assert event.progress.done == 0 and event.progress.total == 0
    assert event.ts.endswith("Z") and "+" not in event.ts
    assert event.payload == {}


def test_paper_defaults_and_links() -> None:
    paper = Paper(paper_id="2107.05580", title="t", abstract="a")
    assert paper.abs_link == "https://arxiv.org/abs/2107.05580"
    assert paper.rerank_text == "t\n\na"
    assert paper.citation_count is None  # unknown is not zero
    assert paper.fulltext_status == "none"


# --- API models --------------------------------------------------------------- #


def test_topic_request_strips_and_bounds() -> None:
    assert TopicRequest(topic="  retrieval   augmented\n generation ").topic == (
        "retrieval augmented generation"
    )
    with pytest.raises(ValidationError):
        TopicRequest(topic="x")
    with pytest.raises(ValidationError):
        TopicRequest(topic="ab" * 200)
    with pytest.raises(ValidationError):
        TopicRequest(topic="   ")  # whitespace-only collapses below min length
