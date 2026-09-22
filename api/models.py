"""Pydantic models: the data contract shared by every pipeline stage (V2).

Three groups live here, implementing plan Appendix A.2 exactly:

1. **Pipeline models** — what stages consume and produce (``Paper``,
   ``RankedPaper``, ``PaperExtraction``, ``StageEvent``).
2. **LLM output schemas** — deliberately narrow: every field the LLM produces
   is a constrained enum, a grounded reference to a supplied paper id, or a
   verbatim quote from supplied text, so a hallucination is a validation
   failure rather than a plausible-looking addition to the map. All strict:
   ``additionalProperties: false`` and every field required (§2.8, B.9 rule 3).
3. **API models** — what the FastAPI layer returns to the frontend. These are
   the contract that Phase 8 generates ``web/lib/api-types.ts`` from.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# Enums / literals
# --------------------------------------------------------------------------- #

StageName = Literal["retrieval", "enrichment", "rerank", "extraction", "layout", "synthesis"]
StageStatus = Literal["running", "done", "error", "skipped"]
EdgeKind = Literal["extends", "contradicts", "applies", "shares_method"]
EdgeSource = Literal["knn", "citation", "llm"]
NarrativeStatus = Literal["pending", "ok", "partial", "fallback", "failed"]
ExtractionStatus = Literal["ok", "failed"]
Novelty = Literal["incremental", "substantial", "unclear"]
LandscapeStatus = Literal["running", "ready", "failed"]

#: Fixed execution order. The SSE stream and the UI timeline both assume this.
#:
#: ``layout`` runs *before* ``synthesis`` even though the user-facing story is
#: retrieval -> rerank -> extraction -> synthesis, because synthesis consumes
#: the computed clusters: it names them and writes the narrative around them
#: (V1's rationale; plan §3.3 keeps it).
STAGE_ORDER: tuple[StageName, ...] = (
    "retrieval",
    "enrichment",
    "rerank",
    "extraction",
    "layout",
    "synthesis",
)

#: HDBSCAN's noise label. Never named as a cluster; kept separate and muted.
UNCLUSTERED_LABEL: int = -1
UNCLUSTERED_NAME: str = "Unclustered"


def utcnow() -> str:
    """ISO-8601 UTC timestamp, second precision, always suffixed ``Z``.

    Imported from this one module everywhere so tests can monkeypatch the clock
    (plan §11.2's time seam).
    """
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Pipeline models
# --------------------------------------------------------------------------- #


class Paper(BaseModel):
    """A single arXiv result, normalized across library versions.

    ``paper_id`` is the **version-stripped** short id (``2107.05580``), because
    ``arxiv.Result.__eq__`` compares ``entry_id`` which *includes* the ``vN``
    suffix — so v1 and v3 of the same paper are distinct results and would both
    land in the map without stripping.
    """

    paper_id: str
    version: str = ""
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    published: str = ""
    updated: str = ""
    primary_category: str = ""
    categories: list[str] = Field(default_factory=list)
    comment: str = ""
    journal_ref: str = ""
    doi: str = ""
    abs_url: str = ""
    pdf_url: str = ""
    citation_count: int | None = None  # None = unknown, NOT zero
    citation_source: str = ""  # semanticscholar | openalex | ""
    openalex_id: str = ""
    s2_paper_id: str = ""
    fulltext_status: Literal["none", "ok", "unavailable"] = "none"

    @property
    def abs_link(self) -> str:
        """Canonical abstract page. arXiv asks that users be sent here."""
        return self.abs_url or f"https://arxiv.org/abs/{self.paper_id}"

    @property
    def rerank_text(self) -> str:
        """Text handed to the cross-encoder and the judge."""
        return f"{self.title}\n\n{self.abstract}"


class arXivQuery(BaseModel):
    """LLM-produced decomposition of a plain-English topic into arXiv syntax."""

    phrases: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)

    @field_validator("phrases", "keywords", "categories", mode="before")
    @classmethod
    def _coerce_none_to_empty(cls, value: object) -> object:
        return [] if value is None else value


class RankedPaper(BaseModel):
    """A paper with its rerank verdict attached.

    ``relevance_score`` is absolute (``sigmoid(logit) * 10``), comparable
    across runs — never min-max normalised within a batch (anti-pattern ledger,
    §2.10). ``rerank_source`` records which signals produced the score, so a
    degraded run is visible rather than silent.
    """

    paper: Paper
    rank: int
    relevance_score: float  # absolute 0-10
    cross_encoder_logit: float | None = None
    citation_prior: float | None = None
    judge_score: float | None = None
    rerank_source: str = "cross-encoder"  # e.g. "cross-encoder+citation+judge"

class PaperExtraction(BaseModel):
    """Structured reading of one abstract (or, in v3, of full-text sections).

    ``evidence`` must contain a verbatim span of the supplied text for each
    populated claim field — use ``evidence_violations`` (below) to check; the
    extraction pipeline refuses to store a field whose span is not a verbatim
    substring (§12.2 layer 3). A failed extraction is still a first-class row:
    ``status='failed'`` with ``error`` saying why, so nothing is silently lost
    (D3).
    """

    problem: str | None = None
    method: str | None = None
    results: str | None = None
    contribution: str | None = None
    limitations: str | None = None
    novelty: Novelty = "unclear"
    evidence: dict[str, str] = Field(default_factory=dict)  # field -> verbatim span
    status: ExtractionStatus = "ok"
    error: str = ""

    # ClassVar, not a field: these are the keys ``evidence`` is expected to
    # cover, not data carried by an instance.
    EXTRACTED_FIELDS: ClassVar[tuple[str, ...]] = (
        "problem",
        "method",
        "results",
        "contribution",
        "limitations",
    )

    @field_validator("evidence", mode="before")
    @classmethod
    def _drop_null_evidence(cls, value: object) -> object:
        """Models emit ``{"results": null}`` for fields they skipped.

        A strict ``dict[str, str]`` would turn the whole extraction into a
        failed one, and the repair loop cannot fix a model that keeps doing
        it. A null quote carries no information; dropping the entry is what
        the model meant.
        """
        if isinstance(value, dict):
            return {k: v for k, v in value.items() if v is not None}
        return value


def evidence_violations(extraction: PaperExtraction, source_text: str) -> list[str]:
    """Field names whose evidence span is NOT a verbatim substring of the source.

    The groundedness gate (§12.2 layer 3): every non-null claim field must be
    backed by a span that appears verbatim in the text the model was shown.
    Also flags spans recorded for null/absent claim fields and claim fields
    populated without any span. Returns ``[]`` when the extraction is fully
    grounded.
    """
    violations: list[str] = []
    for field_name in PaperExtraction.EXTRACTED_FIELDS:
        claim = getattr(extraction, field_name)
        span = extraction.evidence.get(field_name)
        if claim is not None:
            if not span:
                violations.append(field_name)
            elif span not in source_text:
                violations.append(field_name)
        elif span:
            violations.append(field_name)
    return violations


class StageProgress(BaseModel):
    """Intra-stage progress for the SSE stream (§3.3)."""

    done: int = 0
    total: int = 0
    unit: str = ""


class StageEvent(BaseModel):
    """One SSE frame. The frontend timeline consumes exactly this shape."""

    run_id: str
    landscape_id: int | None = None
    stage: StageName
    status: StageStatus
    message: str = ""
    progress: StageProgress = Field(default_factory=StageProgress)
    #: True when the stage completed with reduced confidence (stale cache,
    #: missing citations, judge failure, threshold lowered). Always surfaced.
    degraded: bool = False
    payload: dict = Field(default_factory=dict)
    ts: str = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# LLM output schemas — all strict (B.9 rule 3): additionalProperties false,
# every field required, no free-form dict values.
# --------------------------------------------------------------------------- #


class StrictModel(BaseModel):
    """Base for every LLM output schema: forbids undeclared fields."""

    model_config = ConfigDict(extra="forbid")


class QueryPlan(StrictModel):
    """B.1 — topic → arXiv query."""

    search_query: str
    rationale: str


class JudgeScore(StrictModel):
    """B.2 — one paper's relevance verdict."""

    paper_id: str
    relevance_0_10: float = Field(ge=0.0, le=10.0)


class JudgeBatch(StrictModel):
    """B.2 — the judge's envelope; exactly one entry per supplied id."""

    scores: list[JudgeScore]


class ClusterLabel(StrictModel):
    """B.4 — naming of one computed cluster. Never produced for label -1."""

    local_label: int
    label: str
    description: str


class TypedEdge(StrictModel):
    """B.5 — a relationship classification for a SUPPLIED pair of papers."""

    src_paper_id: str
    dst_paper_id: str
    kind: EdgeKind


class EdgeTyping(StrictModel):
    """B.5 — the edge-typing envelope."""

    edges: list[TypedEdge]


class Narrative(StrictModel):
    """B.6 — synthesis call A: the prose, generated from cluster inputs only."""

    title: str
    summary: str


class Tension(StrictModel):
    """A genuine disagreement between two papers, not merely a difference."""

    statement: str
    paper_a_id: str
    paper_b_id: str


class OpenProblem(StrictModel):
    """A gap that follows from what the supplied papers left undone."""

    statement: str
    why_open: str
    supporting_paper_ids: list[str]


class Claims(StrictModel):
    """B.7 — synthesis call B: tensions and open problems."""

    tensions: list[Tension]
    open_problems: list[OpenProblem]


class ReadingStep(StrictModel):
    """One entry in the suggested reading order."""

    paper_id: str
    position: int
    why: str


class ReadingPath(StrictModel):
    """B.8 — synthesis call C: 5–10 ordered steps."""

    steps: list[ReadingStep]


# --------------------------------------------------------------------------- #
# API models — the wire contract (Phase 8 generates web/lib/api-types.ts)
# --------------------------------------------------------------------------- #


class ClusterOut(BaseModel):
    id: int
    label: str
    description: str = ""
    paper_count: int = 0
    x: float = 0.0
    y: float = 0.0
    color: str = "#94a3b8"
    is_unclustered: bool = False


class PaperInLandscape(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    published: str = ""
    primary_category: str = ""
    categories: list[str] = Field(default_factory=list)
    abs_url: str
    pdf_url: str = ""
    citation_count: int | None = None  # None = unknown, NOT zero
    citation_source: str = ""
    rank: int
    relevance_score: float
    cross_encoder_logit: float | None = None
    #: Percentile rank within this landscape, 0-10. RELATIVE, unlike
    #: ``relevance_score``, and only meaningful for display and sizing.
    relative_score: float | None = None
    rerank_source: str
    cluster_id: int | None = None
    x: float = 0.0
    y: float = 0.0
    is_seed: bool = False
    #: ``None`` when the extraction failed — the paper still appears on the
    #: map (§3.3), and ``extraction_status`` says why it is absent.
    extraction: PaperExtraction | None = None
    extraction_status: ExtractionStatus = "ok"


class EdgeOut(BaseModel):
    src_paper_id: str
    dst_paper_id: str
    kind: EdgeKind
    weight: float
    rationale: str = ""
    source: EdgeSource = "llm"
    #: Deterministic confidence (cosine for knn, 1.0 for a citation).
    confidence: float | None = None


class TensionOut(BaseModel):
    statement: str
    paper_a_id: str
    paper_b_id: str


class OpenProblemOut(BaseModel):
    statement: str
    why_open: str = ""
    supporting_paper_ids: list[str] = Field(default_factory=list)


class ReadingStepOut(BaseModel):
    paper_id: str
    position: int
    why: str = ""
    title: str = ""


class LandscapeSummary(BaseModel):
    id: int
    topic: str
    title: str
    summary: str = ""
    status: LandscapeStatus
    narrative_status: NarrativeStatus = "pending"
    generation: int
    paper_count: int
    cluster_count: int
    cost_usd: float = 0.0
    created_at: str
    updated_at: str


class LandscapeDetail(BaseModel):
    id: int
    topic: str
    title: str
    summary: str
    status: LandscapeStatus
    narrative_status: NarrativeStatus = "pending"
    generation: int
    cost_usd: float = 0.0
    created_at: str
    updated_at: str
    clusters: list[ClusterOut] = Field(default_factory=list)
    papers: list[PaperInLandscape] = Field(default_factory=list)
    edges: list[EdgeOut] = Field(default_factory=list)
    tensions: list[TensionOut] = Field(default_factory=list)
    open_problems: list[OpenProblemOut] = Field(default_factory=list)
    reading_path: list[ReadingStepOut] = Field(default_factory=list)


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"
    llm_configured: bool = False
    llm_provider: str = ""
    llm_model: str = ""
    prompt_version: str = ""


class TopicRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=300)

    @field_validator("topic")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if len(cleaned) < 2:
            raise ValueError("topic must contain at least 2 non-space characters")
        return cleaned


class ExpandRequest(BaseModel):
    max_new_results: int = Field(default=100, ge=1, le=500)


class StreamDone(BaseModel):
    """Terminal SSE frame for a completed run."""

    run_id: str
    landscape_id: int | None = None
    status: LandscapeStatus
    narrative_status: NarrativeStatus = "pending"
    degraded: bool = False


class StreamError(BaseModel):
    """Terminal SSE frame for a failed run. ``retryable`` tells the UI whether
    offering "retry" makes sense (a cold-cache retrieval outage: yes; a
    validation bug: no)."""

    run_id: str
    landscape_id: int | None = None
    stage: str
    message: str
    retryable: bool = False

