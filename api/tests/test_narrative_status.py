"""Tests for Task 7.7 — the narrative_status ladder.

Pinned behaviours (plan §7.7):

* ``ok``       — title+summary present and >=1 structured section is real;
* ``partial``  — title+summary present, but some section used a fallback (or no
  structured section survived);
* ``fallback`` — the prose itself fell back (title/summary are templated);
* ``failed``   — title or summary missing (unreachable in practice: the
  deterministic fallback guarantees a non-empty pair).

The ladder is tested both as a pure function and end-to-end through
:func:`synthesize`.
"""

from __future__ import annotations

from conftest import make_paper

from config import Settings
from models import Claims, Narrative, Paper, ReadingPath, ReadingStep, Tension
from pipeline.synthesize import compute_narrative_status, synthesize

TOPIC = "test time compute"
GOOD_TITLE = "Trading Training for Inference Compute"
GOOD_SUMMARY = (
    "This landscape covers test time compute. "
    "Papers group into search and verification areas. "
    "Open questions remain about optimal allocation."
)


# --------------------------------------------------------------------------- #
# The pure ladder
# --------------------------------------------------------------------------- #

_T = [{"statement": "t", "paper_a_id": "a", "paper_b_id": "b"}]
_OP = [{"statement": "o", "why_open": "w", "supporting_paper_ids": ["a"]}]
_RP = [{"paper_id": "a", "position": 1, "why": "w"}]


def test_ok_when_everything_is_real():
    assert compute_narrative_status("t", "s", _T, _OP, _RP) == "ok"


def test_ok_needs_only_one_structured_section():
    assert compute_narrative_status("t", "s", _T, [], []) == "ok"
    assert compute_narrative_status("t", "s", [], _OP, []) == "ok"
    assert compute_narrative_status("t", "s", [], [], _RP) == "ok"


def test_partial_when_a_section_fell_back():
    assert compute_narrative_status("t", "s", _T, _OP, _RP, claims_fallback=True) == "partial"
    assert compute_narrative_status("t", "s", _T, _OP, _RP, path_fallback=True) == "partial"


def test_partial_when_no_structured_section_survived():
    assert compute_narrative_status("t", "s", [], [], []) == "partial"


def test_fallback_when_prose_fell_back():
    # Prose fallback wins even when structured sections are present.
    assert compute_narrative_status("t", "s", _T, _OP, _RP, prose_fallback=True) == "fallback"


def test_failed_when_prose_missing():
    assert compute_narrative_status("", "s", _T, _OP, _RP) == "failed"
    assert compute_narrative_status("t", "", _T, _OP, _RP) == "failed"


# --------------------------------------------------------------------------- #
# End-to-end through synthesize()
# --------------------------------------------------------------------------- #


class StageCompleter:
    def __init__(self, responses: dict[str, object]):
        self.responses = responses

    def complete_json(self, *, system, user, schema, stage="", **kwargs):
        resp = self.responses.get(stage)
        if isinstance(resp, Exception):
            raise resp
        return resp

    def complete_text(self, *, system, user):
        return None


def _papers(count: int = 5) -> list[Paper]:
    return [
        make_paper(paper_id=f"p{i}", title=f"Compute technique number {i}")
        for i in range(count)
    ]


def _clusters() -> list[dict]:
    return [{"local_label": 0, "label": "Search", "paper_ids": ["p0", "p1"]}]


def test_full_success_is_ok(settings: Settings):
    completer = StageCompleter(
        {
            "synthesis-narrative": Narrative(title=GOOD_TITLE, summary=GOOD_SUMMARY),
            "synthesis-claims": Claims(
                tensions=[Tension(statement="t", paper_a_id="p0", paper_b_id="p1")],
                open_problems=[],
            ),
            "synthesis-reading-path": ReadingPath(
                steps=[
                    ReadingStep(paper_id=f"p{i}", position=i + 1, why="w") for i in range(5)
                ]
            ),
        }
    )
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=completer)
    assert out["narrative_status"] == "ok"


def test_everything_failing_is_fallback_not_failed(settings: Settings):
    """NullCompleter-style run: prose falls back, structured sections too.

    The deterministic prose fallback guarantees a non-empty title/summary, so the
    status is ``fallback`` (templated prose) — never ``failed``.
    """
    failing = StageCompleter({})  # every call returns None
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=failing)
    assert out["narrative_status"] == "fallback"
    assert out["summary"].strip()
    assert out["title"].strip()


def test_only_narrative_surviving_is_partial(settings: Settings):
    completer = StageCompleter(
        {"synthesis-narrative": Narrative(title=GOOD_TITLE, summary=GOOD_SUMMARY)}
    )
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=completer)
    # Claims and reading path fell back (fallback path is still non-empty), so
    # the run is partial — the prose is real but sections used fallbacks.
    assert out["narrative_status"] == "partial"
    assert out["title"] == GOOD_TITLE

