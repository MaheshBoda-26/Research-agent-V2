"""Tests for Task 7.5 — claims call (tensions + open problems) and id validation.

Pinned behaviours (plan Task 7.5):

* a tension whose ``paper_a_id`` or ``paper_b_id`` is not a supplied paper is
  dropped (never rendered against a paper that is not on the map);
* an open problem's ``supporting_paper_ids`` are pruned to supplied ids, and if
  the list is emptied the whole open problem is dropped;
* a failed claims call degrades independently of the narrative and reading path
  (``narrative_status`` reflects it).

:func:`validate_ids` is exercised both as a pure function and through
:func:`synthesize`.
"""

from __future__ import annotations

from conftest import make_paper

from config import Settings
from models import Claims, Narrative, OpenProblem, Paper, ReadingPath, ReadingStep, Tension
from pipeline.synthesize import synthesize, validate_ids

TOPIC = "retrieval augmented generation"
GOOD_TITLE = "Grounding Generative Models in Retrieved Evidence"
GOOD_SUMMARY = (
    "This landscape covers retrieval augmented generation. "
    "Papers group into indexing and fusion areas. "
    "Open questions remain about faithfulness."
)


class StageCompleter:
    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.calls: list[str] = []

    def complete_json(self, *, system, user, schema, stage="", **kwargs):
        self.calls.append(stage)
        resp = self.responses.get(stage)
        if isinstance(resp, Exception):
            raise resp
        return resp

    def complete_text(self, *, system, user):
        return None


def _papers(count: int = 5) -> list[Paper]:
    return [
        make_paper(paper_id=f"p{i}", title=f"Retrieval technique number {i}")
        for i in range(count)
    ]


def _clusters() -> list[dict]:
    return [{"local_label": 0, "label": "Indexing", "paper_ids": ["p0", "p1", "p2"]}]


def _completer(claims) -> StageCompleter:
    return StageCompleter(
        {
            "synthesis-narrative": Narrative(title=GOOD_TITLE, summary=GOOD_SUMMARY),
            "synthesis-claims": claims,
            "synthesis-reading-path": ReadingPath(
                steps=[
                    ReadingStep(paper_id=f"p{i}", position=i + 1, why="foundational")
                    for i in range(5)
                ]
            ),
        }
    )


# --------------------------------------------------------------------------- #
# validate_ids — the pure gate
# --------------------------------------------------------------------------- #


def test_scalar_id_drops_invalid_item():
    items = [
        {"statement": "s1", "paper_a_id": "p0", "paper_b_id": "p1"},
        {"statement": "s2", "paper_a_id": "p0", "paper_b_id": "ghost"},
        {"statement": "s3", "paper_a_id": "ghost", "paper_b_id": "p1"},
    ]
    kept = validate_ids(items, ["paper_a_id", "paper_b_id"], {"p0", "p1"})
    assert [i["statement"] for i in kept] == ["s1"]


def test_list_id_is_pruned_to_valid():
    items = [{"statement": "op", "supporting_paper_ids": ["p0", "ghost", "p1"]}]
    kept = validate_ids(items, ["supporting_paper_ids"], {"p0", "p1"})
    assert len(kept) == 1
    assert kept[0]["supporting_paper_ids"] == ["p0", "p1"]


def test_emptied_list_drops_the_item():
    """An open problem left with zero supporting papers is removed (7.5)."""
    items = [{"statement": "op", "supporting_paper_ids": ["ghost1", "ghost2"]}]
    assert validate_ids(items, ["supporting_paper_ids"], {"p0", "p1"}) == []


def test_originally_empty_list_is_kept():
    """A list that started empty is not an *emptied* list — it is kept."""
    items = [{"statement": "op", "supporting_paper_ids": []}]
    kept = validate_ids(items, ["supporting_paper_ids"], {"p0", "p1"})
    assert kept == [{"statement": "op", "supporting_paper_ids": []}]


# --------------------------------------------------------------------------- #
# Through synthesize()
# --------------------------------------------------------------------------- #


def test_invalid_tension_ids_are_dropped(settings: Settings):
    claims = Claims(
        tensions=[
            Tension(statement="good", paper_a_id="p0", paper_b_id="p1"),
            Tension(statement="bad", paper_a_id="p0", paper_b_id="ghost"),
        ],
        open_problems=[],
    )
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=_completer(claims))
    assert [t["statement"] for t in out["tensions"]] == ["good"]


def test_open_problem_supporting_ids_are_pruned(settings: Settings):
    claims = Claims(
        tensions=[],
        open_problems=[
            OpenProblem(
                statement="kept",
                why_open="gap",
                supporting_paper_ids=["p0", "ghost", "p1"],
            ),
            OpenProblem(
                statement="dropped",
                why_open="gap",
                supporting_paper_ids=["ghost1", "ghost2"],
            ),
        ],
    )
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=_completer(claims))
    assert [op["statement"] for op in out["open_problems"]] == ["kept"]
    assert out["open_problems"][0]["supporting_paper_ids"] == ["p0", "p1"]


def test_failed_claims_call_degrades_independently(settings: Settings):
    """A claims failure leaves the narrative and reading path intact."""
    completer = _completer(RuntimeError("claims exploded"))
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=completer)
    assert out["tensions"] == []
    assert out["open_problems"] == []
    # Narrative (call A) and reading path (call C) still succeeded...
    assert out["title"] == GOOD_TITLE
    assert len(out["reading_path"]) == 5
    # ...but the claims fallback drops the status from ok to partial.
    assert out["narrative_status"] == "partial"


def test_all_invalid_claims_still_persist_empty_lists(settings: Settings):
    claims = Claims(
        tensions=[Tension(statement="bad", paper_a_id="x", paper_b_id="y")],
        open_problems=[OpenProblem(statement="bad", why_open="g", supporting_paper_ids=["x"])],
    )
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=_completer(claims))
    assert out["tensions"] == []
    assert out["open_problems"] == []



# --------------------------------------------------------------------------- #
# The claims prompt budget (content, not just the canned response)
# --------------------------------------------------------------------------- #


def test_claims_prompt_caps_at_forty_papers():
    """Regression: the prompt must return a str and cap the paper list at 40."""
    from prompts.synthesize import build_claims_prompt

    papers = [
        make_paper(paper_id=f"p{i}", title=f"Retrieval technique number {i}")
        for i in range(60)
    ]
    prompt = build_claims_prompt(TOPIC, papers, {})
    assert isinstance(prompt, str)
    assert "[p39]" in prompt
    assert "[p40]" not in prompt

