"""Tests for Task 7.6 — the reading path (call C) and its deterministic fallback.

Pinned behaviours (plan Task 7.6 / B.8):

* positions are re-numbered densely from 1 after ordering (1, 2, 4, 7 -> 1..4);
* the model path must have 5-10 valid steps — otherwise the deterministic
  fallback (survey -> most-cited -> best-ranked) is used;
* steps referencing a paper not on the map are dropped, then renumbered;
* the fallback orders survey/review papers first, then by citation count.
"""

from __future__ import annotations

from conftest import make_paper

from config import Settings
from models import Claims, Narrative, Paper, ReadingPath, ReadingStep
from pipeline.synthesize import _fallback_reading_path, dense_positions, synthesize

TOPIC = "graph neural networks"
GOOD_TITLE = "Message Passing Beyond the Neighbourhood"
GOOD_SUMMARY = (
    "This landscape covers graph neural networks. "
    "Papers group into spectral and spatial areas. "
    "Open questions remain about oversmoothing."
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


def _papers(count: int = 6) -> list[Paper]:
    return [
        make_paper(paper_id=f"p{i}", title=f"GNN technique number {i}")
        for i in range(count)
    ]


def _clusters() -> list[dict]:
    return [{"local_label": 0, "label": "Spectral", "paper_ids": ["p0", "p1", "p2"]}]


def _completer(reading_path) -> StageCompleter:
    return StageCompleter(
        {
            "synthesis-narrative": Narrative(title=GOOD_TITLE, summary=GOOD_SUMMARY),
            "synthesis-claims": Claims(tensions=[], open_problems=[]),
            "synthesis-reading-path": reading_path,
        }
    )


# --------------------------------------------------------------------------- #
# dense_positions — the pure gate
# --------------------------------------------------------------------------- #


def test_dense_positions_renumbers_after_sorting():
    steps = [
        {"paper_id": "a", "position": 1, "why": ""},
        {"paper_id": "b", "position": 2, "why": ""},
        {"paper_id": "c", "position": 4, "why": ""},
        {"paper_id": "d", "position": 7, "why": ""},
    ]
    out = dense_positions(steps)
    assert [s["paper_id"] for s in out] == ["a", "b", "c", "d"]
    assert [s["position"] for s in out] == [1, 2, 3, 4]


def test_dense_positions_orders_by_reported_position():
    # Out-of-order input is sorted by position first, then renumbered.
    steps = [
        {"paper_id": "c", "position": 9, "why": ""},
        {"paper_id": "a", "position": 3, "why": ""},
        {"paper_id": "b", "position": 5, "why": ""},
    ]
    out = dense_positions(steps)
    assert [s["paper_id"] for s in out] == ["a", "b", "c"]
    assert [s["position"] for s in out] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Through synthesize() — 5-10 step enforcement
# --------------------------------------------------------------------------- #


def test_valid_path_is_used_with_dense_positions(settings: Settings):
    path = ReadingPath(
        steps=[
            ReadingStep(paper_id="p0", position=1, why="a"),
            ReadingStep(paper_id="p1", position=2, why="b"),
            ReadingStep(paper_id="p2", position=4, why="c"),
            ReadingStep(paper_id="p3", position=7, why="d"),
            ReadingStep(paper_id="p4", position=9, why="e"),
        ]
    )
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=_completer(path))
    assert [s["position"] for s in out["reading_path"]] == [1, 2, 3, 4, 5]
    assert [s["paper_id"] for s in out["reading_path"]] == ["p0", "p1", "p2", "p3", "p4"]


def test_too_few_steps_falls_back(settings: Settings):
    path = ReadingPath(
        steps=[ReadingStep(paper_id=f"p{i}", position=i + 1, why="x") for i in range(3)]
    )
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=_completer(path))
    # 3 < 5 -> the deterministic fallback is used instead (6 papers available).
    assert len(out["reading_path"]) == 6
    assert [s["position"] for s in out["reading_path"]] == [1, 2, 3, 4, 5, 6]


def test_invalid_step_ids_are_dropped_then_renumbered(settings: Settings):
    path = ReadingPath(
        steps=[
            ReadingStep(paper_id="p0", position=1, why="a"),
            ReadingStep(paper_id="ghost", position=2, why="b"),
            ReadingStep(paper_id="p1", position=3, why="c"),
            ReadingStep(paper_id="p2", position=4, why="d"),
            ReadingStep(paper_id="p3", position=5, why="e"),
        ]
    )
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=_completer(path))
    # "ghost" dropped, 4 valid steps remain — below the 5 minimum, so fallback.
    ids = [s["paper_id"] for s in out["reading_path"]]
    assert "ghost" not in ids
    assert [s["position"] for s in out["reading_path"]] == list(range(1, len(ids) + 1))


def test_failed_reading_path_uses_fallback(settings: Settings):
    completer = _completer(RuntimeError("reading path exploded"))
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=completer)
    assert len(out["reading_path"]) == 6  # fallback over all 6 papers
    assert out["narrative_status"] == "partial"


# --------------------------------------------------------------------------- #
# _fallback_reading_path — deterministic ordering
# --------------------------------------------------------------------------- #


def test_fallback_orders_survey_then_cited_then_rest():
    papers = [
        make_paper(paper_id="rest", title="A niche result", citation_count=None),
        make_paper(paper_id="cited_low", title="Some method", citation_count=5),
        make_paper(paper_id="survey", title="A Survey of Graph Networks", citation_count=2),
        make_paper(paper_id="cited_high", title="A popular method", citation_count=100),
    ]
    path = _fallback_reading_path(papers, {})
    ids = [s["paper_id"] for s in path]
    assert ids[0] == "survey"                 # survey first
    assert ids[1] == "cited_high"             # then most-cited
    assert ids[2] == "cited_low"
    assert ids[3] == "rest"                   # then the rest
    assert [s["position"] for s in path] == [1, 2, 3, 4]


def test_fallback_caps_at_ten_steps():
    papers = [make_paper(paper_id=f"p{i}", citation_count=i) for i in range(15)]
    path = _fallback_reading_path(papers, {})
    assert len(path) == 10
    assert [s["position"] for s in path] == list(range(1, 11))



# --------------------------------------------------------------------------- #
# The reading-path prompt budget (content, not just the canned response)
# --------------------------------------------------------------------------- #


def test_reading_path_prompt_caps_at_thirty_papers():
    """Regression: the prompt must cap the paper list at 30 (B.8)."""
    from prompts.synthesize import build_reading_path_prompt

    papers = [
        make_paper(paper_id=f"p{i}", title=f"GNN technique number {i}")
        for i in range(60)
    ]
    prompt = build_reading_path_prompt(TOPIC, papers, {})
    assert isinstance(prompt, str)
    assert "[p29]" in prompt
    assert "[p30]" not in prompt

