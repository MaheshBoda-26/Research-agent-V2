"""Tests for the FastAPI HTTP layer.

These tests assert the SSE frame shape and error semantics without a real
network.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from dataclasses import replace

from config import Settings
from main import app, sse_frame
from models import StageEvent, StreamDone, StreamError, utcnow
from store import (
    delete_landscape,
    fetch_runs,
    init_db,
    insert_landscape,
    link_paper,
    session,
    upsert_papers,
    upsert_topic,
    update_landscape,
)


def _make_paper(paper_id: str, title: str = "Test Paper") -> Any:
    from models import Paper
    return Paper(
        paper_id=paper_id, version="", title=title, abstract="Abstract text.",
        authors=[], published="", updated="", primary_category="", categories=[],
        comment="", journal_ref="", doi="", abs_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url="", citation_count=None, citation_source="", openalex_id="",
        s2_paper_id="", fulltext_status="none", fetched_at=utcnow(),
    )


def _seed_landscape(conn: sqlite3.Connection, paper_count: int = 3) -> int:
    papers = [_make_paper(f"2107.000{i}", title=f"Paper {i}") for i in range(paper_count)]
    upsert_papers(conn, papers)
    topic_id = upsert_topic(conn, "test topic")
    landscape_id = insert_landscape(conn, topic_id=topic_id, title="Test Landscape", params={})
    for rank, paper in enumerate(papers, start=1):
        link_paper(conn, landscape_id, paper_id=paper.paper_id, rank=rank,
                   relevance_score=10.0 - rank, rerank_source="cross-encoder",
                   is_seed=rank == 1, cross_encoder_logit=float(rank))
    return landscape_id


def _decode_sse_frame(frame: str) -> dict[str, Any]:
    lines = frame.strip().split("\n")
    event_type = ""
    run_id = ""
    data = ""
    for line in lines:
        if line.startswith("event: "):
            event_type = line[len("event: "):]
        elif line.startswith("id: "):
            run_id = line[len("id: "):]
        elif line.startswith("data: "):
            data = line[len("data: "):]
    assert event_type and data, f"Malformed frame: {frame!r}"
    payload = json.loads(data)
    payload["_event_type"] = event_type
    payload["_run_id"] = run_id
    return payload


def test_health_returns_ok() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"


def test_list_landscapes_returns_list() -> None:
    with TestClient(app) as client:
        resp = client.get("/v1/landscapes")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


def test_create_and_read_landscape_via_stream() -> None:
    with TestClient(app) as client:
        with client.stream("POST", "/v1/landscapes/stream", json={"topic": "test topic"}) as resp:
            assert resp.status_code == 200
            frames = []
            for chunk in resp.iter_raw():
                text = chunk.decode("utf-8")
                for frame in text.split("\n\n"):
                    if frame.strip():
                        try:
                            frames.append(_decode_sse_frame(frame))
                        except (json.JSONDecodeError, AssertionError):
                            pass

        assert len(frames) > 0
        statuses = [f.get("status") for f in frames]
        assert "done" in statuses or "ready" in statuses
        landscape_id = frames[-1].get("landscape_id")
        assert landscape_id is not None

        detail_resp = client.get(f"/v1/landscapes/{landscape_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["topic"] == "test topic"
        assert detail["status"] == "ready"
        assert detail["narrative_status"] == "ok"


def test_get_nonexistent_landscape_returns_404() -> None:
    with TestClient(app) as client:
        resp = client.get("/v1/landscapes/99999")
        assert resp.status_code == 404


def test_patch_landscape_status(fresh_db: Path, settings: Settings) -> None:
    test_settings = replace(settings, db_path=fresh_db)
    app.state.settings = test_settings
    with session(test_settings) as conn:
        landscape_id = _seed_landscape(conn)

    with TestClient(app) as client:
        resp = client.patch(f"/v1/landscapes/{landscape_id}", params={"status": "ready"})
        assert resp.status_code == 200
        detail_resp = client.get(f"/v1/landscapes/{landscape_id}")
        assert detail_resp.json()["status"] == "ready"


def test_delete_landscape(fresh_db: Path, settings: Settings) -> None:
    test_settings = replace(settings, db_path=fresh_db)
    app.state.settings = test_settings
    with session(test_settings) as conn:
        landscape_id = _seed_landscape(conn)

    with TestClient(app) as client:
        resp = client.delete(f"/v1/landscapes/{landscape_id}")
        assert resp.status_code == 200
        detail_resp = client.get(f"/v1/landscapes/{landscape_id}")
        assert detail_resp.status_code == 404


def test_sse_frame_shape() -> None:
    event = StageEvent(run_id="test-run", landscape_id=1, stage="retrieval",
                       status="running", message="Fetching...")
    frame = sse_frame("stage", event.model_dump(), "test-run")
    lines = frame.strip().split("\n")
    assert lines[0] == "event: stage"
    assert lines[1] == "id: test-run"
    assert lines[2].startswith("data: ")


def test_stream_done_frame_has_status() -> None:
    done = StreamDone(run_id="r1", landscape_id=1, status="ready",
                      narrative_status="ok", degraded=False)
    payload = done.model_dump()
    assert payload["status"] == "ready"
    assert payload["narrative_status"] == "ok"
    assert payload["degraded"] is False


def test_oversized_landscape_id_returns_404() -> None:
    with TestClient(app) as client:
        resp = client.get("/v1/landscapes/99999999999999999999")
        assert resp.status_code == 404


def test_stream_invalid_topic_returns_422() -> None:
    with TestClient(app) as client:
        resp = client.post("/v1/landscapes/stream", json={"topic": "x"})
        assert resp.status_code == 422


def test_runs_replay(fresh_db: Path, settings: Settings) -> None:
    test_settings = replace(settings, db_path=fresh_db)
    app.state.settings = test_settings
    with session(test_settings) as conn:
        landscape_id = _seed_landscape(conn)
    with session(test_settings) as conn:
        import uuid
        run_id = str(uuid.uuid4())
        from store import record_run
        record_run(conn, run_id, landscape_id, "retrieval", "done", "ok", False, {})
        record_run(conn, run_id, landscape_id, "enrichment", "done", "ok", False, {})

    with TestClient(app) as client:
        resp = client.get(f"/v1/landscapes/{landscape_id}/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 2
        assert runs[0]["stage"] == "retrieval"
        assert runs[1]["stage"] == "enrichment"


def test_runs_nonexistent_landscape_returns_404() -> None:
    with TestClient(app) as client:
        resp = client.get("/v1/landscapes/99999/runs")
        assert resp.status_code == 404


def test_cors_allows_localhost_origin() -> None:
    with TestClient(app) as client:
        resp = client.get("/v1/landscapes", headers={"Origin": "http://localhost:3000"})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
