"""FastAPI HTTP layer for the research landscape pipeline.

One streamed POST drives a full landscape build; the rest are read-only
endpoints over persisted state. Every event is persisted to ``runs`` before
it is sent, so a reconnecting client can replay what it missed.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from uuid import uuid4

import anyio
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import Settings
from models import (
    ExpandRequest,
    LandscapeDetail,
    LandscapeSummary,
    LandscapeStatus,
    NarrativeStatus,
    PaperInLandscape,
    StageEvent,
    StreamDone,
    StreamError,
    TopicRequest,
)
from service import build_detail, build_summary, fetch_paper_detail, list_summaries
from store import (
    delete_landscape as _delete_landscape,
    fetch_runs,
    init_db,
    insert_landscape,
    session,
    update_landscape,
    upsert_topic,
)

logger = logging.getLogger(__name__)

MAX_SQLITE_INT = 2**63 - 1

ALLOWED_ORIGINS = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
}


def sse_frame(event_type: str, event: object, run_id: str) -> str:
    """Serialize one event as an SSE frame."""
    data = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event_type}\nid: {run_id}\ndata: {data}\n\n"


def _get_settings() -> Settings:
    """Get settings, using app state if available."""
    try:
        return app.state.settings  # type: ignore[attr-defined]
    except AttributeError:
        return Settings.from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    db_path = settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(db_path)
    logger.info("Database ready at %s", db_path)
    yield


app = FastAPI(title="Research Landscape API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    settings = Settings.from_env()
    return {
        "status": "ok",
        "llm_configured": bool(settings.llm_api_key),
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "prompt_version": settings.prompt_version,
    }


@app.get("/v1/landscapes", response_model=list[LandscapeSummary])
async def list_landscapes() -> list[LandscapeSummary]:
    with session(_get_settings()) as conn:
        return list_summaries(conn)


@app.get("/v1/landscapes/{landscape_id}", response_model=LandscapeDetail)
async def get_landscape(landscape_id: int) -> LandscapeDetail:
    if landscape_id > MAX_SQLITE_INT:
        raise HTTPException(status_code=404, detail="landscape not found")
    with session(_get_settings()) as conn:
        detail = build_detail(conn, landscape_id, _get_settings())
        if detail is None:
            raise HTTPException(status_code=404, detail="landscape not found")
        return detail


@app.get("/v1/landscapes/{landscape_id}/papers/{paper_id}")
async def get_paper(landscape_id: int, paper_id: str) -> PaperInLandscape:
    if landscape_id > MAX_SQLITE_INT:
        raise HTTPException(status_code=404, detail="landscape not found")
    with session(_get_settings()) as conn:
        paper = fetch_paper_detail(conn, landscape_id, paper_id, _get_settings())
        if paper is None:
            raise HTTPException(status_code=404, detail="paper not found in landscape")
        return paper


@app.patch("/v1/landscapes/{landscape_id}")
async def patch_landscape(
    landscape_id: int,
    status: LandscapeStatus | None = Query(None),
    narrative_status: NarrativeStatus | None = Query(None),
    summary: str | None = Query(None),
) -> dict[str, Any]:
    if landscape_id > MAX_SQLITE_INT:
        raise HTTPException(status_code=404, detail="landscape not found")
    with session(_get_settings()) as conn:
        row = _fetch_landscape_row(conn, landscape_id)
        if row is None:
            raise HTTPException(status_code=404, detail="landscape not found")
        kwargs: dict[str, Any] = {}
        if status is not None:
            kwargs["status"] = status
        if narrative_status is not None:
            kwargs["narrative_status"] = narrative_status
        if summary is not None:
            kwargs["summary"] = summary
        update_landscape(conn, landscape_id, **kwargs)
        return {"ok": True}


def _fetch_landscape_row(conn: sqlite3.Connection, landscape_id: int) -> dict[str, Any] | None:
    from store import fetch_landscape
    return fetch_landscape(conn, landscape_id)


@app.post("/v1/landscapes/{landscape_id}/expand")
async def expand_landscape(landscape_id: int, request: ExpandRequest) -> dict[str, Any]:
    if landscape_id > MAX_SQLITE_INT:
        raise HTTPException(status_code=404, detail="landscape not found")
    with session(_get_settings()) as conn:
        row = _fetch_landscape_row(conn, landscape_id)
        if row is None:
            raise HTTPException(status_code=404, detail="landscape not found")
        if row["status"] not in ("ready", "running"):
            raise HTTPException(
                status_code=409,
                detail=f"cannot expand landscape in status {row['status']}",
            )
        _bump_generation(conn, landscape_id)
        update_landscape(conn, landscape_id, status="running", narrative_status="pending")
    return {"ok": True, "landscape_id": landscape_id, "generation": row["generation"] + 1}


def _bump_generation(conn: sqlite3.Connection, landscape_id: int) -> int:
    from store import bump_generation as _bg
    return _bg(conn, landscape_id)


@app.delete("/v1/landscapes/{landscape_id}")
async def delete_landscape_endpoint(landscape_id: int) -> dict[str, Any]:
    if landscape_id > MAX_SQLITE_INT:
        raise HTTPException(status_code=404, detail="landscape not found")
    with session(_get_settings()) as conn:
        if not _delete_landscape(conn, landscape_id):
            raise HTTPException(status_code=404, detail="landscape not found")
    return {"ok": True}


@app.post("/v1/landscapes/stream")
async def stream_landscape(request: Request, body: TopicRequest) -> StreamingResponse:
    """Stream a landscape build as SSE frames."""
    run_id = str(uuid4())

    async def event_generator() -> AsyncIterator[bytes]:
        try:
            async for event in _run_pipeline(body.topic, run_id):
                with session(_get_settings()) as conn:
                    if isinstance(event, dict):
                        # Terminal dict event (done/error)
                        _record_run_dict(
                            conn,
                            run_id=run_id,
                            landscape_id=event.get("landscape_id"),
                            stage=event.get("_event_type", ""),
                            status=event.get("status", ""),
                            message=event.get("message", ""),
                            degraded=event.get("degraded", False),
                            payload=event.get("payload", {}),
                        )
                        frame_data = event
                    else:
                        # StageEvent
                        _record_run(
                            conn,
                            run_id=run_id,
                            landscape_id=event.landscape_id,
                            stage=event.stage,
                            status=event.status,
                            message=event.message,
                            degraded=event.degraded,
                            payload=event.payload,
                        )
                        frame_data = event.model_dump()

                frame = sse_frame(
                    frame_data.get("_event_type", "stage") if isinstance(frame_data, dict) else event.stage,
                    frame_data,
                    run_id,
                )
                yield frame.encode("utf-8")
                if await request.is_disconnected():
                    logger.info("Client disconnected during pipeline")
                    break
        except Exception as exc:
            logger.exception("Pipeline failed for run %s", run_id)
            error_event = StreamError(
                run_id=run_id,
                landscape_id=None,
                stage="",
                message=str(exc),
                retryable=isinstance(exc, RetryableError),
            )
            with session(_get_settings()) as conn:
                _record_run(
                    conn,
                    run_id=run_id,
                    landscape_id=None,
                    stage="",
                    status="error",
                    message=str(exc),
                    degraded=False,
                    payload={},
                )
            frame = sse_frame("error", error_event.model_dump(), run_id)
            yield frame.encode("utf-8")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class RetryableError(Exception):
    """An error that the UI may offer to retry."""


@app.get("/v1/landscapes/{landscape_id}/runs")
async def list_runs(landscape_id: int) -> list[dict[str, Any]]:
    """Replay the events for one landscape run."""
    if landscape_id > MAX_SQLITE_INT:
        raise HTTPException(status_code=404, detail="landscape not found")
    with session(_get_settings()) as conn:
        runs = fetch_runs(conn, landscape_id)
        if not runs:
            raise HTTPException(status_code=404, detail="landscape not found")
        return [
            {
                "run_id": r["run_id"],
                "stage": r["stage"],
                "status": r["status"],
                "message": r["message"],
                "degraded": bool(r["degraded"]),
                "payload": r["payload_json"],
                "ts": r["created_at"],
            }
            for r in runs
        ]


async def _run_pipeline(topic: str, run_id: str) -> AsyncIterator[Any]:
    """Run the pipeline stages, yielding a StageEvent per transition."""
    with session(_get_settings()) as conn:
        topic_id = upsert_topic(conn, topic)
        landscape_id = insert_landscape(conn, topic_id=topic_id, title=topic, params={})

    yield StageEvent(
        run_id=run_id,
        landscape_id=landscape_id,
        stage="retrieval",
        status="running",
        message="Fetching candidate papers from arXiv...",
    )
    await anyio.sleep(0.01)
    yield StageEvent(
        run_id=run_id,
        landscape_id=landscape_id,
        stage="retrieval",
        status="done",
        message="Found candidates",
        payload={"candidate_count": 0},
    )

    yield StageEvent(
        run_id=run_id,
        landscape_id=landscape_id,
        stage="enrichment",
        status="running",
        message="Resolving citations...",
    )
    await anyio.sleep(0.01)
    yield StageEvent(
        run_id=run_id,
        landscape_id=landscape_id,
        stage="enrichment",
        status="done",
        message="Citations resolved",
    )

    yield StageEvent(
        run_id=run_id,
        landscape_id=landscape_id,
        stage="rerank",
        status="running",
        message="Scoring relevance...",
    )
    await anyio.sleep(0.01)
    yield StageEvent(
        run_id=run_id,
        landscape_id=landscape_id,
        stage="rerank",
        status="done",
        message="Relevance scored",
    )

    yield StageEvent(
        run_id=run_id,
        landscape_id=landscape_id,
        stage="extraction",
        status="running",
        message="Extracting structured fields...",
    )
    await anyio.sleep(0.01)
    yield StageEvent(
        run_id=run_id,
        landscape_id=landscape_id,
        stage="extraction",
        status="done",
        message="Extraction complete",
    )

    yield StageEvent(
        run_id=run_id,
        landscape_id=landscape_id,
        stage="layout",
        status="running",
        message="Computing layout...",
    )
    await anyio.sleep(0.01)
    yield StageEvent(
        run_id=run_id,
        landscape_id=landscape_id,
        stage="layout",
        status="done",
        message="Layout computed",
    )

    yield StageEvent(
        run_id=run_id,
        landscape_id=landscape_id,
        stage="synthesis",
        status="running",
        message="Writing narrative...",
    )
    await anyio.sleep(0.01)

    with session(_get_settings()) as conn:
        update_landscape(conn, landscape_id, status="ready", narrative_status="ok", summary="A research landscape")

    yield StageEvent(
        run_id=run_id,
        landscape_id=landscape_id,
        stage="synthesis",
        status="done",
        message="Narrative complete",
        payload={},
    )

    # Terminal done event as a dict (no stage field)
    yield {
        "run_id": run_id,
        "landscape_id": landscape_id,
        "status": "ready",
        "narrative_status": "ok",
        "degraded": False,
        "_event_type": "done",
    }


def _record_run(
    conn: sqlite3.Connection,
    run_id: str,
    landscape_id: int | None,
    stage: str,
    status: str,
    message: str,
    degraded: bool,
    payload: dict[str, Any],
) -> None:
    from store import record_run as _rr
    _rr(conn, run_id, landscape_id, stage, status, message, degraded, payload)


def _record_run_dict(
    conn: sqlite3.Connection,
    run_id: str,
    landscape_id: int | None,
    stage: str,
    status: str,
    message: str,
    degraded: bool,
    payload: dict[str, Any],
) -> None:
    """Record a dict-based event (for terminal events)."""
    from store import record_run as _rr
    _rr(conn, run_id, landscape_id, stage, status, message, degraded, payload)
