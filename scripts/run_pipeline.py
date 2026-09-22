#!/usr/bin/env python
"""Inspectable pipeline stages — Phase 5, Task 5.5 (extraction skeleton).

Every stage is runnable from here before any UI exists. Later phases add
their stages behind the same CLI:

    python scripts/run_pipeline.py --stage extract --limit 15
    python scripts/run_pipeline.py --stage extract --limit 15 --json
    python scripts/run_pipeline.py --topic "diffusion policy learning" --stage layout

The extract stage prints one row per paper — ``paper_id | status | problem
(truncated)`` — plus a coverage line (``ok / failed / skipped`` and the
ratio). It reads papers from the local database (seeded with ``--seed`` for
offline inspection), runs :func:`pipeline.extract.extract_all`, and persists
every row under ``PROMPT_VERSION`` — so a second run on the same corpus issues
zero LLM calls via the cache.

The layout stage (Task 6.5) runs retrieve -> rerank -> persist membership ->
:func:`pipeline.stages.layout_stage` for a topic, then prints the cluster
table (local label, size, centroid, colour) and the SQL invariant
``SUM(clusters.size) == landscape_papers`` including the unclustered bucket.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Make the api package importable when this script is run from the repo root.
API_ROOT = Path(__file__).resolve().parent.parent / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from config import ConfigError, load_settings  # noqa: E402
from models import Paper  # noqa: E402

STAGES = ("extract", "layout")

#: Human-readable progress goes here. In ``--json`` mode this is redirected to
#: stderr so that stdout carries nothing but parseable JSON.
_PROGRESS = sys.stdout


def say(message: str = "") -> None:
    print(message, file=_PROGRESS)


def _truncate(text: str, width: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _seed_papers(limit: int) -> list[Paper]:
    """Deterministic offline seed corpus for ``--seed`` inspection runs."""
    from tests.conftest import make_paper  # noqa: E402

    papers: list[Paper] = []
    for i in range(limit):
        pid = f"2406.{10000 + i}"
        abstract = (
            f"Problem statement for paper {pid} studied here. "
            f"Method construction for paper {pid} built here. "
            f"Results measurement for paper {pid} reported here with 48.7 percent gain. "
            f"Contribution claim for paper {pid} shown here. "
            "The approach is not evaluated on open-ended generation tasks at all."
        )
        papers.append(make_paper(paper_id=pid, title=f"Seed paper {i} ({pid})", abstract=abstract))
    return papers


def _load_corpus(settings, limit: int, *, seed: bool, seed_grounded: bool = False) -> list[Paper]:
    """Papers for the extract stage: DB rows, or a seeded offline corpus."""
    import store

    if seed or seed_grounded:
        papers = _seed_papers(limit)
        store.init_db(settings.db_path)
        conn = store.connect(settings.db_path)
        try:
            store.upsert_papers(conn, papers)
            if seed_grounded:
                _seed_grounded_extractions(conn, papers, settings.prompt_version)
            conn.commit()
        finally:
            conn.close()
        return papers
    try:
        conn = store.connect(settings.db_path)
    except OSError:
        return []
    try:
        rows = conn.execute("SELECT * FROM papers ORDER BY paper_id ASC LIMIT ?", (limit,)).fetchall()
        return [store._row_to_paper(row) for row in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _seed_grounded_extractions(conn, papers: list[Paper], prompt_version: str) -> None:
    """Write scripted grounded extractions so ``--seed-grounded`` demos ok rows offline."""
    import store
    from models import PaperExtraction

    for paper in papers:
        pid = paper.paper_id
        extraction = PaperExtraction(
            problem=f"Problem statement for paper {pid} studied here.",
            method=f"Method construction for paper {pid} built here.",
            results="Results measurement reported with 48.7 percent gain.",
            contribution=f"Contribution claim for paper {pid} shown here.",
            limitations="The approach is not evaluated on open-ended generation tasks at all.",
            evidence={
                "problem": f"Problem statement for paper {pid} studied here.",
                "method": f"Method construction for paper {pid} built here.",
                "results": f"Results measurement for paper {pid} reported here with 48.7 percent gain.",
                "contribution": f"Contribution claim for paper {pid} shown here.",
                "limitations": "not evaluated on open-ended generation tasks at all.",
            },
            status="ok",
            error="",
        )
        store.upsert_extraction(conn, pid, prompt_version, extraction, model="scripted-seed")


def stage_extract(args, settings) -> int:
    """Run ``--stage extract`` and print the paper table + coverage line."""
    from pipeline.extract import extract_all

    papers = _load_corpus(settings, args.limit, seed=args.seed, seed_grounded=args.seed_grounded)
    say(f"\n[extract] {len(papers)} papers (prompt_version={settings.prompt_version})")
    if not papers:
        say("  (no papers — rerun with --seed to inspect the offline corpus)")
        say("[extract] coverage: ok=0 failed=0 skipped=0 ratio=1.00")
        return 0

    completer = None if args.no_llm else _make_completer(settings)

    def on_progress(event: dict) -> None:
        say(f"[extract] progress {event['done']}/{event['total']}")

    extractions, coverage = asyncio.run(extract_all(papers, settings, completer=completer, progress=on_progress))

    if args.json:
        payload = {
            "coverage": {
                "ok": coverage.ok,
                "failed": coverage.failed,
                "skipped": coverage.skipped,
                "ratio": coverage.ratio,
            },
            "extractions": {pid: extraction.model_dump() for pid, extraction in extractions.items()},
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        say(f"  {'paper_id':<16}  {'status':<7}  problem")
        for paper in papers:
            extraction = extractions.get(paper.paper_id)
            status = extraction.status if extraction else "failed"
            problem = _truncate(extraction.problem if extraction and extraction.problem else "", 70)
            say(f"  {paper.paper_id:<16}  {status:<7}  {problem}")
    say(
        f"[extract] coverage: ok={coverage.ok} failed={coverage.failed} "
        f"skipped={coverage.skipped} ratio={coverage.ratio:.2f}"
    )
    llm_calls = _count_llm_calls(settings)
    say(f"[extract] llm_calls={llm_calls}")
    return 0


def stage_layout(args, settings) -> int:
    """Run ``--topic ... --stage layout``: retrieve → rerank → layout → print.

    Prints the cluster table (local label, size, centroid, colour) and checks
    the Task 6.5 SQL invariant: ``SUM(clusters.paper_count)`` must equal the
    number of linked ``landscape_papers`` **including** the ``-1`` bucket.
    Non-zero exit on a broken invariant, so CI can gate on it.
    """
    import store
    from pipeline import rerank as rerank_mod
    from pipeline import retrieve as retrieve_mod
    from pipeline.stages import layout_stage

    if not args.topic:
        print("--stage layout requires --topic", file=sys.stderr)
        return 2

    say(f"\n[retrieve] searching arXiv for {args.topic!r}")
    try:
        papers = retrieve_mod.fetch_candidates(args.topic, settings, force_refresh=args.refresh)
    except retrieve_mod.RetrievalOffline as exc:
        print(f"[retrieve] {exc}", file=sys.stderr)
        return 1
    say(f"[retrieve] {len(papers)} unique papers after version dedup")
    if not papers:
        print(f"[retrieve] no papers for {args.topic!r}; nothing to lay out", file=sys.stderr)
        return 1

    say(f"\n[rerank] scoring {len(papers)} candidates")
    ranked_all = rerank_mod.rank_papers(args.topic, papers, settings, judge=None)
    ranked = rerank_mod.select_top(ranked_all, settings.rerank_final_count_capped)
    counts: dict[str, int] = {}
    for item in ranked:
        counts[item.rerank_source] = counts.get(item.rerank_source, 0) + 1
    say(f"[rerank] selected {len(ranked)} papers; sources: {counts}")

    # A fresh landscape per CLI run: the orchestrator (Phase 8) owns expand
    # semantics, so a diagnostic run must not silently mutate an existing map.
    with store.session(settings) as conn:
        topic_id = store.upsert_topic(conn, args.topic)
        landscape_id = store.insert_landscape(conn, topic_id=topic_id, title=args.topic)
        store.upsert_papers(conn, [item.paper for item in ranked])
        for item in ranked:
            store.link_paper(
                conn,
                landscape_id,
                paper_id=item.paper.paper_id,
                rank=item.rank,
                relevance_score=item.relevance_score,
                rerank_source=item.rerank_source,
                cross_encoder_logit=item.cross_encoder_logit,
            )

        completer = None if args.no_llm else _make_completer(settings)
        say(f"\n[layout] embedding, projecting, clustering {len(ranked)} papers")
        payload = layout_stage(
            landscape_id,
            ranked,
            args.topic,
            settings,
            conn,
            completer=completer,
        )

        sum_size = conn.execute(
            "SELECT COALESCE(SUM(paper_count), 0) AS s FROM clusters WHERE landscape_id = ?",
            (landscape_id,),
        ).fetchone()["s"]
        papers_in_map = conn.execute(
            "SELECT COUNT(*) AS n FROM landscape_papers WHERE landscape_id = ?",
            (landscape_id,),
        ).fetchone()["n"]

    say(f"\n[layout] landscape {landscape_id}: {payload['cluster_count']} cluster(s), "
        f"{payload['unclustered_count']} unclustered")
    if args.json:
        print(json.dumps({**payload, "landscape_id": landscape_id}, indent=2, ensure_ascii=False))
    else:
        say(f"  {'local':>5}  {'size':>4}  {'x':>7}  {'y':>7}  {'color':<9}  label")
        for row in payload["clusters"]:
            say(f"  {row['local_label']:>5}  {row['paper_count']:>4}  {row['x']:>7.4f}  "
                f"{row['y']:>7.4f}  {row['color']:<9}  {row['label']}")
        say(f"\n[layout] SQL check: SUM(clusters.paper_count)={sum_size} "
            f"landscape_papers={papers_in_map} -> "
            f"{'OK' if sum_size == papers_in_map else 'MISMATCH'}")
    if sum_size != papers_in_map:
        print(f"[layout] invariant broken: cluster sizes sum to {sum_size} but "
              f"{papers_in_map} papers are linked", file=sys.stderr)
        return 1
    return 0


def _count_llm_calls(settings) -> int:
    import sqlite3

    try:
        conn = sqlite3.connect(str(settings.db_path), timeout=30.0)
    except OSError:
        return 0
    try:
        try:
            row = conn.execute("SELECT COUNT(*) AS n FROM llm_calls").fetchone()
        except sqlite3.OperationalError:
            return 0
        if row is None:
            return 0
        return int(row[0])
    finally:
        conn.close()


def _make_completer(settings):
    """Build the LLM client, or ``None`` when no key is configured."""
    if not settings.llm_api_key and settings.llm_provider != "local":
        print("[llm] no LLM key configured; running without it", file=sys.stderr)
        return None
    try:
        from llm.client import build_completer  # noqa: E402
    except ImportError:
        print("[extract] LLM client unavailable; running without it", file=sys.stderr)
        return None
    return build_completer(settings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the research landscape pipeline from the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--topic", default=None, help="Plain-English ML topic (required for --stage layout).")
    parser.add_argument("--stage", choices=STAGES, default="extract", help="Which stage to run and print.")
    parser.add_argument("--limit", type=int, default=15, help="Rows to print (default: 15).")
    parser.add_argument("--seed", action="store_true", help="Seed a deterministic offline corpus first.")
    parser.add_argument(
        "--seed-grounded", action="store_true", help="Seed scripted grounded extractions (offline demo)."
    )
    parser.add_argument("--refresh", action="store_true", help="Bypass the fresh arXiv cache (layout stage).")
    parser.add_argument("--no-llm", action="store_true", help="Skip model calls (ledger papers as failed/skipped).")
    parser.add_argument("--db", default=None, help="Override the database path.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    global _PROGRESS
    args = build_parser().parse_args(argv)
    if args.json:
        _PROGRESS = sys.stderr
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    try:
        settings = load_settings(require_llm=False)
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2
    if args.db:
        from dataclasses import replace

        settings = replace(settings, db_path=Path(args.db))

    if args.stage == "extract":
        return stage_extract(args, settings)
    if args.stage == "layout":
        return stage_layout(args, settings)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
