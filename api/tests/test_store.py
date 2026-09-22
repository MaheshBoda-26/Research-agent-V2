"""Tests for api/store.py — the plan §11.4 row for store:

migrate() upgrades a V1-shaped database with all rows intact; extraction cache
keyed by (paper_id, prompt_version); session() rolls back on exception.
Plus the A.3 surface: CRUD round-trips, citations, source_cache TTL, runs
replay, llm_calls rollup, prune, cascade delete.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import store
from models import PaperExtraction
from store import StoreError

from conftest import make_paper

def _seed_landscape(conn: sqlite3.Connection, paper_count: int = 3) -> int:
    """A topic, a landscape, and paper_count linked papers. Returns landscape id."""
    papers = [make_paper(f"2107.000{i}") for i in range(paper_count)]
    store.upsert_papers(conn, papers)
    topic_id = store.upsert_topic(conn, "retrieval-augmented generation")
    landscape_id = store.insert_landscape(
        conn, topic_id=topic_id, title="RAG", params={"retrieval_max_results": 200}
    )
    for rank, paper in enumerate(papers, start=1):
        store.link_paper(
            conn,
            landscape_id,
            paper_id=paper.paper_id,
            rank=rank,
            relevance_score=10.0 - rank,
            rerank_source="cross-encoder+citation",
            is_seed=rank == 1,
            cross_encoder_logit=float(rank),
        )
    return landscape_id


# --------------------------------------------------------------------------- #
# Topics / papers / landscapes CRUD
# --------------------------------------------------------------------------- #


def test_topic_upsert_is_case_and_whitespace_insensitive(conn) -> None:
    first = store.upsert_topic(conn, "Retrieval-Augmented  Generation")
    second = store.upsert_topic(conn, " retrieval-augmented generation ")
    assert first == second
    assert store.fetch_topic(conn, first)["query_text"] == "Retrieval-Augmented Generation"


def test_paper_upsert_round_trip_and_preserves_enrichment(conn) -> None:
    paper = make_paper("2107.05580", citation_count=128, citation_source="openalex")
    assert store.upsert_papers(conn, [paper]) == 1
    assert store.upsert_papers(conn, [paper]) == 0  # already known
    fetched = store.fetch_paper(conn, "2107.05580")
    assert fetched is not None
    assert fetched.title == "Title for 2107.05580"
    assert fetched.citation_count == 128
    assert fetched.citation_source == "openalex"

    # A retrieval re-run (no enrichment fields) must not wipe resolved citations.
    refreshed = make_paper("2107.05580", version="2")
    store.upsert_papers(conn, [refreshed])
    after = store.fetch_paper(conn, "2107.05580")
    assert after is not None
    assert after.version == "2"
    assert after.citation_count == 128  # preserved, not wiped to NULL


def test_landscape_crud_round_trip(conn) -> None:
    landscape_id = _seed_landscape(conn)
    row = store.fetch_landscape(conn, landscape_id)
    assert row is not None
    assert row["topic"] == "retrieval-augmented generation"
    assert row["narrative_status"] == "pending"
    assert row["status"] == "running"

    store.update_landscape(
        conn, landscape_id, status="ready", narrative_status="ok",
        summary="A map.", cost_usd=0.0123, tokens_in=100, tokens_out=50,
    )
    row = store.fetch_landscape(conn, landscape_id)
    assert row is not None
    assert row["status"] == "ready"
    assert row["narrative_status"] == "ok"
    assert row["cost_usd"] == pytest.approx(0.0123)
    assert row["tokens_in"] == 100 and row["tokens_out"] == 50

    listed = store.list_landscapes(conn)
    assert [item["id"] for item in listed] == [landscape_id]
    assert listed[0]["paper_count"] == 3

    assert store.bump_generation(conn, landscape_id) == 2
    assert store.delete_landscape(conn, landscape_id) is True
    assert store.delete_landscape(conn, landscape_id) is False
    assert store.fetch_landscape(conn, landscape_id) is None


def test_update_landscape_rejects_unknown_field(conn) -> None:
    landscape_id = _seed_landscape(conn)
    with pytest.raises(StoreError, match="unknown fields"):
        store.update_landscape(conn, landscape_id, no_such_column=1)


def test_link_paper_membership_and_fetch(conn) -> None:
    landscape_id = _seed_landscape(conn)
    ids = store.landscape_paper_ids(conn, landscape_id)
    assert ids == {"2107.0000", "2107.0001", "2107.0002"}
    papers = store.fetch_landscape_papers(conn, landscape_id)
    assert [p["rank"] for p in papers] == [1, 2, 3]
    assert papers[0]["is_seed"] is True
    assert papers[0]["rerank_source"] == "cross-encoder+citation"
    assert papers[0]["authors"] == ["A. Author"]

    # Re-linking preserves coordinates (expand must not lose the layout).
    store.update_layout(conn, landscape_id, {"2107.0000": (0.5, -0.25, 7)})
    store.link_paper(
        conn, landscape_id, paper_id="2107.0000", rank=1, relevance_score=9.9,
        rerank_source="cross-encoder+citation",
    )
    relinked = store.fetch_landscape_papers(conn, landscape_id)
    first = next(p for p in relinked if p["paper_id"] == "2107.0000")
    assert first["x"] == 0.5 and first["y"] == -0.25 and first["cluster_id"] == 7
    assert first["relevance_score"] == pytest.approx(9.9)



# --------------------------------------------------------------------------- #
# Embeddings / extractions
# --------------------------------------------------------------------------- #


def test_embedding_round_trip(conn) -> None:
    store.upsert_papers(conn, [make_paper("p1"), make_paper("p2")])
    blob = b"\x01\x02\x03\x04" * 4
    store.upsert_embedding(conn, "p1", "BAAI/bge-base-en-v1.5", blob, 4)
    store.upsert_embeddings(conn, "BAAI/bge-base-en-v1.5", {"p2": blob})
    found = store.fetch_embeddings(conn, ["p1", "p2", "p3"], "BAAI/bge-base-en-v1.5")
    assert set(found) == {"p1", "p2"}
    assert found["p1"] == blob
    assert store.fetch_embeddings(conn, [], "BAAI/bge-base-en-v1.5") == {}


def test_extraction_cache_is_keyed_by_prompt_version(conn) -> None:
    """Task 5.4's store half: writing extract_v2 must not disturb extract_v1."""
    store.upsert_papers(conn, [make_paper("2107.05580")])
    v1 = PaperExtraction(problem="old read", novelty="unclear")
    v2 = PaperExtraction(problem="new read", novelty="substantial")
    store.upsert_extraction(conn, "2107.05580", "extract_v1", v1, model="m")
    store.upsert_extraction(conn, "2107.05580", "extract_v2", v2, model="m")

    old = store.fetch_extractions(conn, ["2107.05580"], "extract_v1")
    assert old["2107.05580"].problem == "old read"
    new = store.fetch_extractions(conn, ["2107.05580"], "extract_v2")
    assert new["2107.05580"].problem == "new read"
    # A bumped prompt deliberately misses the cache.
    assert store.fetch_extractions(conn, ["2107.05580"], "extract_v3") == {}


def test_failed_extraction_round_trip(conn) -> None:
    """D3: a failed extraction is a first-class row, never a silent absence."""
    store.upsert_papers(conn, [make_paper("2107.05580")])
    store.upsert_extraction(
        conn, "2107.05580", "extract_v2",
        PaperExtraction(status="failed", error="timeout after 60s"),
    )
    fetched = store.fetch_extraction(conn, "2107.05580", "extract_v2")
    assert fetched is not None
    assert fetched.status == "failed"
    assert fetched.error == "timeout after 60s"
    assert fetched.problem is None


# --------------------------------------------------------------------------- #
# Clusters / graph / synthesis outputs
# --------------------------------------------------------------------------- #


def test_clusters_edges_and_synthesis_outputs_round_trip(conn) -> None:
    landscape_id = _seed_landscape(conn)
    store.replace_clusters(
        conn, landscape_id,
        [
            {"local_label": 0, "label": "Dense retrieval", "paper_count": 2, "x": 1.0, "y": 2.0},
            {"local_label": -1, "label": "", "paper_count": 1},
        ],
    )
    clusters = store.fetch_clusters(conn, landscape_id)
    assert [c["local_label"] for c in clusters] == [-1, 0]
    assert clusters[1]["label"] == "Dense retrieval"
    assert store.cluster_id_by_label(conn, landscape_id) == {-1: clusters[0]["id"], 0: clusters[1]["id"]}

    store.replace_edges(
        conn, landscape_id,
        [
            {"src_paper_id": "2107.0000", "dst_paper_id": "2107.0001", "kind": "extends",
             "weight": 0.9, "source": "citation", "confidence": 1.0},
            {"src_paper_id": "2107.0001", "dst_paper_id": "2107.0002", "kind": "applies",
             "source": "knn", "confidence": 0.71},
        ],
    )
    edges = store.fetch_edges(conn, landscape_id)
    assert len(edges) == 2
    assert edges[0]["source"] == "citation" and edges[0]["confidence"] == 1.0
    assert edges[1]["source"] == "knn"

    store.replace_tensions(
        conn, landscape_id,
        [{"statement": "A claims X; B claims not-X", "paper_a_id": "2107.0000", "paper_b_id": "2107.0001"}],
    )
    assert store.fetch_tensions(conn, landscape_id)[0]["paper_b_id"] == "2107.0001"

    store.replace_open_problems(
        conn, landscape_id,
        [{"statement": "Multi-hop remains open", "why_open": "nobody benchmarks it",
          "supporting_paper_ids": ["2107.0000"]}],
    )
    problems = store.fetch_open_problems(conn, landscape_id)
    assert problems[0]["supporting_paper_ids"] == ["2107.0000"]

    store.replace_reading_path(
        conn, landscape_id,
        [{"paper_id": "2107.0000", "position": 1, "why": "foundational"},
         {"paper_id": "2107.0002", "position": 2, "why": "recent"}],
    )
    path = store.fetch_reading_path(conn, landscape_id)
    assert [s["position"] for s in path] == [1, 2]
    assert path[0]["title"] == "Title for 2107.0000"

    # replace_* are wholesale: writing again replaces, never appends.
    store.replace_edges(conn, landscape_id, [])
    assert store.fetch_edges(conn, landscape_id) == []


def test_every_output_id_must_reference_a_landscape_paper(conn) -> None:
    """§11.7 referential integrity, run as a SQL assertion."""
    landscape_id = _seed_landscape(conn)
    store.replace_edges(
        conn, landscape_id,
        [{"src_paper_id": "2107.0000", "dst_paper_id": "2107.0001", "kind": "extends"}],
    )
    store.replace_reading_path(conn, landscape_id, [{"paper_id": "2107.0000", "position": 1}])
    dangling = conn.execute(
        """
        SELECT src_paper_id AS pid FROM edges
        WHERE landscape_id = ? AND src_paper_id NOT IN
              (SELECT paper_id FROM landscape_papers WHERE landscape_id = ?)
        UNION ALL
        SELECT dst_paper_id FROM edges
        WHERE landscape_id = ? AND dst_paper_id NOT IN
              (SELECT paper_id FROM landscape_papers WHERE landscape_id = ?)
        UNION ALL
        SELECT paper_id FROM reading_path
        WHERE landscape_id = ? AND paper_id NOT IN
              (SELECT paper_id FROM landscape_papers WHERE landscape_id = ?)
        """,
        (landscape_id,) * 6,
    ).fetchall()
    assert dangling == []



# --------------------------------------------------------------------------- #
# Citations / caches / observability
# --------------------------------------------------------------------------- #


def test_citations_replace_and_fetch(conn) -> None:
    store.upsert_papers(conn, [make_paper(pid) for pid in ("a", "b", "c")])
    store.replace_citations(conn, "a", [("a", "b"), ("a", "c")], "semanticscholar")
    store.replace_citations(conn, "b", [("b", "c")], "openalex")
    triples = store.fetch_citations(conn, ["a", "b"])
    assert ("a", "b", "semanticscholar") in triples
    assert ("a", "c", "semanticscholar") in triples
    assert ("b", "c", "openalex") in triples

    # replace is wholesale per (src, source): refetching S2 drops stale links.
    store.replace_citations(conn, "a", [("a", "c")], "semanticscholar")
    triples = store.fetch_citations(conn, ["a"])
    assert ("a", "b", "semanticscholar") not in triples
    assert ("b", "c", "openalex") in triples  # other source untouched


def test_source_cache_respects_ttl(conn) -> None:
    store.put_cached(conn, "s2:paper:arXiv:2005.11401", "semanticscholar", {"citations": 42}, 3600)
    assert store.get_cached(conn, "s2:paper:arXiv:2005.11401", "semanticscholar") == {"citations": 42}
    # Wrong source namespace misses.
    assert store.get_cached(conn, "s2:paper:arXiv:2005.11401", "openalex") is None
    # Zero TTL expires immediately.
    store.put_cached(conn, "oa:work:W1", "openalex", {"x": 1}, 0)
    assert store.get_cached(conn, "oa:work:W1", "openalex") is None


def test_runs_replay_in_order(conn) -> None:
    landscape_id = _seed_landscape(conn)
    store.insert_run(conn, {"run_id": "r1", "landscape_id": landscape_id,
                            "stage": "retrieval", "status": "running"})
    store.insert_run(conn, {"run_id": "r1", "landscape_id": landscape_id,
                            "stage": "retrieval", "status": "done",
                            "message": "3 papers", "degraded": True,
                            "payload": {"count": 3}})
    events = store.fetch_runs(conn, landscape_id)
    assert [e["status"] for e in events] == ["running", "done"]
    assert events[1]["degraded"] is True
    assert events[1]["payload"] == {"count": 3}
    assert all(e["run_id"] == "r1" for e in events)


def test_llm_calls_and_cost_rollup(conn) -> None:
    landscape_id = _seed_landscape(conn)
    for stage, cost in (("extraction", 0.001), ("extraction", 0.002), ("synthesis", 0.004)):
        store.insert_llm_call(
            conn,
            {
                "run_id": "r1", "landscape_id": landscape_id, "stage": stage,
                "model": "meta/test-model", "provider": "nim",
                "prompt_tokens": 100, "completion_tokens": 20,
                "cost_usd": cost, "latency_ms": 800, "attempt": 1, "ok": True,
            },
        )
    store.insert_llm_call(
        conn,
        {"run_id": "r1", "stage": "extraction", "model": "meta/test-model",
         "provider": "nim", "ok": False, "validation_error": "missing field: problem"},
    )
    rollup = store.llm_cost_rollup(conn, days=30)
    extraction = next(r for r in rollup if r["stage"] == "extraction")
    assert extraction["calls"] == 3
    assert extraction["prompt_tokens"] == 200
    assert extraction["cost_usd"] == pytest.approx(0.003)
    synthesis = next(r for r in rollup if r["stage"] == "synthesis")
    assert synthesis["cost_usd"] == pytest.approx(0.004)


def test_prune_deletes_expired_cache_and_old_runs(conn) -> None:
    landscape_id = _seed_landscape(conn)
    store.put_cached(conn, "old", "openalex", {"x": 1}, 0)  # already expired
    store.put_cached(conn, "fresh", "openalex", {"x": 2}, 3600)
    store.insert_run(conn, {"run_id": "old", "landscape_id": landscape_id,
                            "stage": "retrieval", "status": "done",
                            "created_at": "2020-01-01T00:00:00Z"})
    store.insert_run(conn, {"run_id": "new", "landscape_id": landscape_id,
                            "stage": "retrieval", "status": "done"})
    counts = store.prune(conn, older_than_days=90)
    assert counts == {"source_cache": 1, "runs": 1}
    assert store.get_cached(conn, "fresh", "openalex") == {"x": 2}
    assert [e["run_id"] for e in store.fetch_runs(conn, landscape_id)] == ["new"]



# --------------------------------------------------------------------------- #
# Migration and lifecycle
# --------------------------------------------------------------------------- #

#: The V1 shape of the tables that V2 extends, exactly as V1 created them.
_V1_SCHEMA = """
CREATE TABLE topics (
  id INTEGER PRIMARY KEY, query_text TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE papers (
  paper_id TEXT PRIMARY KEY, version TEXT NOT NULL DEFAULT '',
  title TEXT NOT NULL, abstract TEXT NOT NULL, authors_json TEXT NOT NULL DEFAULT '[]',
  published TEXT DEFAULT '', updated TEXT DEFAULT '', primary_category TEXT DEFAULT '',
  categories_json TEXT NOT NULL DEFAULT '[]', comment TEXT DEFAULT '',
  journal_ref TEXT DEFAULT '', doi TEXT DEFAULT '',
  abs_url TEXT NOT NULL DEFAULT '', pdf_url TEXT DEFAULT '', fetched_at TEXT NOT NULL
);
CREATE TABLE landscapes (
  id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL REFERENCES topics(id),
  title TEXT NOT NULL, summary TEXT DEFAULT '', status TEXT NOT NULL DEFAULT 'running',
  params_json TEXT NOT NULL DEFAULT '{}', generation INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE landscape_papers (
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  paper_id TEXT NOT NULL REFERENCES papers(paper_id),
  rank INTEGER NOT NULL, relevance_score REAL NOT NULL,
  rerank_source TEXT NOT NULL DEFAULT 'cross-encoder', rationale TEXT DEFAULT '',
  is_seed INTEGER NOT NULL DEFAULT 0, cluster_id INTEGER, x REAL, y REAL,
  added_at TEXT NOT NULL, PRIMARY KEY (landscape_id, paper_id)
);
CREATE TABLE edges (
  id INTEGER PRIMARY KEY, landscape_id INTEGER NOT NULL,
  src_paper_id TEXT NOT NULL, dst_paper_id TEXT NOT NULL, kind TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 0.5, rationale TEXT DEFAULT ''
);
CREATE TABLE paper_extractions (
  paper_id TEXT NOT NULL, prompt_version TEXT NOT NULL, model TEXT NOT NULL DEFAULT '',
  problem TEXT, method TEXT, results TEXT, contribution TEXT, limitations TEXT,
  datasets_json TEXT NOT NULL DEFAULT '[]', metrics_json TEXT NOT NULL DEFAULT '[]',
  novelty TEXT DEFAULT 'unclear', evidence_json TEXT NOT NULL DEFAULT '{}',
  confidence REAL, created_at TEXT NOT NULL, PRIMARY KEY (paper_id, prompt_version)
);
CREATE TABLE runs (
  id TEXT PRIMARY KEY, landscape_id INTEGER, stage TEXT NOT NULL, status TEXT NOT NULL,
  message TEXT DEFAULT '', payload_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
"""


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}



def test_migrate_upgrades_a_v1_database_with_rows_intact(tmp_path: Path) -> None:
    db_path = tmp_path / "v1.db"
    conn = store.connect(db_path)
    conn.executescript(_V1_SCHEMA)
    conn.execute("INSERT INTO topics (id, query_text, created_at) VALUES (1, 'rag', '2025-01-01T00:00:00Z')")
    conn.execute(
        "INSERT INTO papers (paper_id, title, abstract, fetched_at) "
        "VALUES ('2107.05580', 'V1 paper', 'abs', '2025-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO landscapes (id, topic_id, title, created_at, updated_at) "
        "VALUES (1, 1, 'V1 landscape', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO edges (landscape_id, src_paper_id, dst_paper_id, kind) "
        "VALUES (1, 'a', 'b', 'extends')"
    )
    conn.execute(
        "INSERT INTO paper_extractions (paper_id, prompt_version, problem, created_at) "
        "VALUES ('2107.05580', 'extract_v1', 'v1 extraction', '2025-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO runs (id, landscape_id, stage, status, created_at) "
        "VALUES ('evt1', 1, 'retrieval', 'done', '2025-01-01T00:00:00Z')"
    )
    conn.commit()
    assert store.schema_version(conn) == 0

    store.migrate(conn)

    # New columns exist with the documented defaults; old rows survive.
    assert store.schema_version(conn) == store.SCHEMA_VERSION == 2
    assert {
        "citation_count", "citation_source", "openalex_id", "s2_paper_id", "fulltext_status",
    } <= _columns(conn, "papers")
    assert {"narrative_status", "cost_usd", "tokens_in", "tokens_out"} <= _columns(conn, "landscapes")
    assert {"source", "confidence"} <= _columns(conn, "edges")
    assert {"status", "error"} <= _columns(conn, "paper_extractions")
    assert {"run_id", "degraded"} <= _columns(conn, "runs")
    assert "cross_encoder_logit" in _columns(conn, "landscape_papers")

    paper = conn.execute("SELECT * FROM papers WHERE paper_id = '2107.05580'").fetchone()
    assert paper["title"] == "V1 paper"
    assert paper["citation_count"] is None  # unknown is not zero
    landscape = conn.execute("SELECT * FROM landscapes WHERE id = 1").fetchone()
    assert landscape["narrative_status"] == "pending"
    edge = conn.execute("SELECT * FROM edges WHERE id = 1").fetchone()
    assert edge["source"] == "llm"  # V1 edges were all LLM-typed
    extraction = conn.execute("SELECT * FROM paper_extractions").fetchone()
    assert extraction["status"] == "ok"
    assert conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"] == 1
    # New tables now exist.
    for table in ("citations", "source_cache", "llm_calls", "schema_meta"):
        assert _columns(conn, table), table
    conn.close()


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    store.init_db(db_path)  # second run must be a no-op
    conn = store.connect(db_path)
    try:
        store.migrate(conn)  # and a third, on a live connection
        assert store.schema_version(conn) == 2
    finally:
        conn.close()


def test_delete_landscape_cascades_but_papers_survive(conn) -> None:
    landscape_id = _seed_landscape(conn)
    store.replace_clusters(conn, landscape_id, [{"local_label": 0, "label": "x", "paper_count": 3}])
    store.replace_edges(
        conn, landscape_id,
        [{"src_paper_id": "2107.0000", "dst_paper_id": "2107.0001", "kind": "extends"}],
    )
    store.replace_tensions(
        conn, landscape_id,
        [{"statement": "s", "paper_a_id": "2107.0000", "paper_b_id": "2107.0001"}],
    )
    store.replace_open_problems(conn, landscape_id, [{"statement": "s"}])
    store.replace_reading_path(conn, landscape_id, [{"paper_id": "2107.0000", "position": 1}])
    store.insert_run(conn, {"run_id": "r1", "landscape_id": landscape_id, "stage": "layout", "status": "done"})
    store.insert_llm_call(conn, {"landscape_id": landscape_id, "stage": "synthesis"})

    assert store.delete_landscape(conn, landscape_id) is True

    assert store.landscape_paper_ids(conn, landscape_id) == set()
    for table in ("clusters", "edges", "tensions", "open_problems", "reading_path", "runs", "llm_calls"):
        count = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        assert count == 0, table
    # Papers are shared infrastructure: they must outlive any one landscape.
    assert store.count_papers(conn) == 3


def test_session_rolls_back_on_exception(settings) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with store.session(settings) as conn:
            store.upsert_topic(conn, "doomed topic")
            raise RuntimeError("boom")
    with store.session(settings) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM topics").fetchone()["n"] == 0


def test_session_commits_on_success(settings) -> None:
    with store.session(settings) as conn:
        topic_id = store.upsert_topic(conn, "committed topic")
    with store.session(settings) as conn:
        assert store.fetch_topic(conn, topic_id)["query_text"] == "committed topic"

