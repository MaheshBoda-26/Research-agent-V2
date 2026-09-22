"""SQLite persistence layer (schema v2).

Deliberately stdlib ``sqlite3`` and not an ORM: the schema is small, every
table is auditable, and there is nothing to invent. WAL mode is enabled so a
streaming pipeline run can keep writing stage events while the API reads a
previous landscape.

V2 keeps V1's 12 tables verbatim as the base and adds the plan §3.2
modifications:

* new columns: ``papers.citation_count/citation_source/openalex_id/
  s2_paper_id/fulltext_status`` (D8), ``paper_extractions.status/error`` (D3),
  ``landscapes.narrative_status/cost_usd/tokens_in/tokens_out`` (D1, D9),
  ``edges.source/confidence`` (D2);
* new tables: ``citations`` (raw links, so edges are recomputable for free),
  ``source_cache`` (TTL cache for S2/OpenAlex), ``llm_calls`` (full audit of
  every model call), and ``schema_meta`` (version bookkeeping);
* ``runs`` gains ``run_id`` and ``degraded`` for the SSE replay path.

Non-negotiables carried over from V1: ``papers.paper_id`` is the
**version-stripped** id; ``paper_extractions`` is keyed
``(paper_id, prompt_version)`` so prompt iteration never forces a re-embed;
embeddings are float32 BLOBs so a grown topic re-projects without re-embedding.

Every function takes an explicit connection. Callers use the ``session()``
context manager, which owns commit/rollback and closes the connection. Opening
a fresh connection per unit of work is intentional: it is cheap under WAL and
keeps connections from crossing thread boundaries when blocking work is pushed
through ``asyncio.to_thread``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT, Settings, load_settings
from models import UNCLUSTERED_LABEL, Paper, PaperExtraction, utcnow

SCHEMA_VERSION: int = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  version INTEGER NOT NULL,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topics (
  id INTEGER PRIMARY KEY,
  query_text TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_topics_query ON topics(lower(trim(query_text)));

CREATE TABLE IF NOT EXISTS papers (
  paper_id     TEXT PRIMARY KEY,
  version      TEXT NOT NULL DEFAULT '',
  title        TEXT NOT NULL,
  abstract     TEXT NOT NULL,
  authors_json TEXT NOT NULL DEFAULT '[]',
  published    TEXT DEFAULT '',
  updated      TEXT DEFAULT '',
  primary_category TEXT DEFAULT '',
  categories_json  TEXT NOT NULL DEFAULT '[]',
  comment TEXT DEFAULT '',
  journal_ref TEXT DEFAULT '',
  doi TEXT DEFAULT '',
  abs_url      TEXT NOT NULL DEFAULT '',
  pdf_url      TEXT DEFAULT '',
  citation_count INTEGER,                 -- NULL = unknown, NOT zero
  citation_source TEXT NOT NULL DEFAULT '',  -- semanticscholar | openalex | ''
  openalex_id  TEXT NOT NULL DEFAULT '',
  s2_paper_id  TEXT NOT NULL DEFAULT '',
  fulltext_status TEXT NOT NULL DEFAULT 'none',  -- none | ok | unavailable
  fetched_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_embeddings (
  paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
  model    TEXT NOT NULL,
  dim      INTEGER NOT NULL,
  vector   BLOB NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (paper_id, model)
);

CREATE TABLE IF NOT EXISTS landscapes (
  id INTEGER PRIMARY KEY,
  topic_id INTEGER NOT NULL REFERENCES topics(id),
  title TEXT NOT NULL,
  summary TEXT DEFAULT '',
  status TEXT NOT NULL DEFAULT 'running',
  narrative_status TEXT NOT NULL DEFAULT 'pending',
  cost_usd REAL NOT NULL DEFAULT 0,
  tokens_in INTEGER NOT NULL DEFAULT 0,
  tokens_out INTEGER NOT NULL DEFAULT 0,
  params_json TEXT NOT NULL DEFAULT '{}',
  generation INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS landscape_papers (
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  paper_id TEXT NOT NULL REFERENCES papers(paper_id),
  rank INTEGER NOT NULL,
  relevance_score REAL NOT NULL,
  cross_encoder_logit REAL,
  rerank_source TEXT NOT NULL DEFAULT 'cross-encoder',
  rationale TEXT DEFAULT '',
  is_seed INTEGER NOT NULL DEFAULT 0,
  cluster_id INTEGER,
  x REAL,
  y REAL,
  added_at TEXT NOT NULL,
  PRIMARY KEY (landscape_id, paper_id)
);
CREATE INDEX IF NOT EXISTS idx_landscape_papers_rank ON landscape_papers(landscape_id, rank);

CREATE TABLE IF NOT EXISTS clusters (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  local_label INTEGER NOT NULL,
  label TEXT DEFAULT '',
  description TEXT DEFAULT '',
  paper_count INTEGER NOT NULL DEFAULT 0,
  x REAL DEFAULT 0,
  y REAL DEFAULT 0,
  color TEXT DEFAULT '#94a3b8'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_clusters_unique ON clusters(landscape_id, local_label);

CREATE TABLE IF NOT EXISTS edges (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  src_paper_id TEXT NOT NULL,
  dst_paper_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 0.5,
  rationale TEXT DEFAULT '',
  source TEXT NOT NULL DEFAULT 'llm',       -- knn | citation | llm
  confidence REAL                           -- cosine for knn, 1.0 for citation
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique
  ON edges(landscape_id, src_paper_id, dst_paper_id, kind);

CREATE TABLE IF NOT EXISTS tensions (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  statement TEXT NOT NULL,
  paper_a_id TEXT DEFAULT '',
  paper_b_id TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS open_problems (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  statement TEXT NOT NULL,
  why_open TEXT DEFAULT '',
  supporting_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS reading_path (
  id INTEGER PRIMARY KEY,
  landscape_id INTEGER NOT NULL REFERENCES landscapes(id) ON DELETE CASCADE,
  paper_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  why TEXT DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_reading_path_unique ON reading_path(landscape_id, position);

CREATE TABLE IF NOT EXISTS paper_extractions (
  paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
  prompt_version TEXT NOT NULL,
  model TEXT NOT NULL DEFAULT '',
  problem TEXT,
  method TEXT,
  results TEXT,
  contribution TEXT,
  limitations TEXT,
  datasets_json TEXT NOT NULL DEFAULT '[]',
  metrics_json TEXT NOT NULL DEFAULT '[]',
  novelty TEXT DEFAULT 'unclear',
  evidence_json TEXT NOT NULL DEFAULT '{}',
  confidence REAL,
  status TEXT NOT NULL DEFAULT 'ok',        -- ok | failed (D3: never silent)
  error TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  PRIMARY KEY (paper_id, prompt_version)
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL DEFAULT '',
  landscape_id INTEGER REFERENCES landscapes(id) ON DELETE CASCADE,
  stage TEXT NOT NULL,
  status TEXT NOT NULL,
  message TEXT DEFAULT '',
  degraded INTEGER NOT NULL DEFAULT 0,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_landscape ON runs(landscape_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_run_id ON runs(run_id, created_at);

CREATE TABLE IF NOT EXISTS citations (      -- raw links, so edges are recomputable for free
  src_paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
  dst_paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
  source TEXT NOT NULL,                     -- semanticscholar | openalex
  fetched_at TEXT NOT NULL,
  PRIMARY KEY (src_paper_id, dst_paper_id, source)
);

CREATE TABLE IF NOT EXISTS source_cache (   -- TTL cache for S2/OpenAlex, mirroring the arXiv disk cache
  key TEXT PRIMARY KEY,                     -- e.g. 's2:paper:arXiv:2005.11401'
  source TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (      -- D9: full audit of every model call
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL DEFAULT '',
  landscape_id INTEGER REFERENCES landscapes(id) ON DELETE CASCADE,
  stage TEXT NOT NULL, model TEXT NOT NULL, provider TEXT NOT NULL,
  prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
  cost_usd REAL DEFAULT 0, latency_ms INTEGER DEFAULT 0,
  attempt INTEGER DEFAULT 1, ok INTEGER NOT NULL DEFAULT 1,
  error TEXT DEFAULT '', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_landscape ON llm_calls(landscape_id);
"""


class StoreError(RuntimeError):
    """Raised when a write cannot be completed."""


# --------------------------------------------------------------------------- #
# Connection handling
# --------------------------------------------------------------------------- #


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open a WAL-mode connection with row access by name."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


#: Columns added after V1's schema shipped. ``CREATE TABLE IF NOT EXISTS``
#: cannot add a column to an existing table, so additive changes are applied
#: explicitly here; a user's database outlives any single schema change.
#: Every NOT NULL addition carries a DEFAULT so ALTER never fails on old rows.
_ADDITIVE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("landscape_papers", "cross_encoder_logit", "REAL"),
    ("papers", "citation_count", "INTEGER"),
    ("papers", "citation_source", "TEXT NOT NULL DEFAULT ''"),
    ("papers", "openalex_id", "TEXT NOT NULL DEFAULT ''"),
    ("papers", "s2_paper_id", "TEXT NOT NULL DEFAULT ''"),
    ("papers", "fulltext_status", "TEXT NOT NULL DEFAULT 'none'"),
    ("landscapes", "narrative_status", "TEXT NOT NULL DEFAULT 'pending'"),
    ("landscapes", "cost_usd", "REAL NOT NULL DEFAULT 0"),
    ("landscapes", "tokens_in", "INTEGER NOT NULL DEFAULT 0"),
    ("landscapes", "tokens_out", "INTEGER NOT NULL DEFAULT 0"),
    ("edges", "source", "TEXT NOT NULL DEFAULT 'llm'"),
    ("edges", "confidence", "REAL"),
    ("paper_extractions", "status", "TEXT NOT NULL DEFAULT 'ok'"),
    ("paper_extractions", "error", "TEXT DEFAULT ''"),
    ("runs", "run_id", "TEXT NOT NULL DEFAULT ''"),
    ("runs", "degraded", "INTEGER NOT NULL DEFAULT 0"),
)


def migrate(conn: sqlite3.Connection) -> None:
    """Bring any older database to SCHEMA_VERSION. Idempotent by construction:
    every step is a CREATE IF NOT EXISTS or an ALTER guarded by PRAGMA
    table_info, so running it twice (or on a fresh database) is a no-op.
    """
    conn.executescript(SCHEMA)
    for table, column, column_type in _ADDITIVE_COLUMNS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:  # table itself missing; CREATE just handled it
            continue
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
    conn.execute(
        "INSERT INTO schema_meta (id, version, applied_at) VALUES (1, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET version = excluded.version, applied_at = excluded.applied_at",
        (SCHEMA_VERSION, utcnow()),
    )


def init_db(db_path: Path | str) -> None:
    """Apply the schema. Idempotent: safe to call on every startup."""
    conn = connect(db_path)
    try:
        migrate(conn)
        conn.commit()
    finally:
        conn.close()


def schema_version(conn: sqlite3.Connection) -> int:
    """The recorded schema version: 0 before any migrate(), 2 current."""
    try:
        row = conn.execute("SELECT version FROM schema_meta WHERE id = 1").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row["version"]) if row else 0


@contextmanager
def session(settings: Settings) -> Iterator[sqlite3.Connection]:
    """Unit-of-work boundary: commit on success, roll back on failure."""
    settings.ensure_dirs()
    conn = connect(settings.db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_load(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def _placeholders(count: int) -> str:
    return ",".join("?" * count)


def _iso_after(seconds: float) -> str:
    """An expiry timestamp in the same format as utcnow()."""
    moment = datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=seconds)
    return moment.isoformat().replace("+00:00", "Z")


def _iso_days_ago(days: int) -> str:
    moment = datetime.now(UTC).replace(microsecond=0) - timedelta(days=days)
    return moment.isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Topics
# --------------------------------------------------------------------------- #


def upsert_topic(conn: sqlite3.Connection, query_text: str) -> int:
    """Return the topic id for ``query_text``, creating it if needed.

    Matching is case- and whitespace-insensitive via the unique index, so
    "RAG" and "rag " resolve to the same topic and can accumulate papers.
    """
    normalized = " ".join(query_text.split())
    row = conn.execute(
        "SELECT id FROM topics WHERE lower(trim(query_text)) = lower(trim(?))",
        (normalized,),
    ).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO topics (query_text, created_at) VALUES (?, ?)",
        (normalized, utcnow()),
    )
    return int(cur.lastrowid)


def fetch_topic(conn: sqlite3.Connection, topic_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------- #
# Papers
# --------------------------------------------------------------------------- #


def upsert_papers(conn: sqlite3.Connection, papers: Sequence[Paper]) -> int:
    """Insert or refresh paper metadata. Returns the number of new rows.

    Metadata is refreshed on conflict because abstracts are occasionally
    revised upstream. Enrichment columns (citation_count, external ids,
    fulltext_status) are NOT refreshed by an empty incoming value: a retrieval
    re-run must never wipe what enrichment resolved (D8). ``COALESCE`` keeps
    the stored value whenever the incoming one is NULL/empty.
    """
    if not papers:
        return 0
    now = utcnow()
    existing = {
        row["paper_id"]
        for row in conn.execute(
            f"SELECT paper_id FROM papers WHERE paper_id IN ({_placeholders(len(papers))})",
            [p.paper_id for p in papers],
        )
    }
    conn.executemany(
        """
        INSERT INTO papers (
            paper_id, version, title, abstract, authors_json, published, updated,
            primary_category, categories_json, comment, journal_ref, doi,
            abs_url, pdf_url, citation_count, citation_source, openalex_id,
            s2_paper_id, fulltext_status, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            version = excluded.version,
            title = excluded.title,
            abstract = excluded.abstract,
            authors_json = excluded.authors_json,
            published = excluded.published,
            updated = excluded.updated,
            primary_category = excluded.primary_category,
            categories_json = excluded.categories_json,
            comment = excluded.comment,
            journal_ref = excluded.journal_ref,
            doi = excluded.doi,
            abs_url = excluded.abs_url,
            pdf_url = excluded.pdf_url,
            citation_count = COALESCE(excluded.citation_count, papers.citation_count),
            citation_source = COALESCE(NULLIF(excluded.citation_source, ''), papers.citation_source),
            openalex_id = COALESCE(NULLIF(excluded.openalex_id, ''), papers.openalex_id),
            s2_paper_id = COALESCE(NULLIF(excluded.s2_paper_id, ''), papers.s2_paper_id),
            fulltext_status = CASE WHEN excluded.fulltext_status != 'none'
                                   THEN excluded.fulltext_status
                                   ELSE papers.fulltext_status END,
            fetched_at = excluded.fetched_at
        """,
        [
            (
                p.paper_id, p.version, p.title, p.abstract, _json_dump(p.authors),
                p.published, p.updated, p.primary_category, _json_dump(p.categories),
                p.comment, p.journal_ref, p.doi, p.abs_link, p.pdf_url,
                p.citation_count, p.citation_source, p.openalex_id, p.s2_paper_id,
                p.fulltext_status, now,
            )
            for p in papers
        ],
    )
    return len([p for p in papers if p.paper_id not in existing])


def _row_to_paper(row: sqlite3.Row) -> Paper:
    return Paper(
        paper_id=row["paper_id"],
        version=row["version"] or "",
        title=row["title"],
        abstract=row["abstract"],
        authors=_json_load(row["authors_json"], []),
        published=row["published"] or "",
        updated=row["updated"] or "",
        primary_category=row["primary_category"] or "",
        categories=_json_load(row["categories_json"], []),
        comment=row["comment"] or "",
        journal_ref=row["journal_ref"] or "",
        doi=row["doi"] or "",
        abs_url=row["abs_url"] or "",
        pdf_url=row["pdf_url"] or "",
        citation_count=row["citation_count"],
        citation_source=row["citation_source"] or "",
        openalex_id=row["openalex_id"] or "",
        s2_paper_id=row["s2_paper_id"] or "",
        fulltext_status=row["fulltext_status"] or "none",
    )


def fetch_paper(conn: sqlite3.Connection, paper_id: str) -> Paper | None:
    row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return _row_to_paper(row) if row else None


def fetch_papers(conn: sqlite3.Connection, paper_ids: Sequence[str]) -> dict[str, Paper]:
    if not paper_ids:
        return {}
    rows = conn.execute(
        f"SELECT * FROM papers WHERE paper_id IN ({_placeholders(len(paper_ids))})",
        list(paper_ids),
    )
    return {row["paper_id"]: _row_to_paper(row) for row in rows}


def count_papers(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"])


# --------------------------------------------------------------------------- #
# Landscapes
# --------------------------------------------------------------------------- #


def insert_landscape(
    conn: sqlite3.Connection, *, topic_id: int, title: str, params: dict[str, Any] | None = None
) -> int:
    now = utcnow()
    cur = conn.execute(
        """
        INSERT INTO landscapes (topic_id, title, summary, status, narrative_status,
                                params_json, generation, created_at, updated_at)
        VALUES (?, ?, '', 'running', 'pending', ?, 1, ?, ?)
        """,
        (topic_id, title, _json_dump(params or {}), now, now),
    )
    return int(cur.lastrowid)


def fetch_landscape(conn: sqlite3.Connection, landscape_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT l.*, t.query_text AS topic
        FROM landscapes l JOIN topics t ON t.id = l.topic_id
        WHERE l.id = ?
        """,
        (landscape_id,),
    ).fetchone()
    return dict(row) if row else None


def find_landscape_for_topic(conn: sqlite3.Connection, topic_id: int) -> dict[str, Any] | None:
    """Most recent landscape for a topic, used by ``expand`` and by re-runs."""
    row = conn.execute(
        """
        SELECT l.*, t.query_text AS topic
        FROM landscapes l JOIN topics t ON t.id = l.topic_id
        WHERE l.topic_id = ?
        ORDER BY l.generation DESC, l.id DESC
        LIMIT 1
        """,
        (topic_id,),
    ).fetchone()
    return dict(row) if row else None


def list_landscapes(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT l.*, t.query_text AS topic,
               (SELECT COUNT(*) FROM landscape_papers lp WHERE lp.landscape_id = l.id) AS paper_count,
               (SELECT COUNT(*) FROM clusters c
                 WHERE c.landscape_id = l.id AND c.local_label != ?) AS cluster_count
        FROM landscapes l JOIN topics t ON t.id = l.topic_id
        ORDER BY l.updated_at DESC, l.id DESC
        """,
        (UNCLUSTERED_LABEL,),
    )
    return [dict(row) for row in rows]


#: Fields ``update_landscape`` may touch — anything else is a bug, not a no-op.
_LANDSCAPE_UPDATABLE = frozenset(
    {"title", "summary", "status", "narrative_status", "cost_usd", "tokens_in", "tokens_out"}
)


def update_landscape(conn: sqlite3.Connection, landscape_id: int, **fields: Any) -> None:
    unknown = set(fields) - _LANDSCAPE_UPDATABLE
    if unknown:
        raise StoreError(f"update_landscape: unknown fields {sorted(unknown)}")
    sets: list[str] = ["updated_at = ?"]
    params: list[Any] = [utcnow()]
    for name, value in fields.items():
        sets.append(f"{name} = ?")
        params.append(value)
    params.append(landscape_id)
    conn.execute(f"UPDATE landscapes SET {', '.join(sets)} WHERE id = ?", params)


def bump_generation(conn: sqlite3.Connection, landscape_id: int) -> int:
    conn.execute(
        "UPDATE landscapes SET generation = generation + 1, updated_at = ? WHERE id = ?",
        (utcnow(), landscape_id),
    )
    row = conn.execute("SELECT generation FROM landscapes WHERE id = ?", (landscape_id,)).fetchone()
    return int(row["generation"]) if row else 1


def delete_landscape(conn: sqlite3.Connection, landscape_id: int) -> bool:
    """Delete a landscape; children cascade, shared papers survive. True if one went."""
    cur = conn.execute("DELETE FROM landscapes WHERE id = ?", (landscape_id,))
    return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# Landscape membership and layout
# --------------------------------------------------------------------------- #


def link_paper(
    conn: sqlite3.Connection,
    landscape_id: int,
    *,
    paper_id: str,
    rank: int,
    relevance_score: float,
    rerank_source: str,
    rationale: str = "",
    is_seed: bool = False,
    cross_encoder_logit: float | None = None,
) -> None:
    """Attach a paper to a landscape, preserving any existing x/y position."""
    conn.execute(
        """
        INSERT INTO landscape_papers (landscape_id, paper_id, rank, relevance_score,
                                      cross_encoder_logit, rerank_source, rationale,
                                      is_seed, added_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(landscape_id, paper_id) DO UPDATE SET
            rank = excluded.rank,
            relevance_score = excluded.relevance_score,
            cross_encoder_logit = excluded.cross_encoder_logit,
            rerank_source = excluded.rerank_source,
            rationale = excluded.rationale,
            is_seed = excluded.is_seed
        """,
        (
            landscape_id, paper_id, rank, relevance_score, cross_encoder_logit,
            rerank_source, rationale, 1 if is_seed else 0, utcnow(),
        ),
    )


def fetch_landscape_papers(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT lp.rank, lp.relevance_score, lp.cross_encoder_logit, lp.rerank_source,
               lp.rationale, lp.is_seed, lp.cluster_id, lp.x, lp.y, lp.added_at,
               p.paper_id, p.version, p.title, p.abstract, p.authors_json,
               p.published, p.primary_category, p.categories_json, p.abs_url, p.pdf_url,
               p.citation_count, p.citation_source
        FROM landscape_papers lp JOIN papers p ON p.paper_id = lp.paper_id
        WHERE lp.landscape_id = ?
        ORDER BY lp.rank ASC, p.paper_id ASC
        """,
        (landscape_id,),
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["authors"] = _json_load(item.pop("authors_json"), [])
        item["categories"] = _json_load(item.pop("categories_json"), [])
        item["is_seed"] = bool(item["is_seed"])
        out.append(item)
    return out


def landscape_paper_ids(conn: sqlite3.Connection, landscape_id: int) -> set[str]:
    rows = conn.execute(
        "SELECT paper_id FROM landscape_papers WHERE landscape_id = ?", (landscape_id,)
    )
    return {row["paper_id"] for row in rows}


def update_layout(
    conn: sqlite3.Connection,
    landscape_id: int,
    layout: dict[str, tuple[float, float, int]],
) -> None:
    """Persist coordinates and cluster assignment: paper_id -> (x, y, cluster)."""
    if not layout:
        return
    conn.executemany(
        "UPDATE landscape_papers SET x = ?, y = ?, cluster_id = ? "
        "WHERE landscape_id = ? AND paper_id = ?",
        [(x, y, cluster, landscape_id, pid) for pid, (x, y, cluster) in layout.items()],
    )


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


def upsert_embedding(
    conn: sqlite3.Connection, paper_id: str, model: str, vector: bytes, dim: int
) -> None:
    """Store one raw float32 payload keyed by (paper_id, model)."""
    conn.execute(
        """
        INSERT INTO paper_embeddings (paper_id, model, dim, vector, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(paper_id, model) DO UPDATE SET
            dim = excluded.dim, vector = excluded.vector, created_at = excluded.created_at
        """,
        (paper_id, model, dim, vector, utcnow()),
    )


def upsert_embeddings(conn: sqlite3.Connection, model: str, vectors: dict[str, bytes]) -> None:
    """Bulk form: ``vectors`` maps paper_id -> BLOB bytes."""
    for paper_id, blob in vectors.items():
        upsert_embedding(conn, paper_id, model, blob, len(blob) // 4)


def fetch_embeddings(
    conn: sqlite3.Connection, paper_ids: Sequence[str], model: str
) -> dict[str, bytes]:
    if not paper_ids:
        return {}
    rows = conn.execute(
        f"SELECT paper_id, vector FROM paper_embeddings WHERE model = ? "
        f"AND paper_id IN ({_placeholders(len(paper_ids))}) ORDER BY paper_id",
        (model, *paper_ids),
    )
    return {row["paper_id"]: bytes(row["vector"]) for row in rows}


# --------------------------------------------------------------------------- #
# Extractions — keyed (paper_id, prompt_version) so prompt iteration never
# forces a re-embed or a re-layout.
# --------------------------------------------------------------------------- #


def upsert_extraction(
    conn: sqlite3.Connection,
    paper_id: str,
    prompt_version: str,
    extraction: PaperExtraction,
    *,
    model: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO paper_extractions (
            paper_id, prompt_version, model, problem, method, results, contribution,
            limitations, novelty, evidence_json, status, error, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id, prompt_version) DO UPDATE SET
            model = excluded.model,
            problem = excluded.problem,
            method = excluded.method,
            results = excluded.results,
            contribution = excluded.contribution,
            limitations = excluded.limitations,
            novelty = excluded.novelty,
            evidence_json = excluded.evidence_json,
            status = excluded.status,
            error = excluded.error,
            created_at = excluded.created_at
        """,
        (
            paper_id, prompt_version, model,
            extraction.problem, extraction.method, extraction.results,
            extraction.contribution, extraction.limitations, extraction.novelty,
            _json_dump(extraction.evidence), extraction.status, extraction.error,
            utcnow(),
        ),
    )


def _row_to_extraction(row: sqlite3.Row) -> PaperExtraction:
    return PaperExtraction(
        problem=row["problem"],
        method=row["method"],
        results=row["results"],
        contribution=row["contribution"],
        limitations=row["limitations"],
        novelty=row["novelty"] or "unclear",
        evidence=_json_load(row["evidence_json"], {}),
        status=row["status"] or "ok",
        error=row["error"] or "",
    )


def fetch_extraction(
    conn: sqlite3.Connection, paper_id: str, prompt_version: str
) -> PaperExtraction | None:
    row = conn.execute(
        "SELECT * FROM paper_extractions WHERE paper_id = ? AND prompt_version = ?",
        (paper_id, prompt_version),
    ).fetchone()
    return _row_to_extraction(row) if row else None


def fetch_extractions(
    conn: sqlite3.Connection, paper_ids: Sequence[str], prompt_version: str
) -> dict[str, PaperExtraction]:
    """Only rows for THIS prompt version — a bumped prompt deliberately misses."""
    if not paper_ids:
        return {}
    rows = conn.execute(
        f"SELECT * FROM paper_extractions WHERE prompt_version = ? "
        f"AND paper_id IN ({_placeholders(len(paper_ids))})",
        (prompt_version, *paper_ids),
    )
    return {row["paper_id"]: _row_to_extraction(row) for row in rows}


# --------------------------------------------------------------------------- #
# Clusters
# --------------------------------------------------------------------------- #


def replace_clusters(
    conn: sqlite3.Connection, landscape_id: int, clusters: Sequence[dict[str, Any]]
) -> None:
    conn.execute("DELETE FROM clusters WHERE landscape_id = ?", (landscape_id,))
    if not clusters:
        return
    conn.executemany(
        """
        INSERT INTO clusters (landscape_id, local_label, label, description,
                              paper_count, x, y, color)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                landscape_id,
                int(c["local_label"]),
                c.get("label", ""),
                c.get("description", ""),
                int(c.get("paper_count", 0)),
                float(c.get("x", 0.0)),
                float(c.get("y", 0.0)),
                c.get("color", "#94a3b8"),
            )
            for c in clusters
        ],
    )


def fetch_clusters(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM clusters WHERE landscape_id = ? ORDER BY local_label ASC",
        (landscape_id,),
    )
    return [dict(row) for row in rows]


def cluster_id_by_label(conn: sqlite3.Connection, landscape_id: int) -> dict[int, int]:
    """Map HDBSCAN's local label (-1, 0, 1, ...) to the clusters table row id."""
    rows = conn.execute(
        "SELECT id, local_label FROM clusters WHERE landscape_id = ?", (landscape_id,)
    )
    return {int(row["local_label"]): int(row["id"]) for row in rows}


def replace_cluster_labels(
    conn: sqlite3.Connection, landscape_id: int, labels: Sequence[Any]
) -> None:
    """Apply LLM labels onto rows already written by the layout stage."""
    for label in labels:
        conn.execute(
            "UPDATE clusters SET label = ?, description = ? WHERE landscape_id = ? AND local_label = ?",
            (label.label, label.description, landscape_id, int(label.local_label)),
        )


# --------------------------------------------------------------------------- #
# Edges, tensions, open problems, reading path
# --------------------------------------------------------------------------- #


def replace_edges(
    conn: sqlite3.Connection, landscape_id: int, edges: Sequence[dict[str, Any]]
) -> None:
    """Replace the landscape's edge set. Each edge dict: src_paper_id,
    dst_paper_id, kind, weight (0.5), rationale (''), source ('llm'),
    confidence (None). ``source`` records provenance (D2): knn | citation | llm.
    """
    conn.execute("DELETE FROM edges WHERE landscape_id = ?", (landscape_id,))
    if not edges:
        return
    conn.executemany(
        """
        INSERT OR IGNORE INTO edges (landscape_id, src_paper_id, dst_paper_id, kind,
                                     weight, rationale, source, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                landscape_id,
                e["src_paper_id"],
                e["dst_paper_id"],
                e["kind"],
                float(e.get("weight", 0.5)),
                e.get("rationale", ""),
                e.get("source", "llm"),
                e.get("confidence"),
            )
            for e in edges
        ],
    )


def fetch_edges(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT src_paper_id, dst_paper_id, kind, weight, rationale, source, confidence "
        "FROM edges WHERE landscape_id = ? ORDER BY id ASC",
        (landscape_id,),
    )
    return [dict(row) for row in rows]


def replace_tensions(
    conn: sqlite3.Connection, landscape_id: int, tensions: Sequence[dict[str, Any]]
) -> None:
    conn.execute("DELETE FROM tensions WHERE landscape_id = ?", (landscape_id,))
    if not tensions:
        return
    conn.executemany(
        "INSERT INTO tensions (landscape_id, statement, paper_a_id, paper_b_id) VALUES (?, ?, ?, ?)",
        [
            (landscape_id, t["statement"], t.get("paper_a_id", ""), t.get("paper_b_id", ""))
            for t in tensions
        ],
    )


def fetch_tensions(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT statement, paper_a_id, paper_b_id FROM tensions WHERE landscape_id = ? ORDER BY id ASC",
        (landscape_id,),
    )
    return [dict(row) for row in rows]


def replace_open_problems(
    conn: sqlite3.Connection, landscape_id: int, problems: Sequence[dict[str, Any]]
) -> None:
    conn.execute("DELETE FROM open_problems WHERE landscape_id = ?", (landscape_id,))
    if not problems:
        return
    conn.executemany(
        "INSERT INTO open_problems (landscape_id, statement, why_open, supporting_json) "
        "VALUES (?, ?, ?, ?)",
        [
            (
                landscape_id,
                p["statement"],
                p.get("why_open", ""),
                _json_dump(p.get("supporting_paper_ids", [])),
            )
            for p in problems
        ],
    )


def fetch_open_problems(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT statement, why_open, supporting_json FROM open_problems "
        "WHERE landscape_id = ? ORDER BY id ASC",
        (landscape_id,),
    )
    out = []
    for row in rows:
        item = dict(row)
        item["supporting_paper_ids"] = _json_load(item.pop("supporting_json"), [])
        out.append(item)
    return out


def replace_reading_path(
    conn: sqlite3.Connection, landscape_id: int, steps: Sequence[dict[str, Any]]
) -> None:
    conn.execute("DELETE FROM reading_path WHERE landscape_id = ?", (landscape_id,))
    if not steps:
        return
    conn.executemany(
        "INSERT INTO reading_path (landscape_id, paper_id, position, why) VALUES (?, ?, ?, ?)",
        [(landscape_id, s["paper_id"], int(s["position"]), s.get("why", "")) for s in steps],
    )


def fetch_reading_path(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT rp.paper_id, rp.position, rp.why, p.title
        FROM reading_path rp LEFT JOIN papers p ON p.paper_id = rp.paper_id
        WHERE rp.landscape_id = ?
        ORDER BY rp.position ASC
        """,
        (landscape_id,),
    )
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------- #
# Citations (raw links, so edges are recomputable for free)
# --------------------------------------------------------------------------- #


def replace_citations(
    conn: sqlite3.Connection, src: str, pairs: Sequence[tuple[str, str]], source: str
) -> None:
    """Replace one source's outgoing links for one paper.

    Each pair is ``(src, dst)``; ``src`` pins the row set being replaced.
    Cross-corpus filtering is the graph layer's job (A.6 citation_edges).
    """
    conn.execute("DELETE FROM citations WHERE src_paper_id = ? AND source = ?", (src, source))
    if not pairs:
        return
    now = utcnow()
    conn.executemany(
        "INSERT OR IGNORE INTO citations (src_paper_id, dst_paper_id, source, fetched_at) "
        "VALUES (?, ?, ?, ?)",
        [(s, d, source, now) for s, d in pairs],
    )


def fetch_citations(
    conn: sqlite3.Connection, paper_ids: Sequence[str]
) -> list[tuple[str, str, str]]:
    """All known citation links touching any of these papers: (src, dst, source)."""
    if not paper_ids:
        return []
    marks = _placeholders(len(paper_ids))
    rows = conn.execute(
        f"SELECT src_paper_id, dst_paper_id, source FROM citations "
        f"WHERE src_paper_id IN ({marks}) OR dst_paper_id IN ({marks})",
        (*paper_ids, *paper_ids),
    )
    return [(row["src_paper_id"], row["dst_paper_id"], row["source"]) for row in rows]


# --------------------------------------------------------------------------- #
# Source cache (TTL) for S2 / OpenAlex
# --------------------------------------------------------------------------- #


def get_cached(conn: sqlite3.Connection, key: str, source: str) -> dict[str, Any] | None:
    """The cached payload if present and unexpired, else None.

    Expired rows are left for ``prune`` — this cache is TTL-strict (unlike the
    arXiv disk cache, which is stale-readable during outages).
    """
    row = conn.execute(
        "SELECT payload_json, expires_at FROM source_cache WHERE key = ? AND source = ?",
        (key, source),
    ).fetchone()
    if not row or row["expires_at"] <= utcnow():
        return None
    return _json_load(row["payload_json"], None)


def put_cached(
    conn: sqlite3.Connection, key: str, source: str, payload: dict[str, Any], ttl_seconds: int
) -> None:
    now = utcnow()
    conn.execute(
        """
        INSERT INTO source_cache (key, source, payload_json, fetched_at, expires_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            source = excluded.source, payload_json = excluded.payload_json,
            fetched_at = excluded.fetched_at, expires_at = excluded.expires_at
        """,
        (key, source, _json_dump(payload), now, _iso_after(ttl_seconds)),
    )


# --------------------------------------------------------------------------- #
# Runs (SSE replay) and llm_calls (per-call audit, D9)
# --------------------------------------------------------------------------- #


def insert_run(conn: sqlite3.Connection, event: dict[str, Any]) -> None:
    """Record a stage event so a client that reconnects can replay progress."""
    conn.execute(
        """
        INSERT OR REPLACE INTO runs (id, run_id, landscape_id, stage, status, message,
                                     degraded, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.get("id") or uuid.uuid4().hex,
            event.get("run_id", ""),
            event.get("landscape_id"),
            event["stage"],
            event["status"],
            event.get("message", ""),
            1 if event.get("degraded") else 0,
            _json_dump(event.get("payload", {})),
            event.get("created_at") or event.get("ts") or utcnow(),
        ),
    )


def fetch_runs(conn: sqlite3.Connection, landscape_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM runs WHERE landscape_id = ? ORDER BY created_at ASC, id ASC",
        (landscape_id,),
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["payload"] = _json_load(item.get("payload_json"), {})
        item["degraded"] = bool(item["degraded"])
        out.append(item)
    return out


def insert_llm_call(conn: sqlite3.Connection, call: dict[str, Any]) -> None:
    """One row per completion attempt — the D9 audit trail (§13.1 llm.call)."""
    conn.execute(
        """
        INSERT INTO llm_calls (run_id, landscape_id, stage, model, provider,
                               prompt_tokens, completion_tokens, cost_usd, latency_ms,
                               attempt, ok, error, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            call.get("run_id", ""),
            call.get("landscape_id"),
            call.get("stage", ""),
            call.get("model", ""),
            call.get("provider", ""),
            int(call.get("prompt_tokens", 0)),
            int(call.get("completion_tokens", 0)),
            float(call.get("cost_usd", 0.0)),
            int(call.get("latency_ms", 0)),
            int(call.get("attempt", 1)),
            1 if call.get("ok", True) else 0,
            call.get("error") or call.get("validation_error") or "",
            call.get("created_at") or utcnow(),
        ),
    )


def llm_cost_rollup(conn: sqlite3.Connection, *, days: int = 30) -> list[dict[str, Any]]:
    """Spend by day, stage and model — the `make cost` table (§13.7)."""
    rows = conn.execute(
        """
        SELECT substr(created_at, 1, 10) AS day, stage, model,
               COUNT(*) AS calls,
               SUM(prompt_tokens) AS prompt_tokens,
               SUM(completion_tokens) AS completion_tokens,
               SUM(cost_usd) AS cost_usd
        FROM llm_calls
        WHERE created_at >= ?
        GROUP BY day, stage, model
        ORDER BY day DESC, stage ASC, model ASC
        """,
        (_iso_days_ago(days),),
    )
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------- #
# Data lifecycle (§13.6)
# --------------------------------------------------------------------------- #


def prune(conn: sqlite3.Connection, *, older_than_days: int) -> dict[str, int]:
    """Drop expired cache rows and run events older than the retention window.

    Returns per-table counts of deleted rows. The caller (scripts/prune.py,
    Phase 8) owns the follow-up VACUUM, which cannot run inside a transaction.
    """
    cutoff = _iso_days_ago(older_than_days)
    expired = conn.execute("DELETE FROM source_cache WHERE expires_at < ?", (utcnow(),)).rowcount
    old_runs = conn.execute("DELETE FROM runs WHERE created_at < ?", (cutoff,)).rowcount
    return {"source_cache": int(expired), "runs": int(old_runs)}


def default_settings() -> Settings:
    """Convenience for scripts that only need paths, not credentials."""
    settings = Settings.from_env()
    settings.ensure_dirs()
    return settings

