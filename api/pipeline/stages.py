"""Pipeline orchestration as an async generator of stage events (V2).

One implementation drives both the SSE endpoint and the CLI. The generator yields
a ``StageEvent`` per transition; the caller decides whether to serialise those as
server-sent events or print them.

Four details are load-bearing:

**Blocking work is pushed to threads.** ``arxiv.Client`` wraps
``requests.Session``, cross-encoder inference is CPU-bound, and UMAP is
single-threaded numba. Called directly from the event loop, any of them stalls
the stream, and the symptom is a UI where all five stages appear at once at the
end instead of completing one by one.

**Connections do not cross threads.** A sqlite connection is opened and used
within a single thread. Workers open their own session inside the worker
function.

**Workers write to shared state rather than returning values.** An async
generator cannot ``return`` a value, so every worker assigns its result onto
``PipelineRun.state``. Because the generator awaits each future before reading
that state, there is no race.

**Intra-stage progress is real.** Extraction over 60 papers takes minutes, so a
progress callback pushes counts back to the event loop through a
``call_soon_threadsafe`` queue, and the generator yields them as the work runs.

Note on identifiers: ``state.local_labels`` maps ``paper_id`` to HDBSCAN's
**local** label (``-1`` for noise, ``0, 1, ...`` for clusters), not to the
``clusters`` table row id. Synthesis depends on that distinction to avoid naming
the unclustered bucket as if it were a research area.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from config import Settings
from llm.protocol import JSONCompleter
from models import (
    STAGE_ORDER,
    UNCLUSTERED_LABEL,
    UNCLUSTERED_NAME,
    Paper,
    PaperExtraction,
    RankedPaper,
    StageEvent,
    StageName,
    StageProgress,
)
from pipeline import cluster as cluster_mod
from pipeline import embed as embed_mod
from pipeline import enrich as enrich_mod
from pipeline import extract as extract_mod
from pipeline import graph as graph_mod
from pipeline import rerank as rerank_mod
from pipeline import retrieve as retrieve_mod
from pipeline import synthesize as synth_mod
from pipeline.cluster import (
    ReducerLike,
    cluster_centroids,
    color_for_label,
    group_labels,
    layout,
    name_clusters,
    normalize_coords,
)
from pipeline.embed import EmbedderLike, embed_papers, embedding_matrix
from store import cluster_id_by_label, landscape_paper_ids, replace_clusters, update_layout
from store import session as db_session

logger = logging.getLogger(__name__)

#: How often the generator checks for progress from a worker thread.
PROGRESS_POLL_SECONDS = 0.25

STAGE_LABELS: dict[StageName, str] = {
    "retrieval": "Searching arXiv",
    "enrichment": "Gathering citations",
    "rerank": "Ranking by relevance",
    "extraction": "Reading each paper",
    "layout": "Mapping the field",
    "synthesis": "Finding tensions and open problems",
}

ProgressCallback = Callable[[int, int], None]


@dataclass
class PipelineState:
    """Everything the stages hand to each other."""

    topic: str
    run_id: str
    landscape_id: int | None = None
    papers: list[Paper] = field(default_factory=list)
    ranked: list[RankedPaper] = field(default_factory=list)
    extractions: dict[str, PaperExtraction] = field(default_factory=dict)
    #: paper_id -> HDBSCAN local label (-1 means unclustered).
    local_labels: dict[str, int] = field(default_factory=dict)
    new_paper_count: int = 0
    coverage: Any = None
    narrative_status: str = "pending"
    degraded: bool = False
    citations: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def paper_ids(self) -> list[str]:
        return [paper.paper_id for paper in self.papers]

    @property
    def cluster_count(self) -> int:
        return len({label for label in self.local_labels.values() if label != UNCLUSTERED_LABEL})

    @property
    def unclustered_count(self) -> int:
        return sum(1 for label in self.local_labels.values() if label == UNCLUSTERED_LABEL)


class PipelineRun:
    """Runs one landscape build, yielding an event per stage transition."""

    def __init__(
        self,
        topic: str,
        settings: Settings,
        *,
        completer: JSONCompleter | None = None,
        expand_landscape_id: int | None = None,
        run_id: str | None = None,
        arxiv_client: Any | None = None,
        scorer: Any | None = None,
        embedder: Any | None = None,
        reducer: Any | None = None,
        clusterer: Any | None = None,
        s2: Any | None = None,
        openalex: Any | None = None,
    ) -> None:
        self.settings = settings
        self.completer = completer
        self.expand_landscape_id = expand_landscape_id
        self.state = PipelineState(topic=topic, run_id=run_id or f"run_{uuid4().hex[:12]}")
        self._arxiv_client = arxiv_client
        self._scorer = scorer
        self._embedder = embedder
        self._reducer = reducer
        self._clusterer = clusterer
        self._s2 = s2
        self._openalex = openalex
        self._llm_sink: list[dict[str, Any]] = []
        if completer is not None and hasattr(completer, "on_call"):
            try:
                completer.on_call = self._record_llm_call  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - a completer may be frozen
                logger.debug("Could not attach llm_calls sink")

    # -- event plumbing ----------------------------------------------------- #

    def _event(
        self,
        stage: StageName,
        status: str,
        message: str = "",
        *,
        progress: StageProgress | None = None,
        payload: dict[str, Any] | None = None,
        degraded: bool | None = None,
    ) -> StageEvent:
        return StageEvent(
            run_id=self.state.run_id,
            landscape_id=self.state.landscape_id,
            stage=stage,
            status=status,  # type: ignore[arg-type]
            message=message or STAGE_LABELS.get(stage, ""),
            progress=progress or StageProgress(),
            payload=payload or {},
            degraded=self.state.degraded if degraded is None else degraded,
        )

    def _record_llm_call(self, **fields: Any) -> None:
        self._llm_sink.append(dict(fields))

    def _flush_llm_calls(self) -> None:
        if not self._llm_sink or self.state.landscape_id is None:
            return
        try:
            with db_session(self.settings) as conn:
                from store import insert_llm_call
                for call in self._llm_sink:
                    insert_llm_call(conn, {**call, "run_id": self.state.run_id, "landscape_id": self.state.landscape_id})
            self._llm_sink.clear()
        except Exception:  # noqa: BLE001 - observability must never break a run
            logger.warning("Could not flush llm_calls buffer")

    def _skip_or_degrade(self, stage: StageName, exc: Exception) -> StageEvent | None:
        if stage in ("enrichment", "rerank", "extraction", "synthesis"):
            self.state.degraded = True
            logger.warning("%s degraded (%s)", stage, exc)
            return self._event(stage, "done", f"{stage} degraded: {exc}", payload={"degraded": True, "error": str(exc)})
        return None

    async def _run_blocking(
        self,
        stage: StageName,
        message: str,
        work: Callable[[ProgressCallback], None],
    ) -> AsyncIterator[StageEvent]:
        """Run blocking ``work`` in a thread, yielding live progress events.

        Yields nothing itself on success; the caller emits the stage's ``done``
        event. Any exception from ``work`` propagates to ``run``.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[int, int]] = asyncio.Queue()

        def report(current: int, total: int) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (current, total))

        future = loop.run_in_executor(None, work, report)
        try:
            while not future.done():
                try:
                    current, total = await asyncio.wait_for(
                        queue.get(), timeout=PROGRESS_POLL_SECONDS
                    )
                except TimeoutError:
                    continue
                yield self._event(
                    stage,
                    "running",
                    message,
                    progress=StageProgress(current=current, total=total),
                )
                # Let the transport flush between ticks.
                await asyncio.sleep(0)

            while not queue.empty():
                current, total = queue.get_nowait()
                yield self._event(
                    stage,
                    "running",
                    message,
                    progress=StageProgress(current=current, total=total),
                )

            await future  # re-raise anything the worker threw
        except BaseException:
            future.cancel()
            raise

    # -- stage 1: retrieval ------------------------------------------------- #

    async def _stage_retrieval(self) -> AsyncIterator[StageEvent]:
        yield self._event("retrieval", "running", f"Searching arXiv for {self.state.topic!r}")

        # ``completer`` is keyword-only on fetch_candidates; passing it
        # positionally would land on ``client`` and inject an LLM as an arXiv
        # client. Use a closure rather than to_thread's positional forwarding.
        _client_kw: dict[str, Any] = {}
        if self._arxiv_client is not None:
            _client_kw["client"] = self._arxiv_client
        papers = await asyncio.to_thread(
            lambda: retrieve_mod.fetch_candidates(
                self.state.topic, self.settings, completer=self.completer, **_client_kw
            )
        )
        if not papers:
            raise retrieve_mod.RetrievalError(
                f"arXiv returned no papers for {self.state.topic!r}. Try a broader topic."
            )

        default_title = self.state.topic

        def persist() -> tuple[int, int]:
            with db_session(self.settings) as conn:
                topic_id = _upsert_topic(conn, self.state.topic)
                if self.expand_landscape_id is None:
                    landscape_id = _insert_landscape(
                        conn, topic_id=topic_id, title=default_title, params=_params(self.settings)
                    )
                else:
                    landscape_id = self.expand_landscape_id
                    _bump_generation(conn, landscape_id)
                    _update_landscape(conn, landscape_id, status="running")
                added = _upsert_papers(conn, papers)
                return landscape_id, added

        landscape_id, added = await asyncio.to_thread(persist)
        self.state.landscape_id = landscape_id
        self.state.new_paper_count = added
        await asyncio.to_thread(self._flush_llm_calls)

        if self.expand_landscape_id is not None:
            # Re-rank the whole corpus, not just the newcomers, so the ordering
            # of an expanded landscape stays globally correct.
            existing = await asyncio.to_thread(self._load_existing_papers)
            merged: dict[str, Paper] = {paper.paper_id: paper for paper in existing}
            for paper in papers:
                merged.setdefault(paper.paper_id, paper)
            papers = list(merged.values())
            logger.info(
                "Expanding landscape %s: %d existing + %d new = %d papers",
                landscape_id,
                len(existing),
                len(merged) - len(existing),
                len(merged),
            )

        self.state.papers = papers
        yield self._event(
            "retrieval",
            "done",
            f"Retrieved {len(papers)} papers",
            payload={"count": len(papers), "new": added, "degraded": self.state.degraded},
        )

    def _load_existing_papers(self) -> list[Paper]:
        with db_session(self.settings) as conn:
            ids = _landscape_paper_ids(conn, self.state.landscape_id)
            papers = [_fetch_paper(conn, paper_id) for paper_id in ids]
        return [paper for paper in papers if paper is not None]

    # -- stage 1b: enrichment (never fatal) ----------------------------------- #

    async def _stage_enrichment(self) -> AsyncIterator[StageEvent]:
        total = len(self.state.papers)
        yield self._event("enrichment", "running", f"Gathering citations for {total} papers",
                          progress=StageProgress(current=0, total=total))
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        def on_progress(event: dict) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, dict(event))
            except Exception:  # noqa: BLE001 - progress must never break enrichment
                pass
        try:
            enriched, triples = await enrich_mod.enrich_papers(
                list(self.state.papers), self.settings, s2=self._s2, openalex=self._openalex, progress=on_progress)
        except Exception as exc:  # noqa: BLE001 - enrichment never fatal
            yield self._skip_or_degrade("enrichment", exc)
            return
        try:
            while True:
                event = queue.get_nowait()
                yield self._event("enrichment", "running",
                                  f"Gathered {event.get('done', 0)}/{event.get('total', total)}",
                                  progress=StageProgress(current=int(event.get("done", 0)), total=int(event.get("total", total))))
        except asyncio.QueueEmpty:
            pass
        self.state.papers = list(enriched)
        self.state.citations = list(triples)
        try:
            def persist_enrich() -> None:
                with db_session(self.settings) as conn:
                    from store import replace_citations, upsert_papers
                    upsert_papers(conn, self.state.papers)
                    by_src: dict[str, list[tuple[str, str]]] = {}
                    for src, dst, _so in triples:
                        by_src.setdefault(src, []).append((src, dst))
                    for src, pairs in by_src.items():
                        source = next((so for s2, _d, so in triples if s2 == src), "semanticscholar")
                        replace_citations(conn, src, pairs, source)
            await asyncio.to_thread(persist_enrich)
        except Exception:  # noqa: BLE001 - persistence failure degrades, not fatal
            logger.warning("Could not persist enrichment results")
        await asyncio.to_thread(self._flush_llm_calls)
        yield self._event("enrichment", "done", f"Gathered citations for {len(self.state.papers)} papers",
                          progress=StageProgress(current=total, total=total),
                          payload={"count": len(self.state.papers), "citation_links": len(self.state.citations), "degraded": self.state.degraded})

    # -- stage 2: rerank ---------------------------------------------------- #

    async def _stage_rerank(self) -> AsyncIterator[StageEvent]:
        total = len(self.state.papers)
        yield self._event(
            "rerank", "running", f"Scoring {total} candidates", progress=StageProgress(current=0, total=total)
        )

        settings = self.settings
        topic = self.state.topic
        papers = self.state.papers
        completer = self.completer

        def work(report: ProgressCallback) -> None:
            report(0, total)
            ranked = rerank_mod.rank_papers(topic, papers, settings, scorer=self._scorer, judge=completer)
            self.state.ranked = rerank_mod.select_top(
                ranked, settings.rerank_final_count_capped
            )
            report(total, total)

        async for event in self._run_blocking("rerank", "Scoring candidates", work):
            yield event

        selected = self.state.ranked
        landscape_id = self.state.landscape_id

        def persist() -> None:
            with db_session(settings) as conn:
                for item in selected:
                    _link_paper(
                        conn,
                        landscape_id,
                        paper_id=item.paper.paper_id,
                        rank=item.rank,
                        relevance_score=item.relevance_score,
                        cross_encoder_logit=item.cross_encoder_logit,
                        rerank_source=item.rerank_source,
                        rationale=item.rationale,
                    )

        await asyncio.to_thread(persist)

        sources: dict[str, int] = {}
        for item in selected:
            sources[item.rerank_source] = sources.get(item.rerank_source, 0) + 1
        yield self._event(
            "rerank",
            "done",
            f"Selected the top {len(selected)} papers",
            payload={"selected": len(selected), "sources": sources},
        )

    # -- stage 3: extraction ------------------------------------------------ #

    async def _stage_extraction(self) -> AsyncIterator[StageEvent]:
        papers = [item.paper for item in self.state.ranked]
        total = len(papers)
        yield self._event(
            "extraction",
            "running",
            f"Reading {total} abstracts",
            progress=StageProgress(current=0, total=total),
        )

        settings = self.settings
        completer = self.completer
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[int, int]] = asyncio.Queue()

        def on_progress(event: dict) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, (int(event.get("done", 0)), int(event.get("total", total))))
            except Exception:  # noqa: BLE001 - progress must never break extraction
                pass

        task = asyncio.create_task(extract_mod.extract_all(papers, settings, completer=completer, progress=on_progress))
        while not task.done():
            try:
                while True:
                    current, total_n = queue.get_nowait()
                    yield self._event("extraction", "running", "Reading abstracts",
                                      progress=StageProgress(current=current, total=total_n))
            except asyncio.QueueEmpty:
                pass
            await asyncio.sleep(PROGRESS_POLL_SECONDS)
        extractions, coverage = task.result()
        self.state.extractions = extractions
        self.state.coverage = coverage
        try:
            while True:
                current, total_n = queue.get_nowait()
                yield self._event("extraction", "running", "Reading abstracts",
                                  progress=StageProgress(current=current, total=total_n))
        except asyncio.QueueEmpty:
            pass

        grounded = sum(
            1 for item in self.state.extractions.values() if extract_mod.has_grounded_content(item)
        )
        if self.completer is None:
            message = f"No LLM configured; skipped reading {total} papers"
        else:
            message = f"Extracted structure from {grounded} of {total} papers"
        yield self._event(
            "extraction",
            "done",
            message,
            payload={"extracted": grounded, "total": total, "llm": self.completer is not None, "degraded": self.state.degraded, "coverage": getattr(self.state.coverage, "ratio", None)},
        )

    # -- stage 4: layout ---------------------------------------------------- #

    async def _stage_layout(self) -> AsyncIterator[StageEvent]:
        papers = [item.paper for item in self.state.ranked]
        total = len(papers)
        yield self._event(
            "layout",
            "running",
            f"Projecting {total} papers",
            progress=StageProgress(current=0, total=total),
        )

        settings = self.settings
        landscape_id = self.state.landscape_id

        def work(report: ProgressCallback) -> None:
            from store import cluster_id_by_label, replace_clusters, update_layout

            report(0, total)
            with db_session(settings) as conn:
                vectors = embed_mod.embed_papers(papers, settings, conn, embedder=self._embedder)
            report(total // 3, total)

            matrix = embed_mod.embedding_matrix([p.paper_id for p in papers], vectors)
            result = cluster_mod.layout(matrix, settings, reducer=self._reducer, clusterer=self._clusterer)
            coords = cluster_mod.normalize_coords(result.coords)
            report((total * 2) // 3, total)

            centroids = cluster_mod.cluster_centroids(coords, result.labels)
            groups = cluster_mod.group_labels(result.labels)
            named = {
                entry["local_label"]: entry
                for entry in cluster_mod.name_clusters(
                    [
                        {
                            "local_label": int(label),
                            "size": len(indices),
                            "topic": self.state.topic,
                            "paper_ids": [papers[i].paper_id for i in indices],
                        }
                        for label, indices in sorted(groups.items())
                    ],
                    papers,
                    settings,
                    completer=self.completer,
                )
            }
            rows = [
                {
                    "local_label": label,
                    "paper_count": len(indices),
                    "x": centroids[label][0],
                    "y": centroids[label][1],
                    "color": cluster_mod.color_for_label(label),
                    "label": (
                        "Unclustered"
                        if label == UNCLUSTERED_LABEL
                        else str(named.get(int(label), {}).get("label") or f"Cluster {label}")
                    ),
                    "description": ("Below the clustering threshold." if label == UNCLUSTERED_LABEL else str(named.get(int(label), {}).get("description") or "")),
                }
                for label, indices in sorted(groups.items())
            ]

            with db_session(settings) as conn:
                replace_clusters(conn, landscape_id, rows)
                mapping = cluster_id_by_label(conn, landscape_id)
                update_layout(
                    conn,
                    landscape_id,
                    {
                        papers[index].paper_id: (
                            float(coords[index][0]),
                            float(coords[index][1]),
                            mapping[int(result.labels[index])],
                        )
                        for index in range(len(papers))
                    },
                )
            # Local labels, not row ids: synthesis needs to recognise noise.
            self.state.local_labels = {
                papers[index].paper_id: int(result.labels[index])
                for index in range(len(papers))
            }
            report(total, total)

        async for event in self._run_blocking("layout", "Projecting papers", work):
            yield event

        yield self._event(
            "layout",
            "done",
            f"Found {self.state.cluster_count} cluster(s), "
            f"{self.state.unclustered_count} unclustered",
            payload={
                "clusters": self.state.cluster_count,
                "unclustered": self.state.unclustered_count,
            },
        )

    # -- stage 5: layout is 4; synthesis is 6th in STAGE_ORDER (phase-7) ------- #

    async def _stage_synthesis(self) -> AsyncIterator[StageEvent]:
        yield self._event("synthesis", "running", "Building the research landscape")
        settings = self.settings
        completer = self.completer
        topic = self.state.topic
        landscape_id = self.state.landscape_id
        papers = [item.paper for item in self.state.ranked] or list(self.state.papers)
        papers_by_id = {p.paper_id: p for p in papers}
        extractions: dict[str, Any] = dict(self.state.extractions)
        for p in papers:
            extractions.setdefault(p.paper_id, None)

        def work(report: ProgressCallback) -> None:
            from store import (
                fetch_clusters,
                fetch_embeddings,
                replace_edges,
                replace_open_problems,
                replace_reading_path,
                replace_tensions,
            )
            report(0, 3)
            with db_session(settings) as conn:
                try:
                    vectors = fetch_embeddings(conn, [p.paper_id for p in papers], settings.embed_model)
                except Exception:
                    vectors = {}
                cluster_rows = fetch_clusters(conn, landscape_id)
            embeddings: dict[str, Any] = {}
            try:
                import numpy as np
                for pid, blob in vectors.items():
                    embeddings[pid] = np.frombuffer(blob, dtype=np.float32).copy()
            except Exception:
                embeddings = {}
            # Clusters for the narrative: persisted rows (local_label/label/description/paper_count).
            clusters = [
                {"local_label": r.get("local_label"), "label": r.get("label"), "description": r.get("description"), "paper_count": r.get("paper_count")}
                for r in cluster_rows
            ]
            unclustered = sum(1 for pid, lab in self.state.local_labels.items() if lab == UNCLUSTERED_LABEL)
            # Deterministic edges first (citation triples from enrichment + kNN).
            try:
                edges, _info = graph_mod.build_graph(
                    [p.paper_id for p in papers], embeddings, list(self.state.citations), topic, settings,
                    completer=completer, papers_by_id=papers_by_id,
                    cluster_of={pid: int(lab) for pid, lab in self.state.local_labels.items()},
                )
            except Exception as exc:  # noqa: BLE001 - graph must not kill synthesis
                logger.warning("Graph build failed (%s); continuing without edges", exc)
                edges = []
            report(1, 3)
            try:
                result = synth_mod.synthesize(clusters, papers, extractions, edges, topic, settings, completer=completer)
            except Exception as exc:  # noqa: BLE001 - synthesis never empty
                logger.warning("Synthesis failed (%s); using deterministic fallback", exc)
                from pipeline.synthesize import fallback_narrative as _fb
                title, summary = _fb(topic, clusters, len(papers), unclustered)
                result = {"title": title, "summary": summary, "narrative_status": "fallback",
                          "tensions": [], "open_problems": [], "reading_path": []}
            report(2, 3)
            with db_session(settings) as conn:
                replace_edges(conn, landscape_id, result.get("edges", edges))
                replace_tensions(conn, landscape_id, result.get("tensions", []))
                replace_open_problems(conn, landscape_id, result.get("open_problems", []))
                replace_reading_path(conn, landscape_id, result.get("reading_path", []))
                _update_landscape(conn, landscape_id, title=result.get("title") or topic,
                                  summary=result.get("summary", ""), status="ready",
                                  narrative_status=result.get("narrative_status", "fallback"))
            self.state.narrative_status = result.get("narrative_status", "fallback")
            self.state.synthesis_result = result  # type: ignore[attr-defined]
            report(3, 3)

        async for event in self._run_blocking("synthesis", "Building the landscape", work):
            yield event
        await asyncio.to_thread(self._flush_llm_calls)
        # Roll up llm cost into the landscape row (D9).
        try:
            with db_session(settings) as conn:
                rows = conn.execute("SELECT COALESCE(SUM(cost_usd),0), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0) FROM llm_calls WHERE landscape_id = ?", (landscape_id,)).fetchone()
                _update_landscape(conn, landscape_id, cost_usd=float(rows[0] or 0.0), tokens_in=int(rows[1] or 0), tokens_out=int(rows[2] or 0))
        except Exception:  # noqa: BLE001 - cost rollup must never break the run
            logger.warning("Could not roll up llm cost")
        summary = await asyncio.to_thread(self._read_landscape_summary)
        yield self._event("synthesis", "done", "Landscape ready", payload=summary)

    def _read_landscape_summary(self) -> dict[str, Any]:
        with db_session(self.settings) as conn:
            return {
                "edges": len(_fetch_edges(conn, self.state.landscape_id)),
                "tensions": len(_fetch_tensions(conn, self.state.landscape_id)),
                "open_problems": len(_fetch_open_problems(conn, self.state.landscape_id)),
                "clusters": self.state.cluster_count,
            }

    # -- driver ------------------------------------------------------------- #

    def _handler(self, stage: StageName) -> Callable[[], AsyncIterator[StageEvent]]:
        return {
            "retrieval": self._stage_retrieval,
            "enrichment": self._stage_enrichment,
            "rerank": self._stage_rerank,
            "extraction": self._stage_extraction,
            "layout": self._stage_layout,
            "synthesis": self._stage_synthesis,
        }[stage]

    async def run(self) -> AsyncIterator[StageEvent]:
        """Yield stage events until the landscape is ready or a stage fails."""
        for stage in STAGE_ORDER:
            try:
                async for event in self._handler(stage)():
                    yield event
            except retrieve_mod.RetrievalThrottled as exc:
                yield self._error(stage, exc, retryable=True)
                await asyncio.to_thread(self._mark_failed, exc)
                return
            except retrieve_mod.RetrievalError as exc:
                # Fatal only when the cache is cold (stale -> degraded inside retrieval).
                retryable = isinstance(exc, retrieve_mod.RetrievalThrottled) or isinstance(
                    exc, retrieve_mod.RetrievalOffline
                )
                yield self._error(stage, exc, retryable=retryable)
                await asyncio.to_thread(self._mark_failed, exc)
                return
            except Exception as exc:  # noqa: BLE001 - report, never crash the stream
                degraded_event = self._skip_or_degrade(stage, exc)
                if degraded_event is not None:
                    yield degraded_event
                    continue
                logger.exception("Stage %s failed", stage)
                yield self._error(stage, exc, retryable=False)
                await asyncio.to_thread(self._mark_failed, exc)
                return

    def _error(self, stage: StageName, exc: Exception, *, retryable: bool) -> StageEvent:
        return self._event(
            stage,
            "error",
            str(exc) or exc.__class__.__name__,
            payload={"retryable": retryable, "error_type": exc.__class__.__name__},
        )

    def _mark_failed(self, exc: Exception) -> None:
        if self.state.landscape_id is None:
            return
        try:
            with db_session(self.settings) as conn:
                _update_landscape(conn, self.state.landscape_id, status="failed")
        except Exception:  # noqa: BLE001 - never mask the original failure
            logger.warning("Could not mark landscape %s failed", self.state.landscape_id)


# --------------------------------------------------------------------------- #
# Store imports, kept behind thin wrappers so the thread boundary is obvious
# and so this module does not import every store symbol at module scope.
# --------------------------------------------------------------------------- #


def _params(settings: Settings) -> dict[str, Any]:
    return {
        "prompt_version": settings.prompt_version,
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
        "rerank_final_count": settings.rerank_final_count,
        "retrieval_max_results": settings.retrieval_max_results,
    }


def _upsert_topic(conn, topic: str) -> int:
    from store import upsert_topic

    return upsert_topic(conn, topic)


def _insert_landscape(conn, *, topic_id: int, title: str, params: dict[str, Any]) -> int:
    from store import insert_landscape

    return insert_landscape(conn, topic_id=topic_id, title=title, params=params)


def _bump_generation(conn, landscape_id: int) -> int:
    from store import bump_generation

    return bump_generation(conn, landscape_id)


def _landscape_paper_ids(conn, landscape_id: int) -> list[str]:
    from store import landscape_paper_ids

    return sorted(landscape_paper_ids(conn, landscape_id))


def _upsert_papers(conn, papers: list[Paper]) -> int:
    from store import upsert_papers

    return upsert_papers(conn, papers)


def _fetch_paper(conn, paper_id: str) -> Paper | None:
    from store import fetch_paper

    return fetch_paper(conn, paper_id)


def _link_paper(conn, landscape_id: int, **kwargs: Any) -> None:
    from store import link_paper

    link_paper(conn, landscape_id, **kwargs)


def _fetch_edges(conn, landscape_id: int) -> list[dict[str, Any]]:
    from store import fetch_edges

    return fetch_edges(conn, landscape_id)


def _fetch_tensions(conn, landscape_id: int) -> list[dict[str, Any]]:
    from store import fetch_tensions

    return fetch_tensions(conn, landscape_id)


def _fetch_open_problems(conn, landscape_id: int) -> list[dict[str, Any]]:
    from store import fetch_open_problems

    return fetch_open_problems(conn, landscape_id)


def _update_landscape(conn, landscape_id: int, **kwargs: Any) -> None:
    from store import update_landscape

    update_landscape(conn, landscape_id, **kwargs)


async def run_pipeline(
    topic: str,
    settings: Settings,
    *,
    completer: JSONCompleter | None = None,
    expand_landscape_id: int | None = None,
    **kwargs: Any,
) -> AsyncIterator[StageEvent]:
    """Convenience wrapper: build a run and iterate it."""
    run = PipelineRun(
        topic, settings, completer=completer, expand_landscape_id=expand_landscape_id, **kwargs
    )
    async for event in run.run():
        yield event


__all__ = ["STAGE_LABELS", "PROGRESS_POLL_SECONDS", "PipelineRun", "PipelineState", "layout_stage", "run_pipeline"]


# --------------------------------------------------------------------------- #
# Layout unit (Phase 6): embed -> project -> cluster -> name -> persist.
# --------------------------------------------------------------------------- #

def layout_stage(
    landscape_id: int,
    ranked: Sequence[RankedPaper],
    topic: str,
    settings: Settings,
    conn: sqlite3.Connection,
    *,
    embedder: EmbedderLike | None = None,
    reducer: ReducerLike | None = None,
    clusterer: Any | None = None,
    completer: JSONCompleter | None = None,
) -> dict[str, Any]:
    """Embed, project, cluster, name, and persist one landscape's layout.

    The papers must already be linked into ``landscape_papers`` (the retrieval
    stage owns membership); this function only writes the two layout columns:
    cluster row id, plus the normalised x/y position of every paper.

    Returns the stage payload: ``clusters`` rows exactly as persisted, the
    ``cluster_count`` / ``unclustered_count`` split, and the ``paper_count``
    the SQL invariant is checked against. Naming runs through
    :func:`pipeline.cluster.name_clusters`, so even with no LLM configured
    every real cluster gets a deterministic tf-idf name rather than an empty
    cell; ``-1`` is persisted as ``Unclustered`` and never passed to the
    labeller.
    """
    papers = [item.paper for item in ranked]
    total = len(papers)
    if total == 0:
        logger.warning("Landscape %d: no linked papers; layout skipped", landscape_id)
        return {
            "clusters": [],
            "cluster_count": 0,
            "unclustered_count": 0,
            "paper_count": 0,
        }

    linked = landscape_paper_ids(conn, landscape_id)
    missing = [paper.paper_id for paper in papers if paper.paper_id not in linked]
    if missing:
        raise ValueError(
            f"{len(missing)} paper/papers are not linked to landscape {landscape_id} "
            f"before layout: {missing[:5]} — run the retrieval stage first"
        )

    vectors = embed_papers(papers, settings, conn, embedder=embedder)
    matrix = embedding_matrix([paper.paper_id for paper in papers], vectors)
    result = layout(matrix, settings, reducer=reducer, clusterer=clusterer)
    coords = normalize_coords(result.coords)
    labels = result.labels

    groups = group_labels(labels)
    centroids = cluster_centroids(coords, labels)
    named = {
        entry["local_label"]: entry
        for entry in name_clusters(
            [
                {
                    "local_label": int(label),
                    "size": len(indices),
                    "topic": topic,
                    "paper_ids": [papers[i].paper_id for i in indices],
                }
                for label, indices in sorted(groups.items())
            ],
            papers,
            settings,
            completer=completer,
        )
    }

    rows: list[dict[str, Any]] = []
    for label, indices in sorted(groups.items()):
        if label == UNCLUSTERED_LABEL:
            label_text, description = UNCLUSTERED_NAME, "Below the clustering threshold."
        else:
            entry = named.get(int(label), {})
            label_text = str(entry.get("label") or f"Cluster {label}")
            description = str(entry.get("description") or "")
        rows.append(
            {
                "local_label": int(label),
                "label": label_text,
                "description": description,
                "paper_count": len(indices),
                "x": float(centroids[int(label)][0]),
                "y": float(centroids[int(label)][1]),
                "color": color_for_label(int(label)),
            }
        )

    # Row ids are re-issued on every run (replace = DELETE + INSERT), so the
    # bridge from local label to row id must be read *after* the replace.
    replace_clusters(conn, landscape_id, rows)
    row_id_by_label = cluster_id_by_label(conn, landscape_id)
    update_layout(
        conn,
        landscape_id,
        {
            papers[index].paper_id: (
                float(coords[index][0]),
                float(coords[index][1]),
                row_id_by_label[int(labels[index])],
            )
            for index in range(total)
        },
    )

    cluster_count = sum(1 for row in rows if row["local_label"] != UNCLUSTERED_LABEL)
    unclustered_count = int(sum(row["paper_count"] for row in rows if row["local_label"] == UNCLUSTERED_LABEL))
    logger.info(
        "Landscape %d laid out: %d papers, %d cluster(s), %d unclustered",
        landscape_id,
        total,
        cluster_count,
        unclustered_count,
    )
    return {
        "clusters": rows,
        "cluster_count": cluster_count,
        "unclustered_count": unclustered_count,
        "paper_count": total,
    }



