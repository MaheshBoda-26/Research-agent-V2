"""Layout and clustering.

Two computations, in this order:

1. **UMAP** projects the embeddings to 2D. A fixed ``random_state`` makes the
   projection reproducible, which matters because the layout is persisted: two
   runs over the same corpus must not produce two different maps.
2. **HDBSCAN** finds dense groups *in the 2D projection*, not in the original
   high-dimensional space. Clustering the projection rather than the source means
   the clusters correspond to the blobs the user can actually see. The tradeoff
   is accepted deliberately: the clusters are slightly less natural than
   high-dimensional ones, in exchange for a map that does not lie about where
   its groups are.

Two traps this module avoids:

* **Noise is not a topic.** HDBSCAN labels outliers ``-1``. Treating ``-1`` as a
  cluster produces one giant meaningless group; it is kept separate and rendered
  as its own muted category.
* **``umap.transform()`` must not be used for growth.** It places new points
  using an approximation that is inconsistent with the fitted embedding, so
  existing nodes visibly jump. On growth the whole projection is recomputed with
  the same seed, and ``landscapes.generation`` is bumped so the UI can animate.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from config import Settings
from llm.protocol import JSONCompleter
from models import UNCLUSTERED_LABEL, ClusterLabel, ClusterNaming, Paper
from prompts.cluster import CLUSTER_LABEL_SYSTEM_PROMPT, build_cluster_naming_prompt

logger = logging.getLogger(__name__)

#: Below this many papers UMAP and HDBSCAN have nothing to work with, so the
#: points are placed on a circle and the clustering is skipped.
MIN_FOR_UMAP = 5


class ReducerLike(Protocol):
    def fit_transform(self, matrix: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Layout:
    """Coordinates and cluster labels, aligned to the caller's id order."""

    coords: np.ndarray  # (n, 2) float32
    labels: np.ndarray  # (n,) int; -1 means unclustered

    @property
    def cluster_count(self) -> int:
        """Number of real clusters. Noise is not counted."""
        return count_clusters(self.labels)

    @property
    def unclustered_count(self) -> int:
        return int(np.sum(self.labels == UNCLUSTERED_LABEL))


def circle_positions(count: int) -> np.ndarray:
    """Fallback placement for inputs too small to project."""
    if count == 0:
        return np.zeros((0, 2), dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def project(
    matrix: np.ndarray,
    settings: Settings,
    *,
    reducer: ReducerLike | None = None,
) -> np.ndarray:
    """Project an (n, dim) matrix to (n, 2). Deterministic for a fixed seed."""
    count = matrix.shape[0]
    if count == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if count < MIN_FOR_UMAP:
        logger.info("Only %d paper(s); using a circular layout instead of UMAP", count)
        return circle_positions(count)

    if reducer is not None:
        return np.asarray(reducer.fit_transform(matrix), dtype=np.float32)

    import umap

    model = umap.UMAP(
        n_neighbors=min(settings.umap_n_neighbors, count - 1),
        min_dist=settings.umap_min_dist,
        n_components=2,
        metric="cosine",
        random_state=settings.umap_random_state,
    )
    return np.asarray(model.fit_transform(matrix), dtype=np.float32)


def min_cluster_size(count: int, settings: Settings) -> int:
    """Adaptive floor: ``max(4, ceil(0.06 * n))`` (§2.7).

    V1's fixed floor collapsed 19 papers into a single cluster; the floor must
    scale with the corpus. ``cluster_min_size_ratio`` (default 0.06) is the knob.
    """
    return max(4, math.ceil(settings.cluster_min_size_ratio * count))


def cluster_labels(
    coords: np.ndarray,
    settings: Settings,
    *,
    clusterer: Any | None = None,
) -> np.ndarray:
    """HDBSCAN over the 2D projection. ``-1`` is noise, never a cluster."""
    count = coords.shape[0]
    if count == 0:
        return np.zeros((0,), dtype=int)
    if count < MIN_FOR_UMAP:
        return np.full((count,), UNCLUSTERED_LABEL, dtype=int)

    if clusterer is not None:
        return np.asarray(clusterer.fit_predict(coords)).ravel().astype(int)

    import hdbscan

    size = min_cluster_size(count, settings)
    model = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=2)
    labels = np.asarray(model.fit_predict(coords)).ravel().astype(int)

    noise = int(np.sum(labels == UNCLUSTERED_LABEL))
    logger.info(
        "Clustered %d papers into %d cluster(s); %d unclustered",
        count,
        count_clusters(labels),
        noise,
    )
    if noise and noise > count // 3:
        logger.warning(
            "%d of %d papers (%.0f%%) are unclustered; consider lowering "
            "HDBSCAN_MIN_SAMPLES",
            noise,
            count,
            100.0 * noise / count,
        )
    return labels


def layout(
    matrix: np.ndarray,
    settings: Settings,
    *,
    reducer: ReducerLike | None = None,
    clusterer: Any | None = None,
) -> Layout:
    """Project and cluster in one call."""
    coords = project(matrix, settings, reducer=reducer)
    labels = cluster_labels(coords, settings, clusterer=clusterer)
    return Layout(coords=coords, labels=labels)


def count_clusters(labels: np.ndarray) -> int:
    """Number of real clusters. ``-1`` is noise and is excluded."""
    present = {int(label) for label in labels if int(label) != UNCLUSTERED_LABEL}
    return len(present)


def group_labels(labels: np.ndarray) -> dict[int, list[int]]:
    """Map each label (including ``-1``) to the row indices carrying it."""
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return groups


def cluster_centroids(coords: np.ndarray, labels: np.ndarray) -> dict[int, tuple[float, float]]:
    """Mean position of each label's members, for placing cluster labels."""
    centroids: dict[int, tuple[float, float]] = {}
    for label, indices in group_labels(labels).items():
        points = coords[indices]
        centroids[label] = (float(points[:, 0].mean()), float(points[:, 1].mean()))
    return centroids


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Scale coordinates into roughly [-1, 1] so the frontend has a stable range.

    The absolute scale of a UMAP projection is arbitrary, so the stored values
    are normalized once here rather than leaving every consumer to guess a zoom
    level. Degenerate spreads (all points identical) collapse to the origin.
    """
    if coords.shape[0] == 0:
        return coords
    centered = coords - coords.mean(axis=0, keepdims=True)
    span = np.abs(centered).max()
    if span <= 0:
        return np.zeros_like(coords)
    return (centered / span).astype(np.float32)


#: Distinguishable palette for cluster coloring, assigned in label order.
CLUSTER_COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#4f46e5",
    "#0d9488",
    "#b45309",
    "#9333ea",
)

#: Deliberately desaturated so the unclustered bucket never reads as a topic.
UNCLUSTERED_COLOR = "#94a3b8"


def color_for_label(label: int) -> str:
    if label == UNCLUSTERED_LABEL:
        return UNCLUSTERED_COLOR
    return CLUSTER_COLORS[label % len(CLUSTER_COLORS)]


def fit_frame_alignment(
    reference: np.ndarray,
    moving: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the rigid transform mapping ``moving`` onto ``reference`` row-by-row.

    UMAP's coordinate frame is deterministic for *identical* input but rotates
    and reflects freely between runs on different input (50 vs 60 points), so
    raw centroids from two generations are not comparable — matching them
    directly maps cluster 0 onto cluster 1. The standard orthogonal-Procrustes
    solution fits rotation + translation on the papers shared by both
    generations; the caller then applies it to the new frame before centroid
    matching. Both inputs must be row-aligned ``(n, 2)`` with n >= 2.
    """
    if reference.shape != moving.shape or reference.shape[0] < 2 or reference.shape[1] != 2:
        raise ValueError(
            f"frame alignment needs matching (n>=2, 2) arrays, got "
            f"{reference.shape} and {moving.shape}"
        )
    ref_center = reference.mean(axis=0)
    mov_center = moving.mean(axis=0)
    u, _, vt = np.linalg.svd((moving - mov_center).T @ (reference - ref_center))
    rotation = u @ vt
    offset = ref_center - mov_center @ rotation
    return rotation, offset


def apply_frame_alignment(
    coords: np.ndarray, alignment: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Apply a ``(rotation, offset)`` pair from :func:`fit_frame_alignment`."""
    rotation, offset = alignment
    return coords @ rotation + offset


def match_labels_by_centroid(
    old_centroids: dict[int, tuple[float, float]],
    new_centroids: dict[int, tuple[float, float]],
) -> dict[int, int]:
    """Remap new local labels onto old ones by nearest-centroid proximity.

    Growth (Task 6.4) re-runs the full projection with the same seed, so
    coordinates are comparable across generations — but HDBSCAN relabels
    arbitrarily (cluster "0" today may be "2" tomorrow). Matching by centroid
    proximity preserves cluster identity so a cluster does not "change name"
    gratuitously. ``-1`` (noise) always maps to itself. Pure and deterministic:
    ties break toward the smaller old label. A genuinely new cluster (no free
    old label left) keeps its own label.
    """
    remap: dict[int, int] = {}
    used_old: set[int] = set()
    for new_label in sorted(new_centroids):
        if new_label == UNCLUSTERED_LABEL:
            remap[new_label] = UNCLUSTERED_LABEL
            continue
        nx, ny = new_centroids[new_label]
        best_old: int | None = None
        best_dist = math.inf
        for old_label in sorted(old_centroids):
            if old_label == UNCLUSTERED_LABEL or old_label in used_old:
                continue
            ox, oy = old_centroids[old_label]
            dist = math.hypot(nx - ox, ny - oy)
            if dist < best_dist:
                best_dist = dist
                best_old = old_label
        if best_old is None:
            remap[new_label] = new_label
        else:
            remap[new_label] = best_old
            used_old.add(best_old)
    return remap


def cluster(matrix: np.ndarray, settings: Settings) -> tuple[np.ndarray, dict]:
    """A.9 contract: labels plus the parameters that actually produced them.

    ``matrix`` is the 2D projection — the deliberate clustering input documented
    in the module docstring. The returned dict records the parameters so a run
    can be reproduced from its log alone.
    """
    labels = cluster_labels(matrix, settings)
    params = {
        "min_cluster_size": min_cluster_size(int(matrix.shape[0]), settings),
        "min_samples": 2,
        "clustered_on": "2d_projection",
    }
    return labels, params


def _tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens — the single tokenizer for the D4 gate."""
    return re.findall(r"[a-z0-9]+", text.lower())


def label_restates_topic(label: str, topic: str, threshold: float = 0.70) -> bool:
    """True when ``threshold`` or more of the label's tokens appear in the topic (D4).

    Overlap is measured on the *label* side: the question is how much of the
    label is just the topic repeated back. An empty label or an empty topic
    overlaps nothing — the separate two-token rule rejects empty labels in
    :func:`name_clusters`.
    """
    label_tokens = set(_tokens(label))
    if not label_tokens:
        return False
    topic_tokens = set(_tokens(topic))
    if not topic_tokens:
        return False
    return len(label_tokens & topic_tokens) / len(label_tokens) >= threshold


def _top_distinctive_term(
    member_titles: list[str],
    corpus_titles: list[str],
    *,
    exclude: set[str],
) -> str | None:
    """Highest mean-tf-idf unigram of ``member_titles`` under the corpus idf.

    tf-idf ranks what is distinctive about this cluster *relative to the whole
    corpus* (scikit-learn is already a dependency), and ``exclude`` strips the
    topic's own words so the replacement cannot restate the topic either.
    """
    members = [t for t in member_titles if t and t.strip()]
    corpus = [t for t in corpus_titles if t and t.strip()]
    if not members or not corpus:
        return None

    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectorizer.fit(corpus)
        member_matrix = vectorizer.transform(members)
    except ValueError:  # empty vocabulary — every token was a stopword
        return None
    scores = np.asarray(member_matrix.mean(axis=0)).ravel()
    vocabulary = vectorizer.get_feature_names_out()
    for index in sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i)):
        term = str(vocabulary[index])
        if scores[index] > 0 and term not in exclude:
            return term
    return None


def _distinguishing_terms(
    member_titles: list[str],
    corpus_titles: list[str],
    *,
    exclude: set[str],
    limit: int = 3,
) -> list[str]:
    """Raw-frequency terms occurring more inside the cluster than outside it.

    Last resort when tf-idf cannot be fitted at all (e.g. every title is
    stopwords): only used to decorate the ``Area {n}`` fallback label.
    """
    if not member_titles:
        return []
    inside = Counter(
        token for title in member_titles for token in _tokens(title) if len(token) >= 3
    )
    if not inside:
        return []
    outside = Counter(token for title in corpus_titles for token in _tokens(title)) - Counter(
        token for title in member_titles for token in _tokens(title)
    )
    ranked = sorted(inside, key=lambda t: (-inside[t], t))
    return [t for t in ranked if t not in exclude and inside[t] > outside.get(t, 0)][:limit]


def _fallback_label(
    local_label: int,
    member_titles: list[str],
    corpus_titles: list[str],
    *,
    exclude: set[str],
) -> str:
    """B.4 fallback ladder: tf-idf term, then ``Area {n}`` with terms, then bare.

    Rung 1 is what a gate-rejected model label becomes; rungs 2 and 3 are the
    total-failure path. Fallback labels are deterministic and are never
    re-gated — the two-token rule exists to police *model* output, and a bare
    tf-idf term is by construction a single token.
    """
    term = _top_distinctive_term(member_titles, corpus_titles, exclude=exclude)
    if term is not None:
        return term
    terms = _distinguishing_terms(member_titles, corpus_titles, exclude=exclude)
    if terms:
        return f"Area {local_label} ({', '.join(terms)})"
    return f"Area {local_label}"


#: Default description when the model supplied none (or nothing at all):
#: honest about how the cluster was named, never inventing prose.
FALLBACK_DESCRIPTION = "Grouped by shared research direction."


def name_clusters(
    clusters: list[dict],
    papers: list[Paper],
    settings: Settings,
    *,
    completer: JSONCompleter | None = None,
) -> list[dict]:
    """Name every real cluster (B.4); ``-1`` is never named.

    Each cluster dict must carry ``local_label``, ``paper_ids`` (drives both
    the exemplar titles and the tf-idf fallback) and ``topic`` — the layout
    stage seeds ``topic`` from the landscape, because the anti-restatement
    gate compares every model label against it. A missing topic degrades the
    gate to the length rule only, and says so in the log.

    Returned dicts are ``{"local_label", "label", "description"}``, one per
    real cluster, sorted by label. Every failure path lands somewhere visible:
    a rejected or absent model label becomes the cluster's top tf-idf term,
    and if that cannot be computed either, ``Area {n}``.
    """
    real = [
        c for c in clusters if int(c.get("local_label", UNCLUSTERED_LABEL)) != UNCLUSTERED_LABEL
    ]
    if not real:
        return []

    topic = next(
        (str(c.get("topic") or "").strip() for c in real if str(c.get("topic") or "").strip()),
        "",
    )
    if not topic:
        logger.warning(
            "Cluster dicts carry no topic; the anti-restatement gate can only apply "
            "the two-token rule"
        )

    titles_by_paper = {paper.paper_id: paper.title for paper in papers}
    corpus_titles = [paper.title for paper in papers]
    member_titles: dict[int, list[str]] = {}
    exemplar_titles: dict[int, list[str]] = {}
    for cluster_ in real:
        local = int(cluster_["local_label"])
        ids = [str(pid) for pid in cluster_.get("paper_ids", [])]
        titles = [titles_by_paper[pid] for pid in ids if pid in titles_by_paper]
        member_titles[local] = titles
        exemplar_titles[local] = titles

    proposed: dict[int, ClusterLabel] = {}
    if completer is None:
        logger.warning(
            "No LLM client available; naming %d cluster(s) by tf-idf fallback", len(real)
        )
    else:
        try:
            response = completer.complete_json(
                system=CLUSTER_LABEL_SYSTEM_PROMPT,
                user=build_cluster_naming_prompt(topic, real, exemplar_titles),
                schema=ClusterNaming,
                stage="cluster-naming",
                temperature=0.0,
            )
        except Exception as exc:  # naming must never sink the stage
            logger.warning("Cluster naming call failed: %s", exc)
            response = None
        if isinstance(response, ClusterNaming):
            known = {int(c["local_label"]) for c in real}
            proposed = {
                entry.local_label: entry
                for entry in response.labels
                if entry.local_label in known
            }
        elif response is not None:
            logger.warning(
                "Cluster naming returned %s, not a ClusterNaming envelope; ignoring it",
                type(response).__name__,
            )

    named: list[dict] = []
    for cluster_ in sorted(real, key=lambda c: int(c["local_label"])):
        local = int(cluster_["local_label"])
        entry = proposed.get(local)
        label = (
            " ".join(entry.label.split())[:80]
            if entry is not None and entry.label.strip()
            else ""
        )
        description = (
            " ".join(entry.description.split())
            if entry is not None and entry.description.strip()
            else ""
        )
        rejected = not label or len(_tokens(label)) <= 2 or label_restates_topic(label, topic)
        if rejected:
            if label:
                logger.warning(
                    "Cluster %d label %r rejected (restates topic or <=2 tokens); "
                    "replacing it with the top tf-idf term",
                    local,
                    label,
                )
            label = _fallback_label(
                local,
                member_titles.get(local, []),
                corpus_titles,
                exclude=set(_tokens(topic)),
            )
        named.append(
            {
                "local_label": local,
                "label": label,
                "description": description or FALLBACK_DESCRIPTION,
            }
        )
    return named


__all__ = [
    "CLUSTER_COLORS",
    "MIN_FOR_UMAP",
    "UNCLUSTERED_COLOR",
    "Layout",
    "apply_frame_alignment",
    "circle_positions",
    "cluster",
    "cluster_centroids",
    "cluster_labels",
    "color_for_label",
    "count_clusters",
    "fit_frame_alignment",
    "group_labels",
    "label_restates_topic",
    "layout",
    "match_labels_by_centroid",
    "min_cluster_size",
    "name_clusters",
    "normalize_coords",
    "project",
]
