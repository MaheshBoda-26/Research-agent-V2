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
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from config import Settings
from models import UNCLUSTERED_LABEL

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
        labels = np.asarray(clusterer.fit_predict(coords)).ravel()
        return labels.astype(int)

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


__all__ = [
    "CLUSTER_COLORS",
    "MIN_FOR_UMAP",
    "UNCLUSTERED_COLOR",
    "Layout",
    "circle_positions",
    "cluster_centroids",
    "cluster_labels",
    "color_for_label",
    "count_clusters",
    "group_labels",
    "layout",
    "match_labels_by_centroid",
    "min_cluster_size",
    "normalize_coords",
    "project",
]
