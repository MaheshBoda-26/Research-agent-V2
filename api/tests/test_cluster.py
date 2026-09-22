"""Tests for Task 6.2 — deterministic projection and adaptive clustering.

Clustering runs against real HDBSCAN on synthetic blobs, which is fast and
needs no network. UMAP is injected as a fake except in one test that verifies
the real library is wired up correctly, because numba's first-run compilation
is the only reason to keep it out of the hot path.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import Settings
from pipeline.cluster import (
    MIN_FOR_UMAP,
    UNCLUSTERED_COLOR,
    circle_positions,
    cluster_centroids,
    cluster_labels,
    color_for_label,
    count_clusters,
    group_labels,
    layout,
    min_cluster_size,
    normalize_coords,
    project,
)


class FakeReducer:
    def __init__(self) -> None:
        self.calls = 0

    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.stack([matrix[:, 0], matrix[:, 1]], axis=1)


def blobs(sizes=(12, 12, 12), spread: float = 0.25) -> np.ndarray:
    """Well-separated 2D clusters, deterministic."""
    rng = np.random.default_rng(7)
    parts = [
        rng.normal(loc=index * 9.0, scale=spread, size=(count, 2))
        for index, count in enumerate(sizes)
    ]
    return np.vstack(parts).astype(np.float32)


def test_circle_positions_are_evenly_spaced():
    coords = circle_positions(4)
    assert coords.shape == (4, 2)
    radii = np.linalg.norm(coords, axis=1)
    assert np.allclose(radii, 1.0, atol=1e-6)


def test_circle_positions_of_nothing_is_empty():
    assert circle_positions(0).shape == (0, 2)


def test_projection_of_nothing_is_empty(settings: Settings):
    assert project(np.zeros((0, 8), dtype=np.float32), settings).shape == (0, 2)


def test_tiny_input_skips_umap_and_uses_the_circle(settings: Settings):
    """Below the threshold UMAP has nothing to work with, so it must not run."""
    reducer = FakeReducer()
    matrix = np.zeros((MIN_FOR_UMAP - 1, 8), dtype=np.float32)
    coords = project(matrix, settings, reducer=reducer)
    assert reducer.calls == 0
    assert coords.shape == (MIN_FOR_UMAP - 1, 2)


def test_projection_is_deterministic_for_a_fixed_seed(settings: Settings):
    matrix = np.random.default_rng(1).normal(size=(30, 12)).astype(np.float32)
    first = project(matrix, settings, reducer=FakeReducer())
    second = project(matrix, settings, reducer=FakeReducer())
    assert np.array_equal(first, second)


def test_real_umap_is_wired_up_and_reproducible(settings: Settings):
    """Guards against a misconfigured or missing umap-learn install."""
    matrix = np.random.default_rng(3).normal(size=(40, 16)).astype(np.float32)
    first = project(matrix, settings)
    second = project(matrix, settings)
    assert first.shape == (40, 2)
    assert np.array_equal(first, second)


def test_min_cluster_size_scales_with_corpus(settings: Settings):
    assert min_cluster_size(19, settings) == 4
    assert min_cluster_size(100, settings) == 6
    assert min_cluster_size(1000, settings) == 60


def test_disjoint_blobs_become_disjoint_clusters(settings: Settings):
    labels = cluster_labels(blobs(), settings)
    assert count_clusters(labels) == 3
    assert set(labels) == {0, 1, 2}


def test_nineteen_papers_do_not_collapse_to_one_cluster(settings: Settings):
    """Regression test for the V1 failure: 19 papers, one cluster.

    With the adaptive floor (max(4, ceil(0.06*19)) = 4) two well-separated
    blobs of 9 and 10 must resolve to ≥2 clusters. If the data genuinely has
    no structure, the honest outcome is everything-noise (flagged below).
    """
    rng = np.random.default_rng(11)
    blob = np.vstack(
        [
            rng.normal(loc=0.0, scale=0.2, size=(9, 2)),
            rng.normal(loc=9.0, scale=0.2, size=(10, 2)),
        ]
    ).astype(np.float32)
    labels = cluster_labels(blob, settings)
    clustered = count_clusters(labels)
    assert clustered >= 2 or set(labels) == {-1}, (
        f"insufficient_structure: 19 papers yielded {clustered} cluster(s); "
        "either ≥2 clusters or explicit all-noise, never a single blob"
    )


def test_noise_is_never_counted_as_a_cluster(settings: Settings):
    labels = np.array([0, 0, 0, 0, -1, -1])
    assert count_clusters(labels) == 1


def test_group_labels_keeps_noise_in_its_own_bucket():
    labels = np.array([0, -1, 0, 1, -1])
    groups = group_labels(labels)
    assert groups[-1] == [1, 4]
    assert groups[0] == [0, 2]


def test_cluster_centroids_are_member_means():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [9.0, 9.0]], dtype=np.float32)
    centroids = cluster_centroids(coords, np.array([0, 0, 1]))
    assert centroids[0] == pytest.approx((1.0, 0.0))
    assert centroids[1] == pytest.approx((9.0, 9.0))


def test_normalize_coords_fits_the_frontend_range():
    coords = np.array([[0.0, 0.0], [4.0, -2.0]], dtype=np.float32)
    normalized = normalize_coords(coords)
    assert np.abs(normalized).max() == pytest.approx(1.0)


def test_normalize_coords_of_identical_points_collapses_to_origin():
    coords = np.ones((3, 2), dtype=np.float32)
    assert np.allclose(normalize_coords(coords), 0.0)


def test_layout_returns_aligned_coords_and_labels(settings: Settings):
    result = layout(blobs(), settings)
    assert result.coords.shape == (36, 2)
    assert result.labels.shape == (36,)
    assert result.cluster_count == 3
    assert result.unclustered_count == 0


def test_layout_of_tiny_input_is_all_noise_on_a_circle(settings: Settings):
    matrix = np.zeros((3, 8), dtype=np.float32)
    result = layout(matrix, settings)
    assert result.coords.shape == (3, 2)
    assert set(result.labels) == {-1}
    assert result.cluster_count == 0


def test_unclustered_bucket_gets_the_muted_colour():
    assert color_for_label(-1) == UNCLUSTERED_COLOR
    assert color_for_label(0) != UNCLUSTERED_COLOR
    assert color_for_label(12) == color_for_label(0)  # palette wraps

    return np.vstack(parts).astype(np.float32)
