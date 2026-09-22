"""Tests for Task 6.4 — growth without lying.

Two guarantees the plan pins down: re-projecting a grown corpus with the same
seed and remapping labels by centroid proximity keeps at least 70% of existing
papers in their old cluster (so clusters do not "change name" gratuitously),
and each expansion bumps ``landscapes.generation`` by exactly 1 so the UI can
animate the re-layout.

Everything here is offline: UMAP/HDBSCAN run for real but are fully
deterministic (fixed ``random_state``), and the store tests use the isolated
tmp_path database from ``conftest``.
"""

from __future__ import annotations

import numpy as np

import store
from config import Settings
from pipeline.cluster import (
    UNCLUSTERED_COLOR,
    apply_frame_alignment,
    cluster_centroids,
    color_for_label,
    count_clusters,
    fit_frame_alignment,
    layout,
    match_labels_by_centroid,
)


def _points(sizes: tuple[int, ...], rng: np.random.Generator, spread: float = 0.4) -> np.ndarray:
    """Deterministic blobs on a circle — centers depend only on ``len(sizes)``.

    Both the base corpus and the growth batch call this with three sizes, so
    the extra papers land beside the same centers as the originals.
    """
    angles = 2 * np.pi * np.arange(len(sizes)) / len(sizes)
    centers = 10.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    parts = [
        rng.normal(loc=centers[i], scale=spread, size=(count, 2))
        for i, count in enumerate(sizes)
    ]
    return np.vstack(parts).astype(np.float32)


# --------------------------------------------------------------------------- #
# The remap itself (pure — no database, no UMAP)
# --------------------------------------------------------------------------- #


def test_remap_is_deterministic():
    old = {0: (0.0, 0.0), 1: (10.0, 0.0)}
    new = {5: (0.2, 0.1), 6: (9.8, 0.3)}
    assert match_labels_by_centroid(old, new) == match_labels_by_centroid(old, new)
    assert match_labels_by_centroid(old, new) == {5: 0, 6: 1}


def test_noise_self_maps_and_never_steals_a_label():
    old = {-1: (0.0, 0.0), 0: (5.0, 5.0)}
    new = {-1: (0.1, 0.1), 3: (5.1, 5.0)}
    assert match_labels_by_centroid(old, new) == {-1: -1, 3: 0}


def test_ties_break_toward_the_smaller_old_label():
    old = {0: (0.0, 0.0), 1: (10.0, 0.0)}
    new = {4: (5.0, 0.0)}  # exactly equidistant
    assert match_labels_by_centroid(old, new) == {4: 0}


def test_genuinely_new_cluster_keeps_its_own_label():
    old = {0: (0.0, 0.0)}
    new = {0: (0.0, 0.0), 7: (100.0, 100.0)}
    assert match_labels_by_centroid(old, new) == {0: 0, 7: 7}


def test_remap_is_injective_over_mapped_labels():
    old = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (20.0, 0.0)}
    new = {0: (0.1, 0.0), 1: (10.1, 0.0), 2: (20.1, 0.0)}
    mapped = [v for k, v in match_labels_by_centroid(old, new).items() if v != -1]
    assert sorted(mapped) == [0, 1, 2]  # no two new clusters claimed one old label


def test_unclustered_color_stays_distinct_from_every_palette_entry():
    assert color_for_label(-1) == UNCLUSTERED_COLOR


def test_frame_alignment_recovers_a_known_rotation():
    reference = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    theta = np.pi / 3
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    offset = np.array([7.0, -2.0], dtype=np.float32)
    moving = (reference - offset) @ rotation  # produce moving, then invert it
    alignment = fit_frame_alignment(reference, moving)
    recovered = apply_frame_alignment(moving, alignment)
    assert np.allclose(recovered, reference, atol=1e-4)


def test_frame_alignment_rejects_malformed_inputs():
    import pytest

    reference = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="frame alignment"):
        fit_frame_alignment(reference, np.zeros((4, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="frame alignment"):
        fit_frame_alignment(np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32))


# --------------------------------------------------------------------------- #
# Growth: re-project, remap, retain
# --------------------------------------------------------------------------- #


def test_growth_keeps_seventy_percent_of_papers_in_their_old_cluster(settings: Settings):
    """Plan Task 6.4: +10 papers on a 50-paper corpus retains >=70% membership."""
    base = _points((17, 17, 16), np.random.default_rng(7))
    assert base.shape == (50, 2)

    old = layout(base, settings)
    assert count_clusters(old.labels) >= 2, "fixture must produce real clusters"

    extra = _points((4, 3, 3), np.random.default_rng(11))  # same three centers, +10 papers
    grown = np.vstack([base, extra]).astype(np.float32)
    assert grown.shape == (60, 2)

    # Same seed, full re-projection — never umap.transform().
    new = layout(grown, settings)
    assert new.coords.shape == (60, 2)

    # The projection frame rotates between runs; align it on the 50 shared
    # papers before centroids from the two generations can be compared.
    alignment = fit_frame_alignment(old.coords, new.coords[:50])
    aligned = apply_frame_alignment(new.coords, alignment)
    remap = match_labels_by_centroid(
        cluster_centroids(old.coords, old.labels),
        cluster_centroids(aligned, new.labels),
    )

    retained = 0
    eligible = 0
    for index in range(50):  # the pre-existing papers only
        old_label = int(old.labels[index])
        if old_label == -1:
            continue  # noise has no cluster identity to preserve
        eligible += 1
        mapped = remap.get(int(new.labels[index]), int(new.labels[index]))
        if mapped == old_label:
            retained += 1

    assert eligible >= 40, "the fixture should cluster nearly all base papers"
    retention = retained / eligible
    assert retention >= 0.70, f"only {retention:.0%} of existing papers kept their cluster"


def test_reprojection_with_the_same_seed_is_byte_identical(settings: Settings):
    """Growth reruns the projection; the same input must yield the same map."""
    matrix = _points((17, 17, 16), np.random.default_rng(7))
    first = layout(matrix, settings)
    second = layout(matrix, settings)
    assert np.array_equal(first.coords, second.coords)
    assert np.array_equal(first.labels, second.labels)


def test_extra_papers_land_in_existing_clusters(settings: Settings):
    """New papers join old clusters (identity preserved), not fresh ones."""
    base = _points((17, 17, 16), np.random.default_rng(7))
    extra = _points((4, 3, 3), np.random.default_rng(11))
    old = layout(base, settings)
    grown = layout(np.vstack([base, extra]).astype(np.float32), settings)

    alignment = fit_frame_alignment(old.coords, grown.coords[:50])
    remap = match_labels_by_centroid(
        cluster_centroids(old.coords, old.labels),
        cluster_centroids(apply_frame_alignment(grown.coords, alignment), grown.labels),
    )
    old_label_set = {int(label) for label in old.labels if int(label) != -1}
    joined = 0
    for index in range(50, 60):  # the ten newcomers
        mapped = remap.get(int(grown.labels[index]), int(grown.labels[index]))
        if mapped in old_label_set:
            joined += 1
    assert joined >= 7, f"only {joined}/10 new papers joined an existing cluster"


# --------------------------------------------------------------------------- #
# generation bumps by exactly 1 per expansion (store contract)
# --------------------------------------------------------------------------- #


def test_generation_starts_at_one_and_bumps_exactly_once(settings: Settings, conn):
    topic_id = store.upsert_topic(conn, "diffusion policy learning")
    landscape_id = store.insert_landscape(
        conn, topic_id=topic_id, title="Diffusion Policy Learning"
    )
    conn.commit()

    row = store.fetch_landscape(conn, landscape_id)
    assert row is not None
    assert row["generation"] == 1

    assert store.bump_generation(conn, landscape_id) == 2
    conn.commit()
    row = store.fetch_landscape(conn, landscape_id)
    assert row is not None
    assert row["generation"] == 2, "one expansion, one increment — never a reset or a jump"


def test_bump_generation_on_unknown_id_is_inert(conn):
    assert store.bump_generation(conn, 999_999) == 1

