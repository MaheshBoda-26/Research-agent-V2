"""Micro-benchmark the cross-encoder scoring the reranker depends on.

Usage: ``.venv/bin/python scripts/bench_encoder.py [--pairs N] [--batch-size B ...]``

Loads ``cross-encoder/ms-marco-MiniLM-L-6-v2`` on CPU and scores synthetic
(query, passage) pairs at batch sizes 16/32/64 (Task 4.1). Prints per-batch
wall time and pairs/sec; the reranker forwards its own batch size, so this
tells us whether 16 is leaving throughput on the table. CPU-only: the service
runs the encoder on CPU in the reference deployment.
"""

from __future__ import annotations

import argparse
import statistics
import time

PAIRS_DEFAULT = 200
WARMUP_PAIRS = 16


def build_pairs(n: int) -> list[tuple[str, str]]:
    topics = [
        "retrieval augmented generation for scientific literature review",
        "diffusion models for protein structure prediction",
        "mixture of experts routing in large language models",
        "contrastive learning for code search embeddings",
    ]
    passage = (
        "We present a method that combines dense retrieval with a reranking "
        "stage over scientific abstracts. Experiments on three benchmarks show "
        "consistent gains in recall at fixed precision budgets, with ablations "
        "isolating the contribution of each component of the pipeline."
    )
    return [(topics[i % len(topics)], f"{passage} Variant {i}.") for i in range(n)]


def bench(model: object, pairs: list[tuple[str, str]], batch_size: int) -> float:
    predict = model.predict  # type: ignore[attr-defined]
    predict(pairs[:WARMUP_PAIRS], batch_size=batch_size, show_progress_bar=False)
    times: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        predict(pairs, batch_size=batch_size, show_progress_bar=False)
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=PAIRS_DEFAULT)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[16, 32, 64])
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model = CrossEncoder(model_name, device="cpu", max_length=512)
    pairs = build_pairs(args.pairs)
    print(f"model={model_name} device=cpu pairs={len(pairs)}")
    for batch_size in args.batch_size:
        seconds = bench(model, pairs, batch_size)
        rate = len(pairs) / seconds
        print(f"batch_size={batch_size:>3} seconds={seconds:.2f} pairs_per_sec={rate:.1f}")
    _ = statistics.median  # keep import used if repetitions change


if __name__ == "__main__":
    main()
