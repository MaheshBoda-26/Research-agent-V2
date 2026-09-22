# Performance measurements

Measured numbers on this machine (macOS arm64, Python 3.11) land here as phases
make them: cross-encoder throughput (Phase 4), embedding throughput (Phase 6),
per-stage timings against the §13.2 budgets.

## Cross-encoder throughput (Phase 4, Task 4.1) — recorded 2026-09-22

`scripts/bench_encoder.py` (200 synthetic query/passage pairs, CPU, warmup 16):

- model=cross-encoder/ms-marco-MiniLM-L-6-v2 device=cpu pairs=200
- batch_size=16 seconds=0.47 pairs_per_sec=429.5
- batch_size=32 seconds=0.49 pairs_per_sec=406.5
- batch_size=64 seconds=0.48 pairs_per_sec=420.9

Throughput is flat across batch sizes on this machine: the reranker's default
batch size of 16 leaves nothing on the table. A 40-candidate stage costs well
under a second on CPU, comfortably inside the rerank slice of the §13.2
per-stage budgets.
