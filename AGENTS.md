# Conventions for coding agents

1. **The plan is the spec.** `research-landscape-agent.md` Appendix A holds the
   interface contracts; implement signatures exactly as written. Appendix B.9's
   cross-prompt rules apply to every prompt. §2.10 is the anti-pattern ledger —
   Phase 12 greps for those patterns, so never introduce them.
2. **Port, don't redesign.** V1 (`../Research-agent`) already solved these
   problems once; copy its proven patterns (config, LLM client repair loop,
   store session handling) and extend them per the plan.
3. **Every expensive artefact is cached; every failure path is recorded.**
   Degradation is surfaced (`degraded=true`, `narrative_status`), never hidden.
4. **Tests are offline and deterministic.** Network and models are injected
   seams (§11.2). A test that can fail because a model had a bad day belongs in
   `evals/`, not `pytest`.
5. **Secrets:** `.env` only, never logged, never `NEXT_PUBLIC_*`, never
   committed. `describe()` redacts `*key*`/`*token*` fields.
6. **Commits:** one per plan task, message taken from the task's `Commit:` line.
