# Agent Contract Changelog

This changelog tracks policy-level contract changes for internal AI coding agents.

Format per entry:
- Date (UTC)
- Contract change
- Files touched
- Reason
- Migration note

## 2026-03-05 (UTC)

- Contract change:
  - Canonical authority and execution behavior clarified for agent runs.
  - Added explicit conflict-stop rule: if AGENTS policy and current code/CLI diverge, stop and ask user.
  - Gate policy normalized around `ΔE_abs = |vqe.energy - ground_state.exact_energy_filtered|`:
    - default hard gate for final conventional VQE: `< 1e-4`
    - pre-VQE HVA/ADAPT: diagnostic only (no default hard-fail)
    - `< 1e-7` retained as optional strict-mode gate.
  - Canonical docs constrained to production-safe command examples.

- Files touched:
  - `AGENTS.md`
  - `README.md`
  - `pipelines/run_guide.md`
  - `docs/LLM_RESEARCH_CONTEXT.md`
  - `docs/optimization_routine_llm.md`
  - `test/test_ai_docs_consistency.py`

- Reason:
  - Reduce wrong turns for internal coding agents and prioritize mathematically correct execution paths.

- Migration note:
  - Treat old references to `1e-7` as strict-mode-only unless a command explicitly declares strict mode.
  - In any AGENTS-vs-code mismatch, request user direction before executing.
