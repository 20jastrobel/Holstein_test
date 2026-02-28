# Hubbard-Holstein Status: Implemented vs Remaining Work

## Executive Summary

The project appears to have moved past core model construction and into integration hardening.
The Hubbard-Holstein (HH) Hamiltonian, variational solvers, and hardcoded trotterized dynamics are in place.
The remaining work is mainly about parity confidence, reproducible reporting, and validation gates.

In practical terms:
- Core HH physics and algorithmic components exist.
- The most important remaining work is compare-path robustness, automated checks, and artifact quality.

## Implemented Capabilities

## 1) Hamiltonian and operator-layer support

- HH Hamiltonian support exists in the hardcoded flow.
- Fermion + phonon register construction is available (with local phonon truncation controls).
- Established operator conventions are already in use:
  - Internal Pauli symbols `e/x/y/z`.
  - Pauli string ordering `q_(n-1)...q_0` with qubit `0` as the rightmost character.
- Jordan-Wigner (JW) helper usage has been integrated in the existing operator stack.

## 2) Ground-state workflows

- Fixed-ansatz VQE is available for HH runs.
- ADAPT-VQE support is available for HH runs.
- Sector-filtered exact-energy comparisons are available (including HH-aware filtering behavior on fermionic sectors).

## 3) Operator pools currently available

ADAPT pool options currently available include:
- `uccsd`
- `cse`
- `full_hamiltonian`
- `hva` (HH-only mode where configured)
- `paop` (alias of `paop_std`)
- `paop_min`
- `paop_std`
- `paop_full`

This means both standard excitation-style pools and polaron-native families are already accessible.

## 4) Time dynamics and trotterization

- Hardcoded trotterization pipeline is in place.
- Time-dependent drive integration exists and is routed through pipeline flags.
- Reference/exact trajectory machinery is present for trajectory-level comparisons.

## 5) Pipeline/runtime structure

- Single-pipeline runners exist for hardcoded and baseline paths.
- Compare pipeline exists for multi-run orchestration and reporting.
- Utility scripts exist for shorthand/scaling runs and regression harnesses.

## Remaining Implementation Work

## 1) HH compare parity confidence (highest priority)

What remains:
- Make HH comparison flow explicit and robust for day-to-day use.
- Ensure HH compare output is unambiguous about which reference path is used.
- Ensure compare metrics include HH-defining parameters and trajectory delta summaries.

Why this matters:
- Without a dependable HH compare path, heavy runs are hard to trust quantitatively.

## 2) Qiskit-side HH strategy

What remains:
- Decide and implement one clear strategy:
  - Full HH parity in Qiskit path, or
  - Explicitly scoped HH reference compare mode, or
  - Deferred Qiskit HH with strict guardrails and clear errors.

Current practical recommendation:
- Keep near-term progress focused on fast HH compare confidence at small `L`, then expand.

## 3) Reporting and PDF completeness

What remains:
- Ensure all HH-relevant PDFs begin with a required parameter manifest section.
- Keep manifest fields complete and consistent across single-run and compare reports.
- Confirm scoreboards include safe-test and HH-relevant delta fields when applicable.

Required manifest fields include:
- Model family/name.
- Ansatz type(s).
- Drive enabled true/false.
- Core physics parameters `t`, `U`, `dv`.
- HH-defining parameters (`omega0`, `g`, `n_ph_max`, boson encoding).
- Run-defining dynamics settings (`t-final`, `num-times`, `suzuki-order`, `trotter-steps`, sampling details).

## 4) Regression and acceptance gate hardening

What remains:
- Formalize a lightweight HH regression matrix for small sizes first.
- Enforce required invariants:
  - A=0 drive safe-test: drive-enabled zero-amplitude matches no-drive trajectory within threshold.
  - HH reduction check: `omega0=0`, `g=0`, no drive should reduce to Hubbard-equivalent behavior (within tolerance).
- Separate fast gating from heavier nightly/overnight sweeps.

## 5) Performance/scaling productization

What remains:
- Keep a strict fast profile for iteration runs.
- Add heavier profile tables for later validation/scaling studies.
- Document when to use each profile and expected runtime budget tiers.

## What Is Likely Not Missing

The current bottleneck is probably not:
- Core HH Hamiltonian term construction.
- Basic HH VQE/ADAPT run capability.
- Access to HH-specific operator pools.
- Basic trotterization machinery.

The current bottleneck is more likely:
- Compare-path confidence.
- Reproducibility/reporting polish.
- Stable, automated acceptance checks.

## Recommended Execution Sequence

## Phase A: Fast confidence pass (now, `L=2`)

- Use short-run HH compare workflows only.
- Skip QPE in routine validation passes.
- Produce JSON-first artifacts.
- Validate safe-test and HH-reduction invariants.

Suggested run profile for quick HH checks:
- `L=2`
- small `t-final`
- modest `num-times`
- modest `trotter-steps`
- `--skip-qpe`

## Phase B: Heavier validation (later today/tonight)

- Extend to larger `L` and longer trajectories.
- Increase trotter depth and reference refinement.
- Compare multiple pools (`hva`, `paop_*`, `full_hamiltonian`) under the same settings.
- Sweep selected `n_ph_max` and drive settings.

## Phase C: Scaling and parity expansion

- Profile memory/time behavior for larger HH instances.
- Tune scaling presets.
- Finalize Qiskit HH parity scope and roadmap based on observed value/cost.

## Milestone Definition of Done (Current Stage)

For this stage, "done enough" means:
- HH quick compare runs at `L=2` are stable and reproducible.
- Required invariants pass consistently.
- HH artifact metadata/manifests are complete.
- Unsupported HH compare modes fail clearly (no silent mismatches).

## Open Questions

- Should full Qiskit HH parity be in the current milestone or a dedicated next milestone?
- Which HH observables are mandatory in compare scoreboards beyond energy/fidelity/density?
- What numeric tolerances should be fixed for quick checks vs heavy nightly checks?

## Bottom Line

You appear to have the core HH implementation already.
The remaining work is primarily integration quality:
- robust HH compare/parity behavior,
- regression gates,
- and polished, reproducible reporting.
