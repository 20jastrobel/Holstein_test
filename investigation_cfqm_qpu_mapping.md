# Investigation: CFQM to QPU / Qiskit circuit mapping gap

## Summary
The repo already has most of the CFQM-to-hardware stack, but it does **not** have one coherent ownership boundary for when CFQM is honestly circuitizable. Numerical CFQM is fully implemented and supports multiple stage-exponential backends, while hardware-facing CFQM currently works only through a circuitized path that implicitly assumes `cfqm_stage_exp=pauli_suzuki2`; that constraint is enforced in some workflows, ignored in others, and missing entirely from the dedicated hardware-validation runner.

## Symptoms
- CFQM numerical propagation exists in the core path.
- Qiskit circuit lowering for CFQM also exists.
- Generic backend/runtime execution of arbitrary circuits exists.
- The dedicated hardware-validation workflow remains Suzuki-only.
- Some workflows expose `--cfqm-stage-exp`, but the noisy/hardware CFQM path does not honor it consistently.

## Investigation Log

### Phase 1 - Initial assessment
**Hypothesis:** The repo has CFQM and Qiskit pieces, but lacks a coherent handoff for CFQM hardware execution.
**Findings:** Initial search showed CFQM in the core propagator path and in a fixed-seed QPU-prep sweep, but did not yet prove an end-to-end hardware execution path.
**Evidence:**
- `src/quantum/time_propagation/cfqm_propagator.py`
- `pipelines/exact_bench/hh_fixed_seed_qpu_prep_sweep.py`
- `pipelines/exact_bench/hh_noise_hardware_validation.py`
**Conclusion:** Confirmed as a likely seam problem; needed cross-file verification.

### Phase 2 - Broad context gathering
**Hypothesis:** The missing support is probably not total absence of CFQM hardware machinery, but a split between numerical CFQM and hardware-facing circuit workflows.
**Findings:** Context discovery showed four distinct subsystems already exist:
1. Numerical CFQM propagation.
2. Shared Qiskit time-dynamics lowering.
3. Generic circuit execution through `ExpectationOracle`.
4. A sequential noisy report path that already executes Suzuki/CFQM circuits.
**Evidence:**
- `src/quantum/time_propagation/cfqm_propagator.py`
- `docs/reports/qiskit_circuit_report.py`
- `pipelines/exact_bench/noise_oracle_runtime.py`
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py`
**Conclusion:** The missing piece is the contract between these subsystems.

### Phase 3 - Numerical CFQM ownership and semantics
**Hypothesis:** The core CFQM owner is numerical/statevector-first and already preserves the required semantics.
**Findings:** Numerical CFQM supports multiple inner-stage exponentiation backends and contains the exact warning/guardrail semantics documented in the run guide.
**Evidence:**
- `src/quantum/time_propagation/cfqm_propagator.py:145-201`
  - `apply_stage_exponential(...)` supports `dense_expm`, `expm_multiply_sparse`, `pauli_suzuki2`, and `auto`.
  - For `pauli_suzuki2`, it warns: `Inner Suzuki-2 makes overall method 2nd order...`.
- `pipelines/hardcoded/hubbard_pipeline.py:1468-1511`
  - Warns when CFQM is used with non-midpoint sampling: `CFQM ignores midpoint/left/right sampling; uses fixed scheme nodes c_j.`
  - Warns again when `cfqm_stage_exp == "pauli_suzuki2"`.
- `pipelines/run_guide.md:540-574`
  - Documents the same CFQM backend semantics and exact warning strings.
- `test/test_cfqm_acceptance.py:1-320`
  - Confirms true higher-order CFQM for dense/sparse stage exponentials and second-order collapse for `pauli_suzuki2`.
**Conclusion:** Confirmed. Numerical CFQM is coherent and already has the correct semantic contract.

### Phase 3 - Shared Qiskit lowering for time dynamics
**Hypothesis:** There is already a reusable CFQM-to-Qiskit lowering layer.
**Findings:** The shared Qiskit lowering layer exists, but it always lowers CFQM stages using Suzuki-2 synthesis.
**Evidence:**
- `docs/reports/qiskit_circuit_report.py:190-220`
  - `build_suzuki2_time_dependent_circuit(...)` appends `PauliEvolutionGate(..., synthesis=SuzukiTrotter(order=2, reps=1, preserve_order=True))` per step.
- `docs/reports/qiskit_circuit_report.py:262-308`
  - `build_cfqm_time_dependent_circuit(...)` builds CFQM stage maps but still appends each stage as `PauliEvolutionGate(..., synthesis=SuzukiTrotter(order=2, reps=1, preserve_order=True))`.
- `pipelines/exact_bench/hh_seq_transition_utils.py:234-259`
  - `build_time_dependent_sparse_qop(...)` converts ordered EXYZ coefficient maps into the `SparsePauliOp` consumed by those builders.
**Conclusion:** Confirmed. The repo already has CFQM circuit lowering, but only for the circuitized inner-Suzuki profile.

### Phase 3 - Generic hardware execution exists
**Hypothesis:** The backend/runtime execution layer is already capable of handling externally built CFQM circuits.
**Findings:** `ExpectationOracle` is generic and can evaluate arbitrary `QuantumCircuit + SparsePauliOp` pairs; tests already cover time-dependent circuits.
**Evidence:**
- `pipelines/exact_bench/noise_oracle_runtime.py:1451-1776`
  - `_build_estimator()` handles ideal, Aer, backend-scheduled, and runtime-backed execution modes.
- `pipelines/exact_bench/noise_oracle_runtime.py:2194-2720`
  - `ExpectationOracle` owns layout priming, execution, artifact capture, and evaluation of arbitrary circuits.
- `test/test_hh_noise_oracle_runtime.py:1-260`
  - Imports `build_cfqm_time_dependent_circuit` and `build_suzuki2_time_dependent_circuit` directly and verifies valid circuit construction.
- `test/test_hh_noise_oracle_runtime.py:820-920`
  - Confirms backend-scheduled execution on time-dependent circuits through the generic oracle path.
**Conclusion:** Eliminated hypothesis: “there is no generic hardware executor for CFQM circuits.” The executor already exists.

### Phase 3 - Existing CFQM hardware-facing path in the sequential noisy report
**Hypothesis:** At least one workflow may already execute CFQM circuits on hardware-facing paths.
**Findings:** The sequential noisy report already executes both Suzuki and CFQM circuits via `ExpectationOracle`.
**Evidence:**
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py:1694-1762`
  - `_run_noisy_method_trajectory(...)` branches on `method`.
  - For `suzuki2`, it calls `_build_suzuki2_time_dependent_circuit(...)`.
  - For CFQM methods, it calls `_build_cfqm_time_dependent_circuit(...)`.
  - Both are evaluated by `ExpectationOracle`.
- `test/test_hh_noise_robustness_benchmarks.py:1-320`
  - Covers the sequential noisy benchmark/report path.
**Conclusion:** Eliminated hypothesis: “CFQM has no hardware-facing execution path at all.” A partial one already exists.

### Phase 3 - The `cfqm_stage_exp` semantic break
**Hypothesis:** The main seam is that noisy/hardware CFQM ignores the same backend-selection semantics used by numerical CFQM.
**Findings:** This is the core mismatch.
**Evidence:**
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py:1694-1719`
  - `_run_noisy_method_trajectory(...)` accepts `cfqm_coeff_drop_abs_tol`, but **not** `cfqm_stage_exp`.
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py:2428-2534`
  - The noiseless path forwards `cfqm_stage_exp`, `cfqm_coeff_drop_abs_tol`, and `cfqm_normalize` into `hc_pipeline._simulate_trajectory(...)`.
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py:4748-4877`
  - CLI exposes `--cfqm-stage-exp`, `--cfqm-coeff-drop-abs-tol`, and `--cfqm-normalize`.
- `pipelines/hardcoded/hh_staged_noise_workflow.py:209-244`
  - The staged noisy wrapper forwards `cfqm_coeff_drop_abs_tol`, but not `cfqm_stage_exp`.
- `pipelines/hardcoded/hh_staged_noise_workflow.py:355`
  - Same omission persists in the final-audit handoff.
**Conclusion:** Confirmed. The repo exposes CFQM backend-selection semantics broadly, but the noisy/hardware CFQM path does not honor them.

### Phase 3 - Dedicated hardware validation is Suzuki-only
**Hypothesis:** The dedicated hardware-validation runner never adopted the shared CFQM circuit path.
**Findings:** Confirmed. It builds only Suzuki-2 `PauliEvolutionGate` trajectories.
**Evidence:**
- `pipelines/exact_bench/hh_noise_hardware_validation.py:801-815`
  - `_trotterized_circuit(...)` only builds Suzuki-2 evolution and rejects any `suzuki_order != 2`.
- `pipelines/exact_bench/hh_noise_hardware_validation.py:1746-1789`
  - `_run_noisy_trotter(...)` always calls `_trotterized_circuit(...)` for each time.
- `pipelines/exact_bench/hh_noise_hardware_validation.py:2665-2840`
  - CLI has `--run-trotter`, `--trotter-steps`, `--suzuki-order`, etc., but no `--propagator` / `--cfqm-stage-exp` options for the hardware trajectory path.
- `pipelines/exact_bench/hh_noise_hardware_validation.py:3078-3115`
  - `main()` routes `--run-trotter` directly into `_run_noisy_trotter(...)`.
**Conclusion:** Confirmed. The dedicated validator is the clearest missing seam.

### Phase 3 - Existing repo acknowledgment that only circuitized CFQM is transpile-honest
**Hypothesis:** Another workflow may already explicitly recognize that only `pauli_suzuki2` CFQM is circuitizable.
**Findings:** The staged circuit-report workflow already encodes this restriction.
**Evidence:**
- `pipelines/hardcoded/hh_staged_workflow.py:2769-2821`
  - Builds CFQM dynamics circuits using `build_cfqm_time_dependent_circuit(...)`.
  - Then sets:
    - `can_transpile_method = not (str(method).startswith("cfqm") and str(cfg.dynamics.cfqm_stage_exp) != "pauli_suzuki2")`
  - Skips CFQM transpile metrics unless `cfqm_stage_exp == "pauli_suzuki2"` with reason: `cfqm_stage_exp must be pauli_suzuki2 for circuitized CFQM metrics`.
**Conclusion:** Confirmed. The repo already knows the honest boundary; it is just not centralized and enforced everywhere.

### Phase 3 - Fixed-seed QPU-prep sweep is metrics-only, not execution
**Hypothesis:** The QPU-prep sweep may already be the intended end-to-end CFQM hardware test path.
**Findings:** It is not. It only consumes staged workflow metrics and proxy counts.
**Evidence:**
- `pipelines/exact_bench/hh_fixed_seed_qpu_prep_sweep.py:138-205`
  - Builds staged noiseless workflow args, including circuit-backend/transpile flags.
- `pipelines/exact_bench/hh_fixed_seed_qpu_prep_sweep.py:243-305`
  - `_candidate_row(...)` reads `circuit_metrics`, transpile summaries, proxy totals, and dynamics trajectories from payload JSON.
- `test/test_hh_fixed_seed_qpu_prep_sweep.py:35-181`
  - Tests monkeypatch `run_staged_hh_noiseless(...)` and assert summary artifacts/metrics only.
**Conclusion:** Eliminated hypothesis: “the prep sweep is already the real hardware validation path.” It is diagnostics/transpile-only.

### Phase 4 - History / evolution check
**Hypothesis:** The mismatch may reflect a later split between newly added CFQM report support and an older dedicated validator.
**Findings:** Commit history supports that the relevant files evolved separately.
**Evidence:**
- `docs/reports/qiskit_circuit_report.py` appears in git only from `696aefd` (Mar 9, 2026).
- `pipelines/hardcoded/hh_staged_noise_workflow.py` begins at `62b5a27` / expands through `aa37478` and later commits.
- `pipelines/exact_bench/hh_noise_hardware_validation.py` has multiple earlier updates across `c31bb22`, `d5918bb`, `aa37478`, `696aefd`, `0d45678`.
**Conclusion:** Likely drift-by-evolution: shared CFQM circuit lowering/report support grew, but the dedicated validator retained its older Suzuki-only routing.

## Root Cause
The root cause is **not** missing Qiskit capability and **not** missing CFQM logic. The root cause is a **missing shared circuitizability contract** between numerical CFQM and hardware-facing workflows.

Today the repo has two different meanings of “CFQM”:
1. **Numerical CFQM**
   - Owned by `src/quantum/time_propagation/cfqm_propagator.py` and `pipelines/hardcoded/hubbard_pipeline.py`.
   - Supports `dense_expm`, `expm_multiply_sparse`, and `pauli_suzuki2`.
   - Preserves CFQM-specific warnings, backend semantics, and normalization options.
2. **Circuitized CFQM**
   - Owned in practice by `docs/reports/qiskit_circuit_report.py` and reused by some noisy/report workflows.
   - Always lowers each CFQM stage through `PauliEvolutionGate(..., SuzukiTrotter(order=2, reps=1, preserve_order=True))`.
   - Therefore corresponds only to the circuitizable case `cfqm_stage_exp == "pauli_suzuki2"`.

The repo already partially acknowledges this in `hh_staged_workflow.py`, where CFQM transpile metrics are gated to `pauli_suzuki2`, but that same rule is not centralized and not enforced in:
- `hh_noise_robustness_seq_report.py`
- `hh_staged_noise_workflow.py`
- `hh_noise_hardware_validation.py`

So the failure mode is workflow incoherence:
- some paths silently reinterpret CFQM as “circuitized inner-Suzuki CFQM,”
- some paths still expose numerical CFQM backend knobs,
- some paths do not support CFQM at all.

## Eliminated Hypotheses
- **Eliminated:** “The repo has no CFQM circuit lowering.”
  - Ruled out by `docs/reports/qiskit_circuit_report.py:262-308`.
- **Eliminated:** “The repo has no hardware executor for CFQM circuits.”
  - Ruled out by `ExpectationOracle` in `pipelines/exact_bench/noise_oracle_runtime.py:2194-2720` and tests in `test/test_hh_noise_oracle_runtime.py`.
- **Eliminated:** “The fixed-seed QPU-prep sweep already provides end-to-end CFQM hardware execution.”
  - Ruled out by `pipelines/exact_bench/hh_fixed_seed_qpu_prep_sweep.py:243-305` and `test/test_hh_fixed_seed_qpu_prep_sweep.py:35-181`.
- **Eliminated:** “CFQM hardware support is entirely absent.”
  - Ruled out by `pipelines/exact_bench/hh_noise_robustness_seq_report.py:1694-1762`.

## Recommendations
1. **Create one thin hardware-facing dynamics shim that owns circuitizability rules.**
   - Best location: a neutral Qiskit-facing module, not a report file.
   - Candidate responsibilities:
     - route `suzuki2` to Suzuki builder,
     - route `cfqm4`/`cfqm6` to CFQM circuit builder **only when** `cfqm_stage_exp == "pauli_suzuki2"`,
     - reject `dense_expm` / `expm_multiply_sparse` for hardware execution with an explicit error,
     - emit/preserve the existing CFQM warning strings.
2. **Thread `cfqm_stage_exp` through all noisy/hardware workflows.**
   - `pipelines/exact_bench/hh_noise_robustness_seq_report.py`
   - `pipelines/hardcoded/hh_staged_noise_workflow.py`
   - `pipelines/exact_bench/hh_noise_hardware_validation.py`
3. **Replace the validator’s Suzuki-only path with the shared shim.**
   - Replace or generalize:
     - `_trotterized_circuit(...)`
     - `_run_noisy_trotter(...)`
   - Add CLI surface for `--propagator` and `--cfqm-stage-exp`.
4. **Unify the staged-report transpile gate with the same helper/predicate.**
   - `pipelines/hardcoded/hh_staged_workflow.py:2803-2821`
   - Avoid duplicating the “CFQM is circuitizable only for `pauli_suzuki2`” rule.
5. **Add characterization tests for semantic honesty.**
   - Allowed: `cfqm4/cfqm6 + pauli_suzuki2` builds/evaluates circuits.
   - Rejected: `cfqm4/cfqm6 + dense_expm` on hardware-facing paths.
   - Warning parity: same warning strings as documented in `run_guide.md`.
   - Keep tests proving the prep sweep remains metrics-only unless explicitly redesigned.

## Preventive Measures
- Centralize “is this method hardware-circuitizable?” in one helper and reuse it everywhere.
- Do not expose hardware-facing CFQM CLI support without explicitly tying it to `cfqm_stage_exp` semantics.
- Keep numerical-only controls (`dense_expm`, `expm_multiply_sparse`, `cfqm_normalize`) clearly separated from circuit/hardware execution.
- Reuse the same predicate for:
  - noisy execution,
  - hardware validation,
  - transpile/proxy metrics,
  - future report labeling.
- Add tests whenever a new dynamics workflow claims CFQM hardware support.
