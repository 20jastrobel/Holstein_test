# Investigation: HH staged noise / Qiskit boundary map

## Summary
The staged HH noisy path is clearly parameterized by `L` from CLI through physics config, stage construction, drive generation, and noisy execution. The main unresolved issues are localized boundary gaps rather than architectural ambiguity: `patch_snapshot` is a hard stub, `frozen_layout` first-run behavior is unspecified in code/tests, `fixed_couplers` is normalized but not consumed downstream, and non-symmetry mitigation/runtime suppression is mostly parsed-and-recorded rather than executed.

## Symptoms
- Need an exact boundary report for core/statevector, Qiskit lowering, noisy simulation, snapshot, transpilation/layout/patch, and mitigation/suppression.
- Need to distinguish implemented behavior from CLI/spec surface only.
- Need exact file/function/test evidence, not design hypotheses.

## Investigation Log

### Initial assessment
**Hypothesis:** The repo already has narrow owners for most boundaries; the main risk is over-claiming partially wired noise/Qiskit features.
**Findings:** Broad context discovery and focused verification both pointed to existing owners in `hh_staged_workflow.py`, `hh_staged_noise_workflow.py`, `noise_model_spec.py`, `noise_snapshot.py`, `noise_aer_builders.py`, `noise_oracle_runtime.py`, and `qiskit_circuit_report.py`.
**Conclusion:** Investigation should focus on partial/mismatch seams, not redesign.

### L flow / staged HH ownership
**Hypothesis:** `L` is internal to both noiseless and noisy staged HH paths, not external to noisy execution.
**Findings:**
- CLI defines `--L` in `pipelines/hardcoded/hh_staged_cli_args.py:10-12`.
- Scaled defaults depend on `L` in `pipelines/hardcoded/hh_staged_workflow.py:430-446`.
- Config resolution starts with `L = int(getattr(args, "L"))` in `pipelines/hardcoded/hh_staged_workflow.py:493-495`.
- Physics config stores `L` in `pipelines/hardcoded/hh_staged_workflow.py:679-692`.
- HH Hamiltonian / reference construction uses `physics.L` in `pipelines/hardcoded/hh_staged_workflow.py:893-921`.
- Stage execution passes `cfg.physics.L` into warm, ADAPT, replay, and qubit-count logic in `pipelines/hardcoded/hh_staged_workflow.py:1250-1407`.
- Drive construction uses `cfg.physics.L` in `pipelines/hardcoded/hh_staged_workflow.py:1464-1486`.
- Noisy config normalization uses `staged_cfg.physics.L` for symmetry mitigation in `pipelines/hardcoded/hh_staged_noise_workflow.py:135-157`.
- Noisy mode/audit execution passes `L` into exact-bench runners in `pipelines/hardcoded/hh_staged_noise_workflow.py:189-232` and `236-271`.
**Test evidence:** `test/test_hh_staged_noiseless_workflow.py:30-55`, `115-245`; `test/test_hh_staged_noise_workflow.py:37-63`, `72-162`.
**Conclusion:** Confirmed. The staged noisy path is parameterized by `L`.

### Ideal/core path
**Hypothesis:** The production parity path remains core/statevector-first, with staged HH orchestrating warm -> ADAPT -> matched-family replay.
**Findings:**
- `run_stage_pipeline()` is the staged owner in `pipelines/hardcoded/hh_staged_workflow.py:1250-1407`.
- Warm path delegates to `_run_hardcoded_vqe()` in `pipelines/hardcoded/hubbard_pipeline.py:665-947`.
- ADAPT path delegates to `_run_hardcoded_adapt_vqe()` in `pipelines/hardcoded/adapt_pipeline.py:1485-...`.
- Replay remains in `pipelines/hardcoded/hh_vqe_from_adapt_family.py:1371-1465` and `1465-...`.
**Test evidence:** `test/test_hh_staged_noiseless_workflow.py:115-245`.
**Conclusion:** Fully owned and exercised.

### Qiskit lowering path
**Hypothesis:** Shared Qiskit lowering is already centralized rather than ad hoc.
**Findings:**
- `docs/reports/qiskit_circuit_report.py` owns Pauli/ansatz/time-dynamics lowering: `pauli_poly_to_sparse_pauli_op()` at `42-59`, `append_reference_state()` at `64-84`, `ansatz_to_circuit()` at `127-148`, `adapt_ops_to_circuit()` at `152-184`, `build_suzuki2_time_dependent_circuit()` at `189-256`, `build_cfqm_time_dependent_circuit()` at `261-346`, `compute_time_dynamics_proxy_cost()` at `351-418`.
- Staged report flow consumes those helpers in `pipelines/hardcoded/hh_staged_workflow.py:1646-1707`.
- Hardware validation reuses the shared ADAPT helper in `pipelines/exact_bench/hh_noise_hardware_validation.py:694-705`.
**Test evidence:** `test/test_hh_noise_oracle_runtime.py:30-121`; `test/test_hh_staged_circuit_report.py:93-198`.
**Conclusion:** Fully owned and exercised.

### Noisy simulation path
**Hypothesis:** Staged noisy orchestration is separate from actual oracle execution.
**Findings:**
- Wrapper/orchestration lives in `pipelines/hardcoded/hh_staged_noise_workflow.py:115-359`.
- Active noisy trajectory owner is `_run_noisy_method_trajectory()` in `pipelines/exact_bench/hh_noise_robustness_seq_report.py:1202-1514`.
- Final noisy audit owner is `_run_noisy_final_state_audit()` in `pipelines/exact_bench/hh_noise_robustness_seq_report.py:1527-1696`.
- Both construct `OracleConfig` and execute via `ExpectationOracle`.
- `ExpectationOracle.evaluate()` lives in `pipelines/exact_bench/noise_oracle_runtime.py:1588-1629`.
**Test evidence:** `test/test_hh_staged_noise_workflow.py:72-162`; `test/test_hh_noise_oracle_runtime.py:122-631`; `test/test_hh_noise_robustness_benchmarks.py:175-280`.
**Conclusion:** Core noisy simulation path is real, but some declared modes remain incomplete.

### Backend-derived snapshot path
**Hypothesis:** Snapshot concerns are distinct from spec and transpilation.
**Findings:**
- Snapshot types are declared in `pipelines/exact_bench/noise_model_spec.py:143-179`.
- Snapshot freeze/load/write is owned by `pipelines/exact_bench/noise_snapshot.py:196-286`.
- `_build_estimator()` freezes or loads snapshots in `pipelines/exact_bench/noise_oracle_runtime.py:610-644`.
- Frozen JSON replay is explicitly rejected for local Aer unless the noise kind is `patch_snapshot` in `pipelines/exact_bench/noise_oracle_runtime.py:641-645`.
**Test evidence:** `test/test_hh_noise_oracle_runtime.py:545-568`.
**Conclusion:** Snapshot capture/serialization is implemented; snapshot-based replay is only partial.

### Transpilation/layout/patch path
**Hypothesis:** Layout locking and fixed-patch transpilation exist, but patch replay may not.
**Findings:**
- Layout lock key derivation is in `pipelines/exact_bench/noise_aer_builders.py:84-101`.
- `_resolve_initial_layout()` chooses a fixed patch or previously stored/persisted layout and otherwise returns `None` in `pipelines/exact_bench/noise_aer_builders.py:144-168`.
- `_store_layout_lock()` persists initial/final layout plus used qubits/edges in `pipelines/exact_bench/noise_aer_builders.py:172-186`.
- `_transpile_and_cache()` and `transpile_and_lock_patch()` build `TranspileSnapshot`, map observables through layout, and record used qubits/edges in `pipelines/exact_bench/noise_aer_builders.py:408-528`.
- `build_patch_snapshot_artifact()` is an explicit stub in `pipelines/exact_bench/noise_aer_builders.py:613-623`.
- `patch_snapshot` flows through spec/oracle surfaces (`noise_model_spec.py:365-377,417,480-481`; `noise_oracle_runtime.py:671-672,1343-1348`) but lands on that stub.
**Test evidence:** positive fixed-patch artifact coverage in `test/test_hh_noise_oracle_runtime.py:640-681`; layout-lock key derivation in `test/test_hh_noise_validation_cli.py:183-198`.
**Conclusion:** Fixed-patch transpilation/layout capture is implemented. `patch_snapshot` is a repo-visible mismatch. First-run `frozen_layout` behavior is not specified by code/tests.

### Mitigation/suppression path
**Hypothesis:** Symmetry mitigation is real; generic mitigation (`readout`/`zne`/`dd`) and runtime suppression are largely surface metadata today.
**Findings:**
- Mitigation normalization is in `pipelines/exact_bench/noise_oracle_runtime.py:137-178`; symmetry normalization is at `181-247`; runtime ideal-reference downgrade is at `251-263`.
- `_build_estimator()` records mitigation/symmetry metadata for statevector, Aer, and runtime at `570-759`.
- Runtime execution is initialized with `RuntimeEstimatorV2(mode=session)` in `pipelines/exact_bench/noise_oracle_runtime.py:732-759`; no runtime resilience/options plumbing was found in that file.
- Local Aer branch uses mitigation only as metadata; execution-time branching exists only for noise kind / backend setup (`610-731`).
- Executable symmetry mitigation is in `ExpectationOracle._maybe_evaluate_symmetry_mitigated()` at `1398-1516`, restricted to diagonal observables and downgraded for runtime.
- `ResolvedNoiseSpec` still classifies runtime/readout/zne/dd into coarse bundles (`noise_model_spec.py:312-328,378-422`).
**Test evidence:**
- Positive normalization/reporting and executable symmetry tests in `test/test_hh_noise_oracle_runtime.py:203-445,633-637`.
- CLI/config parsing tests in `test/test_hh_noise_validation_cli.py:224-275` and `test/test_hh_noise_robustness_benchmarks.py:48-112`.
- No tests mention `qpu_suppressed`, `qpu_layer_learned`, or `runtime_suppressed`.
- No positive execution tests mention mitigation mode `dd`; `zne` only appears in normalization/reporting tests.
**Conclusion:** Symmetry mitigation is wired; other mitigation/suppression paths are partial/declared-only.

## Root Cause
The repo’s boundary structure is mostly already correct, but several noise-related surfaces advertise more than the current execution path delivers:
1. `patch_snapshot` is exposed from CLI/spec/oracle but `noise_aer_builders.py:613-623` still raises `NotImplementedError`.
2. `frozen_layout` depends on prior persisted layout state via `_resolve_initial_layout()` and silently falls back to `None` when no lock exists (`noise_aer_builders.py:144-168`), leaving first-run semantics undefined.
3. `fixed_couplers` is part of `ResolvedNoiseSpec` and validated in `noise_model_spec.py:345,437,482-483`, but there is no downstream consumer in `noise_aer_builders.py`.
4. `readout` / `zne` / `dd` / runtime suppression are normalized and reported (`noise_oracle_runtime.py:137-178,570-759`; `noise_model_spec.py:312-328,378-422`) but not clearly wired into Aer/runtime execution, aside from symmetry mitigation.

## Recommendations
1. Complete `patch_snapshot` only inside `pipelines/exact_bench/noise_aer_builders.py::build_patch_snapshot_artifact()` and exercise it through the existing oracle path.
2. Define first-run `frozen_layout` behavior explicitly in `pipelines/exact_bench/noise_aer_builders.py::_resolve_initial_layout()` and add a direct characterization test.
3. In `pipelines/exact_bench/noise_aer_builders.py`, either consume `resolved_spec.fixed_couplers` or fail fast when present; add a builder/runtime test to lock that behavior.
4. In `pipelines/exact_bench/noise_oracle_runtime.py::_build_estimator()`, either wire `readout`/`zne`/`dd` / runtime-suppression options into the existing backend branches or reject unsupported modes there with explicit errors.
5. Add narrow tests before behavior changes for all currently unexercised declared surfaces (`patch_snapshot`, first-run `frozen_layout`, runtime suppression modes).

## Preventive Measures
- Treat CLI/spec names as insufficient evidence; require an execution-path owner plus a test before claiming support.
- Keep snapshot, transpile/layout, and runtime/oracle concerns separate; current files already provide that separation.
- For future noisy features, add characterization tests in `test/test_hh_noise_oracle_runtime.py` or `test/test_hh_staged_noise_workflow.py` at the same time the surface is exposed.
