# Claude Code Handoff: ADAPT Parameter Carry-Forward Fix (Append-Only Default)

Repo root: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test`  
Scope: **ADAPT internals only** (not final VQE replay/restart behavior)

## 1) Objective

Implement a focused ADAPT fix so ansatz growth is cumulative with preserved prior ADAPT parameters by default.

Confirmed current behavior in ADAPT:
- After each generator append, full theta vector is re-optimized.
- Previously learned theta entries are not reset, but they are altered by joint optimization.

For this task, treat that as the issue to fix.

## 2) Required Target Behavior

At ADAPT step `k -> k+1`:
1. Keep optimized prefix `theta[:k]` fixed.
2. Append exactly one new parameter for the new generator.
3. Optimize only that new parameter.
4. Run next gradient/operator screening on the current optimized ADAPT state.

Add explicit re-optimization policy modes:
- `append_only` (new default): freeze prefix, optimize newest parameter only.
- `full`: legacy behavior (re-optimize all parameters).

## 3) Critical Scope Boundary

Do **not** frame this as a final VQE issue.

Out of scope for this task:
- `vqe_minimize` randomized restart starts in `src/quantum/vqe_latex_python_pairs.py`.
- Any final conventional VQE replay/initialization decisions.

This task is only the ADAPT loop and its wiring.

## 4) Files to Read First

1. `AGENTS.md`
2. `docs/LLM_RESEARCH_CONTEXT.md`
3. `pipelines/hardcoded/adapt_pipeline.py`
4. `pipelines/hardcoded/hubbard_pipeline.py`
5. `src/quantum/vqe_latex_python_pairs.py`

Then check related tests:
- `test/test_adapt_vqe_integration.py`
- `test/test_hubbard_adapt_ref_source.py`
- any ADAPT path tests you touch

## 5) Code Anchors (Current Behavior Evidence)

Use these as starting anchors in `pipelines/hardcoded/adapt_pipeline.py`:
- append operator: line ~1705
- append theta entry: line ~1706
- full re-opt comment: line ~1715
- SPSA optimize with `x0=theta`: line ~1783
- COBYLA optimize with `x0=theta`: line ~1810
- writeback optimized theta: lines ~1799, ~1814
- build current state before gradients: lines ~1553-1563

Wrapper pass-through in `pipelines/hardcoded/hubbard_pipeline.py`:
- `_run_internal_adapt_paop(...)` delegates to `adapt_pipeline._run_hardcoded_adapt_vqe(...)` around lines ~942-1004.

## 6) Implementation Requirements

1. Add ADAPT policy knob in `adapt_pipeline.py`:
- plumb through `parse_args()` and `_run_hardcoded_adapt_vqe(...)`
- allowed values: `append_only`, `full`
- set default to `append_only`

2. Implement `append_only` mechanics inside ADAPT depth optimization:
- keep `theta_before_opt[:-1]` fixed
- optimize scalar (or 1D vector) for newest param only
- objective reconstructs full theta: `theta_trial = concat(fixed_prefix, theta_new)`
- keep both optimizer branches supported (SPSA + COBYLA)

3. Keep finite-angle fallback semantics:
- `init_theta` for new generator still comes from fallback if selected
- in `append_only`, only this appended parameter is optimized

4. Preserve ADAPT invariants:
- next depth gradients computed from current optimized ADAPT state
- compiled Pauli-action cache production path remains intact
- no replacement with uncached per-term loops

5. Wire through hardcoded wrapper:
- update `_run_internal_adapt_paop(...)` and its caller args in `hubbard_pipeline.py`
- ensure internal ADAPT branch can select policy explicitly or inherits intended default

## 7) Tests You Must Add/Update

Add or update tests in `test/test_adapt_vqe_integration.py` (or adjacent ADAPT tests):

1. `append_only` prefix-preservation test:
- run ADAPT for at least 2 depth steps
- assert for each depth transition that old theta prefix is unchanged after optimization
- verify newest parameter can change

2. `full` legacy-path test:
- assert at least one previous theta entry is allowed to change after append+opt
- ensures mode behavior difference is real

3. Wrapper pass-through test:
- verify `hubbard_pipeline.py` internal ADAPT call passes/selects the policy as intended

4. Keep existing ADAPT regression tests green:
- rollback guard
- compiled vs legacy state backend parity
- existing stop-policy diagnostics

## 8) Non-Negotiable Constraints

- Do not edit base operator-core files prohibited by `AGENTS.md`.
- Do not introduce Qiskit into core ADAPT path.
- Keep naming/CLI consistent with existing style.
- Keep changes minimal and localized.
- Do not guess behavior; verify by code and tests.

## 9) Required Return Format

Return exactly these sections in your completion report:

A. **Observed Pre-Change Behavior**  
- exact behavior with `file:line` evidence

B. **Changes Made**  
- exact files/functions edited and reason

C. **Per-Iteration Flow: Before vs After**  
- `select -> append -> theta handling -> optimize -> next gradient`

D. **Minimal Diff Summary**  
- concise list of logic changes

E. **Test Evidence**  
- what was run and what each test proves

F. **Residual Risks / Compatibility Notes**  
- especially around default change to `append_only`

## 10) Done Criteria

Done only when all are true:
1. ADAPT default is `append_only`.
2. `full` legacy behavior remains selectable.
3. Prefix-theta preservation is proven by tests.
4. Internal ADAPT wrapper path is wired and covered.
5. No ADAPT gradient-cache invariant regressions.
