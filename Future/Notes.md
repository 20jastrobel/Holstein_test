### Optional PAOP-LF tweaks (recommended defaults)

**1) Force `paop_r >= 1` for `paop_lf_full` (match prior `paop_full` behavior).**  
Rationale: `paop_lf_full` is intended to include *extended cloud* terms; allowing `paop_r=0` silently removes nonlocal dressing and can weaken the “LF-full” meaning.

Suggested behavior:
- If `--adapt-pool paop_lf_full` and `--paop-r 0`, internally promote to `paop_r = 1`.

**2) Make `hop2` phonon-identity dropping user-configurable (keep current default ON).**  
Current: `paop_hop2(i,j) = K_ij (p_i - p_j)^2` then drop terms that are identity on all phonon qubits.  
Rationale: dropping prevents `hop2` from degenerating into a pure `K_ij` copy (can distort ADAPT selection), but it may be useful to toggle for ablation.

Suggested CLI flag:
- `--paop-hop2-drop-phonon-identity {0,1}` (default `1`)




ansatz factorization, measurement
grouping, symmetry checks

# Notes — “From a VQE–Trotter Hubbard Pipeline to Multiple Publishable Quantum Computing Manuscripts”



## 2) The “six paper” portfolio (high-signal splits)

### A) Eigenphase / leakage diagnostics + correction (VQE → dynamics interface)

* Core question: what *spectral leakage / phase drift* occurs when you evolve a **not-quite-eigenstate** VQE output under product-formula dynamics?
* Deliverable: a **diagnostics-and-correction framework** separating (i) initial-state contamination vs (ii) propagation-induced error, using variance/short-time tests + small subspace (Krylov/QSE-like) phase alignment.

### B) Driven Hubbard nonequilibrium benchmark + error budget

* Deliverable: reproducible driven-observable dataset under Gaussian-envelope drives (densities, doublons, correlators, etc.).
* Key emphasis: a grounded **error budget** decomposing (i) VQE infidelity, (ii) drive discretization + Strang splitting error, (iii) compilation/ordering effects.

### C) Integrator comparison: Strang vs CFQM (time-dependent simulation)

* Deliverable: “control-like workload” benchmark using your drive protocol to compare **Strang** against **commutator-free quasi-Magnus (CFQM/CFET)** schemes.
* Key result aim: identify crossover regimes where CFQM reduces depth for fixed error, with an explicit cost model.

### D) Hubbard–Holstein ADAPT-VQE + unary phonons

* Deliverable: an explicitly digital, symmetry-aware **ADAPT-VQE benchmark for HH** with **unary-encoded truncated phonons**, including operator-pool design and Pareto fronts (qubits vs depth vs measurement cost).
* Includes a “FACT CHECK” style mandate: verify/clean up resource-count and bosonic-operator formulas under truncation/encoding.

###

## 3) Implementation Contract: HH L=4 staged HVA -> ADAPT -> final VQE

### Goal and rationale
- Hartree-Fock-only starts can produce near-zero gradients and stall ADAPT.
- HVA is used as a short seed stage to produce a non-dead reference.
- ADAPT is then used to build expressive ansatz structure efficiently.
- Once marginal ADAPT gains flatten, hand off to fixed-ansatz VQE for final coefficient refinement.

### Default run constants (L=4 HH)
- `adapt_max_depth_probe = 54`
- `adapt_maxiter_probe = 1778`
- Probe gate: `delta_E_abs_best <= 5e-2`
- Production gate: `delta_E_abs_best <= 1e-2`
- Aspirational target (report-only): `<= 5e-3`

### Escalation ladder when probe fails (`delta_E_abs_best > 5e-2`)
1. Escalation A
- `ws_restarts += ceil(L/3)` (for `L=4`, add `2`)
- `ws_maxiter *= 2`
- `adapt_maxiter *= 2`
2. Escalation B
- `adapt_max_depth = ceil(1.25 * adapt_max_depth)`
- Depth cap: `60 * L` (for `L=4`, max `240`)

### Wallclock and stall policy
- Do not run blind forever.
- If a stage emits no new log lines for `>20 min`, mark stalled and proceed to next attempt.
- Hard wallclock caps:
- probe attempts: `1800s`
- production attempts: `7200s`
- Continue on attempt-level errors; do not abort entire workflow because one attempt fails.

### ADAPT handoff cutoff policy (combined criteria)
Use both criteria; hand off ADAPT -> final VQE when either condition is satisfied:

1. Slope + patience cutoff
- Track recent `delta_E_abs` slope over last 3 checkpoints.
- If slope magnitude remains below a threshold for `N` windows and projected gain cannot meet the active gate within remaining budget, stop ADAPT.

2. Gradient-floor cutoff
- Track `max|g|` during ADAPT.
- If `max|g|` remains below `g_min` for `M` checks, stop ADAPT.

Initial tuning placeholders (must be set before production campaign):
- slope threshold
- patience windows `N`
- gradient floor `g_min`
- gradient patience `M`

### Execution sequence
1. Run PROBE attempt with probe settings.
2. Parse probe JSON and compute `delta_E_abs_best`.
3. If probe fails gate, apply Escalation A and rerun probe once.
4. If still failing, apply Escalation B and rerun probe once.
5. On first probe pass, run PRODUCTION with full L=4 defaults.
6. If production misses `1e-2`, keep best run and record as `wallclock/budget-limited`.

### Artifact and reporting contract
- Required summary artifact:
- `artifacts/useful/L4/${TAG}_summary.md`
- Required compact attempts table:
- `artifacts/useful/L4/${TAG}_attempts.csv`
- Required CSV columns:
- `attempt_id,stage,delta_E_abs_best,relative_error_best,gate_pass,runtime_s,stop_reason,artifact_json,artifact_log`
- Keep per-attempt JSON/log/CSV/MD artifacts for probe, probeA, probeB, and production attempts.

### Operator-first benchmark extras to include in summaries
- Pool composition counts (raw and deduplicated A/B pools)
- ADAPT depth reached per stage
- `nfev_total` per stage
- ADAPT stop reasons
- Warm-stage runtime and optimizer effort stats



 If you want true “slope-triggered warm cutoff then continue ADAPT in the
     same run,” I can implement a minimal code change to add that warm handoff
     behavior directly in the script.
Basically, we want the HVA algorithm to stop after a fixed number of depths, when the gradient is less than or equal to some number, say e-3. The depth or the last layer should depend on L. Then we want to have the analogous regimen for the ADAPT VQE, where once the gradient for a certain number of iterations becomes less than some value, we'll stop that. This will be our ansatz that we will assume is sufficiently expressive, and we'll plug it into the Conventional VQE. 
## Phase 2C follow-on ideas (legacy parity + noise validation)

1. Runtime skip artifacts:
- When `--noise-mode runtime` is unavailable (credentials/access), write a structured JSON `skipped` result instead of hard-failing.

2. Baseline registry:
- Add a canonical registry file (for example `artifacts/json/legacy_baselines.json`) that maps named parity anchors to artifact paths and required observables.
- Avoid hardcoded baseline paths in CLI examples.

3. CI parity checks:
- Add a lightweight CI job that runs legacy-parity checks on locked anchors and fails on gate regressions.
- Surface `legacy_parity.passed_all` and max deltas as CI summary outputs.

4. Optional non-matching-grid interpolation mode:
- Keep strict default (`time_grid_match` required).
- Add an explicit opt-in interpolation compare mode for diagnostics only, never as the parity pass/fail gate.
