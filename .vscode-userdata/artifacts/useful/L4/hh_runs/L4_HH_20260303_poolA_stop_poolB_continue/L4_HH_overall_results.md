# L4 HH Results (Human + Agent Brief)

_Updated: 2026-03-04T04:12:10Z_

## 2026-03-04 Decision Update (supersedes active-branch notes below)

- We ran ADAPT(B) and seeded VQE in parallel from the same ADAPT checkpoint.
- We then stopped ADAPT(B) by user request and continue only the seeded VQE run.

### ADAPT vs VQE crossover (this run)

- VQE seeded start: `2026-03-04T01:52:22Z` with `|ΔE|=5.122029e-01`.
- Efficiency crossover (VQE improvement > ADAPT improvement at same elapsed time): `2026-03-04T01:54:25Z` (~123s after VQE start).
- Absolute crossover (VQE best |ΔE| <= ADAPT best |ΔE| at same wallclock): `2026-03-04T02:00:27Z` (~485s after VQE start).

### Stop + current status

- ADAPT(B) stop event (user-requested interrupt): `2026-03-04T04:09:37Z`.
- ADAPT(B) best logged (in-flight heartbeat at stop): `|ΔE|=5.056922e-01` at depth 58.
- ADAPT(B) latest checkpoint JSON before interrupt: `2026-03-04T03:57:17Z`, depth 57, `|ΔE|=5.082622e-01`.
- VQE seeded run remains active.
- VQE latest checkpoint (at update time): `2026-03-04T04:11:16Z`, best `|ΔE|=4.993589e-01`, energy `0.516016845901`.

### Artifact links for this decision

- ADAPT(B) log: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_resume_adaptB_20260303_111311_adapt_cont_B.log
- ADAPT(B) checkpoint JSON: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_resume_adaptB_20260303_111311_adapt_B_B_probe_checkpoint_state.json
- Seeded VQE log: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_parallel_vqe_from_adaptB_seeded_20260303_195209_vqe_heavy.log
- Seeded VQE checkpoint JSON: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_parallel_vqe_from_adaptB_seeded_20260303_195209_vqe_checkpoint_state.json

## Physics + Contract

- problem: HH
- L=4, boundary=open, ordering=blocked, boson_encoding=binary
- n_ph_max=1, sector=(2,2)
- t=1.0, U=4.0, dv=0.0, omega0=1.0, g_ep=0.5
- gates: probe<=5e-2, production<=1e-2, aspirational<=5e-3

## Overall Flow (concise)

1. Warm HVA + sequential ADAPT arms (A then B) baseline run to establish L=4 behavior and checkpoints.
2. Continued A-path ADAPT from checkpoint with gradient-stop policy.
3. Ran conventional VQE from A-derived ADAPT checkpoint (heavy budget) and observed very slow slope.
4. Started parallel B-path ADAPT from the same checkpoint using full pool B (UCCSD+PAOP+HVA).
5. Stopped A continuation; kept B continuation running.

## Convergence Verdicts

- A-path ADAPT (UCCSD+PAOP) latest checkpoint: |ΔE|=6.582533e-01 (FAIL vs all gates).
- A-path conventional VQE (from ADAPT checkpoint): best |ΔE|=6.644438e-01 (FAIL vs all gates).
- B-path ADAPT (UCCSD+PAOP+HVA) latest checkpoint: |ΔE|=6.405563e-01 (still far from gates, but substantially better than A at comparable continuation stage).
- Conclusion: A did not converge with ADAPT+VQE in this campaign; B is the preferred branch to continue.

## Last Iterations (what matters)

### A ADAPT (stopped)

| ts_utc | depth | post-opt |ΔE| | step Δ|ΔE| | pre-opt max|g| |
|---|---:|---:|---:|---:|
| 2026-03-03T17:06:55Z | 40 | 6.598581e-01 | -3.141124e-04 | 4.632332e-02 |
| 2026-03-03T17:15:09Z | 41 | 6.592345e-01 | -6.236192e-04 | 5.024437e-02 |
| 2026-03-03T19:09:45Z | 42 | 6.582533e-01 | -9.811432e-04 | 4.293893e-02 |
- recent mean step Δ|ΔE| (last 3 post-opts): -6.396249e-04
- recent slope d(|ΔE|)/dt from post-opt points: -2.177476e-07 /s

### A conventional VQE (from ADAPT checkpoint)

| ts_utc | restart | elapsed_s | calls | best |ΔE| | slope_last3 (/s) |
|---|---:|---:|---:|---:|---:|
| 2026-03-03T16:25:28Z | 2 | 22850.4 | 6050 | 6.644464e-01 | -1.689649e-08 |
| 2026-03-03T16:26:28Z | 2 | 22910.6 | 6106 | 6.644455e-01 | -1.486808e-08 |
| 2026-03-03T16:27:28Z | 2 | 22970.7 | 6162 | 6.644446e-01 | -1.446639e-08 |
- Interpretation: slope magnitude is ~1e-8 /s (near-flat), i.e., not an efficient path to convergence.

### B ADAPT (active)

| ts_utc | depth | post-opt |ΔE| | step Δ|ΔE| | pre-opt max|g| |
|---|---:|---:|---:|---:|
| 2026-03-03T19:02:22Z | 38 | 6.418790e-01 | -9.383672e-05 | 9.201010e-02 |
| 2026-03-03T19:06:24Z | 39 | 6.411920e-01 | -6.869532e-04 | 8.257167e-02 |
| 2026-03-03T19:10:50Z | 40 | 6.405563e-01 | -6.357455e-04 | 6.934845e-02 |
- recent mean step Δ|ΔE| (last 3 post-opts): -4.721785e-04
- recent slope d(|ΔE|)/dt from post-opt points: -2.603740e-06 /s

## Key Checkpoint Metrics

| branch | generated_utc | depth | energy | |ΔE| | rel err | max|g| | nfev_total | runtime_s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A ADAPT | 2026-03-03T19:09:46Z | 42 | 0.674911253303 | 6.582533e-01 | 3.951597e+01 | 4.293893e-02 | 6455 | 3136.99 |
| B ADAPT | 2026-03-03T19:10:51Z | 40 | 0.657214182806 | 6.405563e-01 | 3.845359e+01 | 6.934845e-02 | 2333 | 1215.86 |
| A VQE best | 2026-03-03T16:28:16Z | - | 0.681101698382 | 6.644438e-01 | 3.988759e+01 | - | - | 23019.69 |

## Artifact Links (JSON first)

- baseline sequential run JSON: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_warmstart_uccsd_paop_hva_seq.json
- A ADAPT checkpoint JSON (stopped branch): /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_resume_adapt_20260303_102846_adapt_A_probe_checkpoint_state.json
- B ADAPT checkpoint JSON (active branch): /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_resume_adaptB_20260303_111311_adapt_B_B_probe_checkpoint_state.json
- A VQE checkpoint JSON (A-derived): /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_vqe_cutover_20260303_025908_vqe_checkpoint_state.json
- A vs B compare CSV: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_resume_adaptB_20260303_111311_A_vs_B_compare.csv
- A vs B compare MD: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_resume_adaptB_20260303_111311_A_vs_B_compare.md

## Supporting Logs

- baseline sequential run log: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_warmstart_uccsd_paop_hva_seq.log
- A ADAPT log: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_resume_adapt_20260303_102846_adapt_cont.log
- B ADAPT log: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_resume_adaptB_20260303_111311_adapt_cont_B.log
- A VQE log: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_vqe_cutover_20260303_025908_vqe_heavy.log

## Live Slope Snapshot (2026-03-03T19:48:48Z)

- branch: Pool B ADAPT (active)
- latest checkpoint utc: 2026-03-03T19:47:43Z
- latest depth: 43
- latest best delta_E_abs: 6.346984e-01
- last 3 post-opt stepDeltaE: ['-1.303097e-03', '-1.006393e-03', '-3.548393e-03']
- mean stepDeltaE (last 3): -1.952628e-03
- mean stepDeltaE (last 5): -1.436116e-03
- slope delta_E_abs vs time (last 3 post-opt): -2.561755e-06 /s
- slope delta_E_abs vs time (last 5 post-opt): -2.620500e-06 /s
- slope best_delta_E_abs vs time (last 10 heartbeats): -3.490775e-07 /s
