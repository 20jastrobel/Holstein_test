# HH VQE Fastcomp Run Record (2026-03-04)

## Seed Source
- adapt_input_json: .vscode-userdata/artifacts/useful/L4/l4_hh_seq_20260302_215706_resume_adaptB_20260303_111311_adapt_B_B_probe_checkpoint_state.json

## Run stopped before completion
- stopped_at_utc: 2026-03-04T19:25:56.490577+00:00
- log: artifacts/logs/l4_hh_seq_20260304_fastcomp_hb_parallel_vqe_hh_hva_tw_from_adaptB_123719.log
- restart_index: 3 / 16
- method: COBYLA
- seed: 7
- progress_every_s: 60.0
- nfev_so_far: 13269
- nfev_restart: 559
- elapsed_s: 2686.3833622494712
- elapsed_restart_s: 120.13850420899689
- best_global_energy: -0.13270974027744376
- current_energy: -0.13270953633028787

## Exact reference
- E_exact_sector: 0.01665790635825912
- delta_to_exact = E_best - E_exact = -0.14936764663570287
- abs_delta_e = 0.14936764663570287

## Active settings at stop
- problem: hh
- L: 4
- boundary: open
- ordering: blocked
- boson_encoding: binary
- n_ph_max: 1
- t: 1.0
- u: 4.0
- dv: 0.0
- omega0: 1.0
- g_ep: 0.5
- vqe_ansatz: hh_hva_tw
- vqe_reps: 1
- vqe_restarts: 16
- vqe_maxiter: 12000
- vqe_energy_backend: one_apply_compiled
- skip_qpe: true
- num_times: 1
- t_final: 0.0
- skip_pdf: true
- output_json: artifacts/json/l4_hh_seq_20260304_fastcomp_hb_parallel_vqe_hh_hva_tw_from_adaptB_123719_fastcomp.json

## Next actions
- Relaunch in parallel from same checkpoint:
  - run A: vqe_reps=2
  - run B: vqe_reps=3

## Parallel Run Teardown + Metrics (2026-03-04)

### Stop action
- stopped active reps2/reps3 fastcomp workers before next launch.
- PIDs issued SIGINT: 93735, 93740

### Stale metrics captured at stop
- reps2 latest heartbeat:
  - nfev_so_far: 1656
  - ts_utc: 2026-03-04T19:55:52.937271+00:00
  - E_best: -0.41386906869680146
  - abs_delta: 0.4305269750550606
- reps3 latest heartbeat:
  - nfev_so_far: 1104
  - ts_utc: 2026-03-04T19:55:52.011675+00:00
  - E_best: -0.09767064086398305
  - abs_delta: 0.11432854722224217

### reps4 launch attempts before cleanup
- 20260304T195515: no `hardcoded_main_start`, no heartbeat captured (run appears to exit before writing log output)
- 20260304T195539: no `hardcoded_main_start`, no heartbeat captured (run appears to exit before writing log output)

## Run Control + Comparison Snapshot (2026-03-04T22:17:39Z)

### Requested stop actions
| target | method | reps | result | matched_pids |
|---|---|---:|---|---|
| kill_cobyla_reps2 | COBYLA | 2 | no_pid | none |
| kill_cobyla_reps3 | COBYLA | 3 | no_pid | none |
| kill_spsa_reps2 | SPSA | 2 | no_pid | none |

### Comparison reference
- E_exact_sector: `0.01665790635825912`
- delta definition: `delta_to_exact = E_best - E_exact_sector`

### Snapshot for requested comparison cohort
| method | reps | role | log | last_event | last_ts_utc | nfev_so_far | E_best | E_current | delta_to_exact | abs_delta_e | note |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
| COBYLA | 4 | baseline | `artifacts/logs/l4_hh_seq_20260304_fastcomp_hb_parallel_reps4_vqe_hh_hva_tw_from_adaptB_setsid_abs.log` | `hardcoded_vqe_heartbeat` | `2026-03-04T21:21:15.544088+00:00` | 1606 | -0.26202387068795385 | -0.26202387068795385 | -0.27868177704621300 | 0.27868177704621300 | interrupted later (`KeyboardInterrupt`) |
| SPSA | 3 | compare-target | `artifacts/logs/l4_hh_seq_20260304_fastcomp_hb_parallel_reps3_vqe_hh_hva_tw_from_adaptB_20260304T214104_spsa.log` | `hardcoded_vqe_restart_start` | `2026-03-04T22:06:32.595609+00:00` | 0 | n/a | n/a | n/a | n/a | no heartbeat yet |
| SPSA | 4 | compare-target | `artifacts/logs/l4_hh_seq_20260304_fastcomp_hb_parallel_reps4_vqe_hh_hva_tw_from_adaptB_20260304T214104_spsa.log` | `hardcoded_vqe_restart_start` | `2026-03-04T22:06:32.516218+00:00` | 0 | n/a | n/a | n/a | n/a | no heartbeat yet |

### Additional stopped-run context requested
| method | reps | role | log | last_event | last_ts_utc | nfev_so_far | E_best | E_current | delta_to_exact | abs_delta_e |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|
| COBYLA | 2 | kill-target | `artifacts/logs/l4_hh_seq_20260304_fastcomp_hb_parallel_reps2_vqe_hh_hva_tw_from_adaptB_20260304T193907.log` | `hardcoded_vqe_heartbeat` | `2026-03-04T21:42:31.599321+00:00` | 9810 | -0.4939335916698295 | -0.49392837635840076 | -0.51059149802808856 | 0.51059149802808856 |
| COBYLA | 3 | kill-target | `artifacts/logs/l4_hh_seq_20260304_fastcomp_hb_parallel_reps3_vqe_hh_hva_tw_from_adaptB_20260304T193907.log` | `hardcoded_vqe_heartbeat` | `2026-03-04T21:42:32.176453+00:00` | 6526 | -0.46086646935087194 | -0.4607208121697815 | -0.47752437570913109 | 0.47752437570913109 |
| SPSA | 2 | kill-target | `artifacts/logs/l4_hh_seq_20260304_fastcomp_hb_parallel_reps2_vqe_hh_hva_tw_from_adaptB_20260304T214104_spsa.log` | `hardcoded_vqe_restart_start` | `2026-03-04T22:06:31.321101+00:00` | 0 | n/a | n/a | n/a | n/a |

### Resumability note
- `hardcoded_hamiltonian_built` and `hardcoded_vqe_restart_start` indicate setup, not a persisted optimizer checkpoint.
- Without a live PID or an explicit optimizer-state handoff artifact, these runs cannot be continued in-place; they must be relaunched.
