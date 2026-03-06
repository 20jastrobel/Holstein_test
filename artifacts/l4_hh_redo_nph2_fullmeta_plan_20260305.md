# L4 HH Redo Plan + Active Run Record (2026-03-05 UTC)

## Objective
- Restart from scratch at `L=4` for HH with:
  - `n_ph_max=2`
  - ADAPT full union pool (`--adapt-pool full_meta`)
  - ADAPT-to-VQE handoff via JSON state export (`initial_state.amplitudes_qn_to_q0`)

## Runtime blockers resolved
- Full dense exact diagonalization OOM at HH L4/n_ph_max=2:
  - Added dense guard (`--dense-eigh-max-dim`, default `8192`), skipping dense eigensolve when Hilbert space is too large.
- HH exact sector energy path OOM:
  - Added sparse-sector HH exact-energy path via `src/quantum/ed_hubbard_holstein.py` + sparse eigensolver fallback.
- Repeated exact-energy recomputation inside ADAPT increased peak memory:
  - Added `exact_gs_override` wiring so `_run_hardcoded_adapt_vqe` reuses the exact value computed in `main`.
- In-session background process reaping:
  - Launch now uses detached `subprocess.Popen(..., start_new_session=True)`.

## Active detached run
- PID: `79447`
- Log: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/logs/l4_hh_redo_nph2_fullmeta_adapt_spsa_detached_20260305T020801.log`
- JSON target: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/json/l4_hh_redo_nph2_fullmeta_adapt_spsa_detached_20260305T020801.json`

## Active configuration (core)
- `--problem hh --L 4 --boundary open --ordering blocked`
- `--boson-encoding binary --n-ph-max 2`
- `--t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5`
- `--adapt-pool full_meta`
- `--adapt-state-backend legacy`
- `--adapt-inner-optimizer SPSA`
- `--adapt-max-depth 160 --adapt-maxiter 5000`
- `--adapt-eps-grad 5e-7 --adapt-eps-energy 1e-9`
- `--adapt-drop-floor 5e-4 --adapt-drop-patience 3 --adapt-drop-min-depth 12 --adapt-grad-floor 2e-2`
- `--adapt-spsa-callback-every 2 --adapt-spsa-progress-every-s 30`
- `--t-final 0.0 --num-times 1 --trotter-steps 1 --skip-pdf`

## Latest heartbeat snapshot (from log)
- stage: `hh_seed_preopt`
- iter: `10`
- nfev_opt_so_far: `20`
- best_fun: `9.842759702330452`
- exact_gs_override: `0.016089430889349953`
- current `|ΔE|` (seed-best vs exact): `9.826670271441102`
- elapsed_opt_s: `36.63831862527877`

## Next step after ADAPT run completes
- Launch VQE handoff from this ADAPT JSON state:
  - `pipelines/hardcoded/hubbard_pipeline.py --initial-state-source adapt_json --adapt-input-json <this_adapt_json> ...`
  - keep HH L4 physics settings aligned with this run.

## Incident + Relaunch Record (2026-03-05 UTC)

### Misconfigured ADAPT run stopped
- stopped_pid: `74597`
- stop_signal: `SIGINT`
- start marker in log:
  - `hardcoded_adapt_main_start`: `2026-03-05T04:43:11.141879+00:00`
  - `adapt_inner_optimizer`: `COBYLA` (unexpected for this workflow)
- failure mode:
  - no `hardcoded_adapt_iter_done` events after ~8h wall-clock
  - run remained in HH seed pre-opt (no depth completion markers)

### Root cause
- Launch command omitted `--adapt-inner-optimizer`, so `adapt_pipeline.py` defaulted to `COBYLA`.

### Replacement run (active)
- PID: `10616`
- launch_utc: `2026-03-05T13:17:27Z`
- Log: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/logs/l4_hh_adapt_fullmeta_spsa_compiled_detached_20260305T131727Z.log`
- JSON target: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/json/l4_hh_adapt_fullmeta_spsa_compiled_detached_20260305T131727Z.json`

### Replacement run settings delta
- `--adapt-inner-optimizer SPSA` (forced)
- `--adapt-state-backend compiled` (forced)
- same checkpoint handoff:
  - `--adapt-ref-json artifacts/json/l4_hh_warm_to_adapt_interrupt_chain_20260305T033710_warm_checkpoint_state.json`
