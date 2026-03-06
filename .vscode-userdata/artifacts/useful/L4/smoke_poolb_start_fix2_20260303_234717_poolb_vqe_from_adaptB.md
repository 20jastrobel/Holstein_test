# L4 HH ADAPT-B-to-Pool-B VQE Summary

Generated UTC: 2026-03-04T05:47:37Z
Tag: smoke_poolb_start_fix2_20260303_234717

## Run Contract
- Problem: HH
- L: 4, boundary: open, ordering: blocked
- sector: (2,2)
- boson_encoding: binary, n_ph_max: 1
- t=1.0, U=4.0, dv=0.0, omega0=1.0, g_ep=0.5
- seed state: Adapt-B checkpoint seed

## VQE Settings
- method: COBYLA
- reps: 1
- restarts: 1
- seed: 7
- maxiter: 5
- progress_every_s: 10.0
- wallclock_cap_s: 30
- pool: uccsd_lifted + paop_lf_std + hva (deduped)

## Key Metrics
- Best |ΔE|: 6.702989596543002
- Best relative_error: 402.3908798850708
- Runtime (s): 9.69181508384645
- nfev_total: 5
- nit_total: 0
- npar: 54
- Best energy: 6.719647502901261
- Exact sector energy: 0.01665790635825912
- Stop reason: Maximum number of function evaluations has been exceeded.
- Gate pass (1e-2): False

## Artifacts
- state JSON: artifacts/useful/L4/smoke_poolb_start_fix2_20260303_234717_poolb_vqe_from_adaptB_state.json
- checkpoint JSON: artifacts/useful/L4/smoke_poolb_start_fix2_20260303_234717_poolb_vqe_from_adaptB_state_checkpoint_state.json
- CSV: artifacts/useful/L4/smoke_poolb_start_fix2_20260303_234717_poolb_vqe_from_adaptB.csv
- log: artifacts/useful/L4/smoke_poolb_start_fix2_20260303_234717_poolb_vqe_from_adaptB.log
- resume input: artifacts/useful/L4/l4_hh_seq_20260302_215706_resume_adaptB_20260303_111311_adapt_B_B_probe_checkpoint_state.json
