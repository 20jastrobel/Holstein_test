# L=3 HH Drive Dynamics from Warm-Start B Depth-15 Proxy

## Provenance
- State JSON: `artifacts/useful/L3/warmstart_states/fix1_warm_start_B_depth15_state.json`
- Trajectory JSON: `artifacts/useful/L3/drive_from_fix1_warm_start_B_depth15.json`
- Trajectory PDF: `artifacts/useful/L3/drive_from_fix1_warm_start_B_depth15.pdf`

## State Build (Bounded Proxy)
- Warm-start budget: reps=3, restarts=1, maxiter=600 (bounded_proxy=True)
- Warm energy: 0.457454230965
- ADAPT depth cap: 15
- ADAPT reached: 15 (stop=max_depth)
- E_exact_sector: 0.244940700128
- E_best(state): 0.430855251475
- ΔE(state): 1.859146e-01
- Relative error(state): 7.590186e-01

## Drive Dynamics Run
- initial_state_source: adapt_json
- drive: enabled=True, A=0.5, omega=2.0, tbar=3.0, phi=0.0, pattern=staggered, time_sampling=midpoint
- trotter_steps=192, reference_steps_multiplier=2, reference_steps=384, num_times=201, t_final=15.0

## Key Trajectory Metrics
- fidelity(t=0): 0.832587963480
- fidelity(t_final): 0.833151871961
- fidelity min/max: 0.832319821379 / 0.833257102969
- energy_total_trotter(t=0): 0.430855251475
- energy_total_trotter(t_final): 0.469626823272
- energy_total_trotter min/max: 0.414234973062 / 0.480414815195
- n_up_site0(t_final): 0.889231834896
- n_dn_site0(t_final): 0.109058098190
- doublon(t_final): 0.209327429259
