# L=3 HH Drive Dynamics from Warm-Start B Full Rebuild

## Provenance
- State JSON: `artifacts/useful/L3/warmstart_states/fix1_warm_start_B_full_state.json`
- Trajectory JSON: `artifacts/useful/L3/drive_from_fix1_warm_start_B_full.json`
- Trajectory PDF: `artifacts/useful/L3/drive_from_fix1_warm_start_B_full.pdf`

## Full-B State Quality (Used For Dynamics)
- E_exact_sector: 0.244940700128
- E_best(state): 0.251448233530
- E_last(state): 0.251448233530
- |ΔE|(state): 6.507533e-03
- Relative error(state): 2.656779e-02
- ADAPT depth reached: 42 (stop=`wallclock_cap`)
- Warm-start E_warm: 0.264627254921
- Full rebuild runtime: 2422.3 s

## Pipeline Internal VQE (Not the propagated branch)
- vqe_ansatz: `hh_hva_ptw`
- E_vqe: 0.422619097010
- E_exact_sector: 0.244940700128
- |ΔE|_vqe: 1.776784e-01
- Relative error_vqe: 7.253935e-01

## Drive Dynamics (propagated from adapt_json state)
- drive: enabled=True, A=0.5, omega=2.0, tbar=3.0, phi=0.0, pattern=staggered, time_sampling=midpoint
- trotter_steps=192, exact_steps_multiplier=2, reference_steps=384, num_times=201, t_final=15.0

## Key Trajectory Metrics
- fidelity(t=0): 0.995909194968
- fidelity(t_final): 0.995881343480
- fidelity min/max: 0.995731981929 / 0.996042925861
- energy_total_trotter(t=0): 0.251448233530
- energy_total_trotter(t_final): 0.276329179848
- energy_total_trotter min/max: 0.228970153402 / 0.285383512962
- n_up_site0(t_final): 0.831810743100
- n_dn_site0(t_final): 0.170221595591
- doublon_avg_trotter(t_final): 0.077129173550
