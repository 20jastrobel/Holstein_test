# L2 Primary vs Crosscheck Summary

## Run Context

- L=2, HH, t=1, U=4, dv=0, omega0=1, g_ep=0.5, n_ph_max=1, boson=binary
- Geometry/indexing: boundary=open, ordering=blocked
- Primary state source: accessibility warm-start export from `fix1_warm_start_C`
- Practical target: near 1e-3 scale (user preference)

## Energy-Accuracy Comparison

| route | abs(DeltaE) | depth | params | nfev | stop_reason |
|---|---:|---:|---:|---:|---|
| primary accessibility export (fix1_warm_start_C) | 1.086613e-03 | 78 | 78 | 4000 | wallclock_cap |
| strict trend crosscheck (A_heavy) | 3.283231e-01 | 6 | 6 | 264 | eps_energy |

## Static Pipeline (from exported state)

- vqe.energy = 1.667242e-01
- vqe.exact_filtered_energy = 1.586679e-01
- |vqe.energy - exact_filtered| = 8.056329e-03

## Drive Pipeline (from exported state)

- Drive settings: A=0.5, omega=2.0, tbar=3.0, phi=0.0, pattern=staggered, sampling=midpoint, t0=0.0
- Fidelity: t0=0.999387998538, mean=0.999195607805, min=0.998321588096, max=0.999388780805, final=0.998602851105

### PAOP Triad Energy Gaps (static-energy observable)

| metric | t=0 | mean | max | t=t_final |
|---|---:|---:|---:|---:|
| |E_gs - E_exact_paop| | 1.102140e-03 | 1.293687e-03 | 2.007281e-03 | 1.261980e-03 |
| |E_gs - E_trotter_paop| | 1.102140e-03 | 1.857465e-03 | 4.761153e-03 | 1.561437e-03 |
| |E_exact_paop - E_trotter_paop| | 0.000000e+00 | 7.150308e-04 | 3.499546e-03 | 2.994573e-04 |

### HVA Companion Energy Gaps (static-energy observable)

| metric | t=0 | mean | max | t=t_final |
|---|---:|---:|---:|---:|
| |E_gs - E_exact_hva| | 8.056329e-03 | 8.941638e-03 | 2.227837e-02 | 8.452034e-03 |
| |E_gs - E_trotter_hva| | 8.056329e-03 | 9.690742e-03 | 2.219540e-02 | 1.056665e-02 |
| |E_exact_hva - E_trotter_hva| | 5.551115e-17 | 9.448662e-04 | 5.422711e-03 | 2.114615e-03 |

## Artifacts

- state: `artifacts/useful/L2/warmstart_states/fix1_warm_start_B_l2_state.json`
- static json/pdf: `artifacts/useful/L2/l2_open_blocked_static_from_fix1B.json` / `artifacts/useful/L2/l2_open_blocked_static_from_fix1B.pdf`
- drive json/pdf: `artifacts/useful/L2/l2_open_blocked_drive_from_fix1B.json` / `artifacts/useful/L2/l2_open_blocked_drive_from_fix1B.pdf`
- crosscheck: `artifacts/useful/L2/l2_uccsd_paop_hva_trend_crosscheck.json`
