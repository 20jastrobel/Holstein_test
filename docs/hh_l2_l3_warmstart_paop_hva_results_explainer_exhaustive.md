# L2/L3 HH Warm-Start PAOP+HVA Exhaustive Results Report (PDF Edition)

## 1. Parameter Manifest (Mandatory Reproducibility Contract)

- Model family/name: `Hubbard` (this report focuses on the Hubbard-Holstein (HH) specialization).
- Ansatz type(s) used: `hh_hva_tw`, `hh_hva_ptw`, ADAPT with `paop_lf_std`, ADAPT pool families `uccsd+paop` and `uccsd+paop+hva`.
- Drive enabled: both `false` (static studies) and `true` (driven study in `drive_from_fix1_warm_start_B_full.json`).
- Core physical parameters: static L2 uses `t=1.0`, `U=2.0`, `dv=0.0`; L3 runs use `t=1.0`, `U=4.0`, `dv=0.0`.
- HH-defining parameters: `omega0=1.0`, `g_ep in {1.0 (L2), 0.5 (L3)}`, `n_ph_max=1`, boson encoding `binary`, ordering `blocked`, boundary `open`.
- Sector conventions used for exact filtered references: L2 `(N_up,N_dn)=(1,1)` and L3 `(N_up,N_dn)=(2,1)`.
- Time-integration parameters for driven run: `t_final=15.0`, `num_times=201`, `trotter_steps=192`, `exact_steps_multiplier=2`, reference steps `384`.
- Drive waveform parameters for driven run: `A=0.5`, `omega=2.0`, `phi=0.0`, `t0=0.0`, `tbar=3.0`, pattern `staggered`, midpoint sampling.

This report expands static quality, optimization cost, ADAPT trace internals, and branch-aware dynamics diagnostics into an exhaustive appendix-first format.

## 2. Source Artifacts and Provenance

- `l2`: `artifacts/useful/L2/H_L2_hh_termwise_regular_lbfgs_t1.0_U2.0_g1_nph1.json`
- `l3_warm`: `artifacts/json/hh_L3_hh_hva_ptw_heavy.json`
- `b_full`: `artifacts/useful/L3/warmstart_states/fix1_warm_start_B_full_state.json`
- `acc`: `artifacts/useful/L3/l3_hh_accessibility_fixes_under8pct.json`
- `trend`: `artifacts/useful/L3/l3_uccsd_paop_hva_trend_full_20260302T000521.json`
- `drive`: `artifacts/useful/L3/drive_from_fix1_warm_start_B_full.json`

All numeric claims below are computed directly from these JSON artifacts.

## 3. Mathematical Definitions and Conventions

Internal Pauli notation convention (repo-level): `e/x/y/z` (identity as `e`).

Pauli string ordering convention: left-to-right is `q_{n-1} ... q_0`; qubit `0` is the rightmost character.

Static HH Hamiltonian split:

$$
H = H_t + H_U + H_{ph} + H_{e-ph}
$$
$$
H_t = -t\sum_{\langle i,j\rangle,\sigma}(c^\dagger_{i\sigma}c_{j\sigma} + c^\dagger_{j\sigma}c_{i\sigma}),
\quad H_U = U\sum_i n_{i\uparrow}n_{i\downarrow}
$$
$$
H_{ph} = \omega_0\sum_i b_i^\dagger b_i,
\quad H_{e-ph} = g\sum_i (n_i - 1)(b_i + b_i^\dagger).
$$

Driven Hamiltonian and waveform:

$$
H(t) = H + H_{drive}(t),
\quad H_{drive}(t)=v(t)\sum_i s_i n_i,
$$
$$
v(t)=A\sin(\omega t+\phi)\exp\left(-\frac{(t-t_0)^2}{2\bar t^2}\right).
$$

Exact filtered benchmark and error metrics:

$$
E_{exact,sector}=\min_{\psi\in\mathcal{H}_{(N_\uparrow,N_\downarrow)}}\langle\psi|H|\psi\rangle,
\quad \Delta E = |E_{best}-E_{exact,sector}|,
\quad \varepsilon_{rel}=\frac{\Delta E}{|E_{exact,sector}|}.
$$

ADAPT ansatz and gradient score:

$$
|\psi_d\rangle = e^{-i\theta_d G_d}\cdots e^{-i\theta_1 G_1}|\psi_0\rangle,
\quad g_m = i\langle\psi_d|[H,G_m]|\psi_d\rangle.
$$

## 4. Model/Complexity Manifest by Run Family

| Family | L | HH qubits | Hilbert size | Hamiltonian terms | Notes |
|---|---:|---:|---:|---:|---|
| L2 strong VQE | 2 | 6 | 64 | 17 | `hh_hva_tw`, L-BFGS-B |
| L3 warm VQE | 3 | 9 | 512 | 27 | `hh_hva_ptw`, COBYLA |
| L3 drive from warm-start B | 3 | 9 | 512 | 27 | branch-aware PAOP/HVA trajectories |

## 5. Static Scoreboard: Energies, Depth, and Cost

| Case | E_exact_sector | E_best | DeltaE | eps_rel | Depth/params | nfev | nit | runtime_s | stop |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| L2 VQE | -0.389553103297 | -0.389552500445 | 6.02851629e-07 | 1.54754673e-06 | reps=6, P=108 | 15042 | 128 | n/a | STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT |
| L3 warm VQE heavy | 0.244940700128 | 0.264627254921 | 0.0196865548 | 0.0803727383 | reps=3, P=39 | 4000 | 0 | n/a | Maximum number of function evaluations has been exceeded. |
| Warm-start B export | 0.244940700128 | 0.251448233530 | 0.0065075334 | 0.0265677913 | depth=42, P=42 | n/a | n/a | 1201.4293 | wallclock_cap |
| Accessibility B | 0.244940700128 | 0.251448233530 | 0.0065075334 | 0.0265677913 | depth=43, P=43 | 8063 | 0 | 1261.9444 | wallclock_cap |
| Accessibility C | 0.244940700128 | 0.249333999503 | 0.00439329938 | 0.0179361755 | depth=38, P=38 | 6227 | 0 | 1208.2765 | wallclock_cap |
| Trend A_medium | 0.244940700128 | 0.245202940355 | 0.000262240227 | 0.00107062741 | depth=20, P=20 | 12640 | 0 | 623.97356 | max_depth |
| Trend A_heavy | 0.244940700128 | 0.245203648021 | 0.000262947893 | 0.00107351654 | depth=36, P=36 | 26833 | 0 | 1722.4704 | max_depth |
| Trend B_medium | 0.244940700128 | 0.245202940355 | 0.000262240227 | 0.00107062741 | depth=20, P=20 | 12640 | 0 | 720.71496 | max_depth |
| Trend B_heavy | 0.244940700128 | 0.245203648021 | 0.000262947893 | 0.00107351654 | depth=36, P=36 | 26833 | 0 | 2379.3171 | max_depth |

## 6. Derived Cost Metrics and Efficiency Ratios

Define:

$$
\kappa_{eval} = \frac{nfev}{P}, \qquad \tau_{eval}=\frac{runtime_s}{nfev}.
$$

- L2 VQE:
  - kappa_eval = 139.277778 evals/parameter
- L3 warm VQE heavy:
  - kappa_eval = 102.564103 evals/parameter
- Accessibility B:
  - tau_eval = 0.156510529 s/eval
- Accessibility C:
  - tau_eval = 0.194038306 s/eval
- Trend A_medium: kappa_eval=632, tau_eval=0.0493649971
- Trend A_heavy: kappa_eval=745.361111, tau_eval=0.0641922406
- Trend B_medium: kappa_eval=632, tau_eval=0.0570185888
- Trend B_heavy: kappa_eval=745.361111, tau_eval=0.0886713056

## 7. Warm-Start Improvement Factors

For warm-start stage define:

$$
F_{impr}=\frac{\Delta E_{warm}}{\Delta E_{best}},
\qquad \eta_d=\frac{\Delta E_{warm}-\Delta E_{best}}{d}.
$$

- B_export: E_warm=0.264627254921, E_best=0.251448233530, E_exact=0.244940700128
  - DeltaE_warm=0.01968655479, DeltaE_best=0.006507533402, F_impr=3.025194582, eta_d=0.0003137862236

- B_accessibility: E_warm=0.264627254921, E_best=0.251448233530, E_exact=0.244940700128
  - DeltaE_warm=0.01968655479, DeltaE_best=0.006507533402, F_impr=3.025194582, eta_d=0.0003064888695

- C_accessibility: E_warm=0.257282636997, E_best=0.249333999503, E_exact=0.244940700128
  - DeltaE_warm=0.01234193687, DeltaE_best=0.004393299375, F_impr=2.809263794, eta_d=0.0002091746709

## 8. Branch-Aware Drive Diagnostics

The drive JSON contains branch order:

1. `exact_gs_filtered`
2. `exact_paop`
3. `trotter_paop`
4. `exact_hva`
5. `trotter_hva`

Aggregate diagnostics from all 201 times:

- Fidelity(paop): min=0.9957319819, mean=0.995892975, max=0.9960429259, final=0.9958813435
- Fidelity(hva):  min=0.8272436534, mean=0.8277501112, max=0.8286952258, final=0.8280706751
- max|Delta E_total| (paop) = 0.001013936364
- max|Delta E_total| (hva)  = 0.002765943995
- mean|Delta E_total| (paop) = 0.0001380074869
- mean|Delta E_total| (hva)  = 0.0004360287629
- max|Delta doublon| (paop) = 0.00555647607
- max|Delta doublon| (hva)  = 0.00473322186
- max energy-gap ratio hva/paop = 2.727926616

Static initial mismatch in same driven file:

- Internal HVA branch DeltaE = 0.177678396882
- Imported PAOP branch DeltaE = 0.00650753340232
- Ratio HVA/PAOP = 27.30349364

## 9. ADAPT Family Selection Detail (Warm-Start B Export)

- Pool type: `paop_lf_std`
- Final depth: 42
- Final parameters: 42
- Stop reason: `wallclock_cap`
- Selected family counts: {'hopdrag': 11, 'disp': 3, 'curdrag': 28}

Full selected-trace appendix is provided later in this report.

## 10. Pool Composition and Trend Experiment Geometry

- Raw pool components: {'uccsd_ferm_only_lifted': 8, 'paop': 7, 'hva': 13}
- Pool A (UCCSD+PAOP): {'raw_sizes': {'uccsd': 8, 'paop': 7}, 'dedup_total': 15, 'dedup_source_presence_counts': {'uccsd': 8, 'paop': 7, 'hva': 0}, 'overlap_count': 0}
- Pool B (UCCSD+PAOP+HVA): {'raw_sizes': {'uccsd': 8, 'paop': 7, 'hva': 13}, 'dedup_total': 28, 'dedup_source_presence_counts': {'uccsd': 8, 'paop': 7, 'hva': 13}, 'overlap_count': 0}

Trend verdict object:

- {'A_uccsd_plus_paop': {'assessable': True, 'likely_convergent_with_more_budget': False, 'medium_delta_E_abs': 0.0002622402274776725, 'heavy_delta_E_abs': 0.0002629478931065188, 'abs_improvement': -7.076656288462768e-07, 'rel_improvement': -0.002698539562952936, 'medium_last_grad_max': 0.004563828566343579, 'heavy_last_grad_max': 0.007721555575588287, 'grad_down': False, 'material_improvement': False, 'heavy_stop_reason': 'max_depth'}, 'B_uccsd_plus_paop_plus_hva': {'assessable': True, 'likely_convergent_with_more_budget': False, 'medium_delta_E_abs': 0.0002622402274776725, 'heavy_delta_E_abs': 0.0002629478931065188, 'abs_improvement': -7.076656288462768e-07, 'rel_improvement': -0.002698539562952936, 'medium_last_grad_max': 0.004563828566343579, 'heavy_last_grad_max': 0.007721555575588287, 'grad_down': False, 'material_improvement': False, 'heavy_stop_reason': 'max_depth'}, 'likely_convergent_with_more_budget': False}

## 11. Key Findings

1. L2 strong VQE is near exact with DeltaE=6.02851629394e-07.
2. L3 warm VQE heavy remains at DeltaE=0.0196865547926 before adaptive refinement.
3. Warm-start B full export reaches DeltaE=0.00650753340231 at depth=42.
4. Accessibility C improves to DeltaE=0.00439329937501 under same wallclock cap class.
5. Best trend run (A_medium) reaches DeltaE=0.000262240227478.
6. In drive dynamics, PAOP branch fidelity stays near 0.99589297 while HVA branch stays near 0.82775011 against the same reference manifold.

## Appendix A. Full 201-Step Trajectory Timeline (Drive Run)

Columns: idx, time, fidelity_paop_trotter, fidelity_hva_trotter, |dE_total_paop|, |dE_total_hva|, |dE_static_paop|, |dE_static_hva|, |dDoublon_paop|, |dDoublon_hva|

```text
000 t= 0.0000 Fpaop=0.995909195 Fhva=0.827522051 dE_tot_p=0.000000000e+00 dE_tot_h=0.000000000e+00 dE_sta_p=0.000000000e+00 dE_sta_h=0.000000000e+00 dD_p=0.000000000e+00 dD_h=0.000000000e+00
001 t= 0.0750 Fpaop=0.995909195 Fhva=0.827522049 dE_tot_p=5.259161995e-10 dE_tot_h=1.029989194e-08 dE_sta_p=4.234793627e-10 dE_sta_h=9.991203087e-09 dD_p=6.446228384e-09 dD_h=4.031415313e-09
002 t= 0.1500 Fpaop=0.995909194 Fhva=0.827522033 dE_tot_p=1.818480944e-08 dE_tot_h=8.361978848e-08 dE_sta_p=2.214350092e-08 dE_sta_h=6.728772128e-08 dD_p=9.380592464e-08 dD_h=6.629052046e-08
003 t= 0.2250 Fpaop=0.995909194 Fhva=0.827521991 dE_tot_p=9.321699851e-08 dE_tot_h=2.679473363e-07 dE_sta_p=1.405324260e-07 dE_sta_h=1.346424694e-07 dD_p=4.219158476e-07 dD_h=3.162415047e-07
004 t= 0.3000 Fpaop=0.995909198 Fhva=0.827521918 dE_tot_p=2.523839935e-07 dE_tot_h=5.664528209e-07 dE_sta_p=4.773994414e-07 dE_sta_h=2.232414464e-08 dD_p=1.140388703e-06 dD_h=8.976292562e-07
005 t= 0.3750 Fpaop=0.995909200 Fhva=0.827521821 dE_tot_p=4.747453728e-07 dE_tot_h=9.228107520e-07 dE_sta_p=1.166822459e-06 dE_sta_h=5.912496404e-07 dD_p=2.287115731e-06 dD_h=1.893653892e-06
006 t= 0.4500 Fpaop=0.995909193 Fhva=0.827521712 dE_tot_p=6.942662646e-07 dE_tot_h=1.222814086e-06 dE_sta_p=2.314355582e-06 dE_sta_h=2.047214015e-06 dD_p=3.748029539e-06 dD_h=3.284047311e-06
007 t= 0.5250 Fpaop=0.995909163 Fhva=0.827521607 dE_tot_p=8.485133023e-07 dE_tot_h=1.308841384e-06 dE_sta_p=3.960282197e-06 dE_sta_h=4.545000225e-06 dD_p=5.307617827e-06 dD_h=4.956588358e-06
008 t= 0.6000 Fpaop=0.995909102 Fhva=0.827521514 dE_tot_p=9.511406966e-07 dE_tot_h=1.005845066e-06 dE_sta_p=6.051094301e-06 dE_sta_h=8.000772439e-06 dD_p=6.768044472e-06 dD_h=6.765493285e-06
009 t= 0.6750 Fpaop=0.995909009 Fhva=0.827521434 dE_tot_p=1.141264606e-06 dE_tot_h=1.589000044e-07 dE_sta_p=8.424698758e-06 dE_sta_h=1.199473008e-05 dD_p=8.070314775e-06 dD_h=8.602435915e-06
010 t= 0.7500 Fpaop=0.995908898 Fhva=0.827521355 dE_tot_p=1.662910886e-06 dE_tot_h=1.319579442e-06 dE_sta_p=1.081937904e-05 dE_sta_h=1.584694319e-05 dD_p=9.340307108e-06 dD_h=1.043707514e-05
011 t= 0.8250 Fpaop=0.995908791 Fhva=0.827521261 dE_tot_p=2.758897424e-06 dE_tot_h=3.399185739e-06 dE_sta_p=1.291302719e-05 dE_sta_h=1.880738731e-05 dD_p=1.081660310e-05 dD_h=1.229889472e-05
012 t= 0.9000 Fpaop=0.995908714 Fhva=0.827521138 dE_tot_p=4.512977988e-06 dE_tot_h=5.888030310e-06 dE_sta_p=1.438420436e-05 dE_sta_h=2.028819991e-05 dD_p=1.268193108e-05 dD_h=1.420456683e-05
013 t= 0.9750 Fpaop=0.995908683 Fhva=0.827520975 dE_tot_p=6.714308952e-06 dE_tot_h=8.409864406e-06 dE_sta_p=1.497131928e-05 dE_sta_h=2.003560367e-05 dD_p=1.488165390e-05 dD_h=1.606864014e-05
014 t= 1.0500 Fpaop=0.995908703 Fhva=0.827520768 dE_tot_p=8.823529706e-06 dE_tot_h=1.041602598e-05 dE_sta_p=1.450648655e-05 dE_sta_h=1.815824586e-05 dD_p=1.703515287e-05 dD_h=1.765193776e-05
015 t= 1.1250 Fpaop=0.995908759 Fhva=0.827520505 dE_tot_p=1.008319018e-05 dE_tot_h=1.124415065e-05 dE_sta_p=1.291977851e-05 dE_sta_h=1.499451812e-05 dD_p=1.851255696e-05 dD_h=1.859118524e-05
016 t= 1.2000 Fpaop=0.995908826 Fhva=0.827520164 dE_tot_p=9.753830009e-06 dE_tot_h=1.022814285e-05 dE_sta_p=1.023208530e-05 dE_sta_h=1.088149555e-05 dD_p=1.867327393e-05 dD_h=1.851813084e-05
017 t= 1.2750 Fpaop=0.995908873 Fhva=0.827519702 dE_tot_p=7.400055592e-06 dE_tot_h=6.849729008e-06 dE_sta_p=6.559167645e-06 dE_sta_h=5.939336113e-06 dD_p=1.718042506e-05 dD_h=1.723207653e-05
018 t= 1.3500 Fpaop=0.995908879 Fhva=0.827519063 dE_tot_p=3.122574394e-06 dE_tot_h=9.027589042e-07 dE_sta_p=2.129168713e-06 dE_sta_h=1.997610516e-08 dD_p=1.425639239e-05 dD_h=1.485657999e-05
019 t= 1.4250 Fpaop=0.995908836 Fhva=0.827518188 dE_tot_p=2.359307409e-06 dE_tot_h=7.376366905e-06 dE_sta_p=2.712202644e-06 dE_sta_h=7.407305075e-06 dD_p=1.075307438e-05 dD_h=1.190521156e-05
020 t= 1.5000 Fpaop=0.995908747 Fhva=0.827517042 dE_tot_p=7.830043666e-06 dE_tot_h=1.726582180e-05 dE_sta_p=7.542939914e-06 dE_sta_h=1.661979487e-05 dD_p=7.974508348e-06 dD_h=9.207917291e-06
021 t= 1.5750 Fpaop=0.995908623 Fhva=0.827515633 dE_tot_p=1.189924312e-05 dE_tot_h=2.764442422e-05 dE_sta_p=1.196121070e-05 dE_sta_h=2.774055237e-05 dD_p=7.285293114e-06 dD_h=7.701186427e-06
022 t= 1.6500 Fpaop=0.995908467 Fhva=0.827514031 dE_tot_p=1.344600489e-05 dE_tot_h=3.721310652e-05 dE_sta_p=1.570049647e-05 dE_sta_h=4.028649024e-05 dD_p=9.630133221e-06 dD_h=8.143847264e-06
023 t= 1.7250 Fpaop=0.995908268 Fhva=0.827512370 dE_tot_p=1.204721880e-05 dE_tot_h=4.480671907e-05 dE_sta_p=1.869533490e-05 dE_sta_h=5.310095929e-05 dD_p=1.514273303e-05 dD_h=1.086287719e-05
024 t= 1.8000 Fpaop=0.995907996 Fhva=0.827510832 dE_tot_p=8.220819400e-06 dE_tot_h=4.970046046e-05 dE_sta_p=2.104079804e-05 dE_sta_h=6.444426805e-05 dD_p=2.301197469e-05 dD_h=1.563939025e-05
025 t= 1.8750 Fpaop=0.995907613 Fhva=0.827509619 dE_tot_p=3.346156600e-06 dE_tot_h=5.179095158e-05 dE_sta_p=2.284416931e-05 dE_sta_h=7.227785508e-05 dD_p=3.169475643e-05 dD_h=2.180389368e-05
026 t= 1.9500 Fpaop=0.995907089 Fhva=0.827508924 dE_tot_p=7.720171837e-07 dE_tot_h=5.155675449e-05 dE_sta_p=2.403656728e-05 dE_sta_h=7.467969889e-05 dD_p=3.943760678e-05 dD_h=2.852976017e-05
027 t= 2.0250 Fpaop=0.995906426 Fhva=0.827508898 dE_tot_p=2.577571297e-06 dE_tot_h=4.978104296e-05 dE_sta_p=2.425207782e-05 dE_sta_h=7.028318360e-05 dD_p=4.493654439e-05 dD_h=3.522229395e-05
028 t= 2.1000 Fpaop=0.995905665 Fhva=0.827509639 dE_tot_p=1.423119283e-06 dE_tot_h=4.712464909e-05 dE_sta_p=2.286394740e-05 dE_sta_h=5.861509534e-05 dD_p=4.788213108e-05 dD_h=4.183807804e-05
029 t= 2.1750 Fpaop=0.995904891 Fhva=0.827511194 dE_tot_p=2.073100393e-06 dE_tot_h=4.372127915e-05 dE_sta_p=1.919709382e-05 dE_sta_h=4.023317113e-05 dD_p=4.914964837e-05 dD_h=4.897160881e-05
030 t= 2.2500 Fpaop=0.995904212 Fhva=0.827513574 dE_tot_p=6.163095011e-06 dE_tot_h=3.897964923e-05 dE_sta_p=1.284959352e-05 dE_sta_h=1.662713078e-05 dD_p=5.051321227e-05 dD_h=5.762543137e-05
031 t= 2.3250 Fpaop=0.995903721 Fhva=0.827516774 dE_tot_p=8.582007800e-06 dE_tot_h=3.170329653e-05 dE_sta_p=3.996805440e-06 dE_sta_h=1.007425372e-05 dD_p=5.394979890e-05 dD_h=6.871184582e-05
032 t= 2.4000 Fpaop=0.995903475 Fhva=0.827520796 dE_tot_p=7.409058200e-06 dE_tot_h=2.050909790e-05 dE_sta_p=6.453008040e-06 dE_sta_h=3.748956700e-05 dD_p=6.077919457e-05 dD_h=8.246445527e-05
033 t= 2.4750 Fpaop=0.995903468 Fhva=0.827525653 dE_tot_p=1.835517300e-06 dE_tot_h=4.403232669e-06 dE_sta_p=1.693514363e-05 dE_sta_h=6.335345358e-05 dD_p=7.097820442e-05 dD_h=9.800594900e-05
034 t= 2.5500 Fpaop=0.995903648 Fhva=0.827531366 dE_tot_p=7.457378867e-06 dE_tot_h=1.668472637e-05 dE_sta_p=2.554361835e-05 dE_sta_h=8.579190197e-05 dD_p=8.296528994e-05 dD_h=1.132879269e-04
035 t= 2.6250 Fpaop=0.995903938 Fhva=0.827537946 dE_tot_p=1.843919031e-05 dE_tot_h=4.158584583e-05 dE_sta_p=3.051320274e-05 dE_sta_h=1.034819897e-04 dD_p=9.398609326e-05 dD_h=1.254952613e-04
036 t= 2.7000 Fpaop=0.995904275 Fhva=0.827545362 dE_tot_p=2.831784756e-05 dE_tot_h=6.789321761e-05 dE_sta_p=3.073805091e-05 dE_sta_h=1.157113958e-04 dD_p=1.010049923e-04 dD_h=1.318371688e-04
037 t= 2.7750 Fpaop=0.995904640 Fhva=0.827553507 dE_tot_p=3.436219477e-05 dE_tot_h=9.229359078e-05 dE_sta_p=2.617174451e-05 dE_sta_h=1.223586191e-04 dD_p=1.018102874e-04 dD_h=1.304935449e-04
038 t= 2.8500 Fpaop=0.995905066 Fhva=0.827562175 dE_tot_p=3.470862265e-05 dE_tot_h=1.111747318e-04 dE_sta_p=1.797073432e-05 dE_sta_h=1.238008843e-04 dD_p=9.594442708e-05 dD_h=1.214030313e-04
039 t= 2.9250 Fpaop=0.995905627 Fhva=0.827571042 dE_tot_p=2.894265885e-05 dE_tot_h=1.214036599e-04 dE_sta_p=8.304797934e-06 dE_sta_h=1.207497318e-04 dD_p=8.510932498e-05 dD_h=1.065966653e-04
040 t= 3.0000 Fpaop=0.995906403 Fhva=0.827579691 dE_tot_p=1.832070891e-05 dE_tot_h=1.211213073e-04 dE_sta_p=1.459940129e-07 dE_sta_h=1.140307505e-04 dD_p=7.285705419e-05 dD_h=8.989506682e-05
041 t= 3.0750 Fpaop=0.995907437 Fhva=0.827587656 dE_tot_p=5.570129815e-06 dE_tot_h=1.103487228e-04 dE_sta_p=4.885472312e-06 dE_sta_h=1.043550795e-04 dD_p=6.360777658e-05 dD_h=7.597047776e-05
042 t= 3.1500 Fpaop=0.995908704 Fhva=0.827594494 dE_tot_p=5.723300657e-06 dE_tot_h=9.117828977e-05 dE_sta_p=4.292629894e-06 dE_sta_h=9.214888652e-05 dD_p=6.126947487e-05 dD_h=6.898315497e-05
043 t= 3.2250 Fpaop=0.995910098 Fhva=0.827599855 dE_tot_p=1.204533366e-05 dE_tot_h=6.737816935e-05 dE_sta_p=1.893395614e-06 dE_sta_h=7.749063905e-05 dD_p=6.789546268e-05 dD_h=7.117884627e-05
044 t= 3.3000 Fpaop=0.995911439 Fhva=0.827603551 dE_tot_p=1.100089837e-05 dE_tot_h=4.339752755e-05 dE_sta_p=1.243819098e-05 dE_sta_h=6.015897973e-05 dD_p=8.284795221e-05 dD_h=8.192002778e-05
045 t= 3.3750 Fpaop=0.995912513 Fhva=0.827605578 dE_tot_p=2.188778228e-06 dE_tot_h=2.299595214e-05 dE_sta_p=2.486572687e-05 dE_sta_h=3.974896874e-05 dD_p=1.028129813e-04 dD_h=9.756736715e-05
046 t= 3.4500 Fpaop=0.995913121 Fhva=0.827606109 dE_tot_p=1.240723555e-05 dE_tot_h=7.940555846e-06 dE_sta_p=3.603229224e-05 dE_sta_h=1.580883792e-05 dD_p=1.227538066e-04 dD_h=1.124107678e-04
047 t= 3.5250 Fpaop=0.995913137 Fhva=0.827605461 dE_tot_p=2.872396761e-05 dE_tot_h=2.702272958e-06 dE_sta_p=4.287821426e-05 dE_sta_h=1.200740215e-05 dD_p=1.375674469e-04 dD_h=1.205079398e-04
048 t= 3.6000 Fpaop=0.995912544 Fhva=0.827604051 dE_tot_p=4.162512288e-05 dE_tot_h=1.229001796e-05 dE_sta_p=4.318701241e-05 dE_sta_h=4.371938275e-05 dD_p=1.439302373e-04 dD_h=1.179222851e-04
049 t= 3.6750 Fpaop=0.995911454 Fhva=0.827602370 dE_tot_p=4.636006952e-05 dE_tot_h=2.529375111e-05 dE_sta_p=3.618938761e-05 dE_sta_h=7.867066158e-05 dD_p=1.416927969e-04 dD_h=1.046013347e-04
050 t= 3.7500 Fpaop=0.995910083 Fhva=0.827600974 dE_tot_p=4.003897357e-05 dE_tot_h=4.532838688e-05 dE_sta_p=2.287633321e-05 dE_sta_h=1.150677419e-04 dD_p=1.342792087e-04 dD_h=8.513073778e-05
051 t= 3.8250 Fpaop=0.995908708 Fhva=0.827600483 dE_tot_p=2.265252293e-05 dE_tot_h=7.321420769e-05 dE_sta_p=5.927117659e-06 dE_sta_h=1.495689343e-04 dD_p=1.278475923e-04 dD_h=6.789040574e-05
052 t= 3.9000 Fpaop=0.995907595 Fhva=0.827601578 dE_tot_p=2.697564381e-06 dE_tot_h=1.058181986e-04 dE_sta_p=1.077675630e-05 dE_sta_h=1.772013936e-04 dD_p=1.293848575e-04 dD_h=6.266496165e-05
053 t= 3.9750 Fpaop=0.995906933 Fhva=0.827604954 dE_tot_p=3.043777402e-05 dE_tot_h=1.361831360e-04 dE_sta_p=2.298471864e-05 dE_sta_h=1.918704057e-04 dD_p=1.442919746e-04 dD_h=7.734585611e-05
054 t= 4.0500 Fpaop=0.995906791 Fhva=0.827611238 dE_tot_p=5.384813859e-05 dE_tot_h=1.550093304e-04 dE_sta_p=2.708070882e-05 dE_sta_h=1.875343411e-04 dD_p=1.742286142e-04 dD_h=1.147839779e-04
055 t= 4.1250 Fpaop=0.995907101 Fhva=0.827620867 dE_tot_p=6.670968445e-05 dE_tot_h=1.530885830e-04 dE_sta_p=2.099928232e-05 dE_sta_h=1.598144443e-04 dD_p=2.159493741e-04 dD_h=1.709279983e-04
056 t= 4.2000 Fpaop=0.995907693 Fhva=0.827633971 dE_tot_p=6.485543901e-05 dE_tot_h=1.239763571e-04 dE_sta_p=4.791536032e-06 dE_sta_h=1.075404797e-04 dD_p=2.615883447e-04 dD_h=2.350546797e-04
057 t= 4.2750 Fpaop=0.995908361 Fhva=0.827650288 dE_tot_p=4.726160044e-05 dE_tot_h=6.611965792e-05 dE_sta_p=1.933047393e-05 dE_sta_h=3.365510854e-05 dD_p=3.004236339e-04 dD_h=2.922531677e-04
058 t= 4.3500 Fpaop=0.995908937 Fhva=0.827669150 dE_tot_p=1.643423982e-05 dE_tot_h=1.617407174e-05 dE_sta_p=4.748226101e-05 dE_sta_h=5.492336365e-05 dD_p=3.217093671e-04 dD_h=3.275748371e-04
059 t= 4.4250 Fpaop=0.995909373 Fhva=0.827689544 dE_tot_p=2.201004927e-05 dE_tot_h=1.132175742e-04 dE_sta_p=7.497684419e-05 dE_sta_h=1.485157346e-04 dD_p=3.178268781e-04 dD_h=3.306551347e-04
060 t= 4.5000 Fpaop=0.995909773 Fhva=0.827710240 dE_tot_p=6.051868871e-05 dE_tot_h=2.117851182e-04 dE_sta_p=9.732637791e-05 dE_sta_h=2.366218960e-04 dD_p=2.868692654e-04 dD_h=2.993613652e-04
061 t= 4.5750 Fpaop=0.995910391 Fhva=0.827729950 dE_tot_p=9.132111389e-05 dE_tot_h=2.978791225e-04 dE_sta_p=1.111249035e-04 dE_sta_h=3.099929784e-04 dD_p=2.338711556e-04 dD_h=2.412180184e-04
062 t= 4.6500 Fpaop=0.995911560 Fhva=0.827747467 dE_tot_p=1.083223290e-04 dE_tot_h=3.597403677e-04 dE_sta_p=1.146394817e-04 dE_sta_h=3.622384446e-04 dD_p=1.702113461e-04 dD_h=1.719694360e-04
063 t= 4.7250 Fpaop=0.995913586 Fhva=0.827761771 dE_tot_p=1.086949650e-04 dE_tot_h=3.903285216e-04 dE_sta_p=1.080399637e-04 dE_sta_h=3.905763088e-04 dD_p=1.111843816e-04 dD_h=1.114935085e-04
064 t= 4.8000 Fpaop=0.995916625 Fhva=0.827772059 dE_tot_p=9.369278924e-05 dE_tot_h=3.886018214e-04 dE_sta_p=9.327417545e-05 dE_sta_h=3.956375892e-04 dD_p=7.225221197e-05 dD_h=7.813272297e-05
065 t= 4.8750 Fpaop=0.995920587 Fhva=0.827777743 dE_tot_p=6.836485011e-05 dE_tot_h=3.592246762e-04 dE_sta_p=7.362811570e-05 dE_sta_h=3.805104798e-04 dD_p=6.491654936e-05 dD_h=8.309536537e-05
066 t= 4.9500 Fpaop=0.995925100 Fhva=0.827778413 dE_tot_p=4.015360743e-05 dE_tot_h=3.107738352e-04 dE_sta_p=5.302354279e-05 dE_sta_h=3.493774080e-04 dD_p=9.336307042e-05 dD_h=1.266918082e-04
067 t= 5.0250 Fpaop=0.995929560 Fhva=0.827773822 dE_tot_p=1.673018468e-05 dE_tot_h=2.529762809e-04 dE_sta_p=3.513835354e-05 dE_sta_h=3.061476728e-04 dD_p=1.529240363e-04 dD_h=1.977268651e-04
068 t= 5.1000 Fpaop=0.995933268 Fhva=0.827763914 dE_tot_p=3.719330207e-06 dE_tot_h=1.938386091e-04 dE_sta_p=2.250697509e-05 dE_sta_h=2.534498055e-04 dD_p=2.309690273e-04 dD_h=2.764579590e-04
069 t= 5.1750 Fpaop=0.995935605 Fhva=0.827748893 dE_tot_p=3.061974214e-06 dE_tot_h=1.375868878e-04 dE_sta_p=1.584032598e-05 dE_sta_h=1.922517804e-04 dD_p=3.101529682e-04 dD_h=3.404053191e-04
070 t= 5.2500 Fpaop=0.995936211 Fhva=0.827729311 dE_tot_p=1.258822277e-05 dE_tot_h=8.408344416e-05 dE_sta_p=1.383172233e-05 dE_sta_h=1.222328949e-04 dD_p=3.732138488e-04 dD_h=3.713213914e-04
071 t= 5.3250 Fpaop=0.995935093 Fhva=0.827706145 dE_tot_p=2.697062609e-05 dE_tot_h=2.989512825e-05 dE_sta_p=1.362453963e-05 dE_sta_h=4.283802766e-05 dD_p=4.079566703e-04 dD_h=3.611370594e-04
072 t= 5.4000 Fpaop=0.995932636 Fhva=0.827680806 dE_tot_p=3.973955338e-05 dE_tot_h=2.937582709e-05 dE_sta_p=1.190177169e-05 dE_sta_h=4.527601679e-05 dD_p=4.108925106e-04 dD_h=3.149084178e-04
073 t= 5.4750 Fpaop=0.995929496 Fhva=0.827655062 dE_tot_p=4.565637189e-05 dE_tot_h=9.632365139e-05 dE_sta_p=6.287210654e-06 dE_sta_h=1.388796703e-04 dD_p=3.883183922e-04 dD_h=2.496675925e-04
074 t= 5.5500 Fpaop=0.995926420 Fhva=0.827630862 dE_tot_p=4.260721353e-05 dE_tot_h=1.695166588e-04 dE_sta_p=3.451893670e-06 dE_sta_h=2.314396631e-04 dD_p=3.543614052e-04 dD_h=1.893729456e-04
075 t= 5.6250 Fpaop=0.995924037 Fhva=0.827610098 dE_tot_p=3.235196382e-05 dE_tot_h=2.426332961e-04 dE_sta_p=1.492245943e-05 dE_sta_h=3.134429361e-04 dD_p=3.264544088e-04 dD_h=1.574372129e-04
076 t= 5.7000 Fpaop=0.995922696 Fhva=0.827594334 dE_tot_p=1.987200072e-05 dE_tot_h=3.053345995e-04 dE_sta_p=2.364664956e-05 dE_sta_h=3.738979216e-04 dD_p=3.195605275e-04 dD_h=1.691518160e-04
077 t= 5.7750 Fpaop=0.995922397 Fhva=0.827584574 dE_tot_p=1.155106446e-05 dE_tot_h=3.456982766e-04 dE_sta_p=2.454788924e-05 dE_sta_h=4.027385906e-04 dD_p=3.409351220e-04 dD_h=2.264279003e-04
078 t= 5.8500 Fpaop=0.995922844 Fhva=0.827581103 dE_tot_p=1.281932562e-05 dE_tot_h=3.535421953e-04 dE_sta_p=1.391163671e-05 dE_sta_h=3.935079400e-04 dD_p=3.871363736e-04 dD_h=3.165922562e-04
079 t= 5.9250 Fpaop=0.995923587 Fhva=0.827583440 dE_tot_p=2.605978063e-05 dE_tot_h=3.236691628e-04 dE_sta_p=8.881557027e-06 dE_sta_h=3.455331614e-04 dD_p=4.443719512e-04 dD_h=4.157248702e-04
080 t= 6.0000 Fpaop=0.995924218 Fhva=0.827590406 dE_tot_p=4.947532900e-05 dE_tot_h=2.580518750e-04 dE_sta_p=4.052013770e-05 dE_sta_h=2.648934322e-04 dD_p=4.922768700e-04 dD_h=4.956070486e-04
081 t= 6.0750 Fpaop=0.995924540 Fhva=0.827600293 dE_tot_p=7.729432421e-05 dE_tot_h=1.662481787e-04 dE_sta_p=7.420035773e-05 dE_sta_h=1.637976683e-04 dD_p=5.101530320e-04 dD_h=5.322167457e-04
082 t= 6.1500 Fpaop=0.995924659 Fhva=0.827611115 dE_tot_p=1.012676563e-04 dE_tot_h=6.381799003e-05 dE_sta_p=1.012802377e-04 dE_sta_h=5.842965017e-05 dD_p=4.838878211e-04 dD_h=5.132136907e-04
083 t= 6.2250 Fpaop=0.995924954 Fhva=0.827620865 dE_tot_p=1.130181243e-04 dE_tot_h=3.092622417e-05 dE_sta_p=1.136738749e-04 dE_sta_h=3.425587141e-05 dD_p=4.114609289e-04 dD_h=4.421458445e-04
084 t= 6.3000 Fpaop=0.995925951 Fhva=0.827627761 dE_tot_p=1.065568475e-04 dE_tot_h=1.010192511e-04 dE_sta_p=1.062662311e-04 dE_sta_h=1.000118014e-04 dD_p=3.052535680e-04 dD_h=3.380800090e-04
085 t= 6.3750 Fpaop=0.995928116 Fhva=0.827630422 dE_tot_p=8.024277902e-05 dE_tot_h=1.347102726e-04 dE_sta_p=7.862650361e-05 dE_sta_h=1.301951309e-04 dD_p=1.902234104e-04 dD_h=2.307080640e-04
086 t= 6.4500 Fpaop=0.995931659 Fhva=0.827627948 dE_tot_p=3.762740771e-05 dE_tot_h=1.280505962e-04 dE_sta_p=3.551883929e-05 dE_sta_h=1.233626693e-04 dD_p=9.817362251e-05 dD_h=1.522979061e-04
087 t= 6.5250 Fpaop=0.995936399 Fhva=0.827619914 dE_tot_p=1.306113800e-05 dE_tot_h=8.549399718e-05 dE_sta_p=1.393896465e-05 dE_sta_h=8.516710018e-05 dD_p=5.949954554e-05 dD_h=1.287593626e-04
088 t= 6.6000 Fpaop=0.995941756 Fhva=0.827606285 dE_tot_p=6.067517118e-05 dE_tot_h=1.835005375e-05 dE_sta_p=5.825277358e-05 dE_sta_h=2.657603828e-05 dD_p=9.460837770e-05 dD_h=1.723307932e-04
089 t= 6.6750 Fpaop=0.995946858 Fhva=0.827587301 dE_tot_p=9.370735370e-05 dE_tot_h=5.847208168e-05 dE_sta_p=8.617063863e-05 dE_sta_h=3.914111604e-05 dD_p=2.074281612e-04 dD_h=2.779151686e-04
090 t= 6.7500 Fpaop=0.995950763 Fhva=0.827563373 dE_tot_p=1.030209516e-04 dE_tot_h=1.305243317e-04 dE_sta_p=8.931213991e-05 dE_sta_h=9.989619282e-05 dD_p=3.829636885e-04 dD_h=4.240645652e-04
091 t= 6.8250 Fpaop=0.995952700 Fhva=0.827535013 dE_tot_p=8.408812746e-05 dE_tot_h=1.875772148e-04 dE_sta_p=6.419617906e-05 dE_sta_h=1.478445881e-04 dD_p=5.898195551e-04 dD_h=5.783371720e-04
092 t= 6.9000 Fpaop=0.995952276 Fhva=0.827502823 dE_tot_p=3.820340908e-05 dE_tot_h=2.260787962e-04 dE_sta_p=1.320323317e-05 dE_sta_h=1.812371181e-04 dD_p=7.872638162e-04 dD_h=7.055898763e-04
093 t= 6.9750 Fpaop=0.995949572 Fhva=0.827467540 dE_tot_p=2.761350027e-05 dE_tot_h=2.496970057e-04 dE_sta_p=5.575381206e-05 dE_sta_h=2.045795021e-04 dD_p=9.351175091e-04 dD_h=7.770268234e-04
094 t= 7.0500 Fpaop=0.995945102 Fhva=0.827430096 dE_tot_p=1.020091286e-04 dE_tot_h=2.678467663e-04 dE_sta_p=1.307953761e-04 dE_sta_h=2.270514537e-04 dD_p=1.003891990e-03 dD_h=7.776703446e-04
095 t= 7.1250 Fpaop=0.995939654 Fhva=0.827391692 dE_tot_p=1.716736524e-04 dE_tot_h=2.926161574e-04 dE_sta_p=1.985488425e-04 dE_sta_h=2.595803874e-04 dD_p=9.824189996e-04 dD_h=7.103588875e-04
096 t= 7.2000 Fpaop=0.995934061 Fhva=0.827353841 dE_tot_p=2.241719234e-04 dE_tot_h=3.348867663e-04 dE_sta_p=2.469862452e-04 dE_sta_h=3.113020581e-04 dD_p=8.808067754e-04 dD_h=5.952625946e-04
097 t= 7.2750 Fpaop=0.995928977 Fhva=0.827318368 dE_tot_p=2.505820323e-04 dE_tot_h=4.006215190e-04 dE_sta_p=2.679744119e-04 dE_sta_h=3.862944342e-04 dD_p=7.277771231e-04 dD_h=4.650047259e-04
098 t= 7.3500 Fpaop=0.995924732 Fhva=0.827287376 dE_tot_p=2.473192636e-04 dE_tot_h=4.882550580e-04 dE_sta_p=2.589232103e-04 dE_sta_h=4.814189723e-04 dD_p=5.629776774e-04 dD_h=3.565274360e-04
099 t= 7.4250 Fpaop=0.995921300 Fhva=0.827263132 dE_tot_p=2.166926573e-04 dE_tot_h=5.878938763e-04 dE_sta_p=2.231106238e-04 dE_sta_h=5.858718784e-04 dD_p=4.263005841e-04 dD_h=3.016207994e-04
100 t= 7.5000 Fpaop=0.995918383 Fhva=0.827247895 dE_tot_p=1.660372745e-04 dE_tot_h=6.826615449e-04 dE_sta_p=1.685846433e-04 dE_sta_h=6.826918950e-04 dD_p=3.471602141e-04 dD_h=3.183925864e-04
101 t= 7.5750 Fpaop=0.995915574 Fhva=0.827243653 dE_tot_p=1.056336075e-04 dE_tot_h=7.520622809e-04 dE_sta_p=1.059213996e-04 dE_sta_h=7.520530283e-04 dD_p=3.367966290e-04 dD_h=4.058238965e-04
102 t= 7.6500 Fpaop=0.995912543 Fhva=0.827251786 dE_tot_p=4.597659658e-05 dE_tot_h=7.767499090e-04 dE_sta_p=4.545482610e-05 dE_sta_h=7.757519795e-04 dD_p=3.858988281e-04 dD_h=5.429448530e-04
103 t= 7.7250 Fpaop=0.995909176 Fhva=0.827272680 dE_tot_p=4.819270517e-06 dE_tot_h=7.436673481e-04 dE_sta_p=5.222679002e-06 dE_sta_h=7.419476957e-04 dD_p=4.683540559e-04 dD_h=6.931669923e-04
104 t= 7.8000 Fpaop=0.995905644 Fhva=0.827305382 dE_tot_p=4.269078914e-05 dE_tot_h=6.502746586e-04 dE_sta_p=4.273595120e-05 dE_sta_h=6.490049767e-04 dD_p=5.501219291e-04 dD_h=8.130949768e-04
105 t= 7.8750 Fpaop=0.995902375 Fhva=0.827347399 dE_tot_p=6.784432655e-05 dE_tot_h=5.066214524e-04 dE_sta_p=6.792059221e-05 dE_sta_h=5.073236839e-04 dD_p=6.006280332e-04 dD_h=8.639512220e-04
106 t= 7.9500 Fpaop=0.995899950 Fhva=0.827394749 dE_tot_p=8.346292973e-05 dE_tot_h=3.344076071e-04 dE_sta_p=8.435077551e-05 dE_sta_h=3.383657164e-04 dD_p=6.031766197e-04 dD_h=8.228683862e-04
107 t= 8.0250 Fpaop=0.995898963 Fhva=0.827442350 dE_tot_p=9.338211838e-05 dE_tot_h=1.628873579e-04 dE_sta_p=9.593147356e-05 dE_sta_h=1.707207730e-04 dD_p=5.610258499e-04 dD_h=6.910097186e-04
108 t= 8.1000 Fpaop=0.995899865 Fhva=0.827484740 dE_tot_p=9.962772454e-05 dE_tot_h=2.235934094e-05 dE_sta_p=1.044582371e-04 dE_sta_h=3.386919689e-05 dD_p=4.969824224e-04 dD_h=4.959469346e-04
109 t= 8.1750 Fpaop=0.995902832 Fhva=0.827517007 dE_tot_p=1.008227251e-04 dE_tot_h=6.319901192e-05 dE_sta_p=1.081249238e-04 dE_sta_h=4.891229567e-05 dD_p=4.463456036e-04 dD_h=2.869663939e-04
110 t= 8.2500 Fpaop=0.995907673 Fhva=0.827535735 dE_tot_p=9.220389084e-05 dE_tot_h=8.233206340e-05 dE_sta_p=1.016758360e-04 dE_sta_h=6.657269198e-05 dD_p=4.451800295e-04 dD_h=1.237699986e-04
111 t= 8.3250 Fpaop=0.995913789 Fhva=0.827539711 dE_tot_p=6.740612594e-05 dE_tot_h=3.922861865e-05 dE_sta_p=7.831859201e-05 dE_sta_h=2.336090035e-05 dD_p=5.175321524e-04 dD_h=6.096064670e-05
112 t= 8.4000 Fpaop=0.995920195 Fhva=0.827530166 dE_tot_p=2.147082800e-05 dE_tot_h=4.741273719e-05 dE_sta_p=3.282744591e-05 dE_sta_h=6.224685892e-05 dD_p=6.657534568e-04 dD_h=1.322070648e-04
113 t= 8.4750 Fpaop=0.995925611 Fhva=0.827510449 dE_tot_p=4.601832205e-05 dE_tot_h=1.497503478e-04 dE_sta_p=3.527067612e-05 dE_sta_h=1.627944329e-04 dD_p=8.673032591e-04 dD_h=3.385560395e-04
114 t= 8.5500 Fpaop=0.995928633 Fhva=0.827485190 dE_tot_p=1.289399908e-04 dE_tot_h=2.390397046e-04 dE_sta_p=1.196972003e-04 dE_sta_h=2.499638123e-04 dD_p=1.079458107e-03 dD_h=6.446927471e-04
115 t= 8.6250 Fpaop=0.995927973 Fhva=0.827459166 dE_tot_p=2.147375217e-04 dE_tot_h=2.943593597e-04 dE_sta_p=2.075653124e-04 dE_sta_h=3.032089294e-04 dD_p=1.250839566e-03 dD_h=9.850720720e-04
116 t= 8.7000 Fpaop=0.995922735 Fhva=0.827436213 dE_tot_p=2.868681162e-04 dE_tot_h=3.090883754e-04 dE_sta_p=2.819088557e-04 dE_sta_h=3.161781842e-04 dD_p=1.336400054e-03 dD_h=1.279167359e-03
117 t= 8.7750 Fpaop=0.995912674 Fhva=0.827418518 dE_tot_p=3.289900506e-04 dE_tot_h=2.929219578e-04 dE_sta_p=3.259736006e-04 dE_sta_h=2.987050033e-04 dD_p=1.311247750e-03 dD_h=1.452324950e-03
118 t= 8.8500 Fpaop=0.995898362 Fhva=0.827406526 dE_tot_p=3.295930646e-04 dE_tot_h=2.685041765e-04 dE_sta_p=3.279511618e-04 dE_sta_h=2.734379351e-04 dD_p=1.178904990e-03 dD_h=1.456707495e-03
119 t= 8.9250 Fpaop=0.995881194 Fhva=0.827399479 dE_tot_p=2.855344228e-04 dE_tot_h=2.634520843e-04 dE_sta_p=2.845872634e-04 dE_sta_h=2.678838433e-04 dD_p=9.712547535e-04 dD_h=1.286256777e-03
120 t= 9.0000 Fpaop=0.995863187 Fhva=0.827396410 dE_tot_p=2.032553267e-04 dE_tot_h=3.001083729e-04 dE_sta_p=2.024137572e-04 dE_sta_h=3.042031121e-04 dD_p=7.400200620e-04 dD_h=9.808344259e-04
121 t= 9.0750 Fpaop=0.995846592 Fhva=0.827397221 dE_tot_p=9.722076685e-05 dE_tot_h=3.862594285e-04 dE_sta_p=9.614034769e-05 dE_sta_h=3.899889229e-04 dD_p=5.422989620e-04 dD_h=6.175110168e-04
122 t= 9.1500 Fpaop=0.995833397 Fhva=0.827403421 dE_tot_p=1.392582845e-05 dE_tot_h=5.099502879e-04 dE_sta_p=1.528571296e-05 dE_sta_h=5.131457612e-04 dD_p=4.245602940e-04 dD_h=2.906571927e-04
123 t= 9.2250 Fpaop=0.995824848 Fhva=0.827418214 dE_tot_p=1.122243064e-04 dE_tot_h=6.403932388e-04 dE_sta_p=1.136475471e-04 dE_sta_h=6.428407513e-04 dD_p=4.099722980e-04 dD_h=8.598383869e-05
124 t= 9.3000 Fpaop=0.995821130 Fhva=0.827445809 dE_tot_p=1.846900189e-04 dE_tot_h=7.351221467e-04 dE_sta_p=1.858290697e-04 dE_sta_h=7.366620550e-04 dD_p=4.928430465e-04 dD_h=5.587809688e-05
125 t= 9.3750 Fpaop=0.995821326 Fhva=0.827490132 dE_tot_p=2.257097230e-04 dE_tot_h=7.515614591e-04 dE_sta_p=2.262351542e-04 dE_sta_h=7.521521297e-04 dD_p=6.416768013e-04 dD_h=2.034903092e-04
126 t= 9.4500 Fpaop=0.995823689 Fhva=0.827553306 dE_tot_p=2.369446091e-04 dE_tot_h=6.596977347e-04 dE_sta_p=2.366629241e-04 dE_sta_h=6.594217187e-04 dD_p=8.096575435e-04 dD_h=4.808387629e-04
127 t= 9.5250 Fpaop=0.995826178 Fhva=0.827634412 dE_tot_p=2.250611016e-04 dE_tot_h=4.520585500e-04 dE_sta_p=2.239653395e-04 dE_sta_h=4.510777997e-04 dD_p=9.491413154e-04 dD_h=8.022228044e-04
128 t= 9.6000 Fpaop=0.995827093 Fhva=0.827728960 dE_tot_p=1.983616942e-04 dE_tot_h=1.479000733e-04 dE_sta_p=1.966108934e-04 dE_sta_h=1.463903390e-04 dD_p=1.025668981e-03 dD_h=1.069590921e-03
129 t= 9.6750 Fpaop=0.995825626 Fhva=0.827829317 dE_tot_p=1.636923369e-04 dE_tot_h=2.098196880e-04 dE_sta_p=1.615432547e-04 dE_sta_h=2.117188554e-04 dD_p=1.027384614e-03 dD_h=1.202619254e-03
130 t= 9.7500 Fpaop=0.995822136 Fhva=0.827926016 dE_tot_p=1.247591262e-04 dE_tot_h=5.639698324e-04 dE_sta_p=1.224856283e-04 dE_sta_h=5.661664173e-04 dD_p=9.673589159e-04 dD_h=1.164400591e-03
131 t= 9.8250 Fpaop=0.995818028 Fhva=0.828009604 dE_tot_p=8.231338096e-05 dE_tot_h=8.566405028e-04 dE_sta_p=8.014359388e-05 dE_sta_h=8.590639860e-04 dD_p=8.785792642e-04 dD_h=9.745418727e-04
132 t= 9.9000 Fpaop=0.995815264 Fhva=0.828072516 dE_tot_p=3.584728647e-05 dE_tot_h=1.042648780e-03 dE_sta_p=3.393242055e-05 dE_sta_h=1.045209095e-03 dD_p=8.035215568e-04 dD_h=7.049984830e-04
133 t= 9.9750 Fpaop=0.995815638 Fhva=0.828110454 dE_tot_p=1.418057469e-05 dE_tot_h=1.099431469e-03 dE_sta_p=1.576865702e-05 dE_sta_h=1.101991887e-03 dD_p=7.816093656e-04 dD_h=4.591607939e-04
134 t=10.0500 Fpaop=0.995820041 Fhva=0.828122908 dE_tot_p=6.474886038e-05 dE_tot_h=1.030738643e-03 dE_sta_p=6.600372122e-05 dE_sta_h=1.033121356e-03 dD_p=8.381482978e-04 dD_h=3.399887707e-04
135 t=10.1250 Fpaop=0.995827967 Fhva=0.828112746 dE_tot_p=1.101367005e-04 dE_tot_h=8.634119826e-04 dE_sta_p=1.110993285e-04 dE_sta_h=8.654354812e-04 dD_p=9.775450247e-04 dD_h=4.166881417e-04
136 t=10.2000 Fpaop=0.995837460 Fhva=0.828085080 dE_tot_p=1.433901253e-04 dE_tot_h=6.385399447e-04 dE_sta_p=1.441328997e-04 dE_sta_h=6.400697758e-04 dD_p=1.182157152e-03 dD_h=7.002888958e-04
137 t=10.2750 Fpaop=0.995845526 Fhva=0.828045830 dE_tot_p=1.587870394e-04 dE_tot_h=3.998169873e-04 dE_sta_p=1.593991578e-04 dE_sta_h=4.008052309e-04 dD_p=1.416507614e-03 dD_h=1.136129810e-03
138 t=10.3500 Fpaop=0.995848943 Fhva=0.828000462 dE_tot_p=1.542072798e-04 dE_tot_h=1.825605593e-04 dE_sta_p=1.547786760e-04 dE_sta_h=1.830563006e-04 dD_p=1.635328001e-03 dD_h=1.616295723e-03
139 t=10.4250 Fpaop=0.995845222 Fhva=0.827953275 dE_tot_p=1.322968130e-04 dE_tot_h=6.421396463e-06 dE_sta_p=1.328998225e-04 dE_sta_h=6.549244776e-06 dD_p=1.793260290e-03 dD_h=2.008976117e-03
140 t=10.5000 Fpaop=0.995833420 Fhva=0.827907372 dE_tot_p=9.978421389e-05 dE_tot_h=1.264536851e-04 dE_sta_p=1.004564998e-04 dE_sta_h=1.265359034e-04 dD_p=1.854074777e-03 dD_h=2.196397434e-03
141 t=10.5750 Fpaop=0.995814551 Fhva=0.827865208 dE_tot_p=6.507516302e-05 dE_tot_h=2.277521008e-04 dE_sta_p=6.581002772e-05 dE_sta_h=2.278986704e-04 dD_p=1.797765206e-03 dD_h=2.110118755e-03
142 t=10.6500 Fpaop=0.995791445 Fhva=0.827829366 dE_tot_p=3.500206790e-05 dE_tot_h=3.153724204e-04 dE_sta_p=3.575106186e-05 dE_sta_h=3.154825913e-04 dD_p=1.624572712e-03 dD_h=1.753021120e-03
143 t=10.7250 Fpaop=0.995768077 Fhva=0.827803173 dE_tot_p=1.202760912e-05 dE_tot_h=4.044548582e-04 dE_sta_p=1.271582328e-05 dE_sta_h=4.044857981e-04 dD_p=1.355614815e-03 dD_h=1.201154517e-03
144 t=10.8000 Fpaop=0.995748550 Fhva=0.827790813 dE_tot_p=6.891930016e-06 dE_tot_h=4.999093632e-04 dE_sta_p=6.342758630e-06 dE_sta_h=4.998703604e-04 dD_p=1.030228255e-03 dD_h=5.845819068e-04
145 t=10.8750 Fpaop=0.995736035 Fhva=0.827796801 dE_tot_p=2.916659088e-05 dE_tot_h=5.926971933e-04 dE_sta_p=2.881515473e-05 dE_sta_h=5.926314612e-04 dD_p=7.003956940e-04 dD_h=5.264167711e-05
146 t=10.9500 Fpaop=0.995731982 Fhva=0.827824924 dE_tot_p=6.340760950e-05 dE_tot_h=6.609312240e-04 dE_sta_p=6.327779009e-05 dE_sta_h=6.608926365e-04 dD_p=4.228329982e-04 dD_h=2.662995511e-04
147 t=11.0250 Fpaop=0.995735830 Fhva=0.827876975 dE_tot_p=1.153073420e-04 dE_tot_h=6.754370694e-04 dE_sta_p=1.153846083e-04 dE_sta_h=6.754676325e-04 dD_p=2.495871334e-04 dD_h=2.989391151e-04
148 t=11.1000 Fpaop=0.995745312 Fhva=0.827951697 dE_tot_p=1.840860832e-04 dE_tot_h=6.081634629e-04 dE_sta_p=1.843248744e-04 dE_sta_h=6.082756111e-04 dD_p=2.183736889e-04 dD_h=4.524178554e-05
149 t=11.1750 Fpaop=0.995757248 Fhva=0.828044325 dE_tot_p=2.609301227e-04 dE_tot_h=4.410705886e-04 dE_sta_p=2.612680460e-04 dE_sta_h=4.412375798e-04 dD_p=3.442819583e-04 dD_h=4.241568633e-04
150 t=11.2500 Fpaop=0.995768565 Fhva=0.828146941 dE_tot_p=3.302180476e-04 dE_tot_h=1.730294056e-04 dE_sta_p=3.305915075e-04 dE_sta_h=1.731865733e-04 dD_p=6.147037712e-04 dD_h=9.900605621e-04
151 t=11.3250 Fpaop=0.995777247 Fhva=0.828249610 dE_tot_p=3.733878368e-04 dE_tot_h=1.771786716e-04 dE_sta_p=3.737446734e-04 dE_sta_h=1.771206078e-04 dD_p=9.891796829e-04 dD_h=1.519383072e-03
152 t=11.4000 Fpaop=0.995782898 Fhva=0.828342060 dE_tot_p=3.743627935e-04 dE_tot_h=5.726390143e-04 dE_sta_p=3.746696112e-04 dE_sta_h=5.727706193e-04 dD_p=1.405154933e-03 dD_h=1.901986963e-03
153 t=11.4750 Fpaop=0.995786751 Fhva=0.828415470 dE_tot_p=3.248538426e-04 dE_tot_h=9.645067317e-04 dE_sta_p=3.250977795e-04 dE_sta_h=9.648925518e-04 dD_p=1.789412499e-03 dD_h=2.079046776e-03
154 t=11.5500 Fpaop=0.995791127 Fhva=0.828463921 dE_tot_p=2.277981912e-04 dE_tot_h=1.301174522e-03 dE_sta_p=2.279842109e-04 dE_sta_h=1.301830552e-03 dD_p=2.073421530e-03 dD_h=2.055187882e-03
155 t=11.6250 Fpaop=0.995798508 Fhva=0.828485162 dE_tot_p=9.770626938e-05 dE_tot_h=1.538224259e-03 dE_sta_p=9.785157659e-05 dE_sta_h=1.539108420e-03 dD_p=2.209405451e-03 dD_h=1.891700429e-03
156 t=11.7000 Fpaop=0.995810528 Fhva=0.828480540 dE_tot_p=4.238895378e-05 dE_tot_h=1.646815851e-03 dE_sta_p=4.226174904e-05 dE_sta_h=1.647835610e-03 dD_p=2.183076925e-03 dD_h=1.683733250e-03
157 t=11.7750 Fpaop=0.995827209 Fhva=0.828454167 dE_tot_p=1.666412048e-04 dE_tot_h=1.618582924e-03 dE_sta_p=1.665107132e-04 dE_sta_h=1.619618001e-03 dD_p=2.019125240e-03 dD_h=1.528855851e-03
158 t=11.8500 Fpaop=0.995846699 Fhva=0.828411629 dE_tot_p=2.531735240e-04 dE_tot_h=1.465986799e-03 dE_sta_p=2.530247608e-04 dE_sta_h=1.466919363e-03 dD_p=1.776873675e-03 dD_h=1.496488076e-03
159 t=11.9250 Fpaop=0.995865628 Fhva=0.828358632 dE_tot_p=2.899597660e-04 dE_tot_h=1.218201112e-03 dE_sta_p=2.897870246e-04 dE_sta_h=1.218943438e-03 dD_p=1.535907768e-03 dD_h=1.606967476e-03
160 t=12.0000 Fpaop=0.995880006 Fhva=0.828299967 dE_tot_p=2.780195363e-04 dE_tot_h=9.136779416e-04 dE_sta_p=2.778266346e-04 dE_sta_h=9.141890860e-04 dD_p=1.374393974e-03 dD_h=1.825849927e-03
161 t=12.0750 Fpaop=0.995886401 Fhva=0.828239057 dE_tot_p=2.308644894e-04 dE_tot_h=5.913169958e-04 dE_sta_p=2.306627226e-04 dE_sta_h=5.916046950e-04 dD_p=1.345454833e-03 dD_h=2.074495587e-03
162 t=12.1500 Fpaop=0.995883065 Fhva=0.828178138 dE_tot_p=1.701981232e-04 dE_tot_h=2.824388855e-04 dE_sta_p=1.700027929e-04 dE_sta_h=2.825483419e-04 dD_p=1.458445292e-03 dD_h=2.253458269e-03
163 t=12.2250 Fpaop=0.995870633 Fhva=0.828118926 dE_tot_p=1.189862483e-04 dE_tot_h=5.495542904e-06 dE_sta_p=1.188128458e-04 dE_sta_h=5.490708205e-06 dD_p=1.671579542e-03 dD_h=2.271919521e-03
164 t=12.3000 Fpaop=0.995852173 Fhva=0.828063447 dE_tot_p=9.386859662e-05 dE_tot_h=2.352837167e-04 dE_sta_p=9.372955862e-05 dE_sta_h=2.353395296e-04 dD_p=1.899853563e-03 dD_h=2.075157761e-03
165 t=12.3750 Fpaop=0.995832517 Fhva=0.828014697 dE_tot_p=9.921687142e-05 dE_tot_h=4.471209667e-04 dE_sta_p=9.911943033e-05 dE_sta_h=4.471796879e-04 dD_p=2.038011932e-03 dD_h=1.662943871e-03
166 t=12.4500 Fpaop=0.995817044 Fhva=0.827976824 dE_tot_p=1.248279369e-04 dE_tot_h=6.426381781e-04 dE_sta_p=1.247731866e-04 dE_sta_h=6.426741259e-04 dD_p=1.993480017e-03 dD_h=1.094342927e-03
167 t=12.5250 Fpaop=0.995810238 Fhva=0.827954722 dE_tot_p=1.483073235e-04 dE_tot_h=8.321889715e-04 dE_sta_p=1.482903891e-04 dE_sta_h=8.321986334e-04 dD_p=1.720197598e-03 dD_h=4.778423274e-04
168 t=12.6000 Fpaop=0.995814448 Fhva=0.827953129 dE_tot_p=1.418581318e-04 dE_tot_h=1.016661806e-03 dE_sta_p=1.418692748e-04 dE_sta_h=1.016658180e-03 dD_p=1.242604207e-03 dD_h=5.089077765e-05
169 t=12.6750 Fpaop=0.995829216 Fhva=0.827975482 dE_tot_p=8.179450987e-05 dE_tot_h=1.182790678e-03 dE_sta_p=8.182100749e-05 dE_sta_h=1.182794785e-03 dD_p=6.605758971e-04 dD_h=3.587216650e-04
170 t=12.7500 Fpaop=0.995851410 Fhva=0.828022901 dE_tot_p=4.194122570e-05 dE_tot_h=1.302497321e-03 dE_sta_p=4.191264695e-05 dE_sta_h=1.302529379e-03 dD_p=1.308964428e-04 dD_h=3.459147208e-04
171 t=12.8250 Fpaop=0.995876132 Fhva=0.828093586 dE_tot_p=2.193025472e-04 dE_tot_h=1.336917073e-03 dE_sta_p=2.192831038e-04 dE_sta_h=1.336989343e-03 dD_p=1.722316418e-04 dD_h=2.831891226e-05
172 t=12.9000 Fpaop=0.995898158 Fhva=0.828182846 dE_tot_p=4.191647554e-04 dE_tot_h=1.244687946e-03 dE_sta_p=4.191613781e-04 dE_sta_h=1.244800651e-03 dD_p=1.080535752e-04 dD_h=7.311206872e-04
173 t=12.9750 Fpaop=0.995913490 Fhva=0.828283702 dE_tot_p=5.948885412e-04 dE_tot_h=9.930032258e-04 dE_sta_p=5.949026133e-04 dE_sta_h=9.931442595e-04 dD_p=3.829866825e-04 dD_h=1.655845392e-03
174 t=13.0500 Fpaop=0.995920530 Fhva=0.828387886 dE_tot_p=6.955319473e-04 dE_tot_h=5.690537085e-04 dE_sta_p=6.955594016e-04 dE_sta_h=5.692019677e-04 dD_p=1.249940346e-03 dD_h=2.637743603e-03
175 t=13.1250 Fpaop=0.995920535 Fhva=0.828486930 dE_tot_p=6.799492234e-04 dE_tot_h=1.097538514e-05 dE_sta_p=6.799820025e-04 dE_sta_h=1.084412234e-05 dD_p=2.331804718e-03 dD_h=3.483848191e-03
176 t=13.2000 Fpaop=0.995917183 Fhva=0.828573050 dE_tot_p=5.298026398e-04 dE_tot_h=6.980622519e-04 dE_sta_p=5.298310912e-04 dE_sta_h=6.979687333e-04 dD_p=3.390926923e-03 dD_h=4.012429363e-03
177 t=13.2750 Fpaop=0.995915379 Fhva=0.828639687 dE_tot_p=2.574290691e-04 dE_tot_h=1.413568017e-03 dE_sta_p=2.574445668e-04 dE_sta_h=1.413524237e-03 dD_p=4.172552292e-03 dD_h=4.094673492e-03
178 t=13.3500 Fpaop=0.995919666 Fhva=0.828681673 dE_tot_p=9.430153450e-05 dE_tot_h=2.059993696e-03 dE_sta_p=9.430447508e-05 dE_sta_h=2.060000477e-03 dD_p=4.474889318e-03 dD_h=3.689482191e-03
179 t=13.4250 Fpaop=0.995932724 Fhva=0.828695226 dE_tot_p=4.607805390e-04 dE_tot_h=2.537919264e-03 dE_sta_p=4.608031612e-04 dE_sta_h=2.537967211e-03 dD_p=4.209404792e-03 dD_h=2.862209698e-03
180 t=13.5000 Fpaop=0.995954415 Fhva=0.828677957 dE_tot_p=7.711365228e-04 dE_tot_h=2.765943995e-03 dE_sta_p=7.711758952e-04 dE_sta_h=2.766017163e-03 dD_p=3.432435250e-03 dD_h=1.780329644e-03
181 t=13.5750 Fpaop=0.995981686 Fhva=0.828629104 dE_tot_p=9.662248799e-04 dE_tot_h=2.699140744e-03 dE_sta_p=9.662749469e-04 dE_sta_h=2.699221537e-03 dD_p=2.336644159e-03 dD_h=6.835430969e-04
182 t=13.6500 Fpaop=0.996009334 Fhva=0.828549987 dE_tot_p=1.013936364e-03 dE_tot_h=2.341234728e-03 dE_sta_p=1.013989586e-03 dE_sta_h=2.341308340e-03 dD_p=1.202424561e-03 dD_h=1.679231621e-04
183 t=13.7250 Fpaop=0.996031427 Fhva=0.828444550 dE_tot_p=9.171902364e-04 dE_tot_h=1.746707350e-03 dE_sta_p=9.172393612e-04 dE_sta_h=1.746764520e-03 dD_p=3.216247235e-04 dD_h=5.565290026e-04
184 t=13.8000 Fpaop=0.996042926 Fhva=0.828319678 dE_tot_p=7.120426866e-04 dE_tot_h=1.011250529e-03 dE_sta_p=7.120822581e-04 dE_sta_h=1.011288187e-03 dD_p=8.489875309e-05 dD_h=3.669002316e-04
185 t=13.8750 Fpaop=0.996041036 Fhva=0.828184989 dE_tot_p=4.563199017e-04 dE_tot_h=2.519949755e-04 dE_sta_p=4.563472031e-04 dE_sta_h=2.520151249e-04 dD_p=6.852418536e-05 dD_h=3.751146776e-04
186 t=13.9500 Fpaop=0.996025890 Fhva=0.828051899 dE_tot_p=2.121489273e-04 dE_tot_h=4.180869339e-04 dE_sta_p=2.121642136e-04 dE_sta_h=4.180792761e-04 dD_p=7.063541440e-04 dD_h=1.497260940e-03
187 t=14.0250 Fpaop=0.996000406 Fhva=0.827932059 dE_tot_p=2.774025210e-05 dE_tot_h=9.151652268e-04 dE_sta_p=2.774626269e-05 dE_sta_h=9.151641932e-04 dD_p=1.613073468e-03 dD_h=2.715202267e-03
188 t=14.1000 Fpaop=0.995969403 Fhva=0.827835455 dE_tot_p=7.590672806e-05 dE_tot_h=1.202139871e-03 dE_sta_p=7.590582441e-05 dE_sta_h=1.202140420e-03 dD_p=2.496716977e-03 dD_h=3.701242722e-03
189 t=14.1750 Fpaop=0.995938285 Fhva=0.827768691 dE_tot_p=1.092020247e-04 dE_tot_h=1.293703298e-03 dE_sta_p=1.092019595e-04 dE_sta_h=1.293702179e-03 dD_p=3.076752421e-03 dD_h=4.171568176e-03
190 t=14.2500 Fpaop=0.995911708 Fhva=0.827733968 dE_tot_p=1.077616021e-04 dE_tot_h=1.247100135e-03 dE_sta_p=1.077592367e-04 dE_sta_h=1.247096180e-03 dD_p=3.170440654e-03 dD_h=3.968749528e-03
191 t=14.3250 Fpaop=0.995892593 Fhva=0.827729058 dE_tot_p=1.178277917e-04 dE_tot_h=1.140962627e-03 dE_sta_p=1.178218972e-04 dE_sta_h=1.140956542e-03 dD_p=2.751519874e-03 dD_h=3.115068620e-03
192 t=14.4000 Fpaop=0.995881731 Fhva=0.827748310 dE_tot_p=1.780605096e-04 dE_tot_h=1.048319610e-03 dE_sta_p=1.780518875e-04 dE_sta_h=1.048313438e-03 dD_p=1.962675058e-03 dD_h=1.818421787e-03
193 t=14.4750 Fpaop=0.995878000 Fhva=0.827784355 dE_tot_p=3.036722158e-04 dE_tot_h=1.011945431e-03 dE_sta_p=3.036631690e-04 dE_sta_h=1.011941819e-03 dD_p=1.076641147e-03 dD_h=4.250403467e-04
194 t=14.5500 Fpaop=0.995879046 Fhva=0.827829934 dE_tot_p=4.782055864e-04 dE_tot_h=1.029855945e-03 dE_sta_p=4.781989651e-04 dE_sta_h=1.029857349e-03 dD_p=4.160437081e-04 dD_h=6.712247011e-04
195 t=14.6250 Fpaop=0.995882158 Fhva=0.827879269 dE_tot_p=6.558866236e-04 dE_tot_h=1.055926297e-03 dE_sta_p=6.558848008e-04 dE_sta_h=1.055934271e-03 dD_p=2.546950552e-04 dD_h=1.139381389e-03
196 t=14.7000 Fpaop=0.995885050 Fhva=0.827928518 dE_tot_p=7.741752638e-04 dE_tot_h=1.016053342e-03 dE_sta_p=7.741793846e-04 dE_sta_h=1.016068082e-03 dD_p=7.292601915e-04 dD_h=8.100771696e-04
197 t=14.7750 Fpaop=0.995886364 Fhva=0.827975249 dE_tot_p=7.729207005e-04 dE_tot_h=8.353695706e-04 dE_sta_p=7.729304163e-04 dE_sta_h=8.353898329e-04 dD_p=1.788051467e-03 dD_h=2.674516266e-04
198 t=14.8500 Fpaop=0.995885778 Fhva=0.828017192 dE_tot_p=6.144058229e-04 dE_tot_h=4.682795057e-04 dE_sta_p=6.144195263e-04 dE_sta_h=4.683029201e-04 dD_p=3.193674592e-03 dD_h=1.829656584e-03
199 t=14.9250 Fpaop=0.995883799 Fhva=0.828050860 dE_tot_p=2.981747743e-04 dE_tot_h=7.822876815e-05 dE_sta_p=2.981901758e-04 dE_sta_h=7.820507991e-05 dD_p=4.581075857e-03 dD_h=3.466144538e-03
200 t=15.0000 Fpaop=0.995881343 Fhva=0.828070675 dE_tot_p=1.340229420e-04 dE_tot_h=7.359294320e-04 dE_sta_p=1.340081403e-04 dE_sta_h=7.359081445e-04 dD_p=5.556476070e-03 dD_h=4.733221860e-03
```

## Appendix B. Warm-Start B Export ADAPT Selected Trace (All Steps)

```text
depth=001 mode=gradient family=hopdrag idx=3 max_grad=2.293880939e-01 E_before=0.264627254921 E_after=0.264032654672 dE_step=-5.946002481e-04 label=paop_lf_std:paop_hopdrag(0,1)
depth=002 mode=gradient family=disp idx=1 max_grad=1.609843096e-01 E_before=0.264032654672 E_after=0.258884641668 dE_step=-5.148013005e-03 label=paop_lf_std:paop_disp(site=1)
depth=003 mode=gradient family=disp idx=2 max_grad=1.108736814e-01 E_before=0.258884641668 E_after=0.254206169570 dE_step=-4.678472098e-03 label=paop_lf_std:paop_disp(site=2)
depth=004 mode=gradient family=curdrag idx=5 max_grad=1.069388852e-01 E_before=0.254206169570 E_after=0.254105813118 dE_step=-1.003564525e-04 label=paop_lf_std:paop_curdrag(0,1)
depth=005 mode=gradient family=curdrag idx=6 max_grad=9.466724343e-02 E_before=0.254105813118 E_after=0.254027833828 dE_step=-7.797928947e-05 label=paop_lf_std:paop_curdrag(1,2)
depth=006 mode=gradient family=disp idx=0 max_grad=7.975291907e-02 E_before=0.254027833828 E_after=0.251517008583 dE_step=-2.510825245e-03 label=paop_lf_std:paop_disp(site=0)
depth=007 mode=gradient family=hopdrag idx=4 max_grad=7.084184353e-02 E_before=0.251517008583 E_after=0.251461078746 dE_step=-5.592983746e-05 label=paop_lf_std:paop_hopdrag(1,2)
depth=008 mode=gradient family=hopdrag idx=3 max_grad=4.815410454e-03 E_before=0.251461078746 E_after=0.251459293373 dE_step=-1.785373319e-06 label=paop_lf_std:paop_hopdrag(0,1)
depth=009 mode=gradient family=curdrag idx=5 max_grad=4.351332993e-03 E_before=0.251459293373 E_after=0.251457304010 dE_step=-1.989362698e-06 label=paop_lf_std:paop_curdrag(0,1)
depth=010 mode=gradient family=curdrag idx=5 max_grad=7.098741228e-03 E_before=0.251457304010 E_after=0.251456719424 dE_step=-5.845856542e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=011 mode=gradient family=curdrag idx=5 max_grad=5.649897881e-03 E_before=0.251456719424 E_after=0.251456106931 dE_step=-6.124934949e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=012 mode=gradient family=hopdrag idx=4 max_grad=5.537000373e-03 E_before=0.251456106931 E_after=0.251455396671 dE_step=-7.102600691e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=013 mode=gradient family=curdrag idx=6 max_grad=1.922185909e-03 E_before=0.251455396671 E_after=0.251455260267 dE_step=-1.364041285e-07 label=paop_lf_std:paop_curdrag(1,2)
depth=014 mode=gradient family=hopdrag idx=3 max_grad=4.130503132e-03 E_before=0.251455260267 E_after=0.251454606729 dE_step=-6.535379444e-07 label=paop_lf_std:paop_hopdrag(0,1)
depth=015 mode=gradient family=curdrag idx=5 max_grad=4.033638655e-03 E_before=0.251454606729 E_after=0.251454344049 dE_step=-2.626795578e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=016 mode=gradient family=curdrag idx=5 max_grad=4.098075252e-03 E_before=0.251454344049 E_after=0.251454570611 dE_step=2.265619132e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=017 mode=gradient family=curdrag idx=5 max_grad=3.667503595e-03 E_before=0.251454570611 E_after=0.251454648497 dE_step=7.788557238e-08 label=paop_lf_std:paop_curdrag(0,1)
depth=018 mode=gradient family=curdrag idx=5 max_grad=6.600165233e-03 E_before=0.251454648497 E_after=0.251454024693 dE_step=-6.238033625e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=019 mode=gradient family=curdrag idx=5 max_grad=3.676968487e-03 E_before=0.251454024693 E_after=0.251454048004 dE_step=2.331078613e-08 label=paop_lf_std:paop_curdrag(0,1)
depth=020 mode=gradient family=curdrag idx=5 max_grad=3.616833232e-03 E_before=0.251454048004 E_after=0.251453873601 dE_step=-1.744030256e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=021 mode=gradient family=hopdrag idx=3 max_grad=3.447147954e-03 E_before=0.251453873601 E_after=0.251453295542 dE_step=-5.780588389e-07 label=paop_lf_std:paop_hopdrag(0,1)
depth=022 mode=gradient family=hopdrag idx=4 max_grad=2.623661975e-03 E_before=0.251453295542 E_after=0.251454003532 dE_step=7.079902229e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=023 mode=gradient family=curdrag idx=5 max_grad=9.470066821e-03 E_before=0.251454003532 E_after=0.251454383338 dE_step=3.798057653e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=024 mode=gradient family=curdrag idx=5 max_grad=1.053508541e-02 E_before=0.251454383338 E_after=0.251453719050 dE_step=-6.642877409e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=025 mode=gradient family=curdrag idx=5 max_grad=9.352872262e-03 E_before=0.251453719050 E_after=0.251452527631 dE_step=-1.191419211e-06 label=paop_lf_std:paop_curdrag(0,1)
depth=026 mode=gradient family=curdrag idx=5 max_grad=2.724262213e-03 E_before=0.251452527631 E_after=0.251453492632 dE_step=9.650005561e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=027 mode=gradient family=curdrag idx=5 max_grad=1.256951115e-02 E_before=0.251453492632 E_after=0.251452495969 dE_step=-9.966628518e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=028 mode=gradient family=curdrag idx=5 max_grad=8.128554948e-03 E_before=0.251452495969 E_after=0.251451861376 dE_step=-6.345924876e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=029 mode=gradient family=hopdrag idx=3 max_grad=2.392972668e-03 E_before=0.251451861376 E_after=0.251451890505 dE_step=2.912843794e-08 label=paop_lf_std:paop_hopdrag(0,1)
depth=030 mode=gradient family=curdrag idx=5 max_grad=2.049325736e-03 E_before=0.251451890505 E_after=0.251454382504 dE_step=2.491999328e-06 label=paop_lf_std:paop_curdrag(0,1)
depth=031 mode=gradient family=curdrag idx=5 max_grad=1.732195102e-02 E_before=0.251454382504 E_after=0.251451151627 dE_step=-3.230877333e-06 label=paop_lf_std:paop_curdrag(0,1)
depth=032 mode=gradient family=curdrag idx=5 max_grad=4.402640968e-03 E_before=0.251451151627 E_after=0.251450974124 dE_step=-1.775027499e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=033 mode=gradient family=hopdrag idx=3 max_grad=2.706988316e-03 E_before=0.251450974124 E_after=0.251450949382 dE_step=-2.474237382e-08 label=paop_lf_std:paop_hopdrag(0,1)
depth=034 mode=gradient family=curdrag idx=5 max_grad=4.849658045e-03 E_before=0.251450949382 E_after=0.251452292354 dE_step=1.342972683e-06 label=paop_lf_std:paop_curdrag(0,1)
depth=035 mode=gradient family=curdrag idx=5 max_grad=1.782041148e-02 E_before=0.251452292354 E_after=0.251450634475 dE_step=-1.657879667e-06 label=paop_lf_std:paop_curdrag(0,1)
depth=036 mode=gradient family=curdrag idx=5 max_grad=1.156375287e-02 E_before=0.251450634475 E_after=0.251449321617 dE_step=-1.312857855e-06 label=paop_lf_std:paop_curdrag(0,1)
depth=037 mode=gradient family=hopdrag idx=4 max_grad=3.056310603e-03 E_before=0.251449321617 E_after=0.251449339155 dE_step=1.753774798e-08 label=paop_lf_std:paop_hopdrag(1,2)
depth=038 mode=gradient family=curdrag idx=5 max_grad=3.758635009e-03 E_before=0.251449339155 E_after=0.251449174156 dE_step=-1.649986006e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=039 mode=gradient family=curdrag idx=5 max_grad=6.881303741e-03 E_before=0.251449174156 E_after=0.251448878990 dE_step=-2.951664149e-07 label=paop_lf_std:paop_curdrag(0,1)
depth=040 mode=gradient family=curdrag idx=5 max_grad=5.238825441e-03 E_before=0.251448878990 E_after=0.251451145741 dE_step=2.266751176e-06 label=paop_lf_std:paop_curdrag(0,1)
depth=041 mode=gradient family=curdrag idx=5 max_grad=1.763416261e-02 E_before=0.251451145741 E_after=0.251448620766 dE_step=-2.524974855e-06 label=paop_lf_std:paop_curdrag(0,1)
depth=042 mode=gradient family=hopdrag idx=4 max_grad=3.126371211e-03 E_before=0.251448620766 E_after=0.251448233530 dE_step=-3.872356432e-07 label=paop_lf_std:paop_hopdrag(1,2)
```

## Appendix C. Accessibility B/C Energy Histories

### Accessibility Rung B History (len=44)

```text
B step=000 E=0.264627254921
B step=001 E=0.264032654672
B step=002 E=0.258884641668
B step=003 E=0.254206169570
B step=004 E=0.254105813118
B step=005 E=0.254027833828
B step=006 E=0.251517008583
B step=007 E=0.251461078746
B step=008 E=0.251459293373
B step=009 E=0.251457304010
B step=010 E=0.251456719424
B step=011 E=0.251456106931
B step=012 E=0.251455396671
B step=013 E=0.251455260267
B step=014 E=0.251454606729
B step=015 E=0.251454344049
B step=016 E=0.251454570611
B step=017 E=0.251454648497
B step=018 E=0.251454024693
B step=019 E=0.251454048004
B step=020 E=0.251453873601
B step=021 E=0.251453295542
B step=022 E=0.251454003532
B step=023 E=0.251454383338
B step=024 E=0.251453719050
B step=025 E=0.251452527631
B step=026 E=0.251453492632
B step=027 E=0.251452495969
B step=028 E=0.251451861376
B step=029 E=0.251451890505
B step=030 E=0.251454382504
B step=031 E=0.251451151627
B step=032 E=0.251450974124
B step=033 E=0.251450949382
B step=034 E=0.251452292354
B step=035 E=0.251450634475
B step=036 E=0.251449321617
B step=037 E=0.251449339155
B step=038 E=0.251449174156
B step=039 E=0.251448878990
B step=040 E=0.251451145741
B step=041 E=0.251448620766
B step=042 E=0.251448233530
B step=043 E=0.251448285999
```

### Accessibility Rung C History (len=39)

```text
C step=000 E=0.257282636997
C step=001 E=0.256658895227
C step=002 E=0.256340718422
C step=003 E=0.256038308823
C step=004 E=0.255912040452
C step=005 E=0.253181250489
C step=006 E=0.250458798104
C step=007 E=0.249393467034
C step=008 E=0.249385651128
C step=009 E=0.249379134530
C step=010 E=0.249378554323
C step=011 E=0.249374513122
C step=012 E=0.249374705888
C step=013 E=0.249370992201
C step=014 E=0.249367114697
C step=015 E=0.249367081285
C step=016 E=0.249365235613
C step=017 E=0.249362907132
C step=018 E=0.249360887408
C step=019 E=0.249344827243
C step=020 E=0.249344757670
C step=021 E=0.249344399867
C step=022 E=0.249343466542
C step=023 E=0.249342422303
C step=024 E=0.249342028043
C step=025 E=0.249342531945
C step=026 E=0.249341140783
C step=027 E=0.249340891387
C step=028 E=0.249340904886
C step=029 E=0.249340844249
C step=030 E=0.249340644678
C step=031 E=0.249339255672
C step=032 E=0.249338450560
C step=033 E=0.249338105363
C step=034 E=0.249339363392
C step=035 E=0.249337002065
C step=036 E=0.249336465441
C step=037 E=0.249335873522
C step=038 E=0.249333999503
```

## Appendix D. Trend Experiment Energy and Gradient Traces

### A_medium (A_uccsd_plus_paop_medium)

```text
A_medium idx=000 E_trace=5.500000000000 grad_max=2.000000000e+00
A_medium idx=001 E_trace=4.500000004791 grad_max=2.000000000e+00
A_medium idx=002 E_trace=1.015564701401 grad_max=4.689267408e-01
A_medium idx=003 E_trace=0.777917786839 grad_max=5.144681954e-01
A_medium idx=004 E_trace=0.535287817246 grad_max=2.716489021e-01
A_medium idx=005 E_trace=0.530439902139 grad_max=2.101001432e-01
A_medium idx=006 E_trace=0.520680642831 grad_max=1.198018325e-01
A_medium idx=007 E_trace=0.514504126578 grad_max=1.198107814e-01
A_medium idx=008 E_trace=0.508312278736 grad_max=2.997232502e-02
A_medium idx=009 E_trace=0.504310225033 grad_max=9.650104682e-02
A_medium idx=010 E_trace=0.421282784421 grad_max=2.333061935e-01
A_medium idx=011 E_trace=0.345160014763 grad_max=2.216665297e-01
A_medium idx=012 E_trace=0.309295234616 grad_max=2.316546025e-01
A_medium idx=013 E_trace=0.294247783461 grad_max=1.488488883e-01
A_medium idx=014 E_trace=0.251543098758 grad_max=6.416392928e-02
A_medium idx=015 E_trace=0.248814216405 grad_max=5.234121692e-02
A_medium idx=016 E_trace=0.246531249223 grad_max=3.291641450e-02
A_medium idx=017 E_trace=0.245258138393 grad_max=3.316727240e-03
A_medium idx=018 E_trace=0.245206429036 grad_max=5.158892421e-03
A_medium idx=019 E_trace=0.245204235933 grad_max=4.563828566e-03
A_medium idx=020 E_trace=0.245202940355 grad_max=nan
```

### A_heavy (A_uccsd_plus_paop_heavy)

```text
A_heavy idx=000 E_trace=5.500000000000 grad_max=2.000000000e+00
A_heavy idx=001 E_trace=4.500000004791 grad_max=2.000000000e+00
A_heavy idx=002 E_trace=1.015564701401 grad_max=4.689267408e-01
A_heavy idx=003 E_trace=0.777917786839 grad_max=5.144681954e-01
A_heavy idx=004 E_trace=0.535287817246 grad_max=2.716489021e-01
A_heavy idx=005 E_trace=0.530439902139 grad_max=2.101001432e-01
A_heavy idx=006 E_trace=0.520680642831 grad_max=1.198018325e-01
A_heavy idx=007 E_trace=0.514504126578 grad_max=1.198107814e-01
A_heavy idx=008 E_trace=0.508312278736 grad_max=2.997232502e-02
A_heavy idx=009 E_trace=0.504310225033 grad_max=9.650104682e-02
A_heavy idx=010 E_trace=0.420674991950 grad_max=2.431388140e-01
A_heavy idx=011 E_trace=0.419624903223 grad_max=2.172496392e-01
A_heavy idx=012 E_trace=0.350267044609 grad_max=2.328306950e-01
A_heavy idx=013 E_trace=0.308429817508 grad_max=1.970364464e-01
A_heavy idx=014 E_trace=0.291491023009 grad_max=1.549592865e-01
A_heavy idx=015 E_trace=0.251658572229 grad_max=7.130977007e-02
A_heavy idx=016 E_trace=0.248188668154 grad_max=4.846304788e-02
A_heavy idx=017 E_trace=0.246504829481 grad_max=3.232254181e-02
A_heavy idx=018 E_trace=0.245255562496 grad_max=1.074727065e-02
A_heavy idx=019 E_trace=0.245232233777 grad_max=6.558860658e-03
A_heavy idx=020 E_trace=0.245228992646 grad_max=2.835205647e-03
A_heavy idx=021 E_trace=0.245218555430 grad_max=2.266011525e-03
A_heavy idx=022 E_trace=0.245216722186 grad_max=3.636154131e-03
A_heavy idx=023 E_trace=0.245216639965 grad_max=2.718311583e-03
A_heavy idx=024 E_trace=0.245215207620 grad_max=5.934594929e-03
A_heavy idx=025 E_trace=0.245214528835 grad_max=7.207446020e-03
A_heavy idx=026 E_trace=0.245213795552 grad_max=3.588232159e-03
A_heavy idx=027 E_trace=0.245211925221 grad_max=6.085148935e-03
A_heavy idx=028 E_trace=0.245210958634 grad_max=3.023789407e-03
A_heavy idx=029 E_trace=0.245210061801 grad_max=3.595696001e-03
A_heavy idx=030 E_trace=0.245209263294 grad_max=2.092585475e-03
A_heavy idx=031 E_trace=0.245207728677 grad_max=2.230874734e-03
A_heavy idx=032 E_trace=0.245208458410 grad_max=1.041215149e-02
A_heavy idx=033 E_trace=0.245205469640 grad_max=4.225013820e-03
A_heavy idx=034 E_trace=0.245206393564 grad_max=1.282004431e-02
A_heavy idx=035 E_trace=0.245204196358 grad_max=7.721555576e-03
A_heavy idx=036 E_trace=0.245203648021 grad_max=nan
```

### B_medium (B_uccsd_plus_paop_plus_hva_medium)

```text
B_medium idx=000 E_trace=5.500000000000 grad_max=2.000000000e+00
B_medium idx=001 E_trace=4.500000004791 grad_max=2.000000000e+00
B_medium idx=002 E_trace=1.015564701401 grad_max=4.689267408e-01
B_medium idx=003 E_trace=0.777917786839 grad_max=5.144681954e-01
B_medium idx=004 E_trace=0.535287817246 grad_max=2.716489021e-01
B_medium idx=005 E_trace=0.530439902139 grad_max=2.101001432e-01
B_medium idx=006 E_trace=0.520680642831 grad_max=1.198018325e-01
B_medium idx=007 E_trace=0.514504126578 grad_max=1.198107814e-01
B_medium idx=008 E_trace=0.508312278736 grad_max=2.997232502e-02
B_medium idx=009 E_trace=0.504310225033 grad_max=9.650104682e-02
B_medium idx=010 E_trace=0.421282784421 grad_max=2.333061935e-01
B_medium idx=011 E_trace=0.345160014763 grad_max=2.216665297e-01
B_medium idx=012 E_trace=0.309295234616 grad_max=2.316546025e-01
B_medium idx=013 E_trace=0.294247783461 grad_max=1.488488883e-01
B_medium idx=014 E_trace=0.251543098758 grad_max=6.416392928e-02
B_medium idx=015 E_trace=0.248814216405 grad_max=5.234121692e-02
B_medium idx=016 E_trace=0.246531249223 grad_max=3.291641450e-02
B_medium idx=017 E_trace=0.245258138393 grad_max=3.316727240e-03
B_medium idx=018 E_trace=0.245206429036 grad_max=5.158892421e-03
B_medium idx=019 E_trace=0.245204235933 grad_max=4.563828566e-03
B_medium idx=020 E_trace=0.245202940355 grad_max=nan
```

### B_heavy (B_uccsd_plus_paop_plus_hva_heavy)

```text
B_heavy idx=000 E_trace=5.500000000000 grad_max=2.000000000e+00
B_heavy idx=001 E_trace=4.500000004791 grad_max=2.000000000e+00
B_heavy idx=002 E_trace=1.015564701401 grad_max=4.689267408e-01
B_heavy idx=003 E_trace=0.777917786839 grad_max=5.144681954e-01
B_heavy idx=004 E_trace=0.535287817246 grad_max=2.716489021e-01
B_heavy idx=005 E_trace=0.530439902139 grad_max=2.101001432e-01
B_heavy idx=006 E_trace=0.520680642831 grad_max=1.198018325e-01
B_heavy idx=007 E_trace=0.514504126578 grad_max=1.198107814e-01
B_heavy idx=008 E_trace=0.508312278736 grad_max=2.997232502e-02
B_heavy idx=009 E_trace=0.504310225033 grad_max=9.650104682e-02
B_heavy idx=010 E_trace=0.420674991950 grad_max=2.431388140e-01
B_heavy idx=011 E_trace=0.419624903223 grad_max=2.172496392e-01
B_heavy idx=012 E_trace=0.350267044609 grad_max=2.328306950e-01
B_heavy idx=013 E_trace=0.308429817508 grad_max=1.970364464e-01
B_heavy idx=014 E_trace=0.291491023009 grad_max=1.549592865e-01
B_heavy idx=015 E_trace=0.251658572229 grad_max=7.130977007e-02
B_heavy idx=016 E_trace=0.248188668154 grad_max=4.846304788e-02
B_heavy idx=017 E_trace=0.246504829481 grad_max=3.232254181e-02
B_heavy idx=018 E_trace=0.245255562496 grad_max=1.074727065e-02
B_heavy idx=019 E_trace=0.245232233777 grad_max=6.558860658e-03
B_heavy idx=020 E_trace=0.245228992646 grad_max=2.835205647e-03
B_heavy idx=021 E_trace=0.245218555430 grad_max=2.266011525e-03
B_heavy idx=022 E_trace=0.245216722186 grad_max=3.636154131e-03
B_heavy idx=023 E_trace=0.245216639965 grad_max=2.718311583e-03
B_heavy idx=024 E_trace=0.245215207620 grad_max=5.934594929e-03
B_heavy idx=025 E_trace=0.245214528835 grad_max=7.207446020e-03
B_heavy idx=026 E_trace=0.245213795552 grad_max=3.588232159e-03
B_heavy idx=027 E_trace=0.245211925221 grad_max=6.085148935e-03
B_heavy idx=028 E_trace=0.245210958634 grad_max=3.023789407e-03
B_heavy idx=029 E_trace=0.245210061801 grad_max=3.595696001e-03
B_heavy idx=030 E_trace=0.245209263294 grad_max=2.092585475e-03
B_heavy idx=031 E_trace=0.245207728677 grad_max=2.230874734e-03
B_heavy idx=032 E_trace=0.245208458410 grad_max=1.041215149e-02
B_heavy idx=033 E_trace=0.245205469640 grad_max=4.225013820e-03
B_heavy idx=034 E_trace=0.245206393564 grad_max=1.282004431e-02
B_heavy idx=035 E_trace=0.245204196358 grad_max=7.721555576e-03
B_heavy idx=036 E_trace=0.245203648021 grad_max=nan
```

## Appendix E. Trend Selected-Operator Trace Summaries

### A_medium selected_trace (len=20)

```text
depth=001 src=uccsd idx=1 max_grad=2.000000000e+00 E_before=5.500000000000 E_after=4.500000004791 dE=-9.999999952e-01 label=uccsd_ferm_lifted::uccsd_sing(alpha:1->2)
depth=002 src=uccsd idx=2 max_grad=2.000000000e+00 E_before=4.500000004791 E_after=1.015564701401 dE=-3.484435303e+00 label=uccsd_ferm_lifted::uccsd_sing(beta:3->4)
depth=003 src=uccsd idx=0 max_grad=4.689267408e-01 E_before=1.015564701401 E_after=0.777917786839 dE=-2.376469146e-01 label=uccsd_ferm_lifted::uccsd_sing(alpha:0->2)
depth=004 src=uccsd idx=3 max_grad=5.144681954e-01 E_before=0.777917786839 E_after=0.535287817246 dE=-2.426299696e-01 label=uccsd_ferm_lifted::uccsd_sing(beta:3->5)
depth=005 src=uccsd idx=6 max_grad=2.716489021e-01 E_before=0.535287817246 E_after=0.530439902139 dE=-4.847915107e-03 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=006 src=paop idx=9 max_grad=2.101001432e-01 E_before=0.530439902139 E_after=0.520680642831 dE=-9.759259308e-03 label=paop_lf_std:paop_disp(site=1)
depth=007 src=paop idx=10 max_grad=1.198018325e-01 E_before=0.520680642831 E_after=0.514504126578 dE=-6.176516253e-03 label=paop_lf_std:paop_disp(site=2)
depth=008 src=paop idx=8 max_grad=1.198107814e-01 E_before=0.514504126578 E_after=0.508312278736 dE=-6.191847842e-03 label=paop_lf_std:paop_disp(site=0)
depth=009 src=uccsd idx=7 max_grad=2.997232502e-02 E_before=0.508312278736 E_after=0.504310225033 dE=-4.002053703e-03 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,5)
depth=010 src=uccsd idx=6 max_grad=9.650104682e-02 E_before=0.504310225033 E_after=0.421282784421 dE=-8.302744061e-02 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=011 src=uccsd idx=1 max_grad=2.333061935e-01 E_before=0.421282784421 E_after=0.345160014763 dE=-7.612276966e-02 label=uccsd_ferm_lifted::uccsd_sing(alpha:1->2)
depth=012 src=uccsd idx=6 max_grad=2.216665297e-01 E_before=0.345160014763 E_after=0.309295234616 dE=-3.586478015e-02 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=013 src=paop idx=12 max_grad=2.316546025e-01 E_before=0.309295234616 E_after=0.294247783461 dE=-1.504745116e-02 label=paop_lf_std:paop_hopdrag(1,2)
depth=014 src=uccsd idx=3 max_grad=1.488488883e-01 E_before=0.294247783461 E_after=0.251543098758 dE=-4.270468470e-02 label=uccsd_ferm_lifted::uccsd_sing(beta:3->5)
depth=015 src=paop idx=10 max_grad=6.416392928e-02 E_before=0.251543098758 E_after=0.248814216405 dE=-2.728882353e-03 label=paop_lf_std:paop_disp(site=2)
depth=016 src=paop idx=8 max_grad=5.234121692e-02 E_before=0.248814216405 E_after=0.246531249223 dE=-2.282967182e-03 label=paop_lf_std:paop_disp(site=0)
depth=017 src=paop idx=9 max_grad=3.291641450e-02 E_before=0.246531249223 E_after=0.245258138393 dE=-1.273110830e-03 label=paop_lf_std:paop_disp(site=1)
depth=018 src=paop idx=9 max_grad=3.316727240e-03 E_before=0.245258138393 E_after=0.245206429036 dE=-5.170935656e-05 label=paop_lf_std:paop_disp(site=1)
depth=019 src=paop idx=12 max_grad=5.158892421e-03 E_before=0.245206429036 E_after=0.245204235933 dE=-2.193103832e-06 label=paop_lf_std:paop_hopdrag(1,2)
depth=020 src=paop idx=11 max_grad=4.563828566e-03 E_before=0.245204235933 E_after=0.245202940355 dE=-1.295577200e-06 label=paop_lf_std:paop_hopdrag(0,1)
```

### A_heavy selected_trace (len=36)

```text
depth=001 src=uccsd idx=1 max_grad=2.000000000e+00 E_before=5.500000000000 E_after=4.500000004791 dE=-9.999999952e-01 label=uccsd_ferm_lifted::uccsd_sing(alpha:1->2)
depth=002 src=uccsd idx=2 max_grad=2.000000000e+00 E_before=4.500000004791 E_after=1.015564701401 dE=-3.484435303e+00 label=uccsd_ferm_lifted::uccsd_sing(beta:3->4)
depth=003 src=uccsd idx=0 max_grad=4.689267408e-01 E_before=1.015564701401 E_after=0.777917786839 dE=-2.376469146e-01 label=uccsd_ferm_lifted::uccsd_sing(alpha:0->2)
depth=004 src=uccsd idx=3 max_grad=5.144681954e-01 E_before=0.777917786839 E_after=0.535287817246 dE=-2.426299696e-01 label=uccsd_ferm_lifted::uccsd_sing(beta:3->5)
depth=005 src=uccsd idx=6 max_grad=2.716489021e-01 E_before=0.535287817246 E_after=0.530439902139 dE=-4.847915107e-03 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=006 src=paop idx=9 max_grad=2.101001432e-01 E_before=0.530439902139 E_after=0.520680642831 dE=-9.759259308e-03 label=paop_lf_std:paop_disp(site=1)
depth=007 src=paop idx=10 max_grad=1.198018325e-01 E_before=0.520680642831 E_after=0.514504126578 dE=-6.176516253e-03 label=paop_lf_std:paop_disp(site=2)
depth=008 src=paop idx=8 max_grad=1.198107814e-01 E_before=0.514504126578 E_after=0.508312278736 dE=-6.191847842e-03 label=paop_lf_std:paop_disp(site=0)
depth=009 src=uccsd idx=7 max_grad=2.997232502e-02 E_before=0.508312278736 E_after=0.504310225033 dE=-4.002053703e-03 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,5)
depth=010 src=uccsd idx=6 max_grad=9.650104682e-02 E_before=0.504310225033 E_after=0.420674991950 dE=-8.363523308e-02 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=011 src=paop idx=12 max_grad=2.431388140e-01 E_before=0.420674991950 E_after=0.419624903223 dE=-1.050088726e-03 label=paop_lf_std:paop_hopdrag(1,2)
depth=012 src=uccsd idx=1 max_grad=2.172496392e-01 E_before=0.419624903223 E_after=0.350267044609 dE=-6.935785861e-02 label=uccsd_ferm_lifted::uccsd_sing(alpha:1->2)
depth=013 src=uccsd idx=6 max_grad=2.328306950e-01 E_before=0.350267044609 E_after=0.308429817508 dE=-4.183722710e-02 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=014 src=paop idx=12 max_grad=1.970364464e-01 E_before=0.308429817508 E_after=0.291491023009 dE=-1.693879450e-02 label=paop_lf_std:paop_hopdrag(1,2)
depth=015 src=uccsd idx=3 max_grad=1.549592865e-01 E_before=0.291491023009 E_after=0.251658572229 dE=-3.983245078e-02 label=uccsd_ferm_lifted::uccsd_sing(beta:3->5)
depth=016 src=paop idx=8 max_grad=7.130977007e-02 E_before=0.251658572229 E_after=0.248188668154 dE=-3.469904075e-03 label=paop_lf_std:paop_disp(site=0)
depth=017 src=paop idx=10 max_grad=4.846304788e-02 E_before=0.248188668154 E_after=0.246504829481 dE=-1.683838673e-03 label=paop_lf_std:paop_disp(site=2)
depth=018 src=paop idx=9 max_grad=3.232254181e-02 E_before=0.246504829481 E_after=0.245255562496 dE=-1.249266984e-03 label=paop_lf_std:paop_disp(site=1)
depth=019 src=paop idx=12 max_grad=1.074727065e-02 E_before=0.245255562496 E_after=0.245232233777 dE=-2.332871948e-05 label=paop_lf_std:paop_hopdrag(1,2)
depth=020 src=paop idx=11 max_grad=6.558860658e-03 E_before=0.245232233777 E_after=0.245228992646 dE=-3.241130687e-06 label=paop_lf_std:paop_hopdrag(0,1)
depth=021 src=uccsd idx=6 max_grad=2.835205647e-03 E_before=0.245228992646 E_after=0.245218555430 dE=-1.043721635e-05 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=022 src=paop idx=12 max_grad=2.266011525e-03 E_before=0.245218555430 E_after=0.245216722186 dE=-1.833244371e-06 label=paop_lf_std:paop_hopdrag(1,2)
depth=023 src=paop idx=12 max_grad=3.636154131e-03 E_before=0.245216722186 E_after=0.245216639965 dE=-8.222027947e-08 label=paop_lf_std:paop_hopdrag(1,2)
depth=024 src=uccsd idx=6 max_grad=2.718311583e-03 E_before=0.245216639965 E_after=0.245215207620 dE=-1.432344932e-06 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=025 src=paop idx=12 max_grad=5.934594929e-03 E_before=0.245215207620 E_after=0.245214528835 dE=-6.787853744e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=026 src=paop idx=12 max_grad=7.207446020e-03 E_before=0.245214528835 E_after=0.245213795552 dE=-7.332826185e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=027 src=paop idx=12 max_grad=3.588232159e-03 E_before=0.245213795552 E_after=0.245211925221 dE=-1.870331226e-06 label=paop_lf_std:paop_hopdrag(1,2)
depth=028 src=paop idx=12 max_grad=6.085148935e-03 E_before=0.245211925221 E_after=0.245210958634 dE=-9.665872238e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=029 src=paop idx=12 max_grad=3.023789407e-03 E_before=0.245210958634 E_after=0.245210061801 dE=-8.968325829e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=030 src=paop idx=12 max_grad=3.595696001e-03 E_before=0.245210061801 E_after=0.245209263294 dE=-7.985074866e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=031 src=paop idx=11 max_grad=2.092585475e-03 E_before=0.245209263294 E_after=0.245207728677 dE=-1.534617378e-06 label=paop_lf_std:paop_hopdrag(0,1)
depth=032 src=paop idx=12 max_grad=2.230874734e-03 E_before=0.245207728677 E_after=0.245208458410 dE=7.297332639e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=033 src=paop idx=12 max_grad=1.041215149e-02 E_before=0.245208458410 E_after=0.245205469640 dE=-2.988769296e-06 label=paop_lf_std:paop_hopdrag(1,2)
depth=034 src=paop idx=12 max_grad=4.225013820e-03 E_before=0.245205469640 E_after=0.245206393564 dE=9.239235986e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=035 src=paop idx=12 max_grad=1.282004431e-02 E_before=0.245206393564 E_after=0.245204196358 dE=-2.197205708e-06 label=paop_lf_std:paop_hopdrag(1,2)
depth=036 src=paop idx=12 max_grad=7.721555576e-03 E_before=0.245204196358 E_after=0.245203648021 dE=-5.483373370e-07 label=paop_lf_std:paop_hopdrag(1,2)
```

### B_medium selected_trace (len=20)

```text
depth=001 src=uccsd idx=1 max_grad=2.000000000e+00 E_before=5.500000000000 E_after=4.500000004791 dE=-9.999999952e-01 label=uccsd_ferm_lifted::uccsd_sing(alpha:1->2)
depth=002 src=uccsd idx=2 max_grad=2.000000000e+00 E_before=4.500000004791 E_after=1.015564701401 dE=-3.484435303e+00 label=uccsd_ferm_lifted::uccsd_sing(beta:3->4)
depth=003 src=uccsd idx=0 max_grad=4.689267408e-01 E_before=1.015564701401 E_after=0.777917786839 dE=-2.376469146e-01 label=uccsd_ferm_lifted::uccsd_sing(alpha:0->2)
depth=004 src=uccsd idx=3 max_grad=5.144681954e-01 E_before=0.777917786839 E_after=0.535287817246 dE=-2.426299696e-01 label=uccsd_ferm_lifted::uccsd_sing(beta:3->5)
depth=005 src=uccsd idx=6 max_grad=2.716489021e-01 E_before=0.535287817246 E_after=0.530439902139 dE=-4.847915107e-03 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=006 src=paop idx=9 max_grad=2.101001432e-01 E_before=0.530439902139 E_after=0.520680642831 dE=-9.759259308e-03 label=paop_lf_std:paop_disp(site=1)
depth=007 src=paop idx=10 max_grad=1.198018325e-01 E_before=0.520680642831 E_after=0.514504126578 dE=-6.176516253e-03 label=paop_lf_std:paop_disp(site=2)
depth=008 src=paop idx=8 max_grad=1.198107814e-01 E_before=0.514504126578 E_after=0.508312278736 dE=-6.191847842e-03 label=paop_lf_std:paop_disp(site=0)
depth=009 src=uccsd idx=7 max_grad=2.997232502e-02 E_before=0.508312278736 E_after=0.504310225033 dE=-4.002053703e-03 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,5)
depth=010 src=uccsd idx=6 max_grad=9.650104682e-02 E_before=0.504310225033 E_after=0.421282784421 dE=-8.302744061e-02 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=011 src=uccsd idx=1 max_grad=2.333061935e-01 E_before=0.421282784421 E_after=0.345160014763 dE=-7.612276966e-02 label=uccsd_ferm_lifted::uccsd_sing(alpha:1->2)
depth=012 src=uccsd idx=6 max_grad=2.216665297e-01 E_before=0.345160014763 E_after=0.309295234616 dE=-3.586478015e-02 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=013 src=paop idx=12 max_grad=2.316546025e-01 E_before=0.309295234616 E_after=0.294247783461 dE=-1.504745116e-02 label=paop_lf_std:paop_hopdrag(1,2)
depth=014 src=uccsd idx=3 max_grad=1.488488883e-01 E_before=0.294247783461 E_after=0.251543098758 dE=-4.270468470e-02 label=uccsd_ferm_lifted::uccsd_sing(beta:3->5)
depth=015 src=paop idx=10 max_grad=6.416392928e-02 E_before=0.251543098758 E_after=0.248814216405 dE=-2.728882353e-03 label=paop_lf_std:paop_disp(site=2)
depth=016 src=paop idx=8 max_grad=5.234121692e-02 E_before=0.248814216405 E_after=0.246531249223 dE=-2.282967182e-03 label=paop_lf_std:paop_disp(site=0)
depth=017 src=paop idx=9 max_grad=3.291641450e-02 E_before=0.246531249223 E_after=0.245258138393 dE=-1.273110830e-03 label=paop_lf_std:paop_disp(site=1)
depth=018 src=paop idx=9 max_grad=3.316727240e-03 E_before=0.245258138393 E_after=0.245206429036 dE=-5.170935656e-05 label=paop_lf_std:paop_disp(site=1)
depth=019 src=paop idx=12 max_grad=5.158892421e-03 E_before=0.245206429036 E_after=0.245204235933 dE=-2.193103832e-06 label=paop_lf_std:paop_hopdrag(1,2)
depth=020 src=paop idx=11 max_grad=4.563828566e-03 E_before=0.245204235933 E_after=0.245202940355 dE=-1.295577200e-06 label=paop_lf_std:paop_hopdrag(0,1)
```

### B_heavy selected_trace (len=36)

```text
depth=001 src=uccsd idx=1 max_grad=2.000000000e+00 E_before=5.500000000000 E_after=4.500000004791 dE=-9.999999952e-01 label=uccsd_ferm_lifted::uccsd_sing(alpha:1->2)
depth=002 src=uccsd idx=2 max_grad=2.000000000e+00 E_before=4.500000004791 E_after=1.015564701401 dE=-3.484435303e+00 label=uccsd_ferm_lifted::uccsd_sing(beta:3->4)
depth=003 src=uccsd idx=0 max_grad=4.689267408e-01 E_before=1.015564701401 E_after=0.777917786839 dE=-2.376469146e-01 label=uccsd_ferm_lifted::uccsd_sing(alpha:0->2)
depth=004 src=uccsd idx=3 max_grad=5.144681954e-01 E_before=0.777917786839 E_after=0.535287817246 dE=-2.426299696e-01 label=uccsd_ferm_lifted::uccsd_sing(beta:3->5)
depth=005 src=uccsd idx=6 max_grad=2.716489021e-01 E_before=0.535287817246 E_after=0.530439902139 dE=-4.847915107e-03 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=006 src=paop idx=9 max_grad=2.101001432e-01 E_before=0.530439902139 E_after=0.520680642831 dE=-9.759259308e-03 label=paop_lf_std:paop_disp(site=1)
depth=007 src=paop idx=10 max_grad=1.198018325e-01 E_before=0.520680642831 E_after=0.514504126578 dE=-6.176516253e-03 label=paop_lf_std:paop_disp(site=2)
depth=008 src=paop idx=8 max_grad=1.198107814e-01 E_before=0.514504126578 E_after=0.508312278736 dE=-6.191847842e-03 label=paop_lf_std:paop_disp(site=0)
depth=009 src=uccsd idx=7 max_grad=2.997232502e-02 E_before=0.508312278736 E_after=0.504310225033 dE=-4.002053703e-03 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,5)
depth=010 src=uccsd idx=6 max_grad=9.650104682e-02 E_before=0.504310225033 E_after=0.420674991950 dE=-8.363523308e-02 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=011 src=paop idx=12 max_grad=2.431388140e-01 E_before=0.420674991950 E_after=0.419624903223 dE=-1.050088726e-03 label=paop_lf_std:paop_hopdrag(1,2)
depth=012 src=uccsd idx=1 max_grad=2.172496392e-01 E_before=0.419624903223 E_after=0.350267044609 dE=-6.935785861e-02 label=uccsd_ferm_lifted::uccsd_sing(alpha:1->2)
depth=013 src=uccsd idx=6 max_grad=2.328306950e-01 E_before=0.350267044609 E_after=0.308429817508 dE=-4.183722710e-02 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=014 src=paop idx=12 max_grad=1.970364464e-01 E_before=0.308429817508 E_after=0.291491023009 dE=-1.693879450e-02 label=paop_lf_std:paop_hopdrag(1,2)
depth=015 src=uccsd idx=3 max_grad=1.549592865e-01 E_before=0.291491023009 E_after=0.251658572229 dE=-3.983245078e-02 label=uccsd_ferm_lifted::uccsd_sing(beta:3->5)
depth=016 src=paop idx=8 max_grad=7.130977007e-02 E_before=0.251658572229 E_after=0.248188668154 dE=-3.469904075e-03 label=paop_lf_std:paop_disp(site=0)
depth=017 src=paop idx=10 max_grad=4.846304788e-02 E_before=0.248188668154 E_after=0.246504829481 dE=-1.683838673e-03 label=paop_lf_std:paop_disp(site=2)
depth=018 src=paop idx=9 max_grad=3.232254181e-02 E_before=0.246504829481 E_after=0.245255562496 dE=-1.249266984e-03 label=paop_lf_std:paop_disp(site=1)
depth=019 src=paop idx=12 max_grad=1.074727065e-02 E_before=0.245255562496 E_after=0.245232233777 dE=-2.332871948e-05 label=paop_lf_std:paop_hopdrag(1,2)
depth=020 src=paop idx=11 max_grad=6.558860658e-03 E_before=0.245232233777 E_after=0.245228992646 dE=-3.241130687e-06 label=paop_lf_std:paop_hopdrag(0,1)
depth=021 src=uccsd idx=6 max_grad=2.835205647e-03 E_before=0.245228992646 E_after=0.245218555430 dE=-1.043721635e-05 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=022 src=paop idx=12 max_grad=2.266011525e-03 E_before=0.245218555430 E_after=0.245216722186 dE=-1.833244371e-06 label=paop_lf_std:paop_hopdrag(1,2)
depth=023 src=paop idx=12 max_grad=3.636154131e-03 E_before=0.245216722186 E_after=0.245216639965 dE=-8.222027947e-08 label=paop_lf_std:paop_hopdrag(1,2)
depth=024 src=uccsd idx=6 max_grad=2.718311583e-03 E_before=0.245216639965 E_after=0.245215207620 dE=-1.432344932e-06 label=uccsd_ferm_lifted::uccsd_dbl(ab:1,3->2,4)
depth=025 src=paop idx=12 max_grad=5.934594929e-03 E_before=0.245215207620 E_after=0.245214528835 dE=-6.787853744e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=026 src=paop idx=12 max_grad=7.207446020e-03 E_before=0.245214528835 E_after=0.245213795552 dE=-7.332826185e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=027 src=paop idx=12 max_grad=3.588232159e-03 E_before=0.245213795552 E_after=0.245211925221 dE=-1.870331226e-06 label=paop_lf_std:paop_hopdrag(1,2)
depth=028 src=paop idx=12 max_grad=6.085148935e-03 E_before=0.245211925221 E_after=0.245210958634 dE=-9.665872238e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=029 src=paop idx=12 max_grad=3.023789407e-03 E_before=0.245210958634 E_after=0.245210061801 dE=-8.968325829e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=030 src=paop idx=12 max_grad=3.595696001e-03 E_before=0.245210061801 E_after=0.245209263294 dE=-7.985074866e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=031 src=paop idx=11 max_grad=2.092585475e-03 E_before=0.245209263294 E_after=0.245207728677 dE=-1.534617378e-06 label=paop_lf_std:paop_hopdrag(0,1)
depth=032 src=paop idx=12 max_grad=2.230874734e-03 E_before=0.245207728677 E_after=0.245208458410 dE=7.297332639e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=033 src=paop idx=12 max_grad=1.041215149e-02 E_before=0.245208458410 E_after=0.245205469640 dE=-2.988769296e-06 label=paop_lf_std:paop_hopdrag(1,2)
depth=034 src=paop idx=12 max_grad=4.225013820e-03 E_before=0.245205469640 E_after=0.245206393564 dE=9.239235986e-07 label=paop_lf_std:paop_hopdrag(1,2)
depth=035 src=paop idx=12 max_grad=1.282004431e-02 E_before=0.245206393564 E_after=0.245204196358 dE=-2.197205708e-06 label=paop_lf_std:paop_hopdrag(1,2)
depth=036 src=paop idx=12 max_grad=7.721555576e-03 E_before=0.245204196358 E_after=0.245203648021 dE=-5.483373370e-07 label=paop_lf_std:paop_hopdrag(1,2)
```



## Appendix F. Compact Full 201-Step Branch Timeline (Single-Line Rows)

Columns: idx, t, F_paop, F_hva, dE_total_paop, dE_total_hva, dE_static_paop, dE_static_hva, dD_paop, dD_hva, stag_paop(ex,tr), stag_hva(ex,tr).

```text
000 t= 0.0000 Fp=0.995909195 Fh=0.827522051 dEtp=0.000000000e+00 dEth=0.000000000e+00 dEsp=0.000000000e+00 dEsh=0.000000000e+00 dDp=0.000000000e+00 dDh=0.000000000e+00 stagp=(0.33341813,0.33341813) stagh=(0.33090048,0.33090048)
001 t= 0.0750 Fp=0.995909195 Fh=0.827522049 dEtp=5.259161995e-10 dEth=1.029989194e-08 dEsp=4.234793627e-10 dEsh=9.991203087e-09 dDp=6.446228384e-09 dDh=4.031415313e-09 stagp=(0.33317586,0.33317586) stagh=(0.32974746,0.32974746)
002 t= 0.1500 Fp=0.995909194 Fh=0.827522033 dEtp=1.818480944e-08 dEth=8.361978848e-08 dEsp=2.214350092e-08 dEsh=6.728772128e-08 dDp=9.380592464e-08 dDh=6.629052046e-08 stagp=(0.33183406,0.33183407) stagh=(0.32807360,0.32807364)
003 t= 0.2250 Fp=0.995909194 Fh=0.827521991 dEtp=9.321699851e-08 dEth=2.679473363e-07 dEsp=1.405324260e-07 dEsh=1.346424694e-07 dDp=4.219158476e-07 dDh=3.162415047e-07 stagp=(0.32844226,0.32844233) stagh=(0.32509669,0.32509690)
004 t= 0.3000 Fp=0.995909198 Fh=0.827521918 dEtp=2.523839935e-07 dEth=5.664528209e-07 dEsp=4.773994414e-07 dEsh=2.232414464e-08 dDp=1.140388703e-06 dDh=8.976292562e-07 stagp=(0.32240621,0.32240647) stagh=(0.32028225,0.32028289)
005 t= 0.3750 Fp=0.995909200 Fh=0.827521821 dEtp=4.747453728e-07 dEth=9.228107520e-07 dEsp=1.166822459e-06 dEsh=5.912496404e-07 dDp=2.287115731e-06 dDh=1.893653892e-06 stagp=(0.31360776,0.31360844) stagh=(0.31344727,0.31344877)
006 t= 0.4500 Fp=0.995909193 Fh=0.827521712 dEtp=6.942662646e-07 dEth=1.222814086e-06 dEsp=2.314355582e-06 dEsh=2.047214015e-06 dDp=3.748029539e-06 dDh=3.284047311e-06 stagp=(0.30245212,0.30245351) stagh=(0.30479495,0.30479777)
007 t= 0.5250 Fp=0.995909163 Fh=0.827521607 dEtp=8.485133023e-07 dEth=1.308841384e-06 dEsp=3.960282197e-06 dEsh=4.545000225e-06 dDp=5.307617827e-06 dDh=4.956588358e-06 stagp=(0.28982596,0.28982839) stagh=(0.29488206,0.29488663)
008 t= 0.6000 Fp=0.995909102 Fh=0.827521514 dEtp=9.511406966e-07 dEth=1.005845066e-06 dEsp=6.051094301e-06 dEsh=8.000772439e-06 dDp=6.768044472e-06 dDh=6.765493285e-06 stagp=(0.27696624,0.27696996) stagh=(0.28453450,0.28454107)
009 t= 0.6750 Fp=0.995909009 Fh=0.827521434 dEtp=1.141264606e-06 dEth=1.589000044e-07 dEsp=8.424698758e-06 dEsh=1.199473008e-05 dDp=8.070314775e-06 dDh=8.602435915e-06 stagp=(0.26526148,0.26526658) stagh=(0.27473258,0.27474109)
010 t= 0.7500 Fp=0.995908898 Fh=0.827521355 dEtp=1.662910886e-06 dEth=1.319579442e-06 dEsp=1.081937904e-05 dEsh=1.584694319e-05 dDp=9.340307108e-06 dDh=1.043707514e-05 stagp=(0.25602762,0.25603393) stagh=(0.26648653,0.26649654)
011 t= 0.8250 Fp=0.995908791 Fh=0.827521261 dEtp=2.758897424e-06 dEth=3.399185739e-06 dEsp=1.291302719e-05 dEsh=1.880738731e-05 dDp=1.081660310e-05 dDh=1.229889472e-05 stagp=(0.25030833,0.25031538) stagh=(0.26071717,0.26072787)
012 t= 0.9000 Fp=0.995908714 Fh=0.827521138 dEtp=4.512977988e-06 dEth=5.888030310e-06 dEsp=1.438420436e-05 dEsh=2.028819991e-05 dDp=1.268193108e-05 dDh=1.420456683e-05 stagp=(0.24874107,0.24874814) stagh=(0.25815050,0.25816081)
013 t= 0.9750 Fp=0.995908683 Fh=0.827520975 dEtp=6.714308952e-06 dEth=8.409864406e-06 dEsp=1.497131928e-05 dEsh=2.003560367e-05 dDp=1.488165390e-05 dDh=1.606864014e-05 stagp=(0.25150747,0.25151372) stagh=(0.25923177,0.25924057)
014 t= 1.0500 Fp=0.995908703 Fh=0.827520768 dEtp=8.823529706e-06 dEth=1.041602598e-05 dEsp=1.450648655e-05 dEsh=1.815824586e-05 dDp=1.703515287e-05 dDh=1.765193776e-05 stagp=(0.25835993,0.25836460) stagh=(0.26406479,0.26407115)
015 t= 1.1250 Fp=0.995908759 Fh=0.827520505 dEtp=1.008319018e-05 dEth=1.124415065e-05 dEsp=1.291977851e-05 dEsh=1.499451812e-05 dDp=1.851255696e-05 dDh=1.859118524e-05 stagp=(0.26869782,0.26870042) stagh=(0.27238438,0.27238782)
016 t= 1.2000 Fp=0.995908826 Fh=0.827520164 dEtp=9.753830009e-06 dEth=1.022814285e-05 dEsp=1.023208530e-05 dEsh=1.088149555e-05 dDp=1.867327393e-05 dDh=1.851813084e-05 stagp=(0.28166243,0.28166294) stagh=(0.28356963,0.28357033)
017 t= 1.2750 Fp=0.995908873 Fh=0.827519702 dEtp=7.400055592e-06 dEth=6.849729008e-06 dEsp=6.559167645e-06 dEsh=5.939336113e-06 dDp=1.718042506e-05 dDh=1.723207653e-05 stagp=(0.29623022,0.29622912) stagh=(0.29670239,0.29670120)
018 t= 1.3500 Fp=0.995908879 Fh=0.827519063 dEtp=3.122574394e-06 dEth=9.027589042e-07 dEsp=2.129168713e-06 dEsh=1.997610516e-08 dDp=1.425639239e-05 dDh=1.485657999e-05 stagp=(0.31129916,0.31129744) stagh=(0.31066822,0.31066662)
019 t= 1.4250 Fp=0.995908836 Fh=0.827518188 dEtp=2.359307409e-06 dEth=7.376366905e-06 dEsp=2.712202644e-06 dEsh=7.407305075e-06 dDp=1.075307438e-05 dDh=1.190521156e-05 stagp=(0.32577539,0.32577448) stagh=(0.32428920,0.32428912)
020 t= 1.5000 Fp=0.995908747 Fh=0.827517042 dEtp=7.830043666e-06 dEth=1.726582180e-05 dEsp=7.542939914e-06 dEsh=1.661979487e-05 dDp=7.974508348e-06 dDh=9.207917291e-06 stagp=(0.33866843,0.33866997) stagh=(0.33647148,0.33647494)
021 t= 1.5750 Fp=0.995908623 Fh=0.827515633 dEtp=1.189924312e-05 dEth=2.764442422e-05 dEsp=1.196121070e-05 dEsh=2.774055237e-05 dDp=7.285293114e-06 dDh=7.701186427e-06 stagp=(0.34919438,0.34920002) stagh=(0.34634636,0.34635511)
022 t= 1.6500 Fp=0.995908467 Fh=0.827514031 dEtp=1.344600489e-05 dEth=3.721310652e-05 dEsp=1.570049647e-05 dEsh=4.028649024e-05 dDp=9.630133221e-06 dDh=8.143847264e-06 stagp=(0.35687293,0.35688401) stagh=(0.35338385,0.35339896)
023 t= 1.7250 Fp=0.995908268 Fh=0.827512370 dEtp=1.204721880e-05 dEth=4.480671907e-05 dEsp=1.869533490e-05 dEsh=5.310095929e-05 dDp=1.514273303e-05 dDh=1.086287719e-05 stagp=(0.36159372,0.36161094) stagh=(0.35745934,0.35748083)
024 t= 1.8000 Fp=0.995907996 Fh=0.827510832 dEtp=8.220819400e-06 dEth=4.970046046e-05 dEsp=2.104079804e-05 dEsh=6.444426805e-05 dDp=2.301197469e-05 dDh=1.563939025e-05 stagp=(0.36362762,0.36365074) stagh=(0.35886038,0.35888697)
025 t= 1.8750 Fp=0.995907613 Fh=0.827509619 dEtp=3.346156600e-06 dEth=5.179095158e-05 dEsp=2.284416931e-05 dEsh=7.227785508e-05 dDp=3.169475643e-05 dDh=2.180389368e-05 stagp=(0.36357059,0.36359824) stagh=(0.35822915,0.35825820)
026 t= 1.9500 Fp=0.995907089 Fh=0.827508924 dEtp=7.720171837e-07 dEth=5.155675449e-05 dEsp=2.403656728e-05 dEsh=7.467969889e-05 dDp=3.943760678e-05 dDh=2.852976017e-05 stagp=(0.36222715,0.36225685) stagh=(0.35644824,0.35647593)
027 t= 2.0250 Fp=0.995906426 Fh=0.827508898 dEtp=2.577571297e-06 dEth=4.978104296e-05 dEsp=2.425207782e-05 dEsh=7.028318360e-05 dDp=4.493654439e-05 dDh=3.522229395e-05 stagp=(0.36045967,0.36048815) stagh=(0.35448914,0.35451091)
028 t= 2.1000 Fp=0.995905665 Fh=0.827509639 dEtp=1.423119283e-06 dEth=4.712464909e-05 dEsp=2.286394740e-05 dEsh=5.861509534e-05 dDp=4.788213108e-05 dDh=4.183807804e-05 stagp=(0.35903968,0.35906341) stagh=(0.35325163,0.35326286)
029 t= 2.1750 Fp=0.995904891 Fh=0.827511194 dEtp=2.073100393e-06 dEth=4.372127915e-05 dEsp=1.919709382e-05 dEsh=4.023317113e-05 dDp=4.914964837e-05 dDh=4.897160881e-05 stagp=(0.35853386,0.35854974) stagh=(0.35342350,0.35342026)
030 t= 2.2500 Fp=0.995904212 Fh=0.827513574 dEtp=6.163095011e-06 dEth=3.897964923e-05 dEsp=1.284959352e-05 dEsh=1.662713078e-05 dDp=5.051321227e-05 dDh=5.762543137e-05 stagp=(0.35924235,0.35924839) stagh=(0.35538329,0.35536309)
031 t= 2.3250 Fp=0.995903721 Fh=0.827516774 dEtp=8.582007800e-06 dEth=3.170329653e-05 dEsp=3.996805440e-06 dEsh=1.007425372e-05 dDp=5.394979890e-05 dDh=6.871184582e-05 stagp=(0.36118834,0.36118421) stagh=(0.35915654,0.35911886)
032 t= 2.4000 Fp=0.995903475 Fh=0.827520796 dEtp=7.409058200e-06 dEth=2.050909790e-05 dEsp=6.453008040e-06 dEsh=3.748956700e-05 dDp=6.077919457e-05 dDh=8.246445527e-05 stagp=(0.36414374,0.36413096) stagh=(0.36442306,0.36436961)
033 t= 2.4750 Fp=0.995903468 Fh=0.827525653 dEtp=1.835517300e-06 dEth=4.403232669e-06 dEsp=1.693514363e-05 dEsh=6.335345358e-05 dDp=7.097820442e-05 dDh=9.800594900e-05 stagp=(0.36767294,0.36765485) stagh=(0.37056452,0.37049920)
034 t= 2.5500 Fp=0.995903648 Fh=0.827531366 dEtp=7.457378867e-06 dEth=1.668472637e-05 dEsp=2.554361835e-05 dEsh=8.579190197e-05 dDp=8.296528994e-05 dDh=1.132879269e-04 stagp=(0.37118373,0.37116504) stagh=(0.37674055,0.37666913)
035 t= 2.6250 Fp=0.995903938 Fh=0.827537946 dEtp=1.843919031e-05 dEth=4.158584583e-05 dEsp=3.051320274e-05 dEsh=1.034819897e-04 dDp=9.398609326e-05 dDh=1.254952613e-04 stagp=(0.37398550,0.37397176) stagh=(0.38198531,0.38191486)
036 t= 2.7000 Fp=0.995904275 Fh=0.827545362 dEtp=2.831784756e-05 dEth=6.789321761e-05 dEsp=3.073805091e-05 dEsh=1.157113958e-04 dDp=1.010049923e-04 dDh=1.318371688e-04 stagp=(0.37536262,0.37535949) stagh=(0.38532126,0.38525941)
037 t= 2.7750 Fp=0.995904640 Fh=0.827553507 dEtp=3.436219477e-05 dEth=9.229359078e-05 dEsp=2.617174451e-05 dEsh=1.223586191e-04 dDp=1.018102874e-04 dDh=1.304935449e-04 stagp=(0.37466863,0.37468114) stagh=(0.38588721,0.38584127)
038 t= 2.8500 Fp=0.995905066 Fh=0.827562175 dEtp=3.470862265e-05 dEth=1.111747318e-04 dEsp=1.797073432e-05 dEsh=1.238008843e-04 dDp=9.594442708e-05 dDh=1.214030313e-04 stagp=(0.37143703,0.37146885) stagh=(0.38307272,0.38304871)
039 t= 2.9250 Fp=0.995905627 Fh=0.827571042 dEtp=2.894265885e-05 dEth=1.214036599e-04 dEsp=8.304797934e-06 dEsh=1.207497318e-04 dDp=8.510932498e-05 dDh=1.065966653e-04 stagp=(0.36548998,0.36554270) stagh=(0.37664112,0.37664280)
040 t= 3.0000 Fp=0.995906403 Fh=0.827579691 dEtp=1.832070891e-05 dEth=1.211213073e-04 dEsp=1.459940129e-07 dEsh=1.140307505e-04 dDp=7.285705419e-05 dDh=8.989506682e-05 stagp=(0.35701629,0.35708893) stagh=(0.36681531,0.36684320)
041 t= 3.0750 Fp=0.995907437 Fh=0.827587656 dEtp=5.570129815e-06 dEth=1.103487228e-04 dEsp=4.885472312e-06 dEsh=1.043550795e-04 dDp=6.360777658e-05 dDh=7.597047776e-05 stagp=(0.34658986,0.34667863) stagh=(0.35429800,0.35434888)
042 t= 3.1500 Fp=0.995908704 Fh=0.827594494 dEtp=5.723300657e-06 dEth=9.117828977e-05 dEsp=4.292629894e-06 dEsh=9.214888652e-05 dDp=6.126947487e-05 dDh=6.898315497e-05 stagp=(0.33511152,0.33520996) stagh=(0.34020691,0.34027370)
043 t= 3.2250 Fp=0.995910098 Fh=0.827599855 dEtp=1.204533366e-05 dEth=6.737816935e-05 dEsp=1.893395614e-06 dEsh=7.749063905e-05 dDp=6.789546268e-05 dDh=7.117884627e-05 stagp=(0.32367742,0.32377716) stagh=(0.32592321,0.32599557)
044 t= 3.3000 Fp=0.995911439 Fh=0.827603551 dEtp=1.100089837e-05 dEth=4.339752755e-05 dEsp=1.243819098e-05 dEsh=6.015897973e-05 dDp=8.284795221e-05 dDh=8.192002778e-05 stagp=(0.31339972,0.31349157) stagh=(0.31287581,0.31294149)
045 t= 3.3750 Fp=0.995912513 Fh=0.827605578 dEtp=2.188778228e-06 dEth=2.299595214e-05 dEsp=2.486572687e-05 dEsh=3.974896874e-05 dDp=1.028129813e-04 dDh=9.756736715e-05 stagp=(0.30522076,0.30529622) stagh=(0.30230530,0.30235203)
046 t= 3.4500 Fp=0.995913121 Fh=0.827606109 dEtp=1.240723555e-05 dEth=7.940555846e-06 dEsp=3.603229224e-05 dEsh=1.580883792e-05 dDp=1.227538066e-04 dDh=1.124107678e-04 stagp=(0.29976525,0.29981800) stagh=(0.29506191,0.29507948)
047 t= 3.5250 Fp=0.995913137 Fh=0.827605461 dEtp=2.872396761e-05 dEth=2.702272958e-06 dEsp=4.287821426e-05 dEsh=1.200740215e-05 dDp=1.375674469e-04 dDh=1.205079398e-04 stagp=(0.29726370,0.29729082) stagh=(0.29148560,0.29146777)
048 t= 3.6000 Fp=0.995912544 Fh=0.827604051 dEtp=4.162512288e-05 dEth=1.229001796e-05 dEsp=4.318701241e-05 dEsh=4.371938275e-05 dDp=1.439302373e-04 dDh=1.179222851e-04 stagp=(0.29755859,0.29756129) stagh=(0.29139491,0.29134067)
049 t= 3.6750 Fp=0.995911454 Fh=0.827602370 dEtp=4.636006952e-05 dEth=2.529375111e-05 dEsp=3.618938761e-05 dEsh=7.867066158e-05 dDp=1.416927969e-04 dDh=1.046013347e-04 stagp=(0.30018081,0.30016442) stagh=(0.29417967,0.29409361)
050 t= 3.7500 Fp=0.995910083 Fh=0.827600974 dEtp=4.003897357e-05 dEth=4.532838688e-05 dEsp=2.287633321e-05 dEsh=1.150677419e-04 dDp=1.342792087e-04 dDh=8.513073778e-05 stagp=(0.30446631,0.30443967) stagh=(0.29896453,0.29885627)
051 t= 3.8250 Fp=0.995908708 Fh=0.827600483 dEtp=2.265252293e-05 dEth=7.321420769e-05 dEsp=5.927117659e-06 dEsh=1.495689343e-04 dDp=1.278475923e-04 dDh=6.789040574e-05 stagp=(0.30967814,0.30965247) stagh=(0.30479366,0.30467648)
052 t= 3.9000 Fp=0.995907595 Fh=0.827601578 dEtp=2.697564381e-06 dEth=1.058181986e-04 dEsp=1.077675630e-05 dEsh=1.772013936e-04 dDp=1.293848575e-04 dDh=6.266496165e-05 stagp=(0.31510684,0.31509428) stagh=(0.31078947,0.31067853)
053 t= 3.9750 Fp=0.995906933 Fh=0.827604954 dEtp=3.043777402e-05 dEth=1.361831360e-04 dEsp=2.298471864e-05 dEsh=1.918704057e-04 dDp=1.442919746e-04 dDh=7.734585611e-05 stagp=(0.32013799,0.32015000) stagh=(0.31625591,0.31616619)
054 t= 4.0500 Fp=0.995906791 Fh=0.827611238 dEtp=5.384813859e-05 dEth=1.550093304e-04 dEsp=2.708070882e-05 dEsh=1.875343411e-04 dDp=1.742286142e-04 dDh=1.147839779e-04 stagp=(0.32429106,0.32433683) stagh=(0.32072106,0.32066544)
055 t= 4.1250 Fp=0.995907101 Fh=0.827620867 dEtp=6.670968445e-05 dEth=1.530885830e-04 dEsp=2.099928232e-05 dEsh=1.598144443e-04 dDp=2.159493741e-04 dDh=1.709279983e-04 stagp=(0.32724221,0.32732722) stagh=(0.32393443,0.32392192)
056 t= 4.2000 Fp=0.995907693 Fh=0.827633971 dEtp=6.485543901e-05 dEth=1.239763571e-04 dEsp=4.791536032e-06 dEsh=1.075404797e-04 dDp=2.615883447e-04 dDh=2.350546797e-04 stagp=(0.32884229,0.32896713) stagh=(0.32584275,0.32587691)
057 t= 4.2750 Fp=0.995908361 Fh=0.827650288 dEtp=4.726160044e-05 dEth=6.611965792e-05 dEsp=1.933047393e-05 dEsh=3.365510854e-05 dDp=3.004236339e-04 dDh=2.922531677e-04 stagp=(0.32913246,0.32929214) stagh=(0.32656333,0.32664117)
058 t= 4.3500 Fp=0.995908937 Fh=0.827669150 dEtp=1.643423982e-05 dEth=1.617407174e-05 dEsp=4.748226101e-05 dEsh=5.492336365e-05 dDp=3.217093671e-04 dDh=3.275748371e-04 stagp=(0.32834990,0.32853380) stagh=(0.32636172,0.32647321)
059 t= 4.4250 Fp=0.995909373 Fh=0.827689544 dEtp=2.201004927e-05 dEth=1.132175742e-04 dEsp=7.497684419e-05 dEsh=1.485157346e-04 dDp=3.178268781e-04 dDh=3.306551347e-04 stagp=(0.32691030,0.32710307) stagh=(0.32562804,0.32575650)
060 t= 4.5000 Fp=0.995909773 Fh=0.827710240 dEtp=6.051868871e-05 dEth=2.117851182e-04 dEsp=9.732637791e-05 dEsh=2.366218960e-04 dDp=2.868692654e-04 dDh=2.993613652e-04 stagp=(0.32535664,0.32554004) stagh=(0.32484017,0.32496393)
061 t= 4.5750 Fp=0.995910391 Fh=0.827729950 dEtp=9.132111389e-05 dEth=2.978791225e-04 dEsp=1.111249035e-04 dEsh=3.099929784e-04 dDp=2.338711556e-04 dDh=2.412180184e-04 stagp=(0.32427372,0.32442938) stagh=(0.32450616,0.32460137)
062 t= 4.6500 Fp=0.995911560 Fh=0.827747467 dEtp=1.083223290e-04 dEth=3.597403677e-04 dEsp=1.146394817e-04 dEsh=3.622384446e-04 dDp=1.702113461e-04 dDh=1.719694360e-04 stagp=(0.32418156,0.32429405) stagh=(0.32508764,0.32513212)
063 t= 4.7250 Fp=0.995913586 Fh=0.827761771 dEtp=1.086949650e-04 dEth=3.903285216e-04 dEsp=1.080399637e-04 dEsh=3.905763088e-04 dDp=1.111843816e-04 dDh=1.114935085e-04 stagp=(0.32543113,0.32549099) stagh=(0.32691744,0.32689480)
064 t= 4.8000 Fp=0.995916625 Fh=0.827772059 dEtp=9.369278924e-05 dEth=3.886018214e-04 dEsp=9.327417545e-05 dEsh=3.956375892e-04 dDp=7.225221197e-05 dDh=7.813272297e-05 stagp=(0.32812974,0.32813550) stagh=(0.33013078,0.33003401)
065 t= 4.8750 Fp=0.995920587 Fh=0.827777743 dEtp=6.836485011e-05 dEth=3.592246762e-04 dEsp=7.362811570e-05 dEsh=3.805104798e-04 dDp=6.491654936e-05 dDh=8.309536537e-05 stagp=(0.33211660,0.33207547) stagh=(0.33462805,0.33446174)
066 t= 4.9500 Fp=0.995925100 Fh=0.827778413 dEtp=4.015360743e-05 dEth=3.107738352e-04 dEsp=5.302354279e-05 dEsh=3.493774080e-04 dDp=9.336307042e-05 dDh=1.266918082e-04 stagp=(0.33699582,0.33692266) stagh=(0.34007931,0.33985988)
067 t= 5.0250 Fp=0.995929560 Fh=0.827773822 dEtp=1.673018468e-05 dEth=2.529762809e-04 dEsp=3.513835354e-05 dEsh=3.061476728e-04 dDp=1.529240363e-04 dDh=1.977268651e-04 stagp=(0.34221734,0.34213207) stagh=(0.34596948,0.34572318)
068 t= 5.1000 Fp=0.995933268 Fh=0.827763914 dEtp=3.719330207e-06 dEth=1.938386091e-04 dEsp=2.250697509e-05 dEsh=2.534498055e-04 dDp=2.309690273e-04 dDh=2.764579590e-04 stagp=(0.34718329,0.34710738) stagh=(0.35167335,0.35143248)
069 t= 5.1750 Fp=0.995935605 Fh=0.827748893 dEtp=3.061974214e-06 dEth=1.375868878e-04 dEsp=1.584032598e-05 dEsh=1.922517804e-04 dDp=3.101529682e-04 dDh=3.404053191e-04 stagp=(0.35135208,0.35130486) stagh=(0.35654497,0.35634297)
070 t= 5.2500 Fp=0.995936211 Fh=0.827729311 dEtp=1.258822277e-05 dEth=8.408344416e-05 dEsp=1.383172233e-05 dEsh=1.222328949e-04 dDp=3.732138488e-04 dDh=3.713213914e-04 stagp=(0.35431645,0.35431209) stagh=(0.36000609,0.35987241)
071 t= 5.3250 Fp=0.995935093 Fh=0.827706145 dEtp=2.697062609e-05 dEth=2.989512825e-05 dEsp=1.362453963e-05 dEsh=4.283802766e-05 dDp=4.079566703e-04 dDh=3.611370594e-04 stagp=(0.35584310,0.35588880) stagh=(0.36162233,0.36157802)
072 t= 5.4000 Fp=0.995932636 Fh=0.827680806 dEtp=3.973955338e-05 dEth=2.937582709e-05 dEsp=1.190177169e-05 dEsh=4.527601679e-05 dDp=4.108925106e-04 dDh=3.149084178e-04 stagp=(0.35587483,0.35597043) stagh=(0.36116022,0.36121483)
073 t= 5.4750 Fp=0.995929496 Fh=0.827655062 dEtp=4.565637189e-05 dEth=9.632365139e-05 dEsp=6.287210654e-06 dEsh=1.388796703e-04 dDp=3.883183922e-04 dDh=2.496675925e-04 stagp=(0.35450643,0.35464535) stagh=(0.35862122,0.35877138)
074 t= 5.5500 Fp=0.995926420 Fh=0.827630862 dEtp=4.260721353e-05 dEth=1.695166588e-04 dEsp=3.451893670e-06 dEsh=2.314396631e-04 dDp=3.543614052e-04 dDh=1.893729456e-04 stagp=(0.35194970,0.35212062) stagh=(0.35424957,0.35447935)
075 t= 5.6250 Fp=0.995924037 Fh=0.827610098 dEtp=3.235196382e-05 dEth=2.426332961e-04 dEsp=1.492245943e-05 dEsh=3.134429361e-04 dDp=3.264544088e-04 dDh=1.574372129e-04 stagp=(0.34849954,0.34868840) stagh=(0.34851090,0.34879379)
076 t= 5.7000 Fp=0.995922696 Fh=0.827594334 dEtp=1.987200072e-05 dEth=3.053345995e-04 dEsp=2.364664956e-05 dEsh=3.738979216e-04 dDp=3.195605275e-04 dDh=1.691518160e-04 stagp=(0.34450603,0.34469791) stagh=(0.34203991,0.34234221)
077 t= 5.7750 Fp=0.995922397 Fh=0.827584574 dEtp=1.155106446e-05 dEth=3.456982766e-04 dEsp=2.454788924e-05 dEsh=4.027385906e-04 dDp=3.409351220e-04 dDh=2.264279003e-04 stagp=(0.34035013,0.34053065) stagh=(0.33555944,0.33584470)
078 t= 5.8500 Fp=0.995922844 Fh=0.827581103 dEtp=1.281932562e-05 dEth=3.535421953e-04 dEsp=1.391163671e-05 dEsh=3.935079400e-04 dDp=3.871363736e-04 dDh=3.165922562e-04 stagp=(0.33641652,0.33657308) stagh=(0.32978006,0.33001414)
079 t= 5.9250 Fp=0.995923587 Fh=0.827583440 dEtp=2.605978063e-05 dEth=3.236691628e-04 dEsp=8.881557027e-06 dEsh=3.455331614e-04 dDp=4.443719512e-04 dDh=4.157248702e-04 stagp=(0.33305934,0.33318196) stagh=(0.32529605,0.32545212)
080 t= 6.0000 Fp=0.995924218 Fh=0.827590406 dEtp=4.947532900e-05 dEth=2.580518750e-04 dEsp=4.052013770e-05 dEsh=2.648934322e-04 dDp=4.922768700e-04 dDh=4.956070486e-04 stagp=(0.33056128,0.33064350) stagh=(0.32249783,0.32256064)
081 t= 6.0750 Fp=0.995924540 Fh=0.827600293 dEtp=7.729432421e-05 dEth=1.662481787e-04 dEsp=7.420035773e-05 dEsh=1.637976683e-04 dDp=5.101530320e-04 dDh=5.322167457e-04 stagp=(0.32909367,0.32913330) stagh=(0.32151981,0.32148842)
082 t= 6.1500 Fp=0.995924659 Fh=0.827611115 dEtp=1.012676563e-04 dEth=6.381799003e-05 dEsp=1.012802377e-04 dEsh=5.842965017e-05 dDp=4.838878211e-04 dDh=5.132136907e-04 stagp=(0.32868838,0.32868812) stagh=(0.32223553,0.32212395)
083 t= 6.2250 Fp=0.995924954 Fh=0.827620865 dEtp=1.130181243e-04 dEth=3.092622417e-05 dEsp=1.136738749e-04 dEsh=3.425587141e-05 dDp=4.114609289e-04 dDh=4.421458445e-04 stagp=(0.32923210,0.32919968) stagh=(0.32430055,0.32413596)
084 t= 6.3000 Fp=0.995925951 Fh=0.827627761 dEtp=1.065568475e-04 dEth=1.010192511e-04 dEsp=1.062662311e-04 dEsh=1.000118014e-04 dDp=3.052535680e-04 dDh=3.380800090e-04 stagp=(0.33048724,0.33043497) stagh=(0.32723141,0.32705023)
085 t= 6.3750 Fp=0.995928116 Fh=0.827630422 dEtp=8.024277902e-05 dEth=1.347102726e-04 dEsp=7.862650361e-05 dEsh=1.301951309e-04 dDp=1.902234104e-04 dDh=2.307080640e-04 stagp=(0.33213640,0.33207997) stagh=(0.33050028,0.33034265)
086 t= 6.4500 Fp=0.995931659 Fh=0.827627948 dEtp=3.762740771e-05 dEth=1.280505962e-04 dEsp=3.551883929e-05 dEsh=1.233626693e-04 dDp=9.817362251e-05 dDh=1.522979061e-04 stagp=(0.33383941,0.33379611) stagh=(0.33362255,0.33352629)
087 t= 6.5250 Fp=0.995936399 Fh=0.827619914 dEtp=1.306113800e-05 dEth=8.549399718e-05 dEsp=1.393896465e-05 dEsh=8.516710018e-05 dDp=5.949954554e-05 dDh=1.287593626e-04 stagp=(0.33528905,0.33527565) stagh=(0.33621926,0.33621427)
088 t= 6.6000 Fp=0.995941756 Fh=0.827606285 dEtp=6.067517118e-05 dEth=1.835005375e-05 dEsp=5.825277358e-05 dEsh=2.657603828e-05 dDp=9.460837770e-05 dDh=1.723307932e-04 stagp=(0.33625272,0.33628339) stagh=(0.33804569,0.33814986)
089 t= 6.6750 Fp=0.995946858 Fh=0.827587301 dEtp=9.370735370e-05 dEth=5.847208168e-05 dEsp=8.617063863e-05 dEsh=3.914111604e-05 dDp=2.074281612e-04 dDh=2.779151686e-04 stagp=(0.33659321,0.33667781) stagh=(0.33898907,0.33920607)
090 t= 6.7500 Fp=0.995950763 Fh=0.827563373 dEtp=1.030209516e-04 dEth=1.305243317e-04 dEsp=8.931213991e-05 dEsh=9.989619282e-05 dDp=3.829636885e-04 dDh=4.240645652e-04 stagp=(0.33626886,0.33641177) stagh=(0.33904600,0.33936530)
091 t= 6.8250 Fp=0.995952700 Fh=0.827535013 dEtp=8.408812746e-05 dEth=1.875772148e-04 dEsp=6.419617906e-05 dEsh=1.478445881e-04 dDp=5.898195551e-04 dDh=5.783371720e-04 stagp=(0.33531943,0.33551903) stagh=(0.33829394,0.33869264)
092 t= 6.9000 Fp=0.995952276 Fh=0.827502823 dEtp=3.820340908e-05 dEth=2.260787962e-04 dEsp=1.320323317e-05 dEsh=1.812371181e-04 dDp=7.872638162e-04 dDh=7.055898763e-04 stagp=(0.33384649,0.33409522) stagh=(0.33686750,0.33731364)
093 t= 6.9750 Fp=0.995949572 Fh=0.827467540 dEtp=2.761350027e-05 dEth=2.496970057e-04 dEsp=5.575381206e-05 dEsh=2.045795021e-04 dDp=9.351175091e-04 dDh=7.770268234e-04 stagp=(0.33199550,0.33228041) stagh=(0.33494418,0.33540098)
094 t= 7.0500 Fp=0.995945102 Fh=0.827430096 dEtp=1.020091286e-04 dEth=2.678467663e-04 dEsp=1.307953761e-04 dEsh=2.270514537e-04 dDp=1.003891990e-03 dDh=7.776703446e-04 stagp=(0.32994269,0.33024650) stagh=(0.33273640,0.33316694)
095 t= 7.1250 Fp=0.995939654 Fh=0.827391692 dEtp=1.716736524e-04 dEth=2.926161574e-04 dEsp=1.985488425e-04 dEsh=2.595803874e-04 dDp=9.824189996e-04 dDh=7.103588875e-04 stagp=(0.32788509,0.32818769) stagh=(0.33048261,0.33085459)
096 t= 7.2000 Fp=0.995934061 Fh=0.827353841 dEtp=2.241719234e-04 dEth=3.348867663e-04 dEsp=2.469862452e-04 dEsh=3.113020581e-04 dDp=8.808067754e-04 dDh=5.952625946e-04 stagp=(0.32602945,0.32631003) stagh=(0.32843034,0.32872040)
097 t= 7.2750 Fp=0.995928977 Fh=0.827318368 dEtp=2.505820323e-04 dEth=4.006215190e-04 dEsp=2.679744119e-04 dEsh=3.862944342e-04 dDp=7.277771231e-04 dDh=4.650047259e-04 stagp=(0.32457614,0.32481566) stagh=(0.32680840,0.32700570)
098 t= 7.3500 Fp=0.995924732 Fh=0.827287376 dEtp=2.473192636e-04 dEth=4.882550580e-04 dEsp=2.589232103e-04 dEsh=4.814189723e-04 dDp=5.629776774e-04 dDh=3.565274360e-04 stagp=(0.32369683,0.32388078) stagh=(0.32579233,0.32590069)
099 t= 7.4250 Fp=0.995921300 Fh=0.827263132 dEtp=2.166926573e-04 dEth=5.878938763e-04 dEsp=2.231106238e-04 dEsh=5.858718784e-04 dDp=4.263005841e-04 dDh=3.016207994e-04 stagp=(0.32350932,0.32363028) stagh=(0.32547219,0.32551030)
100 t= 7.5000 Fp=0.995918383 Fh=0.827247895 dEtp=1.660372745e-04 dEth=6.826615449e-04 dEsp=1.685846433e-04 dEsh=6.826918950e-04 dDp=3.471602141e-04 dDh=3.183925864e-04 stagp=(0.32405534,0.32411478) stagh=(0.32583340,0.32583269)
101 t= 7.5750 Fp=0.995915574 Fh=0.827243653 dEtp=1.056336075e-04 dEth=7.520622809e-04 dEsp=1.059213996e-04 dEsh=7.520530283e-04 dDp=3.367966290e-04 dDh=4.058238965e-04 stagp=(0.32528830,0.32529708) stagh=(0.32675823,0.32675851)
102 t= 7.6500 Fp=0.995912543 Fh=0.827251786 dEtp=4.597659658e-05 dEth=7.767499090e-04 dEsp=4.545482610e-05 dEsh=7.757519795e-04 dDp=3.858988281e-04 dDh=5.429448530e-04 stagp=(0.32707515,0.32705251) stagh=(0.32804888,0.32809218)
103 t= 7.7250 Fp=0.995909176 Fh=0.827272680 dEtp=4.819270517e-06 dEth=7.436673481e-04 dEsp=5.222679002e-06 dEsh=7.419476957e-04 dDp=4.683540559e-04 dDh=6.931669923e-04 stagp=(0.32921278,0.32918375) stagh=(0.32946589,0.32958961)
104 t= 7.8000 Fp=0.995905644 Fh=0.827305382 dEtp=4.269078914e-05 dEth=6.502746586e-04 dEsp=4.273595120e-05 dEsh=6.490049767e-04 dDp=5.501219291e-04 dDh=8.130949768e-04 stagp=(0.33145509,0.33144689) stagh=(0.33077059,0.33100131)
105 t= 7.8750 Fp=0.995902375 Fh=0.827347399 dEtp=6.784432655e-05 dEth=5.066214524e-04 dEsp=6.792059221e-05 dEsh=5.073236839e-04 dDp=6.006280332e-04 dDh=8.639512220e-04 stagp=(0.33354460,0.33358254) stagh=(0.33176019,0.33210947)
106 t= 7.9500 Fp=0.995899950 Fh=0.827394749 dEtp=8.346292973e-05 dEth=3.344076071e-04 dEsp=8.435077551e-05 dEsh=3.383657164e-04 dDp=6.031766197e-04 dDh=8.228683862e-04 stagp=(0.33524252,0.33534638) stagh=(0.33228811,0.33275113)
107 t= 8.0250 Fp=0.995898963 Fh=0.827442350 dEtp=9.338211838e-05 dEth=1.628873579e-04 dEsp=9.593147356e-05 dEsh=1.707207730e-04 dDp=5.610258499e-04 dDh=6.910097186e-04 stagp=(0.33635349,0.33653487) stagh=(0.33226900,0.33282633)
108 t= 8.1000 Fp=0.995899865 Fh=0.827484740 dEtp=9.962772454e-05 dEth=2.235934094e-05 dEsp=1.044582371e-04 dEsh=3.386919689e-05 dDp=4.969824224e-04 dDh=4.959469346e-04 stagp=(0.33674416,0.33700512) stagh=(0.33167427,0.33229607)
109 t= 8.1750 Fp=0.995902832 Fh=0.827517007 dEtp=1.008227251e-04 dEth=6.319901192e-05 dEsp=1.081249238e-04 dEsh=4.891229567e-05 dDp=4.463456036e-04 dDh=2.869663939e-04 stagp=(0.33635671,0.33668976) stagh=(0.33052661,0.33117823)
110 t= 8.2500 Fp=0.995907673 Fh=0.827535735 dEtp=9.220389084e-05 dEth=8.233206340e-05 dEsp=1.016758360e-04 dEsh=6.657269198e-05 dDp=4.451800295e-04 dDh=1.237699986e-04 stagp=(0.33521838,0.33560759) stagh=(0.32890056,0.32954811)
111 t= 8.3250 Fp=0.995913789 Fh=0.827539711 dEtp=6.740612594e-05 dEth=3.922861865e-05 dEsp=7.831859201e-05 dEsh=2.336090035e-05 dDp=5.175321524e-04 dDh=6.096064670e-05 stagp=(0.33344653,0.33386938) stagh=(0.32692995,0.32754481)
112 t= 8.4000 Fp=0.995920195 Fh=0.827530166 dEtp=2.147082800e-05 dEth=4.741273719e-05 dEsp=3.282744591e-05 dEsh=6.224685892e-05 dDp=6.657534568e-04 dDh=1.322070648e-04 stagp=(0.33124580,0.33167572) stagh=(0.32481691,0.32537848)
113 t= 8.4750 Fp=0.995925611 Fh=0.827510449 dEtp=4.601832205e-05 dEth=1.497503478e-04 dEsp=3.527067612e-05 dEsh=1.627944329e-04 dDp=8.673032591e-04 dDh=3.385560395e-04 stagp=(0.32889322,0.32930257) stagh=(0.32283193,0.32332875)
114 t= 8.5500 Fp=0.995928633 Fh=0.827485190 dEtp=1.289399908e-04 dEth=2.390397046e-04 dEsp=1.196972003e-04 dEsh=2.499638123e-04 dDp=1.079458107e-03 dDh=6.446927471e-04 stagp=(0.32670796,0.32707143) stagh=(0.32129531,0.32172489)
115 t= 8.6250 Fp=0.995927973 Fh=0.827459166 dEtp=2.147375217e-04 dEth=2.943593597e-04 dEsp=2.075653124e-04 dEsh=3.032089294e-04 dDp=1.250839566e-03 dDh=9.850720720e-04 stagp=(0.32500651,0.32530477) stagh=(0.32053556,0.32090358)
116 t= 8.7000 Fp=0.995922735 Fh=0.827436213 dEtp=2.868681162e-04 dEth=3.090883754e-04 dEsp=2.819088557e-04 dEsh=3.161781842e-04 dDp=1.336400054e-03 dDh=1.279167359e-03 stagp=(0.32404977,0.32427299) stagh=(0.32082955,0.32114867)
117 t= 8.7750 Fp=0.995912674 Fh=0.827418518 dEtp=3.289900506e-04 dEth=2.929219578e-04 dEsp=3.259736006e-04 dEsh=2.987050033e-04 dDp=1.311247750e-03 dDh=1.452324950e-03 stagp=(0.32399323,0.32414369) stagh=(0.32233783,0.32262628)
118 t= 8.8500 Fp=0.995898362 Fh=0.827406526 dEtp=3.295930646e-04 dEth=2.685041765e-04 dEsp=3.279511618e-04 dEsh=2.734379351e-04 dDp=1.178904990e-03 dDh=1.456707495e-03 stagp=(0.32485331,0.32494636) stagh=(0.32505356,0.32533316)
119 t= 8.9250 Fp=0.995881194 Fh=0.827399479 dEtp=2.855344228e-04 dEth=2.634520843e-04 dEsp=2.845872634e-04 dEsh=2.678838433e-04 dDp=9.712547535e-04 dDh=1.286256777e-03 stagp=(0.32650048,0.32656318) stagh=(0.32878173,0.32907513)
120 t= 9.0000 Fp=0.995863187 Fh=0.827396410 dEtp=2.032553267e-04 dEth=3.001083729e-04 dEsp=2.024137572e-04 dEsh=3.042031121e-04 dDp=7.400200620e-04 dDh=9.808344259e-04 stagp=(0.32868312,0.32875037) stagh=(0.33315792,0.33348513)
121 t= 9.0750 Fp=0.995846592 Fh=0.827397221 dEtp=9.722076685e-05 dEth=3.862594285e-04 dEsp=9.614034769e-05 dEsh=3.899889229e-04 dDp=5.422989620e-04 dDh=6.175110168e-04 stagp=(0.33107756,0.33118614) stagh=(0.33770405,0.33807884)
122 t= 9.1500 Fp=0.995833397 Fh=0.827403421 dEtp=1.392582845e-05 dEth=5.099502879e-04 dEsp=1.528571296e-05 dEsh=5.131457612e-04 dDp=4.245602940e-04 dDh=2.906571927e-04 stagp=(0.33335189,0.33353364) stagh=(0.34190819,0.34233529)
123 t= 9.2250 Fp=0.995824848 Fh=0.827418214 dEtp=1.122243064e-04 dEth=6.403932388e-04 dEsp=1.136475471e-04 dEsh=6.428407513e-04 dDp=4.099722980e-04 dDh=8.598383869e-05 stagp=(0.33522704,0.33550277) stagh=(0.34530849,0.34578267)
124 t= 9.3000 Fp=0.995821130 Fh=0.827445809 dEtp=1.846900189e-04 dEth=7.351221467e-04 dEsp=1.858290697e-04 dEsh=7.366620550e-04 dDp=4.928430465e-04 dDh=5.587809688e-05 stagp=(0.33652028,0.33689576) stagh=(0.34756178,0.34806940)
125 t= 9.3750 Fp=0.995821326 Fh=0.827490132 dEtp=2.257097230e-04 dEth=7.515614591e-04 dEsp=2.262351542e-04 dEsh=7.521521297e-04 dDp=6.416768013e-04 dDh=2.034903092e-04 stagp=(0.33716180,0.33762702) stagh=(0.34848335,0.34900633)
126 t= 9.4500 Fp=0.995823689 Fh=0.827553306 dEtp=2.369446091e-04 dEth=6.596977347e-04 dEsp=2.366629241e-04 dEsh=6.594217187e-04 dDp=8.096575435e-04 dDh=4.808387629e-04 stagp=(0.33718427,0.33771600) stagh=(0.34805434,0.34857537)
127 t= 9.5250 Fp=0.995826178 Fh=0.827634412 dEtp=2.250611016e-04 dEth=4.520585500e-04 dEsp=2.239653395e-04 dEsh=4.510777997e-04 dDp=9.491413154e-04 dDh=8.022228044e-04 stagp=(0.33669316,0.33726008) stagh=(0.34640232,0.34690974)
128 t= 9.6000 Fp=0.995827093 Fh=0.827728960 dEtp=1.983616942e-04 dEth=1.479000733e-04 dEsp=1.966108934e-04 dEsh=1.463903390e-04 dDp=1.025668981e-03 dDh=1.069590921e-03 stagp=(0.33583050,0.33639941) stagh=(0.34376646,0.34425703)
129 t= 9.6750 Fp=0.995825626 Fh=0.827829317 dEtp=1.636923369e-04 dEth=2.098196880e-04 dEsp=1.615432547e-04 dEsh=2.117188554e-04 dDp=1.027384614e-03 dDh=1.202619254e-03 stagp=(0.33474454,0.33528598) stagh=(0.34045909,0.34093757)
130 t= 9.7500 Fp=0.995822136 Fh=0.827926016 dEtp=1.247591262e-04 dEth=5.639698324e-04 dEsp=1.224856283e-04 dEsh=5.661664173e-04 dDp=9.673589159e-04 dDh=1.164400591e-03 stagp=(0.33357294,0.33406507) stagh=(0.33683135,0.33730683)
131 t= 9.8250 Fp=0.995818028 Fh=0.828009604 dEtp=8.231338096e-05 dEth=8.566405028e-04 dEsp=8.014359388e-05 dEsh=8.590639860e-04 dDp=8.785792642e-04 dDh=9.745418727e-04 stagp=(0.33244044,0.33287041) stagh=(0.33324464,0.33372488)
132 t= 9.9000 Fp=0.995815264 Fh=0.828072516 dEtp=3.584728647e-05 dEth=1.042648780e-03 dEsp=3.393242055e-05 dEsh=1.045209095e-03 dDp=8.035215568e-04 dDh=7.049984830e-04 stagp=(0.33146512,0.33182847) stagh=(0.33004500,0.33053083)
133 t= 9.9750 Fp=0.995815638 Fh=0.828110454 dEtp=1.418057469e-05 dEth=1.099431469e-03 dEsp=1.576865702e-05 dEsh=1.101991887e-03 dDp=7.816093656e-04 dDh=4.591607939e-04 stagp=(0.33076440,0.33106321) stagh=(0.32753618,0.32801795)
134 t=10.0500 Fp=0.995820041 Fh=0.828122908 dEtp=6.474886038e-05 dEth=1.030738643e-03 dEsp=6.600372122e-05 dEsh=1.033121356e-03 dDp=8.381482978e-04 dDh=3.399887707e-04 stagp=(0.33045277,0.33069383) stagh=(0.32594944,0.32640715)
135 t=10.1250 Fp=0.995827967 Fh=0.828112746 dEtp=1.101367005e-04 dEth=8.634119826e-04 dEsp=1.110993285e-04 dEsh=8.654354812e-04 dDp=9.775450247e-04 dDh=4.166881417e-04 stagp=(0.33062795,0.33082165) stagh=(0.32541319,0.32582038)
136 t=10.2000 Fp=0.995837460 Fh=0.828085080 dEtp=1.433901253e-04 dEth=6.385399447e-04 dEsp=1.441328997e-04 dEsh=6.400697758e-04 dDp=1.182157152e-03 dDh=7.002888958e-04 stagp=(0.33134821,0.33150857) stagh=(0.32592967,0.32625993)
137 t=10.2750 Fp=0.995845526 Fh=0.828045830 dEtp=1.587870394e-04 dEth=3.998169873e-04 dEsp=1.593991578e-04 dEsh=4.008052309e-04 dDp=1.416507614e-03 dDh=1.136129810e-03 stagp=(0.33260875,0.33275386) stagh=(0.32736746,0.32760172)
138 t=10.3500 Fp=0.995848943 Fh=0.828000462 dEtp=1.542072798e-04 dEth=1.825605593e-04 dEsp=1.547786760e-04 dEsh=1.830563006e-04 dDp=1.635328001e-03 dDh=1.616295723e-03 stagp=(0.33432649,0.33447877) stagh=(0.32947618,0.32960830)
139 t=10.4250 Fp=0.995845222 Fh=0.827953275 dEtp=1.322968130e-04 dEth=6.421396463e-06 dEsp=1.328998225e-04 dEsh=6.549244776e-06 dDp=1.793260290e-03 dDh=2.008976117e-03 stagp=(0.33634074,0.33652601) stagh=(0.33192352,0.33196280)
140 t=10.5000 Fp=0.995833420 Fh=0.827907372 dEtp=9.978421389e-05 dEth=1.264536851e-04 dEsp=1.004564998e-04 dEsh=1.265359034e-04 dDp=1.854074777e-03 dDh=2.196397434e-03 stagp=(0.33843198,0.33867687) stagh=(0.33434823,0.33431828)
141 t=10.5750 Fp=0.995814551 Fh=0.827865208 dEtp=6.507516302e-05 dEth=2.277521008e-04 dEsp=6.581002772e-05 dEsh=2.278986704e-04 dDp=1.797765206e-03 dDh=2.110118755e-03 stagp=(0.34035482,0.34068285) stagh=(0.33641722,0.33635179)
142 t=10.6500 Fp=0.995791445 Fh=0.827829366 dEtp=3.500206790e-05 dEth=3.153724204e-04 dEsp=3.575106186e-05 dEsh=3.154825913e-04 dDp=1.624572712e-03 dDh=1.753021120e-03 stagp=(0.34187701,0.34230414) stagh=(0.33787411,0.33781128)
143 t=10.7250 Fp=0.995768077 Fh=0.827803173 dEtp=1.202760912e-05 dEth=4.044548582e-04 dEsp=1.271582328e-05 dEsh=4.044857981e-04 dDp=1.355614815e-03 dDh=1.201154517e-03 stagp=(0.34281479,0.34334568) stagh=(0.33856936,0.33854549)
144 t=10.8000 Fp=0.995748550 Fh=0.827790813 dEtp=6.891930016e-06 dEth=4.999093632e-04 dEsp=6.342758630e-06 dEsh=4.998703604e-04 dDp=1.030228255e-03 dDh=5.845819068e-04 stagp=(0.34305692,0.34368300) stagh=(0.33846877,0.33851324)
145 t=10.8750 Fp=0.995736035 Fh=0.827796801 dEtp=2.916659088e-05 dEth=5.926971933e-04 dEsp=2.881515473e-05 dEsh=5.926314612e-04 dDp=7.003956940e-04 dDh=5.264167711e-05 stagp=(0.34257427,0.34327433) stagh=(0.33764354,0.33777448)
146 t=10.9500 Fp=0.995731982 Fh=0.827824924 dEtp=6.340760950e-05 dEth=6.609312240e-04 dEsp=6.327779009e-05 dEsh=6.608926365e-04 dDp=4.228329982e-04 dDh=2.662995511e-04 stagp=(0.34141645,0.34215955) stagh=(0.33624944,0.33647032)
147 t=11.0250 Fp=0.995735830 Fh=0.827876975 dEtp=1.153073420e-04 dEth=6.754370694e-04 dEsp=1.153846083e-04 dEsh=6.754676325e-04 dDp=2.495871334e-04 dDh=2.989391151e-04 stagp=(0.33970009,0.34045016) stagh=(0.33450332,0.33480002)
148 t=11.1000 Fp=0.995745312 Fh=0.827951697 dEtp=1.840860832e-04 dEth=6.081634629e-04 dEsp=1.843248744e-04 dEsh=6.082756111e-04 dDp=2.183736889e-04 dDh=4.524178554e-05 stagp=(0.33759399,0.33831509) stagh=(0.33266181,0.33300048)
149 t=11.1750 Fp=0.995757248 Fh=0.828044325 dEtp=2.609301227e-04 dEth=4.410705886e-04 dEsp=2.612680460e-04 dEsh=4.412375798e-04 dDp=3.442819583e-04 dDh=4.241568633e-04 stagp=(0.33530443,0.33596547) stagh=(0.33100226,0.33132893)
150 t=11.2500 Fp=0.995768565 Fh=0.828146941 dEtp=3.302180476e-04 dEth=1.730294056e-04 dEsp=3.305915075e-04 dEsh=1.731865733e-04 dDp=6.147037712e-04 dDh=9.900605621e-04 stagp=(0.33306092,0.33363915) stagh=(0.32980196,0.33004530)
151 t=11.3250 Fp=0.995777247 Fh=0.828249610 dEtp=3.733878368e-04 dEth=1.771786716e-04 dEsp=3.737446734e-04 dEsh=1.771206078e-04 dDp=9.891796829e-04 dDh=1.519383072e-03 stagp=(0.33109994,0.33158296) stagh=(0.32931079,0.32938938)
152 t=11.4000 Fp=0.995782898 Fh=0.828342060 dEtp=3.743627935e-04 dEth=5.726390143e-04 dEsp=3.746696112e-04 dEsh=5.727706193e-04 dDp=1.405154933e-03 dDh=1.901986963e-03 stagp=(0.32964409,0.33003042) stagh=(0.32971545,0.32954974)
153 t=11.4750 Fp=0.995786751 Fh=0.828415470 dEtp=3.248538426e-04 dEth=9.645067317e-04 dEsp=3.250977795e-04 dEsh=9.648925518e-04 dDp=1.789412499e-03 dDh=2.079046776e-03 stagp=(0.32887571,0.32917436) stagh=(0.33109908,0.33062673)
154 t=11.5500 Fp=0.995791127 Fh=0.828463921 dEtp=2.277981912e-04 dEth=1.301174522e-03 dEsp=2.279842109e-04 dEsh=1.301830552e-03 dDp=2.073421530e-03 dDh=2.055187882e-03 stagp=(0.32890757,0.32913677) stagh=(0.33340589,0.33259757)
155 t=11.6250 Fp=0.995798508 Fh=0.828485162 dEtp=9.770626938e-05 dEth=1.538224259e-03 dEsp=9.785157659e-05 dEsh=1.539108420e-03 dDp=2.209405451e-03 dDh=1.891700429e-03 stagp=(0.32975651,0.32994198) stagh=(0.33642339,0.33529483)
156 t=11.7000 Fp=0.995810528 Fh=0.828480540 dEtp=4.238895378e-05 dEth=1.646815851e-03 dEsp=4.226174904e-05 dEsh=1.647835610e-03 dDp=2.183076925e-03 dDh=1.683733250e-03 stagp=(0.33132800,0.33150056) stagh=(0.33979277,0.33840941)
157 t=11.7750 Fp=0.995827209 Fh=0.828454167 dEtp=1.666412048e-04 dEth=1.618582924e-03 dEsp=1.665107132e-04 dEsh=1.619618001e-03 dDp=2.019125240e-03 dDh=1.528855851e-03 stagp=(0.33341806,0.33361073) stagh=(0.34305187,0.34152360)
158 t=11.8500 Fp=0.995846699 Fh=0.828411629 dEtp=2.531735240e-04 dEth=1.465986799e-03 dEsp=2.530247608e-04 dEsh=1.466919363e-03 dDp=1.776873675e-03 dDh=1.496488076e-03 stagp=(0.33573557,0.33598025) stagh=(0.34570534,0.34417151)
159 t=11.9250 Fp=0.995865628 Fh=0.828358632 dEtp=2.899597660e-04 dEth=1.218201112e-03 dEsp=2.897870246e-04 dEsh=1.218943438e-03 dDp=1.535907768e-03 dDh=1.606967476e-03 stagp=(0.33794258,0.33826665) stagh=(0.34730843,0.34591579)
160 t=12.0000 Fp=0.995880006 Fh=0.828299967 dEtp=2.780195363e-04 dEth=9.136779416e-04 dEsp=2.778266346e-04 dEsh=9.141890860e-04 dDp=1.374393974e-03 dDh=1.825849927e-03 stagp=(0.33970564,0.34012896) stagh=(0.34754562,0.34642390)
161 t=12.0750 Fp=0.995886401 Fh=0.828239057 dEtp=2.308644894e-04 dEth=5.913169958e-04 dEsp=2.306627226e-04 dEsh=5.916046950e-04 dDp=1.345454833e-03 dDh=2.074495587e-03 stagp=(0.34074787,0.34128064) stagh=(0.34628643,0.34552674)
162 t=12.1500 Fp=0.995883065 Fh=0.828178138 dEtp=1.701981232e-04 dEth=2.824388855e-04 dEsp=1.700027929e-04 dEsh=2.825483419e-04 dDp=1.458445292e-03 dDh=2.253458269e-03 stagp=(0.34089186,0.34153356) stagh=(0.34360686,0.34324727)
163 t=12.2250 Fp=0.995870633 Fh=0.828118926 dEtp=1.189862483e-04 dEth=5.495542904e-06 dEsp=1.188128458e-04 dEsh=5.490708205e-06 dDp=1.671579542e-03 dDh=2.271919521e-03 stagp=(0.34008592,0.34082528) stagh=(0.33977436,0.33979497)
164 t=12.3000 Fp=0.995852173 Fh=0.828063447 dEtp=9.386859662e-05 dEth=2.352837167e-04 dEsp=9.372955862e-05 dEsh=2.353395296e-04 dDp=1.899853563e-03 dDh=2.075157761e-03 stagp=(0.33841039,0.33922605) stagh=(0.33520324,0.33553067)
165 t=12.3750 Fp=0.995832517 Fh=0.828014697 dEtp=9.921687142e-05 dEth=4.471209667e-04 dEsp=9.911943033e-05 dEsh=4.471796879e-04 dDp=2.038011932e-03 dDh=1.662943871e-03 stagp=(0.33606491,0.33692649) stagh=(0.33039313,0.33091235)
166 t=12.4500 Fp=0.995817044 Fh=0.827976824 dEtp=1.248279369e-04 dEth=6.426381781e-04 dEsp=1.247731866e-04 dEsh=6.426741259e-04 dDp=1.993480017e-03 dDh=1.094342927e-03 stagp=(0.33334054,0.33420984) stagh=(0.32586385,0.32643461)
167 t=12.5250 Fp=0.995810238 Fh=0.827954722 dEtp=1.483073235e-04 dEth=8.321889715e-04 dEsp=1.482903891e-04 dEsh=8.321986334e-04 dDp=1.720197598e-03 dDh=4.778423274e-04 stagp=(0.33058179,0.33141445) stagh=(0.32209688,0.32257195)
168 t=12.6000 Fp=0.995814448 Fh=0.827953129 dEtp=1.418581318e-04 dEth=1.016661806e-03 dEsp=1.418692748e-04 dEsh=1.016658180e-03 dDp=1.242604207e-03 dDh=5.089077765e-05 stagp=(0.32814373,0.32889184) stagh=(0.31948768,0.31973112)
169 t=12.6750 Fp=0.995829216 Fh=0.827975482 dEtp=8.179450987e-05 dEth=1.182790678e-03 dEsp=8.182100749e-05 dEsh=1.182794785e-03 dDp=6.605758971e-04 dDh=3.587216650e-04 stagp=(0.32634838,0.32696465) stagh=(0.31830920,0.31821369)
170 t=12.7500 Fp=0.995851410 Fh=0.828022901 dEtp=4.194122570e-05 dEth=1.302497321e-03 dEsp=4.191264695e-05 dEsh=1.302529379e-03 dDp=1.308964428e-04 dDh=3.459147208e-04 stagp=(0.32544450,0.32588812) stagh=(0.31868476,0.31818714)
171 t=12.8250 Fp=0.995876132 Fh=0.828093586 dEtp=2.193025472e-04 dEth=1.336917073e-03 dEsp=2.192831038e-04 dEsh=1.336989343e-03 dDp=1.722316418e-04 dDh=2.831891226e-05 stagp=(0.32557448,0.32581827) stagh=(0.32057025,0.31966410)
172 t=12.9000 Fp=0.995898158 Fh=0.828182846 dEtp=4.191647554e-04 dEth=1.244687946e-03 dEsp=4.191613781e-04 dEsh=1.244800651e-03 dDp=1.080535752e-04 dDh=7.311206872e-04 stagp=(0.32675253,0.32679020) stagh=(0.32374914,0.32249215)
173 t=12.9750 Fp=0.995913490 Fh=0.828283702 dEtp=5.948885412e-04 dEth=9.930032258e-04 dEsp=5.949026133e-04 dEsh=9.931442595e-04 dDp=3.829866825e-04 dDh=1.655845392e-03 stagp=(0.32885807,0.32870974) stagh=(0.32784603,0.32635935)
174 t=13.0500 Fp=0.995920530 Fh=0.828387886 dEtp=6.955319473e-04 dEth=5.690537085e-04 dEsp=6.955594016e-04 dEsh=5.692019677e-04 dDp=1.249940346e-03 dDh=2.637743603e-03 stagp=(0.33164667,0.33136097) stagh=(0.33236397,0.33082112)
175 t=13.1250 Fp=0.995920535 Fh=0.828486930 dEtp=6.799492234e-04 dEth=1.097538514e-05 dEsp=6.799820025e-04 dEsh=1.084412234e-05 dDp=2.331804718e-03 dDh=3.483848191e-03 stagp=(0.33477844,0.33443000) stagh=(0.33674624,0.33535093)
176 t=13.2000 Fp=0.995917183 Fh=0.828573050 dEtp=5.298026398e-04 dEth=6.980622519e-04 dEsp=5.298310912e-04 dEsh=6.979687333e-04 dDp=3.390926923e-03 dDh=4.012429363e-03 stagp=(0.33786048,0.33754257) stagh=(0.34045571,0.33941075)
177 t=13.2750 Fp=0.995915379 Fh=0.828639687 dEtp=2.574290691e-04 dEth=1.413568017e-03 dEsp=2.574445668e-04 dEsh=1.413524237e-03 dDp=4.172552292e-03 dDh=4.094673492e-03 stagp=(0.34049697,0.34031024) stagh=(0.34305749,0.34253002)
178 t=13.3500 Fp=0.995919666 Fh=0.828681673 dEtp=9.430153450e-05 dEth=2.059993696e-03 dEsp=9.430447508e-05 dEsh=2.060000477e-03 dDp=4.474889318e-03 dDh=3.689482191e-03 stagp=(0.34233909,0.34237821) stagh=(0.34428685,0.34437707)
179 t=13.4250 Fp=0.995932724 Fh=0.828695226 dEtp=4.607805390e-04 dEth=2.537919264e-03 dEsp=4.608031612e-04 dEsh=2.537967211e-03 dDp=4.209404792e-03 dDh=2.862209698e-03 stagp=(0.34312772,0.34346784) stagh=(0.34408582,0.34480668)
180 t=13.5000 Fp=0.995954415 Fh=0.828677957 dEtp=7.711365228e-04 dEth=2.765943995e-03 dEsp=7.711758952e-04 dEsh=2.766017163e-03 dDp=3.432435250e-03 dDh=1.780329644e-03 stagp=(0.34272387,0.34340889) stagh=(0.34259961,0.34387262)
181 t=13.5750 Fp=0.995981686 Fh=0.828629104 dEtp=9.662248799e-04 dEth=2.699140744e-03 dEsp=9.662749469e-04 dEsh=2.699221537e-03 dDp=2.336644159e-03 dDh=6.835430969e-04 stagp=(0.34112514,0.34215906) stagh=(0.34013484,0.34180329)
182 t=13.6500 Fp=0.996009334 Fh=0.828549987 dEtp=1.013936364e-03 dEth=2.341234728e-03 dEsp=1.013989586e-03 dEsh=2.341308340e-03 dDp=1.202424561e-03 dDh=1.679231621e-04 stagp=(0.33846884,0.33981101) stagh=(0.33709246,0.33894886)
183 t=13.7250 Fp=0.996031427 Fh=0.828444550 dEtp=9.171902364e-04 dEth=1.746707350e-03 dEsp=9.172393612e-04 dEsh=1.746764520e-03 dDp=3.216247235e-04 dDh=5.565290026e-04 stagp=(0.33502306,0.33658796) stagh=(0.33389415,0.33571535)
184 t=13.8000 Fp=0.996042926 Fh=0.828319678 dEtp=7.120426866e-04 dEth=1.011250529e-03 dEsp=7.120822581e-04 dEsh=1.011288187e-03 dDp=8.489875309e-05 dDh=3.669002316e-04 stagp=(0.33116666,0.33282884) stagh=(0.33092100,0.33250282)
185 t=13.8750 Fp=0.996041036 Fh=0.828184989 dEtp=4.563199017e-04 dEth=2.519949755e-04 dEsp=4.563472031e-04 dEsh=2.520151249e-04 dDp=6.852418536e-05 dDh=3.751146776e-04 stagp=(0.32735702,0.32896206) stagh=(0.32847613,0.32966071)
186 t=13.9500 Fp=0.996025890 Fh=0.828051899 dEtp=2.121489273e-04 dEth=4.180869339e-04 dEsp=2.121642136e-04 dEsh=4.180792761e-04 dDp=7.063541440e-04 dDh=1.497260940e-03 stagp=(0.32408447,0.32546615) stagh=(0.32677266,0.32746483)
187 t=14.0250 Fp=0.996000406 Fh=0.827932059 dEtp=2.774025210e-05 dEth=9.151652268e-04 dEsp=2.774626269e-05 dEsh=9.151641932e-04 dDp=1.613073468e-03 dDh=2.715202267e-03 stagp=(0.32181293,0.32281632) stagh=(0.32593886,0.32611140)
188 t=14.1000 Fp=0.995969403 Fh=0.827835455 dEtp=7.590672806e-05 dEth=1.202139871e-03 dEsp=7.590582441e-05 dEsh=1.202140420e-03 dDp=2.496716977e-03 dDh=3.701242722e-03 stagp=(0.32091095,0.32141899) stagh=(0.32602727,0.32571810)
189 t=14.1750 Fp=0.995938285 Fh=0.827768691 dEtp=1.092020247e-04 dEth=1.293703298e-03 dEsp=1.092019595e-04 dEsh=1.293702179e-03 dDp=3.076752421e-03 dDh=4.171568176e-03 stagp=(0.32158220,0.32154168) stagh=(0.32701606,0.32632121)
190 t=14.2500 Fp=0.995911708 Fh=0.827733968 dEtp=1.077616021e-04 dEth=1.247100135e-03 dEsp=1.077592367e-04 dEsh=1.247096180e-03 dDp=3.170440654e-03 dDh=3.968749528e-03 stagp=(0.32380985,0.32325084) stagh=(0.32879859,0.32786391)
191 t=14.3250 Fp=0.995892593 Fh=0.827729058 dEtp=1.178277917e-04 dEth=1.140962627e-03 dEsp=1.178218972e-04 dEsh=1.140956542e-03 dDp=2.751519874e-03 dDh=3.115068620e-03 stagp=(0.32733044,0.32637345) stagh=(0.33116658,0.33017867)
192 t=14.4000 Fp=0.995881731 Fh=0.827748310 dEtp=1.780605096e-04 dEth=1.048319610e-03 dEsp=1.780518875e-04 dEsh=1.048313438e-03 dDp=1.962675058e-03 dDh=1.818421787e-03 stagp=(0.33165021,0.33049657) stagh=(0.33380046,0.33297467)
193 t=14.4750 Fp=0.995878000 Fh=0.827784355 dEtp=3.036722158e-04 dEth=1.011945431e-03 dEsp=3.036631690e-04 dEsh=1.011941819e-03 dDp=1.076641147e-03 dDh=4.250403467e-04 stagp=(0.33610823,0.33501287) stagh=(0.33628194,0.33584468)
194 t=14.5500 Fp=0.995879046 Fh=0.827829934 dEtp=4.782055864e-04 dEth=1.029855945e-03 dEsp=4.781989651e-04 dEsh=1.029857349e-03 dDp=4.160437081e-04 dDh=6.712247011e-04 stagp=(0.33997948,0.33920963) stagh=(0.33813904,0.33830231)
195 t=14.6250 Fp=0.995882158 Fh=0.827879269 dEtp=6.558866236e-04 dEth=1.055926297e-03 dEsp=6.558848008e-04 dEsh=1.055934271e-03 dDp=2.546950552e-04 dDh=1.139381389e-03 stagp=(0.34259954,0.34238711) stagh=(0.33892262,0.33985198)
196 t=14.7000 Fp=0.995885050 Fh=0.827928518 dEtp=7.741752638e-04 dEth=1.016053342e-03 dEsp=7.741793846e-04 dEsh=1.016068082e-03 dDp=7.292601915e-04 dDh=8.100771696e-04 stagp=(0.34348526,0.34398313) stagh=(0.33830078,0.34008178)
197 t=14.7750 Fp=0.995886364 Fh=0.827975249 dEtp=7.729207005e-04 dEth=8.353695706e-04 dEsp=7.729304163e-04 dEsh=8.353898329e-04 dDp=1.788051467e-03 dDh=2.674516266e-04 stagp=(0.34242499,0.34367704) stagh=(0.33614803,0.33875920)
198 t=14.8500 Fp=0.995885778 Fh=0.828017192 dEtp=6.144058229e-04 dEth=4.682795057e-04 dEsp=6.144195263e-04 dEsh=4.683029201e-04 dDp=3.193674592e-03 dDh=1.829656584e-03 stagp=(0.33951854,0.34145040) stagh=(0.33260359,0.33590450)
199 t=14.9250 Fp=0.995883799 Fh=0.828050860 dEtp=2.981747743e-04 dEth=7.822876815e-05 dEsp=2.981901758e-04 dEsh=7.820507991e-05 dDp=4.581075857e-03 dDh=3.466144538e-03 stagp=(0.33515899,0.33759122) stagh=(0.32807861,0.33181951)
200 t=15.0000 Fp=0.995881343 Fh=0.828070675 dEtp=1.340229420e-04 dEth=7.359294320e-04 dEsp=1.340081403e-04 dEsh=7.359081445e-04 dDp=5.556476070e-03 dDh=4.733221860e-03 stagp=(0.32996254,0.33264251) stagh=(0.32320525,0.32705952)
```

## Appendix G. Imported Initial-State Amplitudes (adapt_json branch)

All nonzero amplitudes from `initial_state.amplitudes_qn_to_q0` are listed here for strict reproducibility.

```text
bitstring_qn_to_q0=000001011 amp_re=+7.795299609273e-02 amp_im=-4.103311181669e-03 amp_abs=7.806091699747e-02
bitstring_qn_to_q0=000001101 amp_re=+2.195065368334e-01 amp_im=-1.893069845040e-02 amp_abs=2.203213359083e-01
bitstring_qn_to_q0=000001110 amp_re=+3.540728444282e-01 amp_im=+4.760044034753e-03 amp_abs=3.541048392506e-01
bitstring_qn_to_q0=000010011 amp_re=+2.247458817480e-01 amp_im=-3.305500750937e-02 amp_abs=2.271636962284e-01
bitstring_qn_to_q0=000010101 amp_re=+7.178555474208e-01 amp_im=-9.496482604949e-02 amp_abs=7.241097328095e-01
bitstring_qn_to_q0=000010110 amp_re=+2.157590603664e-01 amp_im=-1.861387149352e-02 amp_abs=2.165604958023e-01
bitstring_qn_to_q0=000100011 amp_re=+3.604562555781e-01 amp_im=-7.082314922977e-02 amp_abs=3.673481055514e-01
bitstring_qn_to_q0=000100101 amp_re=+2.208307013597e-01 amp_im=-3.286066075497e-02 amp_abs=2.232622262906e-01
bitstring_qn_to_q0=000100110 amp_re=+9.359100686078e-02 amp_im=-1.281888692776e-02 amp_abs=9.446481052371e-02
bitstring_qn_to_q0=001001011 amp_re=-3.151240911323e-03 amp_im=+5.216096584324e-03 amp_abs=6.094094096598e-03
bitstring_qn_to_q0=001001101 amp_re=-2.163982705799e-02 amp_im=+5.304478905486e-03 amp_abs=2.228047601732e-02
bitstring_qn_to_q0=001001110 amp_re=-8.117352354642e-03 amp_im=-2.564708968807e-03 amp_abs=8.512880907431e-03
bitstring_qn_to_q0=001010011 amp_re=+4.258600037223e-04 amp_im=+5.219939718884e-04 amp_abs=6.736723606162e-04
bitstring_qn_to_q0=001010101 amp_re=-1.766052302774e-03 amp_im=-8.650340529633e-04 amp_abs=1.966526035658e-03
bitstring_qn_to_q0=001010110 amp_re=+1.847814410158e-02 amp_im=+1.177126874108e-03 amp_abs=1.851559983140e-02
bitstring_qn_to_q0=001100011 amp_re=-2.958621996776e-04 amp_im=-2.016194641125e-05 amp_abs=2.965483860707e-04
bitstring_qn_to_q0=001100101 amp_re=+3.424703974936e-03 amp_im=-8.775875900195e-04 amp_abs=3.535358156411e-03
bitstring_qn_to_q0=001100110 amp_re=+6.566963491686e-03 amp_im=-2.811978022956e-03 amp_abs=7.143684616689e-03
bitstring_qn_to_q0=010001011 amp_re=+1.129447141289e-03 amp_im=-2.003338570996e-03 amp_abs=2.299786136797e-03
bitstring_qn_to_q0=010001101 amp_re=+1.932881600856e-02 amp_im=-5.147809391456e-03 amp_abs=2.000257657962e-02
bitstring_qn_to_q0=010001110 amp_re=+6.997321243071e-04 amp_im=+3.910022825506e-04 amp_abs=8.015658617651e-04
bitstring_qn_to_q0=010010011 amp_re=-1.935202089892e-02 amp_im=+1.802530991879e-03 amp_abs=1.943578737404e-02
bitstring_qn_to_q0=010010101 amp_re=+5.352978661890e-03 amp_im=+1.838544533940e-03 amp_abs=5.659914006231e-03
bitstring_qn_to_q0=010010110 amp_re=-1.964154568385e-02 amp_im=-2.560488151375e-03 amp_abs=1.980773627712e-02
bitstring_qn_to_q0=010100011 amp_re=-1.714017197825e-03 amp_im=-2.348016179660e-03 amp_abs=2.907066379425e-03
bitstring_qn_to_q0=010100101 amp_re=+1.219558911989e-02 amp_im=-2.951652480150e-03 amp_abs=1.254769486180e-02
bitstring_qn_to_q0=010100110 amp_re=-2.297885023687e-03 amp_im=-1.528012762821e-03 amp_abs=2.759546807979e-03
bitstring_qn_to_q0=011001011 amp_re=-2.235904513284e-04 amp_im=+1.222585269347e-04 amp_abs=2.548329596687e-04
bitstring_qn_to_q0=011001101 amp_re=-2.321707416628e-03 amp_im=+5.211610879719e-04 amp_abs=2.379481920092e-03
bitstring_qn_to_q0=011001110 amp_re=-1.213411201337e-04 amp_im=-2.746613828368e-04 amp_abs=3.002707822235e-04
bitstring_qn_to_q0=011010011 amp_re=-5.439129419276e-05 amp_im=-5.073246905359e-05 amp_abs=7.437873553804e-05
bitstring_qn_to_q0=011010101 amp_re=-9.790781757467e-05 amp_im=-3.334837077836e-04 amp_abs=3.475590943988e-04
bitstring_qn_to_q0=011010110 amp_re=-1.296312663110e-03 amp_im=-6.345003317770e-04 amp_abs=1.443266154098e-03
bitstring_qn_to_q0=011100011 amp_re=+4.406639561481e-05 amp_im=-1.942792926913e-05 amp_abs=4.815902468040e-05
bitstring_qn_to_q0=011100101 amp_re=+2.288007773079e-04 amp_im=-1.397148105923e-04 amp_abs=2.680858519123e-04
bitstring_qn_to_q0=011100110 amp_re=-1.611703337993e-04 amp_im=-1.898780239592e-04 amp_abs=2.490573036063e-04
bitstring_qn_to_q0=100001011 amp_re=+8.090441421115e-03 amp_im=-2.674625197654e-03 amp_abs=8.521083401564e-03
bitstring_qn_to_q0=100001101 amp_re=-3.633695329233e-04 amp_im=+6.806687756426e-05 amp_abs=3.696897581463e-04
bitstring_qn_to_q0=100001110 amp_re=-1.270638222204e-04 amp_im=+6.626000747456e-05 amp_abs=1.433024895380e-04
bitstring_qn_to_q0=100010011 amp_re=+1.564274691933e-02 amp_im=-5.582096340745e-03 amp_abs=1.660889312204e-02
bitstring_qn_to_q0=100010101 amp_re=-6.356741791450e-04 amp_im=+3.318499351766e-03 amp_abs=3.378834060694e-03
bitstring_qn_to_q0=100010110 amp_re=+3.327122738264e-04 amp_im=+7.261501337208e-04 amp_abs=7.987436847058e-04
bitstring_qn_to_q0=100100011 amp_re=+1.391327882560e-03 amp_im=-1.784181827128e-03 amp_abs=2.262542390552e-03
bitstring_qn_to_q0=100100101 amp_re=-2.372537103860e-02 amp_im=+2.009019139813e-04 amp_abs=2.372622162288e-02
bitstring_qn_to_q0=100100110 amp_re=-9.103501002321e-03 amp_im=+4.823524657377e-04 amp_abs=9.116270860416e-03
bitstring_qn_to_q0=101001011 amp_re=-2.837412931833e-04 amp_im=+5.512181042824e-04 amp_abs=6.199600954464e-04
bitstring_qn_to_q0=101001101 amp_re=+1.450879275207e-05 amp_im=-1.837781505026e-05 amp_abs=2.341472086411e-05
bitstring_qn_to_q0=101001110 amp_re=-1.414023480359e-06 amp_im=+2.491522755384e-07 amp_abs=1.435806135735e-06
bitstring_qn_to_q0=101010011 amp_re=+4.811420123484e-05 amp_im=+2.070396187989e-05 amp_abs=5.237967542846e-05
bitstring_qn_to_q0=101010101 amp_re=+2.146700865608e-05 amp_im=-6.409840172841e-05 amp_abs=6.759761508202e-05
bitstring_qn_to_q0=101010110 amp_re=-3.199425785042e-05 amp_im=+4.605923506196e-05 amp_abs=5.608106338054e-05
bitstring_qn_to_q0=101100011 amp_re=-1.152727789584e-05 amp_im=+2.091989253823e-05 amp_abs=2.388556131848e-05
bitstring_qn_to_q0=101100101 amp_re=-2.625665527700e-04 amp_im=+1.980775011334e-04 amp_abs=3.289010354024e-04
bitstring_qn_to_q0=101100110 amp_re=-5.567486324895e-04 amp_im=+1.687487011572e-04 amp_abs=5.817604007847e-04
bitstring_qn_to_q0=110001011 amp_re=+5.771870519449e-05 amp_im=-1.316019721218e-04 amp_abs=1.437029157522e-04
bitstring_qn_to_q0=110001101 amp_re=+1.084080656045e-06 amp_im=+1.243614314427e-05 amp_abs=1.248330433713e-05
bitstring_qn_to_q0=110001110 amp_re=+9.453463643985e-07 amp_im=+3.749329317937e-07 amp_abs=1.016983014620e-06
bitstring_qn_to_q0=110010011 amp_re=-1.567900414146e-03 amp_im=+8.230423332963e-04 amp_abs=1.770793717822e-03
bitstring_qn_to_q0=110010101 amp_re=+8.031432251047e-06 amp_im=+1.955364669667e-04 amp_abs=1.957013383138e-04
bitstring_qn_to_q0=110010110 amp_re=-1.462064775083e-05 amp_im=-2.563627948708e-05 amp_abs=2.951240699424e-05
bitstring_qn_to_q0=110100011 amp_re=-6.250156108698e-05 amp_im=+8.989756209617e-05 amp_abs=1.094898023066e-04
bitstring_qn_to_q0=110100101 amp_re=-1.081804251878e-03 amp_im=+1.629135777242e-04 amp_abs=1.094002410047e-03
bitstring_qn_to_q0=110100110 amp_re=+2.348132991776e-04 amp_im=+2.041701363090e-04 amp_abs=3.111635101215e-04
bitstring_qn_to_q0=111001011 amp_re=-1.638195908426e-05 amp_im=+8.623042169680e-06 amp_abs=1.851284526210e-05
bitstring_qn_to_q0=111001101 amp_re=-6.578188990613e-07 amp_im=+1.952800062877e-07 amp_abs=6.861923817837e-07
bitstring_qn_to_q0=111001110 amp_re=+5.634365723284e-07 amp_im=+6.798532509001e-08 amp_abs=5.675233699723e-07
bitstring_qn_to_q0=111010011 amp_re=-5.207039402403e-06 amp_im=-3.981895723272e-06 amp_abs=6.555055521442e-06
bitstring_qn_to_q0=111010101 amp_re=+1.614988007680e-06 amp_im=-3.717156119649e-07 amp_abs=1.657214156688e-06
bitstring_qn_to_q0=111010110 amp_re=+3.551129852429e-06 amp_im=-3.269224098892e-07 amp_abs=3.566146588533e-06
bitstring_qn_to_q0=111100011 amp_re=-7.361611013356e-07 amp_im=+1.854372201785e-06 amp_abs=1.995151480433e-06
bitstring_qn_to_q0=111100101 amp_re=-1.733514896735e-05 amp_im=+2.345343615437e-05 amp_abs=2.916455137950e-05
bitstring_qn_to_q0=111100110 amp_re=+1.563462830693e-05 amp_im=+2.145853330071e-05 amp_abs=2.655014602057e-05
```
