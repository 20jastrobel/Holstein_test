# L4 HH Warm-Start Sequential Benchmark Summary

## L3 to L4 Parameter Mapping

| Parameter | L3 successful reference | L4 chosen in this run |
|---|---|---|
| boundary | open | open |
| ordering | blocked | blocked |
| boson encoding | binary | binary |
| n_ph_max | 1 | 1 |
| sector | (2,1) at L=3 | (2,2) at L=4 |
| warm reps | 3 | 4 |
| warm restarts | 5/6 | 6 |
| warm maxiter | 4000/6000 | 6000 |
| probe depth/maxiter | 20/1200 (trend style) | 20/1200 |
| medium depth/maxiter | L3 B/C style heavier | 48/3600 |

## Warm-Start Stage

- Shared warm-start used for both pool arms.
- E_warm: `2.281550e-01`
- |DeltaE|_warm: `2.114971e-01`
- rel_err_warm: `1.269650e+01`
- warm work: nfev=`6000`, nit=`0`

## Run Table

| run_id | ok | |DeltaE| | rel_err | depth | nfev | runtime_s | stop | medium_gate |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| A_probe | True | 1.491916e-01 | 8.956203e+00 | 11 | 1810 | 9.003806e+02 | wallclock_cap | None |
| A_medium | True | 1.466918e-01 | 8.806138e+00 | 15 | 3167 | 1.800741e+03 | wallclock_cap | False |
| B_probe | True | 1.504408e-01 | 9.031193e+00 | 11 | 1708 | 9.004928e+02 | wallclock_cap | None |
| B_medium | True | 1.468441e-01 | 8.815279e+00 | 16 | 3080 | 1.800427e+03 | wallclock_cap | False |

## Comparisons

- medium target: |DeltaE| <= 1.000000e-03
- A_probe encouraging: `True`
- B_probe encouraging: `True`
- medium winner by |DeltaE|: `A_medium`
- A_vs_B medium delta difference (A-B): `-1.522624e-04`

## Final Completion Snapshot (2026-03-03)

- Model: Hubbard-Holstein (HH), `L=4`, open boundary, blocked ordering, binary boson encoding, `n_ph_max=1`.
- Sector: `(n_up, n_dn) = (2,2)`.
- Physics params: `t=1.0`, `U=4.0`, `dv=0.0`, `omega0=1.0`, `g_ep=0.5`.
- Exact sector energy: `E_exact = 1.665791e-02`.
- Warm-start method: `hh_hva_ptw` (reps=`4`, restarts=`6`, maxiter=`6000`).
- Warm-start result: `E_warm = 2.281550e-01`, `|DeltaE|_warm = 2.114971e-01`, `runtime_s = 3.639993e+04`.
- Total run runtime: `4.181008e+04 s` (~11.61 h).

| stage | E_best | \|DeltaE\| | rel_err | depth | stop_reason | runtime_s |
|---|---:|---:|---:|---:|---|---:|
| A_probe | 1.658495e-01 | 1.491916e-01 | 8.956203e+00 | 11 | wallclock_cap | 9.003806e+02 |
| A_medium | 1.633497e-01 | 1.466918e-01 | 8.806138e+00 | 15 | wallclock_cap | 1.800741e+03 |
| B_probe | 1.670987e-01 | 1.504408e-01 | 9.031193e+00 | 11 | wallclock_cap | 9.004928e+02 |
| B_medium | 1.635020e-01 | 1.468441e-01 | 8.815279e+00 | 16 | wallclock_cap | 1.800427e+03 |

- Medium winner by `|DeltaE|`: `A_medium`.
- Medium gate target (`1e-3`) status: failed (`A_medium_pass=false`, `B_medium_pass=false`).
- Artifact JSON: `artifacts/useful/L4/l4_hh_warmstart_uccsd_paop_hva_seq.json`
- Artifact CSV: `artifacts/useful/L4/l4_hh_warmstart_uccsd_paop_hva_seq_summary.csv`
- Artifact Log: `artifacts/useful/L4/l4_hh_warmstart_uccsd_paop_hva_seq.log`
