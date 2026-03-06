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
| warm restarts | 5/6 | 1 |
| warm maxiter | 4000/6000 | 800 |
| probe depth/maxiter | 20/1200 (trend style) | 1/20 |
| medium depth/maxiter | L3 B/C style heavier | 1/20 |

## Warm-Start Stage

- Shared warm-start used for both pool arms.
- E_warm: `7.392223e+00`
- |DeltaE|_warm: `7.375565e+00`
- rel_err_warm: `4.427667e+02`
- warm work: nfev=`17`, nit=`0`

## Run Table

| run_id | ok | |DeltaE| | rel_err | depth | nfev | runtime_s | stop | medium_gate |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| A_probe | True | 6.808898e+00 | 4.087487e+02 | 1 | 21 | 1.351740e+01 | max_depth | None |
| A_medium | True | 6.808898e+00 | 4.087487e+02 | 1 | 21 | 1.519257e+01 | max_depth | False |
| B_probe | True | 6.808898e+00 | 4.087487e+02 | 1 | 21 | 1.692643e+01 | max_depth | None |
| B_medium | True | 6.808898e+00 | 4.087487e+02 | 1 | 21 | 1.987997e+01 | max_depth | False |

## Comparisons

- medium target: |DeltaE| <= 1.000000e-03
- A_probe encouraging: `False`
- B_probe encouraging: `False`
- medium winner by |DeltaE|: `A_medium`
- A_vs_B medium delta difference (A-B): `0.000000e+00`

