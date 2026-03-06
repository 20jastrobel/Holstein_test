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
| warm maxiter | 4000/6000 | 40 |
| probe depth/maxiter | 20/1200 (trend style) | 1/40 |
| medium depth/maxiter | L3 B/C style heavier | 1/40 |

## Warm-Start Stage

- Shared warm-start used for both pool arms.
- E_warm: `4.907473e+00`
- |DeltaE|_warm: `4.890815e+00`
- rel_err_warm: `2.936032e+02`
- warm work: nfev=`40`, nit=`0`

## Run Table

| run_id | ok | |DeltaE| | rel_err | depth | nfev | runtime_s | stop | medium_gate |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| A_probe | True | 4.864852e+00 | 2.920446e+02 | 1 | 21 | 1.407000e+01 | max_depth | None |
| A_medium | True | 4.864852e+00 | 2.920446e+02 | 1 | 21 | 1.441397e+01 | max_depth | False |
| B_probe | True | 4.496304e+00 | 2.699201e+02 | 1 | 20 | 1.721339e+01 | max_depth | None |
| B_medium | True | 4.496304e+00 | 2.699201e+02 | 1 | 20 | 1.711241e+01 | max_depth | False |

## Comparisons

- medium target: |DeltaE| <= 1.000000e-03
- A_probe encouraging: `False`
- B_probe encouraging: `False`
- medium winner by |DeltaE|: `B_medium`
- A_vs_B medium delta difference (A-B): `3.685485e-01`

