# Best Per Method

| method_id | sector | seed | key_knobs | E_best | E_exact_sector | |ΔE| | runtime_s |
|---|---|---:|---|---:|---:|---:|---:|
| m1_hh_hva | (2,1) | 101 | ansatz=hh_hva, reps=1, restarts=1, maxiter=200 | 2.339348979493 | -0.698808385100 | 3.038e+00 | 35.1 |
| m2_hh_hva_tw | (2,1) | 101 | ansatz=hh_hva_tw, reps=1, restarts=1, maxiter=200 | -0.299477337739 | -0.698808385100 | 3.993e-01 | 90.1 |
| m3_adapt_paop_std | (2,1) | 101 | pool=paop_std, depth=4, max_depth=10, maxiter=200 | 2.263932037497 | -0.698808385100 | 2.963e+00 | 45.6 |
| m4_adapt_paop_lf_std | (2,1) | 101 | pool=paop_lf_std, depth=4, max_depth=10, maxiter=200 | 2.263932037497 | -0.698808385100 | 2.963e+00 | 49.2 |
