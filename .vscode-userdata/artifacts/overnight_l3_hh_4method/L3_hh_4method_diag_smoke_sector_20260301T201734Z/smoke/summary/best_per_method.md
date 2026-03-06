# Best Per Method

| method_id | sector | seed | key_knobs | E_best | E_exact_sector | |ΔE| | runtime_s |
|---|---|---:|---|---:|---:|---:|---:|
| m1_hh_hva | (2,1) | 101 | ansatz=hh_hva, reps=1, restarts=1, maxiter=200 | 2.632557973208 | -0.526937719766 | 3.159e+00 | 2.9 |
| m2_hh_hva_tw | (2,1) | 101 | ansatz=hh_hva_tw, reps=1, restarts=1, maxiter=200 | 0.119530941875 | -0.526937719766 | 6.465e-01 | 6.4 |
| m3_adapt_paop_std | (2,1) | 101 | pool=paop_std, depth=10, max_depth=10, maxiter=200 | 2.263932023595 | -0.526937719766 | 2.791e+00 | 11.5 |
| m4_adapt_paop_lf_std | (2,1) | 101 | pool=paop_lf_std, depth=10, max_depth=10, maxiter=200 | 2.263932023595 | -0.526937719766 | 2.791e+00 | 12.0 |
