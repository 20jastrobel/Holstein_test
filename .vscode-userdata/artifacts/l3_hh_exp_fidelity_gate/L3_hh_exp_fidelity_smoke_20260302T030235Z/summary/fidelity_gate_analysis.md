# L=3 HH Exp Fidelity Analysis

## Run Info

- analyzed_at_utc: `2026-03-02T03:05:11.291197+00:00`
- run_dir: `artifacts/l3_hh_exp_fidelity_gate/L3_hh_exp_fidelity_smoke_20260302T030235Z`
- total_rows: `3`
- ok_rows: `3`

## Best Per Method

| method | config | seed | trotter | E_best | E_exact | |ΔE| | runtime_s | leak_flag |
|---|---|---:|---:|---:|---:|---:|---:|---|
| m1_hh_hva | blocked,open,binary | 101 | 1 | 1.132228348137 | -0.526937719766 | 1.659e+00 | 128.4 | True |
| m3_adapt_paop_std | blocked,open,binary | 101 | 1 | 2.263932037497 | -0.526937719766 | 2.791e+00 | 6.4 | False |
| m4_adapt_paop_lf_std | blocked,open,binary | 101 | 1 | 2.263932037497 | -0.526937719766 | 2.791e+00 | 6.5 | False |

## Best Per Method+Config

| method | config | best ΔE | seed | trotter | status |
|---|---|---:|---:|---:|---|
| m1_hh_hva | blocked,open,binary | 1.659e+00 | 101 | 1 | ok |
| m3_adapt_paop_std | blocked,open,binary | 2.791e+00 | 101 | 1 | ok |
| m4_adapt_paop_lf_std | blocked,open,binary | 2.791e+00 | 101 | 1 | ok |

## Ranked OK Runs

| rank | method | config | seed | trotter | |ΔE| | runtime_s |
|---:|---|---|---:|---:|---:|---:|
| 1 | m1_hh_hva | blocked,open,binary | 101 | 1 | 1.659e+00 | 128.4 |
| 2 | m3_adapt_paop_std | blocked,open,binary | 101 | 1 | 2.791e+00 | 6.4 |
| 3 | m4_adapt_paop_lf_std | blocked,open,binary | 101 | 1 | 2.791e+00 | 6.5 |
