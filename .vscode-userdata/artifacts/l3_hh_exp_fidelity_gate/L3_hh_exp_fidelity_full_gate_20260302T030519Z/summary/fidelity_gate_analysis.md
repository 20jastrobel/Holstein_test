# L=3 HH Exp Fidelity Analysis

## Run Info

- analyzed_at_utc: `2026-03-02T04:37:28.092144+00:00`
- run_dir: `artifacts/l3_hh_exp_fidelity_gate/L3_hh_exp_fidelity_full_gate_20260302T030519Z`
- total_rows: `19`
- ok_rows: `16`

## Best Per Method

| method | config | seed | trotter | E_best | E_exact | |ΔE| | runtime_s | leak_flag |
|---|---|---:|---:|---:|---:|---:|---:|---|
| m1_hh_hva | interleaved,periodic,binary | 102 | 1 | 0.905930867248 | -0.698808385100 | 1.605e+00 | 398.7 | True |
| m3_adapt_paop_std | interleaved,periodic,binary | 101 | 4 | 2.263932024910 | -0.698808385100 | 2.963e+00 | 130.8 | False |
| m4_adapt_paop_lf_std | interleaved,periodic,binary | 101 | 4 | 2.263932024910 | -0.698808385100 | 2.963e+00 | 132.4 | False |

## Best Per Method+Config

| method | config | best ΔE | seed | trotter | status |
|---|---|---:|---:|---:|---|
| m1_hh_hva | blocked,open,binary | 1.659e+00 | 101 | 1 | ok |
| m1_hh_hva | interleaved,periodic,binary | 1.605e+00 | 102 | 1 | ok |
| m3_adapt_paop_std | interleaved,periodic,binary | 2.963e+00 | 101 | 4 | ok |
| m4_adapt_paop_lf_std | interleaved,periodic,binary | 2.963e+00 | 101 | 4 | ok |

## Ranked OK Runs

| rank | method | config | seed | trotter | |ΔE| | runtime_s |
|---:|---|---|---:|---:|---:|---:|
| 1 | m1_hh_hva | interleaved,periodic,binary | 102 | 1 | 1.605e+00 | 398.7 |
| 2 | m1_hh_hva | interleaved,periodic,binary | 101 | 1 | 1.627e+00 | 319.3 |
| 3 | m1_hh_hva | interleaved,periodic,binary | 101 | 2 | 1.634e+00 | 425.4 |
| 4 | m1_hh_hva | blocked,open,binary | 101 | 1 | 1.659e+00 | 128.4 |
| 5 | m3_adapt_paop_std | interleaved,periodic,binary | 101 | 4 | 2.963e+00 | 130.8 |
| 6 | m3_adapt_paop_std | interleaved,periodic,binary | 102 | 4 | 2.963e+00 | 131.1 |
| 7 | m4_adapt_paop_lf_std | interleaved,periodic,binary | 101 | 4 | 2.963e+00 | 132.4 |
| 8 | m4_adapt_paop_lf_std | interleaved,periodic,binary | 102 | 4 | 2.963e+00 | 132.6 |
| 9 | m3_adapt_paop_std | interleaved,periodic,binary | 101 | 1 | 2.963e+00 | 7.1 |
| 10 | m3_adapt_paop_std | interleaved,periodic,binary | 102 | 1 | 2.963e+00 | 7.0 |
| 11 | m4_adapt_paop_lf_std | interleaved,periodic,binary | 101 | 1 | 2.963e+00 | 7.5 |
| 12 | m4_adapt_paop_lf_std | interleaved,periodic,binary | 102 | 1 | 2.963e+00 | 7.6 |
| 13 | m3_adapt_paop_std | interleaved,periodic,binary | 101 | 2 | 2.963e+00 | 3.2 |
| 14 | m3_adapt_paop_std | interleaved,periodic,binary | 102 | 2 | 2.963e+00 | 3.2 |
| 15 | m4_adapt_paop_lf_std | interleaved,periodic,binary | 101 | 2 | 2.963e+00 | 3.5 |
| 16 | m4_adapt_paop_lf_std | interleaved,periodic,binary | 102 | 2 | 2.963e+00 | 3.4 |
