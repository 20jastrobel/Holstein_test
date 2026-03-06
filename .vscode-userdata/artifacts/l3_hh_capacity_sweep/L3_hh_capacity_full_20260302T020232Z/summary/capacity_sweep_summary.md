# L=3 HH Capacity Sweep Summary

## Run Meta

- created_utc: `2026-03-02T04:46:43.838200+00:00`
- out_dir: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/l3_hh_capacity_sweep/L3_hh_capacity_full_20260302T020232Z`
- sector: `(2,1)`
- methods: `m1_hh_hva, m3_adapt_paop_std, m4_adapt_paop_lf_std`
- selected_config: `interleaved,periodic,binary`

## Scout Results

| config | mean best ΔE | mean best runtime_s | timeouts |
|---|---:|---:|---:|
| interleaved,periodic,binary | 2.096482e+00 | 176.0 | 0 |
| blocked,open,binary | 2.374061e+00 | 102.5 | 0 |

## Best Per Method

| method | best ΔE | E_best | E_exact_sector | seed | rung | runtime_s |
|---|---:|---:|---:|---:|---|---:|
| m1_hh_hva | 3.639638e-01 | -0.334844555386 | -0.698808385100 | 101 | hva_r2_mi3000 | 274.4 |
| m3_adapt_paop_std | 2.962740e+00 | 2.263932023078 | -0.698808385100 | 101 | adapt_d80_mi3000 | 126.3 |
| m4_adapt_paop_lf_std | 2.962740e+00 | 2.263932023078 | -0.698808385100 | 101 | adapt_d80_mi3000 | 128.8 |

## Rung Progress

| method | rung | n_runs | n_ok | best ΔE | median ΔE | median runtime_s | decision |
|---|---|---:|---:|---:|---:|---:|---|
| m1_hh_hva | hva_r2_mi3000 | 2 | 2 | 3.639638e-01 | 3.639640e-01 | 356.4 | continue |
| m1_hh_hva | hva_r3_mi4500 | 2 | 0 | - | - | - | continue |
| m1_hh_hva | hva_r4_mi6000 | 2 | 0 | - | - | - | continue |
| m1_hh_hva | hva_r5_mi6000 | 2 | 0 | - | - | - | continue |
| m3_adapt_paop_std | adapt_d80_mi3000 | 2 | 2 | 2.962740e+00 | 2.962740e+00 | 126.7 | continue |
| m3_adapt_paop_std | adapt_d120_mi3000 | 2 | 2 | 2.962740e+00 | 2.962740e+00 | 128.0 | stop_adapt_plateau |
| m4_adapt_paop_lf_std | adapt_d80_mi3000 | 2 | 2 | 2.962740e+00 | 2.962740e+00 | 128.8 | continue |
| m4_adapt_paop_lf_std | adapt_d120_mi3000 | 2 | 2 | 2.962740e+00 | 2.962740e+00 | 130.2 | stop_adapt_plateau |
