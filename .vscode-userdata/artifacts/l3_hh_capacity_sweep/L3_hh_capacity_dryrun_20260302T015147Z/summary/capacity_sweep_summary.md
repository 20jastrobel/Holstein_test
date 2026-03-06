# L=3 HH Capacity Sweep Summary

## Run Meta

- created_utc: `2026-03-02T02:02:13.662703+00:00`
- out_dir: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/l3_hh_capacity_sweep/L3_hh_capacity_dryrun_20260302T015147Z`uwmadison.vpn.wisc.edu
- sector: `(2,1)`
- methods: `m1_hh_hva, m3_adapt_paop_std, m4_adapt_paop_lf_std`
- selected_config: `blocked,open,binary`

## Scout Results

| config | mean best ΔE | mean best runtime_s | timeouts |
|---|---:|---:|---:|
| blocked,open,binary | 2.374061e+00 | 104.6 | 0 |

## Best Per Method

| method | best ΔE | E_best | E_exact_sector | seed | rung | runtime_s |
|---|---:|---:|---:|---:|---|---:|
| m1_hh_hva | 1.540442e+00 | 1.013504299097 | -0.526937719766 | 101 | hva_r2_mi3000 | 78.6 |
| m3_adapt_paop_std | 2.790870e+00 | 2.263932023078 | -0.526937719766 | 101 | adapt_d80_mi3000 | 117.8 |
| m4_adapt_paop_lf_std | 2.790870e+00 | 2.263932023078 | -0.526937719766 | 101 | adapt_d80_mi3000 | 115.8 |

## Rung Progress

| method | rung | n_runs | n_ok | best ΔE | median ΔE | median runtime_s | decision |
|---|---|---:|---:|---:|---:|---:|---|
| m1_hh_hva | hva_r2_mi3000 | 1 | 1 | 1.540442e+00 | 1.540442e+00 | 78.6 | continue |
| m3_adapt_paop_std | adapt_d80_mi3000 | 1 | 1 | 2.790870e+00 | 2.790870e+00 | 117.8 | continue |
| m4_adapt_paop_lf_std | adapt_d80_mi3000 | 1 | 1 | 2.790870e+00 | 2.790870e+00 | 115.8 | continue |
