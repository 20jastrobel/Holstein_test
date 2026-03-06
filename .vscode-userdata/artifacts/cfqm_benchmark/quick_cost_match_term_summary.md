# CFQM vs Suzuki proxy benchmark

## Run metadata

- problem=hubbard
- L=2
- t=1.0
- u=4.0
- dv=0.0
- t_final=1.0
- reference_steps=64
- methods=suzuki2,cfqm4,cfqm6
- steps_grid=16,32

## Metrics (vs piecewise_exact reference)

- `S` is the macro-step count (`trotter_steps`), not a cost axis.
| meth | S | final_err | max_err | cx | term_exp | runtime_s |
|---|---:|---:|---:|---:|---:|---:|
| suzuki2 |  16 | 5.385082e-06 | 5.385082e-06 |    384 |     352 |   2.48 |
| suzuki2 |  32 | 3.373690e-07 | 3.373690e-07 |    768 |     704 |   2.44 |
| cfqm4  |  16 | 3.373690e-07 | 3.373690e-07 |    768 |     704 |   2.54 |
| cfqm4  |  32 | 2.109819e-08 | 2.109819e-08 |   1536 |    1408 |   2.83 |
| cfqm6  |  16 | 7.109787e-08 | 7.109787e-08 |   1920 |    1760 |   2.80 |
| cfqm6  |  32 | 4.443073e-09 | 4.443073e-09 |   3840 |    3520 |   3.15 |

## Ordering

- Pareto front is reported by increasing CX cost; each entry keeps best-so-far error.
- suzuki2-16 err=5.385082e-06 cx=384
- suzuki2-32 err=3.373690e-07 cx=768
- cfqm4-32 err=2.109819e-08 cx=1536
- cfqm6-32 err=4.443073e-09 cx=3840

## Equal-cost comparison

- Metric: term_exp_count_total (tolerance=0.0)
- target[1] term_exp_count_total=3.520000e+02
| meth | matched_S | matched_metric | final_err | delta | exact_match |
|---|---:|---:|---:|---:|---|
| cfqm4 |        16 | 7.040000e+02 | 3.373690e-07 | 3.520000e+02 | False |
| cfqm6 |        16 | 1.760000e+03 | 7.109787e-08 | 1.408000e+03 | False |
| suzuki2 |        16 | 3.520000e+02 | 5.385082e-06 | 0.000e+00 | True |
- target[2] term_exp_count_total=7.040000e+02
| meth | matched_S | matched_metric | final_err | delta | exact_match |
|---|---:|---:|---:|---:|---|
| cfqm4 |        16 | 7.040000e+02 | 3.373690e-07 | 0.000e+00 | True |
| cfqm6 |        16 | 1.760000e+03 | 7.109787e-08 | 1.056000e+03 | False |
| suzuki2 |        32 | 7.040000e+02 | 3.373690e-07 | 0.000e+00 | True |
- target[3] term_exp_count_total=1.408000e+03
| meth | matched_S | matched_metric | final_err | delta | exact_match |
|---|---:|---:|---:|---:|---|
| cfqm4 |        32 | 1.408000e+03 | 2.109819e-08 | 0.000e+00 | True |
| cfqm6 |        16 | 1.760000e+03 | 7.109787e-08 | 3.520000e+02 | False |
| suzuki2 |        32 | 7.040000e+02 | 3.373690e-07 | 7.040000e+02 | False |
- target[4] term_exp_count_total=1.760000e+03
| meth | matched_S | matched_metric | final_err | delta | exact_match |
|---|---:|---:|---:|---:|---|
| cfqm4 |        32 | 1.408000e+03 | 2.109819e-08 | 3.520000e+02 | False |
| cfqm6 |        16 | 1.760000e+03 | 7.109787e-08 | 0.000e+00 | True |
| suzuki2 |        32 | 7.040000e+02 | 3.373690e-07 | 1.056000e+03 | False |
- target[5] term_exp_count_total=3.520000e+03
| meth | matched_S | matched_metric | final_err | delta | exact_match |
|---|---:|---:|---:|---:|---|
| cfqm4 |        32 | 1.408000e+03 | 2.109819e-08 | 2.112000e+03 | False |
| cfqm6 |        32 | 3.520000e+03 | 4.443073e-09 | 0.000e+00 | True |
| suzuki2 |        32 | 7.040000e+02 | 3.373690e-07 | 2.816000e+03 | False |


## Note

- Formatting rule: compact markdown rows use fixed-width numeric
- fields to prevent visual spillover in rendered artifacts.
