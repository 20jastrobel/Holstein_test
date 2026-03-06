# CFQM vs Suzuki proxy benchmark

## Run metadata

- problem=hubbard
- L=2
- t=1.0
- u=4.0
- dv=0.0
- t_final=1.0
- reference_steps=64
- methods=suzuki2,cfqm4
- steps_grid=16,32

## Metrics (vs piecewise_exact reference)

- `S` is the macro-step count (`trotter_steps`), not a cost axis.
| meth | S | final_err | max_err | cx | term_exp | runtime_s |
|---|---:|---:|---:|---:|---:|---:|
| suzuki2 |  16 | 5.385082e-06 | 5.385082e-06 |    384 |     352 |   0.86 |
| suzuki2 |  32 | 3.373690e-07 | 3.373690e-07 |    768 |     704 |   0.88 |
| cfqm4  |  16 | 3.373690e-07 | 3.373690e-07 |    768 |     704 |   1.03 |
| cfqm4  |  32 | 2.109819e-08 | 2.109819e-08 |   1536 |    1408 |   1.22 |

## Ordering

- Pareto front is reported by increasing CX cost; each entry keeps best-so-far error.
- suzuki2-16 err=5.385082e-06 cx=384
- suzuki2-32 err=3.373690e-07 cx=768
- cfqm4-32 err=2.109819e-08 cx=1536

## Note

- Formatting rule: compact markdown rows use fixed-width numeric
- fields to prevent visual spillover in rendered artifacts.
