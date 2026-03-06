# CFQM vs Suzuki proxy benchmark

## Run metadata

- problem=hubbard
- L=2
- t=1.0
- u=4.0
- dv=0.0
- t_final=4.0
- reference_steps=128
- methods=suzuki2,cfqm4,cfqm6
- steps_grid=16,32

## Metrics (vs piecewise_exact reference)

| meth | S | final_err | max_err | cx | term_exp | runtime_s |
|---|---:|---:|---:|---:|---:|---:|
| suzuki2 |  16 | 1.180204e-03 | 2.174540e-03 |    384 |     352 |   6.21 |
| suzuki2 |  32 | 5.850520e-04 | 5.850520e-04 |    768 |     704 |   8.99 |
| cfqm4  |  16 | 4.993564e-04 | 5.308571e-04 |    768 |     704 |   9.58 |
| cfqm4  |  32 | 1.419797e-04 | 1.419797e-04 |   1536 |    1408 |  16.59 |
| cfqm6  |  16 | 2.554949e-04 | 2.554949e-04 |   1920 |    1760 |  16.53 |
| cfqm6  |  32 | 6.412992e-05 | 6.412992e-05 |   3840 |    3520 |  29.02 |

## Ordering

- Pareto front is reported by increasing CX cost; each entry keeps best-so-far error.
- suzuki2-16 err=1.180204e-03 cx=384
- cfqm4-16 err=4.993564e-04 cx=768
- cfqm4-32 err=1.419797e-04 cx=1536
- cfqm6-32 err=6.412992e-05 cx=3840

## Note

- Formatting rule: compact markdown rows use fixed-width numeric
- fields.to prevent visual spillover in rendered artifacts.
