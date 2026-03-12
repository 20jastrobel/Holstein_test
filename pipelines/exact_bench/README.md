# Exact-Metrics Benchmark

This folder hosts ED-based benchmarking and validation helpers used with the hardcoded and noise-validation toolchain.

## Current purpose

- Exact references and sweep helpers for cross-checking VQE/ADAPT + time-propagation workflows.
- CFQM/Suzuki benchmark suites for cost-vs-accuracy analysis.
- Noisy and hardware-facing validation utilities with shared oracle logic.

## Key modules

| File | Purpose |
|------|---------|
| `cross_check_suite.py` | Exact benchmark matrix across ansatz/VQE modes with JSON + PDF outputs |
| `cfqm_vs_suzuki_efficiency_suite.py` | Error-vs-cost CFQM/Suzuki benchmarking suite |
| `cfqm_vs_suzuki_qproc_proxy_benchmark.py` | Processor-proxy benchmark runner and summary artifacts |
| `hh_noise_hardware_validation.py` | HH noisy/hardware validation runner (ideal, shots, Aer, runtime modes) |
| `hh_noise_robustness_seq_report.py` | Sequential HH robustness + final/trajectory report workflow |
| `hh_noise_model_repo_guide.py` | Code/docs guide generator for the noise-model stack |
| `benchmark_metrics_proxy.py` | Shared benchmark metric helpers |
| `statevector_kernels.py` | Shared statevector kernels used by exact-bench scripts |
| `noise_oracle_runtime.py` | Noise-oracle helper runtime for validation suites |
| `noise_aer_builders.py` | Noisy Aer circuit/sampler construction helpers |
| `noise_model_spec.py` | Noise model/spec metadata definitions |
| `noise_snapshot.py` | Structured snapshotting helpers for noise diagnostics |

## Relationship to `test/`

- `test/` validates implementation correctness (unit + integration).
- `exact_bench/` produces reference data, diagnostics, and reproducibility artifacts for physics-level runs.

Example: `test/test_ed_crosscheck.py` checks ED numerics; the exact-bench scripts are where you run the broader benchmark workflows.
