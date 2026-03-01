# Exact-Metrics Benchmark

Independent exact-diagonalization (ED) references and accuracy validation tools
for benchmarking the hardcoded quantum algorithms.

## Key properties

- **No Qiskit dependency** — pure numpy / scipy
- **Accuracy oracle** for VQE, Trotter, and future QPE results
- Produces reference eigenvalues, sector-filtered ground states, fidelity benchmarks

## Planned contents

| File | Purpose |
|------|---------|
| `ed_reference_sweep.py` | ED across parameter sweeps (L, t, U, g, ω) → reference JSON |
| `restrict_paulipoly.py` | Sector-restricted Hamiltonian matrix (never builds full 2^n) |
| `accuracy_gate.py` | Load VQE JSON → compare against ED reference → pass/fail |

## Relationship to `test/`

- `test/` verifies **implementation correctness** (unit + integration)
- `exact_bench/` produces **reference data** and **physics-level accuracy reports**

Example:
- `test/test_ed_crosscheck.py` → "does the ED module compute correct eigenvalues?"
- `exact_bench/ed_reference_sweep.py` → "here are the reference eigenvalues for L=2..6, used to gate VQE accuracy"
