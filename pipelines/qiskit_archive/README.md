# Qiskit Archive — Benchmark Only

> **⚠️  This folder is archival.** Files here require Qiskit and are kept for
> reference/comparison benchmarking against the production hardcoded pipelines.
> They are **not** part of the production path.

## Contents

| File | Purpose |
|------|---------|
| `qiskit_baseline.py` | Qiskit-based Hubbard(-Holstein) VQE + Trotter baseline |
| `l2_dual_method.py` | L=2 VQE + exact dynamics, dual-method comparison |
| `compare_hc_vs_qk.py` | Orchestrates hardcoded vs Qiskit side-by-side comparison |
| `hf_circuit.py` | Qiskit `QuantumCircuit` builder for Hartree–Fock reference (dead code, preserved for reference) |
| `DESIGN_NOTE_TIMEDEP.md` | Design note on time-dependent Qiskit baseline |

## When to use

- Sanity-checking a new hardcoded implementation against Qiskit output
- Generating reference data for regression tests
- Historical reference

## When NOT to use

- Any production VQE / Trotter / QPE run
- Anything that needs to run without Qiskit installed
