# Hubbard Pipeline Run Guide

This is the comprehensive runtime guide for the simplified repo layout.

Run from the sub-repo root (`Fermi-Hamil-JW-VQE-TROTTER-PIPELINE/`).

---

## Runtime Scripts

| Script | Purpose |
|--------|---------|
| `pipelines/hardcoded_hubbard_pipeline.py` | Hardcoded Hamiltonian, hardcoded VQE, hardcoded Trotter dynamics, optional QPE |
| `pipelines/qiskit_hubbard_baseline_pipeline.py` | Qiskit Hamiltonian, Qiskit VQE, Qiskit Trotter dynamics, optional QPE |
| `pipelines/compare_hardcoded_vs_qiskit_pipeline.py` | Orchestrator — runs both, compares metrics, writes comparison PDFs |
| `pipelines/manual_compare_jsons.py` | Standalone JSON-vs-JSON consistency checker |
| `pipelines/regression_L2_L3.sh` | Automated L=2/L=3 regression harness |

---

## State Source Behavior

`--initial-state-source` supports:

| Value | Behaviour |
|-------|-----------|
| `vqe` | Dynamics starts from that pipeline's own VQE-optimised state |
| `exact` | Dynamics starts from exact ground state (sector-filtered eigendecomposition) |
| `hf` | Dynamics starts from Hartree-Fock reference state |

If you want apples-to-apples hardcoded vs Qiskit from each ansatz, use `--initial-state-source vqe`.

---

## Complete Parameter Reference

### Model Parameters (all three pipelines)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--L` | int | *required* | Number of lattice sites (single pipelines) |
| `--l-values` | str | `"2,3,4,5"` | Comma-separated lattice sizes (compare pipeline only) |
| `--t` | float | `1.0` | Hopping coefficient t |
| `--u` | float | `4.0` | Onsite interaction U |
| `--dv` | float | `0.0` | Uniform local potential term v (H_v = −v n) |
| `--boundary` | choice | `periodic` | Boundary conditions: `periodic` or `open` |
| `--ordering` | choice | `blocked` | Qubit ordering: `blocked` or `interleaved` |

### Time-Evolution Parameters (all three pipelines)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--t-final` | float | `20.0` | Final evolution time |
| `--num-times` | int | `201` | Number of output time points |
| `--suzuki-order` | int | `2` | Suzuki–Trotter product-formula order |
| `--trotter-steps` | int | `64` | Number of Trotter steps |
| `--term-order` | choice | `sorted` | Term ordering for Trotter product. Hardcoded: `native\|sorted`. Qiskit: `qiskit\|sorted` |

### Time-Dependent Drive Parameters (all three pipelines)

These flags control a Gaussian-envelope sinusoidal onsite density drive:

$$v(t) = A \cdot \sin(\omega t + \phi) \cdot \exp\!\Big(-\frac{(t - t_0)^2}{2\,\bar{t}^{\,2}}\Big)$$

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--enable-drive` | flag | `false` | Enable the time-dependent drive. When absent, no drive flags are forwarded and behaviour is identical to the static case. |
| `--drive-A` | float | `0.0`* | Drive amplitude A. *Compare pipeline default is `1.0`. |
| `--drive-omega` | float | `1.0` | Drive angular frequency ω |
| `--drive-tbar` | float | `1.0`** | Drive Gaussian half-width t̄ (must be > 0). **Compare pipeline default is `5.0`. |
| `--drive-phi` | float | `0.0` | Drive phase offset φ |
| `--drive-t0` | float | `0.0` | Drive start time t₀ |
| `--drive-pattern` | choice | `staggered` | Spatial weight pattern: `staggered`, `dimer_bias`, or `custom` |
| `--drive-custom-s` | str | `null` | JSON array of custom per-site weights, e.g. `'[1.0,-0.5]'`. Required when `--drive-pattern=custom`. |
| `--drive-include-identity` | flag | `false` | Include the identity (global-phase) term from n = (I−Z)/2 decomposition |
| `--drive-time-sampling` | choice | `midpoint` | Time-sampling rule within each Trotter slice: `midpoint`, `left`, or `right` |
| `--exact-steps-multiplier` | int | `1` | Reference-propagator refinement: N_ref = multiplier × trotter_steps. Has no effect when drive is disabled (static reference uses eigendecomposition). |

### VQE Parameters

**Single pipelines** (`hardcoded_hubbard_pipeline.py`, `qiskit_hubbard_baseline_pipeline.py`):

| Flag | Type | Default (HC) | Default (QK) | Description |
|------|------|-------------|-------------|-------------|
| `--vqe-reps` | int | `2` | `2` | Number of ansatz repetitions (circuit depth) |
| `--vqe-restarts` | int | `1` | `3` | Number of independent VQE optimisation restarts |
| `--vqe-seed` | int | `7` | `7` | Random seed for VQE parameter initialisation |
| `--vqe-maxiter` | int | `120` | `120` | Maximum optimiser iterations per restart |

**Compare pipeline** (separate knobs for each sub-pipeline):

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--hardcoded-vqe-reps` | int | `2` | HC ansatz repetitions |
| `--hardcoded-vqe-restarts` | int | `3` | HC restarts |
| `--hardcoded-vqe-seed` | int | `7` | HC seed |
| `--hardcoded-vqe-maxiter` | int | `600` | HC max iterations |
| `--qiskit-vqe-reps` | int | `2` | QK ansatz repetitions |
| `--qiskit-vqe-restarts` | int | `3` | QK restarts |
| `--qiskit-vqe-seed` | int | `7` | QK seed |
| `--qiskit-vqe-maxiter` | int | `600` | QK max iterations |

### QPE Parameters (all three pipelines)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--qpe-eval-qubits` | int | `6` (single) / `5` (compare) | Number of evaluation qubits for QPE |
| `--qpe-shots` | int | `1024` (single) / `256` (compare) | Number of measurement shots |
| `--qpe-seed` | int | `11` | Random seed for QPE simulation |
| `--skip-qpe` | flag | `false` | Skip QPE execution entirely (marks payload as skipped) |

### Initial State

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--initial-state-source` | choice | `vqe` (compare) / `exact` (single) | State for dynamics: `exact`, `vqe`, or `hf` |

### Output / Artifact Controls

**Single pipelines:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-json` | path | auto | Path for output JSON |
| `--output-pdf` | path | auto | Path for output PDF |
| `--skip-pdf` | flag | `false` | Skip PDF generation |

**Compare pipeline:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--artifacts-dir` | path | `artifacts/` | Directory for all generated outputs |
| `--run-pipelines` | flag | `true` | Run both sub-pipelines (use `--no-run-pipelines` to reuse existing JSONs) |
| `--with-per-l-pdfs` | flag | `false` | Include per-L comparison pages in bundle and emit standalone per-L PDFs |

### Drive Amplitude Comparison (compare pipeline only)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--drive-amplitudes` | str | `"0.0,0.2"` | Comma-separated pair `A0,A1`. A0 is the trivial amplitude for the safe-test; A1 is the active amplitude. |
| `--with-drive-amplitude-comparison-pdf` | flag | `false` | Generate amplitude-comparison PDF per L. Runs both pipelines 3× per L (disabled, A0, A1 = 6 sub-runs). Includes safe-test page and VQE delta page. |

---

## Full CLI (defaults)

### Hardcoded pipeline

```bash
python pipelines/hardcoded_hubbard_pipeline.py --help
```

Defaults:

- `--t 1.0 --u 4.0 --dv 0.0`
- `--boundary periodic --ordering blocked`
- `--t-final 20.0 --num-times 201 --suzuki-order 2 --trotter-steps 64`
- `--term-order sorted` (`native|sorted`)
- `--vqe-reps 2 --vqe-restarts 1 --vqe-seed 7 --vqe-maxiter 120`
- `--qpe-eval-qubits 6 --qpe-shots 1024 --qpe-seed 11`
- `--initial-state-source vqe`
- Drive: disabled by default. Enable with `--enable-drive`.

### Qiskit baseline pipeline

```bash
python pipelines/qiskit_hubbard_baseline_pipeline.py --help
```

Defaults:

- `--t 1.0 --u 4.0 --dv 0.0`
- `--boundary periodic --ordering blocked`
- `--t-final 20.0 --num-times 201 --suzuki-order 2 --trotter-steps 64`
- `--term-order sorted` (`qiskit|sorted`)
- `--vqe-reps 2 --vqe-restarts 3 --vqe-seed 7 --vqe-maxiter 120`
- `--qpe-eval-qubits 6 --qpe-shots 1024 --qpe-seed 11`
- `--initial-state-source vqe`
- Drive: disabled by default. Enable with `--enable-drive`.

### Compare runner

```bash
python pipelines/compare_hardcoded_vs_qiskit_pipeline.py --help
```

Defaults:

- `--l-values 2,3,4,5`
- `--run-pipelines` (use `--no-run-pipelines` to reuse JSONs)
- `--t 1.0 --u 4.0 --dv 0.0`
- `--boundary periodic --ordering blocked`
- `--t-final 20.0 --num-times 201 --suzuki-order 2 --trotter-steps 64`
- `--hardcoded-vqe-reps 2 --hardcoded-vqe-restarts 3 --hardcoded-vqe-seed 7 --hardcoded-vqe-maxiter 600`
- `--qiskit-vqe-reps 2 --qiskit-vqe-restarts 3 --qiskit-vqe-seed 7 --qiskit-vqe-maxiter 600`
- `--qpe-eval-qubits 5 --qpe-shots 256 --qpe-seed 11`
- `--initial-state-source vqe` (`exact|vqe|hf`)
- `--artifacts-dir artifacts`
- Drive: disabled by default. Enable with `--enable-drive`.
- Amplitude comparison: disabled by default. Enable with `--with-drive-amplitude-comparison-pdf`.
- `--drive-amplitudes "0.0,0.2"` (only used when amplitude comparison is enabled)

---

## Common Commands

### 1) Run full compare for L=2,3,4 with locked heavy settings

```bash
python pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
  --l-values 2,3,4 \
  --artifacts-dir artifacts \
  --initial-state-source vqe \
  --t 1.0 --u 4.0 --dv 0.0 --boundary periodic --ordering blocked \
  --t-final 20.0 --num-times 401 --suzuki-order 2 --trotter-steps 128 \
  --hardcoded-vqe-reps 2 --hardcoded-vqe-restarts 3 --hardcoded-vqe-maxiter 1800 --hardcoded-vqe-seed 7 \
  --qiskit-vqe-reps 2 --qiskit-vqe-restarts 3 --qiskit-vqe-maxiter 1800 --qiskit-vqe-seed 7 \
  --qpe-eval-qubits 8 --qpe-shots 4096 --qpe-seed 11 \
  --with-per-l-pdfs
```

### 2) Rebuild comparison PDFs/summary from existing JSON

```bash
python pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
  --l-values 2,3,4 \
  --artifacts-dir artifacts \
  --no-run-pipelines \
  --with-per-l-pdfs
```

### 3) Run hardcoded pipeline only

```bash
python pipelines/hardcoded_hubbard_pipeline.py \
  --L 3 --initial-state-source vqe \
  --output-json artifacts/json/H_L3_static_t1.0_U4.0_S64.json \
  --output-pdf artifacts/pdf/H_L3_static_t1.0_U4.0_S64.pdf
```

### 4) Run Qiskit baseline only

```bash
python pipelines/qiskit_hubbard_baseline_pipeline.py \
  --L 3 --initial-state-source vqe \
  --output-json artifacts/json/Q_L3_static_t1.0_U4.0_S64.json \
  --output-pdf artifacts/pdf/Q_L3_static_t1.0_U4.0_S64.pdf
```

### 5) Run with time-dependent drive enabled

```bash
python pipelines/hardcoded_hubbard_pipeline.py \
  --L 2 --initial-state-source vqe \
  --enable-drive --drive-A 0.5 --drive-omega 2.0 --drive-tbar 3.0 \
  --drive-pattern dimer_bias --drive-time-sampling midpoint \
  --t-final 10.0 --num-times 101 --trotter-steps 64 \
  --exact-steps-multiplier 4 \
  --output-json artifacts/json/H_L2_vt_t1.0_U4.0_S64.json \
  --output-pdf artifacts/pdf/H_L2_vt_t1.0_U4.0_S64.pdf
```

### 6) Compare pipeline with drive enabled

```bash
python pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
  --l-values 2,3 --run-pipelines --enable-drive \
  --drive-A 0.5 --drive-omega 2.0 --drive-tbar 3.0 \
  --drive-pattern dimer_bias --drive-time-sampling midpoint \
  --t-final 10.0 --num-times 101 --trotter-steps 64 \
  --exact-steps-multiplier 4 --skip-qpe \
  --with-per-l-pdfs
```

### 7) Amplitude comparison PDF (safe-test + VQE delta)

```bash
python pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
  --l-values 2 --run-pipelines --enable-drive \
  --drive-pattern dimer_bias --drive-omega 2.0 --drive-tbar 2.0 \
  --t-final 2.0 --num-times 21 --trotter-steps 32 --skip-qpe \
  --drive-amplitudes '0.0,0.2' \
  --with-drive-amplitude-comparison-pdf
```

This runs 8 sub-pipeline invocations per L (2 main + 6 amplitude comparison) and generates:
- `pdf/amp_{tag}.pdf` — 5-page PDF with safe-test plots, VQE bars, text summary
- `json/amp_{tag}_metrics.json` — machine-readable safe-test + VQE delta metrics

### 8) Run the L=2/L=3 regression harness

```bash
bash pipelines/regression_L2_L3.sh
```

This writes `_reg` JSON/PDF outputs for L=2 and L=3, runs the compare runner, runs
`manual_compare_jsons.py`, and ends with `REGRESSION PASS` or `REGRESSION FAIL`.

### 9) Manual JSON-vs-JSON consistency check

```bash
python pipelines/manual_compare_jsons.py \
  --hardcoded artifacts/json/H_L3_static_t1.0_U4.0_S64.json \
  --qiskit artifacts/json/Q_L3_static_t1.0_U4.0_S64.json \
  --metrics artifacts/json/HvQ_L3_static_t1.0_U4.0_S64_metrics.json
```

---

## Generated Artifacts

Under `artifacts/` (or the path given by `--artifacts-dir`):

```
artifacts/
├── json/        # All JSON outputs
├── pdf/         # All PDF outputs
└── commands.txt # Exact commands run
```

### Naming convention

Filenames use a **tag** encoding the run config:
`L{L}_{vt|static}_t{t}_U{u}_S{trotter_steps}` — e.g. `L2_static_t1.0_U4.0_S64`.

Prefixes: **H** = hardcoded, **Q** = Qiskit, **HvQ** = comparison, **amp** = amplitude comparison.

### Standard comparison outputs

| File | Description |
|------|-------------|
| `json/H_{tag}.json` | Hardcoded pipeline full output |
| `json/Q_{tag}.json` | Qiskit pipeline full output |
| `json/HvQ_{tag}_metrics.json` | Per-L comparison metrics (fidelity, energy, VQE, QPE deltas) |
| `json/HvQ_summary.json` | Summary across all L values |
| `pdf/HvQ_bundle.pdf` | Multi-page comparison bundle PDF |
| `pdf/HvQ_{tag}.pdf` | Per-L standalone comparison PDF (with `--with-per-l-pdfs`) |
| `commands.txt` | Exact commands that were executed |

### Amplitude comparison outputs (with `--with-drive-amplitude-comparison-pdf`)

| File | Description |
|------|-------------|
| `pdf/amp_{tag}.pdf` | 5-page PDF: command, settings, safe-test plots, VQE bars, text summary |
| `json/amp_{tag}_metrics.json` | Machine-readable: `safe_test`, `delta_vqe_hc_minus_qk_at_A0`, `delta_vqe_hc_minus_qk_at_A1` |
| `json/amp_H_{tag}_{slug}.json` | HC intermediate outputs (slug = `disabled`, `A0`, `A1`) |
| `json/amp_Q_{tag}_{slug}.json` | QK intermediate outputs (slug = `disabled`, `A0`, `A1`) |

### VQE visibility

- Per-L comparison PDFs include explicit VQE bar charts (exact filtered / HC VQE / QK VQE).
- Bundle PDF includes VQE comparison pages and per-L VQE pages (with `--with-per-l-pdfs`).
- Amplitude comparison PDF adds cross-amplitude VQE bar charts with ΔE = HC_VQE − QK_VQE annotations.

### Metrics JSON schema (amplitude comparison)

```json
{
  "generated_utc": "2026-02-21T00:11:04.676856+00:00",
  "L": 2,
  "A0": 0.0,
  "A1": 0.2,
  "safe_test": {
    "passed": true,
    "threshold": 1e-10,
    "hc": { "max_fidelity_delta": 6.66e-16, "max_energy_delta": 0.0 },
    "qk": { "max_fidelity_delta": 8.88e-16, "max_energy_delta": 4.44e-16 }
  },
  "delta_vqe_hc_minus_qk_at_A0": -1.888e-09,
  "delta_vqe_hc_minus_qk_at_A1": -1.888e-09
}
```
