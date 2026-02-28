# HH ADAPT-VQE Validation Suite (HVA)

This suite validates the ADAPT-VQE pipeline on the Hubbard-Holstein (HH) model
for `L=2` and `L=3`.

## Scope

- Problem: `--problem hh`
- Default Ansatz/pool: `--adapt-pool paop_full`
- Reference: exact sector-filtered ground state energy from `exact_ground_energy_sector`
- Acceptance threshold:
  - `L=2`: `abs_delta_e < 1e-2`
  - `L=3`: `abs_delta_e < 1e-1`
  - Can be overridden with `ADAPT_HH_TOL_L2` / `ADAPT_HH_TOL_L3`

## Suite environment overrides

- `ADAPT_HH_TOL_L2`, `ADAPT_HH_TOL_L3`
- `ADAPT_HH_MAXITER_L2`, `ADAPT_HH_MAXITER_L3`
- `ADAPT_HH_MAX_DEPTH_L2`, `ADAPT_HH_MAX_DEPTH_L3`
- `ADAPT_HH_EPS_GRAD_L2`, `ADAPT_HH_EPS_GRAD_L3`
- `ADAPT_HH_EPS_ENERGY_L2`, `ADAPT_HH_EPS_ENERGY_L3`

## Required command entrypoints

- Single pass:
  - `Adapt-VQE-Pipeline/hh_adapt_vqe_validation_suite/run_hh_adapt_vqe_validation.sh`
- Stress loop (default 25 attempts):
  - `Adapt-VQE-Pipeline/hh_adapt_vqe_validation_suite/run_hh_adapt_vqe_validation_stress.sh`

## Default HH physics and ADAPT parameters

- `L`: `2` and `3`
- `t`: `0.2`
- `u`: `0.2`
- `omega0`: `0.2`
- `g-ep`: `0.2`
- `n-ph-max`: `1`
- `boson-encoding`: `binary`
- `boundary`: `open`
- `ordering`: `blocked`
- `adapt-max-depth`: `120`
- `adapt-maxiter`: `1200` (`L=2`) and `20` (`L=3` by default via `ADAPT_HH_MAXITER_L*`)
- `adapt-eps-grad`: `1e-12`
- `adapt-eps-energy`: `1e-10`
- `initial-state-source`: `hf`
- `--adapt-no-repeats`
- `--adapt-no-finite-angle-fallback`
- `--skip-pdf`
- explicit `--output-json`

## Artifacts

All suite outputs are written under:
`Adapt-VQE-Pipeline/hh_adapt_vqe_validation_suite/artifacts/`

The stress script writes per-attempt artifacts in `attempt_<n>/` subdirectories.

## Environment assumptions

Tests assume this repo layout and use the in-repo python modules:
- `Adapt-VQE-Pipeline/pipelines/hardcoded_adapt_pipeline.py`
- `pydephasing.quantum.*`

No changes are required outside `Adapt-VQE-Pipeline/` to run this suite.
