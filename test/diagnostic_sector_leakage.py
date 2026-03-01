#!/usr/bin/env python3
"""Quick diagnostic: does the Trotter propagation leak out of the
particle-number sector at L=2 vs L=3?

# SMOKE TEST — intentionally weak settings (diagnostic only, not convergence test)

Measures:
  1. sector_weight: sum |<basis_i|psi_trot>|^2 over sector basis states
  2. norm_psi: should stay ~1.0 (Trotter is unitary term-by-term)
  3. energy_trotter vs energy_exact: drift = Trotter error, not sector leak

If sector_weight drops significantly below 1.0, sector leakage is confirmed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hubbard_latex_python_pairs import build_hubbard_hamiltonian
from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
from src.quantum.vqe_latex_python_pairs import (
    hamiltonian_matrix,
    half_filled_num_particles,
    apply_pauli_rotation,
    expval_pauli_polynomial,
    HubbardTermwiseAnsatz,
    vqe_minimize,
)

import pipelines.hardcoded.hubbard_pipeline as hp


def measure_sector_weight(psi: np.ndarray, sector_indices: np.ndarray) -> float:
    """Fraction of |psi|^2 that lives in the sector."""
    return float(np.sum(np.abs(psi[sector_indices]) ** 2))


def run_diagnostic(L: int) -> dict:
    t_hop, U, dv = 1.0, 4.0, 0.0
    boundary, ordering = "periodic", "blocked"
    num_particles = half_filled_num_particles(L)
    nq = 2 * L

    print(f"\n{'='*60}")
    print(f"  L={L}  nq={nq}  Hilbert dim=2^{nq}={2**nq}")
    print(f"  sector = n_up={num_particles[0]}, n_dn={num_particles[1]}")
    print(f"{'='*60}")

    # Build Hamiltonian
    H = build_hubbard_hamiltonian(L, t_hop, U, v=dv, indexing=ordering, pbc=(boundary == "periodic"))
    hmat = hamiltonian_matrix(H, nq)

    # Sector indices
    sector_indices = hp._sector_basis_indices(L, num_particles, ordering)
    print(f"  Sector size: {len(sector_indices)} out of {2**nq} basis states")
    print(f"  Sector fraction: {len(sector_indices) / 2**nq:.4f}")

    # Exact ground state (sector-filtered)
    gs_energy, psi_gs = hp._exact_ground_state_sector_filtered(
        hmat, L, num_particles, ordering,
    )
    print(f"  Exact GS energy (filtered): {gs_energy:.8f}")
    print(f"  GS sector weight: {measure_sector_weight(psi_gs, sector_indices):.12f}")

    # Quick VQE — intentionally light
    # SMOKE TEST — intentionally weak settings
    psi_ref = hartree_fock_statevector(L, num_particles, indexing=ordering)
    ansatz = HubbardTermwiseAnsatz(L, t_hop, U, reps=1, indexing=ordering,
                                    pbc=(boundary == "periodic"))
    vqe_res = vqe_minimize(H, ansatz, psi_ref, restarts=1, seed=7, maxiter=200)
    psi_vqe = ansatz.prepare_state(vqe_res.theta, psi_ref)
    vqe_sector_weight = measure_sector_weight(psi_vqe, sector_indices)
    print(f"  VQE energy: {vqe_res.energy:.8f}  (delta={abs(vqe_res.energy - gs_energy):.2e})")
    print(f"  VQE sector weight: {vqe_sector_weight:.12f}")

    # Use exact GS as initial state (to isolate Trotter error from VQE error)
    psi_init = psi_gs.copy()

    # Collect Hamiltonian terms for Trotter (use pipeline's collector)
    _native_order, coeff_map = hp._collect_hardcoded_terms_exyz(H)
    sorted_terms = sorted(coeff_map.items(), key=lambda kv: kv[0])

    # Trotter propagation — light settings
    trotter_steps = 32  # SMOKE TEST — intentionally weak
    t_final = 10.0
    n_times = 51
    times = np.linspace(0.0, t_final, n_times)
    dt = t_final / trotter_steps

    # Precompute exact evolution via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(hmat)

    results = []
    psi_trot = psi_init.copy()

    for ti, t_val in enumerate(times):
        # Exact state at time t
        phases = np.exp(-1j * eigvals * t_val)
        psi_exact = eigvecs @ (phases * (eigvecs.conj().T @ psi_init))

        # Measure
        sw_trot = measure_sector_weight(psi_trot, sector_indices)
        sw_exact = measure_sector_weight(psi_exact, sector_indices)
        norm_trot = float(np.linalg.norm(psi_trot))
        e_trot = float(np.real(psi_trot.conj() @ hmat @ psi_trot))
        e_exact = float(np.real(psi_exact.conj() @ hmat @ psi_exact))
        overlap = float(np.abs(np.vdot(psi_exact, psi_trot)) ** 2)

        results.append({
            "time": round(float(t_val), 6),
            "sector_weight_trotter": sw_trot,
            "sector_weight_exact": sw_exact,
            "norm_trotter": norm_trot,
            "energy_trotter": e_trot,
            "energy_exact": e_exact,
            "state_overlap": overlap,
        })

        if ti == 0 or ti == n_times - 1 or ti % 10 == 0:
            print(f"  t={t_val:6.2f}  sec_wt_trot={sw_trot:.10f}  "
                  f"sec_wt_exact={sw_exact:.10f}  "
                  f"norm={norm_trot:.10f}  "
                  f"E_trot={e_trot:.6f}  overlap={overlap:.8f}")

        # Advance Trotter to next observation time
        if ti < n_times - 1:
            t_next = times[ti + 1]
            # How many Trotter steps between t_val and t_next?
            steps_here = max(1, int(round((t_next - t_val) / dt)))
            dt_actual = (t_next - t_val) / steps_here
            for _ in range(steps_here):
                for label, coeff in sorted_terms:
                    if abs(coeff) < 1e-15:
                        continue
                    angle = -float(np.real(coeff)) * dt_actual
                    psi_trot = apply_pauli_rotation(psi_trot, label, angle)

    # Summary
    sw_all = [r["sector_weight_trotter"] for r in results]
    sw_min = min(sw_all)
    sw_final = sw_all[-1]
    overlap_final = results[-1]["state_overlap"]

    summary = {
        "L": L,
        "nq": nq,
        "hilbert_dim": 2 ** nq,
        "sector_size": len(sector_indices),
        "trotter_steps": trotter_steps,
        "t_final": t_final,
        "n_times": n_times,
        "gs_energy": gs_energy,
        "sector_weight_min": sw_min,
        "sector_weight_final": sw_final,
        "sector_weight_exact_final": results[-1]["sector_weight_exact"],
        "state_overlap_final": overlap_final,
        "energy_drift": abs(results[-1]["energy_trotter"] - results[0]["energy_trotter"]),
    }

    print(f"\n  SUMMARY L={L}:")
    print(f"    sector_weight_min    = {sw_min:.12f}")
    print(f"    sector_weight_final  = {sw_final:.12f}")
    print(f"    state_overlap_final  = {overlap_final:.8f}")
    print(f"    energy_drift         = {summary['energy_drift']:.2e}")

    if sw_min < 0.999:
        print(f"    ⚠️  SECTOR LEAKAGE DETECTED: min sector weight = {sw_min:.6f}")
    else:
        print(f"    ✅  Sector weight stays ≥ 0.999")

    return summary


if __name__ == "__main__":
    t0 = time.perf_counter()
    results = {}
    for L in [2, 3]:
        results[f"L={L}"] = run_diagnostic(L)

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")

    # Compare
    for key, s in results.items():
        print(f"\n  {key}: sector_wt_min={s['sector_weight_min']:.10f}  "
              f"overlap_final={s['state_overlap_final']:.8f}  "
              f"E_drift={s['energy_drift']:.2e}")
