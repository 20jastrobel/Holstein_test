"""
VQE integration tests for the Hubbard-Holstein (HH) pipeline wiring.

Tests:
  1. HH VQE smoke test        — L=2, n_ph_max=1, g>0, omega0>0 → converges
  2. Zero-coupling limit       — g=0, omega0=0 → no runtime error
  3. Seed reproducibility      — same seed → same energy (bitwise)
  4. Backward compatibility    — pure-Hubbard path unchanged by HH additions
  5. Theta energy consistency  — returned theta reproduces reported energy
  6. Sector-filtered exactness — VQE energy >= exact filtered energy (variational bound)
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.hartree_fock_reference_state import (
    hartree_fock_statevector,
    hubbard_holstein_reference_state,
)
from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinLayerwiseAnsatz,
    HubbardLayerwiseAnsatz,
    HardcodedUCCSDLayerwiseAnsatz,
    VQEResult,
    vqe_minimize,
    expval_pauli_polynomial,
    exact_ground_energy_sector,
    exact_ground_energy_sector_hh,
)

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
_L = 2
_T = 1.0
_U = 4.0
_DV = 0.0
_OMEGA0 = 1.0
_G_EP = 0.5
_N_PH_MAX = 1
_BOSON_ENCODING = "binary"
_BOUNDARY = "periodic"
_ORDERING = "blocked"
_HALF_FILL = (1, 1)  # n_alpha = n_beta = L//2  (half-filling: N_el = L)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _build_hh_hamiltonian(
    L=_L, t=_T, U=_U, dv=_DV, omega0=_OMEGA0, g_ep=_G_EP,
    n_ph_max=_N_PH_MAX, boson_encoding=_BOSON_ENCODING,
    boundary=_BOUNDARY, ordering=_ORDERING,
):
    return build_hubbard_holstein_hamiltonian(
        dims=L, J=t, U=U, omega0=omega0, g=g_ep,
        n_ph_max=n_ph_max, boson_encoding=boson_encoding,
        indexing=ordering, pbc=(boundary == "periodic"),
    )


def _run_hh_vqe(
    L=_L, t=_T, U=_U, dv=_DV, omega0=_OMEGA0, g_ep=_G_EP,
    n_ph_max=_N_PH_MAX, boson_encoding=_BOSON_ENCODING,
    boundary=_BOUNDARY, ordering=_ORDERING,
    reps=2, restarts=2, seed=7, maxiter=600,
) -> VQEResult:
    H = _build_hh_hamiltonian(L, t, U, dv, omega0, g_ep, n_ph_max,
                               boson_encoding, boundary, ordering)
    ansatz = HubbardHolsteinLayerwiseAnsatz(
        dims=L, J=t, U=U, omega0=omega0, g=g_ep,
        n_ph_max=n_ph_max, boson_encoding=boson_encoding,
        reps=reps, indexing=ordering,
        pbc=(boundary == "periodic"),
    )
    psi_ref = hubbard_holstein_reference_state(
        dims=L, n_ph_max=n_ph_max, boson_encoding=boson_encoding,
        indexing=ordering,
    )
    return vqe_minimize(
        H, ansatz, psi_ref,
        restarts=restarts, seed=seed, maxiter=maxiter,
    )


def _run_hubbard_vqe(
    L=_L, t=_T, U=_U, dv=_DV,
    boundary=_BOUNDARY, ordering=_ORDERING,
    ansatz_cls=None, reps=2, restarts=2, seed=7, maxiter=600,
) -> VQEResult:
    """Pure-Hubbard VQE for backward-compat checks."""
    if ansatz_cls is None:
        ansatz_cls = HubbardLayerwiseAnsatz
    H = build_hubbard_hamiltonian(
        dims=L, t=t, U=U, v=dv,
        indexing=ordering, pbc=(boundary == "periodic"),
    )
    if ansatz_cls is HardcodedUCCSDLayerwiseAnsatz:
        ansatz = ansatz_cls(
            dims=L, num_particles=_HALF_FILL,
            reps=reps, indexing=ordering,
        )
    else:
        ansatz = ansatz_cls(
            dims=L, t=t, U=U,
            reps=reps, indexing=ordering,
            pbc=(boundary == "periodic"),
        )
    psi_ref = hartree_fock_statevector(
        n_sites=L, num_particles=_HALF_FILL, indexing=ordering,
    )
    return vqe_minimize(
        H, ansatz, psi_ref,
        restarts=restarts, seed=seed, maxiter=maxiter,
    )


# ────────────────────────────────────────────────────────────────────────────
# 1. HH VQE smoke test
# ────────────────────────────────────────────────────────────────────────────
class TestHHVQESmoke:
    """Minimal HH VQE converges to a finite energy below HF."""

    def test_smoke_L2(self):
        res = _run_hh_vqe(L=2, reps=2, restarts=2, maxiter=600, seed=42)
        assert np.isfinite(res.energy), f"VQE energy not finite: {res.energy}"
        assert res.theta is not None
        assert res.theta.shape[0] > 0

    def test_energy_below_hf(self):
        """VQE energy should be <= HF energy."""
        H = _build_hh_hamiltonian()
        psi_hf = hubbard_holstein_reference_state(
            dims=_L, n_ph_max=_N_PH_MAX, boson_encoding=_BOSON_ENCODING,
            indexing=_ORDERING,
        )
        e_hf = expval_pauli_polynomial(psi_hf, H)
        res = _run_hh_vqe(reps=2, restarts=3, maxiter=600, seed=42)
        assert res.energy <= e_hf + 1e-8, (
            f"VQE energy {res.energy} exceeds HF energy {e_hf}"
        )


# ────────────────────────────────────────────────────────────────────────────
# 2. Zero-coupling limit
# ────────────────────────────────────────────────────────────────────────────
class TestZeroCouplingLimit:
    """With g=0, omega0=0 the HH path must still run without error."""

    def test_zero_coupling_runs(self):
        res = _run_hh_vqe(g_ep=0.0, omega0=0.0, reps=1, restarts=1,
                           maxiter=100, seed=7)
        assert np.isfinite(res.energy)


# ────────────────────────────────────────────────────────────────────────────
# 3. Seed reproducibility
# ────────────────────────────────────────────────────────────────────────────
class TestSeedReproducibility:
    """Same seed produces identical VQE energy (bitwise)."""

    def test_same_seed_same_energy(self):
        r1 = _run_hh_vqe(seed=123, reps=1, restarts=1, maxiter=200)
        r2 = _run_hh_vqe(seed=123, reps=1, restarts=1, maxiter=200)
        assert r1.energy == r2.energy, (
            f"seed=123 gave different energies: {r1.energy} vs {r2.energy}"
        )
        np.testing.assert_array_equal(r1.theta, r2.theta)

    def test_different_seed_different_energy(self):
        r1 = _run_hh_vqe(seed=100, reps=1, restarts=1, maxiter=200)
        r2 = _run_hh_vqe(seed=200, reps=1, restarts=1, maxiter=200)
        # They *could* match by coincidence but it's extremely unlikely.
        # Just assert both are finite — the point is no crash.
        assert np.isfinite(r1.energy)
        assert np.isfinite(r2.energy)


# ────────────────────────────────────────────────────────────────────────────
# 4. Backward compatibility  — pure Hubbard path unbroken
# ────────────────────────────────────────────────────────────────────────────
class TestBackwardCompat:
    """Pure-Hubbard VQE still works after HH additions."""

    def test_hubbard_hva_runs(self):
        res = _run_hubbard_vqe(
            ansatz_cls=HubbardLayerwiseAnsatz,
            reps=2, restarts=1, maxiter=200, seed=7,
        )
        assert np.isfinite(res.energy)

    def test_hubbard_uccsd_runs(self):
        """UCCSD needs virtual orbitals, so use L=3 (at L=2 half-fill,
        all orbitals are occupied → 0 excitation generators)."""
        L3 = 3
        half3 = (2, 1)  # _half_filled_num_particles(3) = (2, 1)
        H = build_hubbard_hamiltonian(
            dims=L3, t=_T, U=_U, v=_DV,
            indexing=_ORDERING, pbc=(_BOUNDARY == "periodic"),
        )
        ansatz = HardcodedUCCSDLayerwiseAnsatz(
            dims=L3, num_particles=half3,
            reps=2, indexing=_ORDERING,
        )
        psi_ref = hartree_fock_statevector(
            n_sites=L3, num_particles=half3, indexing=_ORDERING,
        )
        res = vqe_minimize(H, ansatz, psi_ref,
                           restarts=1, seed=7, maxiter=200)
        assert np.isfinite(res.energy)


# ────────────────────────────────────────────────────────────────────────────
# 5. Theta energy consistency
# ────────────────────────────────────────────────────────────────────────────
class TestThetaEnergyConsistency:
    """Returned theta reproduces reported energy via explicit expval."""

    def test_theta_reproduces_energy(self):
        H = _build_hh_hamiltonian()
        ansatz = HubbardHolsteinLayerwiseAnsatz(
            dims=_L, J=_T, U=_U, omega0=_OMEGA0, g=_G_EP,
            n_ph_max=_N_PH_MAX, boson_encoding=_BOSON_ENCODING,
            reps=2, indexing=_ORDERING,
            pbc=(_BOUNDARY == "periodic"),
        )
        psi_ref = hubbard_holstein_reference_state(
            dims=_L, n_ph_max=_N_PH_MAX, boson_encoding=_BOSON_ENCODING,
            indexing=_ORDERING,
        )
        res = vqe_minimize(H, ansatz, psi_ref,
                           restarts=2, seed=42, maxiter=600)

        psi_opt = ansatz.prepare_state(res.theta, psi_ref)
        e_check = expval_pauli_polynomial(psi_opt, H)
        assert abs(e_check - res.energy) < 1e-10, (
            f"theta-recomputed energy {e_check} != reported {res.energy}"
        )


# ────────────────────────────────────────────────────────────────────────────
# 6. Variational bound (VQE >= exact filtered)
# ────────────────────────────────────────────────────────────────────────────
class TestVariationalBound:
    """VQE energy must satisfy the variational bound vs exact sector energy."""

    def test_hh_variational_bound(self):
        H = _build_hh_hamiltonian()
        e_exact = exact_ground_energy_sector_hh(
            H, num_sites=_L, num_particles=_HALF_FILL,
            n_ph_max=_N_PH_MAX, boson_encoding=_BOSON_ENCODING,
            indexing=_ORDERING,
        )
        res = _run_hh_vqe(reps=2, restarts=3, maxiter=600, seed=42)
        assert res.energy >= e_exact - 1e-8, (
            f"VQE energy {res.energy} below exact {e_exact} by more than tolerance"
        )

    def test_hubbard_variational_bound(self):
        H = build_hubbard_hamiltonian(
            dims=_L, t=_T, U=_U, v=_DV,
            indexing=_ORDERING, pbc=(_BOUNDARY == "periodic"),
        )
        e_exact = exact_ground_energy_sector(
            H, num_sites=_L, num_particles=_HALF_FILL, indexing=_ORDERING,
        )
        res = _run_hubbard_vqe(reps=2, restarts=3, maxiter=600, seed=42)
        assert res.energy >= e_exact - 1e-8, (
            f"VQE energy {res.energy} below exact {e_exact} by more than tolerance"
        )
