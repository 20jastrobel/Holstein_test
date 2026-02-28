"""
Acceptance tests for Hubbard-Holstein reference state and ansatz classes.

Checks:
  1. HH reference state shape & norm
  2. Deterministic/reproducible construction
  3. HH layerwise ansatz parameter count scales with reps
  4. prepare_state output norm == 1
  5. Same-theta determinism
  6. Hubbard-only ansatz path unbroken (backward compat)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.quantum.hartree_fock_reference_state import (
    hubbard_holstein_reference_state,
    hartree_fock_statevector,
)
from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_holstein_hamiltonian,
    n_sites_from_dims,
)
from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinLayerwiseAnsatz,
    HubbardTermwiseAnsatz,
    HubbardLayerwiseAnsatz,
    basis_state,
)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture(params=[1, 2, 3])
def n_ph_max(request):
    return request.param


@pytest.fixture(params=[2, 3])
def L(request):
    return request.param


# ────────────────────────────────────────────────────────────────────────────
# 1. HH reference state: shape & norm
# ────────────────────────────────────────────────────────────────────────────

class TestHHReferenceStateShape:
    """HH reference state dimension must be 2^(n_ferm + n_bos)."""

    def test_shape_L2_nph1(self):
        psi = hubbard_holstein_reference_state(dims=2, n_ph_max=1)
        n_ferm = 4
        qpb = boson_qubits_per_site(1)  # 1
        n_total = n_ferm + 2 * qpb       # 6
        assert psi.shape == (1 << n_total,), f"expected dim 2^{n_total}, got {psi.shape}"

    def test_shape_L3_nph3(self):
        psi = hubbard_holstein_reference_state(dims=3, n_ph_max=3)
        n_ferm = 6
        qpb = boson_qubits_per_site(3)  # 2
        n_total = n_ferm + 3 * qpb       # 12
        assert psi.shape == (1 << n_total,)

    def test_shape_parametric(self, L, n_ph_max):
        psi = hubbard_holstein_reference_state(dims=L, n_ph_max=n_ph_max)
        n_ferm = 2 * L
        qpb = boson_qubits_per_site(n_ph_max)
        n_total = n_ferm + L * qpb
        assert psi.shape == (1 << n_total,)


class TestHHReferenceStateNorm:
    """HH reference state must have ||psi|| = 1 exactly."""

    def test_norm_L2_nph1(self):
        psi = hubbard_holstein_reference_state(dims=2, n_ph_max=1)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-14

    def test_norm_parametric(self, L, n_ph_max):
        psi = hubbard_holstein_reference_state(dims=L, n_ph_max=n_ph_max)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-14


# ────────────────────────────────────────────────────────────────────────────
# 2. Deterministic construction
# ────────────────────────────────────────────────────────────────────────────

class TestHHReferenceStateDeterminism:
    """Two calls with identical args must produce bit-for-bit identical output."""

    def test_deterministic(self, L, n_ph_max):
        psi_a = hubbard_holstein_reference_state(dims=L, n_ph_max=n_ph_max)
        psi_b = hubbard_holstein_reference_state(dims=L, n_ph_max=n_ph_max)
        np.testing.assert_array_equal(psi_a, psi_b)


# ────────────────────────────────────────────────────────────────────────────
# 3. HH layerwise ansatz: parameter count scales with reps
# ────────────────────────────────────────────────────────────────────────────

class TestHHLayerwiseAnsatzParamCount:
    """num_parameters must scale linearly with reps."""

    def test_param_count_reps_1_vs_2(self):
        kw = dict(dims=2, J=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=1)
        a1 = HubbardHolsteinLayerwiseAnsatz(**kw, reps=1)
        a2 = HubbardHolsteinLayerwiseAnsatz(**kw, reps=2)
        assert a2.num_parameters == 2 * a1.num_parameters

    def test_param_count_reps_1_vs_3(self):
        kw = dict(dims=2, J=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=1)
        a1 = HubbardHolsteinLayerwiseAnsatz(**kw, reps=1)
        a3 = HubbardHolsteinLayerwiseAnsatz(**kw, reps=3)
        assert a3.num_parameters == 3 * a1.num_parameters

    def test_param_count_positive(self, L, n_ph_max):
        a = HubbardHolsteinLayerwiseAnsatz(
            dims=L, J=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=n_ph_max, reps=1,
        )
        assert a.num_parameters > 0

    def test_nq_matches_hamiltonian(self):
        """Ansatz nq must match the Hamiltonian qubit count."""
        L, n_ph_max = 2, 1
        a = HubbardHolsteinLayerwiseAnsatz(
            dims=L, J=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=n_ph_max,
        )
        H = build_hubbard_holstein_hamiltonian(
            dims=L, J=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=n_ph_max,
        )
        terms = H.return_polynomial()
        nq_H = int(terms[0].nqubit())
        assert a.nq == nq_H


# ────────────────────────────────────────────────────────────────────────────
# 4. prepare_state output norm == 1
# ────────────────────────────────────────────────────────────────────────────

class TestHHLayerwiseAnsatzNorm:
    """Output state from prepare_state must have unit norm."""

    def test_zero_theta_norm(self):
        a = HubbardHolsteinLayerwiseAnsatz(
            dims=2, J=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=1, reps=1,
        )
        psi_ref = hubbard_holstein_reference_state(dims=2, n_ph_max=1)
        theta = np.zeros(a.num_parameters)
        psi_out = a.prepare_state(theta, psi_ref)
        assert abs(np.linalg.norm(psi_out) - 1.0) < 1e-12

    def test_random_theta_norm(self):
        a = HubbardHolsteinLayerwiseAnsatz(
            dims=2, J=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=1, reps=2,
        )
        psi_ref = hubbard_holstein_reference_state(dims=2, n_ph_max=1)
        rng = np.random.default_rng(42)
        theta = rng.normal(size=a.num_parameters) * 0.3
        psi_out = a.prepare_state(theta, psi_ref)
        assert abs(np.linalg.norm(psi_out) - 1.0) < 1e-12


# ────────────────────────────────────────────────────────────────────────────
# 5. Same-theta determinism
# ────────────────────────────────────────────────────────────────────────────

class TestHHLayerwiseAnsatzDeterminism:
    """Same theta + same ref -> identical output."""

    def test_same_theta_same_output(self):
        a = HubbardHolsteinLayerwiseAnsatz(
            dims=2, J=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=1, reps=1,
        )
        psi_ref = hubbard_holstein_reference_state(dims=2, n_ph_max=1)
        rng = np.random.default_rng(99)
        theta = rng.normal(size=a.num_parameters) * 0.5
        psi_a = a.prepare_state(theta, psi_ref)
        psi_b = a.prepare_state(theta, psi_ref)
        np.testing.assert_allclose(psi_a, psi_b, atol=1e-14)


# ────────────────────────────────────────────────────────────────────────────
# 6. Hubbard-only path backward compat
# ────────────────────────────────────────────────────────────────────────────

class TestHubbardOnlyBackwardCompat:
    """Existing Hubbard-only ansatz classes must still work identically."""

    def test_hubbard_termwise_L2(self):
        a = HubbardTermwiseAnsatz(dims=2, t=1.0, U=4.0, reps=1)
        assert a.num_parameters > 0
        nq = a.nq
        psi_ref = basis_state(nq, "0" * nq)
        theta = np.zeros(a.num_parameters)
        psi_out = a.prepare_state(theta, psi_ref)
        assert abs(np.linalg.norm(psi_out) - 1.0) < 1e-14

    def test_hubbard_layerwise_L2(self):
        a = HubbardLayerwiseAnsatz(dims=2, t=1.0, U=4.0, reps=1)
        assert a.num_parameters > 0
        nq = a.nq
        psi_ref = basis_state(nq, "0" * nq)
        theta = np.zeros(a.num_parameters)
        psi_out = a.prepare_state(theta, psi_ref)
        assert abs(np.linalg.norm(psi_out) - 1.0) < 1e-14

    def test_hubbard_termwise_param_unchanged(self):
        """L=2, reps=1, t=1, U=4, no potential: param count must not change."""
        a = HubbardTermwiseAnsatz(dims=2, t=1.0, U=4.0, reps=1, pbc=True)
        # 1D chain L=2, PBC: 1 edge * 2 spins = 2 hop + 2 onsite = 4 terms per rep
        # But with PBC on L=2 chain: edges (0,1) only (undirected, no self-loop).
        # So: 1 edge * 2 spins (hop) + 2 sites (onsite) = 4
        assert a.num_parameters == 4


# ────────────────────────────────────────────────────────────────────────────
# 7. Reference state: phonon vacuum factoring
# ────────────────────────────────────────────────────────────────────────────

class TestHHReferencePhononVacuum:
    """HH ref state must have phonon qubits all in |0> (vacuum)."""

    def test_phonon_vacuum_L2_nph1(self):
        psi = hubbard_holstein_reference_state(dims=2, n_ph_max=1, indexing="blocked")
        # The only nonzero amplitude is at a basis index where all phonon bits are 0.
        idx_nonzero = np.flatnonzero(psi)
        assert len(idx_nonzero) == 1, "Reference state must be a single comp-basis state"
        basis_idx = idx_nonzero[0]
        # phonon bits are the upper bits: n_ferm = 4, n_bos = 2, n_total = 6
        n_ferm = 4
        qpb = boson_qubits_per_site(1)
        n_bos = 2 * qpb
        # bits n_ferm..n_total-1 must be 0
        for b in range(n_ferm, n_ferm + n_bos):
            assert (basis_idx >> b) & 1 == 0, f"phonon qubit {b} not in vacuum"
