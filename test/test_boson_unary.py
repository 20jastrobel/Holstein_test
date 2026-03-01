#!/usr/bin/env python3
"""Unit tests for unary (one-hot) boson encoding.

Tests cover:
  1. qpb and qubit-index mapping
  2. displacement (x) operator has only XX/YY terms with real coefficients
  3. number operator has only I and Z terms
  4. Hermiticity of H_ph and x
  5. full HH Hamiltonian builds without error
  6. binary vs unary spectrum match (projected onto one-hot subspace)
  7. unary reference state phonon vacuum bits
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── imports ─────────────────────────────────────────────────────────────────
from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    phonon_qubit_indices_for_site,
    boson_number_operator,
    boson_displacement_operator,
    boson_operator,
    boson_unary_number_operator,
    boson_unary_displacement_operator,
    boson_unary_bdag_operator,
    boson_unary_b_operator,
    build_holstein_phonon_energy,
    build_holstein_coupling,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.hartree_fock_reference_state import (
    hubbard_holstein_reference_state,
)
from src.quantum.vqe_latex_python_pairs import (
    hamiltonian_matrix,
    pauli_matrix,
)

# ── helpers ─────────────────────────────────────────────────────────────────

def _term_strings(poly) -> list[tuple[str, complex]]:
    """Return [(pauli_string, coeff), …] from a PauliPolynomial."""
    return [(t.pw2strng(), complex(t.p_coeff)) for t in poly.return_polynomial()]


def _pauli_letters_in_poly(poly) -> set[str]:
    """Return the set of Pauli letters appearing in non-identity positions."""
    letters: set[str] = set()
    for ps, _ in _term_strings(poly):
        for ch in ps:
            if ch not in ("e", "I"):
                letters.add(ch.upper())
    return letters


def _is_hermitian(mat: np.ndarray, tol: float = 1e-12) -> bool:
    return np.allclose(mat, mat.conj().T, atol=tol)


def _one_hot_projector(n_sites: int, n_ph_max: int, n_ferm: int) -> np.ndarray:
    """Build projector onto the one-hot (physical) boson subspace ⊗ full fermion.

    For each site, exactly one of the (N_b+1) unary qubits must be |1⟩.
    Fermion qubits are unconstrained.

    Returns a (dim_total, dim_phys) matrix where dim_phys = count of valid
    basis states, so P @ P.T is the projector.
    """
    qpb = n_ph_max + 1
    n_bos = n_sites * qpb
    n_total = n_ferm + n_bos
    dim_total = 1 << n_total

    valid_indices: list[int] = []
    for idx in range(dim_total):
        # Check each site's boson block is one-hot.
        ok = True
        for site in range(n_sites):
            base = n_ferm + site * qpb
            count = 0
            for q in range(qpb):
                count += (idx >> (base + q)) & 1
            if count != 1:
                ok = False
                break
        if ok:
            valid_indices.append(idx)

    # Build projection matrix (dim_total × len(valid_indices))
    P = np.zeros((dim_total, len(valid_indices)), dtype=complex)
    for col, row in enumerate(valid_indices):
        P[row, col] = 1.0
    return P


def _binary_physical_projector(
    n_sites: int, n_ph_max: int, n_ferm: int,
) -> np.ndarray:
    """Build projector onto the physical boson subspace for binary encoding.

    For each site, the integer encoded in the qpb qubits must be ≤ n_ph_max.
    Fermion qubits are unconstrained.

    Returns a (dim_total, dim_phys) matrix.
    """
    qpb = int(math.ceil(math.log2(n_ph_max + 1))) if n_ph_max > 0 else 1
    n_bos = n_sites * qpb
    n_total = n_ferm + n_bos
    dim_total = 1 << n_total

    valid_indices: list[int] = []
    for idx in range(dim_total):
        ok = True
        for site in range(n_sites):
            base = n_ferm + site * qpb
            # Extract the binary integer for this site's boson register
            site_val = 0
            for q in range(qpb):
                site_val += ((idx >> (base + q)) & 1) << q
            if site_val > n_ph_max:
                ok = False
                break
        if ok:
            valid_indices.append(idx)

    P = np.zeros((dim_total, len(valid_indices)), dtype=complex)
    for col, row in enumerate(valid_indices):
        P[row, col] = 1.0
    return P


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

class TestUnaryQPBAndMapping(unittest.TestCase):
    """PATCH D – test 1: qpb == N_b + 1, phonon_qubit_indices correct."""

    def test_qpb_equals_nph_plus_one(self) -> None:
        for n_ph_max in (1, 2, 3, 5):
            with self.subTest(n_ph_max=n_ph_max):
                qpb = boson_qubits_per_site(n_ph_max, encoding="unary")
                self.assertEqual(qpb, n_ph_max + 1)

    def test_binary_qpb_unchanged(self) -> None:
        """Regression: binary qpb must still be ceil(log2(N_b+1))."""
        self.assertEqual(boson_qubits_per_site(1, encoding="binary"), 1)
        self.assertEqual(boson_qubits_per_site(2, encoding="binary"), 2)
        self.assertEqual(boson_qubits_per_site(3, encoding="binary"), 2)
        self.assertEqual(boson_qubits_per_site(7, encoding="binary"), 3)

    def test_phonon_qubit_indices_unary(self) -> None:
        """For L=2, n_ph_max=1: each site gets 2 unary qubits starting at 4."""
        n_sites, n_ph_max = 2, 1
        qpb = boson_qubits_per_site(n_ph_max, encoding="unary")
        self.assertEqual(qpb, 2)
        q0 = phonon_qubit_indices_for_site(
            0, n_sites=n_sites, qpb=qpb, fermion_qubits=4
        )
        q1 = phonon_qubit_indices_for_site(
            1, n_sites=n_sites, qpb=qpb, fermion_qubits=4
        )
        self.assertEqual(q0, [4, 5])
        self.assertEqual(q1, [6, 7])


class TestUnaryDisplacementXXYYOnly(unittest.TestCase):
    """PATCH D – test 2: x has only XX and YY terms with real coefficients."""

    def test_x_letters(self) -> None:
        for n_ph_max in (1, 2, 3):
            with self.subTest(n_ph_max=n_ph_max):
                qpb = n_ph_max + 1
                nq_total = 4 + 2 * qpb  # L=2, 4 fermion qubits + 2 sites
                qubits_site0 = list(range(4, 4 + qpb))

                x = boson_unary_displacement_operator(
                    "JW", nq_total, qubits_site0, n_ph_max=n_ph_max,
                )

                letters = _pauli_letters_in_poly(x)
                self.assertTrue(
                    letters <= {"X", "Y"},
                    f"Expected only X,Y but got {letters}",
                )

    def test_x_coefficients_are_real(self) -> None:
        n_ph_max = 2
        qpb = n_ph_max + 1
        nq_total = 4 + 2 * qpb
        qubits = list(range(4, 4 + qpb))

        x = boson_unary_displacement_operator(
            "JW", nq_total, qubits, n_ph_max=n_ph_max,
        )
        for ps, coeff in _term_strings(x):
            self.assertAlmostEqual(
                coeff.imag, 0.0, places=14,
                msg=f"Imaginary part non-zero for term {ps}: {coeff}",
            )


class TestUnaryNumberIZOnly(unittest.TestCase):
    """PATCH D – test 3: number operator has only I and Z terms."""

    def test_n_letters(self) -> None:
        for n_ph_max in (1, 2, 3):
            with self.subTest(n_ph_max=n_ph_max):
                qpb = n_ph_max + 1
                nq_total = 4 + 2 * qpb
                qubits = list(range(4, 4 + qpb))

                n_op = boson_unary_number_operator(
                    "JW", nq_total, qubits, n_ph_max=n_ph_max,
                )

                # Every term should contain only 'e' (identity) and 'z'
                for ps, _ in _term_strings(n_op):
                    for ch in ps:
                        self.assertIn(
                            ch.lower(), ("e", "z"),
                            f"Unexpected letter '{ch}' in number operator term {ps}",
                        )

    def test_n_coefficients_are_real(self) -> None:
        n_ph_max = 2
        qpb = n_ph_max + 1
        nq_total = 4 + 2 * qpb
        qubits = list(range(4, 4 + qpb))

        n_op = boson_unary_number_operator(
            "JW", nq_total, qubits, n_ph_max=n_ph_max,
        )
        for ps, coeff in _term_strings(n_op):
            self.assertAlmostEqual(
                coeff.imag, 0.0, places=14,
                msg=f"Imaginary part non-zero for term {ps}: {coeff}",
            )


class TestUnaryHermiticity(unittest.TestCase):
    """PATCH D – test 4: H_ph and x are Hermitian."""

    def test_phonon_energy_hermitian(self) -> None:
        """H_ph = ω₀ Σ (n + ½) must be Hermitian."""
        L, n_ph_max, omega0 = 2, 2, 1.5
        H_ph = build_holstein_phonon_energy(
            dims=L, omega0=omega0, n_ph_max=n_ph_max,
            boson_encoding="unary", repr_mode="JW",
        )
        mat = hamiltonian_matrix(H_ph)
        self.assertTrue(
            _is_hermitian(mat),
            "H_ph (unary) is not Hermitian",
        )

    def test_displacement_hermitian(self) -> None:
        """x = b + b† must be Hermitian."""
        n_ph_max = 2
        qpb = n_ph_max + 1
        nq_total = 4 + 2 * qpb
        qubits = list(range(4, 4 + qpb))

        x = boson_unary_displacement_operator(
            "JW", nq_total, qubits, n_ph_max=n_ph_max,
        )
        mat = hamiltonian_matrix(x)
        self.assertTrue(
            _is_hermitian(mat),
            "x (unary displacement) is not Hermitian",
        )

    def test_number_operator_hermitian(self) -> None:
        """n must be Hermitian."""
        n_ph_max = 2
        qpb = n_ph_max + 1
        nq_total = 4 + 2 * qpb
        qubits = list(range(4, 4 + qpb))

        n_op = boson_unary_number_operator(
            "JW", nq_total, qubits, n_ph_max=n_ph_max,
        )
        mat = hamiltonian_matrix(n_op)
        self.assertTrue(
            _is_hermitian(mat),
            "n (unary number) is not Hermitian",
        )


class TestUnaryFullHHBuilds(unittest.TestCase):
    """PATCH D – test 5: full HH Hamiltonian builds without error."""

    def test_build_hh_unary_no_error(self) -> None:
        H = build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0, U=4.0,
            omega0=1.0, g=0.5,
            n_ph_max=1,
            boson_encoding="unary",
            indexing="blocked",
            pbc=True,
        )
        nterms = H.count_number_terms()
        self.assertGreater(nterms, 0, "HH (unary) Hamiltonian has no terms")

    def test_build_hh_unary_hermitian(self) -> None:
        H = build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0, U=4.0,
            omega0=1.0, g=0.5,
            n_ph_max=1,
            boson_encoding="unary",
            indexing="blocked",
            pbc=True,
        )
        mat = hamiltonian_matrix(H)
        self.assertTrue(_is_hermitian(mat), "HH (unary) Hamiltonian is not Hermitian")


class TestBinaryVsUnarySpectrum(unittest.TestCase):
    """PATCH D – test 6: binary and unary eigenvalues match on one-hot subspace.

    Strategy: For the same physical parameters, build both binary and unary
    HH Hamiltonians. Project the unary Hamiltonian onto the physical (one-hot)
    subspace. The projected eigenvalues must match the binary eigenvalues.
    """

    def _get_spectra(self, L: int, n_ph_max: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (binary projected eigenvalues sorted, unary projected eigenvalues sorted).

        Both are projected onto their respective physical boson subspaces to
        handle n_ph_max values where binary qpb encodes extra unphysical states.
        """
        t, U, omega0, g = 1.0, 4.0, 1.0, 0.5

        H_bin = build_hubbard_holstein_hamiltonian(
            dims=L, J=t, U=U, omega0=omega0, g=g,
            n_ph_max=n_ph_max, boson_encoding="binary",
            indexing="blocked", pbc=True,
        )
        H_un = build_hubbard_holstein_hamiltonian(
            dims=L, J=t, U=U, omega0=omega0, g=g,
            n_ph_max=n_ph_max, boson_encoding="unary",
            indexing="blocked", pbc=True,
        )

        mat_bin = hamiltonian_matrix(H_bin)
        mat_un = hamiltonian_matrix(H_un)

        n_ferm = 2 * L

        # Project binary onto physical subspace (state ≤ n_ph_max per site)
        P_bin = _binary_physical_projector(L, n_ph_max, n_ferm)
        mat_bin_proj = P_bin.conj().T @ mat_bin @ P_bin
        evals_bin = np.sort(np.linalg.eigvalsh(mat_bin_proj).real)

        # Project unary onto one-hot subspace
        P_un = _one_hot_projector(L, n_ph_max, n_ferm)
        mat_un_proj = P_un.conj().T @ mat_un @ P_un
        evals_un = np.sort(np.linalg.eigvalsh(mat_un_proj).real)

        return evals_bin, evals_un

    def test_L2_nph1_spectrum_match(self) -> None:
        evals_bin, evals_proj = self._get_spectra(L=2, n_ph_max=1)
        np.testing.assert_allclose(
            evals_bin, evals_proj, atol=1e-10,
            err_msg="L=2 n_ph_max=1 binary vs unary projected spectra differ",
        )

    def test_L2_nph2_spectrum_match(self) -> None:
        evals_bin, evals_proj = self._get_spectra(L=2, n_ph_max=2)
        np.testing.assert_allclose(
            evals_bin, evals_proj, atol=1e-10,
            err_msg="L=2 n_ph_max=2 binary vs unary projected spectra differ",
        )


class TestUnaryReferenceStateVacuum(unittest.TestCase):
    """PATCH D – test 7: phonon vacuum bits correct for unary encoding."""

    def test_vacuum_state_overlap(self) -> None:
        """Reference state must have phonon vacuum = one-hot |n=0⟩ per site."""
        L, n_ph_max = 2, 1
        psi = hubbard_holstein_reference_state(
            dims=L,
            n_ph_max=n_ph_max,
            boson_encoding="unary",
            indexing="blocked",
        )
        # Dimension check: 2^(4 + 2*2) = 2^8 = 256
        n_ferm = 2 * L
        qpb = n_ph_max + 1  # 2 for unary
        n_total = n_ferm + L * qpb  # 4 + 4 = 8
        self.assertEqual(psi.shape[0], 1 << n_total)

        # The state must be a single basis vector
        nonzero = np.where(np.abs(psi) > 1e-14)[0]
        self.assertEqual(len(nonzero), 1, "Reference state is not a single basis vector")
        idx = nonzero[0]

        # Check phonon bits: each site should have qubit[0]=1, rest=0
        for site in range(L):
            base = n_ferm + site * qpb
            for q_offset in range(qpb):
                bit = (idx >> (base + q_offset)) & 1
                if q_offset == 0:
                    self.assertEqual(
                        bit, 1,
                        f"Site {site} qubit[0] should be 1 (vacuum), got 0",
                    )
                else:
                    self.assertEqual(
                        bit, 0,
                        f"Site {site} qubit[{q_offset}] should be 0, got 1",
                    )

    def test_binary_vacuum_unchanged(self) -> None:
        """Regression: binary reference state phonon sector = all zeros."""
        L, n_ph_max = 2, 1
        psi = hubbard_holstein_reference_state(
            dims=L,
            n_ph_max=n_ph_max,
            boson_encoding="binary",
            indexing="blocked",
        )
        nonzero = np.where(np.abs(psi) > 1e-14)[0]
        self.assertEqual(len(nonzero), 1)
        idx = nonzero[0]

        n_ferm = 2 * L
        qpb = boson_qubits_per_site(n_ph_max, encoding="binary")
        for site in range(L):
            base = n_ferm + site * qpb
            for q_offset in range(qpb):
                bit = (idx >> (base + q_offset)) & 1
                self.assertEqual(
                    bit, 0,
                    f"Binary vacuum: site {site} qubit[{q_offset}] should be 0",
                )

    def test_unary_state_normalised(self) -> None:
        psi = hubbard_holstein_reference_state(
            dims=2, n_ph_max=1, boson_encoding="unary", indexing="blocked",
        )
        self.assertAlmostEqual(np.linalg.norm(psi), 1.0, places=14)


class TestUnaryFactoryDispatch(unittest.TestCase):
    """Ensure boson_operator dispatches to unary helpers via encoding kwarg."""

    def test_number_via_factory(self) -> None:
        """boson_operator(which='n', encoding='unary') == boson_unary_number_operator."""
        n_ph_max = 2
        qpb = n_ph_max + 1
        nq = 4 + 2 * qpb
        qubits = list(range(4, 4 + qpb))

        via_factory = hamiltonian_matrix(
            boson_operator("JW", nq, qubits, which="n", n_ph_max=n_ph_max, encoding="unary")
        )
        via_direct = hamiltonian_matrix(
            boson_unary_number_operator("JW", nq, qubits, n_ph_max=n_ph_max)
        )
        np.testing.assert_allclose(via_factory, via_direct, atol=1e-14)

    def test_x_via_factory(self) -> None:
        n_ph_max = 2
        qpb = n_ph_max + 1
        nq = 4 + 2 * qpb
        qubits = list(range(4, 4 + qpb))

        via_factory = hamiltonian_matrix(
            boson_operator("JW", nq, qubits, which="x", n_ph_max=n_ph_max, encoding="unary")
        )
        via_direct = hamiltonian_matrix(
            boson_unary_displacement_operator("JW", nq, qubits, n_ph_max=n_ph_max)
        )
        np.testing.assert_allclose(via_factory, via_direct, atol=1e-14)

    def test_convenience_wrappers_dispatch_unary(self) -> None:
        """boson_number_operator / boson_displacement_operator with encoding='unary'."""
        n_ph_max = 2
        qpb = n_ph_max + 1
        nq = 4 + 2 * qpb
        qubits = list(range(4, 4 + qpb))

        n_factory = hamiltonian_matrix(
            boson_number_operator("JW", nq, qubits, n_ph_max=n_ph_max, encoding="unary")
        )
        n_direct = hamiltonian_matrix(
            boson_unary_number_operator("JW", nq, qubits, n_ph_max=n_ph_max)
        )
        np.testing.assert_allclose(n_factory, n_direct, atol=1e-14)

        x_factory = hamiltonian_matrix(
            boson_displacement_operator("JW", nq, qubits, n_ph_max=n_ph_max, encoding="unary")
        )
        x_direct = hamiltonian_matrix(
            boson_unary_displacement_operator("JW", nq, qubits, n_ph_max=n_ph_max)
        )
        np.testing.assert_allclose(x_factory, x_direct, atol=1e-14)


class TestUnaryBBdagAlgebra(unittest.TestCase):
    """Verify b + b† == x, and b†b eigenvalues on one-hot subspace are 0..N_b."""

    def test_b_plus_bdag_equals_x(self) -> None:
        n_ph_max = 2
        qpb = n_ph_max + 1
        nq = 4 + 2 * qpb
        qubits = list(range(4, 4 + qpb))

        mat_b = hamiltonian_matrix(
            boson_unary_b_operator("JW", nq, qubits, n_ph_max=n_ph_max)
        )
        mat_bdag = hamiltonian_matrix(
            boson_unary_bdag_operator("JW", nq, qubits, n_ph_max=n_ph_max)
        )
        mat_x = hamiltonian_matrix(
            boson_unary_displacement_operator("JW", nq, qubits, n_ph_max=n_ph_max)
        )
        np.testing.assert_allclose(mat_b + mat_bdag, mat_x, atol=1e-14)

    def test_bdag_b_eigenvalues_one_hot(self) -> None:
        """b†b projected onto one-hot subspace has eigenvalues {0, 1, …, N_b}."""
        n_ph_max = 2
        qpb = n_ph_max + 1
        # Minimal: 0 fermion qubits, 1 site, just the boson register
        nq = qpb
        qubits = list(range(qpb))

        mat_b = hamiltonian_matrix(
            boson_unary_b_operator("JW", nq, qubits, n_ph_max=n_ph_max)
        )
        mat_bdag = hamiltonian_matrix(
            boson_unary_bdag_operator("JW", nq, qubits, n_ph_max=n_ph_max)
        )
        mat_bdag_b = mat_bdag @ mat_b

        # Project onto one-hot subspace (1 site, 0 fermion qubits)
        P = _one_hot_projector(n_sites=1, n_ph_max=n_ph_max, n_ferm=0)
        mat_proj = P.conj().T @ mat_bdag_b @ P
        evals = np.sort(np.linalg.eigvalsh(mat_proj).real)

        expected = np.arange(n_ph_max + 1, dtype=float)
        np.testing.assert_allclose(
            evals, expected, atol=1e-12,
            err_msg=f"b†b eigenvalues on one-hot subspace: {evals} != {expected}",
        )

    def test_bdag_is_adjoint_of_b(self) -> None:
        """b† matrix is the conjugate transpose of b."""
        n_ph_max = 2
        qpb = n_ph_max + 1
        nq = 4 + 2 * qpb
        qubits = list(range(4, 4 + qpb))

        mat_b = hamiltonian_matrix(
            boson_unary_b_operator("JW", nq, qubits, n_ph_max=n_ph_max)
        )
        mat_bdag = hamiltonian_matrix(
            boson_unary_bdag_operator("JW", nq, qubits, n_ph_max=n_ph_max)
        )
        np.testing.assert_allclose(mat_bdag, mat_b.conj().T, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
