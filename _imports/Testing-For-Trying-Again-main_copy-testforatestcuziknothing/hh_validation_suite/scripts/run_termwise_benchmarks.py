#!/usr/bin/env python3
"""
HH term-wise ansatz benchmark harness.

This keeps experimental sweeps isolated to the validation suite folder.
"""

from __future__ import annotations

import numpy as np

from pydephasing.quantum.ed_hubbard_holstein import (
    build_hh_sector_basis,
    build_hh_sector_hamiltonian_ed,
    matrix_to_dense,
)
from pydephasing.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
from pydephasing.quantum.vqe_latex_python_pairs_test import (
    PauliPolynomial,
    apply_exp_pauli_polynomial,
    hubbard_holstein_reference_state,
    vqe_minimize,
)
from pydephasing.quantum.pauli_words import PauliTerm


def exact_sector_energy(dims: int) -> float:
    num_particles = ((dims + 1) // 2, dims // 2)
    basis = build_hh_sector_basis(
        dims=dims,
        n_ph_max=1,
        num_particles=num_particles,
        indexing="blocked",
        boson_encoding="binary",
    )
    H_ed = build_hh_sector_hamiltonian_ed(
        dims=dims,
        J=0.2,
        U=0.2,
        omega0=0.2,
        g=0.0,
        n_ph_max=1,
        num_particles=num_particles,
        indexing="blocked",
        boson_encoding="binary",
        pbc=False,
        sparse=True,
        basis=basis,
        include_zero_point=True,
    )
    return float(np.min(np.linalg.eigvalsh(matrix_to_dense(H_ed))))


def termwise_ansatz(H, *, reps: int, term_sort: str, normalize_coeff: bool):
    terms = list(H.return_polynomial())
    if term_sort == "magnitude":
        terms = sorted(terms, key=lambda term: abs(complex(term.p_coeff)), reverse=True)
    elif term_sort == "reverse":
        terms = list(reversed(terms))

    nq = int(terms[0].nqubit())
    repr_mode = H._repr_mode  # type: ignore[attr-defined]
    base_terms = [
        PauliPolynomial(
            repr_mode,
            [PauliTerm(nq, ps=term.pw2strng(), pc=1.0 if normalize_coeff else complex(term.p_coeff))],
        )
        for term in terms
    ]

    class _Ansatz:
        def __init__(self, base_terms, reps):
            self.nq = int(nq)
            self.base_terms = base_terms
            self.num_parameters = len(base_terms) * int(reps)
            self.reps = int(reps)

        def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray, **_kwargs) -> np.ndarray:
            psi = np.array(psi_ref, copy=True)
            k = 0
            for _ in range(self.reps):
                for term in self.base_terms:
                    psi = apply_exp_pauli_polynomial(psi, term, float(theta[k]))
                    k += 1
            return psi

    return _Ansatz(base_terms, reps)


def run_sweep():
    print("[HH termwise benchmark] exact sector energies")
    exact = {dims: exact_sector_energy(dims) for dims in (2, 3)}
    for dims, e_exact in exact.items():
        print(f"  L={dims}: exact={e_exact:.12f}")

    methods = ("SLSQP", "BFGS", "COBYLA")
    configs = []
    for dims in (2, 3):
        H = build_hubbard_holstein_hamiltonian(
            dims=dims,
            J=0.2,
            U=0.2,
            omega0=0.2,
            g=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            pbc=False,
            v_t=None,
            v0=None,
            t_eval=None,
        )
        psi_ref = hubbard_holstein_reference_state(
            dims=dims,
            n_ph_max=1,
            boson_encoding="binary",
            indexing="blocked",
        )

        for reps in (1, 2, 3):
            for normalize_coeff in (True, False):
                for term_sort in ("magnitude", "reverse", "input"):
                    configs.append((dims, reps, normalize_coeff, term_sort))

        print(f"[HH termwise benchmark] running L={dims}")
        for reps, normalize_coeff, term_sort in [
            (1, True, "magnitude"),
            (1, True, "reverse"),
            (1, False, "input"),
            (2, True, "magnitude"),
            (2, False, "input"),
        ]:
            ansatz = termwise_ansatz(
                H,
                reps=reps,
                term_sort=term_sort,
                normalize_coeff=normalize_coeff,
            )
            for method in methods:
                res = vqe_minimize(
                    H,
                    ansatz,
                    psi_ref,
                    restarts=4,
                    seed=11,
                    method=method,
                    maxiter=80,
                )
                gap = res.energy - exact[dims]
                print(
                    f"L={dims} reps={reps} norm={normalize_coeff} "
                    f"sort={term_sort:8s} method={method:9s} "
                    f"E={res.energy:.12f} gap={gap:.12f}"
                )


if __name__ == "__main__":
    run_sweep()
