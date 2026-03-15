from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.time_propagation.projected_real_time import (
    ProjectedRealTimeConfig,
    build_tangent_vectors,
    run_exact_driven_reference,
    run_projected_real_time_trajectory,
    solve_mclachlan_step,
    state_fidelity,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _basis(dim: int, idx: int) -> np.ndarray:
    out = np.zeros(dim, dtype=complex)
    out[int(idx)] = 1.0
    return out


def _term(label: str, *, nq: int = 1, coeff: complex = 1.0) -> AnsatzTerm:
    return AnsatzTerm(
        label=f"term_{label}",
        polynomial=PauliPolynomial("JW", [PauliTerm(int(nq), ps=str(label), pc=complex(coeff))]),
    )


def test_build_tangent_vectors_matches_single_pauli_derivative() -> None:
    psi_ref = _basis(2, 0)
    psi, tangents = build_tangent_vectors(
        psi_ref,
        [_term("x")],
        np.array([0.0], dtype=float),
        tangent_eps=1e-7,
    )

    assert np.allclose(psi, psi_ref)
    assert len(tangents) == 1
    assert np.allclose(np.asarray(tangents[0]), np.array([0.0, -1.0j], dtype=complex), atol=1e-5)


def test_solve_mclachlan_step_solves_regularized_system() -> None:
    tangent = np.array([0.0, 1.0], dtype=complex)
    hpsi = np.array([0.0, 1.0j], dtype=complex)
    theta_dot, diag = solve_mclachlan_step([tangent], hpsi, lambda_reg=1e-8, svd_rcond=1e-12)

    assert theta_dot.shape == (1,)
    assert abs(float(theta_dot[0]) - 0.99999999) < 1e-6
    assert diag["regularization_used"] is True
    assert diag["solve_mode"] == "solve"


def test_run_projected_real_time_trajectory_zero_hamiltonian_keeps_state() -> None:
    psi_ref = _basis(2, 0)
    result = run_projected_real_time_trajectory(
        psi_ref,
        [_term("x")],
        np.zeros((2, 2), dtype=complex),
        config=ProjectedRealTimeConfig(t_final=0.5, num_times=5, ode_substeps=2),
    )

    assert result.theta_history.shape == (5, 1)
    assert np.allclose(result.theta_history, 0.0)
    for state in result.states:
        assert np.allclose(state, psi_ref)
    assert all(abs(float(row["state_norm"]) - 1.0) < 1e-10 for row in result.trajectory_rows)


def test_run_exact_driven_reference_static_matches_analytic_state() -> None:
    psi_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    hmat = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    result = run_exact_driven_reference(
        psi_plus,
        hmat,
        t_final=0.5,
        num_times=3,
        reference_steps=8,
        drive_coeff_provider_exyz=None,
    )

    expected = np.array([np.exp(-0.5j), np.exp(0.5j)], dtype=complex) / np.sqrt(2.0)
    assert result.times.shape == (3,)
    assert state_fidelity(result.states[-1], expected) > 1.0 - 1e-12
    assert all(abs(float(row["state_norm"]) - 1.0) < 1e-10 for row in result.trajectory_rows)
