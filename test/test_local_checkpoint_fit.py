from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.time_propagation.local_checkpoint_fit import (
    CheckpointFitConfig,
    LocalPauliAnsatzSpec,
    build_local_pauli_ansatz_terms,
    fit_checkpoint_target_state,
    fit_checkpoint_trajectory,
)


def test_build_local_pauli_ansatz_terms_counts_chain_terms() -> None:
    terms = build_local_pauli_ansatz_terms(
        LocalPauliAnsatzSpec(
            num_qubits=4,
            reps=2,
            single_axes=("y",),
            entangler_axes=("zz",),
        )
    )
    assert len(terms) == 2 * (4 + 3)
    assert str(terms[0].label) == "local_y(q=0)_rep1"
    assert str(terms[-1].label) == "local_zz(q=2,3)_rep2"


def test_fit_checkpoint_target_state_recovers_one_qubit_rotation() -> None:
    terms = build_local_pauli_ansatz_terms(
        LocalPauliAnsatzSpec(
            num_qubits=1,
            reps=1,
            single_axes=("y",),
            entangler_axes=(),
        )
    )
    executor = CompiledAnsatzExecutor(list(terms))
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    target_theta = np.asarray([0.23], dtype=float)
    psi_target = executor.prepare_state(target_theta, psi_ref)

    result = fit_checkpoint_target_state(
        psi_ref,
        psi_target,
        terms,
        config=CheckpointFitConfig(maxiter=40),
    )

    assert float(result.fidelity) >= 1.0 - 1e-10
    assert float(result.objective) <= 1e-10
    assert np.allclose(result.theta, target_theta, atol=1e-6)


def test_fit_checkpoint_trajectory_warm_starts_across_times() -> None:
    terms = build_local_pauli_ansatz_terms(
        LocalPauliAnsatzSpec(
            num_qubits=1,
            reps=1,
            single_axes=("y",),
            entangler_axes=(),
        )
    )
    executor = CompiledAnsatzExecutor(list(terms))
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    theta_targets = [0.0, 0.1, 0.2]
    target_states = [executor.prepare_state(np.asarray([theta], dtype=float), psi_ref) for theta in theta_targets]

    result = fit_checkpoint_trajectory(
        psi_ref,
        target_states,
        [0.0, 0.1, 0.2],
        terms,
        config=CheckpointFitConfig(maxiter=30),
    )

    assert result.theta_history.shape == (3, 1)
    assert len(result.states) == 3
    assert all(float(row["fidelity"]) >= 1.0 - 1e-10 for row in result.solver_rows)
