from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Ensure repo root is on path (same pattern as other integration tests).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hartree_fock_reference_state import (
    hartree_fock_statevector,
    hubbard_holstein_reference_state,
)
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
import src.quantum.vqe_latex_python_pairs as vqe_mod
from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinLayerwiseAnsatz,
    HubbardLayerwiseAnsatz,
    vqe_minimize,
)


def test_vqe_energy_backend_one_apply_matches_legacy_hubbard():
    H = build_hubbard_hamiltonian(
        dims=2,
        t=1.0,
        U=4.0,
        v=0.1,
        indexing="blocked",
        pbc=True,
    )
    ansatz = HubbardLayerwiseAnsatz(
        dims=2,
        t=1.0,
        U=4.0,
        v=0.1,
        reps=1,
        indexing="blocked",
        pbc=True,
    )
    psi_ref = hartree_fock_statevector(n_sites=2, num_particles=(1, 1), indexing="blocked")

    legacy = vqe_minimize(
        H,
        ansatz,
        psi_ref,
        restarts=1,
        seed=123,
        maxiter=120,
        energy_backend="legacy",
    )
    fast = vqe_minimize(
        H,
        ansatz,
        psi_ref,
        restarts=1,
        seed=123,
        maxiter=120,
        energy_backend="one_apply_compiled",
    )

    assert np.isfinite(legacy.energy)
    assert np.isfinite(fast.energy)
    assert abs(fast.energy - legacy.energy) < 1e-9


def test_vqe_minimize_prefers_restart_best_energy_over_terminal_optimizer_energy(monkeypatch: pytest.MonkeyPatch):
    class _DummyAnsatz:
        num_parameters = 1

        def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float)

    call_index = {"value": 0}

    def _fake_expval(psi: np.ndarray, _h: object) -> float:
        return float(np.asarray(psi, dtype=float).reshape(-1)[0])

    def _fake_minimize(objective, x0, method=None, bounds=None, options=None):
        call_index["value"] += 1
        if call_index["value"] == 1:
            objective(np.array([0.03], dtype=float))
            objective(np.array([1.95], dtype=float))
            return SimpleNamespace(
                fun=1.95,
                x=np.array([1.95], dtype=float),
                nfev=2,
                nit=2,
                success=True,
                message="restart1-terminal-worse-than-best",
            )
        objective(np.array([0.88], dtype=float))
        return SimpleNamespace(
            fun=0.88,
            x=np.array([0.88], dtype=float),
            nfev=1,
            nit=1,
            success=True,
            message="restart2-terminal",
        )

    monkeypatch.setattr(vqe_mod, "expval_pauli_polynomial", _fake_expval)
    monkeypatch.setattr(vqe_mod, "_try_import_scipy_minimize", lambda: _fake_minimize)

    result = vqe_minimize(
        H=object(),
        ansatz=_DummyAnsatz(),
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        restarts=2,
        seed=7,
        method="Powell",
        maxiter=5,
        bounds=None,
        energy_backend="legacy",
    )

    assert result.energy == pytest.approx(0.03)
    assert result.best_restart == 0
    assert np.allclose(result.theta, np.array([0.03], dtype=float))


def test_vqe_energy_backend_one_apply_matches_legacy_hh():
    H = build_hubbard_holstein_hamiltonian(
        dims=2,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        indexing="blocked",
        pbc=True,
    )
    ansatz = HubbardHolsteinLayerwiseAnsatz(
        dims=2,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        reps=1,
        indexing="blocked",
        pbc=True,
    )
    psi_ref = hubbard_holstein_reference_state(
        dims=2,
        n_ph_max=1,
        boson_encoding="binary",
        indexing="blocked",
    )

    legacy = vqe_minimize(
        H,
        ansatz,
        psi_ref,
        restarts=1,
        seed=321,
        maxiter=180,
        energy_backend="legacy",
    )
    fast = vqe_minimize(
        H,
        ansatz,
        psi_ref,
        restarts=1,
        seed=321,
        maxiter=180,
        energy_backend="one_apply_compiled",
    )

    assert np.isfinite(legacy.energy)
    assert np.isfinite(fast.energy)
    assert abs(fast.energy - legacy.energy) < 1e-9
