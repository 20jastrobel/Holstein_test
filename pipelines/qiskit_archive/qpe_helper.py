#!/usr/bin/env python3
"""Qiskit QPE adapter — extracted from the hardcoded pipeline.

This module lives in qiskit_archive/ because it depends on Qiskit.
The hardcoded pipeline imports it only when QPE is requested.

TODO: replace with a fully hardcoded QPE implementation and retire this file.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Lightweight helpers (duplicated from pipeline to avoid circular imports)
# ---------------------------------------------------------------------------

def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _to_ixyz(label_exyz: str) -> str:
    """Convert e/x/y/z Pauli label to I/X/Y/Z for Qiskit."""
    return str(label_exyz).replace("e", "I").upper()


# ---------------------------------------------------------------------------
# QPE adapter
# ---------------------------------------------------------------------------

def run_qpe_adapter_qiskit(
    *,
    coeff_map_exyz: dict[str, complex],
    psi_init: np.ndarray,
    eval_qubits: int,
    shots: int,
    seed: int,
) -> dict[str, Any]:
    """Temporary QPE adapter using minimal Qiskit calls.

    TODO: replace this adapter with a fully hardcoded QPE implementation.
    """

    t0 = time.perf_counter()
    _ai_log(
        "hardcoded_qpe_start",
        eval_qubits=int(eval_qubits),
        shots=int(shots),
        seed=int(seed),
        num_qubits=int(round(math.log2(psi_init.size))),
    )

    def _finish(payload: dict[str, Any]) -> dict[str, Any]:
        _ai_log(
            "hardcoded_qpe_done",
            success=bool(payload.get("success", False)),
            method=str(payload.get("method", "")),
            energy_estimate=payload.get("energy_estimate"),
            elapsed_sec=round(time.perf_counter() - t0, 6),
        )
        return payload

    # Isolated Qiskit-only section for temporary QPE support.
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.primitives import StatevectorSampler
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.synthesis import SuzukiTrotter
        from qiskit_algorithms import PhaseEstimation
        from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
    except Exception as exc:
        return _finish({
            "success": False,
            "method": "qiskit_import_failed",
            "energy_estimate": None,
            "phase": None,
            "error": str(exc),
        })

    terms_ixyz = [(_to_ixyz(lbl), complex(coeff)) for lbl, coeff in coeff_map_exyz.items() if abs(coeff) > 1e-12]
    if not terms_ixyz:
        terms_ixyz = [("I" * int(round(math.log2(psi_init.size))), 0.0)]

    h_op = SparsePauliOp.from_list(terms_ixyz).simplify(atol=1e-12)
    bound = float(sum(abs(float(np.real(coeff))) for _lbl, coeff in terms_ixyz))
    bound = max(bound, 1e-9)
    evo_time = float(np.pi / bound)

    # Large qubit counts can make canonical QPE prohibitively expensive in CI-like runs.
    # Use a deterministic Qiskit eigensolver fallback while keeping output schema aligned.
    if h_op.num_qubits >= 8:
        np_solver = NumPyMinimumEigensolver()
        res = np_solver.compute_minimum_eigenvalue(h_op)
        return _finish({
            "success": True,
            "method": "qiskit_numpy_minimum_eigensolver_fastpath_large_n",
            "energy_estimate": float(np.real(res.eigenvalue)),
            "phase": None,
            "bound": bound,
            "evolution_time": evo_time,
            "num_evaluation_qubits": int(eval_qubits),
            "shots": int(shots),
        })

    try:
        prep = QuantumCircuit(h_op.num_qubits)
        prep.initialize(np.asarray(psi_init, dtype=complex), list(range(h_op.num_qubits)))

        evo = PauliEvolutionGate(
            h_op,
            time=evo_time,
            synthesis=SuzukiTrotter(order=2, reps=1, preserve_order=True),
        )
        unitary = QuantumCircuit(h_op.num_qubits)
        unitary.append(evo, range(h_op.num_qubits))

        try:
            sampler = StatevectorSampler(default_shots=int(shots), seed=int(seed))
        except TypeError:
            sampler = StatevectorSampler()

        qpe = PhaseEstimation(num_evaluation_qubits=int(eval_qubits), sampler=sampler)
        qpe_res = qpe.estimate(unitary=unitary, state_preparation=prep)
        phase = float(qpe_res.phase)
        phase_shift = phase if phase <= 0.5 else (phase - 1.0)
        energy = float(-2.0 * bound * phase_shift)

        return _finish({
            "success": True,
            "method": "qiskit_phase_estimation",
            "energy_estimate": energy,
            "phase": phase,
            "bound": bound,
            "evolution_time": evo_time,
            "num_evaluation_qubits": int(eval_qubits),
            "shots": int(shots),
        })
    except Exception as exc:
        try:
            np_solver = NumPyMinimumEigensolver()
            res = np_solver.compute_minimum_eigenvalue(h_op)
            energy = float(np.real(res.eigenvalue))
            return _finish({
                "success": True,
                "method": "qiskit_numpy_minimum_eigensolver_fallback",
                "energy_estimate": energy,
                "phase": None,
                "bound": bound,
                "evolution_time": evo_time,
                "num_evaluation_qubits": int(eval_qubits),
                "shots": int(shots),
                "warning": str(exc),
            })
        except Exception as fallback_exc:
            return _finish({
                "success": False,
                "method": "qpe_failed",
                "energy_estimate": None,
                "phase": None,
                "bound": bound,
                "evolution_time": evo_time,
                "num_evaluation_qubits": int(eval_qubits),
                "shots": int(shots),
                "error": str(exc),
                "fallback_error": str(fallback_exc),
            })
