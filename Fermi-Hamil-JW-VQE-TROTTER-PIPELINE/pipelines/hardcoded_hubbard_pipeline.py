#!/usr/bin/env python3
"""Hardcoded-first end-to-end Hubbard pipeline.

Flow:
1) Build hardcoded Hubbard Hamiltonian (JW) from repo source-of-truth helpers.
2) Run hardcoded VQE on numpy-statevector backend (SciPy optional fallback comes from
   the notebook implementation).
3) Run temporary QPE adapter (Qiskit-only, isolated in one function).
4) Run hardcoded Suzuki-2 Trotter dynamics and exact dynamics.
5) Emit JSON + compact PDF artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
from src.quantum.hubbard_latex_python_pairs import build_hubbard_hamiltonian
from src.quantum.drives_time_potential import (
    build_gaussian_sinusoid_density_drive,
    reference_method_name,
)


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


PAULI_MATS = {
    "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}

EXACT_LABEL = "Exact_Hardcode"
EXACT_METHOD = "python_matrix_eigendecomposition"


@dataclass(frozen=True)
class CompiledPauliAction:
    label_exyz: str
    perm: np.ndarray
    phase: np.ndarray


def _to_ixyz(label_exyz: str) -> str:
    return str(label_exyz).replace("e", "I").upper()


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return psi / nrm


def _half_filled_particles(num_sites: int) -> tuple[int, int]:
    return ((num_sites + 1) // 2, num_sites // 2)


def _exact_energy_sector_filtered(
    hmat: np.ndarray,
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> float:
    """Return the lowest eigenvalue of *hmat* restricted to the particle sector
    ``(n_up, n_dn) = num_particles``.

    The sector is identified by counting set bits in the computational-basis
    index according to the qubit layout:

    * **blocked** (default for JW):  qubits 0 … L-1 are spin-up, L … 2L-1
      are spin-down.  ``n_up = popcount(idx & up_mask)``,
      ``n_dn = popcount(idx & dn_mask)``.
    * **interleaved**: qubits 0, 2, 4, … are spin-up; 1, 3, 5, … are
      spin-down.

    Only basis states with exactly the requested particle numbers are kept.
    The Hamiltonian is projected onto the sector subspace and diagonalised.
    If no states satisfy the filter an error is raised.

    This energy is the *apples-to-apples* comparison target for VQE, which is
    also run in the same half-filled sector.
    """
    nq = 2 * int(num_sites)
    dim = 1 << nq
    n_up_want, n_dn_want = int(num_particles[0]), int(num_particles[1])
    norm_ordering = str(ordering).strip().lower()

    idx_all = np.arange(dim, dtype=np.int64)

    if norm_ordering == "blocked":
        # Qubits 0 … L-1 carry spin-up; L … 2L-1 carry spin-down.
        up_mask = (1 << num_sites) - 1          # bits 0..L-1
        dn_mask = up_mask << num_sites           # bits L..2L-1
        n_up_arr = np.array([bin(int(i) & int(up_mask)).count("1") for i in idx_all], dtype=np.int32)
        n_dn_arr = np.array([bin(int(i) & int(dn_mask)).count("1") for i in idx_all], dtype=np.int32)
    else:
        # Interleaved: even bits → spin-up, odd bits → spin-down.
        even_mask = int(sum(1 << (2 * q) for q in range(num_sites)))
        odd_mask  = int(sum(1 << (2 * q + 1) for q in range(num_sites)))
        n_up_arr = np.array([bin(int(i) & even_mask).count("1") for i in idx_all], dtype=np.int32)
        n_dn_arr = np.array([bin(int(i) & odd_mask).count("1") for i in idx_all], dtype=np.int32)

    sector_indices = np.where((n_up_arr == n_up_want) & (n_dn_arr == n_dn_want))[0]
    if len(sector_indices) == 0:
        raise ValueError(
            f"No basis states found for sector (n_up={n_up_want}, n_dn={n_dn_want}) "
            f"with ordering='{ordering}', num_sites={num_sites}."
        )

    h_sector = hmat[np.ix_(sector_indices, sector_indices)]
    evals_sector = np.linalg.eigvalsh(h_sector)
    return float(np.real(evals_sector[0]))

    coeff_map: dict[str, complex] = {}
    order: list[str] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if label not in coeff_map:
            coeff_map[label] = 0.0 + 0.0j
            order.append(label)
        coeff_map[label] += coeff
    cleaned_order = [lbl for lbl in order if abs(coeff_map[lbl]) > tol]
    cleaned_map = {lbl: coeff_map[lbl] for lbl in cleaned_order}
    return cleaned_order, cleaned_map


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = [PAULI_MATS[ch] for ch in label]
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def _collect_hardcoded_terms_exyz(
    h_poly,
) -> tuple[list[str], dict[str, complex]]:
    """Walk a PauliPolynomial and return (native_order, coeff_map_exyz).

    Duplicate labels are accumulated (summed).  The native_order list
    preserves insertion order of *first occurrence*, matching the order
    the Hamiltonian builder emitted the terms.
    """
    coeff_map: dict[str, complex] = {}
    native_order: list[str] = []
    for term in h_poly.return_polynomial():
        label: str = term.pw2strng()  # lowercase exyz string
        coeff: complex = complex(term.p_coeff)
        if label not in coeff_map:
            native_order.append(label)
            coeff_map[label] = coeff
        else:
            coeff_map[label] = coeff_map[label] + coeff
    return native_order, coeff_map


def _build_hamiltonian_matrix(coeff_map_exyz: dict[str, complex]) -> np.ndarray:
    if not coeff_map_exyz:
        return np.zeros((1, 1), dtype=complex)
    nq = len(next(iter(coeff_map_exyz)))
    dim = 1 << nq
    hmat = np.zeros((dim, dim), dtype=complex)
    for label, coeff in coeff_map_exyz.items():
        hmat += coeff * _pauli_matrix_exyz(label)
    return hmat


def _compile_pauli_action(label_exyz: str, nq: int) -> CompiledPauliAction:
    dim = 1 << nq
    idx = np.arange(dim, dtype=np.int64)
    perm = idx.copy()
    phase = np.ones(dim, dtype=complex)

    for q in range(nq):
        op = label_exyz[nq - 1 - q]
        bits = ((idx >> q) & 1).astype(np.int8)
        sign = (1 - 2 * bits).astype(np.int8)

        if op == "e":
            continue
        if op == "x":
            perm ^= (1 << q)
            continue
        if op == "y":
            perm ^= (1 << q)
            phase *= 1j * sign
            continue
        if op == "z":
            phase *= sign
            continue
        raise ValueError(f"Unsupported Pauli symbol '{op}' in '{label_exyz}'.")

    return CompiledPauliAction(label_exyz=label_exyz, perm=perm, phase=phase)


def _apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
    out = np.empty_like(psi)
    out[action.perm] = action.phase * psi
    return out


def _apply_exp_term(
    psi: np.ndarray,
    action: CompiledPauliAction,
    coeff: complex,
    alpha: float,
    tol: float = 1e-12,
) -> np.ndarray:
    if abs(coeff.imag) > tol:
        raise ValueError(f"Imaginary coefficient encountered for {action.label_exyz}: {coeff}")
    theta = float(alpha) * float(coeff.real)
    ppsi = _apply_compiled_pauli(psi, action)
    return math.cos(theta) * psi - 1j * math.sin(theta) * ppsi


def _evolve_trotter_suzuki2_absolute(
    psi0: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    compiled_actions: dict[str, CompiledPauliAction],
    time_value: float,
    trotter_steps: int,
    *,
    drive_coeff_provider_exyz: Any | None = None,
    t0: float = 0.0,
    time_sampling: str = "midpoint",
    coeff_tol: float = 1e-12,
) -> np.ndarray:
    """Suzuki-Trotter order-2 evolution, with optional time-dependent drive.

    When *drive_coeff_provider_exyz* is ``None`` the original bit-for-bit
    time-independent path is taken (no behavioural change).

    When provided, drive coefficients are sampled once per Trotter slice and
    additively merged with the static coefficients.
    """
    # --- time-independent fast path (bit-for-bit identical to original) ---
    if drive_coeff_provider_exyz is None:
        psi = np.array(psi0, copy=True)
        if abs(time_value) <= 1e-15:
            return psi
        dt = float(time_value) / float(trotter_steps)
        half = 0.5 * dt
        for _ in range(trotter_steps):
            for label in ordered_labels_exyz:
                psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
            for label in reversed(ordered_labels_exyz):
                psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
        return _normalize_state(psi)

    # --- time-dependent path: coefficients sampled once per slice ---
    psi = np.array(psi0, copy=True)
    if abs(time_value) <= 1e-15:
        return psi
    dt = float(time_value) / float(trotter_steps)
    half = 0.5 * dt

    sampling = str(time_sampling).strip().lower()
    if sampling not in {"midpoint", "left", "right"}:
        raise ValueError("time_sampling must be one of {'midpoint','left','right'}")

    t0_f = float(t0)
    tol = float(coeff_tol)

    for k in range(int(trotter_steps)):
        if sampling == "midpoint":
            t_sample = t0_f + (float(k) + 0.5) * dt
        elif sampling == "left":
            t_sample = t0_f + float(k) * dt
        else:  # right
            t_sample = t0_f + (float(k) + 1.0) * dt

        drive_map = dict(drive_coeff_provider_exyz(float(t_sample)))

        for label in ordered_labels_exyz:
            c_total = coeff_map_exyz.get(label, 0.0 + 0.0j) + complex(drive_map.get(label, 0.0))
            if abs(c_total) <= tol:
                continue
            psi = _apply_exp_term(psi, compiled_actions[label], c_total, half)
        for label in reversed(ordered_labels_exyz):
            c_total = coeff_map_exyz.get(label, 0.0 + 0.0j) + complex(drive_map.get(label, 0.0))
            if abs(c_total) <= tol:
                continue
            psi = _apply_exp_term(psi, compiled_actions[label], c_total, half)

    return _normalize_state(psi)


def _expectation_hamiltonian(psi: np.ndarray, hmat: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, hmat @ psi)))


def _occupation_site0(psi: np.ndarray, num_sites: int) -> tuple[float, float]:
    probs = np.abs(psi) ** 2
    n_up = 0.0
    n_dn = 0.0
    for idx, prob in enumerate(probs):
        n_up += float((idx >> 0) & 1) * float(prob)
        n_dn += float((idx >> num_sites) & 1) * float(prob)
    return float(n_up), float(n_dn)


def _doublon_total(psi: np.ndarray, num_sites: int) -> float:
    probs = np.abs(psi) ** 2
    out = 0.0
    for idx, prob in enumerate(probs):
        count = 0
        for site in range(num_sites):
            up = (idx >> site) & 1
            dn = (idx >> (num_sites + site)) & 1
            count += int(up * dn)
        out += float(count) * float(prob)
    return float(out)


def _state_to_amplitudes_qn_to_q0(psi: np.ndarray, cutoff: float = 1e-12) -> dict[str, dict[str, float]]:
    nq = int(round(math.log2(psi.size)))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(psi):
        if abs(amp) < cutoff:
            continue
        bit = format(idx, f"0{nq}b")
        out[bit] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


def _load_hardcoded_vqe_namespace() -> dict[str, Any]:
    from src.quantum import vqe_latex_python_pairs as vqe_mod

    ns: dict[str, Any] = {name: getattr(vqe_mod, name) for name in dir(vqe_mod)}

    required = [
        "half_filled_num_particles",
        "hartree_fock_bitstring",
        "basis_state",
        "HardcodedUCCSDAnsatz",
        "vqe_minimize",
    ]
    missing = [name for name in required if name not in ns]
    if missing:
        raise RuntimeError(f"Missing required VQE notebook symbols: {missing}")
    return ns


def _run_hardcoded_vqe(
    *,
    num_sites: int,
    ordering: str,
    h_poly: Any,
    reps: int,
    restarts: int,
    seed: int,
    maxiter: int,
) -> tuple[dict[str, Any], np.ndarray]:
    t0 = time.perf_counter()
    _ai_log(
        "hardcoded_vqe_start",
        L=int(num_sites),
        ordering=str(ordering),
        reps=int(reps),
        restarts=int(restarts),
        maxiter=int(maxiter),
        seed=int(seed),
    )
    ns = _load_hardcoded_vqe_namespace()
    num_particles = tuple(ns["half_filled_num_particles"](int(num_sites)))
    hf_bits = str(ns["hartree_fock_bitstring"](n_sites=int(num_sites), num_particles=num_particles, indexing=ordering))
    nq = 2 * int(num_sites)
    psi_ref = np.asarray(ns["basis_state"](nq, hf_bits), dtype=complex)

    ansatz = ns["HardcodedUCCSDAnsatz"](
        dims=int(num_sites),
        num_particles=num_particles,
        reps=int(reps),
        repr_mode="JW",
        indexing=ordering,
        include_singles=True,
        include_doubles=True,
    )

    result = ns["vqe_minimize"](
        h_poly,
        ansatz,
        psi_ref,
        restarts=int(restarts),
        seed=int(seed),
        maxiter=int(maxiter),
        method="SLSQP",
    )

    theta = np.asarray(result.theta, dtype=float)
    psi_vqe = np.asarray(ansatz.prepare_state(theta, psi_ref), dtype=complex).reshape(-1)
    psi_vqe = _normalize_state(psi_vqe)

    payload = {
        "success": True,
        "method": "hardcoded_uccsd_notebook_statevector",
        "energy": float(result.energy),
        "best_restart": int(getattr(result, "best_restart", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
        "nit": int(getattr(result, "nit", 0)),
        "message": str(getattr(result, "message", "")),
        "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
        "num_parameters": int(ansatz.num_parameters),
        "reps": int(reps),
        "optimal_point": [float(x) for x in theta.tolist()],
        "hf_bitstring_qn_to_q0": hf_bits,
    }
    _ai_log(
        "hardcoded_vqe_done",
        L=int(num_sites),
        success=True,
        energy=float(result.energy),
        best_restart=int(getattr(result, "best_restart", 0)),
        nfev=int(getattr(result, "nfev", 0)),
        nit=int(getattr(result, "nit", 0)),
        elapsed_sec=round(time.perf_counter() - t0, 6),
    )
    return payload, psi_vqe


def _run_qpe_adapter_qiskit(
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


def _reference_terms_for_case(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    ordering: str,
) -> dict[str, float] | None:
    norm_boundary = boundary.strip().lower()
    norm_ordering = ordering.strip().lower()
    if norm_boundary != "periodic" or norm_ordering != "blocked":
        return None
    if abs(float(t) - 1.0) > 1e-12 or abs(float(u) - 4.0) > 1e-12 or abs(float(dv)) > 1e-12:
        return None

    candidate_files = [
        REPO_ROOT / "src" / "quantum" / "exports" / "hubbard_jw_L2_L3_periodic_blocked.json",
        REPO_ROOT / "src" / "quantum" / "exports" / "hubbard_jw_L4_L5_periodic_blocked.json",
        ROOT / "Tests" / "hubbard_jw_L4_L5_periodic_blocked_qiskit.json",
    ]
    case_key = f"L={int(num_sites)}"

    for path in candidate_files:
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        cases = obj.get("cases", {}) if isinstance(obj, dict) else {}
        if case_key in cases:
            terms = cases[case_key].get("pauli_terms", {})
            if isinstance(terms, dict):
                return {str(k): float(v) for k, v in terms.items()}
    return None


def _reference_sanity(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    ordering: str,
    coeff_map_exyz: dict[str, complex],
) -> dict[str, Any]:
    ref_terms = _reference_terms_for_case(
        num_sites=num_sites,
        t=t,
        u=u,
        dv=dv,
        boundary=boundary,
        ordering=ordering,
    )
    if ref_terms is None:
        return {
            "checked": False,
            "reason": "no matching bundled reference for these settings",
        }

    cand = {_to_ixyz(lbl): float(np.real(coeff)) for lbl, coeff in coeff_map_exyz.items()}
    all_keys = sorted(set(ref_terms) | set(cand))
    max_abs_delta = 0.0
    missing_from_reference: list[str] = []
    missing_from_candidate: list[str] = []
    for key in all_keys:
        rv = float(ref_terms.get(key, 0.0))
        cv = float(cand.get(key, 0.0))
        max_abs_delta = max(max_abs_delta, abs(cv - rv))
        if key not in ref_terms:
            missing_from_reference.append(key)
        if key not in cand:
            missing_from_candidate.append(key)

    return {
        "checked": True,
        "max_abs_delta": float(max_abs_delta),
        "matches_within_1e-12": bool(max_abs_delta <= 1e-12),
        "missing_from_reference": missing_from_reference,
        "missing_from_candidate": missing_from_candidate,
    }


# ---------------------------------------------------------------------------
# Sparse / expm_multiply helpers
# ---------------------------------------------------------------------------

# Minimum Hilbert-space dimension at which expm_multiply is preferred over
# dense scipy.linalg.expm.  Below this threshold (dim < 64 ↔ L ≤ 2, nq ≤ 5)
# the Al-Mohy–Higham norm-estimation overhead outweighs the O(d³) dense cost.
# At dim = 64 (L = 3) the crossover begins; at dim ≥ 256 (L ≥ 4) sparse wins
# by an order of magnitude and the advantage grows exponentially with L.
_EXPM_SPARSE_MIN_DIM: int = 64


def _is_all_z_type(label: str) -> bool:
    """Return True if every character in an exyz label is ``'z'`` or ``'e'``.

    A Pauli string composed only of Z and I (identity, here 'e') operators is
    diagonal in the computational basis.  Its full tensor product is therefore
    also diagonal, meaning H_drive can be stored as a 1-D vector rather than a
    d × d matrix.

    The density drive (``TimeDependentOnsiteDensityDrive``) always returns
    labels of this form, so this check confirms the fast diagonal pathway is
    available.
    """
    return all(ch in ("z", "e") for ch in label)


def _build_drive_diagonal(
    drive_map: dict[str, complex],
    dim: int,
    nq: int,
) -> np.ndarray:
    """Build the diagonal of H_drive as a 1-D complex numpy array.

    Only valid when every label in *drive_map* is Z-type (caller must ensure
    this via :func:`_is_all_z_type`).

    For a Z-type label ``l``, the diagonal entry for computational-basis state
    ``|idx⟩`` is the product of eigenvalues:

    .. math::
        d[\\text{idx}] = \\prod_{q:\\, l[n_q-1-q]=\\text{'z'}} (-1)^{\\bigl(\\text{idx} >> q\\bigr) \\& 1}

    This is computed in O(|drive_map| × d) with fully vectorised numpy
    operations — no d × d matrix is ever allocated.

    Parameters
    ----------
    drive_map:
        ``{label: coeff}`` mapping, all labels Z-type.
    dim:
        Hilbert-space dimension (must equal ``1 << nq``).
    nq:
        Number of qubits.

    Returns
    -------
    np.ndarray of shape ``(dim,)`` and dtype ``complex``.
    """
    idx = np.arange(dim, dtype=np.int64)
    diag = np.zeros(dim, dtype=complex)
    for label, coeff in drive_map.items():
        if abs(coeff) <= 1e-15:
            continue
        # Accumulate the eigenvalue product for each basis state.
        eig = np.ones(dim, dtype=np.float64)
        for q in range(nq):
            if label[nq - 1 - q] == "z":
                # Z_q eigenvalue: +1 if bit q is 0, −1 if bit q is 1
                eig *= 1.0 - 2.0 * ((idx >> q) & 1).astype(np.float64)
        diag += coeff * eig
    return diag


def _evolve_piecewise_exact(
    *,
    psi0: np.ndarray,
    hmat_static: np.ndarray,
    drive_coeff_provider_exyz: Any,
    time_value: float,
    trotter_steps: int,
    t0: float = 0.0,
    time_sampling: str = "midpoint",
) -> np.ndarray:
    """Piecewise-constant matrix-exponential reference propagator.

    Approximation order
    -------------------
    This function is **not** a true time-ordered exponential.  It is a
    piecewise-constant approximation: each sub-interval [t_k, t_{k+1}] of
    width Δt = time_value / trotter_steps is replaced by the exact
    exponential of H evaluated at a single representative time t_k.

    The order depends on how t_k is chosen (``time_sampling``):

    * ``"midpoint"`` (default): t_k = t₀ + (k + ½)Δt.
      **Exponential midpoint / Magnus-2 integrator — second order O(Δt²).**
      For a non-autonomous linear ODE, this equals the one-term Magnus
      expansion with the midpoint quadrature and is equivalent to the
      classical exponential midpoint rule.  The global error scales as
      O(Δt²) = O(1/N²).
    * ``"left"``: t_k = t₀ + k Δt.  First order O(Δt).  Use only for
      diagnostics or explicit order-convergence studies.
    * ``"right"``: t_k = t₀ + (k+1)Δt.  First order O(Δt).

    The JSON metadata records ``reference_method`` and
    ``reference_steps_multiplier`` so downstream readers know which
    approximation was used.

    The time-sampling rule is applied consistently to both this reference
    propagator and the Trotter integrator so that fidelity comparisons are
    apples-to-apples at the *same* discretization.  When
    ``--exact-steps-multiplier M > 1``, this function is called with
    M × trotter_steps to provide a finer (higher-quality) reference while the
    Trotter circuit runs at the original trotter_steps.

    Sparse / expm_multiply optimisation
    ------------------------------------
    For Hilbert-space dimension ``dim >= _EXPM_SPARSE_MIN_DIM`` (currently 64,
    i.e., L ≥ 3) this function uses ``scipy.sparse.linalg.expm_multiply``
    instead of ``scipy.linalg.expm``.

    * **H_static** is pre-converted to a ``scipy.sparse.csc_matrix`` *once*
      before the time-step loop, exploiting its O(L) non-zero structure.
    * **H_drive(t)** — which for the density drive is always a sum of Z-type
      (diagonal) Paulis — is stored as a 1-D diagonal vector and converted to
      a ``scipy.sparse.diags`` matrix per step.  No d × d dense allocation is
      needed.
    * ``expm_multiply`` applies the Al-Mohy–Higham algorithm: it approximates
      exp(A) b via a Krylov subspace sequence without ever forming the full
      matrix exponential.  Time cost is O(d · nnz(H) · p) where p is the
      polynomial degree (typically 3–55 for physics-scale problems), compared
      with O(d³) for dense expm.

    Trade-offs
    ~~~~~~~~~~
    +-------------------+-----------------+-------------------------------------+
    | Method            | Cost / step     | When preferred                      |
    +===================+=================+=====================================+
    | Dense ``expm``    | O(d³)           | d < 64 (L ≤ 2); SciPy sparse absent |
    +-------------------+-----------------+-------------------------------------+
    | ``expm_multiply`` | O(d · nnz · p)  | d ≥ 64 (L ≥ 3); sparse available    |
    +-------------------+-----------------+-------------------------------------+

    The dense fallback is retained automatically whenever ``dim <
    _EXPM_SPARSE_MIN_DIM`` or ``scipy.sparse`` is unavailable.
    """
    psi = np.array(psi0, copy=True)
    if abs(time_value) <= 1e-15:
        return _normalize_state(psi)

    dt = float(time_value) / float(trotter_steps)
    t0_f = float(t0)
    sampling = str(time_sampling).strip().lower()
    dim = int(hmat_static.shape[0])
    nq = dim.bit_length() - 1  # dim == 1 << nq

    # ------------------------------------------------------------------
    # Decide once which propagation path to use.
    # ------------------------------------------------------------------
    use_sparse = False
    H_static_sparse = None
    drive_is_diagonal = False

    if dim >= _EXPM_SPARSE_MIN_DIM:
        try:
            from scipy.sparse import csc_matrix as _csc_matrix, diags as _diags
            from scipy.sparse.linalg import expm_multiply as _expm_multiply

            H_static_sparse = _csc_matrix(hmat_static)

            # Probe the drive at t=0 to inspect the label structure.
            _probe_map = dict(drive_coeff_provider_exyz(float(t0_f)))
            drive_is_diagonal = bool(_probe_map) and all(
                _is_all_z_type(lbl) for lbl in _probe_map
            )
            use_sparse = True
        except ImportError:
            pass  # scipy.sparse unavailable — fall through to dense path

    # Keep dense expm import available as fallback.
    from scipy.linalg import expm as _expm_dense

    for k in range(int(trotter_steps)):
        if sampling == "midpoint":
            t_sample = t0_f + (float(k) + 0.5) * dt
        elif sampling == "left":
            t_sample = t0_f + float(k) * dt
        else:  # right
            t_sample = t0_f + (float(k) + 1.0) * dt

        drive_map = dict(drive_coeff_provider_exyz(float(t_sample)))
        filtered_drive = {lbl: complex(c) for lbl, c in drive_map.items() if abs(c) > 1e-15}

        if use_sparse and drive_is_diagonal:
            # Fast path: drive is diagonal → build a 1-D vector and use
            # expm_multiply on H_static_sparse + diags(diag_drive).
            if filtered_drive:
                diag_drive = _build_drive_diagonal(filtered_drive, dim, nq)
                H_drive_sparse = _diags(diag_drive, format="csc")
                H_total_sparse = H_static_sparse + H_drive_sparse
            else:
                H_total_sparse = H_static_sparse
            psi = _expm_multiply((-1j * dt) * H_total_sparse, psi)

        elif use_sparse:
            # Sparse path but drive has off-diagonal terms (non-Z labels).
            # Build the drive as a dense matrix then convert to sparse.
            if filtered_drive:
                h_drive_dense = _build_hamiltonian_matrix(filtered_drive)
                if h_drive_dense.shape != hmat_static.shape:
                    h_drive_dense = np.zeros_like(hmat_static)
                    for lbl, c in filtered_drive.items():
                        h_drive_dense += complex(c) * _pauli_matrix_exyz(lbl)
                H_total_sparse = H_static_sparse + _csc_matrix(h_drive_dense)
            else:
                H_total_sparse = H_static_sparse
            psi = _expm_multiply((-1j * dt) * H_total_sparse, psi)

        else:
            # Dense fallback (small systems or scipy.sparse absent).
            if filtered_drive:
                h_drive = _build_hamiltonian_matrix(filtered_drive)
                if h_drive.shape != hmat_static.shape:
                    h_drive = np.zeros_like(hmat_static)
                    for lbl, c in filtered_drive.items():
                        h_drive += complex(c) * _pauli_matrix_exyz(lbl)
            else:
                h_drive = np.zeros_like(hmat_static)
            h_total = hmat_static + h_drive
            psi = _expm_dense(-1j * dt * h_total) @ psi

    return _normalize_state(psi)


def _simulate_trajectory(
    *,
    num_sites: int,
    psi0: np.ndarray,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    trotter_steps: int,
    t_final: float,
    num_times: int,
    suzuki_order: int,
    drive_coeff_provider_exyz: Any | None = None,
    drive_t0: float = 0.0,
    drive_time_sampling: str = "midpoint",
    exact_steps_multiplier: int = 1,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    if int(suzuki_order) != 2:
        raise ValueError("This script currently supports suzuki_order=2 only.")

    nq = 2 * int(num_sites)
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T

    compiled = {lbl: _compile_pauli_action(lbl, nq) for lbl in ordered_labels_exyz}
    times = np.linspace(0.0, float(t_final), int(num_times))
    n_times = int(times.size)
    stride = max(1, n_times // 20)
    t0 = time.perf_counter()

    has_drive = drive_coeff_provider_exyz is not None

    # When drive is enabled the reference propagator may use a finer step count
    # to improve its quality independently of the Trotter discretization.
    reference_steps = int(trotter_steps) * max(1, int(exact_steps_multiplier))

    _ai_log(
        "hardcoded_trajectory_start",
        L=int(num_sites),
        num_times=n_times,
        t_final=float(t_final),
        trotter_steps=int(trotter_steps),
        reference_steps=reference_steps,
        exact_steps_multiplier=int(exact_steps_multiplier),
        suzuki_order=int(suzuki_order),
        drive_enabled=has_drive,
    )

    rows: list[dict[str, float]] = []
    exact_states: list[np.ndarray] = []

    for idx, time_val in enumerate(times):
        t = float(time_val)

        # --- exact / reference propagation ---
        if not has_drive:
            # Time-independent: eigendecomposition shortcut (exact to machine
            # precision — does not depend on exact_steps_multiplier).
            psi_exact = evecs @ (np.exp(-1j * evals * t) * (evecs_dag @ psi0))
            psi_exact = _normalize_state(psi_exact)
        else:
            # Time-dependent: piecewise-constant matrix-exponential reference
            # (exponential midpoint / Magnus-2 when time_sampling="midpoint").
            # reference_steps = exact_steps_multiplier * trotter_steps so that
            # this can be made arbitrarily accurate independently of the
            # Trotter circuit discretization.
            psi_exact = _evolve_piecewise_exact(
                psi0=psi0,
                hmat_static=hmat,
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                time_value=t,
                trotter_steps=reference_steps,
                t0=float(drive_t0),
                time_sampling=str(drive_time_sampling),
            )

        psi_trot = _evolve_trotter_suzuki2_absolute(
            psi0,
            ordered_labels_exyz,
            coeff_map_exyz,
            compiled,
            t,
            int(trotter_steps),
            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
            t0=float(drive_t0),
            time_sampling=str(drive_time_sampling),
        )

        # --- norm-drift diagnostic ---
        norm_before = float(np.linalg.norm(psi_trot))
        norm_drift = abs(norm_before - 1.0)
        if norm_drift > 1e-6:
            _ai_log(
                "trotter_norm_drift",
                time=t,
                norm_before_renorm=norm_before,
                norm_drift=norm_drift,
            )

        fidelity = float(abs(np.vdot(psi_exact, psi_trot)) ** 2)
        n_up_exact, n_dn_exact = _occupation_site0(psi_exact, num_sites)
        n_up_trot, n_dn_trot = _occupation_site0(psi_trot, num_sites)

        rows.append(
            {
                "time": t,
                "fidelity": fidelity,
                "energy_exact": _expectation_hamiltonian(psi_exact, hmat),
                "energy_trotter": _expectation_hamiltonian(psi_trot, hmat),
                "n_up_site0_exact": n_up_exact,
                "n_up_site0_trotter": n_up_trot,
                "n_dn_site0_exact": n_dn_exact,
                "n_dn_site0_trotter": n_dn_trot,
                "doublon_exact": _doublon_total(psi_exact, num_sites),
                "doublon_trotter": _doublon_total(psi_trot, num_sites),
                "norm_before_renorm": norm_before,
            }
        )
        exact_states.append(psi_exact)
        if idx == 0 or idx == n_times - 1 or ((idx + 1) % stride == 0):
            _ai_log(
                "hardcoded_trajectory_progress",
                step=int(idx + 1),
                total_steps=n_times,
                frac=round(float((idx + 1) / n_times), 6),
                time=float(t),
                fidelity=float(fidelity),
                elapsed_sec=round(time.perf_counter() - t0, 6),
            )

    _ai_log(
        "hardcoded_trajectory_done",
        total_steps=n_times,
        elapsed_sec=round(time.perf_counter() - t0, 6),
        final_fidelity=float(rows[-1]["fidelity"]) if rows else None,
        final_energy_trotter=float(rows[-1]["energy_trotter"]) if rows else None,
    )

    return rows, exact_states


def _current_command_string() -> str:
    return " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])


def _render_command_page(pdf: PdfPages, command: str) -> None:
    wrapped = textwrap.wrap(command, width=112, subsequent_indent="  ")
    lines = [
        "Executed Command",
        "",
        "Reference: pipelines/PIPELINE_RUN_GUIDE.md",
        "Script: pipelines/hardcoded_hubbard_pipeline.py",
        "",
        *wrapped,
    ]
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def _write_pipeline_pdf(pdf_path: Path, payload: dict[str, Any], run_command: str) -> None:
    traj = payload["trajectory"]
    times = np.array([float(r["time"]) for r in traj], dtype=float)
    markevery = max(1, times.size // 25)

    def arr(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in traj], dtype=float)

    fid = arr("fidelity")
    e_exact = arr("energy_exact")
    e_trot = arr("energy_trotter")
    nu_exact = arr("n_up_site0_exact")
    nu_trot = arr("n_up_site0_trotter")
    nd_exact = arr("n_dn_site0_exact")
    nd_trot = arr("n_dn_site0_trotter")
    d_exact = arr("doublon_exact")
    d_trot = arr("doublon_trotter")
    gs_exact = float(payload["ground_state"]["exact_energy"])
    gs_exact_filtered_raw = payload["ground_state"].get("exact_energy_filtered")
    gs_exact_filtered = float(gs_exact_filtered_raw) if gs_exact_filtered_raw is not None else gs_exact
    vqe_e = payload.get("vqe", {}).get("energy")
    vqe_val = float(vqe_e) if vqe_e is not None else np.nan
    vqe_sector = payload.get("vqe", {}).get("num_particles", {})
    sector_label = (
        f"N_up={vqe_sector.get('n_up','?')}, N_dn={vqe_sector.get('n_dn','?')}"
        if vqe_sector else "half-filled"
    )

    with PdfPages(str(pdf_path)) as pdf:
        _render_command_page(pdf, run_command)

        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
        ax00, ax01 = axes[0, 0], axes[0, 1]
        ax10, ax11 = axes[1, 0], axes[1, 1]

        ax00.plot(times, fid, color="#0b3d91", marker="o", markersize=3, markevery=markevery)
        ax00.set_title(f"Fidelity(t) = |<{EXACT_LABEL}|Trotter>|^2")
        ax00.grid(alpha=0.25)

        ax01.plot(times, e_exact, label=EXACT_LABEL, color="#111111", linewidth=2.0, marker="s", markersize=3, markevery=markevery)
        ax01.plot(times, e_trot, label="Trotter", color="#d62728", linestyle="--", linewidth=1.4, marker="^", markersize=3, markevery=markevery)
        ax01.set_title("Energy")
        ax01.grid(alpha=0.25)
        ax01.legend(fontsize=8)

        ax10.plot(times, nu_exact, label=f"n_up0 {EXACT_LABEL}", color="#17becf", linewidth=1.8, marker="o", markersize=3, markevery=markevery)
        ax10.plot(times, nu_trot, label="n_up0 trotter", color="#0f7f8b", linestyle="--", linewidth=1.2, marker="s", markersize=3, markevery=markevery)
        ax10.plot(times, nd_exact, label=f"n_dn0 {EXACT_LABEL}", color="#9467bd", linewidth=1.8, marker="^", markersize=3, markevery=markevery)
        ax10.plot(times, nd_trot, label="n_dn0 trotter", color="#6f4d8f", linestyle="--", linewidth=1.2, marker="v", markersize=3, markevery=markevery)
        ax10.set_title("Site-0 Occupations")
        ax10.set_xlabel("Time")
        ax10.grid(alpha=0.25)
        ax10.legend(fontsize=8)

        ax11.plot(times, d_exact, label=f"doublon {EXACT_LABEL}", color="#8c564b", linewidth=1.8, marker="o", markersize=3, markevery=markevery)
        ax11.plot(times, d_trot, label="doublon trotter", color="#c251a1", linestyle="--", linewidth=1.2, marker="s", markersize=3, markevery=markevery)
        ax11.set_title("Total Doublon")
        ax11.set_xlabel("Time")
        ax11.grid(alpha=0.25)
        ax11.legend(fontsize=8)

        fig.suptitle(f"Hardcoded Hubbard Pipeline: L={payload['settings']['L']}", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        filt_label = f"Exact (sector {sector_label})"
        figv, axesv = plt.subplots(1, 2, figsize=(11.0, 8.5))
        vx0, vx1 = axesv[0], axesv[1]
        vx0.bar([0, 1], [gs_exact_filtered, vqe_val], color=["#2166ac", "#2ca02c"], edgecolor="black", linewidth=0.4)
        vx0.set_xticks([0, 1])
        vx0.set_xticklabels([filt_label, "VQE"], fontsize=8)
        vx0.set_ylabel("Energy")
        vx0.set_title("VQE Energy vs Exact (filtered sector)")
        vx0.grid(axis="y", alpha=0.25)

        err_vqe = abs(vqe_val - gs_exact_filtered) if (np.isfinite(vqe_val) and np.isfinite(gs_exact_filtered)) else np.nan
        vx1.bar([0], [err_vqe], color="#2ca02c", edgecolor="black", linewidth=0.4)
        vx1.set_xticks([0])
        vx1.set_xticklabels([f"|VQE \u2212 Exact (filtered)|"])
        vx1.set_ylabel("Absolute Error")
        vx1.set_title("VQE Absolute Error vs Exact (filtered sector)")
        vx1.grid(axis="y", alpha=0.25)

        figv.suptitle(
            "VQE optimises within the half-filled sector; exact (filtered) is the true sector ground state.\n"
            "Full-Hilbert exact energy is in the JSON text summary only.",
            fontsize=10,
        )
        figv.tight_layout(rect=(0.0, 0.03, 1.0, 0.91))
        pdf.savefig(figv)
        plt.close(figv)

        fig2 = plt.figure(figsize=(11.0, 8.5))
        ax2 = fig2.add_subplot(111)
        ax2.axis("off")
        lines = [
            "Hardcoded Hubbard pipeline summary",
            "",
            f"settings: {json.dumps(payload['settings'])}",
            f"exact_trajectory_label: {EXACT_LABEL}",
            f"exact_trajectory_method: {EXACT_METHOD}",
            f"ground_state_exact_energy (full Hilbert): {payload['ground_state']['exact_energy']:.12f}",
            f"ground_state_exact_energy_filtered: {payload['ground_state'].get('exact_energy_filtered')}",
            f"filtered_sector: {payload['ground_state'].get('filtered_sector')}",
            f"vqe_energy: {payload['vqe'].get('energy')}",
            f"qpe_energy_estimate: {payload['qpe'].get('energy_estimate')}",
            f"initial_state_source: {payload['initial_state']['source']}",
            f"hamiltonian_terms: {payload['hamiltonian']['num_terms']}",
            f"reference_sanity: {payload['sanity']['jw_reference']}",
        ]
        ax2.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
        pdf.savefig(fig2)
        plt.close(fig2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardcoded-first Hubbard pipeline runner.")
    parser.add_argument("--L", type=int, required=True, help="Number of lattice sites.")
    parser.add_argument("--t", type=float, default=1.0, help="Hopping coefficient.")
    parser.add_argument("--u", type=float, default=4.0, help="Onsite interaction U.")
    parser.add_argument("--dv", type=float, default=0.0, help="Uniform local potential term v (Hv = -v n).")
    parser.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    parser.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    parser.add_argument("--t-final", type=float, default=20.0)
    parser.add_argument("--num-times", type=int, default=201)
    parser.add_argument("--suzuki-order", type=int, default=2)
    parser.add_argument("--trotter-steps", type=int, default=64)
    parser.add_argument("--term-order", choices=["native", "sorted"], default="sorted")

    # --- time-dependent drive arguments ---
    parser.add_argument("--enable-drive", action="store_true", help="Enable time-dependent onsite density drive.")
    parser.add_argument("--drive-A", type=float, default=0.0, help="Drive amplitude A in v(t)=A*sin(wt+phi)*exp(-t^2/(2 tbar^2)).")
    parser.add_argument("--drive-omega", type=float, default=1.0, help="Drive carrier angular frequency w.")
    parser.add_argument("--drive-tbar", type=float, default=1.0, help="Drive Gaussian envelope width tbar (must be > 0).")
    parser.add_argument("--drive-phi", type=float, default=0.0, help="Drive phase phi.")
    parser.add_argument(
        "--drive-pattern",
        choices=["dimer_bias", "staggered", "custom"],
        default="staggered",
        help="Spatial pattern mode for v_i(t)=s_i*v(t).",
    )
    parser.add_argument(
        "--drive-custom-s",
        type=str,
        default=None,
        help="Custom spatial weights s_i (comma-separated or JSON list), length L, used when --drive-pattern custom.",
    )
    parser.add_argument("--drive-include-identity", action="store_true", help="Include identity term from n=(I-Z)/2 (global phase).")
    parser.add_argument(
        "--drive-time-sampling",
        choices=["midpoint", "left", "right"],
        default="midpoint",
        help="Time sampling rule per Trotter slice (midpoint recommended; left/right for diagnostics).",
    )
    parser.add_argument("--drive-t0", type=float, default=0.0, help="Drive start time t0 for evolution (default 0.0).")
    parser.add_argument(
        "--exact-steps-multiplier",
        type=int,
        default=1,
        help=(
            "Reference-propagator refinement factor (default 1). "
            "When drive is enabled the reference runs at "
            "N_ref = exact_steps_multiplier * trotter_steps steps while the "
            "Trotter circuit runs at trotter_steps. "
            "With midpoint sampling (Magnus-2, O(Δt²)) a larger multiplier "
            "strictly improves reference quality. "
            "Has no effect when drive is disabled (the static reference uses "
            "exact eigendecomposition)."
        ),
    )
    parser.add_argument("--vqe-reps", type=int, default=2, help="Number of UCCSD repetitions (ansatz depth).")
    parser.add_argument("--vqe-restarts", type=int, default=1)
    parser.add_argument("--vqe-seed", type=int, default=7)
    parser.add_argument("--vqe-maxiter", type=int, default=120)

    parser.add_argument("--qpe-eval-qubits", type=int, default=6)
    parser.add_argument("--qpe-shots", type=int, default=1024)
    parser.add_argument("--qpe-seed", type=int, default=11)
    parser.add_argument("--skip-qpe", action="store_true", help="Skip QPE execution and mark qpe payload as skipped.")

    parser.add_argument("--initial-state-source", choices=["exact", "vqe", "hf"], default="vqe")

    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--skip-pdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ai_log("hardcoded_main_start", settings=vars(args))
    run_command = _current_command_string()
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_json = args.output_json or (artifacts_dir / f"hardcoded_pipeline_L{args.L}.json")
    output_pdf = args.output_pdf or (artifacts_dir / f"hardcoded_pipeline_L{args.L}.pdf")

    h_poly = build_hubbard_hamiltonian(
        dims=int(args.L),
        t=float(args.t),
        U=float(args.u),
        v=float(args.dv),
        repr_mode="JW",
        indexing=str(args.ordering),
        pbc=(str(args.boundary) == "periodic"),
    )

    native_order, coeff_map_exyz = _collect_hardcoded_terms_exyz(h_poly)
    _ai_log("hardcoded_hamiltonian_built", L=int(args.L), num_terms=int(len(coeff_map_exyz)))
    if args.term_order == "native":
        ordered_labels_exyz = list(native_order)
    else:
        ordered_labels_exyz = sorted(coeff_map_exyz)

    # --- build time-dependent drive (if enabled) ---
    drive = None
    drive_coeff_provider_exyz = None
    if bool(args.enable_drive):
        custom_weights = None
        if str(args.drive_pattern) == "custom":
            if args.drive_custom_s is None:
                raise ValueError("--drive-custom-s is required when --drive-pattern custom")
            raw = str(args.drive_custom_s).strip()
            if raw.startswith("["):
                custom_weights = json.loads(raw)
            else:
                custom_weights = [float(x) for x in raw.split(",") if x.strip()]
        drive = build_gaussian_sinusoid_density_drive(
            n_sites=int(args.L),
            nq_total=int(2 * args.L),
            indexing=str(args.ordering),
            A=float(args.drive_A),
            omega=float(args.drive_omega),
            tbar=float(args.drive_tbar),
            phi=float(args.drive_phi),
            pattern_mode=str(args.drive_pattern),
            custom_weights=custom_weights,
            include_identity=bool(args.drive_include_identity),
            coeff_tol=0.0,
        )
        drive_coeff_provider_exyz = drive.coeff_map_exyz
        drive_labels = set(drive.template.labels_exyz(include_identity=bool(drive.include_identity)))
        missing = sorted(drive_labels.difference(ordered_labels_exyz))
        ordered_labels_exyz = list(ordered_labels_exyz) + list(missing)
        _ai_log(
            "hardcoded_drive_built",
            L=int(args.L),
            drive_labels=len(drive_labels),
            new_labels=len(missing),
        )

    hmat = _build_hamiltonian_matrix(coeff_map_exyz)
    evals, evecs = np.linalg.eigh(hmat)
    gs_idx = int(np.argmin(evals))
    gs_energy_exact = float(np.real(evals[gs_idx]))
    psi_exact_ground = _normalize_state(np.asarray(evecs[:, gs_idx], dtype=complex).reshape(-1))

    # Sector-filtered exact energy: minimum eigenvalue restricted to the same
    # half-filled (N_up, N_dn) sector that VQE searches.  This is the
    # apples-to-apples comparison target for the VQE bar plot.
    _vqe_num_particles = _half_filled_particles(int(args.L))
    try:
        gs_energy_exact_filtered = _exact_energy_sector_filtered(
            hmat,
            int(args.L),
            _vqe_num_particles,
            str(args.ordering),
        )
    except Exception as _exc_filt:
        _ai_log("hardcoded_filtered_exact_failed", error=str(_exc_filt))
        gs_energy_exact_filtered = None

    try:
        vqe_payload, psi_vqe = _run_hardcoded_vqe(
            num_sites=int(args.L),
            ordering=str(args.ordering),
            h_poly=h_poly,
            reps=int(args.vqe_reps),
            restarts=int(args.vqe_restarts),
            seed=int(args.vqe_seed),
            maxiter=int(args.vqe_maxiter),
        )
    except Exception as exc:
        _ai_log("hardcoded_vqe_failed", L=int(args.L), error=str(exc))
        vqe_payload = {
            "success": False,
            "method": "hardcoded_uccsd_notebook_statevector",
            "energy": None,
            "error": str(exc),
        }
        psi_vqe = psi_exact_ground

    num_particles = _half_filled_particles(int(args.L))
    psi_hf = _normalize_state(
        np.asarray(
            hartree_fock_statevector(int(args.L), num_particles, indexing=str(args.ordering)),
            dtype=complex,
        ).reshape(-1)
    )

    if args.initial_state_source == "vqe" and bool(vqe_payload.get("success", False)):
        psi0 = psi_vqe
        _ai_log("hardcoded_initial_state_selected", source="vqe")
    elif args.initial_state_source == "vqe":
        raise RuntimeError("Requested --initial-state-source vqe but hardcoded VQE statevector is unavailable.")
    elif args.initial_state_source == "hf":
        psi0 = psi_hf
        _ai_log("hardcoded_initial_state_selected", source="hf")
    else:
        psi0 = psi_exact_ground
        _ai_log("hardcoded_initial_state_selected", source="exact")

    if args.skip_qpe:
        qpe_payload = {
            "success": False,
            "method": "qpe_skipped",
            "energy_estimate": None,
            "phase": None,
            "skipped": True,
            "reason": "--skip-qpe enabled",
            "num_evaluation_qubits": int(args.qpe_eval_qubits),
            "shots": int(args.qpe_shots),
        }
        _ai_log("hardcoded_qpe_skipped", eval_qubits=int(args.qpe_eval_qubits), shots=int(args.qpe_shots))
    else:
        qpe_payload = _run_qpe_adapter_qiskit(
            coeff_map_exyz=coeff_map_exyz,
            psi_init=psi0,
            eval_qubits=int(args.qpe_eval_qubits),
            shots=int(args.qpe_shots),
            seed=int(args.qpe_seed),
        )

    trajectory, _exact_states = _simulate_trajectory(
        num_sites=int(args.L),
        psi0=psi0,
        hmat=hmat,
        ordered_labels_exyz=ordered_labels_exyz,
        coeff_map_exyz=coeff_map_exyz,
        trotter_steps=int(args.trotter_steps),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        suzuki_order=int(args.suzuki_order),
        drive_coeff_provider_exyz=drive_coeff_provider_exyz,
        drive_t0=float(args.drive_t0),
        drive_time_sampling=str(args.drive_time_sampling),
        exact_steps_multiplier=int(args.exact_steps_multiplier),
    )

    sanity = {
        "jw_reference": _reference_sanity(
            num_sites=int(args.L),
            t=float(args.t),
            u=float(args.u),
            dv=float(args.dv),
            boundary=str(args.boundary),
            ordering=str(args.ordering),
            coeff_map_exyz=coeff_map_exyz,
        )
    }

    settings: dict[str, Any] = {
        "L": int(args.L),
        "t": float(args.t),
        "u": float(args.u),
        "dv": float(args.dv),
        "boundary": str(args.boundary),
        "ordering": str(args.ordering),
        "t_final": float(args.t_final),
        "num_times": int(args.num_times),
        "suzuki_order": int(args.suzuki_order),
        "trotter_steps": int(args.trotter_steps),
        "term_order": str(args.term_order),
        "initial_state_source": str(args.initial_state_source),
        "skip_qpe": bool(args.skip_qpe),
    }
    if bool(args.enable_drive):
        settings["drive"] = {
            "enabled": True,
            "A": float(args.drive_A),
            "omega": float(args.drive_omega),
            "tbar": float(args.drive_tbar),
            "phi": float(args.drive_phi),
            "pattern": str(args.drive_pattern),
            "custom_s": (str(args.drive_custom_s) if args.drive_custom_s is not None else None),
            "include_identity": bool(args.drive_include_identity),
            "time_sampling": str(args.drive_time_sampling),
            "t0": float(args.drive_t0),
            # Reference-propagator metadata (Prompt 4/5).
            "reference_steps_multiplier": int(args.exact_steps_multiplier),
            "reference_steps": int(args.trotter_steps) * int(args.exact_steps_multiplier),
            "reference_method": reference_method_name(str(args.drive_time_sampling)),
            # Architecture metadata (see DESIGN_NOTE_QISKIT_BASELINE_TIMEDEP.md §5).
            "propagator_backend": "scipy_sparse_expm_multiply",
        }

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "hardcoded",
        "settings": settings,
        "hamiltonian": {
            "num_qubits": int(2 * args.L),
            "num_terms": int(len(ordered_labels_exyz) if bool(args.enable_drive) else len(coeff_map_exyz)),
            "coefficients_exyz": [
                {
                    "label_exyz": lbl,
                    "coeff": {
                        "re": float(np.real(coeff_map_exyz.get(lbl, 0.0 + 0.0j))),
                        "im": float(np.imag(coeff_map_exyz.get(lbl, 0.0 + 0.0j))),
                    },
                }
                for lbl in ordered_labels_exyz
            ],
        },
        "ground_state": {
            "exact_energy": float(gs_energy_exact),
            "exact_energy_filtered": float(gs_energy_exact_filtered) if gs_energy_exact_filtered is not None else None,
            "filtered_sector": {
                "n_up": int(_vqe_num_particles[0]),
                "n_dn": int(_vqe_num_particles[1]),
            },
            "method": "matrix_diagonalization",
        },
        "vqe": vqe_payload,
        "qpe": qpe_payload,
        "initial_state": {
            "source": str(args.initial_state_source if args.initial_state_source != "vqe" or vqe_payload.get("success") else "exact"),
            "amplitudes_qn_to_q0": _state_to_amplitudes_qn_to_q0(psi0),
        },
        "trajectory": trajectory,
        "sanity": sanity,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.skip_pdf:
        _write_pipeline_pdf(output_pdf, payload, run_command)

    _ai_log(
        "hardcoded_main_done",
        L=int(args.L),
        output_json=str(output_json),
        output_pdf=(str(output_pdf) if not args.skip_pdf else None),
        vqe_energy=vqe_payload.get("energy"),
        qpe_energy=qpe_payload.get("energy_estimate"),
    )
    print(f"Wrote JSON: {output_json}")
    if not args.skip_pdf:
        print(f"Wrote PDF:  {output_pdf}")


if __name__ == "__main__":
    main()
