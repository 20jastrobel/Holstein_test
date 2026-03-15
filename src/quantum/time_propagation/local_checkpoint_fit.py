"""Local checkpoint-fit dynamics utilities for fixed-seed HH hardware screens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


@dataclass(frozen=True)
class LocalPauliAnsatzSpec:
    num_qubits: int
    reps: int
    single_axes: tuple[str, ...] = ("y",)
    entangler_axes: tuple[str, ...] = ("xx", "zz")
    entangler_edges: tuple[tuple[int, int], ...] | None = None
    repr_mode: str = "JW"


@dataclass(frozen=True)
class CheckpointFitConfig:
    optimizer_method: str = "L-BFGS-B"
    maxiter: int = 60
    gtol: float = 1e-8
    ftol: float = 1e-12
    angle_bound: float = math.pi
    param_shift: float = math.pi / 2.0
    coefficient_tolerance: float = 1e-12


@dataclass(frozen=True)
class CheckpointFitStepResult:
    theta: np.ndarray
    state: np.ndarray
    fidelity: float
    objective: float
    solver_row: dict[str, Any]


@dataclass(frozen=True)
class CheckpointFitTrajectoryResult:
    times: np.ndarray
    theta_history: np.ndarray
    states: tuple[np.ndarray, ...]
    solver_rows: tuple[dict[str, Any], ...]


_MATH_NORMALIZE_STATE = r"\hat{\psi}=\psi/\|\psi\|"


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    vec = np.asarray(psi, dtype=complex).reshape(-1)
    nrm = float(np.linalg.norm(vec))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return vec / nrm


_MATH_CHAIN_EDGES = r"\mathcal{E}_{\mathrm{chain}}=\{(q,q+1)\ |\ q=0,\dots,n_q-2\}"


def default_chain_edges(num_qubits: int) -> tuple[tuple[int, int], ...]:
    nq = int(num_qubits)
    if nq < 1:
        raise ValueError("num_qubits must be positive.")
    return tuple((int(q), int(q + 1)) for q in range(nq - 1))


_MATH_PAULI_WORD = r"P(\{(q,\sigma_q)\})=\bigotimes_{j=n_q-1}^{0}\sigma_j,\ \sigma_j=e\ \text{unless assigned}"


def _pauli_word(num_qubits: int, ops: dict[int, str]) -> str:
    nq = int(num_qubits)
    chars = ["e"] * nq
    for qubit, axis in ops.items():
        q = int(qubit)
        if q < 0 or q >= nq:
            raise ValueError(f"qubit index {q} out of range for nq={nq}.")
        sym = str(axis).strip().lower()
        if sym not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported Pauli axis {axis!r}.")
        chars[nq - 1 - q] = sym
    return "".join(chars)


_MATH_MONOMIAL_TERM = r"H_k=P_k,\ \ U_k(\theta_k)=\exp(-i\theta_k P_k)"


def _monomial_term(
    *,
    num_qubits: int,
    label: str,
    ops: dict[int, str],
    repr_mode: str,
) -> AnsatzTerm:
    word = _pauli_word(int(num_qubits), dict(ops))
    poly = PauliPolynomial(str(repr_mode), [PauliTerm(int(num_qubits), ps=word, pc=1.0)])
    return AnsatzTerm(label=str(label), polynomial=poly)


_MATH_BUILD_LOCAL_ANSA = (
    r"\mathcal{A}=\prod_{\ell=1}^{r}\left[\prod_{q,a\in S}\exp(-i\theta_{\ell,q,a}\sigma_a^{(q)})"
    r"\prod_{(i,j),b\in E}\exp(-i\theta_{\ell,i,j,b}\sigma_b^{(i)}\sigma_b^{(j)})\right]"
)


def build_local_pauli_ansatz_terms(spec: LocalPauliAnsatzSpec) -> tuple[AnsatzTerm, ...]:
    nq = int(spec.num_qubits)
    reps = int(spec.reps)
    if nq < 1:
        raise ValueError("num_qubits must be positive.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    single_axes = tuple(str(axis).strip().lower() for axis in spec.single_axes)
    entangler_axes = tuple(str(axis).strip().lower() for axis in spec.entangler_axes)
    if any(axis not in {"x", "y", "z"} for axis in single_axes):
        raise ValueError("single_axes must contain only x/y/z.")
    if any(axis not in {"xx", "yy", "zz"} for axis in entangler_axes):
        raise ValueError("entangler_axes must contain only xx/yy/zz.")
    edges = tuple(spec.entangler_edges or default_chain_edges(int(nq)))
    terms: list[AnsatzTerm] = []
    for rep_idx in range(reps):
        for axis in single_axes:
            for qubit in range(nq):
                terms.append(
                    _monomial_term(
                        num_qubits=nq,
                        label=f"local_{axis}(q={qubit})_rep{rep_idx + 1}",
                        ops={int(qubit): str(axis)},
                        repr_mode=str(spec.repr_mode),
                    )
                )
        for axis in entangler_axes:
            pauli_axis = str(axis)[0]
            for q0, q1 in edges:
                terms.append(
                    _monomial_term(
                        num_qubits=nq,
                        label=f"local_{axis}(q={int(q0)},{int(q1)})_rep{rep_idx + 1}",
                        ops={int(q0): pauli_axis, int(q1): pauli_axis},
                        repr_mode=str(spec.repr_mode),
                    )
                )
    return tuple(terms)


_MATH_STATE_FIDELITY = r"F(\psi,\phi)=|\langle\phi|\psi\rangle|^2"


def state_fidelity(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    lhs = _normalize_state(psi_a)
    rhs = _normalize_state(psi_b)
    return float(abs(np.vdot(rhs, lhs)) ** 2)


_MATH_FIDELITY_OBJECTIVE = (
    r"\mathcal{L}(\theta)=1-F(\psi(\theta),\psi_{\mathrm{target}}),"
    r"\quad \partial_k \psi(\theta)=\frac{\psi(\theta+s e_k)-\psi(\theta-s e_k)}{2\sin s}"
)


def _evaluate_fidelity_objective(
    theta: np.ndarray,
    *,
    executor: CompiledAnsatzExecutor,
    reference_state: np.ndarray,
    target_state: np.ndarray,
    param_shift: float,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    psi = _normalize_state(executor.prepare_state(theta_vec, reference_state))
    target = _normalize_state(target_state)
    overlap = np.vdot(target, psi)
    fidelity = float(abs(overlap) ** 2)
    objective = float(max(0.0, 1.0 - fidelity))
    if int(theta_vec.size) == 0:
        return objective, np.zeros(0, dtype=float), psi, fidelity

    shift = float(param_shift)
    denom = 2.0 * math.sin(shift)
    if abs(denom) <= 1e-15:
        raise ValueError("param_shift leads to zero parameter-shift denominator.")

    grad = np.zeros(int(theta_vec.size), dtype=float)
    for idx in range(int(theta_vec.size)):
        theta_plus = np.asarray(theta_vec, dtype=float).copy()
        theta_minus = np.asarray(theta_vec, dtype=float).copy()
        theta_plus[idx] += shift
        theta_minus[idx] -= shift
        psi_plus = _normalize_state(executor.prepare_state(theta_plus, reference_state))
        psi_minus = _normalize_state(executor.prepare_state(theta_minus, reference_state))
        dpsi = (psi_plus - psi_minus) / denom
        grad[idx] = float(-2.0 * np.real(np.conjugate(overlap) * np.vdot(target, dpsi)))
    return objective, grad, psi, fidelity


def _try_import_scipy_minimize() -> Any:
    try:
        from scipy.optimize import minimize  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Checkpoint fitting requires scipy.optimize.minimize. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return minimize


_MATH_FIT_CHECKPOINT = r"\theta^\star(t)=\arg\min_\theta 1-|\langle\psi_{\mathrm{target}}(t)|U(\theta)|\psi_{\mathrm{ref}}\rangle|^2"


def fit_checkpoint_target_state(
    reference_state: np.ndarray,
    target_state: np.ndarray,
    terms: Sequence[AnsatzTerm],
    *,
    config: CheckpointFitConfig,
    theta_init: np.ndarray | None = None,
    executor: CompiledAnsatzExecutor | None = None,
) -> CheckpointFitStepResult:
    minimize = _try_import_scipy_minimize()
    ref = _normalize_state(reference_state)
    target = _normalize_state(target_state)
    compiled = executor or CompiledAnsatzExecutor(
        list(terms),
        coefficient_tolerance=float(config.coefficient_tolerance),
    )
    npar = int(len(terms))
    theta0 = (
        np.zeros(npar, dtype=float)
        if theta_init is None
        else np.asarray(theta_init, dtype=float).reshape(-1)
    )
    if int(theta0.size) != npar:
        raise ValueError(f"theta_init length {int(theta0.size)} does not match term count {npar}.")

    eval_counter = {"fresh_calls": 0}
    cache: dict[str, Any] = {}

    def _evaluate(theta_raw: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
        theta_vec = np.asarray(theta_raw, dtype=float).reshape(-1)
        cached_theta = cache.get("theta", None)
        if isinstance(cached_theta, np.ndarray) and np.array_equal(theta_vec, cached_theta):
            return (
                float(cache["objective"]),
                np.asarray(cache["gradient"], dtype=float),
                np.asarray(cache["state"], dtype=complex),
                float(cache["fidelity"]),
            )
        objective, gradient, state, fidelity = _evaluate_fidelity_objective(
            theta_vec,
            executor=compiled,
            reference_state=ref,
            target_state=target,
            param_shift=float(config.param_shift),
        )
        cache["theta"] = np.asarray(theta_vec, dtype=float).copy()
        cache["objective"] = float(objective)
        cache["gradient"] = np.asarray(gradient, dtype=float).copy()
        cache["state"] = np.asarray(state, dtype=complex).copy()
        cache["fidelity"] = float(fidelity)
        eval_counter["fresh_calls"] = int(eval_counter["fresh_calls"]) + 1
        return objective, gradient, state, fidelity

    if state_fidelity(ref, target) >= 1.0 - 1e-14 and np.allclose(theta0, 0.0):
        objective, gradient, state, fidelity = _evaluate(theta0)
        return CheckpointFitStepResult(
            theta=np.asarray(theta0, dtype=float).copy(),
            state=np.asarray(state, dtype=complex).copy(),
            fidelity=float(fidelity),
            objective=float(objective),
            solver_row={
                "success": True,
                "status": 0,
                "message": "Exact reference-target match at initialization.",
                "nit": 0,
                "nfev": int(eval_counter["fresh_calls"]),
                "njev": 0,
                "optimizer_method": str(config.optimizer_method),
            },
        )

    bounds = [(-float(config.angle_bound), float(config.angle_bound))] * npar
    result = minimize(
        lambda x: _evaluate(x)[0],
        theta0,
        jac=lambda x: _evaluate(x)[1],
        method=str(config.optimizer_method),
        bounds=bounds,
        options={
            "maxiter": int(config.maxiter),
            "gtol": float(config.gtol),
            "ftol": float(config.ftol),
        },
    )
    objective, gradient, state, fidelity = _evaluate(np.asarray(result.x, dtype=float))
    return CheckpointFitStepResult(
        theta=np.asarray(result.x, dtype=float).reshape(-1),
        state=np.asarray(state, dtype=complex).copy(),
        fidelity=float(fidelity),
        objective=float(objective),
        solver_row={
            "success": bool(result.success),
            "status": int(getattr(result, "status", 0)),
            "message": str(getattr(result, "message", "")),
            "nit": int(getattr(result, "nit", 0)),
            "nfev": int(getattr(result, "nfev", eval_counter["fresh_calls"])),
            "njev": int(getattr(result, "njev", eval_counter["fresh_calls"])),
            "optimizer_method": str(config.optimizer_method),
            "objective": float(objective),
            "gradient_norm": float(np.linalg.norm(np.asarray(gradient, dtype=float))),
        },
    )


_MATH_FIT_TRAJECTORY = r"\theta^\star(t_n)\leftarrow \mathrm{warm\_start}(\theta^\star(t_{n-1}))"


def fit_checkpoint_trajectory(
    reference_state: np.ndarray,
    target_states: Sequence[np.ndarray],
    times: Sequence[float],
    terms: Sequence[AnsatzTerm],
    *,
    config: CheckpointFitConfig,
    theta_init: np.ndarray | None = None,
) -> CheckpointFitTrajectoryResult:
    time_vec = np.asarray(times, dtype=float).reshape(-1)
    if int(time_vec.size) != int(len(target_states)):
        raise ValueError("times and target_states must have the same length.")
    if int(time_vec.size) < 1:
        raise ValueError("Need at least one target checkpoint.")

    compiled = CompiledAnsatzExecutor(
        list(terms),
        coefficient_tolerance=float(config.coefficient_tolerance),
    )
    npar = int(len(terms))
    theta_prev = (
        np.zeros(npar, dtype=float)
        if theta_init is None
        else np.asarray(theta_init, dtype=float).reshape(-1)
    )
    if int(theta_prev.size) != npar:
        raise ValueError(f"theta_init length {int(theta_prev.size)} does not match term count {npar}.")

    states: list[np.ndarray] = []
    theta_rows: list[np.ndarray] = []
    solver_rows: list[dict[str, Any]] = []
    for idx, target_state in enumerate(target_states):
        fit_result = fit_checkpoint_target_state(
            reference_state,
            np.asarray(target_state, dtype=complex),
            terms,
            config=config,
            theta_init=theta_prev,
            executor=compiled,
        )
        theta_prev = np.asarray(fit_result.theta, dtype=float).copy()
        states.append(np.asarray(fit_result.state, dtype=complex))
        theta_rows.append(np.asarray(fit_result.theta, dtype=float))
        solver_rows.append(
            {
                "time": float(time_vec[idx]),
                "fidelity": float(fit_result.fidelity),
                "objective": float(fit_result.objective),
                **dict(fit_result.solver_row),
            }
        )

    return CheckpointFitTrajectoryResult(
        times=np.asarray(time_vec, dtype=float),
        theta_history=np.asarray(theta_rows, dtype=float),
        states=tuple(np.asarray(state, dtype=complex) for state in states),
        solver_rows=tuple(dict(row) for row in solver_rows),
    )


__all__ = [
    "CheckpointFitConfig",
    "CheckpointFitStepResult",
    "CheckpointFitTrajectoryResult",
    "LocalPauliAnsatzSpec",
    "build_local_pauli_ansatz_terms",
    "default_chain_edges",
    "fit_checkpoint_target_state",
    "fit_checkpoint_trajectory",
    "state_fidelity",
]
