"""Projected real-time dynamics utilities for fixed-generator HH surrogates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.vqe_latex_python_pairs import apply_exp_pauli_polynomial


_EXPM_SPARSE_MIN_DIM: int = 64


@dataclass(frozen=True)
class ProjectedRealTimeConfig:
    t_final: float
    num_times: int
    ode_substeps: int = 4
    tangent_eps: float = 1e-6
    lambda_reg: float = 1e-8
    svd_rcond: float = 1e-12
    coefficient_tolerance: float = 1e-12
    sort_terms: bool = True


@dataclass(frozen=True)
class ProjectedRealTimeResult:
    times: np.ndarray
    theta_history: np.ndarray
    states: tuple[np.ndarray, ...]
    trajectory_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ExactDrivenReferenceResult:
    times: np.ndarray
    states: tuple[np.ndarray, ...]
    energies_total: np.ndarray
    trajectory_rows: tuple[dict[str, Any], ...]


_MATH_NORMALIZE_STATE = r"\hat{\psi}=\psi/\|\psi\|"


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    vec = np.asarray(psi, dtype=complex).reshape(-1)
    nrm = float(np.linalg.norm(vec))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return vec / nrm


_MATH_PAULI_MATRIX_EXYZ = r"P(\ell)=\bigotimes_{q=n-1}^{0}\sigma_{\ell_q}"


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = {
        "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    }
    if str(label) == "":
        raise ValueError("Pauli label must be non-empty.")
    out = mats[str(label)[0]]
    for ch in str(label)[1:]:
        out = np.kron(out, mats[ch])
    return out


_MATH_DRIVE_IS_Z_TYPE = r"\ell\ \mathrm{diagonal}\iff \forall q,\ \ell_q\in\{e,z\}"


def _is_all_z_type(label: str) -> bool:
    return all(ch in {"e", "z"} for ch in str(label))


_MATH_DRIVE_DIAGONAL = r"d(\mathrm{idx})=\sum_\ell c_\ell \prod_{q:\ell_q=z}(-1)^{((\mathrm{idx}\gg q)\&1)}"


def _build_drive_diagonal(
    drive_map: Mapping[str, complex],
    *,
    dim: int,
    nq: int,
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    idx = np.arange(int(dim), dtype=np.int64)
    diag = np.zeros(int(dim), dtype=complex)
    local_cache = {} if cache is None else cache
    for label, coeff in drive_map.items():
        coeff_c = complex(coeff)
        if abs(coeff_c) <= 1e-15:
            continue
        eig = local_cache.get(str(label), None)
        if eig is None:
            eig = np.ones(int(dim), dtype=np.float64)
            for q in range(int(nq)):
                if str(label)[int(nq) - 1 - q] == "z":
                    eig *= 1.0 - 2.0 * ((idx >> q) & 1).astype(np.float64)
            local_cache[str(label)] = eig
        diag += coeff_c * eig
    return diag


_MATH_TERM_POLY = r"G_j=\mathrm{poly}(term_j)"


def _term_polynomial(term: Any) -> Any:
    if hasattr(term, "polynomial"):
        return term.polynomial
    return term


_MATH_APPLY_ORDERED_TERMS = r"U(\theta)|\psi_0\rangle=\prod_{j=1}^{K}\widetilde{\exp}(-i\theta_j G_j)|\psi_0\rangle"


def _apply_ordered_terms(
    reference_state: np.ndarray,
    terms: Sequence[Any],
    theta: np.ndarray,
    *,
    coefficient_tolerance: float,
    sort_terms: bool,
) -> np.ndarray:
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    if int(theta_vec.size) != int(len(terms)):
        raise ValueError(
            f"theta length {int(theta_vec.size)} does not match term count {int(len(terms))}."
        )
    psi = _normalize_state(reference_state)
    for idx, term in enumerate(terms):
        psi = apply_exp_pauli_polynomial(
            psi,
            _term_polynomial(term),
            float(theta_vec[idx]),
            coefficient_tolerance=float(coefficient_tolerance),
            sort_terms=bool(sort_terms),
        )
    return _normalize_state(psi)


_MATH_TANGENT_VECTORS = r"\partial_j|\psi(\theta)\rangle=U_K\cdots U_{j+1}\,\partial_{\theta_j}(U_j)\,U_{j-1}\cdots U_1|\psi_0\rangle"


def build_tangent_vectors(
    reference_state: np.ndarray,
    terms: Sequence[Any],
    theta: np.ndarray,
    *,
    tangent_eps: float = 1e-6,
    coefficient_tolerance: float = 1e-12,
    sort_terms: bool = True,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    if int(theta_vec.size) != int(len(terms)):
        raise ValueError(
            f"theta length {int(theta_vec.size)} does not match term count {int(len(terms))}."
        )
    eps = float(tangent_eps)
    if eps <= 0.0:
        raise ValueError("tangent_eps must be > 0.")
    if not terms:
        return _normalize_state(reference_state), ()

    prefix_states: list[np.ndarray] = [_normalize_state(reference_state)]
    for idx, term in enumerate(terms):
        prefix_states.append(
            apply_exp_pauli_polynomial(
                prefix_states[-1],
                _term_polynomial(term),
                float(theta_vec[idx]),
                coefficient_tolerance=float(coefficient_tolerance),
                sort_terms=bool(sort_terms),
            )
        )

    tangents: list[np.ndarray] = []
    for idx, term in enumerate(terms):
        psi_before = prefix_states[idx]
        poly = _term_polynomial(term)
        psi_plus = apply_exp_pauli_polynomial(
            psi_before,
            poly,
            float(theta_vec[idx] + eps),
            coefficient_tolerance=float(coefficient_tolerance),
            sort_terms=bool(sort_terms),
        )
        psi_minus = apply_exp_pauli_polynomial(
            psi_before,
            poly,
            float(theta_vec[idx] - eps),
            coefficient_tolerance=float(coefficient_tolerance),
            sort_terms=bool(sort_terms),
        )
        tangent = (psi_plus - psi_minus) / (2.0 * eps)
        for tail_idx in range(idx + 1, int(len(terms))):
            tangent = apply_exp_pauli_polynomial(
                tangent,
                _term_polynomial(terms[tail_idx]),
                float(theta_vec[tail_idx]),
                coefficient_tolerance=float(coefficient_tolerance),
                sort_terms=bool(sort_terms),
            )
        tangents.append(np.asarray(tangent, dtype=complex).reshape(-1))
    return _normalize_state(prefix_states[-1]), tuple(tangents)


_MATH_SOLVE_MCLACHLAN = r"A_{ij}=\Re\langle \partial_i\psi|\partial_j\psi\rangle,\ C_i=\Im\langle \partial_i\psi|H|\psi\rangle,\ \dot{\theta}=(A+\lambda I)^+C"


def solve_mclachlan_step(
    tangents: Sequence[np.ndarray],
    hpsi: np.ndarray,
    *,
    lambda_reg: float = 1e-8,
    svd_rcond: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not tangents:
        return np.zeros(0, dtype=float), {
            "condition_number": 1.0,
            "regularization": float(lambda_reg),
            "regularization_used": bool(lambda_reg > 0.0),
            "solve_mode": "empty",
            "residual_norm": 0.0,
        }

    tangent_mat = np.column_stack([np.asarray(vec, dtype=complex).reshape(-1) for vec in tangents])
    hpsi_vec = np.asarray(hpsi, dtype=complex).reshape(-1)
    amat = np.real(np.conjugate(tangent_mat).T @ tangent_mat)
    cvec = np.imag(np.conjugate(tangent_mat).T @ hpsi_vec)
    cond = float(np.linalg.cond(amat))
    reg = float(max(0.0, lambda_reg))
    solve_mode = "solve"
    system = amat + reg * np.eye(int(amat.shape[0]), dtype=float)
    try:
        theta_dot = np.linalg.solve(system, cvec)
    except np.linalg.LinAlgError:
        solve_mode = "pinv"
        theta_dot = np.linalg.pinv(system, rcond=float(svd_rcond)) @ cvec
    theta_dot = np.asarray(theta_dot, dtype=float).reshape(-1)
    if not np.all(np.isfinite(theta_dot)):
        solve_mode = "pinv"
        theta_dot = np.asarray(
            np.linalg.pinv(system, rcond=float(svd_rcond)) @ cvec,
            dtype=float,
        ).reshape(-1)
    residual = np.asarray(system @ theta_dot - cvec, dtype=float).reshape(-1)
    return theta_dot, {
        "condition_number": float(cond),
        "regularization": float(reg),
        "regularization_used": bool(reg > 0.0),
        "solve_mode": str(solve_mode),
        "residual_norm": float(np.linalg.norm(residual)),
        "matrix_rank": int(np.linalg.matrix_rank(system)),
    }


_MATH_TOTAL_H_ACTION = r"H(t)|\psi\rangle=(H_{\mathrm{static}}+H_{\mathrm{drive}}(t))|\psi\rangle"


def _apply_total_hamiltonian(
    psi: np.ndarray,
    hmat_static: np.ndarray,
    *,
    drive_coeff_provider_exyz: Any | None,
    t_physical: float,
    drive_diag_cache: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
    hpsi = np.asarray(hmat_static, dtype=complex) @ psi_vec
    if drive_coeff_provider_exyz is None:
        return hpsi, None
    drive_map = {
        str(label): complex(coeff)
        for label, coeff in dict(drive_coeff_provider_exyz(float(t_physical))).items()
        if abs(complex(coeff)) > 1e-15
    }
    if not drive_map:
        return hpsi, None
    nq = int(round(math.log2(int(psi_vec.size))))
    if all(_is_all_z_type(label) for label in drive_map):
        diag = _build_drive_diagonal(
            drive_map,
            dim=int(psi_vec.size),
            nq=int(nq),
            cache=drive_diag_cache,
        )
        return hpsi + diag * psi_vec, np.asarray(diag, dtype=complex)

    h_drive = np.zeros((int(psi_vec.size), int(psi_vec.size)), dtype=complex)
    for label, coeff in drive_map.items():
        h_drive += complex(coeff) * _pauli_matrix_exyz(label)
    return hpsi + h_drive @ psi_vec, None


_MATH_RHS = r"\dot{\theta}(t)=\mathcal{P}_{\theta}(H(t),|\psi(\theta)\rangle)"


def _rhs_theta_dot(
    t_rel: float,
    theta: np.ndarray,
    *,
    reference_state: np.ndarray,
    terms: Sequence[Any],
    hmat_static: np.ndarray,
    drive_coeff_provider_exyz: Any | None,
    drive_t0: float,
    tangent_eps: float,
    lambda_reg: float,
    svd_rcond: float,
    coefficient_tolerance: float,
    sort_terms: bool,
    drive_diag_cache: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    psi, tangents = build_tangent_vectors(
        reference_state,
        terms,
        theta,
        tangent_eps=float(tangent_eps),
        coefficient_tolerance=float(coefficient_tolerance),
        sort_terms=bool(sort_terms),
    )
    hpsi, drive_diag = _apply_total_hamiltonian(
        psi,
        hmat_static,
        drive_coeff_provider_exyz=drive_coeff_provider_exyz,
        t_physical=float(drive_t0) + float(t_rel),
        drive_diag_cache=drive_diag_cache,
    )
    theta_dot, solve_diag = solve_mclachlan_step(
        tangents,
        hpsi,
        lambda_reg=float(lambda_reg),
        svd_rcond=float(svd_rcond),
    )
    diagnostics = dict(solve_diag)
    diagnostics["state_norm"] = float(np.linalg.norm(psi))
    diagnostics["theta_norm"] = float(np.linalg.norm(np.asarray(theta, dtype=float)))
    diagnostics["theta_dot_norm"] = float(np.linalg.norm(theta_dot))
    diagnostics["drive_diag_used"] = bool(drive_diag is not None)
    return theta_dot, psi, diagnostics


_MATH_RUN_PROJECTED_RT = r"\theta_{n+1}=\mathrm{RK4}(\dot{\theta},\Delta t),\ |\psi_n\rangle=U(\theta_n)|\psi_0\rangle"


def run_projected_real_time_trajectory(
    reference_state: np.ndarray,
    terms: Sequence[Any],
    hmat_static: np.ndarray,
    *,
    config: ProjectedRealTimeConfig,
    drive_coeff_provider_exyz: Any | None = None,
    drive_t0: float = 0.0,
    theta_init: np.ndarray | None = None,
) -> ProjectedRealTimeResult:
    cfg = config
    if float(cfg.t_final) < 0.0:
        raise ValueError("t_final must be >= 0.")
    if int(cfg.num_times) < 2:
        raise ValueError("num_times must be >= 2.")
    if int(cfg.ode_substeps) < 1:
        raise ValueError("ode_substeps must be >= 1.")
    theta = (
        np.zeros(int(len(terms)), dtype=float)
        if theta_init is None
        else np.asarray(theta_init, dtype=float).reshape(-1)
    )
    if int(theta.size) != int(len(terms)):
        raise ValueError(
            f"theta_init length {int(theta.size)} does not match term count {int(len(terms))}."
        )
    times = np.linspace(0.0, float(cfg.t_final), int(cfg.num_times))
    drive_diag_cache: dict[str, np.ndarray] = {}
    states: list[np.ndarray] = []
    theta_rows: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for time_idx, t_rel in enumerate(times):
        theta_dot_now, psi_now, diag_now = _rhs_theta_dot(
            float(t_rel),
            theta,
            reference_state=np.asarray(reference_state, dtype=complex),
            terms=terms,
            hmat_static=np.asarray(hmat_static, dtype=complex),
            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
            drive_t0=float(drive_t0),
            tangent_eps=float(cfg.tangent_eps),
            lambda_reg=float(cfg.lambda_reg),
            svd_rcond=float(cfg.svd_rcond),
            coefficient_tolerance=float(cfg.coefficient_tolerance),
            sort_terms=bool(cfg.sort_terms),
            drive_diag_cache=drive_diag_cache,
        )
        states.append(np.asarray(psi_now, dtype=complex))
        theta_rows.append(np.asarray(theta, dtype=float).copy())
        rows.append(
            {
                "time": float(t_rel),
                "state_norm": float(diag_now["state_norm"]),
                "theta_norm": float(diag_now["theta_norm"]),
                "theta_dot_norm": float(diag_now["theta_dot_norm"]),
                "condition_number": float(diag_now["condition_number"]),
                "regularization": float(diag_now["regularization"]),
                "regularization_used": bool(diag_now["regularization_used"]),
                "solve_mode": str(diag_now["solve_mode"]),
                "residual_norm": float(diag_now["residual_norm"]),
                "matrix_rank": int(diag_now["matrix_rank"]),
                "drive_diag_used": bool(diag_now["drive_diag_used"]),
            }
        )
        if time_idx >= int(times.size) - 1:
            continue
        dt = float(times[time_idx + 1] - times[time_idx]) / float(cfg.ode_substeps)
        t_step = float(t_rel)
        for sub_idx in range(int(cfg.ode_substeps)):
            t_sub = t_step + float(sub_idx) * dt
            k1, _psi1, _diag1 = _rhs_theta_dot(
                t_sub,
                theta,
                reference_state=np.asarray(reference_state, dtype=complex),
                terms=terms,
                hmat_static=np.asarray(hmat_static, dtype=complex),
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                drive_t0=float(drive_t0),
                tangent_eps=float(cfg.tangent_eps),
                lambda_reg=float(cfg.lambda_reg),
                svd_rcond=float(cfg.svd_rcond),
                coefficient_tolerance=float(cfg.coefficient_tolerance),
                sort_terms=bool(cfg.sort_terms),
                drive_diag_cache=drive_diag_cache,
            )
            k2, _psi2, _diag2 = _rhs_theta_dot(
                t_sub + 0.5 * dt,
                theta + 0.5 * dt * k1,
                reference_state=np.asarray(reference_state, dtype=complex),
                terms=terms,
                hmat_static=np.asarray(hmat_static, dtype=complex),
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                drive_t0=float(drive_t0),
                tangent_eps=float(cfg.tangent_eps),
                lambda_reg=float(cfg.lambda_reg),
                svd_rcond=float(cfg.svd_rcond),
                coefficient_tolerance=float(cfg.coefficient_tolerance),
                sort_terms=bool(cfg.sort_terms),
                drive_diag_cache=drive_diag_cache,
            )
            k3, _psi3, _diag3 = _rhs_theta_dot(
                t_sub + 0.5 * dt,
                theta + 0.5 * dt * k2,
                reference_state=np.asarray(reference_state, dtype=complex),
                terms=terms,
                hmat_static=np.asarray(hmat_static, dtype=complex),
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                drive_t0=float(drive_t0),
                tangent_eps=float(cfg.tangent_eps),
                lambda_reg=float(cfg.lambda_reg),
                svd_rcond=float(cfg.svd_rcond),
                coefficient_tolerance=float(cfg.coefficient_tolerance),
                sort_terms=bool(cfg.sort_terms),
                drive_diag_cache=drive_diag_cache,
            )
            k4, _psi4, _diag4 = _rhs_theta_dot(
                t_sub + dt,
                theta + dt * k3,
                reference_state=np.asarray(reference_state, dtype=complex),
                terms=terms,
                hmat_static=np.asarray(hmat_static, dtype=complex),
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                drive_t0=float(drive_t0),
                tangent_eps=float(cfg.tangent_eps),
                lambda_reg=float(cfg.lambda_reg),
                svd_rcond=float(cfg.svd_rcond),
                coefficient_tolerance=float(cfg.coefficient_tolerance),
                sort_terms=bool(cfg.sort_terms),
                drive_diag_cache=drive_diag_cache,
            )
            theta = np.asarray(
                theta + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4),
                dtype=float,
            ).reshape(-1)

    return ProjectedRealTimeResult(
        times=np.asarray(times, dtype=float),
        theta_history=np.asarray(theta_rows, dtype=float),
        states=tuple(np.asarray(state, dtype=complex) for state in states),
        trajectory_rows=tuple(dict(row) for row in rows),
    )


_MATH_EXACT_REFERENCE = r"|\psi_{\mathrm{exact}}(t)\rangle=\prod_{k=0}^{N-1}\exp(-i\Delta t\,H(t_k))|\psi_0\rangle"


def run_exact_driven_reference(
    initial_state: np.ndarray,
    hmat_static: np.ndarray,
    *,
    t_final: float,
    num_times: int,
    reference_steps: int,
    drive_coeff_provider_exyz: Any | None = None,
    drive_t0: float = 0.0,
    time_sampling: str = "midpoint",
) -> ExactDrivenReferenceResult:
    if float(t_final) < 0.0:
        raise ValueError("t_final must be >= 0.")
    if int(num_times) < 2:
        raise ValueError("num_times must be >= 2.")
    if int(reference_steps) < 1:
        raise ValueError("reference_steps must be >= 1.")
    sampling = str(time_sampling).strip().lower()
    if sampling not in {"midpoint", "left", "right"}:
        raise ValueError("time_sampling must be one of {'midpoint','left','right'}.")

    psi0 = _normalize_state(initial_state)
    h_static = np.asarray(hmat_static, dtype=complex)
    dim = int(psi0.size)
    nq = int(round(math.log2(int(dim))))
    times = np.linspace(0.0, float(t_final), int(num_times))
    drive_diag_cache: dict[str, np.ndarray] = {}

    use_sparse = int(dim) >= int(_EXPM_SPARSE_MIN_DIM)
    h_static_sparse = None
    sparse_diags = None
    expm_multiply = None
    dense_expm = None
    if use_sparse:
        from scipy.sparse import csc_matrix, diags
        from scipy.sparse.linalg import expm_multiply as scipy_expm_multiply

        h_static_sparse = csc_matrix(h_static)
        sparse_diags = diags
        expm_multiply = scipy_expm_multiply
    else:
        from scipy.linalg import expm as scipy_dense_expm

        dense_expm = scipy_dense_expm

    states: list[np.ndarray] = []
    energies: list[float] = []
    rows: list[dict[str, Any]] = []

    for t_rel in times:
        if abs(float(t_rel)) <= 1e-15:
            psi = np.asarray(psi0, dtype=complex).copy()
        elif drive_coeff_provider_exyz is None:
            evals, evecs = np.linalg.eigh(h_static)
            coeffs = np.conjugate(evecs).T @ psi0
            psi = evecs @ (np.exp(-1.0j * float(t_rel) * evals) * coeffs)
        else:
            psi = np.asarray(psi0, dtype=complex).copy()
            dt = float(t_rel) / float(reference_steps)
            for step_idx in range(int(reference_steps)):
                if sampling == "midpoint":
                    t_sample = float(drive_t0) + (float(step_idx) + 0.5) * dt
                elif sampling == "left":
                    t_sample = float(drive_t0) + float(step_idx) * dt
                else:
                    t_sample = float(drive_t0) + (float(step_idx) + 1.0) * dt
                drive_map = {
                    str(label): complex(coeff)
                    for label, coeff in dict(drive_coeff_provider_exyz(float(t_sample))).items()
                    if abs(complex(coeff)) > 1e-15
                }
                if drive_map and not all(_is_all_z_type(label) for label in drive_map):
                    h_drive = np.zeros_like(h_static)
                    for label, coeff in drive_map.items():
                        h_drive += complex(coeff) * _pauli_matrix_exyz(label)
                    h_total_dense = h_static + h_drive
                    if dense_expm is None:
                        from scipy.linalg import expm as scipy_dense_expm

                        dense_expm = scipy_dense_expm
                    psi = dense_expm(-1.0j * dt * h_total_dense) @ psi
                elif use_sparse and h_static_sparse is not None and sparse_diags is not None and expm_multiply is not None:
                    diag = (
                        _build_drive_diagonal(
                            drive_map,
                            dim=int(dim),
                            nq=int(nq),
                            cache=drive_diag_cache,
                        )
                        if drive_map
                        else np.zeros(int(dim), dtype=complex)
                    )
                    h_total_sparse = h_static_sparse if not np.any(diag) else h_static_sparse + sparse_diags(diag, format="csc")
                    psi = expm_multiply((-1.0j * dt) * h_total_sparse, psi)
                else:
                    if dense_expm is None:
                        from scipy.linalg import expm as scipy_dense_expm

                        dense_expm = scipy_dense_expm
                    diag = (
                        _build_drive_diagonal(
                            drive_map,
                            dim=int(dim),
                            nq=int(nq),
                            cache=drive_diag_cache,
                        )
                        if drive_map
                        else np.zeros(int(dim), dtype=complex)
                    )
                    h_total_dense = h_static + np.diag(diag)
                    psi = dense_expm(-1.0j * dt * h_total_dense) @ psi
        psi = _normalize_state(psi)
        hpsi_total, _drive_diag = _apply_total_hamiltonian(
            psi,
            h_static,
            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
            t_physical=float(drive_t0) + float(t_rel),
            drive_diag_cache=drive_diag_cache,
        )
        energy = float(np.real(np.vdot(psi, hpsi_total)))
        states.append(np.asarray(psi, dtype=complex))
        energies.append(float(energy))
        rows.append(
            {
                "time": float(t_rel),
                "state_norm": float(np.linalg.norm(psi)),
                "energy_total_exact": float(energy),
            }
        )

    return ExactDrivenReferenceResult(
        times=np.asarray(times, dtype=float),
        states=tuple(np.asarray(state, dtype=complex) for state in states),
        energies_total=np.asarray(energies, dtype=float),
        trajectory_rows=tuple(dict(row) for row in rows),
    )


_MATH_TOTAL_ENERGY = r"E(t)=\Re\langle\psi|H(t)|\psi\rangle"


def expectation_total_hamiltonian(
    psi: np.ndarray,
    hmat_static: np.ndarray,
    *,
    drive_coeff_provider_exyz: Any | None = None,
    t_physical: float = 0.0,
) -> float:
    hpsi, _drive_diag = _apply_total_hamiltonian(
        psi,
        hmat_static,
        drive_coeff_provider_exyz=drive_coeff_provider_exyz,
        t_physical=float(t_physical),
        drive_diag_cache={},
    )
    return float(np.real(np.vdot(np.asarray(psi, dtype=complex).reshape(-1), hpsi)))


_MATH_STATE_FIDELITY = r"F(\psi,\phi)=|\langle\psi|\phi\rangle|^2"


def state_fidelity(psi_left: np.ndarray, psi_right: np.ndarray) -> float:
    left = _normalize_state(psi_left)
    right = _normalize_state(psi_right)
    return float(abs(np.vdot(left, right)) ** 2)


__all__ = [
    "ExactDrivenReferenceResult",
    "ProjectedRealTimeConfig",
    "ProjectedRealTimeResult",
    "build_tangent_vectors",
    "expectation_total_hamiltonian",
    "run_exact_driven_reference",
    "run_projected_real_time_trajectory",
    "solve_mclachlan_step",
    "state_fidelity",
]
