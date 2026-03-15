#!/usr/bin/env python3
"""Scoring and reduced-path derivative accounting for HH continuation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.hh_continuation_types import (
    CandidateFeatures,
    CompileCostEstimate,
    CurvatureOracle,
    MeasurementCacheStats,
    MeasurementPlan,
    NoveltyOracle,
)
from pipelines.hardcoded.hh_continuation_motifs import motif_bonus_for_generator
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.pauli_actions import apply_compiled_pauli


@dataclass(frozen=True)
class SimpleScoreConfig:
    lambda_F: float = 1.0
    lambda_compile: float = 0.05
    lambda_measure: float = 0.02
    lambda_leak: float = 0.0
    z_alpha: float = 0.0
    wD: float = 0.0
    wG: float = 0.0
    wC: float = 0.0
    wc: float = 0.0
    depth_ref: float = 1.0
    group_ref: float = 1.0
    shot_ref: float = 1.0
    family_ref: float = 1.0
    lifetime_cost_mode: str = "off"
    score_version: str = "append_screen_v1"


@dataclass(frozen=True)
class FullScoreConfig:
    z_alpha: float = 0.0
    lambda_F: float = 1.0
    lambda_H: float = 1e-6
    rho: float = 0.25
    eta_L: float = 0.0
    gamma_N: float = 1.0
    wD: float = 0.2
    wG: float = 0.15
    wC: float = 0.15
    wP: float = 0.1
    wc: float = 0.1
    depth_ref: float = 1.0
    group_ref: float = 1.0
    shot_ref: float = 1.0
    optdim_ref: float = 1.0
    reuse_ref: float = 1.0
    family_ref: float = 1.0
    novelty_eps: float = 1e-6
    shortlist_fraction: float = 0.2
    shortlist_size: int = 12
    batch_target_size: int = 2
    batch_size_cap: int = 3
    batch_near_degenerate_ratio: float = 0.9
    compat_overlap_weight: float = 0.4
    compat_comm_weight: float = 0.2
    compat_curv_weight: float = 0.2
    compat_sched_weight: float = 0.2
    leakage_cap: float = 1e6
    lifetime_cost_mode: str = "off"
    remaining_evaluations_proxy_mode: str = "none"
    lifetime_weight: float = 0.05
    motif_bonus_weight: float = 0.0
    metric_floor: float = 1e-12
    reduced_metric_collapse_rel_tol: float = 1e-8
    ridge_growth_factor: float = 10.0
    ridge_max_steps: int = 12
    score_version: str = "append_full_v1"


@dataclass(frozen=True)
class _ScaffoldDerivativeContext:
    psi_state: np.ndarray
    hpsi_state: np.ndarray
    refit_window_indices: tuple[int, ...]
    dpsi_window: tuple[np.ndarray, ...]
    tangents_window: tuple[np.ndarray, ...]
    Q_window: np.ndarray
    H_window_hessian: np.ndarray


class Phase1CompileCostOracle:
    """Built-in math expression:
    D_proxy = n_new_pauli + n_rot + shift_span + active_count
    """

    def estimate(
        self,
        *,
        candidate_term_count: int,
        position_id: int,
        append_position: int,
        refit_active_count: int,
    ) -> CompileCostEstimate:
        new_pauli_actions = float(max(1, int(candidate_term_count)))
        new_rotation_steps = float(max(1, int(candidate_term_count)))
        position_shift_span = float(abs(int(append_position) - int(position_id)))
        refit_active = float(max(0, int(refit_active_count)))
        total = float(new_pauli_actions + new_rotation_steps + position_shift_span + refit_active)
        return CompileCostEstimate(
            new_pauli_actions=new_pauli_actions,
            new_rotation_steps=new_rotation_steps,
            position_shift_span=position_shift_span,
            refit_active_count=refit_active,
            proxy_total=total,
        )


class MeasurementCacheAudit:
    """Phase 1 accounting-only grouped reuse tracker."""

    def __init__(
        self,
        nominal_shots_per_group: int = 1,
        *,
        plan_version: str = "phase1_grouped_label_reuse",
        grouping_mode: str = "grouped_label_reuse",
    ) -> None:
        self._seen_groups: set[str] = set()
        self._nominal_shots = int(max(1, nominal_shots_per_group))
        self._plan_version = str(plan_version)
        self._grouping_mode = str(grouping_mode)

    def clone(self) -> "MeasurementCacheAudit":
        cloned = MeasurementCacheAudit(
            nominal_shots_per_group=int(self._nominal_shots),
            plan_version=str(self._plan_version),
            grouping_mode=str(self._grouping_mode),
        )
        cloned._seen_groups = set(self._seen_groups)
        return cloned

    def snapshot(self) -> dict[str, Any]:
        return {
            "seen_groups": sorted(str(x) for x in self._seen_groups),
            "nominal_shots_per_group": int(self._nominal_shots),
            "plan_version": str(self._plan_version),
            "grouping_mode": str(self._grouping_mode),
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "MeasurementCacheAudit":
        cloned = cls(
            nominal_shots_per_group=int(snapshot.get("nominal_shots_per_group", 1)),
            plan_version=str(snapshot.get("plan_version", "phase1_grouped_label_reuse")),
            grouping_mode=str(snapshot.get("grouping_mode", "grouped_label_reuse")),
        )
        cloned._seen_groups = {
            str(x)
            for x in snapshot.get("seen_groups", [])
            if str(x) != ""
        }
        return cloned

    def plan_for(self, group_keys: Iterable[str]) -> MeasurementPlan:
        keys = [str(k) for k in group_keys if str(k) != ""]
        unique_keys: list[str] = []
        for key in keys:
            if key not in unique_keys:
                unique_keys.append(key)
        return MeasurementPlan(
            plan_version=str(self._plan_version),
            group_keys=list(unique_keys),
            nominal_shots_per_group=int(self._nominal_shots),
            grouping_mode=str(self._grouping_mode),
        )

    def estimate(self, group_keys: Iterable[str]) -> MeasurementCacheStats:
        plan = self.plan_for(group_keys)
        unique_keys = list(plan.group_keys)

        groups_total = int(len(unique_keys))
        groups_reused = 0
        for key in unique_keys:
            if key in self._seen_groups:
                groups_reused += 1
        groups_new = int(groups_total - groups_reused)
        shots_reused = float(groups_reused * self._nominal_shots)
        shots_new = float(groups_new * self._nominal_shots)
        reuse_count_cost = float(groups_new)
        return MeasurementCacheStats(
            groups_total=groups_total,
            groups_reused=int(groups_reused),
            groups_new=int(groups_new),
            shots_reused=shots_reused,
            shots_new=shots_new,
            reuse_count_cost=reuse_count_cost,
        )

    def commit(self, group_keys: Iterable[str]) -> None:
        for key in group_keys:
            key_s = str(key)
            if key_s != "":
                self._seen_groups.add(key_s)

    def summary(self) -> dict[str, float]:
        return {
            "groups_known": float(len(self._seen_groups)),
            "nominal_shots_per_group": float(self._nominal_shots),
            "plan_version": str(self._plan_version),
            "grouping_mode": str(self._grouping_mode),
        }


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _replace_feature(feat: CandidateFeatures, **updates: Any) -> CandidateFeatures:
    return CandidateFeatures(**{**feat.__dict__, **updates})


def normalize(value: float, ref: float) -> float:
    denom = float(ref)
    if not math.isfinite(denom) or denom <= 0.0:
        return float(max(0.0, value))
    return float(max(0.0, value) / denom)


def trust_region_drop(g_lcb: float, h_eff: float, F: float, rho: float) -> float:
    if float(g_lcb) <= 0.0:
        return 0.0
    F_pos = float(max(float(F), 1e-12))
    h_eff_pos = float(max(0.0, h_eff))
    alpha_max = float(rho) / float(math.sqrt(F_pos))
    if h_eff_pos > 0.0:
        alpha_newton = float(g_lcb) / h_eff_pos
        if alpha_newton <= alpha_max:
            return float(0.5 * float(g_lcb) * float(g_lcb) / h_eff_pos)
    return float(float(g_lcb) * alpha_max - 0.5 * h_eff_pos * alpha_max * alpha_max)


def remaining_evaluations_proxy(
    *,
    current_depth: int | None,
    max_depth: int | None,
    mode: str,
) -> float:
    mode_key = str(mode).strip().lower()
    if mode_key == "none":
        return 0.0
    depth_now = 0 if current_depth is None else int(max(0, current_depth))
    depth_cap = depth_now if max_depth is None else int(max(depth_now, max_depth))
    if mode_key == "remaining_depth":
        return float(max(1, depth_cap - depth_now + 1))
    raise ValueError("remaining_evaluations_proxy_mode must be 'none' or 'remaining_depth'")


def family_repeat_cost_from_history(
    *,
    history_rows: Sequence[Mapping[str, Any]],
    candidate_family: str,
) -> float:
    fam = str(candidate_family).strip()
    if fam == "":
        return 0.0
    tail = [row for row in history_rows if isinstance(row, Mapping) and row.get("candidate_family") is not None]
    if not tail:
        return 0.0
    if str(tail[-1].get("candidate_family", "")).strip() != fam:
        return 0.0
    streak = 0
    for row in reversed(tail):
        if str(row.get("candidate_family", "")).strip() != fam:
            break
        streak += 1
    return float(streak)


def lifetime_weight_components(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> dict[str, float]:
    rem = float(max(0.0, feat.remaining_evaluations_proxy))
    if str(cfg.lifetime_cost_mode).strip().lower() == "off":
        return {
            "remaining_evaluations_proxy": float(rem),
            "depth_life": 0.0,
            "total": 0.0,
        }
    depth_life = float(max(1.0, rem) * normalize(float(feat.depth_cost), float(cfg.depth_ref)))
    return {
        "remaining_evaluations_proxy": float(rem),
        "depth_life": float(depth_life),
        "total": float(depth_life),
    }


def _depth_life_cost(feat: CandidateFeatures, cfg: FullScoreConfig) -> float:
    base = normalize(float(feat.depth_cost), float(cfg.depth_ref))
    if str(cfg.lifetime_cost_mode).strip().lower() == "off":
        return float(base)
    rem = float(max(1.0, feat.remaining_evaluations_proxy))
    return float(rem * base)


def _score_denominator(feat: CandidateFeatures, cfg: FullScoreConfig) -> float:
    return float(
        1.0
        + float(cfg.wD) * float(_depth_life_cost(feat, cfg))
        + float(cfg.wG) * normalize(float(feat.new_group_cost), float(cfg.group_ref))
        + float(cfg.wC) * normalize(float(feat.new_shot_cost), float(cfg.shot_ref))
        + float(cfg.wc) * normalize(float(feat.family_repeat_cost), float(cfg.family_ref))
    )


# ---------------------------------------------------------------------------
# Active append-only scoring surface
# ---------------------------------------------------------------------------

def _screen_denominator(feat: CandidateFeatures, cfg: SimpleScoreConfig) -> float:
    wD = float(cfg.wD if cfg.wD != 0.0 else cfg.lambda_compile)
    wG = float(cfg.wG if cfg.wG != 0.0 else cfg.lambda_measure)
    wC = float(cfg.wC if cfg.wC != 0.0 else cfg.lambda_measure)
    wc = float(cfg.wc if cfg.wc != 0.0 else cfg.lambda_measure)
    depth_life = normalize(float(feat.depth_cost), float(cfg.depth_ref))
    if str(cfg.lifetime_cost_mode).strip().lower() != "off":
        depth_life *= float(max(1.0, feat.remaining_evaluations_proxy))
    return float(
        1.0
        + wD * depth_life
        + wG * normalize(float(feat.new_group_cost), float(cfg.group_ref))
        + wC * normalize(float(feat.new_shot_cost), float(cfg.shot_ref))
        + wc * normalize(float(feat.family_repeat_cost), float(cfg.family_ref))
    )


def simple_v1_score(
    feat: CandidateFeatures,
    cfg: SimpleScoreConfig,
) -> float:
    if not bool(feat.stage_gate_open):
        return float("-inf")
    g_lcb = float(max(0.0, feat.g_lcb))
    F_raw = float(max(float(feat.F_raw if feat.F_raw is not None else feat.F_metric), 1e-12))
    if g_lcb <= 0.0:
        return 0.0
    lamF = float(max(float(cfg.lambda_F), 1e-12))
    delta_e_screen = 0.5 * g_lcb * g_lcb / (lamF * F_raw)
    return float(delta_e_screen / float(_screen_denominator(feat, cfg)))


def full_v2_score(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> tuple[float, str]:
    if not bool(feat.stage_gate_open):
        return float("-inf"), "blocked_stage"
    g_lcb = float(max(0.0, feat.g_lcb))
    if g_lcb <= 0.0:
        return 0.0, "nonpositive_gradient"

    F_red_raw = feat.F_red if feat.F_red is not None else feat.F_raw
    if F_red_raw is None:
        return 0.0, "missing_reduced_path_metric"
    h_eff = float(feat.h_eff if feat.h_eff is not None else (feat.h_raw if feat.h_raw is not None else 0.0))
    F_red = float(max(float(F_red_raw), float(cfg.metric_floor)))
    novelty = 1.0 if feat.novelty is None else float(min(1.0, max(0.0, feat.novelty)))
    if str(feat.curvature_mode).startswith("append_exact_metric_collapse") or novelty <= 0.0:
        return 0.0, "reduced_metric_collapse"
    delta_e = trust_region_drop(g_lcb, float(max(0.0, h_eff)), F_red, float(cfg.rho))
    if delta_e <= 0.0:
        return 0.0, "nonpositive_trust_region_drop"
    score = (float(novelty) ** float(cfg.gamma_N)) * float(delta_e) / float(_score_denominator(feat, cfg))
    if feat.ridge_used is not None and float(feat.ridge_used) > float(max(cfg.lambda_H, 0.0)):
        return float(score), "append_exact_reduced_path_ridge_grown"
    if len(feat.refit_window_indices) == 0:
        return float(score), "append_exact_empty_window"
    return float(score), "append_exact_reduced_path"


# ---------------------------------------------------------------------------
# Shortlist ranking helpers
# ---------------------------------------------------------------------------

def shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    score_key: str = "simple_score",
) -> list[dict[str, Any]]:
    ranked = sorted(
        [dict(rec) for rec in records],
        key=lambda rec: (
            -float(rec.get(score_key, float("-inf"))),
            -float(rec.get("simple_score", float("-inf"))),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return []
    total = int(len(ranked))
    target = int(max(1, min(total, cfg.shortlist_size, math.ceil(float(cfg.shortlist_fraction) * total))))
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(ranked[:target], start=1):
        updated = dict(rec)
        feat = updated.get("feature", None)
        if isinstance(feat, CandidateFeatures):
            updated["feature"] = _replace_feature(
                feat,
                shortlist_rank=int(idx),
                shortlist_size=int(target),
            )
        out.append(updated)
    return out


# ---------------------------------------------------------------------------
# Exact append-only reduced-path derivative helpers
# ---------------------------------------------------------------------------

def _executor_for_terms(
    terms: Sequence[Any],
    *,
    pauli_action_cache: dict[str, Any] | None,
) -> CompiledAnsatzExecutor:
    return CompiledAnsatzExecutor(
        list(terms),
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        pauli_action_cache=pauli_action_cache,
    )


def _rotation_triplet(vec: np.ndarray, step: Any, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vec_arr = np.asarray(vec, dtype=complex).reshape(-1)
    coeff = float(step.coeff_real)
    pvec = apply_compiled_pauli(vec_arr, step.action)
    phi = float(theta) * coeff
    c = math.cos(phi)
    s = math.sin(phi)
    u_vec = c * vec_arr - 1j * s * pvec
    d_vec = -coeff * s * vec_arr - 1j * coeff * c * pvec
    s_vec = -(coeff * coeff) * u_vec
    return np.asarray(u_vec, dtype=complex), np.asarray(d_vec, dtype=complex), np.asarray(s_vec, dtype=complex)


def _horizontal_tangent(psi_state: np.ndarray, dpsi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    dpsi_vec = np.asarray(dpsi, dtype=complex).reshape(-1)
    overlap = complex(np.vdot(psi, dpsi_vec))
    return np.asarray(dpsi_vec - overlap * psi, dtype=complex)


def _tangent_overlap_matrix(tangents: Sequence[np.ndarray]) -> np.ndarray:
    n = int(len(tangents))
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = float(np.real(np.vdot(tangents[i], tangents[j])))
            out[i, j] = val
            out[j, i] = val
    return out


def _energy_hessian_entry(
    *,
    dpsi_left: np.ndarray,
    dpsi_right: np.ndarray,
    d2psi: np.ndarray,
    hpsi_state: np.ndarray,
    hdpsi_right: np.ndarray,
) -> float:
    return float(
        2.0
        * np.real(
            np.vdot(np.asarray(d2psi, dtype=complex), np.asarray(hpsi_state, dtype=complex))
            + np.vdot(np.asarray(dpsi_left, dtype=complex), np.asarray(hdpsi_right, dtype=complex))
        )
    )


def _propagate_executor_derivatives(
    *,
    executor: CompiledAnsatzExecutor,
    theta: np.ndarray,
    psi_ref: np.ndarray,
    active_indices: Sequence[int],
) -> tuple[np.ndarray, list[np.ndarray], list[list[np.ndarray]]]:
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    active = [int(i) for i in active_indices]
    psi = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
    n_active = int(len(active))
    dpsi = [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
    d2psi = [[np.zeros_like(psi, dtype=complex) for _ in range(n_active)] for __ in range(n_active)]
    if n_active == 0:
        return executor.prepare_state(theta_vec, psi), dpsi, d2psi

    active_map = {int(global_idx): int(local_idx) for local_idx, global_idx in enumerate(active)}
    plans = list(getattr(executor, "_plans", []))
    if len(plans) != int(theta_vec.size):
        raise ValueError(f"theta length mismatch: got {theta_vec.size}, expected {len(plans)}.")

    for global_idx, plan in enumerate(plans):
        theta_k = float(theta_vec[global_idx])
        local = active_map.get(int(global_idx), None)
        for step in getattr(plan, "steps", ()):  # pragma: no branch - tuple in normal path
            old_psi = psi
            old_dpsi = dpsi
            old_d2psi = d2psi

            psi_u, psi_d, psi_s = _rotation_triplet(old_psi, step, theta_k)
            psi = psi_u

            next_dpsi: list[np.ndarray] = []
            d_old: list[np.ndarray] = []
            for idx in range(n_active):
                vec_u, vec_d, _vec_s = _rotation_triplet(old_dpsi[idx], step, theta_k)
                next_dpsi.append(vec_u)
                d_old.append(vec_d)
            if local is not None:
                next_dpsi[int(local)] = np.asarray(next_dpsi[int(local)] + psi_d, dtype=complex)

            next_d2psi: list[list[np.ndarray]] = [
                [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
                for __ in range(n_active)
            ]
            for row in range(n_active):
                for col in range(n_active):
                    vec_u, _vec_d, _vec_s = _rotation_triplet(old_d2psi[row][col], step, theta_k)
                    updated = vec_u
                    if local is not None:
                        if row == int(local):
                            updated = np.asarray(updated + d_old[col], dtype=complex)
                        if col == int(local):
                            updated = np.asarray(updated + d_old[row], dtype=complex)
                        if row == int(local) and col == int(local):
                            updated = np.asarray(updated + psi_s, dtype=complex)
                    next_d2psi[row][col] = np.asarray(updated, dtype=complex)

            dpsi = next_dpsi
            d2psi = next_d2psi

    return np.asarray(psi, dtype=complex), dpsi, d2psi


def _propagate_append_candidate(
    *,
    candidate_term: Any,
    psi_state: np.ndarray,
    window_dpsi: Sequence[np.ndarray],
    pauli_action_cache: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    cand_exec = _executor_for_terms([candidate_term], pauli_action_cache=pauli_action_cache)
    plan = list(getattr(cand_exec, "_plans", []))
    if not plan:
        zero = np.zeros_like(np.asarray(psi_state, dtype=complex).reshape(-1), dtype=complex)
        return zero, zero, [np.zeros_like(zero) for _ in window_dpsi]
    steps = list(getattr(plan[0], "steps", ()))
    psi = np.asarray(psi_state, dtype=complex).reshape(-1).copy()
    cand_dpsi = np.zeros_like(psi, dtype=complex)
    cand_d2psi = np.zeros_like(psi, dtype=complex)
    win_dpsi = [np.asarray(vec, dtype=complex).reshape(-1).copy() for vec in window_dpsi]
    cand_win_d2 = [np.zeros_like(psi, dtype=complex) for _ in window_dpsi]

    for step in steps:
        old_psi = psi
        old_cand_dpsi = cand_dpsi
        old_cand_d2psi = cand_d2psi
        old_win_dpsi = win_dpsi
        old_cand_win_d2 = cand_win_d2

        psi_u, psi_d, psi_s = _rotation_triplet(old_psi, step, 0.0)
        cand_u, cand_d, _cand_s = _rotation_triplet(old_cand_dpsi, step, 0.0)
        cand2_u, _cand2_d, _cand2_s = _rotation_triplet(old_cand_d2psi, step, 0.0)

        psi = psi_u
        cand_dpsi = np.asarray(cand_u + psi_d, dtype=complex)
        cand_d2psi = np.asarray(cand2_u + cand_d + cand_d + psi_s, dtype=complex)

        next_win_dpsi: list[np.ndarray] = []
        next_cand_win_d2: list[np.ndarray] = []
        for idx, win_vec in enumerate(old_win_dpsi):
            win_u, win_d, _win_s = _rotation_triplet(win_vec, step, 0.0)
            cross_u, _cross_d, _cross_s = _rotation_triplet(old_cand_win_d2[idx], step, 0.0)
            next_win_dpsi.append(np.asarray(win_u, dtype=complex))
            next_cand_win_d2.append(np.asarray(cross_u + win_d, dtype=complex))
        win_dpsi = next_win_dpsi
        cand_win_d2 = next_cand_win_d2

    return cand_dpsi, cand_d2psi, cand_win_d2


def _regularized_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    base_ridge: float,
    growth_factor: float,
    max_steps: int,
    require_pd: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    mat = np.asarray(matrix, dtype=float)
    vec = np.asarray(rhs, dtype=float).reshape(-1)
    n = int(mat.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float), float(max(base_ridge, 0.0)), np.zeros((0, 0), dtype=float)
    eye = np.eye(n, dtype=float)
    ridge = float(max(base_ridge, 0.0))
    if ridge == 0.0:
        ridge = 1e-12
    mat_sym = 0.5 * (mat + mat.T)
    for _ in range(int(max(1, max_steps))):
        trial = mat_sym + ridge * eye
        try:
            if require_pd:
                np.linalg.cholesky(trial)
            sol = np.linalg.solve(trial, vec)
            return np.asarray(sol, dtype=float), float(ridge), np.asarray(trial, dtype=float)
        except Exception:
            ridge *= float(max(growth_factor, 2.0))
    trial = mat_sym + ridge * eye
    if require_pd:
        np.linalg.cholesky(trial)
    sol = np.linalg.solve(trial, vec)
    return np.asarray(sol, dtype=float), float(ridge), np.asarray(trial, dtype=float)


class Phase2NoveltyOracle:
    """Exact ordered-state tangent context for append-only reduced-path scoring."""

    def prepare_scaffold_context(
        self,
        *,
        selected_ops: Sequence[Any],
        theta: np.ndarray,
        psi_ref: np.ndarray,
        psi_state: np.ndarray,
        h_compiled: CompiledPolynomialAction,
        hpsi_state: np.ndarray,
        refit_window_indices: Sequence[int],
        pauli_action_cache: dict[str, Any] | None = None,
    ) -> _ScaffoldDerivativeContext:
        inherited_window = [int(i) for i in refit_window_indices]
        psi_current = np.asarray(psi_state, dtype=complex).reshape(-1)
        hpsi_current = np.asarray(hpsi_state, dtype=complex).reshape(-1)
        if not inherited_window:
            return _ScaffoldDerivativeContext(
                psi_state=psi_current,
                hpsi_state=hpsi_current,
                refit_window_indices=tuple(),
                dpsi_window=tuple(),
                tangents_window=tuple(),
                Q_window=np.zeros((0, 0), dtype=float),
                H_window_hessian=np.zeros((0, 0), dtype=float),
            )

        executor = _executor_for_terms(selected_ops, pauli_action_cache=pauli_action_cache)
        _psi_final, dpsi_window, d2psi_window = _propagate_executor_derivatives(
            executor=executor,
            theta=np.asarray(theta, dtype=float),
            psi_ref=np.asarray(psi_ref, dtype=complex),
            active_indices=inherited_window,
        )
        tangents_window = [
            _horizontal_tangent(psi_current, dpsi_vec)
            for dpsi_vec in dpsi_window
        ]
        q_window = _tangent_overlap_matrix(tangents_window)
        hdpsi_window = [
            apply_compiled_polynomial(np.asarray(dpsi_vec, dtype=complex), h_compiled)
            for dpsi_vec in dpsi_window
        ]
        m = int(len(inherited_window))
        hess = np.zeros((m, m), dtype=float)
        for row in range(m):
            for col in range(m):
                hess[row, col] = _energy_hessian_entry(
                    dpsi_left=dpsi_window[row],
                    dpsi_right=dpsi_window[col],
                    d2psi=d2psi_window[row][col],
                    hpsi_state=hpsi_current,
                    hdpsi_right=hdpsi_window[col],
                )
        hess = 0.5 * (hess + hess.T)
        return _ScaffoldDerivativeContext(
            psi_state=psi_current,
            hpsi_state=hpsi_current,
            refit_window_indices=tuple(inherited_window),
            dpsi_window=tuple(np.asarray(x, dtype=complex) for x in dpsi_window),
            tangents_window=tuple(np.asarray(x, dtype=complex) for x in tangents_window),
            Q_window=np.asarray(q_window, dtype=float),
            H_window_hessian=np.asarray(hess, dtype=float),
        )

    def estimate(
        self,
        *,
        scaffold_context: _ScaffoldDerivativeContext,
        candidate_label: str,
        candidate_term: Any,
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
        novelty_eps: float = 1e-6,
    ) -> Mapping[str, Any]:
        del compiled_cache, novelty_eps  # reduced-path novelty is completed after curvature correction.
        cand_dpsi, cand_d2psi, cand_window_d2 = _propagate_append_candidate(
            candidate_term=candidate_term,
            psi_state=scaffold_context.psi_state,
            window_dpsi=list(scaffold_context.dpsi_window),
            pauli_action_cache=pauli_action_cache,
        )
        cand_tangent = _horizontal_tangent(scaffold_context.psi_state, cand_dpsi)
        q_window = np.asarray(
            [
                float(np.real(np.vdot(tang_j, cand_tangent)))
                for tang_j in scaffold_context.tangents_window
            ],
            dtype=float,
        )
        F_raw = float(max(0.0, np.real(np.vdot(cand_tangent, cand_tangent))))
        return {
            "novelty_mode": "append_exact_tangent_context_v1",
            "candidate_dpsi": np.asarray(cand_dpsi, dtype=complex),
            "candidate_d2psi": np.asarray(cand_d2psi, dtype=complex),
            "candidate_window_d2": [np.asarray(x, dtype=complex) for x in cand_window_d2],
            "candidate_tangent": np.asarray(cand_tangent, dtype=complex),
            "F_raw": float(F_raw),
            "Q_window": np.asarray(scaffold_context.Q_window, dtype=float),
            "q_window": np.asarray(q_window, dtype=float),
        }


class Phase2CurvatureOracle:
    """Exact analytic Hessian blocks for the append-only reduced path."""

    def estimate(
        self,
        *,
        base_feature: CandidateFeatures,
        novelty_info: Mapping[str, Any],
        scaffold_context: _ScaffoldDerivativeContext,
        h_compiled: CompiledPolynomialAction,
        cfg: FullScoreConfig,
        optimizer_memory: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        del optimizer_memory
        F_raw = float(max(0.0, novelty_info.get("F_raw", base_feature.F_raw or base_feature.F_metric)))
        q_window = np.asarray(novelty_info.get("q_window", []), dtype=float).reshape(-1)
        Q_window = np.asarray(novelty_info.get("Q_window", scaffold_context.Q_window), dtype=float)
        cand_dpsi = np.asarray(novelty_info.get("candidate_dpsi"), dtype=complex).reshape(-1)
        cand_d2psi = np.asarray(novelty_info.get("candidate_d2psi"), dtype=complex).reshape(-1)
        cand_window_d2 = [
            np.asarray(x, dtype=complex).reshape(-1)
            for x in novelty_info.get("candidate_window_d2", [])
        ]
        hdpsi_candidate = apply_compiled_polynomial(cand_dpsi, h_compiled)
        h_raw = _energy_hessian_entry(
            dpsi_left=cand_dpsi,
            dpsi_right=cand_dpsi,
            d2psi=cand_d2psi,
            hpsi_state=scaffold_context.hpsi_state,
            hdpsi_right=hdpsi_candidate,
        )

        b_mixed = np.zeros(len(scaffold_context.refit_window_indices), dtype=float)
        for idx, dpsi_window in enumerate(scaffold_context.dpsi_window):
            if idx >= len(cand_window_d2):
                break
            b_mixed[idx] = _energy_hessian_entry(
                dpsi_left=dpsi_window,
                dpsi_right=cand_dpsi,
                d2psi=cand_window_d2[idx],
                hpsi_state=scaffold_context.hpsi_state,
                hdpsi_right=hdpsi_candidate,
            )

        H_window = np.asarray(scaffold_context.H_window_hessian, dtype=float)
        if H_window.size == 0:
            h_eff = float(h_raw)
            F_red = float(max(F_raw, float(cfg.metric_floor)))
            q_reduced = np.zeros(0, dtype=float)
            novelty = 1.0
            ridge_used = float(max(cfg.lambda_H, 0.0))
            M_window = np.zeros((0, 0), dtype=float)
            mode = "append_exact_empty_window"
        else:
            minv_b, ridge_used, M_window = _regularized_solve(
                H_window,
                b_mixed,
                base_ridge=float(max(cfg.lambda_H, 0.0)),
                growth_factor=float(max(cfg.ridge_growth_factor, 2.0)),
                max_steps=int(max(1, cfg.ridge_max_steps)),
                require_pd=True,
            )
            h_eff = float(h_raw - float(b_mixed.T @ minv_b))
            F_red_exact = float(
                F_raw
                - 2.0 * float(q_window.T @ minv_b)
                + float(minv_b.T @ Q_window @ minv_b)
            )
            F_red = float(max(F_red_exact, float(cfg.metric_floor)))
            q_reduced = np.asarray(q_window - Q_window @ minv_b, dtype=float)
            collapse_floor = max(
                float(cfg.metric_floor),
                float(cfg.reduced_metric_collapse_rel_tol) * float(max(F_raw, float(cfg.metric_floor))),
            )
            metric_collapse = bool(F_red_exact <= collapse_floor)
            if metric_collapse:
                novelty = 0.0
                mode = "append_exact_metric_collapse_v1"
            else:
                qsol, _nov_ridge, _Qreg = _regularized_solve(
                    Q_window,
                    q_reduced,
                    base_ridge=float(max(cfg.novelty_eps, 0.0)),
                    growth_factor=float(max(cfg.ridge_growth_factor, 2.0)),
                    max_steps=int(max(1, cfg.ridge_max_steps)),
                    require_pd=True,
                )
                novelty_raw = 1.0 - float(q_reduced.T @ qsol) / float(F_red)
                novelty = float(min(1.0, max(0.0, novelty_raw)))
                mode = (
                    "append_exact_window_hessian_ridge_grown_v1"
                    if float(ridge_used) > float(max(cfg.lambda_H, 0.0))
                    else "append_exact_window_hessian_v1"
                )

        return {
            "h_raw": float(h_raw),
            "b_mixed": [float(x) for x in b_mixed.tolist()],
            "H_window_hessian": [[float(x) for x in row] for row in H_window.tolist()],
            "M_window": [[float(x) for x in row] for row in np.asarray(M_window, dtype=float).tolist()],
            "h_eff": float(h_eff),
            "F_red": float(F_red),
            "q_reduced": [float(x) for x in q_reduced.tolist()],
            "novelty": float(novelty),
            "ridge_used": float(ridge_used),
            "curvature_mode": str(mode),
        }


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_full_candidate_features(
    *,
    base_feature: CandidateFeatures,
    candidate_term: Any,
    cfg: FullScoreConfig,
    novelty_oracle: NoveltyOracle,
    curvature_oracle: CurvatureOracle,
    scaffold_context: _ScaffoldDerivativeContext,
    h_compiled: CompiledPolynomialAction,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    optimizer_memory: Mapping[str, Any] | None = None,
    motif_library: Mapping[str, Any] | None = None,
    target_num_sites: int | None = None,
) -> CandidateFeatures:
    novelty_info = novelty_oracle.estimate(
        scaffold_context=scaffold_context,
        candidate_label=str(base_feature.candidate_label),
        candidate_term=candidate_term,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
        novelty_eps=float(cfg.novelty_eps),
    )
    curvature_info = curvature_oracle.estimate(
        base_feature=base_feature,
        novelty_info=novelty_info,
        scaffold_context=scaffold_context,
        h_compiled=h_compiled,
        cfg=cfg,
        optimizer_memory=optimizer_memory,
    )
    feat = _replace_feature(
        base_feature,
        novelty=float(curvature_info.get("novelty", 1.0)),
        novelty_mode=str(novelty_info.get("novelty_mode", "append_exact_tangent_context_v1")),
        curvature_mode=str(curvature_info.get("curvature_mode", "append_exact_window_hessian_v1")),
        F_metric=float(max(0.0, novelty_info.get("F_raw", base_feature.F_metric))),
        metric_proxy=float(max(0.0, novelty_info.get("F_raw", base_feature.metric_proxy))),
        F_raw=float(max(0.0, novelty_info.get("F_raw", base_feature.F_raw or base_feature.F_metric))),
        Q_window=[[float(x) for x in row] for row in np.asarray(novelty_info.get("Q_window", np.zeros((0, 0), dtype=float)), dtype=float).tolist()],
        q_window=[float(x) for x in np.asarray(novelty_info.get("q_window", []), dtype=float).tolist()],
        h_raw=float(curvature_info.get("h_raw", 0.0)),
        b_mixed=[float(x) for x in curvature_info.get("b_mixed", [])],
        H_window_hessian=[[float(x) for x in row] for row in curvature_info.get("H_window_hessian", [])],
        M_window=[[float(x) for x in row] for row in curvature_info.get("M_window", [])],
        h_eff=float(curvature_info.get("h_eff", 0.0)),
        F_red=float(curvature_info.get("F_red", novelty_info.get("F_raw", 0.0))),
        q_reduced=[float(x) for x in curvature_info.get("q_reduced", [])],
        ridge_used=float(curvature_info.get("ridge_used", max(cfg.lambda_H, 0.0))),
        h_hat=float(curvature_info.get("h_raw", 0.0)),
        b_hat=[float(x) for x in curvature_info.get("b_mixed", [])],
        H_window=[[float(x) for x in row] for row in curvature_info.get("H_window_hessian", [])],
        score_version=str(cfg.score_version),
        placeholder_hooks={
            **dict(base_feature.placeholder_hooks),
            "novelty_oracle": True,
            "curvature_oracle": True,
            "full_v2_score": True,
        },
    )
    if isinstance(base_feature.generator_metadata, Mapping) and isinstance(motif_library, Mapping):
        motif_bonus, motif_meta = motif_bonus_for_generator(
            generator_metadata=base_feature.generator_metadata,
            motif_library=motif_library,
            target_num_sites=int(max(0, target_num_sites or 0)),
        )
        feat = _replace_feature(
            feat,
            motif_bonus=float(motif_bonus),
            motif_source=(
                str(motif_library.get("source_tag", "payload"))
                if bool(motif_bonus) else str(feat.motif_source)
            ),
            motif_metadata=(dict(motif_meta) if isinstance(motif_meta, Mapping) else feat.motif_metadata),
        )
    feat = _replace_feature(
        feat,
        lifetime_weight_components=dict(lifetime_weight_components(feat, cfg)),
        lifetime_cost_mode=str(cfg.lifetime_cost_mode),
        remaining_evaluations_proxy_mode=str(cfg.remaining_evaluations_proxy_mode),
    )
    score, fallback_mode = full_v2_score(feat, cfg)
    return _replace_feature(
        feat,
        full_v2_score=float(score),
        actual_fallback_mode=str(fallback_mode),
    )


def build_candidate_features(
    *,
    stage_name: str,
    candidate_label: str,
    candidate_family: str,
    candidate_pool_index: int,
    position_id: int,
    append_position: int,
    positions_considered: list[int],
    gradient_signed: float,
    metric_proxy: float,
    sigma_hat: float,
    refit_window_indices: list[int],
    compile_cost: CompileCostEstimate,
    measurement_stats: MeasurementCacheStats,
    leakage_penalty: float,
    stage_gate_open: bool,
    leakage_gate_open: bool,
    trough_probe_triggered: bool,
    trough_detected: bool,
    cfg: SimpleScoreConfig,
    generator_metadata: Mapping[str, Any] | None = None,
    symmetry_spec: Mapping[str, Any] | None = None,
    symmetry_mode: str = "none",
    symmetry_mitigation_mode: str = "off",
    motif_metadata: Mapping[str, Any] | None = None,
    motif_bonus: float = 0.0,
    motif_source: str = "none",
    current_depth: int | None = None,
    max_depth: int | None = None,
    lifetime_cost_mode: str = "off",
    remaining_evaluations_proxy_mode: str = "none",
    family_repeat_cost: float = 0.0,
) -> CandidateFeatures:
    """Built-in math expression:
    g_lcb = max(|g| - z_alpha * sigma_hat, 0)
    """
    g_abs = float(abs(float(gradient_signed)))
    g_lcb = max(g_abs - float(cfg.z_alpha) * float(max(0.0, sigma_hat)), 0.0)
    raw_metric = float(max(0.0, metric_proxy))
    remaining_eval_proxy = remaining_evaluations_proxy(
        current_depth=current_depth,
        max_depth=max_depth,
        mode=str(remaining_evaluations_proxy_mode),
    )
    feat = CandidateFeatures(
        stage_name=str(stage_name),
        candidate_label=str(candidate_label),
        candidate_family=str(candidate_family),
        candidate_pool_index=int(candidate_pool_index),
        position_id=int(position_id),
        append_position=int(append_position),
        positions_considered=[int(x) for x in positions_considered],
        g_signed=float(gradient_signed),
        g_abs=float(g_abs),
        g_lcb=float(g_lcb),
        sigma_hat=float(max(0.0, sigma_hat)),
        F_metric=float(raw_metric),
        metric_proxy=float(raw_metric),
        novelty=None,
        curvature_mode="append_screen_only",
        novelty_mode="none",
        refit_window_indices=[int(i) for i in refit_window_indices],
        compiled_position_cost_proxy={
            "new_pauli_actions": float(compile_cost.new_pauli_actions),
            "new_rotation_steps": float(compile_cost.new_rotation_steps),
            "position_shift_span": float(compile_cost.position_shift_span),
            "refit_active_count": float(compile_cost.refit_active_count),
            "proxy_total": float(compile_cost.proxy_total),
        },
        measurement_cache_stats={
            "groups_total": float(measurement_stats.groups_total),
            "groups_reused": float(measurement_stats.groups_reused),
            "groups_new": float(measurement_stats.groups_new),
            "shots_reused": float(measurement_stats.shots_reused),
            "shots_new": float(measurement_stats.shots_new),
            "reuse_count_cost": float(measurement_stats.reuse_count_cost),
        },
        leakage_penalty=float(max(0.0, leakage_penalty)),
        stage_gate_open=bool(stage_gate_open),
        leakage_gate_open=bool(leakage_gate_open),
        trough_probe_triggered=bool(trough_probe_triggered),
        trough_detected=bool(trough_detected),
        simple_score=None,
        score_version=str(cfg.score_version),
        F_raw=float(raw_metric),
        depth_cost=float(compile_cost.new_pauli_actions + compile_cost.new_rotation_steps),
        new_group_cost=float(measurement_stats.groups_new),
        new_shot_cost=float(measurement_stats.shots_new),
        opt_dim_cost=float(len(refit_window_indices)),
        reuse_count_cost=float(measurement_stats.reuse_count_cost),
        family_repeat_cost=float(max(0.0, family_repeat_cost)),
        generator_id=(
            str(generator_metadata.get("generator_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("generator_id") is not None
            else None
        ),
        template_id=(
            str(generator_metadata.get("template_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("template_id") is not None
            else None
        ),
        is_macro_generator=bool(generator_metadata.get("is_macro_generator", False)) if isinstance(generator_metadata, Mapping) else False,
        parent_generator_id=(
            str(generator_metadata.get("parent_generator_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("parent_generator_id") is not None
            else None
        ),
        generator_metadata=(dict(generator_metadata) if isinstance(generator_metadata, Mapping) else None),
        symmetry_spec=(dict(symmetry_spec) if isinstance(symmetry_spec, Mapping) else None),
        symmetry_mode=str(symmetry_mode),
        symmetry_mitigation_mode=str(symmetry_mitigation_mode),
        motif_metadata=(dict(motif_metadata) if isinstance(motif_metadata, Mapping) else None),
        motif_bonus=float(max(0.0, motif_bonus)),
        motif_source=str(motif_source),
        remaining_evaluations_proxy=float(remaining_eval_proxy),
        remaining_evaluations_proxy_mode=str(remaining_evaluations_proxy_mode),
        lifetime_cost_mode=str(lifetime_cost_mode),
        lifetime_weight_components={
            "remaining_evaluations_proxy": float(remaining_eval_proxy),
        },
        placeholder_hooks={
            "novelty_oracle": False,
            "curvature_oracle": False,
            "full_v2_score": False,
            "qn_spsa_refresh": False,
            "motif_metadata": False,
            "symmetry_metadata": bool(isinstance(symmetry_spec, Mapping)),
        },
    )
    score = simple_v1_score(feat, cfg)
    return _replace_feature(feat, simple_score=float(score))


# ---------------------------------------------------------------------------
# Legacy compatibility helpers (inactive in active append-only path)
# ---------------------------------------------------------------------------

def _pauli_labels_from_term(term: Any) -> list[str]:
    labels: list[str] = []
    if term is None or not hasattr(term, "polynomial"):
        return labels
    for poly_term in term.polynomial.return_polynomial():
        labels.append(str(poly_term.pw2strng()))
    return labels


def _support_set(term: Any) -> set[int]:
    support: set[int] = set()
    labels = _pauli_labels_from_term(term)
    for label in labels:
        for idx, ch in enumerate(str(label)):
            if ch != "e":
                support.add(int(idx))
    return support


def _pauli_strings_commute(lhs: str, rhs: str) -> bool:
    anticomm = 0
    for a, b in zip(str(lhs), str(rhs)):
        if a == "e" or b == "e" or a == b:
            continue
        anticomm += 1
    return bool((anticomm % 2) == 0)


def _polynomials_commute(term_a: Any, term_b: Any) -> bool:
    labels_a = _pauli_labels_from_term(term_a)
    labels_b = _pauli_labels_from_term(term_b)
    if not labels_a or not labels_b:
        return True
    for lhs in labels_a:
        for rhs in labels_b:
            if not _pauli_strings_commute(lhs, rhs):
                return False
    return True


def compatibility_penalty(
    *,
    record_a: Mapping[str, Any],
    record_b: Mapping[str, Any],
    cfg: FullScoreConfig,
    psi_state: np.ndarray | None = None,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> dict[str, float]:
    del compiled_cache, pauli_action_cache
    feat_a = record_a.get("feature")
    feat_b = record_b.get("feature")
    term_a = record_a.get("candidate_term")
    term_b = record_b.get("candidate_term")
    if not isinstance(feat_a, CandidateFeatures) or not isinstance(feat_b, CandidateFeatures):
        return {"support_overlap": 0.0, "noncommutation": 0.0, "cross_curvature": 0.0, "schedule": 0.0, "total": 0.0}

    supp_a = _support_set(term_a)
    supp_b = _support_set(term_b)
    union = len(supp_a | supp_b)
    support_overlap = 0.0 if union == 0 else float(len(supp_a & supp_b) / union)
    noncomm = 0.0 if _polynomials_commute(term_a, term_b) else 1.0

    cross_curv = 0.0
    vec_a = np.asarray(
        feat_a.q_reduced if feat_a.q_reduced is not None else (feat_a.b_mixed if feat_a.b_mixed is not None else []),
        dtype=float,
    )
    vec_b = np.asarray(
        feat_b.q_reduced if feat_b.q_reduced is not None else (feat_b.b_mixed if feat_b.b_mixed is not None else []),
        dtype=float,
    )
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom > 0.0:
        cross_curv = float(min(1.0, abs(float(vec_a @ vec_b)) / denom))
    elif psi_state is not None:
        cross_curv = float(support_overlap)

    win_a = set(int(i) for i in feat_a.refit_window_indices)
    win_b = set(int(i) for i in feat_b.refit_window_indices)
    union_w = len(win_a | win_b)
    schedule = 0.0 if union_w == 0 else float(len(win_a & win_b) / union_w)
    total = (
        float(cfg.compat_overlap_weight) * float(support_overlap)
        + float(cfg.compat_comm_weight) * float(noncomm)
        + float(cfg.compat_curv_weight) * float(cross_curv)
        + float(cfg.compat_sched_weight) * float(schedule)
    )
    return {
        "support_overlap": float(support_overlap),
        "noncommutation": float(noncomm),
        "cross_curvature": float(cross_curv),
        "schedule": float(schedule),
        "total": float(total),
    }


class CompatibilityPenaltyOracle:
    def __init__(
        self,
        *,
        cfg: FullScoreConfig,
        psi_state: np.ndarray | None = None,
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.psi_state = None if psi_state is None else np.asarray(psi_state, dtype=complex)
        self.compiled_cache = compiled_cache
        self.pauli_action_cache = pauli_action_cache

    def penalty(self, record_a: Mapping[str, Any], record_b: Mapping[str, Any]) -> dict[str, float]:
        return compatibility_penalty(
            record_a=record_a,
            record_b=record_b,
            cfg=self.cfg,
            psi_state=self.psi_state,
            compiled_cache=self.compiled_cache,
            pauli_action_cache=self.pauli_action_cache,
        )


def greedy_batch_select(
    ranked_records: Sequence[Mapping[str, Any]],
    compat_oracle: CompatibilityPenaltyOracle,
    cfg: FullScoreConfig,
) -> tuple[list[dict[str, Any]], float]:
    del compat_oracle, cfg
    ranked = sorted(
        [dict(rec) for rec in ranked_records],
        key=lambda rec: (
            -float(rec.get("full_v2_score", float("-inf"))),
            -float(rec.get("simple_score", float("-inf"))),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return [], 0.0
    return [dict(ranked[0])], 0.0
