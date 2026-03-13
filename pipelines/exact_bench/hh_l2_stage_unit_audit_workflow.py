#!/usr/bin/env python3
"""Dedicated HH L=2, n_ph_max=2 stage-unit energy-contribution audit workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_staged_cli_args import build_staged_hh_parser
from pipelines.hardcoded.hh_staged_workflow import (
    StageExecutionResult,
    StagedHHConfig,
    resolve_staged_hh_config,
    run_stage_pipeline,
)
from src.quantum.vqe_latex_python_pairs import apply_exp_pauli_polynomial


_AUDIT_OUTPUT_JSON = REPO_ROOT / "artifacts" / "json" / "hh_l2_stage_unit_audit.json"
_AUDIT_OUTPUT_CSV = REPO_ROOT / "artifacts" / "json" / "hh_l2_stage_unit_audit.csv"
_STAGE_TAG_DEFAULT = "hh_l2_stage_unit_audit_stagechain"
_SUMMARY_TOP_K = 10


@dataclass(frozen=True)
class AuditWorkflowConfig:
    output_json: Path = _AUDIT_OUTPUT_JSON
    output_csv: Path = _AUDIT_OUTPUT_CSV
    stage_tag: str = _STAGE_TAG_DEFAULT
    t: float = 1.0
    u: float = 2.0
    dv: float = 0.0
    omega0: float = 1.0
    g_ep: float = 1.0
    warm_ansatz: str = "hh_hva_ptw"
    adapt_pool: str | None = "paop_lf_std"
    adapt_continuation_mode: str = "phase3_v1"


@dataclass(frozen=True)
class AuditUnit:
    stage: str
    unit_id: str
    unit_index: int
    sequence_order: int
    unit_kind: str
    unit_label: str
    base_label: str
    theta_value: float
    polynomials: tuple[Any, ...]
    insertion_position: int | None = None
    final_order_index: int | None = None
    unit_parameter_count: int = 1
    unit_logical_2q_count: int = 0
    unit_logical_depth: int = 1


@dataclass(frozen=True)
class StageAuditSpec:
    stage: str
    reference_state: np.ndarray
    expected_full_state: np.ndarray
    units_in_acceptance_order: tuple[AuditUnit, ...]
    full_order_ids: tuple[str, ...]
    prefix_order_ids: tuple[tuple[str, ...], ...]
    reference_energy: float
    stage_metadata: dict[str, Any] = field(default_factory=dict)


"""
E(ψ) = Re[ ψ† H ψ ]
"""
def _state_energy(hmat: np.ndarray, psi: np.ndarray) -> float:
    vec = np.asarray(psi, dtype=complex).reshape(-1)
    return float(np.real(np.vdot(vec, np.asarray(hmat, dtype=complex) @ vec)))


"""
ψ_norm = ψ / ||ψ||
"""
def _normalize_state(psi: np.ndarray) -> np.ndarray:
    vec = np.asarray(psi, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("Statevector has zero norm.")
    return vec / norm


"""
d(ψ, φ) = || ψ - e^{i arg(<φ|ψ>)} φ ||
"""
def _state_distance_up_to_global_phase(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    a = _normalize_state(np.asarray(psi_a, dtype=complex).reshape(-1))
    b = _normalize_state(np.asarray(psi_b, dtype=complex).reshape(-1))
    overlap = complex(np.vdot(b, a))
    phase = 1.0 + 0.0j if abs(overlap) <= 1e-15 else overlap / abs(overlap)
    return float(np.linalg.norm(a - phase * b))


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows_list = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows_list[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


"""
H_audit := locked(HH, L=2, n_ph_max=2, trotter_steps=128,
                  warm/final reps=3, restarts=4, maxiter=1500, optimizer=SPSA)
"""
def build_locked_staged_hh_audit_config(workflow_cfg: AuditWorkflowConfig) -> StagedHHConfig:
    parser = build_staged_hh_parser(
        description=(
            "Dedicated HH L=2, n_ph_max=2 stage-unit audit: local/noiseless only, "
            "no patch selection, no noisy execution, no mitigation."
        )
    )
    argv = [
        "--L",
        "2",
        "--n-ph-max",
        "2",
        "--t",
        f"{float(workflow_cfg.t):.16g}",
        "--u",
        f"{float(workflow_cfg.u):.16g}",
        "--dv",
        f"{float(workflow_cfg.dv):.16g}",
        "--omega0",
        f"{float(workflow_cfg.omega0):.16g}",
        "--g-ep",
        f"{float(workflow_cfg.g_ep):.16g}",
        "--warm-ansatz",
        str(workflow_cfg.warm_ansatz),
        "--ordering",
        "blocked",
        "--boundary",
        "open",
        "--adapt-continuation-mode",
        str(workflow_cfg.adapt_continuation_mode),
        "--phase1-no-prune",
        "--tag",
        str(workflow_cfg.stage_tag),
        "--state-export-prefix",
        str(workflow_cfg.stage_tag),
        "--skip-pdf",
    ]
    if workflow_cfg.adapt_pool is not None:
        argv.extend(["--adapt-pool", str(workflow_cfg.adapt_pool)])
    base_cfg = resolve_staged_hh_config(parser.parse_args(argv))
    return replace(
        base_cfg,
        warm_start=replace(
            base_cfg.warm_start,
            reps=3,
            restarts=4,
            maxiter=1500,
            method="SPSA",
        ),
        adapt=replace(
            base_cfg.adapt,
            pool=(None if workflow_cfg.adapt_pool is None else str(workflow_cfg.adapt_pool)),
            continuation_mode=str(workflow_cfg.adapt_continuation_mode),
            max_depth=80,
            maxiter=2222,
            eps_grad=5e-7,
            eps_energy=1e-9,
            inner_optimizer="SPSA",
            phase1_prune_enabled=False,
        ),
        replay=replace(
            base_cfg.replay,
            reps=3,
            restarts=4,
            maxiter=1500,
            method="SPSA",
        ),
        dynamics=replace(
            base_cfg.dynamics,
            trotter_steps=128,
            enable_drive=False,
        ),
        default_provenance={
            **dict(base_cfg.default_provenance),
            "audit_locked_profile": "AGENTS.hh_L2_nph2.audit_locked_profile",
            "audit_ordering": "audit.fixed=blocked",
            "audit_boundary": "audit.fixed=open",
            "audit_adapt_pool": (
                "audit.fixed=paop_lf_std" if workflow_cfg.adapt_pool is None else "cli"
            ),
            "audit_phase1_prune_enabled": "audit.fixed=false",
            "audit_adapt_max_depth": "audit.fixed=80",
            "audit_adapt_maxiter": "audit.fixed=2222",
            "audit_adapt_eps_grad": "audit.fixed=5e-7",
            "audit_adapt_eps_energy": "audit.fixed=1e-9",
        },
    )


"""
U_unit(θ) = Π_m exp(-i θ P_m)
"""
def _apply_single_polynomial(psi: np.ndarray, polynomial: Any, theta_value: float) -> np.ndarray:
    return np.asarray(
        apply_exp_pauli_polynomial(
            np.asarray(psi, dtype=complex).reshape(-1),
            polynomial,
            float(theta_value),
        ),
        dtype=complex,
    ).reshape(-1)


"""
U_prefix = U_{j_k} ... U_{j_2} U_{j_1}
"""
def _apply_unit(psi: np.ndarray, unit: AuditUnit) -> np.ndarray:
    out = np.asarray(psi, dtype=complex).reshape(-1)
    for polynomial in unit.polynomials:
        out = _apply_single_polynomial(out, polynomial, float(unit.theta_value))
    return _normalize_state(out)


"""
ψ(order) = U_{order[-1]} ... U_{order[0]} ψ_ref
"""
def _apply_order(
    reference_state: np.ndarray,
    units_by_id: Mapping[str, AuditUnit],
    ordered_unit_ids: Sequence[str],
) -> np.ndarray:
    psi = np.asarray(reference_state, dtype=complex).reshape(-1)
    for unit_id in ordered_unit_ids:
        psi = _apply_unit(psi, units_by_id[str(unit_id)])
    return _normalize_state(psi)


def _term_label_exyz(term: Any) -> str:
    return str(term.pw2strng())


def _polynomial_support_width(polynomial: Any, *, tol: float = 1e-12) -> int:
    active_qubits: set[int] = set()
    for term in list(polynomial.return_polynomial()):
        coeff = complex(getattr(term, "p_coeff", 0.0 + 0.0j))
        if abs(coeff) <= float(tol):
            continue
        label = _term_label_exyz(term)
        for q, ch in enumerate(reversed(label)):
            if ch in {"x", "y", "z"}:
                active_qubits.add(int(q))
    return int(len(active_qubits))


"""
C_2q(unit) = Σ_poly 2 · max(width(poly) - 1, 0)
D(unit) = number_of_constituent_polynomials(unit)
"""
def _unit_cost_metrics(polynomials: Sequence[Any]) -> tuple[int, int]:
    logical_2q = 0
    logical_depth = 0
    for polynomial in polynomials:
        logical_depth += 1
        logical_2q += int(2 * max(_polynomial_support_width(polynomial) - 1, 0))
    return int(logical_2q), int(logical_depth)


def _make_unit(
    *,
    stage: str,
    unit_index: int,
    unit_kind: str,
    unit_label: str,
    base_label: str,
    theta_value: float,
    polynomials: Sequence[Any],
    insertion_position: int | None,
    final_order_index: int | None,
) -> AuditUnit:
    logical_2q, logical_depth = _unit_cost_metrics(polynomials)
    return AuditUnit(
        stage=str(stage),
        unit_id=f"{stage}:{int(unit_index)}",
        unit_index=int(unit_index),
        sequence_order=0,
        unit_kind=str(unit_kind),
        unit_label=str(unit_label),
        base_label=str(base_label),
        theta_value=float(theta_value),
        polynomials=tuple(polynomials),
        insertion_position=(None if insertion_position is None else int(insertion_position)),
        final_order_index=(None if final_order_index is None else int(final_order_index)),
        unit_parameter_count=1,
        unit_logical_2q_count=int(logical_2q),
        unit_logical_depth=int(logical_depth),
    )


def _assign_sequence_orders(stage_specs: Sequence[StageAuditSpec]) -> tuple[StageAuditSpec, ...]:
    next_sequence = 1
    updated_specs: list[StageAuditSpec] = []
    for spec in stage_specs:
        remapped_units: list[AuditUnit] = []
        for unit in spec.units_in_acceptance_order:
            remapped_units.append(replace(unit, sequence_order=int(next_sequence)))
            next_sequence += 1
        updated_specs.append(
            replace(
                spec,
                units_in_acceptance_order=tuple(remapped_units),
            )
        )
    return tuple(updated_specs)


def _warm_stage_spec(stage_result: StageExecutionResult) -> StageAuditSpec:
    warm_ctx = stage_result.warm_circuit_context
    if not isinstance(warm_ctx, Mapping) or warm_ctx.get("ansatz") is None:
        raise ValueError("Warm stage circuit context is unavailable.")
    ansatz = warm_ctx["ansatz"]
    theta = np.asarray(warm_ctx.get("theta", []), dtype=float).reshape(-1)
    psi_ref = np.asarray(warm_ctx.get("reference_state"), dtype=complex).reshape(-1)
    reps = int(getattr(ansatz, "reps", 1))
    units: list[AuditUnit] = []
    if hasattr(ansatz, "layer_term_groups") and list(getattr(ansatz, "layer_term_groups", [])):
        groups = list(getattr(ansatz, "layer_term_groups"))
        k = 0
        for rep_idx in range(reps):
            for group_idx, (group_name, group_terms) in enumerate(groups, start=1):
                if k >= int(theta.size):
                    raise ValueError("Warm layerwise theta traversal exceeded theta size.")
                units.append(
                    _make_unit(
                        stage="warm_start",
                        unit_index=len(units) + 1,
                        unit_kind="layer_sub_block",
                        unit_label=f"layer{int(rep_idx + 1)}:{str(group_name)}",
                        base_label=str(group_name),
                        theta_value=float(theta[k]),
                        polynomials=[term.polynomial for term in list(group_terms)],
                        insertion_position=len(units),
                        final_order_index=len(units),
                    )
                )
                k += 1
        if k != int(theta.size):
            raise ValueError("Warm layerwise theta traversal did not consume the full theta vector.")
    else:
        base_terms = list(getattr(ansatz, "base_terms", []))
        if not base_terms:
            raise ValueError("Warm ansatz is missing base_terms.")
        k = 0
        for rep_idx in range(reps):
            for term in base_terms:
                if k >= int(theta.size):
                    raise ValueError("Warm termwise theta traversal exceeded theta size.")
                units.append(
                    _make_unit(
                        stage="warm_start",
                        unit_index=len(units) + 1,
                        unit_kind="logical_block",
                        unit_label=f"layer{int(rep_idx + 1)}:{str(term.label)}",
                        base_label=str(term.label),
                        theta_value=float(theta[k]),
                        polynomials=[term.polynomial],
                        insertion_position=len(units),
                        final_order_index=len(units),
                    )
                )
                k += 1
        if k != int(theta.size):
            raise ValueError("Warm termwise theta traversal did not consume the full theta vector.")
    full_order = tuple(unit.unit_id for unit in units)
    prefix_orders = tuple(tuple(unit.unit_id for unit in units[:idx]) for idx in range(1, len(units) + 1))
    return StageAuditSpec(
        stage="warm_start",
        reference_state=np.asarray(psi_ref, dtype=complex).reshape(-1),
        expected_full_state=np.asarray(stage_result.psi_warm, dtype=complex).reshape(-1),
        units_in_acceptance_order=tuple(units),
        full_order_ids=full_order,
        prefix_order_ids=prefix_orders,
        reference_energy=_state_energy(stage_result.hmat, psi_ref),
        stage_metadata={
            "ansatz_name": str(warm_ctx.get("ansatz_name", stage_result.warm_payload.get("ansatz", "unknown"))),
            "reps": int(reps),
        },
    )


def _adapt_acceptance_events(adapt_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    history = adapt_payload.get("history", [])
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ValueError("ADAPT payload missing history rows required for acceptance-order reconstruction.")
    events: list[dict[str, Any]] = []
    for row_index, row in enumerate(history, start=1):
        if not isinstance(row, Mapping):
            continue
        labels = row.get("selected_ops", None)
        positions = row.get("selected_positions", None)
        if (
            isinstance(labels, Sequence)
            and not isinstance(labels, (str, bytes))
            and isinstance(positions, Sequence)
            and not isinstance(positions, (str, bytes))
            and len(labels) > 0
            and len(labels) == len(positions)
        ):
            original_positions_seen: list[int] = []
            for batch_index, (label_raw, pos_raw) in enumerate(zip(labels, positions), start=1):
                pos_orig = int(pos_raw)
                pos_eff = int(pos_orig + sum(1 for prev in original_positions_seen if prev <= pos_orig))
                original_positions_seen.append(int(pos_orig))
                events.append(
                    {
                        "base_label": str(label_raw),
                        "effective_position": int(pos_eff),
                        "history_row_index": int(row_index),
                        "batch_index": int(batch_index),
                    }
                )
            continue
        selected_op = row.get("selected_op", None)
        if selected_op is None:
            continue
        events.append(
            {
                "base_label": str(selected_op),
                "effective_position": int(row.get("selected_position", len(events))),
                "history_row_index": int(row_index),
                "batch_index": 1,
            }
        )
    if not events:
        raise ValueError("ADAPT history did not contain any accepted-operator events.")
    return events


def _adapt_stage_spec(stage_result: StageExecutionResult) -> StageAuditSpec:
    adapt_ctx = stage_result.adapt_circuit_context
    if not isinstance(adapt_ctx, Mapping) or adapt_ctx.get("selected_ops") is None:
        raise ValueError("ADAPT stage circuit context is unavailable.")
    selected_ops = list(adapt_ctx.get("selected_ops", []))
    theta_final = np.asarray(adapt_ctx.get("theta", []), dtype=float).reshape(-1)
    psi_ref = np.asarray(adapt_ctx.get("reference_state"), dtype=complex).reshape(-1)
    if len(selected_ops) != int(theta_final.size):
        raise ValueError("ADAPT selected ops / theta size mismatch.")
    acceptance_events = _adapt_acceptance_events(stage_result.adapt_payload)
    seed_prefix_depth = int(len(selected_ops) - len(acceptance_events))
    if seed_prefix_depth < 0:
        raise ValueError(
            "ADAPT acceptance-event count exceeds final selected-operator count. "
            f"events={len(acceptance_events)} final={len(selected_ops)}"
        )

    mixed_order: list[int] = [-(idx + 1) for idx in range(seed_prefix_depth)]
    prefix_orders_raw: list[tuple[int, ...]] = []
    for event_index, event in enumerate(acceptance_events):
        pos = max(0, min(len(mixed_order), int(event["effective_position"])))
        mixed_order.insert(pos, int(event_index))
        prefix_orders_raw.append(tuple(int(x) for x in mixed_order if int(x) >= 0))

    final_order_event_indices = [int(x) for x in mixed_order if int(x) >= 0]
    final_order_positions = [idx for idx, marker in enumerate(mixed_order) if int(marker) >= 0]
    if len(final_order_event_indices) != len(acceptance_events):
        raise ValueError("ADAPT mixed-order reconstruction lost accepted-event markers.")
    if final_order_positions != list(range(seed_prefix_depth, len(selected_ops))):
        raise ValueError(
            "ADAPT stage-unit audit requires accepted insertions to remain after the seeded prefix. "
            f"seed_prefix_depth={seed_prefix_depth} final_event_positions={final_order_positions[:8]}"
        )

    psi_stage_ref = np.asarray(psi_ref, dtype=complex).reshape(-1)
    for prefix_index in range(seed_prefix_depth):
        seed_op = selected_ops[int(prefix_index)]
        psi_stage_ref = _apply_single_polynomial(
            np.asarray(psi_stage_ref, dtype=complex).reshape(-1),
            seed_op.polynomial,
            float(theta_final[prefix_index]),
        )
    psi_stage_ref = _normalize_state(psi_stage_ref)

    expected_labels = [str(op.label) for op in selected_ops[seed_prefix_depth:]]
    actual_labels = [str(acceptance_events[idx]["base_label"]) for idx in final_order_event_indices]
    if actual_labels != expected_labels:
        raise ValueError(
            "ADAPT acceptance-order reconstruction does not match final accepted-operator order. "
            f"reconstructed={actual_labels[:8]} expected={expected_labels[:8]}"
        )

    units_by_event_index: dict[int, AuditUnit] = {}
    for accepted_offset, event_index in enumerate(final_order_event_indices):
        full_order_index = int(seed_prefix_depth + accepted_offset)
        event = acceptance_events[int(event_index)]
        op = selected_ops[int(full_order_index)]
        units_by_event_index[int(event_index)] = _make_unit(
            stage="adapt_vqe",
            unit_index=int(event_index + 1),
            unit_kind="accepted_operator_insertion",
            unit_label=f"accept{int(event_index + 1)}:{str(op.label)}",
            base_label=str(op.label),
            theta_value=float(theta_final[full_order_index]),
            polynomials=[op.polynomial],
            insertion_position=int(event["effective_position"]),
            final_order_index=int(full_order_index),
        )
    units = tuple(units_by_event_index[idx] for idx in range(len(acceptance_events)))
    full_order_ids = tuple(units_by_event_index[idx].unit_id for idx in final_order_event_indices)
    prefix_order_ids = tuple(
        tuple(units_by_event_index[idx].unit_id for idx in order_event_indices)
        for order_event_indices in prefix_orders_raw
    )
    return StageAuditSpec(
        stage="adapt_vqe",
        reference_state=np.asarray(psi_stage_ref, dtype=complex).reshape(-1),
        expected_full_state=np.asarray(stage_result.psi_adapt, dtype=complex).reshape(-1),
        units_in_acceptance_order=units,
        full_order_ids=full_order_ids,
        prefix_order_ids=prefix_order_ids,
        reference_energy=_state_energy(stage_result.hmat, psi_stage_ref),
        stage_metadata={
            "pool_type": str(adapt_ctx.get("pool_type", stage_result.adapt_payload.get("pool_type", "unknown"))),
            "continuation_mode": str(
                adapt_ctx.get("continuation_mode", stage_result.adapt_payload.get("continuation_mode", "unknown"))
            ),
            "ansatz_depth": int(len(units)),
            "seed_prefix_depth": int(seed_prefix_depth),
        },
    )


def _replay_stage_spec(stage_result: StageExecutionResult) -> StageAuditSpec:
    replay_ctx = stage_result.replay_circuit_context
    if not isinstance(replay_ctx, Mapping) or replay_ctx.get("ansatz") is None:
        raise ValueError("Replay stage circuit context is unavailable.")
    ansatz = replay_ctx["ansatz"]
    theta = np.asarray(replay_ctx.get("theta", []), dtype=float).reshape(-1)
    psi_ref = np.asarray(replay_ctx.get("reference_state"), dtype=complex).reshape(-1)
    reps = int(getattr(ansatz, "reps", 1))
    base_terms = list(getattr(ansatz, "base_terms", []))
    if not base_terms:
        raise ValueError("Replay ansatz is missing base_terms.")
    units: list[AuditUnit] = []
    k = 0
    for rep_idx in range(reps):
        for term in base_terms:
            if k >= int(theta.size):
                raise ValueError("Replay theta traversal exceeded theta size.")
            units.append(
                _make_unit(
                    stage="conventional_replay",
                    unit_index=len(units) + 1,
                    unit_kind="logical_block",
                    unit_label=f"layer{int(rep_idx + 1)}:{str(term.label)}",
                    base_label=str(term.label),
                    theta_value=float(theta[k]),
                    polynomials=[term.polynomial],
                    insertion_position=len(units),
                    final_order_index=len(units),
                )
            )
            k += 1
    if k != int(theta.size):
        raise ValueError("Replay theta traversal did not consume the full theta vector.")
    full_order = tuple(unit.unit_id for unit in units)
    prefix_orders = tuple(tuple(unit.unit_id for unit in units[:idx]) for idx in range(1, len(units) + 1))
    return StageAuditSpec(
        stage="conventional_replay",
        reference_state=np.asarray(psi_ref, dtype=complex).reshape(-1),
        expected_full_state=np.asarray(stage_result.psi_final, dtype=complex).reshape(-1),
        units_in_acceptance_order=tuple(units),
        full_order_ids=full_order,
        prefix_order_ids=prefix_orders,
        reference_energy=_state_energy(stage_result.hmat, psi_ref),
        stage_metadata={
            "reps": int(reps),
            "family_info": dict(replay_ctx.get("family_info", {})),
            "seed_policy": str(replay_ctx.get("resolved_seed_policy", "unknown")),
        },
    )


def build_stage_audit_specs(stage_result: StageExecutionResult) -> tuple[StageAuditSpec, ...]:
    specs = (
        _warm_stage_spec(stage_result),
        _adapt_stage_spec(stage_result),
        _replay_stage_spec(stage_result),
    )
    return _assign_sequence_orders(specs)


def _prefix_hash(stage: str, units: Sequence[AuditUnit]) -> str:
    manifest = {
        "stage": str(stage),
        "units": [
            {
                "unit_id": str(unit.unit_id),
                "unit_label": str(unit.unit_label),
                "theta_value": float(unit.theta_value),
                "unit_kind": str(unit.unit_kind),
            }
            for unit in units
        ],
    }
    return hashlib.sha1(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _row_summary_view(row: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "stage",
        "unit_index",
        "unit_label",
        "base_label",
        "sequence_order",
        "delta_energy_from_previous",
        "removal_penalty",
        "energy_gain_per_2q",
        "energy_gain_per_depth",
        "unit_logical_2q_count",
        "unit_logical_depth",
    )
    return {str(key): row.get(key) for key in keep}


def _rank_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows_list = [dict(row) for row in rows]
    per2q = [row for row in rows_list if row.get("energy_gain_per_2q") is not None]
    return {
        "top_k": int(_SUMMARY_TOP_K),
        "smallest_delta_energy_from_previous": [
            _row_summary_view(row)
            for row in sorted(rows_list, key=lambda row: float(row.get("delta_energy_from_previous", float("inf"))))[:_SUMMARY_TOP_K]
        ],
        "smallest_removal_penalty": [
            _row_summary_view(row)
            for row in sorted(rows_list, key=lambda row: float(row.get("removal_penalty", float("inf"))))[:_SUMMARY_TOP_K]
        ],
        "worst_energy_gain_per_2q": [
            _row_summary_view(row)
            for row in sorted(per2q, key=lambda row: float(row.get("energy_gain_per_2q", float("inf"))))[:_SUMMARY_TOP_K]
        ],
        "worst_energy_gain_per_depth": [
            _row_summary_view(row)
            for row in sorted(rows_list, key=lambda row: float(row.get("energy_gain_per_depth", float("inf"))))[:_SUMMARY_TOP_K]
        ],
    }


"""
Δ_prefix[k] = E_prefix[k-1] - E_prefix[k]
Δ_remove[k] = E_full_without_k - E_full
"""
def compute_stage_audit_rows(spec: StageAuditSpec, hmat: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    units_by_id = {str(unit.unit_id): unit for unit in spec.units_in_acceptance_order}
    full_units = [units_by_id[unit_id] for unit_id in spec.full_order_ids]
    psi_full = _apply_order(spec.reference_state, units_by_id, spec.full_order_ids)
    full_energy = _state_energy(hmat, psi_full)
    reconstruction_error = _state_distance_up_to_global_phase(psi_full, spec.expected_full_state)
    full_circuit_hash = _prefix_hash(spec.stage, full_units)

    rows: list[dict[str, Any]] = []
    previous_energy = float(spec.reference_energy)
    for prefix_order_ids, unit in zip(spec.prefix_order_ids, spec.units_in_acceptance_order):
        prefix_units = [units_by_id[unit_id] for unit_id in prefix_order_ids]
        psi_prefix = _apply_order(spec.reference_state, units_by_id, prefix_order_ids)
        energy_prefix = _state_energy(hmat, psi_prefix)
        delta_energy = float(previous_energy - energy_prefix)
        removal_order_ids = tuple(unit_id for unit_id in spec.full_order_ids if str(unit_id) != str(unit.unit_id))
        psi_minus = _apply_order(spec.reference_state, units_by_id, removal_order_ids)
        energy_minus = _state_energy(hmat, psi_minus)
        removal_penalty = float(energy_minus - full_energy)
        prefix_parameter_count = int(len(prefix_units))
        prefix_logical_2q_count = int(sum(int(item.unit_logical_2q_count) for item in prefix_units))
        prefix_logical_depth = int(sum(int(item.unit_logical_depth) for item in prefix_units))
        energy_gain_per_2q = (
            None
            if int(unit.unit_logical_2q_count) <= 0
            else float(delta_energy / float(unit.unit_logical_2q_count))
        )
        energy_gain_per_depth = float(delta_energy / float(max(1, unit.unit_logical_depth)))
        rows.append(
            {
                "stage": str(spec.stage),
                "unit_index": int(unit.unit_index),
                "unit_kind": str(unit.unit_kind),
                "unit_label": str(unit.unit_label),
                "base_label": str(unit.base_label),
                "sequence_order": int(unit.sequence_order),
                "theta_value": float(unit.theta_value),
                "reference_energy": float(spec.reference_energy),
                "energy_prefix": float(energy_prefix),
                "delta_energy_from_previous": float(delta_energy),
                "energy_full": float(full_energy),
                "energy_full_minus_unit": float(energy_minus),
                "removal_penalty": float(removal_penalty),
                "parameter_count": int(prefix_parameter_count),
                "logical_2q_count": int(prefix_logical_2q_count),
                "logical_depth": int(prefix_logical_depth),
                "unit_logical_2q_count": int(unit.unit_logical_2q_count),
                "unit_logical_depth": int(unit.unit_logical_depth),
                "energy_gain_per_2q": energy_gain_per_2q,
                "energy_gain_per_depth": float(energy_gain_per_depth),
                "circuit_hash": str(_prefix_hash(spec.stage, prefix_units)),
                "removal_circuit_hash": str(_prefix_hash(spec.stage, [units_by_id[uid] for uid in removal_order_ids])),
                "full_circuit_hash": str(full_circuit_hash),
            }
        )
        previous_energy = float(energy_prefix)

    stage_summary = {
        "stage": str(spec.stage),
        "reference_energy": float(spec.reference_energy),
        "full_energy": float(full_energy),
        "unit_count": int(len(spec.units_in_acceptance_order)),
        "full_circuit_hash": str(full_circuit_hash),
        "reconstruction_error_global_phase_aligned": float(reconstruction_error),
        "stage_metadata": dict(spec.stage_metadata),
        "acceptance_order_labels": [str(unit.unit_label) for unit in spec.units_in_acceptance_order],
        "final_order_labels": [str(units_by_id[unit_id].unit_label) for unit_id in spec.full_order_ids],
    }
    return rows, stage_summary


def build_audit_payload(
    workflow_cfg: AuditWorkflowConfig,
    staged_cfg: StagedHHConfig,
    stage_result: StageExecutionResult,
) -> dict[str, Any]:
    specs = build_stage_audit_specs(stage_result)
    all_rows: list[dict[str, Any]] = []
    stage_summaries: dict[str, Any] = {}
    for spec in specs:
        rows, stage_summary = compute_stage_audit_rows(spec, stage_result.hmat)
        all_rows.extend(rows)
        stage_summaries[str(spec.stage)] = stage_summary

    payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_l2_stage_unit_audit",
        "model_family": "Hubbard-Holstein (HH)",
        "audit_scope": {
            "local_only": True,
            "noiseless_only": True,
            "noise_enabled": False,
            "patch_selection_enabled": False,
            "mitigation_enabled": False,
            "removal_variant": "fixed_parameter_only",
            "reoptimized_removals_included": False,
        },
        "settings": {
            "physics": asdict(staged_cfg.physics),
            "warm_start": asdict(staged_cfg.warm_start),
            "adapt": asdict(staged_cfg.adapt),
            "replay": asdict(staged_cfg.replay),
            "dynamics": asdict(staged_cfg.dynamics),
            "gates": asdict(staged_cfg.gates),
            "default_provenance": dict(staged_cfg.default_provenance),
            "audit_locked_profile": {
                "L": 2,
                "n_ph_max": 2,
                "trotter_steps": 128,
                "warm_reps": 3,
                "warm_restarts": 4,
                "warm_maxiter": 1500,
                "final_reps": 3,
                "final_restarts": 4,
                "final_maxiter": 1500,
                "optimizer": "SPSA",
                "ordering": "blocked",
                "boundary": "open",
                "adapt_pool": staged_cfg.adapt.pool,
                "adapt_continuation_mode": staged_cfg.adapt.continuation_mode,
                "adapt_max_depth": staged_cfg.adapt.max_depth,
                "adapt_maxiter": staged_cfg.adapt.maxiter,
                "adapt_eps_grad": staged_cfg.adapt.eps_grad,
                "adapt_eps_energy": staged_cfg.adapt.eps_energy,
                "phase1_prune_enabled": staged_cfg.adapt.phase1_prune_enabled,
            },
        },
        "artifacts": {
            "output_json": str(workflow_cfg.output_json),
            "output_csv": str(workflow_cfg.output_csv),
            "stage_workflow": asdict(staged_cfg.artifacts),
        },
        "metric_semantics": {
            "energy_prefix": "Stage-local exact energy after applying accepted units up to this point with final fixed parameters.",
            "delta_energy_from_previous": "Stage-local reference/prefix energy lowering: previous_energy - current_prefix_energy.",
            "energy_full_minus_unit": "Stage-local exact energy after removing this unit from the final stage structure at fixed final parameters.",
            "removal_penalty": "energy_full_minus_unit - energy_full.",
            "parameter_count": "Cumulative stage-prefix parameter count.",
            "logical_2q_count": "Cumulative stage-prefix logical two-qubit proxy count based on stage-native polynomial support widths; not transpiled hardware counts.",
            "logical_depth": "Cumulative stage-prefix logical depth proxy = count of stage-native polynomial exponentials in the prefix.",
            "circuit_hash": "Hash of the stage-prefix logical-unit manifest, not a transpiled boundary circuit hash.",
            "energy_gain_per_2q": "delta_energy_from_previous divided by unit_logical_2q_count when unit_logical_2q_count > 0.",
            "energy_gain_per_depth": "delta_energy_from_previous divided by unit_logical_depth.",
        },
        "stage_summaries": stage_summaries,
        "rows": all_rows,
        "summary": _rank_rows(all_rows),
        "boundary_metrics": {
            "included": False,
            "reason": "Deferred in first-pass implementation; stage-native logical audit is primary.",
        },
    }
    return payload


def emit_audit_files(payload: Mapping[str, Any], workflow_cfg: AuditWorkflowConfig) -> None:
    rows = payload.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("Audit payload rows must be a sequence.")
    _write_json(Path(workflow_cfg.output_json), payload)
    _write_csv(Path(workflow_cfg.output_csv), [dict(row) for row in rows if isinstance(row, Mapping)])


def run_hh_l2_stage_unit_audit(
    workflow_cfg: AuditWorkflowConfig,
    *,
    stage_result: StageExecutionResult | None = None,
) -> dict[str, Any]:
    staged_cfg = build_locked_staged_hh_audit_config(workflow_cfg)
    resolved_stage_result = run_stage_pipeline(staged_cfg) if stage_result is None else stage_result
    payload = build_audit_payload(workflow_cfg, staged_cfg, resolved_stage_result)
    emit_audit_files(payload, workflow_cfg)
    return payload


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dedicated hardcoded HH L=2, n_ph_max=2 stage-unit audit. "
            "Exact/noiseless/local-only. No patch selection, no noisy execution, no mitigation."
        )
    )
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=2.0)
    parser.add_argument("--dv", type=float, default=0.0)
    parser.add_argument("--omega0", type=float, default=1.0)
    parser.add_argument("--g-ep", type=float, default=1.0, dest="g_ep")
    parser.add_argument("--warm-ansatz", choices=["hh_hva", "hh_hva_ptw"], default="hh_hva_ptw")
    parser.add_argument("--adapt-pool", type=str, default="paop_lf_std")
    parser.add_argument(
        "--adapt-continuation-mode",
        choices=["legacy", "phase1_v1", "phase2_v1", "phase3_v1"],
        default="phase3_v1",
    )
    parser.add_argument("--stage-tag", type=str, default=_STAGE_TAG_DEFAULT)
    parser.add_argument("--output-json", type=Path, default=_AUDIT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=_AUDIT_OUTPUT_CSV)
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> AuditWorkflowConfig:
    args = build_cli_parser().parse_args(list(argv) if argv is not None else None)
    return AuditWorkflowConfig(
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
        stage_tag=str(args.stage_tag),
        t=float(args.t),
        u=float(args.u),
        dv=float(args.dv),
        omega0=float(args.omega0),
        g_ep=float(args.g_ep),
        warm_ansatz=str(args.warm_ansatz),
        adapt_pool=(None if args.adapt_pool is None else str(args.adapt_pool)),
        adapt_continuation_mode=str(args.adapt_continuation_mode),
    )


def format_compact_summary(payload: Mapping[str, Any]) -> list[str]:
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    lines = [
        f"audit_json={payload.get('artifacts', {}).get('output_json', '') if isinstance(payload.get('artifacts', {}), Mapping) else ''}",
        f"audit_csv={payload.get('artifacts', {}).get('output_csv', '') if isinstance(payload.get('artifacts', {}), Mapping) else ''}",
    ]
    for key in (
        "smallest_delta_energy_from_previous",
        "smallest_removal_penalty",
        "worst_energy_gain_per_2q",
        "worst_energy_gain_per_depth",
    ):
        items = summary.get(key, []) if isinstance(summary, Mapping) else []
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes)) and len(items) > 0:
            best = items[0]
            if isinstance(best, Mapping):
                lines.append(
                    f"{key}={best.get('stage')}#{best.get('unit_index')}:{best.get('unit_label')}"
                )
    return lines


__all__ = [
    "AuditUnit",
    "AuditWorkflowConfig",
    "StageAuditSpec",
    "build_audit_payload",
    "build_cli_parser",
    "build_locked_staged_hh_audit_config",
    "build_stage_audit_specs",
    "compute_stage_audit_rows",
    "emit_audit_files",
    "format_compact_summary",
    "parse_cli_args",
    "run_hh_l2_stage_unit_audit",
]
