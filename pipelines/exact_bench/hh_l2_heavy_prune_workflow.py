#!/usr/bin/env python3
"""Offline-heavy HH L=2 open-boundary pruning workflow over one staged baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from docs.reports.qiskit_circuit_report import (
    adapt_ops_to_circuit,
    ansatz_to_circuit,
    expand_pauli_evolution_once,
)
from pipelines.exact_bench.hh_l2_stage_unit_audit_workflow import (
    AuditWorkflowConfig,
    build_locked_staged_hh_audit_config,
)
import pipelines.exact_bench.hh_l2_logical_screen_workflow as logical_wf
from pipelines.hardcoded import hh_vqe_from_adapt_family as replay_mod
from pipelines.hardcoded.hh_staged_workflow import StageExecutionResult, StagedHHConfig, run_stage_pipeline


REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_JSON = REPO_ROOT / "artifacts" / "json" / "hh_l2_heavy_prune.json"
_DEFAULT_OUTPUT_CSV = REPO_ROOT / "artifacts" / "json" / "hh_l2_heavy_prune.csv"
_DEFAULT_RUN_ROOT = REPO_ROOT / "artifacts" / "json" / "hh_l2_heavy_prune_run"
_DEFAULT_TAG = "hh_l2_heavy_prune"
_DEFAULT_BASIS_GATES = ("cx", "rz", "sx", "x")


@dataclass(frozen=True)
class HeavyPrunePlan:
    ablation_id: str
    ablation_kind: str
    description: str
    keep_operator_indices: tuple[int, ...]
    removed_operator_indices: tuple[int, ...]
    removed_label: str | None = None
    ranking_entries: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class HeavyPruneConfig:
    output_json: Path = _DEFAULT_OUTPUT_JSON
    output_csv: Path = _DEFAULT_OUTPUT_CSV
    run_root: Path = _DEFAULT_RUN_ROOT
    tag: str = _DEFAULT_TAG
    t: float = 1.0
    u: float = 4.0
    dv: float = 0.0
    omega0: float = 1.0
    g_ep: float = 1.0
    warm_ansatz: str = "hh_hva_ptw"
    adapt_pool: str | None = "paop_lf_std"
    adapt_continuation_mode: str = "phase3_v1"
    ordering: str = "blocked"
    boundary: str = "open"
    warm_seed: int = 7
    adapt_seed: int = 11
    final_seed: int = 19
    include_prefix_50: bool = False
    weakest_single_count: int = 6
    weakest_cumulative_count: int = 6
    warm_vqe_reps_override: int | None = None
    warm_vqe_restarts_override: int | None = None
    warm_vqe_maxiter_override: int | None = None
    adapt_max_depth_override: int | None = None
    adapt_drop_min_depth_override: int | None = None
    adapt_maxiter_override: int | None = None
    final_vqe_reps_override: int | None = None
    final_vqe_restarts_override: int | None = None
    final_vqe_maxiter_override: int | None = None
    basis_gates: tuple[str, ...] = _DEFAULT_BASIS_GATES
    transpile_optimization_level: int = 1


# math: timestamp_now = UTC ISO-8601 string
def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# math: jsonable(x) = recursive conversion of numpy / Path / mappings / sequences to JSON-safe values
def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(v) for v in value]
    return value


# math: write_json(path, payload) = json.dump(jsonable(payload))
def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=False), encoding="utf-8")


# math: write_csv(path, rows) = header union + per-row JSON-safe scalarization
def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded: dict[str, Any] = {}
            for key in fieldnames:
                raw = _jsonable(row.get(key))
                if isinstance(raw, (dict, list)):
                    encoded[key] = json.dumps(raw, sort_keys=True)
                else:
                    encoded[key] = raw
            writer.writerow(encoded)


# math: cfg = locked_audit_profile + explicit open-boundary seeds/artifacts/overrides
def build_heavy_prune_staged_cfg(cfg: HeavyPruneConfig) -> tuple[StagedHHConfig, AuditWorkflowConfig, Path]:
    run_dir = Path(cfg.run_root) / str(cfg.tag)
    audit_cfg = AuditWorkflowConfig(
        output_json=run_dir / "audit.json",
        output_csv=run_dir / "audit.csv",
        stage_tag=str(cfg.tag),
        t=float(cfg.t),
        u=float(cfg.u),
        dv=float(cfg.dv),
        omega0=float(cfg.omega0),
        g_ep=float(cfg.g_ep),
        warm_ansatz=str(cfg.warm_ansatz),
        adapt_pool=(None if cfg.adapt_pool is None else str(cfg.adapt_pool)),
        adapt_continuation_mode=str(cfg.adapt_continuation_mode),
        ordering=str(cfg.ordering),
        boundary=str(cfg.boundary),
    )
    staged_cfg = build_locked_staged_hh_audit_config(audit_cfg)
    staged_cfg = replace(
        staged_cfg,
        warm_start=replace(
            staged_cfg.warm_start,
            seed=int(cfg.warm_seed),
            reps=(int(cfg.warm_vqe_reps_override) if cfg.warm_vqe_reps_override is not None else int(staged_cfg.warm_start.reps)),
            restarts=(int(cfg.warm_vqe_restarts_override) if cfg.warm_vqe_restarts_override is not None else int(staged_cfg.warm_start.restarts)),
            maxiter=(int(cfg.warm_vqe_maxiter_override) if cfg.warm_vqe_maxiter_override is not None else int(staged_cfg.warm_start.maxiter)),
        ),
        adapt=replace(
            staged_cfg.adapt,
            seed=int(cfg.adapt_seed),
            max_depth=(int(cfg.adapt_max_depth_override) if cfg.adapt_max_depth_override is not None else int(staged_cfg.adapt.max_depth)),
            drop_min_depth=(int(cfg.adapt_drop_min_depth_override) if cfg.adapt_drop_min_depth_override is not None else staged_cfg.adapt.drop_min_depth),
            maxiter=(int(cfg.adapt_maxiter_override) if cfg.adapt_maxiter_override is not None else int(staged_cfg.adapt.maxiter)),
        ),
        replay=replace(
            staged_cfg.replay,
            seed=int(cfg.final_seed),
            reps=(int(cfg.final_vqe_reps_override) if cfg.final_vqe_reps_override is not None else int(staged_cfg.replay.reps)),
            restarts=(int(cfg.final_vqe_restarts_override) if cfg.final_vqe_restarts_override is not None else int(staged_cfg.replay.restarts)),
            maxiter=(int(cfg.final_vqe_maxiter_override) if cfg.final_vqe_maxiter_override is not None else int(staged_cfg.replay.maxiter)),
        ),
        warm_checkpoint=replace(
            staged_cfg.warm_checkpoint,
            state_export_dir=run_dir / "state_export",
            state_export_prefix=str(cfg.tag),
        ),
        artifacts=replace(
            staged_cfg.artifacts,
            tag=str(cfg.tag),
            output_json=run_dir / "staged_noiseless.json",
            output_pdf=run_dir / "staged_noiseless.pdf",
            handoff_json=run_dir / "handoff.json",
            warm_checkpoint_json=run_dir / "warm_checkpoint.json",
            warm_cutover_json=run_dir / "warm_cutover.json",
            replay_output_json=run_dir / "replay_full.json",
            replay_output_csv=run_dir / "replay_full.csv",
            replay_output_md=run_dir / "replay_full.md",
            replay_output_log=run_dir / "replay_full.log",
            workflow_log=run_dir / "workflow.log",
            skip_pdf=True,
        ),
    )
    return staged_cfg, audit_cfg, run_dir


# math: metrics(circuit) = raw + expanded_once + transpiled summaries under a fixed basis
def _circuit_cost_metrics(
    circuit: Any,
    *,
    basis_gates: Sequence[str],
    optimization_level: int,
) -> dict[str, Any]:
    try:
        from qiskit import transpile
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Heavy-prune circuit-cost metrics require qiskit. Original error: {type(exc).__name__}: {exc}") from exc

    def _bundle_metrics(qc: Any) -> dict[str, Any]:
        count_ops = {str(key): int(val) for key, val in dict(qc.count_ops()).items()}
        one_q = int(sum(int(count_ops.get(name, 0)) for name in ("rz", "sx", "x", "u", "u1", "u2", "u3")))
        return {
            "num_qubits": int(qc.num_qubits),
            "depth": int(qc.depth()),
            "size": int(qc.size()),
            "num_parameters": int(len(getattr(qc, "parameters", []))),
            "count_ops": count_ops,
            "cx_count": int(count_ops.get("cx", 0)),
            "one_q_count": int(one_q),
        }

    expanded = expand_pauli_evolution_once(circuit)
    transpiled = transpile(
        expanded,
        basis_gates=[str(x) for x in basis_gates],
        optimization_level=int(optimization_level),
    )
    return {
        "basis_gates": [str(x) for x in basis_gates],
        "optimization_level": int(optimization_level),
        "raw": _bundle_metrics(circuit),
        "expanded_once": _bundle_metrics(expanded),
        "transpiled": _bundle_metrics(transpiled),
    }


# math: stage_costs = circuit-cost summaries for warm/adapt/replay stage circuits when contexts exist
def _build_stage_circuit_costs(stage_result: StageExecutionResult, cfg: HeavyPruneConfig) -> dict[str, Any]:
    basis = tuple(str(x) for x in cfg.basis_gates)
    opt_level = int(cfg.transpile_optimization_level)
    costs: dict[str, Any] = {}

    warm_ctx = stage_result.warm_circuit_context
    if isinstance(warm_ctx, Mapping) and warm_ctx.get("ansatz") is not None:
        warm_circuit = ansatz_to_circuit(
            warm_ctx["ansatz"],
            np.asarray(warm_ctx.get("theta", []), dtype=float),
            num_qubits=int(warm_ctx.get("num_qubits", stage_result.nq_total)),
            reference_state=np.asarray(warm_ctx.get("reference_state"), dtype=complex),
        )
        costs["warm_start"] = _circuit_cost_metrics(warm_circuit, basis_gates=basis, optimization_level=opt_level)

    adapt_ctx = stage_result.adapt_circuit_context
    if isinstance(adapt_ctx, Mapping) and adapt_ctx.get("selected_ops") is not None:
        adapt_circuit = adapt_ops_to_circuit(
            list(adapt_ctx.get("selected_ops", [])),
            np.asarray(adapt_ctx.get("theta", []), dtype=float),
            num_qubits=int(adapt_ctx.get("num_qubits", stage_result.nq_total)),
            reference_state=np.asarray(adapt_ctx.get("reference_state"), dtype=complex),
        )
        costs["adapt_vqe"] = _circuit_cost_metrics(adapt_circuit, basis_gates=basis, optimization_level=opt_level)

    replay_ctx = stage_result.replay_circuit_context
    if isinstance(replay_ctx, Mapping) and replay_ctx.get("ansatz") is not None:
        replay_circuit = ansatz_to_circuit(
            replay_ctx["ansatz"],
            np.asarray(replay_ctx.get("theta", []), dtype=float),
            num_qubits=int(replay_ctx.get("num_qubits", stage_result.nq_total)),
            reference_state=np.asarray(replay_ctx.get("reference_state"), dtype=complex),
        )
        costs["conventional_replay"] = _circuit_cost_metrics(replay_circuit, basis_gates=basis, optimization_level=opt_level)

    return costs


# math: replay_cost(diag) = transpiled summary of the ablated replay ansatz at best_theta
def _replay_circuit_cost_from_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    basis_gates: Sequence[str],
    optimization_level: int,
) -> dict[str, Any]:
    ansatz = diagnostics.get("ansatz", None)
    best_theta = diagnostics.get("best_theta", None)
    reference_state = diagnostics.get("reference_state", None)
    num_qubits = diagnostics.get("num_qubits", None)
    if ansatz is None or best_theta is None or reference_state is None or num_qubits is None:
        raise ValueError("Replay diagnostics are missing ansatz, best_theta, reference_state, or num_qubits.")
    replay_circuit = ansatz_to_circuit(
        ansatz,
        np.asarray(best_theta, dtype=float),
        num_qubits=int(num_qubits),
        reference_state=np.asarray(reference_state, dtype=complex),
    )
    return _circuit_cost_metrics(
        replay_circuit,
        basis_gates=basis_gates,
        optimization_level=int(optimization_level),
    )


# math: priority(plan) = semantic precedence for duplicate keep-sets
def _plan_priority(*, ablation_id: str, ablation_kind: str) -> int:
    if str(ablation_id) == "full_replay_baseline":
        return 100
    priorities = {
        "single_operator_drop": 80,
        "cumulative_operator_drop": 70,
        "accepted_prefix_truncation": 60,
        "tail_truncation": 50,
        "baseline": 40,
    }
    return int(priorities.get(str(ablation_kind), 0))


# math: plans = baseline + accepted-prefix truncations + ranked weakest single/cumulative accepted-unit drops
def _build_ranked_prune_plans(
    *,
    handoff_payload: Mapping[str, Any],
    ranked_adapt_units: Sequence[Mapping[str, Any]],
    seed_prefix_depth: int,
    include_prefix_50: bool,
    weakest_single_count: int,
    weakest_cumulative_count: int,
) -> tuple[HeavyPrunePlan, ...]:
    adapt_block = handoff_payload.get("adapt_vqe", {})
    operators = adapt_block.get("operators", []) if isinstance(adapt_block, Mapping) else []
    n_ops = int(len(operators))
    if n_ops <= 0:
        raise ValueError("Handoff payload must include adapt_vqe.operators for heavy-prune screening.")

    seed_prefix_depth_int = int(max(0, min(int(seed_prefix_depth), n_ops)))
    accepted_ranked: list[dict[str, Any]] = []
    seen_final_order: set[int] = set()
    for entry in ranked_adapt_units:
        idx = int(entry.get("final_order_index", -1))
        if idx < int(seed_prefix_depth_int) or idx >= n_ops or idx in seen_final_order:
            continue
        seen_final_order.add(int(idx))
        accepted_ranked.append(dict(entry))

    accepted_final_order = sorted(int(entry["final_order_index"]) for entry in accepted_ranked)
    accepted_count = int(len(accepted_final_order))
    accepted_by_final_order = {int(entry["final_order_index"]): dict(entry) for entry in accepted_ranked}
    keep_order: list[tuple[int, ...]] = []
    plans_by_keep: dict[tuple[int, ...], tuple[int, HeavyPrunePlan]] = {}

    def _removed_entries_for_keep(keep_key: Sequence[int]) -> tuple[dict[str, Any], ...]:
        keep_set = {int(i) for i in keep_key}
        removed_final_indices = [idx for idx in accepted_final_order if int(idx) not in keep_set]
        return tuple(dict(accepted_by_final_order[idx]) for idx in removed_final_indices if int(idx) in accepted_by_final_order)

    def _append(
        *,
        ablation_id: str,
        ablation_kind: str,
        description: str,
        keep_operator_indices: Sequence[int],
        ranking_entries: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        keep_key = tuple(sorted(int(i) for i in keep_operator_indices))
        if len(keep_key) <= 0:
            return
        removed = tuple(i for i in range(n_ops) if i not in keep_key)
        removed_label = None if len(removed) != 1 else str(operators[int(removed[0])])
        ranking_payload = _removed_entries_for_keep(keep_key) if ranking_entries is None else tuple(dict(item) for item in ranking_entries)
        candidate = HeavyPrunePlan(
            ablation_id=str(ablation_id),
            ablation_kind=str(ablation_kind),
            description=str(description),
            keep_operator_indices=keep_key,
            removed_operator_indices=tuple(int(i) for i in removed),
            removed_label=removed_label,
            ranking_entries=tuple(dict(item) for item in ranking_payload),
        )
        candidate_priority = _plan_priority(ablation_id=str(ablation_id), ablation_kind=str(ablation_kind))
        existing = plans_by_keep.get(keep_key)
        if existing is None:
            keep_order.append(keep_key)
            plans_by_keep[keep_key] = (candidate_priority, candidate)
            return
        existing_priority, _ = existing
        if int(candidate_priority) > int(existing_priority):
            plans_by_keep[keep_key] = (candidate_priority, candidate)

    _append(
        ablation_id="full_replay_baseline",
        ablation_kind="baseline",
        description="Canonical staged matched-family replay baseline.",
        keep_operator_indices=tuple(range(n_ops)),
    )

    if accepted_count > 1:
        prefix_75_keep = int(max(1, math.floor(0.75 * accepted_count)))
        if prefix_75_keep < accepted_count:
            keep = tuple(range(seed_prefix_depth_int)) + tuple(accepted_final_order[:prefix_75_keep])
            _append(
                ablation_id="accepted_prefix_75",
                ablation_kind="accepted_prefix_truncation",
                description="Keep the full seed prefix and the first floor(0.75 * accepted_count) accepted operators in final order.",
                keep_operator_indices=keep,
                ranking_entries=None,
            )
        if bool(include_prefix_50):
            prefix_50_keep = int(max(1, math.floor(0.50 * accepted_count)))
            if prefix_50_keep < accepted_count:
                keep = tuple(range(seed_prefix_depth_int)) + tuple(accepted_final_order[:prefix_50_keep])
                _append(
                    ablation_id="accepted_prefix_50",
                    ablation_kind="accepted_prefix_truncation",
                    description="Keep the full seed prefix and the first floor(0.50 * accepted_count) accepted operators in final order.",
                    keep_operator_indices=keep,
                    ranking_entries=None,
                )
        tail_idx = int(max(accepted_final_order))
        _append(
            ablation_id="tail_drop_1",
            ablation_kind="tail_truncation",
            description="Drop the last accepted operator in final order while preserving the seed prefix.",
            keep_operator_indices=tuple(i for i in range(n_ops) if int(i) != int(tail_idx)),
            ranking_entries=None,
        )
        weakest_single_cap = int(max(1, min(int(weakest_single_count), accepted_count)))
        for rank, entry in enumerate(accepted_ranked[:weakest_single_cap], start=1):
            idx = int(entry["final_order_index"])
            _append(
                ablation_id=("drop_weakest_accepted" if rank == 1 else f"drop_ranked_weakest_{rank:02d}"),
                ablation_kind="single_operator_drop",
                description=(
                    "Drop the weakest accepted operator by audit removal-penalty ranking."
                    if rank == 1
                    else f"Drop the rank-{rank} weakest accepted operator by audit removal-penalty ranking."
                ),
                keep_operator_indices=tuple(i for i in range(n_ops) if int(i) != int(idx)),
                ranking_entries=(entry,),
            )
        weakest_cumulative_cap = int(max(0, min(int(weakest_cumulative_count), accepted_count)))
        for k in range(2, weakest_cumulative_cap + 1):
            removed_entries = tuple(dict(entry) for entry in accepted_ranked[:k])
            removed_indices = sorted(int(entry["final_order_index"]) for entry in removed_entries)
            _append(
                ablation_id=f"drop_ranked_cumulative_{k:02d}",
                ablation_kind="cumulative_operator_drop",
                description=f"Drop the bottom-{k} accepted operators by audit removal-penalty ranking.",
                keep_operator_indices=tuple(i for i in range(n_ops) if int(i) not in set(removed_indices)),
                ranking_entries=removed_entries,
            )

    return tuple(plans_by_keep[key][1] for key in keep_order)


# math: payload' = payload with adapt_vqe operators/theta filtered by keep indices and heavy-prune meta attached
def _build_heavy_prune_handoff_payload(
    handoff_payload: Mapping[str, Any],
    *,
    plan: HeavyPrunePlan,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(_jsonable(handoff_payload)))
    adapt_block = payload.get("adapt_vqe", None)
    if not isinstance(adapt_block, dict):
        raise ValueError("Handoff payload missing adapt_vqe block.")
    operators = list(adapt_block.get("operators", []))
    theta = list(adapt_block.get("optimal_point", []))
    if len(operators) != len(theta):
        raise ValueError("Handoff payload adapt_vqe operators/theta mismatch.")
    keep = [int(i) for i in plan.keep_operator_indices]
    adapt_block["operators"] = [str(operators[i]) for i in keep]
    adapt_block["optimal_point"] = [float(theta[i]) for i in keep]
    adapt_block["ansatz_depth"] = int(len(keep))
    adapt_block["num_parameters"] = int(len(keep))
    continuation = payload.get("continuation", None)
    if isinstance(continuation, dict):
        selected_meta = continuation.get("selected_generator_metadata", None)
        if isinstance(selected_meta, Sequence) and not isinstance(selected_meta, (str, bytes)) and len(selected_meta) == len(operators):
            continuation["selected_generator_metadata"] = [selected_meta[i] for i in keep]
        replay_contract = continuation.get("replay_contract", None)
        if isinstance(replay_contract, dict):
            replay_contract["adapt_depth"] = int(len(keep))
            replay_contract["derived_num_parameters"] = int(len(keep) * int(replay_contract.get("reps", 1)))
    meta = payload.get("meta", None)
    if not isinstance(meta, dict):
        meta = {}
        payload["meta"] = meta
    meta["heavy_prune_ablation"] = {
        "ablation_id": str(plan.ablation_id),
        "ablation_kind": str(plan.ablation_kind),
        "description": str(plan.description),
        "keep_operator_indices": [int(i) for i in plan.keep_operator_indices],
        "removed_operator_indices": [int(i) for i in plan.removed_operator_indices],
        "removed_label": plan.removed_label,
        "ranking_entries": [dict(item) for item in plan.ranking_entries],
    }
    return payload


# math: pareto(rows) = nondominated rows under (delta_abs, cx, depth) minimization
def _pareto_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for row in rows:
        delta = row.get("replay_delta_abs", None)
        cx = row.get("replay_transpiled_cx_count", None)
        depth = row.get("replay_transpiled_depth", None)
        if delta is None or cx is None or depth is None:
            continue
        valid.append(dict(row))

    front: list[dict[str, Any]] = []
    for i, row_i in enumerate(valid):
        a = (
            float(row_i["replay_delta_abs"]),
            int(row_i["replay_transpiled_cx_count"]),
            int(row_i["replay_transpiled_depth"]),
        )
        dominated = False
        for j, row_j in enumerate(valid):
            if i == j:
                continue
            b = (
                float(row_j["replay_delta_abs"]),
                int(row_j["replay_transpiled_cx_count"]),
                int(row_j["replay_transpiled_depth"]),
            )
            if (
                b[0] <= a[0]
                and b[1] <= a[1]
                and b[2] <= a[2]
                and (b[0] < a[0] or b[1] < a[1] or b[2] < a[2])
            ):
                dominated = True
                break
        if not dominated:
            front.append(dict(row_i))
    front.sort(
        key=lambda row: (
            float(row.get("replay_delta_abs", float("inf"))),
            int(row.get("replay_transpiled_cx_count", 10**9)),
            int(row.get("replay_transpiled_depth", 10**9)),
            str(row.get("ablation_id", "")),
        )
    )
    return front


# math: row = flat heavy-prune metrics for one baseline/ablation replay outcome
def _build_heavy_prune_row(
    *,
    cfg: HeavyPruneConfig,
    stage_metrics: Mapping[str, Any],
    audit_ctx: Mapping[str, Any],
    baseline_payload: Mapping[str, Any],
    replay_payload: Mapping[str, Any],
    plan: HeavyPrunePlan,
    replay_output_json: Path,
    handoff_input_json: Path,
    replay_cost: Mapping[str, Any] | None,
    baseline_replay_cost: Mapping[str, Any] | None,
) -> dict[str, Any]:
    baseline_vqe = baseline_payload.get("vqe", {}) if isinstance(baseline_payload, Mapping) else {}
    replay_vqe = replay_payload.get("vqe", {}) if isinstance(replay_payload, Mapping) else {}
    replay_exact = replay_payload.get("exact", {}) if isinstance(replay_payload, Mapping) else {}
    removed_indices = [int(i) for i in plan.removed_operator_indices]
    removed_labels = []
    ranking_entries = [dict(item) for item in plan.ranking_entries]
    if ranking_entries:
        removed_labels = [str(item.get("base_label", item.get("unit_label", ""))) for item in ranking_entries]
    row = {
        "tag": str(cfg.tag),
        "t": float(cfg.t),
        "u": float(cfg.u),
        "dv": float(cfg.dv),
        "omega0": float(cfg.omega0),
        "g_ep": float(cfg.g_ep),
        "ordering": str(cfg.ordering),
        "boundary": str(cfg.boundary),
        "warm_seed": int(cfg.warm_seed),
        "adapt_seed": int(cfg.adapt_seed),
        "final_seed": int(cfg.final_seed),
        "ablation_id": str(plan.ablation_id),
        "ablation_kind": str(plan.ablation_kind),
        "ablation_description": str(plan.description),
        "keep_operator_count": int(len(plan.keep_operator_indices)),
        "removed_operator_count": int(len(plan.removed_operator_indices)),
        "removed_operator_indices": list(removed_indices),
        "removed_labels": list(removed_labels),
        "seed_prefix_depth": int(audit_ctx.get("seed_prefix_depth", 0)),
        "accepted_insertion_count": int(audit_ctx.get("accepted_insertion_count", 0)),
        "final_adapt_depth": int(stage_metrics.get("final_adapt_depth", 0)),
        "warm_delta_abs": float(stage_metrics.get("warm_delta_abs", float("nan"))),
        "adapt_delta_abs": float(stage_metrics.get("adapt_delta_abs", float("nan"))),
        "baseline_replay_delta_abs": float(baseline_vqe.get("abs_delta_e", float("nan"))),
        "replay_delta_abs": float(replay_vqe.get("abs_delta_e", float("nan"))),
        "replay_relative_error_abs": float(replay_vqe.get("relative_error_abs", float("nan"))),
        "replay_energy": float(replay_vqe.get("energy", float("nan"))),
        "replay_exact_energy": float(replay_exact.get("E_exact_sector", float("nan"))),
        "replay_num_parameters": int(replay_vqe.get("num_parameters", 0)),
        "replay_runtime_s": float(replay_vqe.get("runtime_s", float("nan"))),
        "replay_stop_reason": str(replay_vqe.get("stop_reason", replay_vqe.get("message", ""))),
        "replay_gate_pass_1e2": bool(replay_vqe.get("gate_pass_1e2", False)),
        "replay_delta_vs_baseline_abs_delta": float(
            float(replay_vqe.get("abs_delta_e", float("nan"))) - float(baseline_vqe.get("abs_delta_e", float("nan")))
        ),
        "audit_json": str(audit_ctx.get("audit_json", "")),
        "audit_csv": str(audit_ctx.get("audit_csv", "")),
        "handoff_input_json": str(handoff_input_json),
        "replay_output_json": str(replay_output_json),
        "ranking_entries": ranking_entries,
    }
    if isinstance(replay_cost, Mapping):
        tx = replay_cost.get("transpiled", {}) if isinstance(replay_cost.get("transpiled", {}), Mapping) else {}
        row.update(
            {
                "replay_transpiled_depth": int(tx.get("depth", 0)),
                "replay_transpiled_size": int(tx.get("size", 0)),
                "replay_transpiled_cx_count": int(tx.get("cx_count", 0)),
                "replay_transpiled_one_q_count": int(tx.get("one_q_count", 0)),
            }
        )
    else:
        row.update(
            {
                "replay_transpiled_depth": None,
                "replay_transpiled_size": None,
                "replay_transpiled_cx_count": None,
                "replay_transpiled_one_q_count": None,
            }
        )
    if isinstance(baseline_replay_cost, Mapping):
        tx0 = baseline_replay_cost.get("transpiled", {}) if isinstance(baseline_replay_cost.get("transpiled", {}), Mapping) else {}
        row.update(
            {
                "baseline_replay_transpiled_depth": int(tx0.get("depth", 0)),
                "baseline_replay_transpiled_cx_count": int(tx0.get("cx_count", 0)),
                "replay_depth_delta_vs_baseline": (
                    None if row["replay_transpiled_depth"] is None else int(row["replay_transpiled_depth"]) - int(tx0.get("depth", 0))
                ),
                "replay_cx_delta_vs_baseline": (
                    None if row["replay_transpiled_cx_count"] is None else int(row["replay_transpiled_cx_count"]) - int(tx0.get("cx_count", 0))
                ),
            }
        )
    return row


# math: heavy_prune = one heavy staged baseline + ranked replay ablations + Pareto summary
def run_hh_l2_heavy_prune(cfg: HeavyPruneConfig) -> dict[str, Any]:
    staged_cfg, audit_cfg, run_dir = build_heavy_prune_staged_cfg(cfg)
    stage_result = run_stage_pipeline(staged_cfg)
    audit_ctx = logical_wf._build_audit_context(
        stage_result=stage_result,
        staged_cfg=staged_cfg,
        audit_cfg=audit_cfg,
    )
    baseline_payload = dict(stage_result.replay_payload)
    stage_metrics = logical_wf._summarize_stage_baseline(stage_result, last_k=3)
    handoff_payload = logical_wf._load_json(Path(staged_cfg.artifacts.handoff_json))
    plans = _build_ranked_prune_plans(
        handoff_payload=handoff_payload,
        ranked_adapt_units=audit_ctx.get("ranked_adapt_units", []),
        seed_prefix_depth=int(audit_ctx.get("seed_prefix_depth", 0)),
        include_prefix_50=bool(cfg.include_prefix_50),
        weakest_single_count=int(cfg.weakest_single_count),
        weakest_cumulative_count=int(cfg.weakest_cumulative_count),
    )
    stage_costs = _build_stage_circuit_costs(stage_result, cfg)
    baseline_replay_cost = stage_costs.get("conventional_replay", None)

    rows: list[dict[str, Any]] = []
    baseline_plan = next(plan for plan in plans if str(plan.ablation_id) == "full_replay_baseline")
    rows.append(
        _build_heavy_prune_row(
            cfg=cfg,
            stage_metrics=stage_metrics,
            audit_ctx=audit_ctx,
            baseline_payload=baseline_payload,
            replay_payload=baseline_payload,
            plan=baseline_plan,
            replay_output_json=Path(staged_cfg.artifacts.replay_output_json),
            handoff_input_json=Path(staged_cfg.artifacts.handoff_json),
            replay_cost=baseline_replay_cost,
            baseline_replay_cost=baseline_replay_cost,
        )
    )

    for plan in plans:
        if str(plan.ablation_id) == "full_replay_baseline":
            continue
        handoff_path = Path(run_dir) / f"handoff_{plan.ablation_id}.json"
        _write_json(handoff_path, _build_heavy_prune_handoff_payload(handoff_payload, plan=plan))
        replay_cfg = logical_wf._build_replay_cfg(
            staged_cfg,
            adapt_input_json=handoff_path,
            run_dir=run_dir,
            ablation_id=str(plan.ablation_id),
        )
        replay_diag: dict[str, Any] = {}
        replay_payload = replay_mod.run(replay_cfg, diagnostics_out=replay_diag)
        replay_cost = _replay_circuit_cost_from_diagnostics(
            replay_diag,
            basis_gates=tuple(str(x) for x in cfg.basis_gates),
            optimization_level=int(cfg.transpile_optimization_level),
        )
        rows.append(
            _build_heavy_prune_row(
                cfg=cfg,
                stage_metrics=stage_metrics,
                audit_ctx=audit_ctx,
                baseline_payload=baseline_payload,
                replay_payload=replay_payload,
                plan=plan,
                replay_output_json=Path(replay_cfg.output_json),
                handoff_input_json=handoff_path,
                replay_cost=replay_cost,
                baseline_replay_cost=baseline_replay_cost,
            )
        )

    pareto_rows = _pareto_rows(rows)
    payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_l2_heavy_prune",
        "model_family": "Hubbard-Holstein (HH)",
        "scope": {
            "local_only": True,
            "noiseless_only": True,
            "noise_enabled": False,
            "boundary_metrics_enabled": True,
            "pruning_level": "accepted_adapt_units_only",
            "native_gate_pruning_enabled": False,
        },
        "settings": asdict(cfg),
        "artifacts": {
            "output_json": str(cfg.output_json),
            "output_csv": str(cfg.output_csv),
            "run_root": str(cfg.run_root),
            "staged_workflow_json": str(staged_cfg.artifacts.output_json),
            "handoff_json": str(staged_cfg.artifacts.handoff_json),
            "audit_json": str(audit_cfg.output_json),
            "audit_csv": str(audit_cfg.output_csv),
        },
        "metric_semantics": {
            "ranking_entries": "Accepted ADAPT units ranked by (removal_penalty, delta_energy_from_previous, -final_order_index, unit_index).",
            "replay_delta_abs": "Final matched-family replay absolute energy error after ablation and replay refit.",
            "replay_transpiled_cx_count": "CX count after one PauliEvolution expansion pass and basis-gate transpilation.",
            "replay_transpiled_depth": "Circuit depth after one PauliEvolution expansion pass and basis-gate transpilation.",
            "pareto_front": "Nondominated rows under simultaneous minimization of (replay_delta_abs, replay_transpiled_cx_count, replay_transpiled_depth).",
        },
        "baseline": {
            "stage_metrics": dict(stage_metrics),
            "stage_circuit_costs": dict(stage_costs),
            "ranked_adapt_units": [dict(row) for row in audit_ctx.get("ranked_adapt_units", [])],
            "weakest_adapt_unit": (
                dict(audit_ctx["weakest_adapt_unit"])
                if isinstance(audit_ctx.get("weakest_adapt_unit"), Mapping)
                else None
            ),
            "seed_prefix_depth": int(audit_ctx.get("seed_prefix_depth", 0)),
            "accepted_insertion_count": int(audit_ctx.get("accepted_insertion_count", 0)),
        },
        "plans": [asdict(plan) for plan in plans],
        "rows": rows,
        "summary": {
            "plan_count": int(len(plans)),
            "pareto_front": pareto_rows,
            "best_delta_abs_row": (
                dict(min(rows, key=lambda row: float(row.get("replay_delta_abs", float("inf")))))
                if rows
                else None
            ),
            "lowest_cx_row": (
                dict(min(
                    [row for row in rows if row.get("replay_transpiled_cx_count") is not None],
                    key=lambda row: (int(row.get("replay_transpiled_cx_count", 10**9)), float(row.get("replay_delta_abs", float("inf")))),
                ))
                if any(row.get("replay_transpiled_cx_count") is not None for row in rows)
                else None
            ),
        },
    }
    _write_json(Path(cfg.output_json), payload)
    _write_csv(Path(cfg.output_csv), rows)
    return payload


# math: parse_basis("cx,rz,sx,x") = tuple[str, ...]
def _parse_basis_gates(raw: str) -> tuple[str, ...]:
    parts = [str(part).strip() for part in str(raw).split(",")]
    out = tuple(part for part in parts if part != "")
    if not out:
        raise ValueError("--basis-gates must contain at least one gate name.")
    return out


# math: cli = HeavyPruneConfig(args)
def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "HH L=2 heavy-circuit offline pruning workflow. Builds one open-boundary staged baseline, "
            "ranks accepted ADAPT units by audit contribution, runs replay ablations, and reports a Pareto set."
        )
    )
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--dv", type=float, default=0.0)
    parser.add_argument("--omega0", type=float, default=1.0)
    parser.add_argument("--g-ep", type=float, default=1.0, dest="g_ep")
    parser.add_argument("--warm-ansatz", choices=["hh_hva", "hh_hva_ptw"], default="hh_hva_ptw")
    parser.add_argument("--adapt-pool", type=str, default="paop_lf_std")
    parser.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    parser.add_argument("--boundary", choices=["open", "periodic"], default="open")
    parser.add_argument(
        "--adapt-continuation-mode",
        choices=["legacy", "phase1_v1", "phase2_v1", "phase3_v1"],
        default="phase3_v1",
    )
    parser.add_argument("--warm-seed", type=int, default=7)
    parser.add_argument("--adapt-seed", type=int, default=11)
    parser.add_argument("--final-seed", type=int, default=19)
    parser.add_argument("--include-prefix-50", action="store_true")
    parser.add_argument("--weakest-single-count", type=int, default=6)
    parser.add_argument("--weakest-cumulative-count", type=int, default=6)
    parser.add_argument("--warm-vqe-reps", type=int, default=None)
    parser.add_argument("--warm-vqe-restarts", type=int, default=None)
    parser.add_argument("--warm-vqe-maxiter", type=int, default=None)
    parser.add_argument("--adapt-max-depth", type=int, default=None)
    parser.add_argument("--adapt-drop-min-depth", type=int, default=None)
    parser.add_argument("--adapt-maxiter", type=int, default=None)
    parser.add_argument("--final-vqe-reps", type=int, default=None)
    parser.add_argument("--final-vqe-restarts", type=int, default=None)
    parser.add_argument("--final-vqe-maxiter", type=int, default=None)
    parser.add_argument("--basis-gates", type=str, default="cx,rz,sx,x")
    parser.add_argument("--transpile-optimization-level", type=int, default=1)
    parser.add_argument("--tag", type=str, default=_DEFAULT_TAG)
    parser.add_argument("--run-root", type=Path, default=_DEFAULT_RUN_ROOT)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=_DEFAULT_OUTPUT_CSV)
    return parser


# math: positive_int(raw) = int(raw) when raw > 0 else error
def parse_cli_args(argv: Sequence[str] | None = None) -> HeavyPruneConfig:
    args = build_cli_parser().parse_args(list(argv) if argv is not None else None)

    def _maybe_positive_int(raw: Any, *, flag: str) -> int | None:
        if raw is None:
            return None
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{flag} must be a positive integer when provided.")
        return value

    if int(args.weakest_single_count) <= 0:
        raise ValueError("--weakest-single-count must be a positive integer.")
    if int(args.weakest_cumulative_count) <= 0:
        raise ValueError("--weakest-cumulative-count must be a positive integer.")
    opt_level = int(args.transpile_optimization_level)
    if opt_level < 0 or opt_level > 3:
        raise ValueError("--transpile-optimization-level must be in {0,1,2,3}.")

    return HeavyPruneConfig(
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
        run_root=Path(args.run_root),
        tag=str(args.tag),
        t=float(args.t),
        u=float(args.u),
        dv=float(args.dv),
        omega0=float(args.omega0),
        g_ep=float(args.g_ep),
        warm_ansatz=str(args.warm_ansatz),
        adapt_pool=(None if args.adapt_pool is None else str(args.adapt_pool)),
        adapt_continuation_mode=str(args.adapt_continuation_mode),
        ordering=str(args.ordering),
        boundary=str(args.boundary),
        warm_seed=int(args.warm_seed),
        adapt_seed=int(args.adapt_seed),
        final_seed=int(args.final_seed),
        include_prefix_50=bool(args.include_prefix_50),
        weakest_single_count=int(args.weakest_single_count),
        weakest_cumulative_count=int(args.weakest_cumulative_count),
        warm_vqe_reps_override=_maybe_positive_int(args.warm_vqe_reps, flag="--warm-vqe-reps"),
        warm_vqe_restarts_override=_maybe_positive_int(args.warm_vqe_restarts, flag="--warm-vqe-restarts"),
        warm_vqe_maxiter_override=_maybe_positive_int(args.warm_vqe_maxiter, flag="--warm-vqe-maxiter"),
        adapt_max_depth_override=_maybe_positive_int(args.adapt_max_depth, flag="--adapt-max-depth"),
        adapt_drop_min_depth_override=_maybe_positive_int(args.adapt_drop_min_depth, flag="--adapt-drop-min-depth"),
        adapt_maxiter_override=_maybe_positive_int(args.adapt_maxiter, flag="--adapt-maxiter"),
        final_vqe_reps_override=_maybe_positive_int(args.final_vqe_reps, flag="--final-vqe-reps"),
        final_vqe_restarts_override=_maybe_positive_int(args.final_vqe_restarts, flag="--final-vqe-restarts"),
        final_vqe_maxiter_override=_maybe_positive_int(args.final_vqe_maxiter, flag="--final-vqe-maxiter"),
        basis_gates=_parse_basis_gates(str(args.basis_gates)),
        transpile_optimization_level=int(opt_level),
    )


# math: summary = compact artifact + Pareto headline
def format_compact_summary(payload: Mapping[str, Any]) -> list[str]:
    artifacts = payload.get("artifacts", {}) if isinstance(payload, Mapping) else {}
    summary = payload.get("summary", {}) if isinstance(payload, Mapping) else {}
    lines = [
        f"heavy_prune_json={artifacts.get('output_json', '') if isinstance(artifacts, Mapping) else ''}",
        f"heavy_prune_csv={artifacts.get('output_csv', '') if isinstance(artifacts, Mapping) else ''}",
    ]
    pareto = summary.get("pareto_front", []) if isinstance(summary, Mapping) else []
    if isinstance(pareto, Sequence) and not isinstance(pareto, (str, bytes)) and pareto:
        best = pareto[0]
        if isinstance(best, Mapping):
            lines.append(
                f"pareto_head={best.get('ablation_id')} delta={best.get('replay_delta_abs')} cx={best.get('replay_transpiled_cx_count')}"
            )
    return lines


__all__ = [
    "HeavyPruneConfig",
    "HeavyPrunePlan",
    "build_cli_parser",
    "build_heavy_prune_staged_cfg",
    "format_compact_summary",
    "parse_cli_args",
    "run_hh_l2_heavy_prune",
]
