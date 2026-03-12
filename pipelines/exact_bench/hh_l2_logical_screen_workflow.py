#!/usr/bin/env python3
"""Wrapper-only HH L=2 logical robustness screen over staged baseline + audit + replay ablations."""

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

from pipelines.exact_bench.hh_l2_stage_unit_audit_workflow import (
    AuditWorkflowConfig,
    build_audit_payload,
    build_locked_staged_hh_audit_config,
    build_stage_audit_specs,
    compute_stage_audit_rows,
    emit_audit_files,
)
from pipelines.hardcoded import hh_vqe_from_adapt_family as replay_mod
from pipelines.hardcoded.hh_staged_workflow import StageExecutionResult, StagedHHConfig, run_stage_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_JSON = REPO_ROOT / "artifacts" / "json" / "hh_l2_logical_screen.json"
_DEFAULT_OUTPUT_CSV = REPO_ROOT / "artifacts" / "json" / "hh_l2_logical_screen.csv"
_DEFAULT_RUN_ROOT = REPO_ROOT / "artifacts" / "json" / "hh_l2_logical_screen_runs"
_DEFAULT_SCREEN_TAG = "hh_l2_logical_screen"
_DEFAULT_LAST_K = 3
_DEFAULT_STRESS_POINT_COUNT = 2
_CORE_ABLATION_IDS = ("full_replay_baseline", "prefix_75", "tail_drop_1", "drop_weakest_accepted")


@dataclass(frozen=True)
class HamiltonianPoint:
    t: float
    u: float
    dv: float
    omega0: float
    g_ep: float

    def point_id(self) -> str:
        return f"U{_tag_float(self.u)}_g{_tag_float(self.g_ep)}_w{_tag_float(self.omega0)}"


@dataclass(frozen=True)
class SeedTriple:
    seed_index: int
    warm_seed: int
    adapt_seed: int
    final_seed: int


@dataclass(frozen=True)
class ReplayAblationPlan:
    ablation_id: str
    ablation_kind: str
    description: str
    keep_operator_indices: tuple[int, ...]
    removed_operator_indices: tuple[int, ...]
    removed_label: str | None = None


@dataclass(frozen=True)
class LogicalScreenConfig:
    output_json: Path
    output_csv: Path
    run_root: Path
    screen_tag: str
    points: tuple[HamiltonianPoint, ...]
    seed_count: int
    last_k: int
    stress_point_count: int
    warm_ansatz: str
    adapt_pool: str | None
    adapt_continuation_mode: str
    include_prefix_50: bool = False


@dataclass
class BaselineRunRecord:
    point: HamiltonianPoint
    seeds: SeedTriple
    staged_cfg: StagedHHConfig
    audit_cfg: AuditWorkflowConfig
    run_dir: Path
    handoff_json: Path
    baseline_payload: dict[str, Any]
    baseline_row: dict[str, Any]
    baseline_delta_abs: float
    weakest_adapt_unit: dict[str, Any] | None
    audit_extrema: dict[str, Any]


# math: timestamp_now = UTC ISO-8601 string

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# math: tag(x) = decimal string with '.' -> 'p' and '-' -> 'm'

def _tag_float(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


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


# math: representative_6 = fixed six-point HH sandbox subset

def _point_preset_representative_6(*, t: float, dv: float) -> tuple[HamiltonianPoint, ...]:
    return (
        HamiltonianPoint(t=t, u=0.5, dv=dv, g_ep=0.25, omega0=0.5),
        HamiltonianPoint(t=t, u=0.5, dv=dv, g_ep=2.0, omega0=0.5),
        HamiltonianPoint(t=t, u=8.0, dv=dv, g_ep=0.25, omega0=0.5),
        HamiltonianPoint(t=t, u=8.0, dv=dv, g_ep=2.0, omega0=0.5),
        HamiltonianPoint(t=t, u=4.0, dv=dv, g_ep=1.0, omega0=2.0),
        HamiltonianPoint(t=t, u=8.0, dv=dv, g_ep=2.0, omega0=2.0),
    )


# math: full_18 = U-grid × g-grid × omega-grid

def _point_preset_full_18(*, t: float, dv: float) -> tuple[HamiltonianPoint, ...]:
    out: list[HamiltonianPoint] = []
    for u in (0.5, 4.0, 8.0):
        for g_ep in (0.25, 1.0, 2.0):
            for omega0 in (0.5, 2.0):
                out.append(HamiltonianPoint(t=t, u=u, dv=dv, g_ep=g_ep, omega0=omega0))
    return tuple(out)


# math: parse_points(raw) = tuple((u_i, g_i, omega_i)) from ';'-separated 'u:g:omega' tokens

def _parse_points(raw: str, *, t: float, dv: float) -> tuple[HamiltonianPoint, ...]:
    points: list[HamiltonianPoint] = []
    for token in raw.split(";"):
        item = token.strip()
        if item == "":
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise ValueError(
                "Each custom point must use 'u:g_ep:omega0' format; "
                f"got '{item}'."
            )
        u_raw, g_raw, omega_raw = parts
        points.append(
            HamiltonianPoint(
                t=float(t),
                u=float(u_raw),
                dv=float(dv),
                g_ep=float(g_raw),
                omega0=float(omega_raw),
            )
        )
    if not points:
        raise ValueError("Custom point list was empty.")
    return tuple(points)


# math: points = custom(raw) if raw else preset(name)

def resolve_screen_points(
    *,
    point_preset: str,
    raw_points: str | None,
    t: float,
    dv: float,
) -> tuple[HamiltonianPoint, ...]:
    if raw_points is not None and str(raw_points).strip() != "":
        return _parse_points(str(raw_points), t=float(t), dv=float(dv))
    if str(point_preset) == "representative_6":
        return _point_preset_representative_6(t=float(t), dv=float(dv))
    if str(point_preset) == "full_18":
        return _point_preset_full_18(t=float(t), dv=float(dv))
    raise ValueError(f"Unknown point preset '{point_preset}'.")


# math: seeds(i) = (7, 11, 19) + 100 i

def _seed_triplet(seed_index: int) -> SeedTriple:
    base = 100 * int(seed_index)
    return SeedTriple(
        seed_index=int(seed_index + 1),
        warm_seed=int(7 + base),
        adapt_seed=int(11 + base),
        final_seed=int(19 + base),
    )


# math: last_k_tail(vals, k) = suffix of length min(k, len(vals)) with summary statistics

def _summarize_last_k_marginal_gains(history: Sequence[Mapping[str, Any]], *, last_k: int) -> dict[str, Any]:
    values = [
        float(row.get("delta_abs_drop_from_prev"))
        for row in history
        if isinstance(row, Mapping) and row.get("delta_abs_drop_from_prev") is not None
    ]
    k = max(1, int(last_k))
    tail = values[-k:]
    if not tail:
        return {
            "last_k": int(k),
            "tail_count": 0,
            "tail_values": [],
            "tail_mean": None,
            "tail_min": None,
            "tail_max": None,
            "tail_last": None,
            "tail_positive_fraction": None,
        }
    tail_arr = np.asarray(tail, dtype=float)
    return {
        "last_k": int(k),
        "tail_count": int(tail_arr.size),
        "tail_values": [float(x) for x in tail_arr.tolist()],
        "tail_mean": float(np.mean(tail_arr)),
        "tail_min": float(np.min(tail_arr)),
        "tail_max": float(np.max(tail_arr)),
        "tail_last": float(tail_arr[-1]),
        "tail_positive_fraction": float(np.mean(tail_arr > 0.0)),
    }


# math: stall_count = count(drop_low_signal or depth_rollback or delta_abs_drop_from_prev <= 0)

def _summarize_stall_signals(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    low_signal_count = 0
    rollback_count = 0
    optimizer_memory_reuse_count = 0
    nonpositive_gain_count = 0
    stall_step_count = 0
    for row in history:
        if not isinstance(row, Mapping):
            continue
        low_signal = bool(row.get("drop_low_signal", False))
        rollback = bool(row.get("depth_rollback", False))
        optimizer_memory_reused = bool(row.get("optimizer_memory_reused", False))
        delta_raw = row.get("delta_abs_drop_from_prev", None)
        nonpositive_gain = False
        if delta_raw is not None:
            delta_val = float(delta_raw)
            nonpositive_gain = bool(delta_val <= 0.0)
        low_signal_count += int(low_signal)
        rollback_count += int(rollback)
        optimizer_memory_reuse_count += int(optimizer_memory_reused)
        nonpositive_gain_count += int(nonpositive_gain)
        stall_step_count += int(low_signal or rollback or nonpositive_gain)
    return {
        "stall_step_count": int(stall_step_count),
        "drop_low_signal_count": int(low_signal_count),
        "depth_rollback_count": int(rollback_count),
        "optimizer_memory_reuse_count": int(optimizer_memory_reuse_count),
        "nonpositive_marginal_gain_count": int(nonpositive_gain_count),
    }


# math: stage_drops = (E_warm - E_adapt, E_adapt - E_replay, E_warm - E_replay)

def _summarize_stage_baseline(stage_result: StageExecutionResult, *, last_k: int) -> dict[str, Any]:
    warm_energy = float(stage_result.warm_payload.get("energy", float("nan")))
    warm_exact = float(stage_result.warm_payload.get("exact_filtered_energy", float("nan")))
    adapt_energy = float(stage_result.adapt_payload.get("energy", float("nan")))
    adapt_exact = float(stage_result.adapt_payload.get("exact_gs_energy", float("nan")))
    replay_vqe = stage_result.replay_payload.get("vqe", {})
    replay_exact = stage_result.replay_payload.get("exact", {})
    final_energy = float(replay_vqe.get("energy", float("nan")))
    final_exact = float(replay_exact.get("E_exact_sector", float("nan")))
    history_raw = stage_result.adapt_payload.get("history", [])
    history = [row for row in history_raw if isinstance(row, Mapping)] if isinstance(history_raw, Sequence) else []
    continuation = stage_result.adapt_payload.get("continuation", {})
    rescue_history = continuation.get("rescue_history", []) if isinstance(continuation, Mapping) else []
    marginal = _summarize_last_k_marginal_gains(history, last_k=int(last_k))
    stall = _summarize_stall_signals(history)
    return {
        "warm_energy": float(warm_energy),
        "warm_exact_energy": float(warm_exact),
        "warm_delta_abs": float(abs(warm_energy - warm_exact)),
        "adapt_energy": float(adapt_energy),
        "adapt_exact_energy": float(adapt_exact),
        "adapt_delta_abs": float(abs(adapt_energy - adapt_exact)),
        "replay_energy": float(final_energy),
        "replay_exact_energy": float(final_exact),
        "replay_delta_abs": float(abs(final_energy - final_exact)),
        "warm_to_adapt_energy_drop": float(warm_energy - adapt_energy),
        "adapt_to_replay_energy_drop": float(adapt_energy - final_energy),
        "warm_to_replay_energy_drop": float(warm_energy - final_energy),
        "accepted_operator_count": int(len(stage_result.adapt_payload.get("operators", []))),
        "history_length": int(len(history)),
        "rescue_count": int(len(rescue_history)) if isinstance(rescue_history, Sequence) else 0,
        **marginal,
        **stall,
    }


# math: extrema(rows) = min prefix/removal/gain scalars for a stage row set

def _stage_row_extrema(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "min_delta_energy_from_previous": None,
            "min_removal_penalty": None,
            "worst_energy_gain_per_depth": None,
            "worst_energy_gain_per_2q": None,
        }
    deltas = [float(row.get("delta_energy_from_previous")) for row in rows]
    removals = [float(row.get("removal_penalty")) for row in rows]
    depth_gains = [float(row.get("energy_gain_per_depth")) for row in rows]
    twoq_gains = [
        float(row.get("energy_gain_per_2q"))
        for row in rows
        if row.get("energy_gain_per_2q") is not None
    ]
    return {
        "min_delta_energy_from_previous": float(min(deltas)),
        "min_removal_penalty": float(min(removals)),
        "worst_energy_gain_per_depth": float(min(depth_gains)),
        "worst_energy_gain_per_2q": (float(min(twoq_gains)) if twoq_gains else None),
    }


# math: weakest = argmin(removal_penalty, delta_energy_from_previous, -final_order_index, unit_index)

def _select_weakest_adapt_unit(
    *,
    units_in_acceptance_order: Sequence[Any],
    adapt_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    candidates: list[tuple[tuple[float, float, int, int], dict[str, Any]]] = []
    for unit, row in zip(units_in_acceptance_order, adapt_rows):
        final_order_index = getattr(unit, "final_order_index", None)
        if final_order_index is None:
            continue
        entry = {
            "unit_index": int(getattr(unit, "unit_index")),
            "unit_label": str(getattr(unit, "unit_label")),
            "base_label": str(getattr(unit, "base_label")),
            "final_order_index": int(final_order_index),
            "insertion_position": int(getattr(unit, "insertion_position")),
            "removal_penalty": float(row.get("removal_penalty", float("nan"))),
            "delta_energy_from_previous": float(row.get("delta_energy_from_previous", float("nan"))),
        }
        key = (
            float(entry["removal_penalty"]),
            float(entry["delta_energy_from_previous"]),
            -int(entry["final_order_index"]),
            int(entry["unit_index"]),
        )
        candidates.append((key, entry))
    if len(candidates) <= 1:
        return None
    candidates.sort(key=lambda item: item[0])
    return dict(candidates[0][1])


# math: audit_ctx = emitted audit payload + stage rows + weakest adapt unit summary

def _build_audit_context(
    *,
    stage_result: StageExecutionResult,
    staged_cfg: StagedHHConfig,
    audit_cfg: AuditWorkflowConfig,
) -> dict[str, Any]:
    payload = build_audit_payload(audit_cfg, staged_cfg, stage_result)
    emit_audit_files(payload, audit_cfg)
    rows_by_stage: dict[str, list[dict[str, Any]]] = {}
    specs_by_stage: dict[str, Any] = {}
    for spec in build_stage_audit_specs(stage_result):
        rows, _stage_summary = compute_stage_audit_rows(spec, stage_result.hmat)
        rows_by_stage[str(spec.stage)] = [dict(row) for row in rows]
        specs_by_stage[str(spec.stage)] = spec
    adapt_spec = specs_by_stage.get("adapt_vqe", None)
    adapt_rows = rows_by_stage.get("adapt_vqe", [])
    weakest = None
    if adapt_spec is not None:
        weakest = _select_weakest_adapt_unit(
            units_in_acceptance_order=getattr(adapt_spec, "units_in_acceptance_order", []),
            adapt_rows=adapt_rows,
        )
    return {
        "payload": payload,
        "audit_json": Path(audit_cfg.output_json),
        "audit_csv": Path(audit_cfg.output_csv),
        "rows_by_stage": rows_by_stage,
        "weakest_adapt_unit": weakest,
        "audit_extrema": {
            "warm": _stage_row_extrema(rows_by_stage.get("warm_start", [])),
            "adapt": _stage_row_extrema(rows_by_stage.get("adapt_vqe", [])),
            "replay": _stage_row_extrema(rows_by_stage.get("conventional_replay", [])),
        },
    }


# math: cfg(point, seed_i) = locked audit config with unique artifacts and per-stage seeds

def build_screen_staged_cfg(
    screen_cfg: LogicalScreenConfig,
    *,
    point: HamiltonianPoint,
    seed_index: int,
) -> tuple[StagedHHConfig, AuditWorkflowConfig, SeedTriple, Path]:
    seeds = _seed_triplet(int(seed_index))
    stage_tag = f"{screen_cfg.screen_tag}_{point.point_id()}_seed{int(seeds.seed_index):02d}"
    run_dir = Path(screen_cfg.run_root) / stage_tag
    audit_cfg = AuditWorkflowConfig(
        output_json=run_dir / "audit.json",
        output_csv=run_dir / "audit.csv",
        stage_tag=stage_tag,
        t=float(point.t),
        u=float(point.u),
        dv=float(point.dv),
        omega0=float(point.omega0),
        g_ep=float(point.g_ep),
        warm_ansatz=str(screen_cfg.warm_ansatz),
        adapt_pool=(None if screen_cfg.adapt_pool is None else str(screen_cfg.adapt_pool)),
        adapt_continuation_mode=str(screen_cfg.adapt_continuation_mode),
    )
    staged_cfg = build_locked_staged_hh_audit_config(audit_cfg)
    staged_cfg = replace(
        staged_cfg,
        warm_start=replace(staged_cfg.warm_start, seed=int(seeds.warm_seed)),
        adapt=replace(staged_cfg.adapt, seed=int(seeds.adapt_seed)),
        replay=replace(staged_cfg.replay, seed=int(seeds.final_seed)),
        warm_checkpoint=replace(
            staged_cfg.warm_checkpoint,
            state_export_dir=run_dir / "state_export",
            state_export_prefix=stage_tag,
        ),
        artifacts=replace(
            staged_cfg.artifacts,
            tag=stage_tag,
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
    return staged_cfg, audit_cfg, seeds, run_dir


# math: load_json(path) = json.parse(text(path))

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# math: replay_cfg = staged_cfg.replay + ablation-specific artifact paths + handoff path

def _build_replay_cfg(
    staged_cfg: StagedHHConfig,
    *,
    adapt_input_json: Path,
    run_dir: Path,
    ablation_id: str,
) -> replay_mod.RunConfig:
    suffix = str(ablation_id)
    return replay_mod.RunConfig(
        adapt_input_json=Path(adapt_input_json),
        output_json=Path(run_dir) / f"replay_{suffix}.json",
        output_csv=Path(run_dir) / f"replay_{suffix}.csv",
        output_md=Path(run_dir) / f"replay_{suffix}.md",
        output_log=Path(run_dir) / f"replay_{suffix}.log",
        tag=f"{staged_cfg.artifacts.tag}_{suffix}",
        generator_family=str(staged_cfg.replay.generator_family),
        fallback_family=str(staged_cfg.replay.fallback_family),
        legacy_paop_key=str(staged_cfg.replay.legacy_paop_key),
        replay_seed_policy=str(staged_cfg.replay.replay_seed_policy),
        replay_continuation_mode=str(staged_cfg.replay.continuation_mode),
        L=int(staged_cfg.physics.L),
        t=float(staged_cfg.physics.t),
        u=float(staged_cfg.physics.u),
        dv=float(staged_cfg.physics.dv),
        omega0=float(staged_cfg.physics.omega0),
        g_ep=float(staged_cfg.physics.g_ep),
        n_ph_max=int(staged_cfg.physics.n_ph_max),
        boson_encoding=str(staged_cfg.physics.boson_encoding),
        ordering=str(staged_cfg.physics.ordering),
        boundary=str(staged_cfg.physics.boundary),
        sector_n_up=int(staged_cfg.physics.sector_n_up),
        sector_n_dn=int(staged_cfg.physics.sector_n_dn),
        reps=int(staged_cfg.replay.reps),
        restarts=int(staged_cfg.replay.restarts),
        maxiter=int(staged_cfg.replay.maxiter),
        method=str(staged_cfg.replay.method),
        seed=int(staged_cfg.replay.seed),
        energy_backend=str(staged_cfg.replay.energy_backend),
        progress_every_s=float(staged_cfg.replay.progress_every_s),
        wallclock_cap_s=int(staged_cfg.replay.wallclock_cap_s),
        paop_r=int(staged_cfg.replay.paop_r),
        paop_split_paulis=bool(staged_cfg.replay.paop_split_paulis),
        paop_prune_eps=float(staged_cfg.replay.paop_prune_eps),
        paop_normalization=str(staged_cfg.replay.paop_normalization),
        spsa_a=float(staged_cfg.replay.spsa_a),
        spsa_c=float(staged_cfg.replay.spsa_c),
        spsa_alpha=float(staged_cfg.replay.spsa_alpha),
        spsa_gamma=float(staged_cfg.replay.spsa_gamma),
        spsa_A=float(staged_cfg.replay.spsa_A),
        spsa_avg_last=int(staged_cfg.replay.spsa_avg_last),
        spsa_eval_repeats=int(staged_cfg.replay.spsa_eval_repeats),
        spsa_eval_agg=str(staged_cfg.replay.spsa_eval_agg),
        replay_freeze_fraction=float(staged_cfg.replay.replay_freeze_fraction),
        replay_unfreeze_fraction=float(staged_cfg.replay.replay_unfreeze_fraction),
        replay_full_fraction=float(staged_cfg.replay.replay_full_fraction),
        replay_qn_spsa_refresh_every=int(staged_cfg.replay.replay_qn_spsa_refresh_every),
        replay_qn_spsa_refresh_mode=str(staged_cfg.replay.replay_qn_spsa_refresh_mode),
        phase3_symmetry_mitigation_mode=str(staged_cfg.replay.phase3_symmetry_mitigation_mode),
    )


# math: ablation_plans = baseline + optional prefix cuts + tail drop + weakest accepted drop

def _build_replay_ablation_plans(
    *,
    handoff_payload: Mapping[str, Any],
    weakest_adapt_unit: Mapping[str, Any] | None,
    include_prefix_50: bool,
) -> tuple[ReplayAblationPlan, ...]:
    adapt_block = handoff_payload.get("adapt_vqe", {})
    operators = adapt_block.get("operators", []) if isinstance(adapt_block, Mapping) else []
    n_ops = int(len(operators))
    if n_ops <= 0:
        raise ValueError("Handoff payload must include adapt_vqe.operators for replay ablation screening.")
    plans: list[ReplayAblationPlan] = []
    seen_keeps: set[tuple[int, ...]] = set()

    def _append(plan: ReplayAblationPlan) -> None:
        keep_key = tuple(int(x) for x in plan.keep_operator_indices)
        if len(keep_key) <= 0:
            return
        if keep_key in seen_keeps:
            return
        seen_keeps.add(keep_key)
        plans.append(plan)

    full_keep = tuple(range(n_ops))
    _append(
        ReplayAblationPlan(
            ablation_id="full_replay_baseline",
            ablation_kind="baseline",
            description="Canonical staged matched-family replay baseline.",
            keep_operator_indices=full_keep,
            removed_operator_indices=tuple(),
            removed_label=None,
        )
    )
    prefix_75_n = max(1, int(math.floor(0.75 * n_ops)))
    if prefix_75_n < n_ops:
        _append(
            ReplayAblationPlan(
                ablation_id="prefix_75",
                ablation_kind="prefix_truncation",
                description="Keep the first floor(0.75 * depth) accepted operators in final order.",
                keep_operator_indices=tuple(range(prefix_75_n)),
                removed_operator_indices=tuple(range(prefix_75_n, n_ops)),
            )
        )
    if bool(include_prefix_50):
        prefix_50_n = max(1, int(math.floor(0.50 * n_ops)))
        if prefix_50_n < n_ops:
            _append(
                ReplayAblationPlan(
                    ablation_id="prefix_50",
                    ablation_kind="prefix_truncation",
                    description="Keep the first floor(0.50 * depth) accepted operators in final order.",
                    keep_operator_indices=tuple(range(prefix_50_n)),
                    removed_operator_indices=tuple(range(prefix_50_n, n_ops)),
                )
            )
    if n_ops > 1:
        _append(
            ReplayAblationPlan(
                ablation_id="tail_drop_1",
                ablation_kind="tail_truncation",
                description="Drop the last accepted operator in final order.",
                keep_operator_indices=tuple(range(n_ops - 1)),
                removed_operator_indices=(int(n_ops - 1),),
                removed_label=str(operators[int(n_ops - 1)]),
            )
        )
    weakest_index = None if weakest_adapt_unit is None else weakest_adapt_unit.get("final_order_index", None)
    if n_ops > 1 and weakest_index is not None:
        weakest_index_int = int(weakest_index)
        if 0 <= weakest_index_int < n_ops:
            keep = tuple(i for i in range(n_ops) if i != weakest_index_int)
            _append(
                ReplayAblationPlan(
                    ablation_id="drop_weakest_accepted",
                    ablation_kind="single_operator_drop",
                    description="Drop the weakest accepted operator by ADAPT audit removal penalty / prefix-gain ranking.",
                    keep_operator_indices=keep,
                    removed_operator_indices=(int(weakest_index_int),),
                    removed_label=str(operators[int(weakest_index_int)]),
                )
            )
    return tuple(plans)


# math: payload' = payload with adapt_vqe operators/theta filtered by keep indices and ablation meta attached

def _build_ablated_handoff_payload(
    handoff_payload: Mapping[str, Any],
    *,
    plan: ReplayAblationPlan,
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
    meta["logical_screen_ablation"] = {
        "ablation_id": str(plan.ablation_id),
        "ablation_kind": str(plan.ablation_kind),
        "description": str(plan.description),
        "keep_operator_indices": [int(i) for i in plan.keep_operator_indices],
        "removed_operator_indices": [int(i) for i in plan.removed_operator_indices],
        "removed_label": plan.removed_label,
    }
    return payload


# math: replay(plan) = run existing matched-family replay from ablated handoff payload

def _run_replay_ablation(
    *,
    staged_cfg: StagedHHConfig,
    run_dir: Path,
    handoff_payload: Mapping[str, Any],
    plan: ReplayAblationPlan,
) -> tuple[dict[str, Any], Path]:
    handoff_path = Path(run_dir) / f"handoff_{plan.ablation_id}.json"
    _write_json(handoff_path, _build_ablated_handoff_payload(handoff_payload, plan=plan))
    replay_cfg = _build_replay_cfg(staged_cfg, adapt_input_json=handoff_path, run_dir=run_dir, ablation_id=plan.ablation_id)
    return replay_mod.run(replay_cfg), Path(replay_cfg.output_json)


# math: row = flat screen metrics for one (point, seed, ablation)

def _build_screen_row(
    *,
    point: HamiltonianPoint,
    seeds: SeedTriple,
    stage_metrics: Mapping[str, Any],
    audit_ctx: Mapping[str, Any],
    baseline_payload: Mapping[str, Any],
    replay_payload: Mapping[str, Any],
    plan: ReplayAblationPlan,
    replay_output_json: Path,
    handoff_input_json: Path,
) -> dict[str, Any]:
    baseline_vqe = baseline_payload.get("vqe", {}) if isinstance(baseline_payload, Mapping) else {}
    baseline_exact = baseline_payload.get("exact", {}) if isinstance(baseline_payload, Mapping) else {}
    replay_vqe = replay_payload.get("vqe", {}) if isinstance(replay_payload, Mapping) else {}
    replay_exact = replay_payload.get("exact", {}) if isinstance(replay_payload, Mapping) else {}
    weakest = audit_ctx.get("weakest_adapt_unit", None) if isinstance(audit_ctx, Mapping) else None
    audit_extrema = audit_ctx.get("audit_extrema", {}) if isinstance(audit_ctx, Mapping) else {}
    row = {
        "point_id": str(point.point_id()),
        "t": float(point.t),
        "u": float(point.u),
        "dv": float(point.dv),
        "omega0": float(point.omega0),
        "g_ep": float(point.g_ep),
        "seed_index": int(seeds.seed_index),
        "warm_seed": int(seeds.warm_seed),
        "adapt_seed": int(seeds.adapt_seed),
        "final_seed": int(seeds.final_seed),
        "ablation_id": str(plan.ablation_id),
        "ablation_kind": str(plan.ablation_kind),
        "ablation_description": str(plan.description),
        "ablation_selected_operator_count": int(len(plan.keep_operator_indices)),
        "ablation_removed_operator_count": int(len(plan.removed_operator_indices)),
        "ablation_removed_operator_indices": [int(i) for i in plan.removed_operator_indices],
        "ablation_removed_label": plan.removed_label,
        "accepted_operator_count": int(stage_metrics.get("accepted_operator_count", 0)),
        "warm_energy": float(stage_metrics.get("warm_energy", float("nan"))),
        "warm_exact_energy": float(stage_metrics.get("warm_exact_energy", float("nan"))),
        "warm_delta_abs": float(stage_metrics.get("warm_delta_abs", float("nan"))),
        "adapt_energy": float(stage_metrics.get("adapt_energy", float("nan"))),
        "adapt_exact_energy": float(stage_metrics.get("adapt_exact_energy", float("nan"))),
        "adapt_delta_abs": float(stage_metrics.get("adapt_delta_abs", float("nan"))),
        "baseline_replay_energy": float(baseline_vqe.get("energy", float("nan"))),
        "baseline_replay_exact_energy": float(baseline_exact.get("E_exact_sector", float("nan"))),
        "baseline_replay_delta_abs": float(baseline_vqe.get("abs_delta_e", float("nan"))),
        "replay_energy": float(replay_vqe.get("energy", float("nan"))),
        "replay_exact_energy": float(replay_exact.get("E_exact_sector", float("nan"))),
        "replay_delta_abs": float(replay_vqe.get("abs_delta_e", float("nan"))),
        "replay_relative_error_abs": float(replay_vqe.get("relative_error_abs", float("nan"))),
        "replay_num_parameters": int(replay_vqe.get("num_parameters", 0)),
        "replay_runtime_s": float(replay_vqe.get("runtime_s", float("nan"))),
        "replay_stop_reason": str(replay_vqe.get("stop_reason", replay_vqe.get("message", ""))),
        "replay_gate_pass_1e2": bool(replay_vqe.get("gate_pass_1e2", False)),
        "replay_delta_vs_baseline_energy": float(
            float(replay_vqe.get("energy", float("nan"))) - float(baseline_vqe.get("energy", float("nan")))
        ),
        "replay_delta_vs_baseline_abs_delta": float(
            float(replay_vqe.get("abs_delta_e", float("nan"))) - float(baseline_vqe.get("abs_delta_e", float("nan")))
        ),
        "warm_to_adapt_energy_drop": float(stage_metrics.get("warm_to_adapt_energy_drop", float("nan"))),
        "adapt_to_replay_energy_drop": float(stage_metrics.get("adapt_to_replay_energy_drop", float("nan"))),
        "warm_to_replay_energy_drop": float(stage_metrics.get("warm_to_replay_energy_drop", float("nan"))),
        "history_length": int(stage_metrics.get("history_length", 0)),
        "last_k": int(stage_metrics.get("last_k", _DEFAULT_LAST_K)),
        "last_k_tail_count": int(stage_metrics.get("tail_count", 0)),
        "last_k_marginal_gains": list(stage_metrics.get("tail_values", [])),
        "last_k_marginal_gain_mean": stage_metrics.get("tail_mean", None),
        "last_k_marginal_gain_min": stage_metrics.get("tail_min", None),
        "last_k_marginal_gain_max": stage_metrics.get("tail_max", None),
        "last_k_marginal_gain_last": stage_metrics.get("tail_last", None),
        "last_k_positive_fraction": stage_metrics.get("tail_positive_fraction", None),
        "rescue_count": int(stage_metrics.get("rescue_count", 0)),
        "stall_step_count": int(stage_metrics.get("stall_step_count", 0)),
        "drop_low_signal_count": int(stage_metrics.get("drop_low_signal_count", 0)),
        "depth_rollback_count": int(stage_metrics.get("depth_rollback_count", 0)),
        "optimizer_memory_reuse_count": int(stage_metrics.get("optimizer_memory_reuse_count", 0)),
        "nonpositive_marginal_gain_count": int(stage_metrics.get("nonpositive_marginal_gain_count", 0)),
        "weakest_adapt_unit_label": (None if weakest is None else str(weakest.get("unit_label"))),
        "weakest_adapt_final_order_index": (None if weakest is None else int(weakest.get("final_order_index"))),
        "weakest_adapt_removal_penalty": (None if weakest is None else float(weakest.get("removal_penalty"))),
        "weakest_adapt_delta_energy_from_previous": (
            None if weakest is None else float(weakest.get("delta_energy_from_previous"))
        ),
        "adapt_min_delta_energy_from_previous": audit_extrema.get("adapt", {}).get("min_delta_energy_from_previous"),
        "adapt_min_removal_penalty": audit_extrema.get("adapt", {}).get("min_removal_penalty"),
        "replay_min_delta_energy_from_previous": audit_extrema.get("replay", {}).get("min_delta_energy_from_previous"),
        "replay_min_removal_penalty": audit_extrema.get("replay", {}).get("min_removal_penalty"),
        "audit_json": str(audit_ctx.get("audit_json", "")),
        "audit_csv": str(audit_ctx.get("audit_csv", "")),
        "handoff_input_json": str(handoff_input_json),
        "replay_output_json": str(replay_output_json),
        "seed_role": "unassigned",
        "seed_rank_within_point": None,
    }
    return row


# math: baseline_rank(point) = sort seeds by baseline replay delta_abs then seed_index

def _annotate_seed_roles(rows: list[dict[str, Any]]) -> None:
    baseline_by_point: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("ablation_id")) != "full_replay_baseline":
            continue
        baseline_by_point.setdefault(str(row.get("point_id")), []).append(row)
    for point_rows in baseline_by_point.values():
        point_rows.sort(key=lambda row: (float(row.get("baseline_replay_delta_abs", float("inf"))), int(row.get("seed_index", 0))))
        if not point_rows:
            continue
        median_idx = len(point_rows) // 2
        for idx, row in enumerate(point_rows):
            row["seed_rank_within_point"] = int(idx + 1)
            if idx == 0:
                row["seed_role"] = "best"
            elif idx == median_idx:
                row["seed_role"] = "median"
            elif idx == len(point_rows) - 1:
                row["seed_role"] = "worst"
            else:
                row["seed_role"] = "interior"


# math: median_record(point) = middle baseline seed by sorted replay delta_abs

def _select_median_records(records: Sequence[BaselineRunRecord]) -> tuple[BaselineRunRecord, ...]:
    grouped: dict[str, list[BaselineRunRecord]] = {}
    for record in records:
        grouped.setdefault(record.point.point_id(), []).append(record)
    selected: list[BaselineRunRecord] = []
    for point_id in sorted(grouped.keys()):
        point_records = grouped[point_id]
        point_records.sort(key=lambda rec: (float(rec.baseline_delta_abs), int(rec.seeds.seed_index)))
        selected.append(point_records[len(point_records) // 2])
    return tuple(selected)


# math: summary(point) = baseline spread + rescue/stall maxima; stress points = top-N by spread/stall tuple

def _build_screen_summary(rows: Sequence[Mapping[str, Any]], *, stress_point_count: int) -> dict[str, Any]:
    baseline_rows = [row for row in rows if str(row.get("ablation_id")) == "full_replay_baseline"]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in baseline_rows:
        grouped.setdefault(str(row.get("point_id")), []).append(row)
    per_point: list[dict[str, Any]] = []
    for point_id in sorted(grouped.keys()):
        point_rows = sorted(grouped[point_id], key=lambda row: int(row.get("seed_index", 0)))
        delta_vals = np.asarray([float(row.get("baseline_replay_delta_abs", float("nan"))) for row in point_rows], dtype=float)
        rescue_vals = [int(row.get("rescue_count", 0)) for row in point_rows]
        stall_vals = [int(row.get("stall_step_count", 0)) for row in point_rows]
        entry = {
            "point_id": str(point_id),
            "seed_count": int(len(point_rows)),
            "baseline_replay_delta_abs_mean": float(np.mean(delta_vals)),
            "baseline_replay_delta_abs_std": float(np.std(delta_vals)),
            "baseline_replay_delta_abs_min": float(np.min(delta_vals)),
            "baseline_replay_delta_abs_max": float(np.max(delta_vals)),
            "max_rescue_count": int(max(rescue_vals) if rescue_vals else 0),
            "max_stall_step_count": int(max(stall_vals) if stall_vals else 0),
            "best_seed_index": int(min(point_rows, key=lambda row: (float(row.get("baseline_replay_delta_abs", float("inf"))), int(row.get("seed_index", 0)))).get("seed_index", 0)),
            "median_seed_index": int(sorted(point_rows, key=lambda row: (float(row.get("baseline_replay_delta_abs", float("inf"))), int(row.get("seed_index", 0))))[len(point_rows) // 2].get("seed_index", 0)),
            "worst_seed_index": int(max(point_rows, key=lambda row: (float(row.get("baseline_replay_delta_abs", float("-inf"))), -int(row.get("seed_index", 0)))).get("seed_index", 0)),
        }
        per_point.append(entry)
    stress_points = sorted(
        per_point,
        key=lambda entry: (
            -float(entry.get("baseline_replay_delta_abs_std", 0.0)),
            -int(entry.get("max_stall_step_count", 0)),
            -int(entry.get("max_rescue_count", 0)),
            str(entry.get("point_id", "")),
        ),
    )[: max(0, int(stress_point_count))]
    return {
        "baseline_points": per_point,
        "stress_points": stress_points,
        "median_seed_ablation_scope": "core_only",
    }


# math: screen = baselines(all seeds) + ablations(median seed only) + consolidated artifact rows

def run_hh_l2_logical_screen(screen_cfg: LogicalScreenConfig) -> dict[str, Any]:
    baseline_records: list[BaselineRunRecord] = []
    rows: list[dict[str, Any]] = []

    for point in screen_cfg.points:
        for seed_index in range(int(screen_cfg.seed_count)):
            staged_cfg, audit_cfg, seeds, run_dir = build_screen_staged_cfg(screen_cfg, point=point, seed_index=seed_index)
            stage_result = run_stage_pipeline(staged_cfg)
            audit_ctx = _build_audit_context(stage_result=stage_result, staged_cfg=staged_cfg, audit_cfg=audit_cfg)
            baseline_payload = dict(stage_result.replay_payload)
            stage_metrics = _summarize_stage_baseline(stage_result, last_k=int(screen_cfg.last_k))
            baseline_plan = ReplayAblationPlan(
                ablation_id="full_replay_baseline",
                ablation_kind="baseline",
                description="Canonical staged matched-family replay baseline.",
                keep_operator_indices=tuple(range(int(stage_metrics.get("accepted_operator_count", 0)))),
                removed_operator_indices=tuple(),
            )
            baseline_row = _build_screen_row(
                point=point,
                seeds=seeds,
                stage_metrics=stage_metrics,
                audit_ctx=audit_ctx,
                baseline_payload=baseline_payload,
                replay_payload=baseline_payload,
                plan=baseline_plan,
                replay_output_json=Path(staged_cfg.artifacts.replay_output_json),
                handoff_input_json=Path(staged_cfg.artifacts.handoff_json),
            )
            rows.append(baseline_row)
            baseline_records.append(
                BaselineRunRecord(
                    point=point,
                    seeds=seeds,
                    staged_cfg=staged_cfg,
                    audit_cfg=audit_cfg,
                    run_dir=Path(run_dir),
                    handoff_json=Path(staged_cfg.artifacts.handoff_json),
                    baseline_payload=baseline_payload,
                    baseline_row=baseline_row,
                    baseline_delta_abs=float(baseline_row["baseline_replay_delta_abs"]),
                    weakest_adapt_unit=(
                        dict(audit_ctx["weakest_adapt_unit"])
                        if isinstance(audit_ctx.get("weakest_adapt_unit"), Mapping)
                        else None
                    ),
                    audit_extrema=dict(audit_ctx.get("audit_extrema", {})),
                )
            )

    _annotate_seed_roles(rows)
    median_records = _select_median_records(baseline_records)
    for record in median_records:
        handoff_payload = _load_json(record.handoff_json)
        stage_metrics = {
            key: record.baseline_row[key]
            for key in (
                "warm_energy",
                "warm_exact_energy",
                "warm_delta_abs",
                "adapt_energy",
                "adapt_exact_energy",
                "adapt_delta_abs",
                "warm_to_adapt_energy_drop",
                "adapt_to_replay_energy_drop",
                "warm_to_replay_energy_drop",
                "accepted_operator_count",
                "history_length",
                "last_k",
                "last_k_tail_count",
                "last_k_marginal_gains",
                "last_k_marginal_gain_mean",
                "last_k_marginal_gain_min",
                "last_k_marginal_gain_max",
                "last_k_marginal_gain_last",
                "last_k_positive_fraction",
                "rescue_count",
                "stall_step_count",
                "drop_low_signal_count",
                "depth_rollback_count",
                "optimizer_memory_reuse_count",
                "nonpositive_marginal_gain_count",
            )
            if key in record.baseline_row
        }
        stage_metrics["tail_values"] = list(record.baseline_row.get("last_k_marginal_gains", []))
        stage_metrics["tail_count"] = int(record.baseline_row.get("last_k_tail_count", 0))
        stage_metrics["tail_mean"] = record.baseline_row.get("last_k_marginal_gain_mean", None)
        stage_metrics["tail_min"] = record.baseline_row.get("last_k_marginal_gain_min", None)
        stage_metrics["tail_max"] = record.baseline_row.get("last_k_marginal_gain_max", None)
        stage_metrics["tail_last"] = record.baseline_row.get("last_k_marginal_gain_last", None)
        audit_ctx = {
            "audit_json": Path(record.audit_cfg.output_json),
            "audit_csv": Path(record.audit_cfg.output_csv),
            "weakest_adapt_unit": record.weakest_adapt_unit,
            "audit_extrema": dict(record.audit_extrema),
        }
        plans = _build_replay_ablation_plans(
            handoff_payload=handoff_payload,
            weakest_adapt_unit=record.weakest_adapt_unit,
            include_prefix_50=bool(screen_cfg.include_prefix_50),
        )
        for plan in plans:
            if str(plan.ablation_id) == "full_replay_baseline":
                continue
            replay_payload, replay_output_json = _run_replay_ablation(
                staged_cfg=record.staged_cfg,
                run_dir=record.run_dir,
                handoff_payload=handoff_payload,
                plan=plan,
            )
            ablation_row = _build_screen_row(
                point=record.point,
                seeds=record.seeds,
                stage_metrics=stage_metrics,
                audit_ctx=audit_ctx,
                baseline_payload=record.baseline_payload,
                replay_payload=replay_payload,
                plan=plan,
                replay_output_json=replay_output_json,
                handoff_input_json=record.run_dir / f"handoff_{plan.ablation_id}.json",
            )
            ablation_row["seed_role"] = str(record.baseline_row.get("seed_role", "median"))
            ablation_row["seed_rank_within_point"] = record.baseline_row.get("seed_rank_within_point")
            rows.append(ablation_row)

    summary = _build_screen_summary(rows, stress_point_count=int(screen_cfg.stress_point_count))
    payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_l2_logical_screen",
        "model_family": "Hubbard-Holstein (HH)",
        "screen_scope": {
            "local_only": True,
            "noiseless_only": True,
            "noise_enabled": False,
            "patch_selection_enabled": False,
            "mitigation_enabled": False,
            "boundary_metrics_enabled": False,
            "median_seed_ablation_only": True,
        },
        "settings": {
            "screen_tag": str(screen_cfg.screen_tag),
            "seed_count": int(screen_cfg.seed_count),
            "last_k_marginal_gain_window": int(screen_cfg.last_k),
            "stress_point_count": int(screen_cfg.stress_point_count),
            "warm_ansatz": str(screen_cfg.warm_ansatz),
            "adapt_pool": screen_cfg.adapt_pool,
            "adapt_continuation_mode": str(screen_cfg.adapt_continuation_mode),
            "include_prefix_50": bool(screen_cfg.include_prefix_50),
            "core_ablation_ids": list(_CORE_ABLATION_IDS),
            "points": [asdict(point) for point in screen_cfg.points],
        },
        "metric_semantics": {
            "accepted_operator_count": "Final accepted ADAPT operator count from the canonical staged baseline.",
            "last_k_marginal_gains": "Tail of ADAPT delta_abs_drop_from_prev values from accepted iterations only.",
            "stall_step_count": "Count of ADAPT history rows with drop_low_signal or depth_rollback or nonpositive marginal gain.",
            "weakest_adapt_removal_penalty": "Smallest fixed-parameter ADAPT audit removal penalty chosen for single-operator replay ablation.",
            "replay_delta_vs_baseline_energy": "Ablated matched-family replay energy minus canonical staged replay energy at the same Hamiltonian point and seed.",
            "replay_delta_vs_baseline_abs_delta": "Ablated matched-family replay abs_delta_e minus canonical staged replay abs_delta_e at the same Hamiltonian point and seed.",
        },
        "artifacts": {
            "output_json": str(screen_cfg.output_json),
            "output_csv": str(screen_cfg.output_csv),
            "run_root": str(screen_cfg.run_root),
        },
        "rows": rows,
        "summary": summary,
    }
    _write_json(Path(screen_cfg.output_json), payload)
    _write_csv(Path(screen_cfg.output_csv), rows)
    return payload


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wrapper-only HH L=2 logical robustness screen. Runs canonical staged baselines, "
            "emits stage-unit audits, selects the median seed per point, and runs reduced matched-family replay ablations."
        )
    )
    parser.add_argument("--point-preset", choices=["representative_6", "full_18"], default="representative_6")
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help="Optional custom point list using 'u:g_ep:omega0;u:g_ep:omega0'. Overrides --point-preset.",
    )
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--dv", type=float, default=0.0)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--last-k", type=int, default=_DEFAULT_LAST_K)
    parser.add_argument("--stress-point-count", type=int, default=_DEFAULT_STRESS_POINT_COUNT)
    parser.add_argument("--warm-ansatz", choices=["hh_hva", "hh_hva_ptw"], default="hh_hva_ptw")
    parser.add_argument("--adapt-pool", type=str, default="paop_lf_std")
    parser.add_argument(
        "--adapt-continuation-mode",
        choices=["legacy", "phase1_v1", "phase2_v1", "phase3_v1"],
        default="phase3_v1",
    )
    parser.add_argument("--include-prefix-50", action="store_true")
    parser.add_argument("--screen-tag", type=str, default=_DEFAULT_SCREEN_TAG)
    parser.add_argument("--run-root", type=Path, default=_DEFAULT_RUN_ROOT)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=_DEFAULT_OUTPUT_CSV)
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> LogicalScreenConfig:
    args = build_cli_parser().parse_args(list(argv) if argv is not None else None)
    seed_count = int(args.seed_count)
    if seed_count <= 0 or (seed_count % 2) == 0:
        raise ValueError("--seed-count must be a positive odd integer so a unique median seed exists.")
    points = resolve_screen_points(
        point_preset=str(args.point_preset),
        raw_points=(None if args.points is None else str(args.points)),
        t=float(args.t),
        dv=float(args.dv),
    )
    return LogicalScreenConfig(
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
        run_root=Path(args.run_root),
        screen_tag=str(args.screen_tag),
        points=tuple(points),
        seed_count=int(seed_count),
        last_k=int(args.last_k),
        stress_point_count=int(args.stress_point_count),
        warm_ansatz=str(args.warm_ansatz),
        adapt_pool=(None if args.adapt_pool is None else str(args.adapt_pool)),
        adapt_continuation_mode=str(args.adapt_continuation_mode),
        include_prefix_50=bool(args.include_prefix_50),
    )


def format_compact_summary(payload: Mapping[str, Any]) -> list[str]:
    artifacts = payload.get("artifacts", {}) if isinstance(payload, Mapping) else {}
    summary = payload.get("summary", {}) if isinstance(payload, Mapping) else {}
    lines = [
        f"screen_json={artifacts.get('output_json', '') if isinstance(artifacts, Mapping) else ''}",
        f"screen_csv={artifacts.get('output_csv', '') if isinstance(artifacts, Mapping) else ''}",
    ]
    stress_points = summary.get("stress_points", []) if isinstance(summary, Mapping) else []
    if isinstance(stress_points, Sequence) and not isinstance(stress_points, (str, bytes)):
        for idx, entry in enumerate(stress_points[:2], start=1):
            if isinstance(entry, Mapping):
                lines.append(
                    f"stress_point_{idx}={entry.get('point_id')} std={entry.get('baseline_replay_delta_abs_std')}"
                )
    return lines


__all__ = [
    "BaselineRunRecord",
    "HamiltonianPoint",
    "LogicalScreenConfig",
    "ReplayAblationPlan",
    "SeedTriple",
    "build_cli_parser",
    "build_screen_staged_cfg",
    "format_compact_summary",
    "parse_cli_args",
    "resolve_screen_points",
    "run_hh_l2_logical_screen",
]