#!/usr/bin/env python3
"""Shared staged Hubbard-Holstein noiseless workflow orchestration.

This module keeps the stage-chain logic out of the existing monolithic
entrypoints. It reuses the production hardcoded primitives instead of
re-implementing warm-start VQE, ADAPT, replay, or time dynamics.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.reports.pdf_utils import (
    HAS_MATPLOTLIB,
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)
from docs.reports.qiskit_circuit_report import (
    adapt_ops_to_circuit,
    ansatz_to_circuit,
    build_time_dynamics_circuit,
    compute_time_dynamics_proxy_cost,
    is_cfqm_dynamics_method,
    render_circuit_page,
    render_circuit_summary_page,
    time_dynamics_circuitization_reason,
    transpile_circuit_metrics,
    warn_time_dynamics_circuit_semantics,
)
from pipelines.hardcoded import adapt_pipeline as adapt_mod
from pipelines.hardcoded import hh_vqe_from_adapt_family as replay_mod
from pipelines.hardcoded import hubbard_pipeline as hc_pipeline

_PREPARED_STATE = replay_mod._PREPARED_STATE
_REFERENCE_STATE = replay_mod._REFERENCE_STATE
from pipelines.hardcoded.handoff_state_bundle import (
    HandoffStateBundleConfig,
    write_handoff_state_bundle,
)
from src.quantum.drives_time_potential import (
    build_gaussian_sinusoid_density_drive,
    reference_method_name,
)
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinLayerwiseAnsatz,
    HubbardHolsteinPhysicalTermwiseAnsatz,
    HubbardHolsteinTermwiseAnsatz,
    exact_ground_energy_sector_hh,
)


_ALLOWED_NOISELESS_METHODS = ("suzuki2", "cfqm4", "cfqm6", "piecewise_exact")


@dataclass(frozen=True)
class PhysicsConfig:
    L: int
    t: float
    u: float
    dv: float
    omega0: float
    g_ep: float
    n_ph_max: int
    boson_encoding: str
    ordering: str
    boundary: str
    sector_n_up: int
    sector_n_dn: int


@dataclass(frozen=True)
class WarmStartConfig:
    ansatz_name: str
    reps: int
    restarts: int
    maxiter: int
    method: str
    seed: int
    progress_every_s: float
    energy_backend: str
    spsa_a: float
    spsa_c: float
    spsa_alpha: float
    spsa_gamma: float
    spsa_A: float
    spsa_avg_last: int
    spsa_eval_repeats: int
    spsa_eval_agg: str


@dataclass(frozen=True)
class SeedRefineConfig:
    family: str | None
    reps: int
    maxiter: int
    optimizer: str


@dataclass(frozen=True)
class AdaptConfig:
    pool: str | None
    continuation_mode: str
    max_depth: int
    maxiter: int
    eps_grad: float
    eps_energy: float
    drop_floor: float | None
    drop_patience: int | None
    drop_min_depth: int | None
    grad_floor: float | None
    seed: int
    inner_optimizer: str
    allow_repeats: bool
    finite_angle_fallback: bool
    finite_angle: float
    finite_angle_min_improvement: float
    disable_hh_seed: bool
    reopt_policy: str
    window_size: int
    window_topk: int
    full_refit_every: int
    final_full_refit: bool
    beam_live_branches: int
    beam_children_per_parent: int | None
    beam_terminated_keep: int | None
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str
    spsa_a: float
    spsa_c: float
    spsa_alpha: float
    spsa_gamma: float
    spsa_A: float
    spsa_avg_last: int
    spsa_eval_repeats: int
    spsa_eval_agg: str
    spsa_callback_every: int
    spsa_progress_every_s: float
    phase1_lambda_F: float
    phase1_lambda_compile: float
    phase1_lambda_measure: float
    phase1_lambda_leak: float
    phase1_score_z_alpha: float
    phase1_probe_max_positions: int
    phase1_plateau_patience: int
    phase1_trough_margin_ratio: float
    phase1_prune_enabled: bool
    phase1_prune_fraction: float
    phase1_prune_max_candidates: int
    phase1_prune_max_regression: float
    phase3_motif_source_json: Path | None
    phase3_symmetry_mitigation_mode: str
    phase3_enable_rescue: bool
    phase3_lifetime_cost_mode: str
    phase3_runtime_split_mode: str


@dataclass(frozen=True)
class ReplayConfig:
    enabled: bool
    generator_family: str
    fallback_family: str
    legacy_paop_key: str
    replay_seed_policy: str
    continuation_mode: str
    reps: int
    restarts: int
    maxiter: int
    method: str
    seed: int
    energy_backend: str
    progress_every_s: float
    wallclock_cap_s: int
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str
    spsa_a: float
    spsa_c: float
    spsa_alpha: float
    spsa_gamma: float
    spsa_A: float
    spsa_avg_last: int
    spsa_eval_repeats: int
    spsa_eval_agg: str
    replay_freeze_fraction: float
    replay_unfreeze_fraction: float
    replay_full_fraction: float
    replay_qn_spsa_refresh_every: int
    replay_qn_spsa_refresh_mode: str
    phase3_symmetry_mitigation_mode: str


@dataclass(frozen=True)
class DynamicsConfig:
    enabled: bool
    methods: tuple[str, ...]
    t_final: float
    num_times: int
    trotter_steps: int
    exact_steps_multiplier: int
    fidelity_subspace_energy_tol: float
    cfqm_stage_exp: str
    cfqm_coeff_drop_abs_tol: float
    cfqm_normalize: bool
    enable_drive: bool
    drive_A: float
    drive_omega: float
    drive_tbar: float
    drive_phi: float
    drive_pattern: str
    drive_custom_s: str | None
    drive_include_identity: bool
    drive_time_sampling: str
    drive_t0: float


@dataclass(frozen=True)
class FixedFinalStateConfig:
    json_path: Path
    strict_match: bool


@dataclass(frozen=True)
class CircuitMetricConfig:
    backend_name: str | None
    use_fake_backend: bool
    optimization_level: int
    seed_transpiler: int


@dataclass(frozen=True)
class WarmCheckpointConfig:
    stop_energy: float | None
    stop_delta_abs: float | None
    state_export_dir: Path
    state_export_prefix: str
    resume_from_warm_checkpoint: Path | None
    handoff_from_warm_checkpoint: Path | None


@dataclass(frozen=True)
class ArtifactConfig:
    tag: str
    output_json: Path
    output_pdf: Path
    handoff_json: Path
    warm_checkpoint_json: Path
    warm_cutover_json: Path
    replay_output_json: Path
    replay_output_csv: Path
    replay_output_md: Path
    replay_output_log: Path
    workflow_log: Path
    skip_pdf: bool


@dataclass(frozen=True)
class GateConfig:
    ecut_1: float
    ecut_2: float


@dataclass(frozen=True)
class StagedHHConfig:
    physics: PhysicsConfig
    warm_start: WarmStartConfig
    seed_refine: SeedRefineConfig
    adapt: AdaptConfig
    replay: ReplayConfig
    dynamics: DynamicsConfig
    fixed_final_state: FixedFinalStateConfig | None
    circuit_metrics: CircuitMetricConfig
    warm_checkpoint: WarmCheckpointConfig
    artifacts: ArtifactConfig
    gates: GateConfig
    smoke_test_intentionally_weak: bool = False
    default_provenance: dict[str, str] = field(default_factory=dict)
    external_noise_handle: dict[str, Any] | None = None


@dataclass
class StageExecutionResult:
    h_poly: Any
    hmat: np.ndarray
    ordered_labels_exyz: list[str]
    coeff_map_exyz: dict[str, complex]
    nq_total: int
    psi_hf: np.ndarray
    psi_warm: np.ndarray
    psi_adapt: np.ndarray
    psi_final: np.ndarray
    warm_payload: dict[str, Any]
    adapt_payload: dict[str, Any]
    replay_payload: dict[str, Any]
    psi_seed_refine: np.ndarray | None = None
    seed_refine_payload: dict[str, Any] | None = None
    fixed_final_state_import: dict[str, Any] | None = None
    warm_circuit_context: dict[str, Any] | None = None
    adapt_circuit_context: dict[str, Any] | None = None
    replay_circuit_context: dict[str, Any] | None = None


"""
Δ_rel(E, E_ref) = |E - E_ref| / max(|E_ref|, 1e-14)
"""
def _relative_error_abs(value: float, reference: float) -> float:
    return float(abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-14))


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _append_workflow_log(cfg: StagedHHConfig, event: str, **fields: Any) -> None:
    payload = {
        "ts_utc": _now_utc(),
        "event": str(event),
        **_jsonable(fields),
    }
    log_path = Path(cfg.artifacts.workflow_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _handoff_bundle_cfg(cfg: StagedHHConfig) -> HandoffStateBundleConfig:
    return HandoffStateBundleConfig(
        L=int(cfg.physics.L),
        t=float(cfg.physics.t),
        U=float(cfg.physics.u),
        dv=float(cfg.physics.dv),
        omega0=float(cfg.physics.omega0),
        g_ep=float(cfg.physics.g_ep),
        n_ph_max=int(cfg.physics.n_ph_max),
        boson_encoding=str(cfg.physics.boson_encoding),
        ordering=str(cfg.physics.ordering),
        boundary=str(cfg.physics.boundary),
        sector_n_up=int(cfg.physics.sector_n_up),
        sector_n_dn=int(cfg.physics.sector_n_dn),
    )


def _seed_refine_state_json_path(cfg: StagedHHConfig) -> Path:
    return Path(cfg.warm_checkpoint.state_export_dir) / (
        f"{cfg.warm_checkpoint.state_export_prefix}_seed_refine_state.json"
    )


def _build_seed_refine_run_cfg(cfg: StagedHHConfig) -> replay_mod.RunConfig:
    return replay_mod.RunConfig(
        adapt_input_json=Path(cfg.artifacts.warm_cutover_json),
        output_json=Path(cfg.artifacts.output_json).with_name(f"{cfg.artifacts.tag}_seed_refine.json"),
        output_csv=Path(cfg.artifacts.replay_output_csv).with_name(f"{cfg.artifacts.tag}_seed_refine.csv"),
        output_md=Path(cfg.artifacts.replay_output_md).with_name(f"{cfg.artifacts.tag}_seed_refine.md"),
        output_log=Path(cfg.artifacts.replay_output_log).with_name(f"{cfg.artifacts.tag}_seed_refine.log"),
        tag=f"{cfg.artifacts.tag}_seed_refine",
        generator_family=str(cfg.seed_refine.family or ""),
        fallback_family="full_meta",
        legacy_paop_key=str(cfg.replay.legacy_paop_key),
        replay_seed_policy="auto",
        replay_continuation_mode=None,
        L=int(cfg.physics.L),
        t=float(cfg.physics.t),
        u=float(cfg.physics.u),
        dv=float(cfg.physics.dv),
        omega0=float(cfg.physics.omega0),
        g_ep=float(cfg.physics.g_ep),
        n_ph_max=int(cfg.physics.n_ph_max),
        boson_encoding=str(cfg.physics.boson_encoding),
        ordering=str(cfg.physics.ordering),
        boundary=str(cfg.physics.boundary),
        sector_n_up=int(cfg.physics.sector_n_up),
        sector_n_dn=int(cfg.physics.sector_n_dn),
        reps=int(cfg.seed_refine.reps),
        restarts=int(cfg.replay.restarts),
        maxiter=int(cfg.seed_refine.maxiter),
        method=str(cfg.seed_refine.optimizer),
        seed=int(cfg.replay.seed),
        energy_backend=str(cfg.replay.energy_backend),
        progress_every_s=float(cfg.replay.progress_every_s),
        wallclock_cap_s=int(cfg.replay.wallclock_cap_s),
        paop_r=int(cfg.replay.paop_r),
        paop_split_paulis=bool(cfg.replay.paop_split_paulis),
        paop_prune_eps=float(cfg.replay.paop_prune_eps),
        paop_normalization=str(cfg.replay.paop_normalization),
        spsa_a=float(cfg.replay.spsa_a),
        spsa_c=float(cfg.replay.spsa_c),
        spsa_alpha=float(cfg.replay.spsa_alpha),
        spsa_gamma=float(cfg.replay.spsa_gamma),
        spsa_A=float(cfg.replay.spsa_A),
        spsa_avg_last=int(cfg.replay.spsa_avg_last),
        spsa_eval_repeats=int(cfg.replay.spsa_eval_repeats),
        spsa_eval_agg=str(cfg.replay.spsa_eval_agg),
        replay_freeze_fraction=float(cfg.replay.replay_freeze_fraction),
        replay_unfreeze_fraction=float(cfg.replay.replay_unfreeze_fraction),
        replay_full_fraction=float(cfg.replay.replay_full_fraction),
        replay_qn_spsa_refresh_every=int(cfg.replay.replay_qn_spsa_refresh_every),
        replay_qn_spsa_refresh_mode=str(cfg.replay.replay_qn_spsa_refresh_mode),
        phase3_symmetry_mitigation_mode=str(cfg.replay.phase3_symmetry_mitigation_mode),
    )


def _build_seed_provenance(
    cfg: StagedHHConfig,
    seed_refine_payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if cfg.seed_refine.family is None:
        return None
    pool_meta = seed_refine_payload.get("pool", {}) if isinstance(seed_refine_payload, Mapping) else {}
    motif_families_raw: list[str] = []
    if isinstance(pool_meta, Mapping):
        raw_many = pool_meta.get("motif_families", None)
        if isinstance(raw_many, Sequence) and not isinstance(raw_many, (str, bytes)):
            motif_families_raw.extend(str(x) for x in raw_many if str(x).strip())
        raw_one = pool_meta.get("motif_family", None)
        if raw_one is not None:
            motif_families_raw.append(str(raw_one))
    motif_families = sorted({str(x) for x in motif_families_raw if str(x).strip()})
    family_kind = (
        str(pool_meta.get("family_kind"))
        if isinstance(pool_meta, Mapping) and pool_meta.get("family_kind") is not None
        else "explicit_family"
    )
    return {
        "warm_ansatz": str(cfg.warm_start.ansatz_name),
        "refine_family": str(cfg.seed_refine.family),
        "refine_family_kind": str(family_kind),
        "refine_paop_motif_families": list(motif_families),
        "refine_reps": int(cfg.seed_refine.reps),
    }


def _warm_stop_required(cfg: StagedHHConfig) -> bool:
    return (
        cfg.warm_checkpoint.stop_energy is not None
        or cfg.warm_checkpoint.stop_delta_abs is not None
    )


def _warm_stop_status(
    cfg: StagedHHConfig,
    *,
    energy: float,
    exact_filtered_energy: float,
) -> dict[str, Any]:
    delta_abs = float(abs(float(energy) - float(exact_filtered_energy)))
    hit_energy = (
        cfg.warm_checkpoint.stop_energy is not None
        and float(energy) <= float(cfg.warm_checkpoint.stop_energy)
    )
    hit_delta = (
        cfg.warm_checkpoint.stop_delta_abs is not None
        and float(delta_abs) <= float(cfg.warm_checkpoint.stop_delta_abs)
    )
    reasons: list[str] = []
    if bool(hit_energy):
        reasons.append("warm_stop_energy")
    if bool(hit_delta):
        reasons.append("warm_stop_delta_abs")
    return {
        "triggered": bool(reasons),
        "reason": ("+".join(reasons) if reasons else None),
        "delta_abs": float(delta_abs),
        "hit_energy": bool(hit_energy),
        "hit_delta": bool(hit_delta),
        "stop_energy": (
            None if cfg.warm_checkpoint.stop_energy is None else float(cfg.warm_checkpoint.stop_energy)
        ),
        "stop_delta_abs": (
            None
            if cfg.warm_checkpoint.stop_delta_abs is None
            else float(cfg.warm_checkpoint.stop_delta_abs)
        ),
    }


def _write_warm_checkpoint_bundle(
    cfg: StagedHHConfig,
    *,
    path: Path,
    psi_state: np.ndarray,
    energy: float,
    exact_filtered_energy: float,
    theta: Sequence[float] | None,
    role: str,
    cutoff_status: Mapping[str, Any],
    event_meta: Mapping[str, Any] | None = None,
    source_json: Path | None = None,
) -> None:
    meta = {
        "pipeline": "hh_staged_noiseless",
        "workflow_tag": str(cfg.artifacts.tag),
        "stage": "warm_start_hva",
        "checkpoint_role": str(role),
        "warm_ansatz": str(cfg.warm_start.ansatz_name),
        "optimizer_method": str(cfg.warm_start.method),
        "warm_stop_energy": cutoff_status.get("stop_energy"),
        "warm_stop_delta_abs": cutoff_status.get("stop_delta_abs"),
        "cutoff_triggered": bool(cutoff_status.get("triggered", False)),
        "cutoff_reason": cutoff_status.get("reason"),
        "delta_abs": float(cutoff_status.get("delta_abs", float("nan"))),
    }
    if theta is not None:
        meta["warm_optimal_point"] = [float(x) for x in theta]
    if source_json is not None:
        meta["source_json"] = str(source_json)
    if isinstance(event_meta, Mapping):
        for key in ("restart_index", "restarts_total", "nfev_so_far", "nfev_restart", "elapsed_s", "elapsed_restart_s"):
            if key in event_meta:
                meta[key] = _jsonable(event_meta.get(key))
    write_handoff_state_bundle(
        path=Path(path),
        psi_state=np.asarray(psi_state, dtype=complex).reshape(-1),
        cfg=_handoff_bundle_cfg(cfg),
        source="warm_vqe",
        exact_energy=float(exact_filtered_energy),
        energy=float(energy),
        delta_E_abs=float(cutoff_status.get("delta_abs", abs(float(energy) - float(exact_filtered_energy)))),
        relative_error_abs=float(_relative_error_abs(float(energy), float(exact_filtered_energy))),
        meta=meta,
        handoff_state_kind="prepared_state",
    )


def _expected_adapt_ref_args(cfg: StagedHHConfig) -> SimpleNamespace:
    return SimpleNamespace(
        L=int(cfg.physics.L),
        problem="hh",
        ordering=str(cfg.physics.ordering),
        boundary=str(cfg.physics.boundary),
        t=float(cfg.physics.t),
        u=float(cfg.physics.u),
        dv=float(cfg.physics.dv),
        omega0=float(cfg.physics.omega0),
        g_ep=float(cfg.physics.g_ep),
        n_ph_max=int(cfg.physics.n_ph_max),
        boson_encoding=str(cfg.physics.boson_encoding),
    )


def _resolve_checkpoint_energy(payload: Mapping[str, Any]) -> float | None:
    for block_name, field_name in (("adapt_vqe", "energy"), ("vqe", "energy")):
        block = payload.get(block_name, {})
        if not isinstance(block, Mapping):
            continue
        raw = block.get(field_name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except Exception:
            continue
        if np.isfinite(value):
            return float(value)
    return None


def _read_json_object(json_path: Path, *, label: str) -> dict[str, Any]:
    if not json_path.exists():
        raise FileNotFoundError(f"{label} not found: {json_path}")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a top-level object: {json_path}")
    return raw


def _resolve_json_reference(source_json: Path, raw_candidate: Any) -> Path | None:
    if raw_candidate in {None, "", "none"}:
        return None
    candidate = Path(raw_candidate).expanduser()
    search_paths: list[Path]
    if candidate.is_absolute():
        search_paths = [candidate]
    else:
        search_paths = [
            source_json.parent / candidate,
            REPO_ROOT / candidate,
            candidate,
        ]
    for path in search_paths:
        if path.exists():
            return path.resolve()
    return None


def _resolve_fixed_final_state_payload(
    source_json: Path,
) -> tuple[Path, dict[str, Any], str | None]:
    raw_payload = _read_json_object(source_json, label="Fixed final-state JSON")
    if isinstance(raw_payload.get("initial_state"), Mapping):
        return source_json, raw_payload, None

    candidate_refs: list[tuple[str, Any]] = []
    artifacts = raw_payload.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        intermediate = artifacts.get("intermediate", {})
        if isinstance(intermediate, Mapping):
            candidate_refs.append(
                ("artifacts.intermediate.adapt_handoff_json", intermediate.get("adapt_handoff_json"))
            )
            candidate_refs.append(
                ("artifacts.intermediate.fixed_final_state_json", intermediate.get("fixed_final_state_json"))
            )
    stage_pipeline = raw_payload.get("stage_pipeline", {})
    if isinstance(stage_pipeline, Mapping):
        adapt_block = stage_pipeline.get("adapt_vqe", {})
        if isinstance(adapt_block, Mapping):
            candidate_refs.append(
                ("stage_pipeline.adapt_vqe.handoff_json", adapt_block.get("handoff_json"))
            )
        fixed_import = stage_pipeline.get("fixed_final_state_import", {})
        if isinstance(fixed_import, Mapping):
            candidate_refs.append(
                (
                    "stage_pipeline.fixed_final_state_import.source_json",
                    fixed_import.get("source_json"),
                )
            )

    attempted_refs: list[str] = []
    for ref_label, raw_candidate in candidate_refs:
        if raw_candidate in {None, "", "none"}:
            continue
        attempted_refs.append(f"{ref_label}={raw_candidate}")
        candidate_json = _resolve_json_reference(source_json, raw_candidate)
        if candidate_json is None:
            continue
        candidate_payload = _read_json_object(
            candidate_json,
            label=f"Resolved fixed final-state JSON via {ref_label}",
        )
        if isinstance(candidate_payload.get("initial_state"), Mapping):
            return candidate_json, candidate_payload, ref_label

    attempted_text = (
        ""
        if not attempted_refs
        else " Attempted reusable references: " + ", ".join(attempted_refs) + "."
    )
    raise ValueError(
        "Fixed final-state JSON {} is missing top-level initial_state.amplitudes_qn_to_q0, "
        "and no reusable staged handoff bundle could be resolved. Pass a handoff-style bundle "
        "directly or a staged workflow JSON with artifacts.intermediate.adapt_handoff_json."
        "{}".format(source_json, attempted_text)
    )


def _build_fixed_final_state_import(
    cfg: StagedHHConfig,
    *,
    source_json: Path,
    raw_payload: Mapping[str, Any],
    nq_total: int,
    requested_source_json: Path | None = None,
    resolved_via: str | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    psi_final, import_meta = adapt_mod._load_adapt_initial_state(Path(source_json), int(nq_total))
    exact_energy = adapt_mod._resolve_exact_energy_from_payload(raw_payload)
    if exact_energy is None:
        raise ValueError(
            f"Fixed final-state JSON {source_json} is missing an exact-energy field needed for dynamics baselines."
        )
    imported_energy = _resolve_checkpoint_energy(raw_payload)
    if imported_energy is None:
        raise ValueError(
            f"Fixed final-state JSON {source_json} is missing an adapt/vqe energy for provenance."
        )
    mismatches = adapt_mod._validate_adapt_ref_metadata_for_exact_reuse(
        adapt_settings=import_meta.get("settings", {}),
        args=_expected_adapt_ref_args(cfg),
        is_hh=True,
    )
    if mismatches and bool(cfg.fixed_final_state and cfg.fixed_final_state.strict_match):
        raise ValueError(
            "fixed final-state settings mismatch: "
            + "; ".join(str(item) for item in mismatches)
        )

    delta_abs = float(abs(float(imported_energy) - float(exact_energy)))
    fixed_import = {
        "source_json": str(requested_source_json if requested_source_json is not None else source_json),
        "resolved_json": str(source_json),
        "resolved_via": resolved_via,
        "strict_match": bool(cfg.fixed_final_state and cfg.fixed_final_state.strict_match),
        "mismatches": [str(item) for item in mismatches],
        "initial_state_source": import_meta.get("initial_state_source"),
        "energy": float(imported_energy),
        "exact_energy": float(exact_energy),
        "delta_abs": float(delta_abs),
        "relative_error_abs": float(_relative_error_abs(float(imported_energy), float(exact_energy))),
    }
    warm_payload = {
        "ansatz": str(cfg.warm_start.ansatz_name),
        "energy": float(imported_energy),
        "exact_filtered_energy": float(exact_energy),
        "optimizer_method": str(cfg.warm_start.method),
        "message": "skipped_fixed_final_state_json",
        "checkpoint_json_latest": str(source_json),
        "checkpoint_json_used": str(source_json),
        "cutoff_triggered": False,
        "cutoff_reason": None,
        "resumed_from_checkpoint": False,
        "skipped": True,
        "skip_reason": "fixed_final_state_json",
    }
    adapt_payload = {
        "energy": float(imported_energy),
        "exact_gs_energy": float(exact_energy),
        "ansatz_depth": 0,
        "pool_type": str(cfg.adapt.pool or cfg.adapt.continuation_mode),
        "continuation_mode": str(cfg.adapt.continuation_mode),
        "stop_reason": "skipped_fixed_final_state_json",
        "adapt_ref_json": str(source_json),
        "initial_state_source": "fixed_final_state_json",
        "skipped": True,
        "skip_reason": "fixed_final_state_json",
    }
    replay_payload = {
        "generator_family": {
            "requested": "fixed_final_state_json",
            "resolved": "fixed_final_state_json",
            "resolution_source": "fixed_final_state_json",
        },
        "seed_baseline": {"theta_policy": "fixed_final_state_json"},
        "exact": {"E_exact_sector": float(exact_energy)},
        "vqe": {
            "energy": float(imported_energy),
            "abs_delta_e": float(delta_abs),
            "relative_error_abs": float(_relative_error_abs(float(imported_energy), float(exact_energy))),
            "stop_reason": "skipped_fixed_final_state_json",
        },
        "replay_contract": {"continuation_mode": str(cfg.replay.continuation_mode)},
        "best_state": {
            "amplitudes_qn_to_q0": hc_pipeline._state_to_amplitudes_qn_to_q0(
                np.asarray(psi_final, dtype=complex).reshape(-1)
            )
        },
        "skipped": True,
        "skip_reason": "fixed_final_state_json",
        "source_json": str(source_json),
    }
    return (
        np.asarray(psi_final, dtype=complex).reshape(-1),
        fixed_import,
        warm_payload,
        adapt_payload,
        replay_payload,
    )


def _write_fixed_final_state_sidecars(
    cfg: StagedHHConfig,
    *,
    psi_final: np.ndarray,
    fixed_import: Mapping[str, Any],
    replay_payload: Mapping[str, Any],
) -> None:
    cfg.artifacts.handoff_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.replay_output_json.parent.mkdir(parents=True, exist_ok=True)
    write_handoff_state_bundle(
        path=cfg.artifacts.handoff_json,
        psi_state=np.asarray(psi_final, dtype=complex).reshape(-1),
        cfg=_handoff_bundle_cfg(cfg),
        source="fixed_final_state_json",
        exact_energy=float(fixed_import.get("exact_energy", float("nan"))),
        energy=float(fixed_import.get("energy", float("nan"))),
        delta_E_abs=float(fixed_import.get("delta_abs", float("nan"))),
        relative_error_abs=float(fixed_import.get("relative_error_abs", float("nan"))),
        meta={
            "pipeline": "hh_staged_noiseless",
            "workflow_tag": str(cfg.artifacts.tag),
            "stage_chain": ["hf_reference", "fixed_final_state_import", "final_only_noiseless_dynamics"],
            "fixed_final_state_json": str(fixed_import.get("source_json", "")),
        },
        handoff_state_kind="prepared_state",
    )
    _write_json(
        cfg.artifacts.replay_output_json,
        {
            "generated_utc": _now_utc(),
            "pipeline": "hh_staged_noiseless",
            "mode": "fixed_final_state_import",
            "source_json": str(fixed_import.get("source_json", "")),
            "fixed_final_state_import": dict(fixed_import),
            "exact": dict(replay_payload.get("exact", {})),
            "vqe": dict(replay_payload.get("vqe", {})),
            "best_state": dict(replay_payload.get("best_state", {})),
        },
    )


def _run_warm_start_stage(
    cfg: StagedHHConfig,
    *,
    h_poly: Any,
    psi_hf: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, Path]:
    nq_total = int(round(math.log2(int(np.asarray(psi_hf, dtype=complex).size))))
    exact_filtered_energy: float | None = None
    resumed_from_checkpoint = False
    resume_seed_json: Path | None = None
    resume_initial_point: list[float] | None = None
    checkpoint_state: dict[str, Any] = {
        "best_energy": float("inf"),
        "checkpoint_json": None,
    }

    _append_workflow_log(
        cfg,
        "warm_stage_start",
        stop_energy=cfg.warm_checkpoint.stop_energy,
        stop_delta_abs=cfg.warm_checkpoint.stop_delta_abs,
        checkpoint_json=str(cfg.artifacts.warm_checkpoint_json),
    )

    def _load_checkpoint_for_handoff(
        checkpoint_json: Path,
    ) -> tuple[dict[str, Any], np.ndarray, Mapping[str, Any], list[float] | None, float, float, dict[str, Any]]:
        raw = json.loads(Path(checkpoint_json).read_text(encoding="utf-8"))
        psi_seed, adapt_ref_meta = adapt_mod._load_adapt_initial_state(Path(checkpoint_json), int(nq_total))
        mismatches = adapt_mod._validate_adapt_ref_metadata_for_exact_reuse(
            adapt_settings=adapt_ref_meta.get("settings", {}),
            args=_expected_adapt_ref_args(cfg),
            is_hh=True,
        )
        if mismatches:
            raise ValueError(
                "warm checkpoint settings mismatch: "
                + "; ".join(str(x) for x in mismatches)
            )
        exact_seed_energy = adapt_mod._resolve_exact_energy_from_payload(raw)
        if exact_seed_energy is None:
            raise ValueError("Warm checkpoint missing ground_state.exact_energy_filtered.")
        seed_energy = _resolve_checkpoint_energy(raw)
        if seed_energy is None:
            raise ValueError("Warm checkpoint missing adapt_vqe.energy.")
        raw_meta = raw.get("meta", {})
        seed_theta = None
        if isinstance(raw_meta, Mapping):
            theta_raw = raw_meta.get("warm_optimal_point")
            if isinstance(theta_raw, Sequence) and not isinstance(theta_raw, (str, bytes)):
                seed_theta = [float(x) for x in theta_raw]
        cutoff = _warm_stop_status(
            cfg,
            energy=float(seed_energy),
            exact_filtered_energy=float(exact_seed_energy),
        )
        return (
            raw,
            np.asarray(psi_seed, dtype=complex).reshape(-1),
            raw_meta if isinstance(raw_meta, Mapping) else {},
            seed_theta,
            float(exact_seed_energy),
            float(seed_energy),
            cutoff,
        )

    handoff_path = cfg.warm_checkpoint.handoff_from_warm_checkpoint
    if handoff_path is not None:
        handoff_json = Path(handoff_path)
        (
            _raw_handoff,
            psi_handoff,
            handoff_meta,
            handoff_theta,
            exact_filtered_energy,
            warm_energy,
            cutoff_status,
        ) = _load_checkpoint_for_handoff(handoff_json)
        _write_warm_checkpoint_bundle(
            cfg,
            path=cfg.artifacts.warm_checkpoint_json,
            psi_state=psi_handoff,
            energy=float(warm_energy),
            exact_filtered_energy=float(exact_filtered_energy),
            theta=handoff_theta,
            role="warm_checkpoint",
            cutoff_status=cutoff_status,
            event_meta=handoff_meta,
            source_json=handoff_json,
        )
        _write_warm_checkpoint_bundle(
            cfg,
            path=cfg.artifacts.warm_cutover_json,
            psi_state=psi_handoff,
            energy=float(warm_energy),
            exact_filtered_energy=float(exact_filtered_energy),
            theta=handoff_theta,
            role="warm_cutover",
            cutoff_status=cutoff_status,
            event_meta=handoff_meta,
            source_json=handoff_json,
        )
        checkpoint_state["best_energy"] = float(warm_energy)
        checkpoint_state["checkpoint_json"] = Path(cfg.artifacts.warm_checkpoint_json)
        warm_payload = {
            "success": True,
            "method": "hh_staged_warm_handoff_checkpoint",
            "energy": float(warm_energy),
            "ansatz": str(cfg.warm_start.ansatz_name),
            "exact_filtered_energy": float(exact_filtered_energy),
            "optimizer_method": str(cfg.warm_start.method),
            "message": "warm_handoff_from_checkpoint",
            "optimal_point": handoff_theta,
            "checkpoint_json_latest": str(cfg.artifacts.warm_checkpoint_json),
            "checkpoint_json_used": str(cfg.artifacts.warm_cutover_json),
            "cutoff_triggered": bool(cutoff_status["triggered"]),
            "cutoff_reason": cutoff_status.get("reason"),
            "cutoff_delta_abs": float(cutoff_status["delta_abs"]),
            "resumed_from_checkpoint": False,
            "handoff_from_checkpoint": True,
            "handoff_checkpoint_json": str(handoff_json),
        }
        _append_workflow_log(
            cfg,
            "warm_handoff_checkpoint_loaded",
            checkpoint_json=str(handoff_json),
            checkpoint_json_latest=str(cfg.artifacts.warm_checkpoint_json),
            checkpoint_json_used=str(cfg.artifacts.warm_cutover_json),
            energy=float(warm_energy),
            exact_filtered_energy=float(exact_filtered_energy),
            delta_abs=float(cutoff_status["delta_abs"]),
            cutoff_triggered=bool(cutoff_status["triggered"]),
            cutoff_reason=cutoff_status.get("reason"),
        )
        _append_workflow_log(
            cfg,
            "warm_stage_complete",
            checkpoint_json=str(cfg.artifacts.warm_cutover_json),
            checkpoint_json_latest=str(cfg.artifacts.warm_checkpoint_json),
            energy=float(warm_energy),
            exact_filtered_energy=float(exact_filtered_energy),
            delta_abs=float(cutoff_status["delta_abs"]),
            cutoff_triggered=bool(cutoff_status["triggered"]),
            cutoff_reason=cutoff_status.get("reason"),
            handoff_from_checkpoint=True,
            handoff_checkpoint_json=str(handoff_json),
        )
        return warm_payload, psi_handoff, Path(cfg.artifacts.warm_cutover_json)

    resume_path = cfg.warm_checkpoint.resume_from_warm_checkpoint
    if resume_path is not None:
        resume_json = Path(resume_path)
        (
            _raw_resume,
            psi_resume,
            raw_meta,
            resume_theta,
            exact_filtered_energy,
            warm_energy,
            cutoff_status,
        ) = _load_checkpoint_for_handoff(resume_json)
        cutoff_status = _warm_stop_status(
            cfg,
            energy=float(warm_energy),
            exact_filtered_energy=float(exact_filtered_energy),
        )
        if resume_theta is None:
            raise ValueError(
                "Resume checkpoint missing meta.warm_optimal_point; cannot continue warm optimization."
            )
        _write_warm_checkpoint_bundle(
            cfg,
            path=cfg.artifacts.warm_checkpoint_json,
            psi_state=np.asarray(psi_resume, dtype=complex).reshape(-1),
            energy=float(warm_energy),
            exact_filtered_energy=float(exact_filtered_energy),
            theta=resume_theta,
            role="warm_checkpoint",
            cutoff_status=cutoff_status,
            event_meta=(raw_meta if isinstance(raw_meta, Mapping) else None),
            source_json=resume_json,
        )
        checkpoint_state["best_energy"] = float(warm_energy)
        checkpoint_state["checkpoint_json"] = Path(cfg.artifacts.warm_checkpoint_json)
        warm_payload = {
            "success": True,
            "method": "hh_staged_warm_resume_checkpoint",
            "energy": float(warm_energy),
            "ansatz": str(cfg.warm_start.ansatz_name),
            "exact_filtered_energy": float(exact_filtered_energy),
            "optimizer_method": str(cfg.warm_start.method),
            "message": "warm_resumed_from_checkpoint",
            "optimal_point": resume_theta,
            "checkpoint_json_latest": str(cfg.artifacts.warm_checkpoint_json),
            "checkpoint_json_used": str(cfg.artifacts.warm_checkpoint_json),
            "cutoff_triggered": bool(cutoff_status["triggered"]),
            "cutoff_reason": cutoff_status.get("reason"),
            "cutoff_delta_abs": float(cutoff_status["delta_abs"]),
            "resumed_from_checkpoint": True,
            "resume_checkpoint_json": str(resume_json),
        }
        _append_workflow_log(
            cfg,
            "warm_resume_checkpoint_loaded",
            checkpoint_json=str(resume_json),
            energy=float(warm_energy),
            exact_filtered_energy=float(exact_filtered_energy),
            delta_abs=float(cutoff_status["delta_abs"]),
            cutoff_triggered=bool(cutoff_status["triggered"]),
            cutoff_reason=cutoff_status.get("reason"),
        )
        if bool(cutoff_status["triggered"]):
            _write_warm_checkpoint_bundle(
                cfg,
                path=cfg.artifacts.warm_cutover_json,
                psi_state=np.asarray(psi_resume, dtype=complex).reshape(-1),
                energy=float(warm_energy),
                exact_filtered_energy=float(exact_filtered_energy),
                theta=resume_theta,
                role="warm_cutover",
                cutoff_status=cutoff_status,
                event_meta=(raw_meta if isinstance(raw_meta, Mapping) else None),
                source_json=resume_json,
            )
            warm_payload["checkpoint_json_used"] = str(cfg.artifacts.warm_cutover_json)
            _append_workflow_log(
                cfg,
                "warm_cutoff_triggered",
                checkpoint_json=str(cfg.artifacts.warm_cutover_json),
                checkpoint_json_latest=str(cfg.artifacts.warm_checkpoint_json),
                energy=float(warm_energy),
                exact_filtered_energy=float(exact_filtered_energy),
                delta_abs=float(cutoff_status["delta_abs"]),
                cutoff_triggered=True,
                cutoff_reason=cutoff_status.get("reason"),
                resumed_from_checkpoint=True,
                resume_checkpoint_json=str(resume_json),
            )
            return warm_payload, np.asarray(psi_resume, dtype=complex).reshape(-1), Path(cfg.artifacts.warm_cutover_json)
        resumed_from_checkpoint = True
        resume_seed_json = resume_json
        resume_initial_point = list(resume_theta)
        _append_workflow_log(
            cfg,
            "warm_resume_checkpoint_continue",
            checkpoint_json=str(cfg.artifacts.warm_checkpoint_json),
            resume_checkpoint_json=str(resume_json),
            energy=float(warm_energy),
            exact_filtered_energy=float(exact_filtered_energy),
            delta_abs=float(cutoff_status["delta_abs"]),
        )

    if exact_filtered_energy is None:
        exact_filtered_energy = float(
            exact_ground_energy_sector_hh(
                h_poly,
                num_sites=int(cfg.physics.L),
                num_particles=(int(cfg.physics.sector_n_up), int(cfg.physics.sector_n_dn)),
                n_ph_max=int(cfg.physics.n_ph_max),
                boson_encoding=str(cfg.physics.boson_encoding),
                indexing=str(cfg.physics.ordering),
            )
        )
    
    ansatz = _build_hh_warm_ansatz(cfg)

    def _emit_checkpoint(theta_values: Sequence[float], energy_value: float, event_meta: Mapping[str, Any] | None) -> None:
        psi_best = hc_pipeline._normalize_state(
            np.asarray(
                ansatz.prepare_state(np.asarray(theta_values, dtype=float), np.asarray(psi_hf, dtype=complex)),
                dtype=complex,
            ).reshape(-1)
        )
        cutoff_status = _warm_stop_status(
            cfg,
            energy=float(energy_value),
            exact_filtered_energy=float(exact_filtered_energy),
        )
        _write_warm_checkpoint_bundle(
            cfg,
            path=cfg.artifacts.warm_checkpoint_json,
            psi_state=psi_best,
            energy=float(energy_value),
            exact_filtered_energy=float(exact_filtered_energy),
            theta=theta_values,
            role="warm_checkpoint",
            cutoff_status=cutoff_status,
            event_meta=event_meta,
        )
        checkpoint_state["best_energy"] = float(energy_value)
        checkpoint_state["checkpoint_json"] = Path(cfg.artifacts.warm_checkpoint_json)
        _append_workflow_log(
            cfg,
            "warm_new_best_checkpoint",
            checkpoint_json=str(cfg.artifacts.warm_checkpoint_json),
            energy=float(energy_value),
            exact_filtered_energy=float(exact_filtered_energy),
            delta_abs=float(cutoff_status["delta_abs"]),
            cutoff_triggered=bool(cutoff_status["triggered"]),
            cutoff_reason=cutoff_status.get("reason"),
            restart_index=(None if event_meta is None else event_meta.get("restart_index")),
            nfev_so_far=(None if event_meta is None else event_meta.get("nfev_so_far")),
        )

    def _progress_observer(event: Mapping[str, Any]) -> None:
        if str(event.get("event", "")) != "new_best":
            return
        raw_energy = event.get("energy_best_global")
        theta_values = event.get("theta_restart_best", event.get("theta_current"))
        if not isinstance(raw_energy, (int, float)):
            return
        if not isinstance(theta_values, Sequence) or isinstance(theta_values, (str, bytes)):
            return
        energy_value = float(raw_energy)
        if not np.isfinite(energy_value):
            return
        if energy_value >= float(checkpoint_state["best_energy"]) - 1e-15:
            return
        _emit_checkpoint(theta_values, energy_value, event)

    def _early_stop_checker(event: Mapping[str, Any]) -> bool:
        raw_energy = event.get("energy_best_global")
        if not isinstance(raw_energy, (int, float)):
            return False
        cutoff_status = _warm_stop_status(
            cfg,
            energy=float(raw_energy),
            exact_filtered_energy=float(exact_filtered_energy),
        )
        return bool(cutoff_status["triggered"])

    warm_payload, psi_warm = hc_pipeline._run_hardcoded_vqe(
        num_sites=int(cfg.physics.L),
        ordering=str(cfg.physics.ordering),
        boundary=str(cfg.physics.boundary),
        hopping_t=float(cfg.physics.t),
        onsite_u=float(cfg.physics.u),
        potential_dv=float(cfg.physics.dv),
        h_poly=h_poly,
        reps=int(cfg.warm_start.reps),
        restarts=int(cfg.warm_start.restarts),
        seed=int(cfg.warm_start.seed),
        maxiter=int(cfg.warm_start.maxiter),
        method=str(cfg.warm_start.method),
        energy_backend=str(cfg.warm_start.energy_backend),
        vqe_progress_every_s=float(cfg.warm_start.progress_every_s),
        progress_observer=_progress_observer,
        emit_theta_in_progress=True,
        return_best_on_keyboard_interrupt=True,
        early_stop_checker=(
            _early_stop_checker if _warm_stop_required(cfg) else None
        ),
        initial_point=resume_initial_point,
        ansatz_name=str(cfg.warm_start.ansatz_name),
        spsa_a=float(cfg.warm_start.spsa_a),
        spsa_c=float(cfg.warm_start.spsa_c),
        spsa_alpha=float(cfg.warm_start.spsa_alpha),
        spsa_gamma=float(cfg.warm_start.spsa_gamma),
        spsa_A=float(cfg.warm_start.spsa_A),
        spsa_avg_last=int(cfg.warm_start.spsa_avg_last),
        spsa_eval_repeats=int(cfg.warm_start.spsa_eval_repeats),
        spsa_eval_agg=str(cfg.warm_start.spsa_eval_agg),
        problem="hh",
        omega0=float(cfg.physics.omega0),
        g_ep=float(cfg.physics.g_ep),
        n_ph_max=int(cfg.physics.n_ph_max),
        boson_encoding=str(cfg.physics.boson_encoding),
    )
    warm_energy = float(warm_payload.get("energy", float("nan")))
    cutoff_status = _warm_stop_status(
        cfg,
        energy=float(warm_energy),
        exact_filtered_energy=float(exact_filtered_energy),
    )
    theta_final = warm_payload.get("optimal_point", [])
    if (
        checkpoint_state["checkpoint_json"] is None
        or float(warm_energy) < float(checkpoint_state["best_energy"]) - 1e-15
    ):
        if isinstance(theta_final, Sequence) and not isinstance(theta_final, (str, bytes)) and len(theta_final) > 0:
            _emit_checkpoint(theta_final, float(warm_energy), {"event": "run_end"})
        else:
            _write_warm_checkpoint_bundle(
                cfg,
                path=cfg.artifacts.warm_checkpoint_json,
                psi_state=np.asarray(psi_warm, dtype=complex).reshape(-1),
                energy=float(warm_energy),
                exact_filtered_energy=float(exact_filtered_energy),
                theta=None,
                role="warm_checkpoint",
                cutoff_status=cutoff_status,
            )
            checkpoint_state["best_energy"] = float(warm_energy)
            checkpoint_state["checkpoint_json"] = Path(cfg.artifacts.warm_checkpoint_json)
    if _warm_stop_required(cfg) and not bool(cutoff_status["triggered"]):
        _append_workflow_log(
            cfg,
            "warm_cutoff_not_reached_continue",
            checkpoint_json=str(cfg.artifacts.warm_checkpoint_json),
            energy=float(warm_energy),
            exact_filtered_energy=float(exact_filtered_energy),
            delta_abs=float(cutoff_status["delta_abs"]),
            stop_energy=cfg.warm_checkpoint.stop_energy,
            stop_delta_abs=cfg.warm_checkpoint.stop_delta_abs,
        )
    _write_warm_checkpoint_bundle(
        cfg,
        path=cfg.artifacts.warm_cutover_json,
        psi_state=np.asarray(psi_warm, dtype=complex).reshape(-1),
        energy=float(warm_energy),
        exact_filtered_energy=float(exact_filtered_energy),
        theta=(
            [float(x) for x in theta_final]
            if isinstance(theta_final, Sequence) and not isinstance(theta_final, (str, bytes))
            else None
        ),
        role="warm_cutover",
        cutoff_status=cutoff_status,
        source_json=Path(cfg.artifacts.warm_checkpoint_json),
    )
    _append_workflow_log(
        cfg,
        ("warm_cutoff_triggered" if bool(cutoff_status["triggered"]) else "warm_stage_complete"),
        checkpoint_json=str(cfg.artifacts.warm_cutover_json),
        checkpoint_json_latest=str(cfg.artifacts.warm_checkpoint_json),
        energy=float(warm_energy),
        exact_filtered_energy=float(exact_filtered_energy),
        delta_abs=float(cutoff_status["delta_abs"]),
        cutoff_triggered=bool(cutoff_status["triggered"]),
        cutoff_reason=cutoff_status.get("reason"),
    )
    warm_payload["exact_filtered_energy"] = float(exact_filtered_energy)
    warm_payload["checkpoint_json_latest"] = str(cfg.artifacts.warm_checkpoint_json)
    warm_payload["checkpoint_json_used"] = str(cfg.artifacts.warm_cutover_json)
    warm_payload["cutoff_triggered"] = bool(cutoff_status["triggered"])
    warm_payload["cutoff_reason"] = cutoff_status.get("reason")
    warm_payload["cutoff_delta_abs"] = float(cutoff_status["delta_abs"])
    warm_payload["resumed_from_checkpoint"] = bool(resumed_from_checkpoint)
    warm_payload["handoff_from_checkpoint"] = False
    if resume_seed_json is not None:
        warm_payload["resume_checkpoint_json"] = str(resume_seed_json)
    return warm_payload, np.asarray(psi_warm, dtype=complex).reshape(-1), Path(cfg.artifacts.warm_cutover_json)


def _bool_flag(raw: Any) -> bool:
    if isinstance(raw, bool):
        return bool(raw)
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Could not interpret boolean flag value {raw!r}.")


def _parse_noiseless_methods(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        parts = [x.strip().lower() for x in raw.split(",") if x.strip()]
    else:
        parts = [str(x).strip().lower() for x in raw if str(x).strip()]
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part not in _ALLOWED_NOISELESS_METHODS:
            raise ValueError(
                f"Unsupported noiseless method '{part}'. Expected subset of {_ALLOWED_NOISELESS_METHODS}."
            )
        if part in seen:
            continue
        seen.add(part)
        out.append(part)
    if not out:
        raise ValueError("At least one noiseless propagation method is required.")
    return tuple(out)


def _parse_drive_custom_weights(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    if text.startswith("["):
        vals = json.loads(text)
    else:
        vals = [float(x) for x in text.split(",") if x.strip()]
    return [float(x) for x in vals]


def _half_filled_particles(L: int) -> tuple[int, int]:
    n_up, n_dn = hc_pipeline._half_filled_particles(int(L))
    return int(n_up), int(n_dn)


def _hh_nq_total(L: int, n_ph_max: int, boson_encoding: str) -> int:
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    return int(2 * int(L) + int(L) * qpb)


def _default_output_tag(
    *,
    L: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    ordering: str,
    boundary: str,
    sector_n_up: int,
    sector_n_dn: int,
    drive_enabled: bool,
    drive_pattern: str,
    drive_A: float,
    drive_omega: float,
    drive_tbar: float,
    drive_phi: float,
    drive_time_sampling: str,
    noiseless_methods: str,
    adapt_continuation_mode: str,
    warm_ansatz: str,
    seed_refine_family: str | None,
    fixed_final_state_json: str | None,
    circuit_backend_name: str | None,
    circuit_use_fake_backend: bool,
) -> str:
    drive_label = "drive" if bool(drive_enabled) else "static"
    spec = {
        "L": int(L),
        "t": float(t),
        "u": float(u),
        "dv": float(dv),
        "omega0": float(omega0),
        "g_ep": float(g_ep),
        "n_ph_max": int(n_ph_max),
        "ordering": str(ordering),
        "boundary": str(boundary),
        "sector_n_up": int(sector_n_up),
        "sector_n_dn": int(sector_n_dn),
        "drive_enabled": bool(drive_enabled),
        "drive_pattern": str(drive_pattern),
        "drive_A": float(drive_A),
        "drive_omega": float(drive_omega),
        "drive_tbar": float(drive_tbar),
        "drive_phi": float(drive_phi),
        "drive_time_sampling": str(drive_time_sampling),
        "noiseless_methods": str(noiseless_methods),
        "adapt_continuation_mode": str(adapt_continuation_mode),
        "warm_ansatz": str(warm_ansatz),
        "seed_refine_family": seed_refine_family,
        "fixed_final_state_json": fixed_final_state_json,
        "circuit_backend_name": circuit_backend_name,
        "circuit_use_fake_backend": bool(circuit_use_fake_backend),
    }
    digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    refine_label = "" if seed_refine_family in {None, ""} else f"_refine{str(seed_refine_family)}"
    return (
        f"hh_staged_L{int(L)}_{drive_label}_"
        f"t{float(t):g}_U{float(u):g}_dv{float(dv):g}_w{float(omega0):g}_g{float(g_ep):g}_"
        f"nph{int(n_ph_max)}_warm{str(warm_ansatz)}{refine_label}_{digest}"
    )


"""
ws_reps(L) = L
ws_restarts(L) = ceil(5L/3)
ws_maxiter(L) = round(4000 L^2 / 9)
adapt_max_depth(L) = 40L
adapt_maxiter(L) = round(5000 L^2 / 9)
final_reps(L) = L
final_restarts(L) := ws_restarts(L)   [workflow inference]
final_maxiter(L) := ws_maxiter(L)     [workflow inference]
t_final(L) = 5L
trotter_steps(L) = 64L
num_times(L) = 1 + ceil(200L/3)
exact_steps_multiplier(L) = ceil((L + 1)/2)
"""
def _scaled_defaults(L: int) -> dict[str, Any]:
    L_int = int(L)
    return {
        "warm_reps": int(max(1, L_int)),
        "warm_restarts": int(max(1, math.ceil((5.0 * L_int) / 3.0))),
        "warm_maxiter": int(max(200, round((4000.0 * L_int * L_int) / 9.0))),
        "adapt_max_depth": int(max(15, 40 * L_int)),
        "adapt_maxiter": int(max(300, round((5000.0 * L_int * L_int) / 9.0))),
        "adapt_eps_grad": 5e-7,
        "adapt_eps_energy": 1e-9,
        "final_reps": int(max(1, L_int)),
        "final_restarts": int(max(1, math.ceil((5.0 * L_int) / 3.0))),
        "final_maxiter": int(max(200, round((4000.0 * L_int * L_int) / 9.0))),
        "t_final": float(5.0 * L_int),
        "trotter_steps": int(64 * L_int),
        "num_times": int(1 + math.ceil((200.0 * L_int) / 3.0)),
        "exact_steps_multiplier": int(math.ceil((L_int + 1) / 2.0)),
    }


def _resolve_with_default(
    *,
    name: str,
    raw: Any,
    default: Any,
    provenance: dict[str, str],
    default_source: str,
) -> Any:
    if raw is None:
        provenance[name] = str(default_source)
        return default
    provenance[name] = "cli"
    return raw


def _enforce_not_weaker(
    *,
    cfg_values: Mapping[str, Any],
    baseline: Mapping[str, Any],
    smoke_test_intentionally_weak: bool,
) -> None:
    if bool(smoke_test_intentionally_weak):
        return
    checks = {
        "warm_reps": int(cfg_values["warm_reps"]) >= int(baseline["warm_reps"]),
        "warm_restarts": int(cfg_values["warm_restarts"]) >= int(baseline["warm_restarts"]),
        "warm_maxiter": int(cfg_values["warm_maxiter"]) >= int(baseline["warm_maxiter"]),
        "adapt_max_depth": int(cfg_values["adapt_max_depth"]) >= int(baseline["adapt_max_depth"]),
        "adapt_maxiter": int(cfg_values["adapt_maxiter"]) >= int(baseline["adapt_maxiter"]),
        "final_reps": int(cfg_values["final_reps"]) >= int(baseline["final_reps"]),
        "final_restarts": int(cfg_values["final_restarts"]) >= int(baseline["final_restarts"]),
        "final_maxiter": int(cfg_values["final_maxiter"]) >= int(baseline["final_maxiter"]),
        "trotter_steps": int(cfg_values["trotter_steps"]) >= int(baseline["trotter_steps"]),
    }
    if bool(cfg_values.get("seed_refine_enabled", False)):
        checks.update(
            {
                "seed_refine_reps": int(cfg_values["seed_refine_reps"]) >= int(baseline["final_reps"]),
                "seed_refine_maxiter": int(cfg_values["seed_refine_maxiter"]) >= int(baseline["final_maxiter"]),
            }
        )
    failed = [key for key, ok in checks.items() if not bool(ok)]
    if failed:
        raise ValueError(
            "Under-parameterized staged HH run rejected. "
            f"Failed fields: {failed}. Baseline defaults: {dict(baseline)}. "
            "Use --smoke-test-intentionally-weak only for explicit smoke tests."
        )


def resolve_staged_hh_config(args: Any) -> StagedHHConfig:
    L = int(getattr(args, "L"))
    defaults = _scaled_defaults(L)
    provenance: dict[str, str] = {}
    warm_ansatz = str(
        _resolve_with_default(
            name="warm_ansatz",
            raw=getattr(args, "warm_ansatz", None),
            default="hh_hva_ptw",
            provenance=provenance,
            default_source="workflow.warm_ansatz.default=hh_hva_ptw",
        )
    )

    sector_n_up_raw = getattr(args, "sector_n_up", None)
    sector_n_dn_raw = getattr(args, "sector_n_dn", None)
    sector_n_up_default, sector_n_dn_default = _half_filled_particles(L)

    sector_n_up = int(sector_n_up_default if sector_n_up_raw is None else sector_n_up_raw)
    sector_n_dn = int(sector_n_dn_default if sector_n_dn_raw is None else sector_n_dn_raw)
    if (int(sector_n_up), int(sector_n_dn)) != (int(sector_n_up_default), int(sector_n_dn_default)):
        raise ValueError(
            "hh_staged_noiseless currently supports only the half-filled sector across all stages. "
            "Non-default --sector-n-up/--sector-n-dn overrides are not yet plumbed through warm-start and ADAPT."
        )

    tag = _resolve_with_default(
        name="tag",
        raw=getattr(args, "tag", None),
        default=_default_output_tag(
            L=L,
            t=float(getattr(args, "t")),
            u=float(getattr(args, "u")),
            dv=float(getattr(args, "dv")),
            omega0=float(getattr(args, "omega0")),
            g_ep=float(getattr(args, "g_ep")),
            n_ph_max=int(getattr(args, "n_ph_max")),
            ordering=str(getattr(args, "ordering")),
            boundary=str(getattr(args, "boundary")),
            sector_n_up=int(sector_n_up),
            sector_n_dn=int(sector_n_dn),
            drive_enabled=bool(getattr(args, "enable_drive")),
            drive_pattern=str(getattr(args, "drive_pattern")),
            drive_A=float(getattr(args, "drive_A")),
            drive_omega=float(getattr(args, "drive_omega")),
            drive_tbar=float(getattr(args, "drive_tbar")),
            drive_phi=float(getattr(args, "drive_phi")),
            drive_time_sampling=str(getattr(args, "drive_time_sampling")),
            noiseless_methods=str(getattr(args, "noiseless_methods")),
            adapt_continuation_mode=str(getattr(args, "adapt_continuation_mode")),
            warm_ansatz=str(warm_ansatz),
            seed_refine_family=(
                None
                if getattr(args, "seed_refine_family", None) is None
                else str(getattr(args, "seed_refine_family"))
            ),
            fixed_final_state_json=(
                None
                if getattr(args, "fixed_final_state_json", None) is None
                else str(Path(getattr(args, "fixed_final_state_json")))
            ),
            circuit_backend_name=(
                None
                if getattr(args, "circuit_backend_name", None) in {None, ""}
                else str(getattr(args, "circuit_backend_name"))
            ),
            circuit_use_fake_backend=bool(getattr(args, "circuit_use_fake_backend", False)),
        ),
        provenance=provenance,
        default_source="workflow.tag.default",
    )

    output_json = Path(
        _resolve_with_default(
            name="output_json",
            raw=getattr(args, "output_json", None),
            default=REPO_ROOT / "artifacts" / "json" / f"{tag}.json",
            provenance=provenance,
            default_source="artifacts/json/<tag>.json",
        )
    )
    output_pdf = Path(
        _resolve_with_default(
            name="output_pdf",
            raw=getattr(args, "output_pdf", None),
            default=REPO_ROOT / "artifacts" / "pdf" / f"{tag}.pdf",
            provenance=provenance,
            default_source="artifacts/pdf/<tag>.pdf",
        )
    )
    state_export_dir = Path(
        _resolve_with_default(
            name="state_export_dir",
            raw=getattr(args, "state_export_dir", None),
            default=REPO_ROOT / "artifacts" / "json",
            provenance=provenance,
            default_source="artifacts/json",
        )
    )
    state_export_prefix = str(
        _resolve_with_default(
            name="state_export_prefix",
            raw=getattr(args, "state_export_prefix", None),
            default=str(tag),
            provenance=provenance,
            default_source="workflow.state_export_prefix := tag",
        )
    )
    resume_from_warm_checkpoint = (
        None
        if getattr(args, "resume_from_warm_checkpoint", None) is None
        else Path(getattr(args, "resume_from_warm_checkpoint"))
    )
    handoff_from_warm_checkpoint = (
        None
        if getattr(args, "handoff_from_warm_checkpoint", None) is None
        else Path(getattr(args, "handoff_from_warm_checkpoint"))
    )
    if resume_from_warm_checkpoint is not None and handoff_from_warm_checkpoint is not None:
        raise ValueError(
            "Use either --resume-from-warm-checkpoint or --handoff-from-warm-checkpoint, not both."
        )

    handoff_json = REPO_ROOT / "artifacts" / "json" / f"{tag}_adapt_handoff.json"
    warm_checkpoint_json = state_export_dir / f"{state_export_prefix}_warm_checkpoint_state.json"
    warm_cutover_json = state_export_dir / f"{state_export_prefix}_warm_cutover_state.json"
    replay_output_json = REPO_ROOT / "artifacts" / "json" / f"{tag}_replay.json"
    replay_output_csv = REPO_ROOT / "artifacts" / "json" / f"{tag}_replay.csv"
    replay_output_md = REPO_ROOT / "artifacts" / "useful" / f"L{L}" / f"{tag}_replay.md"
    replay_output_log = REPO_ROOT / "artifacts" / "logs" / f"{tag}_replay.log"
    workflow_log = REPO_ROOT / "artifacts" / "logs" / f"{tag}.log"

    cfg_values = {
        "warm_reps": _resolve_with_default(
            name="warm_reps",
            raw=getattr(args, "warm_reps", None),
            default=defaults["warm_reps"],
            provenance=provenance,
            default_source="run_guide.ws_reps(L)=L",
        ),
        "warm_restarts": _resolve_with_default(
            name="warm_restarts",
            raw=getattr(args, "warm_restarts", None),
            default=defaults["warm_restarts"],
            provenance=provenance,
            default_source="run_guide.ws_restarts(L)=ceil(5L/3)",
        ),
        "warm_maxiter": _resolve_with_default(
            name="warm_maxiter",
            raw=getattr(args, "warm_maxiter", None),
            default=defaults["warm_maxiter"],
            provenance=provenance,
            default_source="run_guide.ws_maxiter(L)=round(4000L^2/9)",
        ),
        "adapt_max_depth": _resolve_with_default(
            name="adapt_max_depth",
            raw=getattr(args, "adapt_max_depth", None),
            default=defaults["adapt_max_depth"],
            provenance=provenance,
            default_source="run_guide.adapt_max_depth(L)=40L",
        ),
        "adapt_maxiter": _resolve_with_default(
            name="adapt_maxiter",
            raw=getattr(args, "adapt_maxiter", None),
            default=defaults["adapt_maxiter"],
            provenance=provenance,
            default_source="run_guide.adapt_maxiter(L)=round(5000L^2/9)",
        ),
        "adapt_eps_grad": _resolve_with_default(
            name="adapt_eps_grad",
            raw=getattr(args, "adapt_eps_grad", None),
            default=defaults["adapt_eps_grad"],
            provenance=provenance,
            default_source="run_guide.adapt_eps_grad=5e-7",
        ),
        "adapt_eps_energy": _resolve_with_default(
            name="adapt_eps_energy",
            raw=getattr(args, "adapt_eps_energy", None),
            default=defaults["adapt_eps_energy"],
            provenance=provenance,
            default_source="run_guide.adapt_eps_energy=1e-9",
        ),
        "final_reps": _resolve_with_default(
            name="final_reps",
            raw=getattr(args, "final_reps", None),
            default=defaults["final_reps"],
            provenance=provenance,
            default_source="run_guide.vqe_reps(L)=L",
        ),
        "final_restarts": _resolve_with_default(
            name="final_restarts",
            raw=getattr(args, "final_restarts", None),
            default=defaults["final_restarts"],
            provenance=provenance,
            default_source="workflow.final_restarts := warm_restarts(L)",
        ),
        "final_maxiter": _resolve_with_default(
            name="final_maxiter",
            raw=getattr(args, "final_maxiter", None),
            default=defaults["final_maxiter"],
            provenance=provenance,
            default_source="workflow.final_maxiter := warm_maxiter(L)",
        ),
        "seed_refine_reps": _resolve_with_default(
            name="seed_refine_reps",
            raw=getattr(args, "seed_refine_reps", None),
            default=defaults["final_reps"],
            provenance=provenance,
            default_source="workflow.seed_refine_reps := final_reps(L)",
        ),
        "seed_refine_maxiter": _resolve_with_default(
            name="seed_refine_maxiter",
            raw=getattr(args, "seed_refine_maxiter", None),
            default=defaults["final_maxiter"],
            provenance=provenance,
            default_source="workflow.seed_refine_maxiter := final_maxiter(L)",
        ),
        "seed_refine_enabled": bool(getattr(args, "seed_refine_family", None) not in {None, "", "none"}),
        "t_final": _resolve_with_default(
            name="t_final",
            raw=getattr(args, "t_final", None),
            default=defaults["t_final"],
            provenance=provenance,
            default_source="run_guide.t_final(L)=5L",
        ),
        "trotter_steps": _resolve_with_default(
            name="trotter_steps",
            raw=getattr(args, "trotter_steps", None),
            default=defaults["trotter_steps"],
            provenance=provenance,
            default_source="run_guide.trotter_steps(L)=64L",
        ),
        "num_times": _resolve_with_default(
            name="num_times",
            raw=getattr(args, "num_times", None),
            default=defaults["num_times"],
            provenance=provenance,
            default_source="run_guide.num_times(L)=1+ceil(200L/3)",
        ),
        "exact_steps_multiplier": _resolve_with_default(
            name="exact_steps_multiplier",
            raw=getattr(args, "exact_steps_multiplier", None),
            default=defaults["exact_steps_multiplier"],
            provenance=provenance,
            default_source="run_guide.exact_steps_multiplier(L)=ceil((L+1)/2)",
        ),
    }
    _enforce_not_weaker(
        cfg_values=cfg_values,
        baseline=defaults,
        smoke_test_intentionally_weak=bool(getattr(args, "smoke_test_intentionally_weak", False)),
    )

    physics = PhysicsConfig(
        L=L,
        t=float(getattr(args, "t")),
        u=float(getattr(args, "u")),
        dv=float(getattr(args, "dv")),
        omega0=float(getattr(args, "omega0")),
        g_ep=float(getattr(args, "g_ep")),
        n_ph_max=int(getattr(args, "n_ph_max")),
        boson_encoding=str(getattr(args, "boson_encoding")),
        ordering=str(getattr(args, "ordering")),
        boundary=str(getattr(args, "boundary")),
        sector_n_up=int(sector_n_up),
        sector_n_dn=int(sector_n_dn),
    )
    warm_start = WarmStartConfig(
        ansatz_name=str(warm_ansatz),
        reps=int(cfg_values["warm_reps"]),
        restarts=int(cfg_values["warm_restarts"]),
        maxiter=int(cfg_values["warm_maxiter"]),
        method=str(getattr(args, "warm_method")),
        seed=int(getattr(args, "warm_seed")),
        progress_every_s=float(getattr(args, "warm_progress_every_s")),
        energy_backend=str(getattr(args, "vqe_energy_backend")),
        spsa_a=float(getattr(args, "vqe_spsa_a")),
        spsa_c=float(getattr(args, "vqe_spsa_c")),
        spsa_alpha=float(getattr(args, "vqe_spsa_alpha")),
        spsa_gamma=float(getattr(args, "vqe_spsa_gamma")),
        spsa_A=float(getattr(args, "vqe_spsa_A")),
        spsa_avg_last=int(getattr(args, "vqe_spsa_avg_last")),
        spsa_eval_repeats=int(getattr(args, "vqe_spsa_eval_repeats")),
        spsa_eval_agg=str(getattr(args, "vqe_spsa_eval_agg")),
    )
    seed_refine = SeedRefineConfig(
        family=(
            None
            if getattr(args, "seed_refine_family", None) in {None, "", "none"}
            else str(getattr(args, "seed_refine_family"))
        ),
        reps=int(cfg_values["seed_refine_reps"]),
        maxiter=int(cfg_values["seed_refine_maxiter"]),
        optimizer=str(
            _resolve_with_default(
                name="seed_refine_optimizer",
                raw=getattr(args, "seed_refine_optimizer", None),
                default="SPSA",
                provenance=provenance,
                default_source="workflow.seed_refine_optimizer.default=SPSA",
            )
        ),
    )
    adapt_mode = str(getattr(args, "adapt_continuation_mode"))
    adapt = AdaptConfig(
        pool=(None if getattr(args, "adapt_pool", None) in {None, "", "none"} else str(getattr(args, "adapt_pool"))),
        continuation_mode=adapt_mode,
        max_depth=int(cfg_values["adapt_max_depth"]),
        maxiter=int(cfg_values["adapt_maxiter"]),
        eps_grad=float(cfg_values["adapt_eps_grad"]),
        eps_energy=float(cfg_values["adapt_eps_energy"]),
        drop_floor=(
            None
            if getattr(args, "adapt_drop_floor", None) is None
            else float(getattr(args, "adapt_drop_floor"))
        ),
        drop_patience=(
            None
            if getattr(args, "adapt_drop_patience", None) is None
            else int(getattr(args, "adapt_drop_patience"))
        ),
        drop_min_depth=(
            None
            if getattr(args, "adapt_drop_min_depth", None) is None
            else int(getattr(args, "adapt_drop_min_depth"))
        ),
        grad_floor=(
            None
            if getattr(args, "adapt_grad_floor", None) is None
            else float(getattr(args, "adapt_grad_floor"))
        ),
        seed=int(getattr(args, "adapt_seed")),
        inner_optimizer=str(getattr(args, "adapt_inner_optimizer")),
        allow_repeats=bool(getattr(args, "adapt_allow_repeats")),
        finite_angle_fallback=bool(getattr(args, "adapt_finite_angle_fallback")),
        finite_angle=float(getattr(args, "adapt_finite_angle")),
        finite_angle_min_improvement=float(getattr(args, "adapt_finite_angle_min_improvement")),
        disable_hh_seed=bool(getattr(args, "adapt_disable_hh_seed")),
        reopt_policy=str(getattr(args, "adapt_reopt_policy")),
        window_size=int(getattr(args, "adapt_window_size")),
        window_topk=int(getattr(args, "adapt_window_topk")),
        full_refit_every=int(getattr(args, "adapt_full_refit_every")),
        final_full_refit=bool(getattr(args, "adapt_final_full_refit")),
        beam_live_branches=int(getattr(args, "adapt_beam_live_branches")),
        beam_children_per_parent=(
            None
            if getattr(args, "adapt_beam_children_per_parent", None) is None
            else int(getattr(args, "adapt_beam_children_per_parent"))
        ),
        beam_terminated_keep=(
            None
            if getattr(args, "adapt_beam_terminated_keep", None) is None
            else int(getattr(args, "adapt_beam_terminated_keep"))
        ),
        paop_r=int(getattr(args, "paop_r")),
        paop_split_paulis=bool(getattr(args, "paop_split_paulis")),
        paop_prune_eps=float(getattr(args, "paop_prune_eps")),
        paop_normalization=str(getattr(args, "paop_normalization")),
        spsa_a=float(getattr(args, "adapt_spsa_a")),
        spsa_c=float(getattr(args, "adapt_spsa_c")),
        spsa_alpha=float(getattr(args, "adapt_spsa_alpha")),
        spsa_gamma=float(getattr(args, "adapt_spsa_gamma")),
        spsa_A=float(getattr(args, "adapt_spsa_A")),
        spsa_avg_last=int(getattr(args, "adapt_spsa_avg_last")),
        spsa_eval_repeats=int(getattr(args, "adapt_spsa_eval_repeats")),
        spsa_eval_agg=str(getattr(args, "adapt_spsa_eval_agg")),
        spsa_callback_every=int(getattr(args, "adapt_spsa_callback_every")),
        spsa_progress_every_s=float(getattr(args, "adapt_spsa_progress_every_s")),
        phase1_lambda_F=float(getattr(args, "phase1_lambda_F")),
        phase1_lambda_compile=float(getattr(args, "phase1_lambda_compile")),
        phase1_lambda_measure=float(getattr(args, "phase1_lambda_measure")),
        phase1_lambda_leak=float(getattr(args, "phase1_lambda_leak")),
        phase1_score_z_alpha=float(getattr(args, "phase1_score_z_alpha")),
        phase1_probe_max_positions=int(getattr(args, "phase1_probe_max_positions")),
        phase1_plateau_patience=int(getattr(args, "phase1_plateau_patience")),
        phase1_trough_margin_ratio=float(getattr(args, "phase1_trough_margin_ratio")),
        phase1_prune_enabled=bool(getattr(args, "phase1_prune_enabled")),
        phase1_prune_fraction=float(getattr(args, "phase1_prune_fraction")),
        phase1_prune_max_candidates=int(getattr(args, "phase1_prune_max_candidates")),
        phase1_prune_max_regression=float(getattr(args, "phase1_prune_max_regression")),
        phase3_motif_source_json=(
            None
            if getattr(args, "phase3_motif_source_json", None) is None
            else Path(getattr(args, "phase3_motif_source_json"))
        ),
        phase3_symmetry_mitigation_mode=str(getattr(args, "phase3_symmetry_mitigation_mode")),
        phase3_enable_rescue=bool(getattr(args, "phase3_enable_rescue")),
        phase3_lifetime_cost_mode=str(getattr(args, "phase3_lifetime_cost_mode")),
        phase3_runtime_split_mode=str(getattr(args, "phase3_runtime_split_mode")),
    )
    replay_mode_raw = getattr(args, "replay_continuation_mode", None)
    replay_mode = adapt_mode if replay_mode_raw in {None, "", "auto"} else str(replay_mode_raw)
    provenance["replay_continuation_mode"] = (
        "workflow.replay_mode := adapt_continuation_mode"
        if replay_mode_raw in {None, "", "auto"}
        else "cli"
    )
    replay = ReplayConfig(
        enabled=bool(getattr(args, "run_replay", False)),
        generator_family="match_adapt",
        fallback_family="full_meta",
        legacy_paop_key=str(getattr(args, "legacy_paop_key")),
        replay_seed_policy=str(getattr(args, "replay_seed_policy")),
        continuation_mode=str(replay_mode),
        reps=int(cfg_values["final_reps"]),
        restarts=int(cfg_values["final_restarts"]),
        maxiter=int(cfg_values["final_maxiter"]),
        method=str(getattr(args, "final_method")),
        seed=int(getattr(args, "final_seed")),
        energy_backend=str(getattr(args, "vqe_energy_backend")),
        progress_every_s=float(getattr(args, "final_progress_every_s")),
        wallclock_cap_s=int(getattr(args, "replay_wallclock_cap_s")),
        paop_r=int(getattr(args, "paop_r")),
        paop_split_paulis=bool(getattr(args, "paop_split_paulis")),
        paop_prune_eps=float(getattr(args, "paop_prune_eps")),
        paop_normalization=str(getattr(args, "paop_normalization")),
        spsa_a=float(getattr(args, "vqe_spsa_a")),
        spsa_c=float(getattr(args, "vqe_spsa_c")),
        spsa_alpha=float(getattr(args, "vqe_spsa_alpha")),
        spsa_gamma=float(getattr(args, "vqe_spsa_gamma")),
        spsa_A=float(getattr(args, "vqe_spsa_A")),
        spsa_avg_last=int(getattr(args, "vqe_spsa_avg_last")),
        spsa_eval_repeats=int(getattr(args, "vqe_spsa_eval_repeats")),
        spsa_eval_agg=str(getattr(args, "vqe_spsa_eval_agg")),
        replay_freeze_fraction=float(getattr(args, "replay_freeze_fraction")),
        replay_unfreeze_fraction=float(getattr(args, "replay_unfreeze_fraction")),
        replay_full_fraction=float(getattr(args, "replay_full_fraction")),
        replay_qn_spsa_refresh_every=int(getattr(args, "replay_qn_spsa_refresh_every")),
        replay_qn_spsa_refresh_mode=str(getattr(args, "replay_qn_spsa_refresh_mode")),
        phase3_symmetry_mitigation_mode=str(getattr(args, "phase3_symmetry_mitigation_mode")),
    )
    dynamics = DynamicsConfig(
        enabled=bool(getattr(args, "run_dynamics", False)),
        methods=_parse_noiseless_methods(getattr(args, "noiseless_methods")),
        t_final=float(cfg_values["t_final"]),
        num_times=int(cfg_values["num_times"]),
        trotter_steps=int(cfg_values["trotter_steps"]),
        exact_steps_multiplier=int(cfg_values["exact_steps_multiplier"]),
        fidelity_subspace_energy_tol=float(getattr(args, "fidelity_subspace_energy_tol")),
        cfqm_stage_exp=str(getattr(args, "cfqm_stage_exp")),
        cfqm_coeff_drop_abs_tol=float(getattr(args, "cfqm_coeff_drop_abs_tol")),
        cfqm_normalize=bool(getattr(args, "cfqm_normalize")),
        enable_drive=bool(getattr(args, "enable_drive")),
        drive_A=float(getattr(args, "drive_A")),
        drive_omega=float(getattr(args, "drive_omega")),
        drive_tbar=float(getattr(args, "drive_tbar")),
        drive_phi=float(getattr(args, "drive_phi")),
        drive_pattern=str(getattr(args, "drive_pattern")),
        drive_custom_s=getattr(args, "drive_custom_s", None),
        drive_include_identity=bool(getattr(args, "drive_include_identity")),
        drive_time_sampling=str(getattr(args, "drive_time_sampling")),
        drive_t0=float(getattr(args, "drive_t0")),
    )
    fixed_final_state = None
    fixed_final_state_json = getattr(args, "fixed_final_state_json", None)
    if fixed_final_state_json is not None:
        fixed_final_state = FixedFinalStateConfig(
            json_path=Path(fixed_final_state_json),
            strict_match=bool(getattr(args, "fixed_final_state_strict_match", True)),
        )
    circuit_backend_name_raw = getattr(args, "circuit_backend_name", None)
    circuit_backend_name = (
        None if circuit_backend_name_raw in {None, ""} else str(circuit_backend_name_raw)
    )
    circuit_use_fake_backend = bool(getattr(args, "circuit_use_fake_backend", False))
    if circuit_use_fake_backend and circuit_backend_name is None:
        raise ValueError("--circuit-use-fake-backend requires --circuit-backend-name.")
    circuit_metrics = CircuitMetricConfig(
        backend_name=circuit_backend_name,
        use_fake_backend=circuit_use_fake_backend,
        optimization_level=int(getattr(args, "circuit_transpile_optimization_level", 3)),
        seed_transpiler=int(getattr(args, "circuit_seed_transpiler", 7)),
    )
    warm_checkpoint = WarmCheckpointConfig(
        stop_energy=(
            None
            if getattr(args, "warm_stop_energy", None) is None
            else float(getattr(args, "warm_stop_energy"))
        ),
        stop_delta_abs=(
            None
            if getattr(args, "warm_stop_delta_abs", None) is None
            else float(getattr(args, "warm_stop_delta_abs"))
        ),
        state_export_dir=Path(state_export_dir),
        state_export_prefix=str(state_export_prefix),
        resume_from_warm_checkpoint=(
            None if resume_from_warm_checkpoint is None else Path(resume_from_warm_checkpoint)
        ),
        handoff_from_warm_checkpoint=(
            None if handoff_from_warm_checkpoint is None else Path(handoff_from_warm_checkpoint)
        ),
    )
    artifacts = ArtifactConfig(
        tag=str(tag),
        output_json=Path(output_json),
        output_pdf=Path(output_pdf),
        handoff_json=Path(handoff_json),
        warm_checkpoint_json=Path(warm_checkpoint_json),
        warm_cutover_json=Path(warm_cutover_json),
        replay_output_json=Path(replay_output_json),
        replay_output_csv=Path(replay_output_csv),
        replay_output_md=Path(replay_output_md),
        replay_output_log=Path(replay_output_log),
        workflow_log=Path(workflow_log),
        skip_pdf=bool(getattr(args, "skip_pdf", False)),
    )
    gates = GateConfig(
        ecut_1=float(
            _resolve_with_default(
                name="ecut_1",
                raw=getattr(args, "ecut_1", None),
                default=1e-1,
                provenance=provenance,
                default_source="run_guide.ecut_1=1e-1",
            )
        ),
        ecut_2=float(
            _resolve_with_default(
                name="ecut_2",
                raw=getattr(args, "ecut_2", None),
                default=1e-4,
                provenance=provenance,
                default_source="run_guide.ecut_2=1e-4",
            )
        ),
    )
    return StagedHHConfig(
        physics=physics,
        warm_start=warm_start,
        seed_refine=seed_refine,
        adapt=adapt,
        replay=replay,
        dynamics=dynamics,
        fixed_final_state=fixed_final_state,
        circuit_metrics=circuit_metrics,
        warm_checkpoint=warm_checkpoint,
        artifacts=artifacts,
        gates=gates,
        smoke_test_intentionally_weak=bool(getattr(args, "smoke_test_intentionally_weak", False)),
        default_provenance=dict(provenance),
    )


def _build_hh_context(cfg: StagedHHConfig) -> tuple[Any, np.ndarray, list[str], dict[str, complex], np.ndarray]:
    physics = cfg.physics
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=int(physics.L),
        J=float(physics.t),
        U=float(physics.u),
        omega0=float(physics.omega0),
        g=float(physics.g_ep),
        n_ph_max=int(physics.n_ph_max),
        boson_encoding=str(physics.boson_encoding),
        v_t=None,
        v0=float(physics.dv),
        t_eval=None,
        include_zero_point=True,
        repr_mode="JW",
        indexing=str(physics.ordering),
        pbc=(str(physics.boundary).strip().lower() == "periodic"),
    )
    ordered_labels_exyz, coeff_map_exyz = hc_pipeline._collect_hardcoded_terms_exyz(h_poly)
    hmat = hc_pipeline._build_hamiltonian_matrix(coeff_map_exyz)
    psi_hf = hc_pipeline._normalize_state(
        np.asarray(
            hubbard_holstein_reference_state(
                dims=int(physics.L),
                num_particles=(int(physics.sector_n_up), int(physics.sector_n_dn)),
                n_ph_max=int(physics.n_ph_max),
                boson_encoding=str(physics.boson_encoding),
                indexing=str(physics.ordering),
            ),
            dtype=complex,
        ).reshape(-1)
    )
    return h_poly, np.asarray(hmat, dtype=complex), list(ordered_labels_exyz), dict(coeff_map_exyz), psi_hf


def _handoff_continuation_meta(adapt_payload: Mapping[str, Any]) -> dict[str, Any]:
    continuation = adapt_payload.get("continuation", {})
    if not isinstance(continuation, Mapping):
        continuation = {}
    return {
        "continuation_mode": str(adapt_payload.get("continuation_mode", continuation.get("mode", "legacy"))),
        "continuation_scaffold": (
            dict(adapt_payload.get("scaffold_fingerprint_lite", {}))
            if isinstance(adapt_payload.get("scaffold_fingerprint_lite", {}), Mapping)
            else None
        ),
        "optimizer_memory": (
            dict(continuation.get("optimizer_memory", {}))
            if isinstance(continuation.get("optimizer_memory", {}), Mapping)
            else None
        ),
        "selected_generator_metadata": (
            [dict(x) for x in continuation.get("selected_generator_metadata", [])]
            if isinstance(continuation.get("selected_generator_metadata", []), Sequence)
            else None
        ),
        "generator_split_events": (
            [dict(x) for x in continuation.get("generator_split_events", [])]
            if isinstance(continuation.get("generator_split_events", []), Sequence)
            else None
        ),
        "motif_library": (
            dict(continuation.get("motif_library", {}))
            if isinstance(continuation.get("motif_library", {}), Mapping)
            else None
        ),
        "motif_usage": (
            dict(continuation.get("motif_usage", {}))
            if isinstance(continuation.get("motif_usage", {}), Mapping)
            else None
        ),
        "symmetry_mitigation": (
            dict(continuation.get("symmetry_mitigation", {}))
            if isinstance(continuation.get("symmetry_mitigation", {}), Mapping)
            else None
        ),
        "rescue_history": (
            [dict(x) for x in continuation.get("rescue_history", [])]
            if isinstance(continuation.get("rescue_history", []), Sequence)
            else None
        ),
        "prune_summary": (
            dict(adapt_payload.get("prune_summary", {}))
            if isinstance(adapt_payload.get("prune_summary", {}), Mapping)
            else None
        ),
        "pre_prune_scaffold": (
            dict(adapt_payload.get("pre_prune_scaffold", {}))
            if isinstance(adapt_payload.get("pre_prune_scaffold", {}), Mapping)
            else None
        ),
    }


def _infer_replay_family_from_operator_labels(labels: Any) -> tuple[str | None, str | None]:
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        return None, None
    families: set[str] = set()
    for raw_label in labels:
        label = str(raw_label).strip()
        family: str | None = None
        if label.startswith("hh_termwise_"):
            family = "full_meta"
        elif ":" in label:
            family = _canonical_replay_family(label.split(":", 1)[0])
        if family is not None:
            families.add(str(family))
    if len(families) == 1:
        return next(iter(families)), "adapt_payload.operators"
    if len(families) > 1:
        return None, "adapt_payload.operators(mixed)"
    return None, None


def _infer_handoff_adapt_pool(cfg: StagedHHConfig, adapt_payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    continuation = adapt_payload.get("continuation", {})
    if not isinstance(continuation, Mapping):
        continuation = {}

    metadata_records = continuation.get("selected_generator_metadata", [])
    if isinstance(metadata_records, Sequence) and not isinstance(metadata_records, (str, bytes)):
        selected_families = sorted(
            {
                _canonical_replay_family(rec.get("family_id"))
                for rec in metadata_records
                if isinstance(rec, Mapping)
            }
        )
        selected_families = [x for x in selected_families if x is not None]
        if len(selected_families) == 1:
            return selected_families[0], "continuation.selected_generator_metadata.family_id"
        if len(selected_families) > 1:
            # Mixed canonical families in selected generators are treated as provenance ambiguity;
            # force fallback.
            return None, "continuation.selected_generator_metadata.family_id(mixed)"

    labels_family, labels_source = _infer_replay_family_from_operator_labels(adapt_payload.get("operators", []))
    if labels_source is not None:
        return labels_family, labels_source

    record_sets: list[Any] = []
    motif_library = continuation.get("motif_library", {})
    if isinstance(motif_library, Mapping):
        record_sets.append(motif_library.get("records", []))

    for records in record_sets:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            continue
        families = sorted(
            {
                _canonical_replay_family(rec.get("family_id"))
                for rec in records
                if isinstance(rec, Mapping)
            }
        )
        families = [x for x in families if x is not None]
        if len(families) == 1:
            return str(families[0]), "continuation.motif_library.records"

    raw_pool = adapt_payload.get("pool_type")
    raw_pool2 = _canonical_replay_family(raw_pool)
    if raw_pool2 is not None:
        return str(raw_pool2), "adapt_payload.pool_type"

    direct_pool = _canonical_replay_family(cfg.adapt.pool)
    if direct_pool is not None:
        return direct_pool, "cfg.adapt.pool"
    return None, None


def _canonical_replay_family(raw: Any) -> str | None:
    if raw is None:
        return None
    val = str(raw).strip().lower()
    return val if val in replay_mod.EXPLICIT_FAMILIES else None


def _seed_policy_for_handoff_state(raw_state: str, raw_policy: Any) -> tuple[str, str]:
    policy = str(raw_policy).strip().lower()
    if policy not in replay_mod.REPLAY_SEED_POLICIES:
        raise ValueError(f"Invalid replay_seed_policy '{policy}'.")
    if policy == "auto":
        if raw_state == _PREPARED_STATE:
            return policy, "residual_only"
        if raw_state == _REFERENCE_STATE:
            return policy, "scaffold_plus_zero"
        raise ValueError(
            "Cannot resolve handoff replay_seed_policy='auto': expected handoff_state_kind "
            "prepared_state or reference_state."
        )
    return policy, policy


def _build_replay_contract(
    cfg: StagedHHConfig,
    handoff_adapt_pool: str | None,
    handoff_adapt_pool_source: str | None = None,
) -> dict[str, Any]:
    handoff_state_kind = _PREPARED_STATE
    requested = str(cfg.replay.generator_family).strip().lower()
    fallback_family = _canonical_replay_family(cfg.replay.fallback_family)
    if fallback_family is None:
        raise ValueError(f"Invalid fallback_family '{cfg.replay.fallback_family}' in replay config.")

    if requested == "match_adapt":
        resolved = handoff_adapt_pool or fallback_family
        source = handoff_adapt_pool_source if handoff_adapt_pool is not None else "fallback_family"
        fallback_used = handoff_adapt_pool is None
        requested_field = "match_adapt"
    else:
        requested_canon = _canonical_replay_family(requested)
        if requested_canon is None:
            raise ValueError(f"Invalid replay generator_family '{requested}' in config.")
        resolved = requested_canon
        source = "cli.generator_family"
        fallback_used = False
        requested_field = requested_canon

    requested_seed_policy, resolved_seed_policy = _seed_policy_for_handoff_state(
        handoff_state_kind,
        cfg.replay.replay_seed_policy,
    )

    return {
        "contract_version": int(replay_mod.REPLAY_CONTRACT_VERSION),
        "generator_family": {
            "requested": requested_field,
            "resolved": resolved,
            "resolution_source": source,
            "fallback_family": fallback_family,
            "fallback_used": bool(fallback_used),
        },
        "seed_policy_requested": str(requested_seed_policy),
        "seed_policy_resolved": str(resolved_seed_policy),
        "handoff_state_kind": _PREPARED_STATE,
        "provenance_source": "explicit",
        "continuation_mode": str(cfg.replay.continuation_mode),
        "contract_seed_hint": "built-in_staged_writer",
    }


def _run_seed_refine_stage(
    cfg: StagedHHConfig,
    *,
    h_poly: Any,
    psi_ref: np.ndarray,
    exact_filtered_energy: float,
) -> tuple[dict[str, Any], np.ndarray, Path]:
    if cfg.seed_refine.family is None:
        raise ValueError("seed refine is disabled for this staged HH config.")

    stage_json = _seed_refine_state_json_path(cfg)
    run_cfg = _build_seed_refine_run_cfg(cfg)
    _append_workflow_log(
        cfg,
        "seed_refine_stage_start",
        family=str(cfg.seed_refine.family),
        reps=int(cfg.seed_refine.reps),
        maxiter=int(cfg.seed_refine.maxiter),
        optimizer=str(cfg.seed_refine.optimizer),
        state_json=str(stage_json),
    )

    family_ctx = replay_mod.build_family_ansatz_context(
        run_cfg,
        psi_ref=np.asarray(psi_ref, dtype=complex).reshape(-1),
        h_poly=h_poly,
        family=str(cfg.seed_refine.family),
        e_exact=float(exact_filtered_energy),
    )
    ansatz = family_ctx["ansatz"]
    seed_theta = np.asarray(family_ctx["seed_theta"], dtype=float)
    t0 = time.perf_counter()
    vqe_res = replay_mod.vqe_minimize(
        h_poly,
        ansatz,
        np.asarray(psi_ref, dtype=complex).reshape(-1),
        restarts=int(run_cfg.restarts),
        seed=int(run_cfg.seed),
        initial_point=seed_theta,
        use_initial_point_first_restart=True,
        method=str(run_cfg.method),
        maxiter=int(run_cfg.maxiter),
        progress_every_s=float(run_cfg.progress_every_s),
        progress_label="hh_staged_seed_refine",
        track_history=False,
        emit_theta_in_progress=False,
        return_best_on_keyboard_interrupt=True,
        spsa_a=float(run_cfg.spsa_a),
        spsa_c=float(run_cfg.spsa_c),
        spsa_alpha=float(run_cfg.spsa_alpha),
        spsa_gamma=float(run_cfg.spsa_gamma),
        spsa_A=float(run_cfg.spsa_A),
        spsa_avg_last=int(run_cfg.spsa_avg_last),
        spsa_eval_repeats=int(run_cfg.spsa_eval_repeats),
        spsa_eval_agg=str(run_cfg.spsa_eval_agg),
        energy_backend=str(run_cfg.energy_backend),
    )
    runtime_s = float(time.perf_counter() - t0)
    theta_best = np.asarray(vqe_res.theta, dtype=float)
    e_best = float(vqe_res.energy)
    vqe_success = bool(vqe_res.success)
    vqe_message = str(vqe_res.message)
    if not bool(vqe_success):
        _append_workflow_log(
            cfg,
            "seed_refine_stage_failed",
            family=str(cfg.seed_refine.family),
            message=str(vqe_message),
            runtime_s=float(runtime_s),
        )
        raise RuntimeError(
            f"Seed refine VQE failed for family '{cfg.seed_refine.family}': {vqe_message}"
        )

    psi_best = np.asarray(ansatz.prepare_state(theta_best, np.asarray(psi_ref, dtype=complex).reshape(-1)), dtype=complex).reshape(-1)
    psi_best = hc_pipeline._normalize_state(psi_best)
    delta_abs = float(abs(e_best - float(exact_filtered_energy)))
    rel_abs = float(delta_abs / max(abs(float(exact_filtered_energy)), 1e-14))
    stop_reason = "converged" if bool(vqe_success) else str(vqe_message)
    payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_staged_seed_refine",
        "settings": {
            "problem": "hh",
            "L": int(cfg.physics.L),
            "t": float(cfg.physics.t),
            "u": float(cfg.physics.u),
            "dv": float(cfg.physics.dv),
            "omega0": float(cfg.physics.omega0),
            "g_ep": float(cfg.physics.g_ep),
            "n_ph_max": int(cfg.physics.n_ph_max),
            "boson_encoding": str(cfg.physics.boson_encoding),
            "ordering": str(cfg.physics.ordering),
            "boundary": str(cfg.physics.boundary),
            "sector_n_up": int(cfg.physics.sector_n_up),
            "sector_n_dn": int(cfg.physics.sector_n_dn),
            "reps": int(cfg.seed_refine.reps),
            "restarts": int(run_cfg.restarts),
            "maxiter": int(cfg.seed_refine.maxiter),
            "method": str(cfg.seed_refine.optimizer),
            "seed": int(run_cfg.seed),
            "energy_backend": str(run_cfg.energy_backend),
            "paop_r": int(run_cfg.paop_r),
            "paop_split_paulis": bool(run_cfg.paop_split_paulis),
            "paop_prune_eps": float(run_cfg.paop_prune_eps),
            "paop_normalization": str(run_cfg.paop_normalization),
        },
        "generator_family": dict(family_ctx["family_info"]),
        "pool": dict(family_ctx["pool_meta"]),
        "seed_baseline": {
            "theta_policy": "all_zero",
            "energy": float(family_ctx["seed_energy"]),
            "abs_delta_e": float(family_ctx["seed_delta_abs"]),
            "relative_error_abs": float(family_ctx["seed_relative_abs"]),
        },
        "exact": {"E_exact_sector": float(exact_filtered_energy)},
        "vqe": {
            "success": bool(vqe_success),
            "message": str(vqe_message),
            "method": str(cfg.seed_refine.optimizer),
            "energy": float(e_best),
            "abs_delta_e": float(delta_abs),
            "relative_error_abs": float(rel_abs),
            "best_restart": int(vqe_res.best_restart),
            "nfev": int(vqe_res.nfev),
            "nit": int(vqe_res.nit),
            "num_parameters": int(ansatz.num_parameters),
            "runtime_s": float(runtime_s),
            "stop_reason": str(stop_reason),
        },
        "initial_state": {
            "source": f"warm_start_{cfg.warm_start.ansatz_name}",
            "nq_total": int(family_ctx["nq"]),
            "handoff_state_kind": "prepared_state",
            "amplitudes_qn_to_q0": hc_pipeline._state_to_amplitudes_qn_to_q0(np.asarray(psi_ref, dtype=complex).reshape(-1)),
        },
        "best_state": {
            "amplitudes_qn_to_q0": hc_pipeline._state_to_amplitudes_qn_to_q0(psi_best),
            "best_theta": [float(x) for x in theta_best.tolist()],
        },
        "state_json": str(stage_json),
    }
    seed_provenance = _build_seed_provenance(cfg, payload)
    write_handoff_state_bundle(
        path=stage_json,
        psi_state=psi_best,
        cfg=_handoff_bundle_cfg(cfg),
        source="seed_refine_vqe",
        exact_energy=float(exact_filtered_energy),
        energy=float(e_best),
        delta_E_abs=float(delta_abs),
        relative_error_abs=float(rel_abs),
        meta={
            "pipeline": "hh_staged_noiseless",
            "workflow_tag": str(cfg.artifacts.tag),
            "stage": "seed_refine_vqe",
            "stage_chain": ["hf_reference", "warm_start_hva", "seed_refine_vqe"],
        },
        handoff_state_kind="prepared_state",
        vqe_payload=dict(payload["vqe"]),
        seed_provenance=seed_provenance,
    )
    _append_workflow_log(
        cfg,
        "seed_refine_stage_complete",
        family=str(cfg.seed_refine.family),
        energy=float(e_best),
        exact_energy=float(exact_filtered_energy),
        delta_abs=float(delta_abs),
        state_json=str(stage_json),
    )
    return payload, psi_best, stage_json


def _write_adapt_handoff(
    cfg: StagedHHConfig,
    adapt_payload: Mapping[str, Any],
    psi_adapt: np.ndarray,
    *,
    seed_provenance: Mapping[str, Any] | None = None,
) -> None:
    exact_energy = float(adapt_payload.get("exact_gs_energy", float("nan")))
    energy = float(adapt_payload.get("energy", float("nan")))
    continuation_meta = _handoff_continuation_meta(adapt_payload)
    handoff_adapt_pool, handoff_adapt_pool_source = _infer_handoff_adapt_pool(cfg, adapt_payload)
    stage_chain = ["hf_reference", "warm_start_hva"]
    if cfg.seed_refine.family is not None:
        stage_chain.append("seed_refine_vqe")
    stage_chain.extend(["adapt_vqe", "matched_family_replay"])
    write_handoff_state_bundle(
        path=cfg.artifacts.handoff_json,
        psi_state=np.asarray(psi_adapt, dtype=complex).reshape(-1),
        cfg=_handoff_bundle_cfg(cfg),
        source="adapt_vqe",
        exact_energy=float(exact_energy),
        energy=float(energy),
        delta_E_abs=float(adapt_payload.get("abs_delta_e", abs(energy - exact_energy))),
        relative_error_abs=float(_relative_error_abs(energy, exact_energy)),
        meta={
            "pipeline": "hh_staged_noiseless",
            "workflow_tag": str(cfg.artifacts.tag),
            "stage_chain": stage_chain,
        },
        adapt_operators=[str(x) for x in adapt_payload.get("operators", [])],
        adapt_optimal_point=[float(x) for x in adapt_payload.get("optimal_point", [])],
        adapt_pool_type=handoff_adapt_pool,
        settings_adapt_pool=handoff_adapt_pool,
        handoff_state_kind="prepared_state",
        continuation_mode=str(continuation_meta.get("continuation_mode", cfg.adapt.continuation_mode)),
        continuation_scaffold=continuation_meta.get("continuation_scaffold"),
        replay_contract=_build_replay_contract(
            cfg,
            handoff_adapt_pool=handoff_adapt_pool,
            handoff_adapt_pool_source=handoff_adapt_pool_source,
        ),
        optimizer_memory=continuation_meta.get("optimizer_memory"),
        selected_generator_metadata=continuation_meta.get("selected_generator_metadata"),
        generator_split_events=continuation_meta.get("generator_split_events"),
        motif_library=continuation_meta.get("motif_library"),
        motif_usage=continuation_meta.get("motif_usage"),
        symmetry_mitigation=continuation_meta.get("symmetry_mitigation"),
        rescue_history=continuation_meta.get("rescue_history"),
        prune_summary=continuation_meta.get("prune_summary"),
        pre_prune_scaffold=continuation_meta.get("pre_prune_scaffold"),
        replay_contract_hint={
            "generator_family": str(cfg.replay.generator_family),
            "fallback_family": str(cfg.replay.fallback_family),
            "replay_seed_policy": str(cfg.replay.replay_seed_policy),
            "replay_continuation_mode": str(cfg.replay.continuation_mode),
        },
        seed_provenance=(dict(seed_provenance) if isinstance(seed_provenance, Mapping) else None),
    )


def _staged_ansatz_manifest(cfg: StagedHHConfig) -> str:
    parts = [f"warm: {cfg.warm_start.ansatz_name}"]
    if cfg.seed_refine.family is not None:
        parts.append(f"seed refine: {cfg.seed_refine.family}")
    parts.append(f"ADAPT: {cfg.adapt.continuation_mode}")
    parts.append(
        "final: matched-family replay"
        if bool(cfg.replay.enabled)
        else "final: replay disabled"
    )
    return "; ".join(parts)


def _build_hh_warm_ansatz(cfg: StagedHHConfig) -> Any:
    common_kwargs = {
        "dims": int(cfg.physics.L),
        "J": float(cfg.physics.t),
        "U": float(cfg.physics.u),
        "omega0": float(cfg.physics.omega0),
        "g": float(cfg.physics.g_ep),
        "n_ph_max": int(cfg.physics.n_ph_max),
        "boson_encoding": str(cfg.physics.boson_encoding),
        "reps": int(cfg.warm_start.reps),
        "repr_mode": "JW",
        "indexing": str(cfg.physics.ordering),
        "pbc": (str(cfg.physics.boundary).strip().lower() == "periodic"),
    }
    ansatz_name = str(cfg.warm_start.ansatz_name).strip().lower()
    if ansatz_name == "hh_hva":
        return HubbardHolsteinLayerwiseAnsatz(**common_kwargs)
    if ansatz_name == "hh_hva_tw":
        return HubbardHolsteinTermwiseAnsatz(**common_kwargs)
    if ansatz_name == "hh_hva_ptw":
        return HubbardHolsteinPhysicalTermwiseAnsatz(**common_kwargs)
    raise ValueError(f"Unsupported staged HH warm ansatz {cfg.warm_start.ansatz_name!r}.")


def _assemble_stage_circuit_contexts(
    *,
    cfg: StagedHHConfig,
    psi_hf: np.ndarray,
    warm_payload: Mapping[str, Any],
    adapt_diagnostics: Mapping[str, Any] | None,
    replay_diagnostics: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any] | None]:
    warm_ctx: dict[str, Any] | None = None
    adapt_ctx: dict[str, Any] | None = None
    replay_ctx: dict[str, Any] | None = None

    warm_theta_raw = warm_payload.get("optimal_point", None)
    if isinstance(warm_theta_raw, Sequence) and not isinstance(warm_theta_raw, (str, bytes)):
        warm_theta = np.asarray([float(x) for x in warm_theta_raw], dtype=float)
        if int(warm_theta.size) > 0:
            warm_ctx = {
                "ansatz": _build_hh_warm_ansatz(cfg),
                "theta": np.asarray(warm_theta, dtype=float).copy(),
                "reference_state": np.asarray(psi_hf, dtype=complex).reshape(-1).copy(),
                "num_qubits": int(round(math.log2(int(np.asarray(psi_hf).size)))),
                "ansatz_name": str(cfg.warm_start.ansatz_name),
            }

    if isinstance(adapt_diagnostics, Mapping) and adapt_diagnostics:
        adapt_ctx = {
            "selected_ops": list(adapt_diagnostics.get("selected_ops", [])),
            "theta": np.asarray(adapt_diagnostics.get("theta", []), dtype=float).copy(),
            "reference_state": np.asarray(adapt_diagnostics.get("reference_state"), dtype=complex).reshape(-1).copy(),
            "num_qubits": int(adapt_diagnostics.get("num_qubits", 0)),
            "pool_type": str(adapt_diagnostics.get("pool_type", cfg.adapt.pool or cfg.adapt.continuation_mode)),
            "continuation_mode": str(adapt_diagnostics.get("continuation_mode", cfg.adapt.continuation_mode)),
        }

    if isinstance(replay_diagnostics, Mapping) and replay_diagnostics:
        replay_ctx = {
            "ansatz": replay_diagnostics.get("ansatz"),
            "theta": np.asarray(replay_diagnostics.get("best_theta", []), dtype=float).copy(),
            "seed_theta": np.asarray(replay_diagnostics.get("seed_theta", []), dtype=float).copy(),
            "reference_state": np.asarray(replay_diagnostics.get("reference_state"), dtype=complex).reshape(-1).copy(),
            "num_qubits": int(replay_diagnostics.get("num_qubits", 0)),
            "family_info": dict(replay_diagnostics.get("family_info", {})),
            "handoff_state_kind": str(replay_diagnostics.get("handoff_state_kind", "prepared_state")),
            "provenance_source": str(replay_diagnostics.get("provenance_source", "explicit")),
            "resolved_seed_policy": str(replay_diagnostics.get("resolved_seed_policy", cfg.replay.replay_seed_policy)),
        }

    return {
        "warm_circuit_context": warm_ctx,
        "adapt_circuit_context": adapt_ctx,
        "replay_circuit_context": replay_ctx,
    }


def _workflow_stage_chain(cfg: StagedHHConfig, *, fixed_mode: bool) -> list[str]:
    if fixed_mode:
        chain = ["hf_reference", "fixed_final_state_import"]
    else:
        chain = ["hf_reference", "warm_start_hva"]
        if cfg.seed_refine.family is not None:
            chain.append("seed_refine_vqe")
        chain.append("adapt_vqe")
        if bool(cfg.replay.enabled):
            chain.append("matched_family_replay")
    if bool(cfg.dynamics.enabled):
        chain.append("final_only_noiseless_dynamics")
    return chain


def _terminal_reference_energy(stage_result: StageExecutionResult, cfg: StagedHHConfig) -> tuple[float, str]:
    if bool(cfg.replay.enabled) and not bool(stage_result.replay_payload.get("skipped", False)):
        replay_exact = float(stage_result.replay_payload.get("exact", {}).get("E_exact_sector", float("nan")))
        if math.isfinite(replay_exact):
            return replay_exact, "stage_pipeline.conventional_replay.exact_energy"
    adapt_exact = float(stage_result.adapt_payload.get("exact_gs_energy", float("nan")))
    return adapt_exact, "stage_pipeline.adapt_vqe.exact_energy"


def run_stage_pipeline(cfg: StagedHHConfig) -> StageExecutionResult:
    h_poly, hmat, ordered_labels_exyz, coeff_map_exyz, psi_hf = _build_hh_context(cfg)
    adapt_diagnostics: dict[str, Any] = {}
    replay_diagnostics: dict[str, Any] = {}
    _append_workflow_log(
        cfg,
        "stage_pipeline_start",
        tag=str(cfg.artifacts.tag),
        output_json=str(cfg.artifacts.output_json),
        warm_checkpoint_json=str(cfg.artifacts.warm_checkpoint_json),
        warm_cutover_json=str(cfg.artifacts.warm_cutover_json),
    )
    if cfg.fixed_final_state is not None:
        requested_source_json = Path(cfg.fixed_final_state.json_path)
        source_json, raw_payload, resolved_via = _resolve_fixed_final_state_payload(
            requested_source_json
        )
        psi_final, fixed_import, warm_payload, adapt_payload, replay_payload = _build_fixed_final_state_import(
            cfg,
            source_json=source_json,
            raw_payload=raw_payload,
            nq_total=_hh_nq_total(cfg.physics.L, cfg.physics.n_ph_max, cfg.physics.boson_encoding),
            requested_source_json=requested_source_json,
            resolved_via=resolved_via,
        )
        _write_fixed_final_state_sidecars(
            cfg,
            psi_final=np.asarray(psi_final, dtype=complex).reshape(-1),
            fixed_import=fixed_import,
            replay_payload=replay_payload,
        )
        _append_workflow_log(
            cfg,
            "fixed_final_state_import",
            source_json=str(requested_source_json),
            resolved_json=str(source_json),
            resolved_via=resolved_via,
            strict_match=bool(cfg.fixed_final_state.strict_match),
            mismatch_count=int(len(fixed_import.get("mismatches", []))),
        )
        return StageExecutionResult(
            h_poly=h_poly,
            hmat=np.asarray(hmat, dtype=complex),
            ordered_labels_exyz=list(ordered_labels_exyz),
            coeff_map_exyz=dict(coeff_map_exyz),
            nq_total=int(_hh_nq_total(cfg.physics.L, cfg.physics.n_ph_max, cfg.physics.boson_encoding)),
            psi_hf=np.asarray(psi_hf, dtype=complex).reshape(-1),
            psi_warm=np.asarray(psi_final, dtype=complex).reshape(-1),
            psi_adapt=np.asarray(psi_final, dtype=complex).reshape(-1),
            psi_final=np.asarray(psi_final, dtype=complex).reshape(-1),
            warm_payload=dict(warm_payload),
            adapt_payload=dict(adapt_payload),
            replay_payload=dict(replay_payload),
            fixed_final_state_import=dict(fixed_import),
            warm_circuit_context=None,
            adapt_circuit_context=None,
            replay_circuit_context=None,
        )

    warm_payload, psi_warm, warm_seed_json = _run_warm_start_stage(
        cfg,
        h_poly=h_poly,
        psi_hf=np.asarray(psi_hf, dtype=complex).reshape(-1),
    )
    adapt_seed_json = Path(warm_seed_json)
    psi_seed_refine: np.ndarray | None = None
    seed_refine_payload: dict[str, Any] | None = None
    if cfg.seed_refine.family is not None:
        seed_refine_payload, psi_seed_refine, adapt_seed_json = _run_seed_refine_stage(
            cfg,
            h_poly=h_poly,
            psi_ref=np.asarray(psi_warm, dtype=complex).reshape(-1),
            exact_filtered_energy=float(warm_payload.get("exact_filtered_energy", float("nan"))),
        )
    _append_workflow_log(
        cfg,
        "adapt_seed_checkpoint_selected",
        checkpoint_json=str(adapt_seed_json),
        source_stage=("seed_refine_vqe" if seed_refine_payload is not None else "warm_start_hva"),
        energy=float(
            seed_refine_payload.get("vqe", {}).get("energy", warm_payload.get("energy", float("nan")))
            if isinstance(seed_refine_payload, Mapping)
            else warm_payload.get("energy", float("nan"))
        ),
        exact_filtered_energy=float(warm_payload.get("exact_filtered_energy", float("nan"))),
        cutoff_triggered=bool(warm_payload.get("cutoff_triggered", False)),
        cutoff_reason=warm_payload.get("cutoff_reason"),
    )

    adapt_payload, psi_adapt = adapt_mod._run_hardcoded_adapt_vqe(
        h_poly=h_poly,
        num_sites=int(cfg.physics.L),
        ordering=str(cfg.physics.ordering),
        problem="hh",
        adapt_pool=cfg.adapt.pool,
        t=float(cfg.physics.t),
        u=float(cfg.physics.u),
        dv=float(cfg.physics.dv),
        boundary=str(cfg.physics.boundary),
        omega0=float(cfg.physics.omega0),
        g_ep=float(cfg.physics.g_ep),
        n_ph_max=int(cfg.physics.n_ph_max),
        boson_encoding=str(cfg.physics.boson_encoding),
        max_depth=int(cfg.adapt.max_depth),
        eps_grad=float(cfg.adapt.eps_grad),
        eps_energy=float(cfg.adapt.eps_energy),
        adapt_drop_floor=cfg.adapt.drop_floor,
        adapt_drop_patience=cfg.adapt.drop_patience,
        adapt_drop_min_depth=cfg.adapt.drop_min_depth,
        adapt_grad_floor=cfg.adapt.grad_floor,
        maxiter=int(cfg.adapt.maxiter),
        seed=int(cfg.adapt.seed),
        adapt_inner_optimizer=str(cfg.adapt.inner_optimizer),
        adapt_spsa_a=float(cfg.adapt.spsa_a),
        adapt_spsa_c=float(cfg.adapt.spsa_c),
        adapt_spsa_alpha=float(cfg.adapt.spsa_alpha),
        adapt_spsa_gamma=float(cfg.adapt.spsa_gamma),
        adapt_spsa_A=float(cfg.adapt.spsa_A),
        adapt_spsa_avg_last=int(cfg.adapt.spsa_avg_last),
        adapt_spsa_eval_repeats=int(cfg.adapt.spsa_eval_repeats),
        adapt_spsa_eval_agg=str(cfg.adapt.spsa_eval_agg),
        adapt_spsa_callback_every=int(cfg.adapt.spsa_callback_every),
        adapt_spsa_progress_every_s=float(cfg.adapt.spsa_progress_every_s),
        allow_repeats=bool(cfg.adapt.allow_repeats),
        finite_angle_fallback=bool(cfg.adapt.finite_angle_fallback),
        finite_angle=float(cfg.adapt.finite_angle),
        finite_angle_min_improvement=float(cfg.adapt.finite_angle_min_improvement),
        paop_r=int(cfg.adapt.paop_r),
        paop_split_paulis=bool(cfg.adapt.paop_split_paulis),
        paop_prune_eps=float(cfg.adapt.paop_prune_eps),
        paop_normalization=str(cfg.adapt.paop_normalization),
        disable_hh_seed=bool(cfg.adapt.disable_hh_seed),
        adapt_ref_json=Path(adapt_seed_json),
        adapt_reopt_policy=str(cfg.adapt.reopt_policy),
        adapt_window_size=int(cfg.adapt.window_size),
        adapt_window_topk=int(cfg.adapt.window_topk),
        adapt_full_refit_every=int(cfg.adapt.full_refit_every),
        adapt_final_full_refit=bool(cfg.adapt.final_full_refit),
        adapt_beam_live_branches=int(cfg.adapt.beam_live_branches),
        adapt_beam_children_per_parent=cfg.adapt.beam_children_per_parent,
        adapt_beam_terminated_keep=cfg.adapt.beam_terminated_keep,
        adapt_continuation_mode=str(cfg.adapt.continuation_mode),
        phase1_lambda_F=float(cfg.adapt.phase1_lambda_F),
        phase1_lambda_compile=float(cfg.adapt.phase1_lambda_compile),
        phase1_lambda_measure=float(cfg.adapt.phase1_lambda_measure),
        phase1_lambda_leak=float(cfg.adapt.phase1_lambda_leak),
        phase1_score_z_alpha=float(cfg.adapt.phase1_score_z_alpha),
        phase1_probe_max_positions=int(cfg.adapt.phase1_probe_max_positions),
        phase1_plateau_patience=int(cfg.adapt.phase1_plateau_patience),
        phase1_trough_margin_ratio=float(cfg.adapt.phase1_trough_margin_ratio),
        phase1_prune_enabled=bool(cfg.adapt.phase1_prune_enabled),
        phase1_prune_fraction=float(cfg.adapt.phase1_prune_fraction),
        phase1_prune_max_candidates=int(cfg.adapt.phase1_prune_max_candidates),
        phase1_prune_max_regression=float(cfg.adapt.phase1_prune_max_regression),
        phase3_motif_source_json=cfg.adapt.phase3_motif_source_json,
        phase3_symmetry_mitigation_mode=str(cfg.adapt.phase3_symmetry_mitigation_mode),
        phase3_enable_rescue=bool(cfg.adapt.phase3_enable_rescue),
        phase3_lifetime_cost_mode=str(cfg.adapt.phase3_lifetime_cost_mode),
        phase3_runtime_split_mode=str(cfg.adapt.phase3_runtime_split_mode),
        diagnostics_out=adapt_diagnostics,
    )
    adapt_payload["adapt_ref_json"] = str(adapt_seed_json)
    adapt_payload["initial_state_source"] = "adapt_ref_json"
    _append_workflow_log(
        cfg,
        "adapt_seed_checkpoint_used",
        checkpoint_json=str(adapt_seed_json),
        adapt_ref_base_depth=adapt_payload.get("adapt_ref_base_depth"),
        exact_gs_energy=adapt_payload.get("exact_gs_energy"),
    )

    _write_adapt_handoff(
        cfg,
        adapt_payload,
        np.asarray(psi_adapt, dtype=complex).reshape(-1),
        seed_provenance=_build_seed_provenance(cfg, seed_refine_payload),
    )
    nq_total = _hh_nq_total(cfg.physics.L, cfg.physics.n_ph_max, cfg.physics.boson_encoding)
    if bool(cfg.replay.enabled):
        replay_cfg = replay_mod.RunConfig(
            adapt_input_json=Path(cfg.artifacts.handoff_json),
            output_json=Path(cfg.artifacts.replay_output_json),
            output_csv=Path(cfg.artifacts.replay_output_csv),
            output_md=Path(cfg.artifacts.replay_output_md),
            output_log=Path(cfg.artifacts.replay_output_log),
            tag=f"{cfg.artifacts.tag}_replay",
            generator_family=str(cfg.replay.generator_family),
            fallback_family=str(cfg.replay.fallback_family),
            legacy_paop_key=str(cfg.replay.legacy_paop_key),
            replay_seed_policy=str(cfg.replay.replay_seed_policy),
            replay_continuation_mode=str(cfg.replay.continuation_mode),
            L=int(cfg.physics.L),
            t=float(cfg.physics.t),
            u=float(cfg.physics.u),
            dv=float(cfg.physics.dv),
            omega0=float(cfg.physics.omega0),
            g_ep=float(cfg.physics.g_ep),
            n_ph_max=int(cfg.physics.n_ph_max),
            boson_encoding=str(cfg.physics.boson_encoding),
            ordering=str(cfg.physics.ordering),
            boundary=str(cfg.physics.boundary),
            sector_n_up=int(cfg.physics.sector_n_up),
            sector_n_dn=int(cfg.physics.sector_n_dn),
            reps=int(cfg.replay.reps),
            restarts=int(cfg.replay.restarts),
            maxiter=int(cfg.replay.maxiter),
            method=str(cfg.replay.method),
            seed=int(cfg.replay.seed),
            energy_backend=str(cfg.replay.energy_backend),
            progress_every_s=float(cfg.replay.progress_every_s),
            wallclock_cap_s=int(cfg.replay.wallclock_cap_s),
            paop_r=int(cfg.replay.paop_r),
            paop_split_paulis=bool(cfg.replay.paop_split_paulis),
            paop_prune_eps=float(cfg.replay.paop_prune_eps),
            paop_normalization=str(cfg.replay.paop_normalization),
            spsa_a=float(cfg.replay.spsa_a),
            spsa_c=float(cfg.replay.spsa_c),
            spsa_alpha=float(cfg.replay.spsa_alpha),
            spsa_gamma=float(cfg.replay.spsa_gamma),
            spsa_A=float(cfg.replay.spsa_A),
            spsa_avg_last=int(cfg.replay.spsa_avg_last),
            spsa_eval_repeats=int(cfg.replay.spsa_eval_repeats),
            spsa_eval_agg=str(cfg.replay.spsa_eval_agg),
            replay_freeze_fraction=float(cfg.replay.replay_freeze_fraction),
            replay_unfreeze_fraction=float(cfg.replay.replay_unfreeze_fraction),
            replay_full_fraction=float(cfg.replay.replay_full_fraction),
            replay_qn_spsa_refresh_every=int(cfg.replay.replay_qn_spsa_refresh_every),
            replay_qn_spsa_refresh_mode=str(cfg.replay.replay_qn_spsa_refresh_mode),
            phase3_symmetry_mitigation_mode=str(cfg.replay.phase3_symmetry_mitigation_mode),
        )
        try:
            replay_payload = replay_mod.run(replay_cfg, diagnostics_out=replay_diagnostics)
        except TypeError as exc:
            if "diagnostics_out" not in str(exc):
                raise
            replay_payload = replay_mod.run(replay_cfg)
            replay_diagnostics = {}
        best_state = replay_payload.get("best_state", {})
        if not isinstance(best_state, Mapping):
            raise ValueError("Replay payload missing best_state block.")
        amplitudes = best_state.get("amplitudes_qn_to_q0", None)
        if not isinstance(amplitudes, Mapping):
            raise ValueError("Replay payload missing best_state.amplitudes_qn_to_q0.")
        psi_final = hc_pipeline._state_from_amplitudes_qn_to_q0(amplitudes, int(nq_total))
        psi_final = hc_pipeline._normalize_state(np.asarray(psi_final, dtype=complex).reshape(-1))
    else:
        replay_payload = {
            "generator_family": {
                "requested": str(cfg.replay.generator_family),
                "resolved": None,
                "fallback": str(cfg.replay.fallback_family),
            },
            "seed_baseline": {},
            "replay_contract": {
                "continuation_mode": str(cfg.replay.continuation_mode),
                "seed_policy_requested": str(cfg.replay.replay_seed_policy),
            },
            "vqe": {},
            "exact": {},
            "skipped": True,
            "skip_reason": "run_replay_false",
        }
        psi_final = hc_pipeline._normalize_state(np.asarray(psi_adapt, dtype=complex).reshape(-1))
        _append_workflow_log(
            cfg,
            "replay_stage_skipped",
            reason="run_replay_false",
            adapt_handoff_json=str(cfg.artifacts.handoff_json),
        )
    circuit_contexts = _assemble_stage_circuit_contexts(
        cfg=cfg,
        psi_hf=np.asarray(psi_hf, dtype=complex).reshape(-1),
        warm_payload=warm_payload,
        adapt_diagnostics=adapt_diagnostics,
        replay_diagnostics=replay_diagnostics,
    )

    return StageExecutionResult(
        h_poly=h_poly,
        hmat=np.asarray(hmat, dtype=complex),
        ordered_labels_exyz=list(ordered_labels_exyz),
        coeff_map_exyz=dict(coeff_map_exyz),
        nq_total=int(nq_total),
        psi_hf=np.asarray(psi_hf, dtype=complex).reshape(-1),
        psi_warm=np.asarray(psi_warm, dtype=complex).reshape(-1),
        psi_adapt=np.asarray(psi_adapt, dtype=complex).reshape(-1),
        psi_final=np.asarray(psi_final, dtype=complex).reshape(-1),
        warm_payload=dict(warm_payload),
        adapt_payload=dict(adapt_payload),
        replay_payload=dict(replay_payload),
        psi_seed_refine=(
            None if psi_seed_refine is None else np.asarray(psi_seed_refine, dtype=complex).reshape(-1)
        ),
        seed_refine_payload=(None if seed_refine_payload is None else dict(seed_refine_payload)),
        warm_circuit_context=circuit_contexts["warm_circuit_context"],
        adapt_circuit_context=circuit_contexts["adapt_circuit_context"],
        replay_circuit_context=circuit_contexts["replay_circuit_context"],
    )


def _build_drive_provider(
    *,
    cfg: StagedHHConfig,
    nq_total: int,
    ordered_labels_exyz: Sequence[str],
) -> tuple[Any | None, dict[str, Any] | None, list[str], dict[str, Any] | None]:
    if not bool(cfg.dynamics.enable_drive):
        return None, None, list(ordered_labels_exyz), None
    custom_weights = None
    if str(cfg.dynamics.drive_pattern) == "custom":
        custom_weights = _parse_drive_custom_weights(cfg.dynamics.drive_custom_s)
        if custom_weights is None:
            raise ValueError("--drive-custom-s is required when --drive-pattern custom.")
    drive = build_gaussian_sinusoid_density_drive(
        n_sites=int(cfg.physics.L),
        nq_total=int(nq_total),
        indexing=str(cfg.physics.ordering),
        A=float(cfg.dynamics.drive_A),
        omega=float(cfg.dynamics.drive_omega),
        tbar=float(cfg.dynamics.drive_tbar),
        phi=float(cfg.dynamics.drive_phi),
        pattern_mode=str(cfg.dynamics.drive_pattern),
        custom_weights=custom_weights,
        include_identity=bool(cfg.dynamics.drive_include_identity),
        coeff_tol=0.0,
    )
    drive_labels = set(drive.template.labels_exyz(include_identity=bool(drive.include_identity)))
    ordered = list(ordered_labels_exyz)
    missing = sorted(drive_labels.difference(ordered))
    ordered.extend(missing)
    profile = {
        "A": float(cfg.dynamics.drive_A),
        "omega": float(cfg.dynamics.drive_omega),
        "tbar": float(cfg.dynamics.drive_tbar),
        "phi": float(cfg.dynamics.drive_phi),
        "pattern": str(cfg.dynamics.drive_pattern),
        "custom_weights": custom_weights,
        "include_identity": bool(cfg.dynamics.drive_include_identity),
        "time_sampling": str(cfg.dynamics.drive_time_sampling),
        "t0": float(cfg.dynamics.drive_t0),
    }
    meta = {
        "reference_method": str(reference_method_name(str(cfg.dynamics.drive_time_sampling))),
        "missing_drive_labels_added": int(len(missing)),
        "drive_label_count": int(len(drive_labels)),
    }
    return drive.coeff_map_exyz, meta, ordered, profile


def _run_noiseless_profile(
    *,
    cfg: StagedHHConfig,
    psi_seed: np.ndarray,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    drive_enabled: bool,
    ground_state_reference_energy: float,
    ground_state_reference_source: str,
) -> dict[str, Any]:
    drive_provider = None
    drive_meta = None
    drive_profile = None
    ordered_for_run = list(ordered_labels_exyz)
    if drive_enabled:
        drive_provider, drive_meta, ordered_for_run, drive_profile = _build_drive_provider(
            cfg=cfg,
            nq_total=int(round(math.log2(int(np.asarray(psi_seed).size)))),
            ordered_labels_exyz=ordered_labels_exyz,
        )

    method_payloads: dict[str, Any] = {}
    reference_rows: list[dict[str, Any]] | None = None
    psi_seed_arr = np.asarray(psi_seed, dtype=complex).reshape(-1)
    ground_state_energy = float(ground_state_reference_energy)

    for method in cfg.dynamics.methods:
        rows, _ = hc_pipeline._simulate_trajectory(
            num_sites=int(cfg.physics.L),
            ordering=str(cfg.physics.ordering),
            psi0_legacy_trot=np.asarray(psi_seed_arr, dtype=complex),
            psi0_paop_trot=np.asarray(psi_seed_arr, dtype=complex),
            psi0_hva_trot=np.asarray(psi_seed_arr, dtype=complex),
            legacy_branch_label="replay",
            psi0_exact_ref=np.asarray(psi_seed_arr, dtype=complex),
            fidelity_subspace_basis_v0=np.asarray(psi_seed_arr, dtype=complex).reshape(-1, 1),
            fidelity_subspace_energy_tol=float(cfg.dynamics.fidelity_subspace_energy_tol),
            hmat=np.asarray(hmat, dtype=complex),
            ordered_labels_exyz=list(ordered_for_run),
            coeff_map_exyz=dict(coeff_map_exyz),
            trotter_steps=int(cfg.dynamics.trotter_steps),
            t_final=float(cfg.dynamics.t_final),
            num_times=int(cfg.dynamics.num_times),
            suzuki_order=2,
            drive_coeff_provider_exyz=drive_provider,
            drive_t0=float(cfg.dynamics.drive_t0 if drive_enabled else 0.0),
            drive_time_sampling=str(cfg.dynamics.drive_time_sampling),
            exact_steps_multiplier=(int(cfg.dynamics.exact_steps_multiplier) if drive_enabled else 1),
            propagator=str(method),
            cfqm_stage_exp=str(cfg.dynamics.cfqm_stage_exp),
            cfqm_coeff_drop_abs_tol=float(cfg.dynamics.cfqm_coeff_drop_abs_tol),
            cfqm_normalize=bool(cfg.dynamics.cfqm_normalize),
        )
        rows_with_metrics: list[dict[str, Any]] = []
        for row in rows:
            row_out = dict(row)
            row_out["abs_energy_error_vs_ground_state"] = float(
                abs(float(row_out["energy_total_trotter"]) - ground_state_energy)
            )
            rows_with_metrics.append(row_out)
        reference_rows = rows_with_metrics if reference_rows is None else reference_rows
        final_row = rows_with_metrics[-1]
        final_reference_error = float(
            abs(float(final_row["energy_total_trotter"]) - float(final_row["energy_total_exact"]))
        )
        method_payloads[str(method)] = {
            "propagator": str(method),
            "trajectory": rows_with_metrics,
            "final": {
                "energy_total_trotter": float(final_row["energy_total_trotter"]),
                "energy_total_exact": float(final_row["energy_total_exact"]),
                "abs_energy_total_error": float(final_reference_error),
                "abs_energy_total_error_vs_reference": float(final_reference_error),
                "abs_energy_error_vs_ground_state": float(final_row["abs_energy_error_vs_ground_state"]),
                "fidelity": float(final_row["fidelity"]),
                "doublon_trotter": float(final_row["doublon_trotter"]),
                "doublon_exact": float(final_row["doublon_exact"]),
            },
            "settings": {
                "trotter_steps": int(cfg.dynamics.trotter_steps),
                "num_times": int(cfg.dynamics.num_times),
                "t_final": float(cfg.dynamics.t_final),
                "cfqm_stage_exp": str(cfg.dynamics.cfqm_stage_exp),
                "cfqm_coeff_drop_abs_tol": float(cfg.dynamics.cfqm_coeff_drop_abs_tol),
                "cfqm_normalize": bool(cfg.dynamics.cfqm_normalize),
            },
        }

    assert reference_rows is not None
    return {
        "drive_enabled": bool(drive_enabled),
        "drive_profile": drive_profile,
        "drive_meta": drive_meta,
        "times": [float(row["time"]) for row in reference_rows],
        "ground_state_reference": {
            "energy": float(ground_state_energy),
            "kind": "filtered_sector_ground_state_static",
            "source": str(ground_state_reference_source),
        },
        "reference": {
            "kind": "seeded_exact_reference",
            "initial_state": "psi_final",
            "method": (
                "eigendecomposition"
                if not drive_enabled
                else str(reference_method_name(str(cfg.dynamics.drive_time_sampling)))
            ),
            "energy_total_exact": [float(row["energy_total_exact"]) for row in reference_rows],
            "doublon_exact": [float(row["doublon_exact"]) for row in reference_rows],
        },
        "methods": method_payloads,
    }


def run_noiseless_profiles(stage_result: StageExecutionResult, cfg: StagedHHConfig) -> dict[str, Any]:
    terminal_exact, terminal_source = _terminal_reference_energy(stage_result, cfg)
    profiles = {
        "static": _run_noiseless_profile(
            cfg=cfg,
            psi_seed=stage_result.psi_final,
            hmat=stage_result.hmat,
            ordered_labels_exyz=stage_result.ordered_labels_exyz,
            coeff_map_exyz=stage_result.coeff_map_exyz,
            drive_enabled=False,
            ground_state_reference_energy=terminal_exact,
            ground_state_reference_source=terminal_source,
        )
    }
    if bool(cfg.dynamics.enable_drive):
        profiles["drive"] = _run_noiseless_profile(
            cfg=cfg,
            psi_seed=stage_result.psi_final,
            hmat=stage_result.hmat,
            ordered_labels_exyz=stage_result.ordered_labels_exyz,
            coeff_map_exyz=stage_result.coeff_map_exyz,
            drive_enabled=True,
            ground_state_reference_energy=terminal_exact,
            ground_state_reference_source=terminal_source,
        )
    return {"profiles": profiles}


def _empty_qiskit_circuit(num_qubits: int) -> Any:
    from qiskit import QuantumCircuit

    return QuantumCircuit(int(num_qubits))


def _transpile_target_metadata(cfg: StagedHHConfig) -> dict[str, Any] | None:
    if cfg.circuit_metrics.backend_name is None:
        return None
    return {
        "backend_name": str(cfg.circuit_metrics.backend_name),
        "use_fake_backend": bool(cfg.circuit_metrics.use_fake_backend),
        "optimization_level": int(cfg.circuit_metrics.optimization_level),
        "seed_transpiler": int(cfg.circuit_metrics.seed_transpiler),
        "basis_gates": ["rz", "sx", "x", "cx"],
    }


def _transpile_metrics_or_error(
    cfg: StagedHHConfig,
    *,
    circuit: Any | None,
    enabled: bool = True,
    reason: str | None = None,
) -> dict[str, Any] | None:
    if circuit is None:
        return None
    if cfg.circuit_metrics.backend_name is None:
        return None
    if not bool(enabled):
        return {
            "target": _transpile_target_metadata(cfg),
            "skipped": True,
            "reason": str(reason or "transpile_metrics_disabled"),
        }
    try:
        return transpile_circuit_metrics(
            circuit,
            backend_name=str(cfg.circuit_metrics.backend_name),
            use_fake_backend=bool(cfg.circuit_metrics.use_fake_backend),
            optimization_level=int(cfg.circuit_metrics.optimization_level),
            seed_transpiler=int(cfg.circuit_metrics.seed_transpiler),
        )
    except Exception as exc:
        return {
            "target": _transpile_target_metadata(cfg),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _strip_circuit_objects(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_circuit_objects(val)
            for key, val in value.items()
            if str(key) != "circuit"
        }
    if isinstance(value, list):
        return [_strip_circuit_objects(item) for item in value]
    return value


def build_stage_circuit_report_artifacts(
    stage_result: StageExecutionResult,
    cfg: StagedHHConfig,
) -> dict[str, Any]:
    stage_bundles: dict[str, dict[str, Any] | None] = {
        "warm_start": None,
        "adapt_vqe": None,
        "conventional_replay": None,
    }

    warm_ctx = stage_result.warm_circuit_context
    if isinstance(warm_ctx, Mapping) and warm_ctx.get("ansatz") is not None:
        warm_circuit = ansatz_to_circuit(
            warm_ctx["ansatz"],
            np.asarray(warm_ctx.get("theta", []), dtype=float),
            num_qubits=int(warm_ctx.get("num_qubits", stage_result.nq_total)),
            reference_state=np.asarray(warm_ctx.get("reference_state"), dtype=complex),
        )
        stage_bundles["warm_start"] = {
            "title": f"L={int(cfg.physics.L)} warm HH-HVA",
            "circuit": warm_circuit,
            "metadata": {
                "ansatz": str(cfg.warm_start.ansatz_name),
                "reps": int(cfg.warm_start.reps),
                "energy": float(stage_result.warm_payload.get("energy", float("nan"))),
                "exact_energy": float(stage_result.warm_payload.get("exact_filtered_energy", float("nan"))),
                "delta_abs": float(
                    abs(
                        float(stage_result.warm_payload.get("energy", float("nan")))
                        - float(stage_result.warm_payload.get("exact_filtered_energy", float("nan")))
                    )
                ),
                "transpile_metrics": _transpile_metrics_or_error(cfg, circuit=warm_circuit),
            },
            "notes": [
                "Representative view keeps PauliEvolutionGate blocks intact.",
                "Expanded view applies one circuit-definition decomposition pass.",
            ],
        }

    adapt_ctx = stage_result.adapt_circuit_context
    adapt_circuit = None
    if isinstance(adapt_ctx, Mapping) and adapt_ctx.get("reference_state") is not None:
        adapt_circuit = adapt_ops_to_circuit(
            list(adapt_ctx.get("selected_ops", [])),
            np.asarray(adapt_ctx.get("theta", []), dtype=float),
            num_qubits=int(adapt_ctx.get("num_qubits", stage_result.nq_total)),
            reference_state=np.asarray(adapt_ctx.get("reference_state"), dtype=complex),
        )
        stage_bundles["adapt_vqe"] = {
            "title": f"L={int(cfg.physics.L)} ADAPT-VQE",
            "circuit": adapt_circuit,
            "metadata": {
                "depth": int(stage_result.adapt_payload.get("ansatz_depth", 0)),
                "pool_type": str(adapt_ctx.get("pool_type", stage_result.adapt_payload.get("pool_type", ""))),
                "continuation_mode": str(
                    adapt_ctx.get("continuation_mode", stage_result.adapt_payload.get("continuation_mode", ""))
                ),
                "energy": float(stage_result.adapt_payload.get("energy", float("nan"))),
                "exact_energy": float(stage_result.adapt_payload.get("exact_gs_energy", float("nan"))),
                "stop_reason": str(stage_result.adapt_payload.get("stop_reason", "")),
                "transpile_metrics": _transpile_metrics_or_error(cfg, circuit=adapt_circuit),
            },
            "notes": [
                "Circuit uses the actual selected ADAPT generators and optimized theta values.",
                "Reference state is the warm-stage output handed to ADAPT.",
            ],
        }

    replay_ctx = stage_result.replay_circuit_context
    replay_circuit = None
    if isinstance(replay_ctx, Mapping) and replay_ctx.get("ansatz") is not None:
        replay_circuit = ansatz_to_circuit(
            replay_ctx["ansatz"],
            np.asarray(replay_ctx.get("theta", []), dtype=float),
            num_qubits=int(replay_ctx.get("num_qubits", stage_result.nq_total)),
            reference_state=np.asarray(replay_ctx.get("reference_state"), dtype=complex),
        )
        stage_bundles["conventional_replay"] = {
            "title": f"L={int(cfg.physics.L)} matched-family replay",
            "circuit": replay_circuit,
            "metadata": {
                "family_info": dict(replay_ctx.get("family_info", {})),
                "reps": int(cfg.replay.reps),
                "seed_policy": str(replay_ctx.get("resolved_seed_policy", cfg.replay.replay_seed_policy)),
                "energy": float(stage_result.replay_payload.get("vqe", {}).get("energy", float("nan"))),
                "exact_energy": float(stage_result.replay_payload.get("exact", {}).get("E_exact_sector", float("nan"))),
                "stop_reason": str(stage_result.replay_payload.get("vqe", {}).get("stop_reason", "")),
                "transpile_metrics": _transpile_metrics_or_error(cfg, circuit=replay_circuit),
            },
            "notes": [
                "Replay circuit uses the matched ADAPT-family ansatz with the final optimized replay theta values.",
                "This is the conventional non-ADAPT VQE stage in the staged HH workflow.",
            ],
        }

    dynamics_bundles: dict[str, dict[str, Any]] = {}
    if not bool(cfg.dynamics.enabled):
        return {
            "transpile_target": _transpile_target_metadata(cfg),
            "stages": stage_bundles,
            "dynamics": dynamics_bundles,
        }

    ordered_for_run = list(stage_result.ordered_labels_exyz)
    drive_provider = None
    drive_profile = None
    if bool(cfg.dynamics.enable_drive):
        drive_provider, _drive_meta, ordered_for_run, drive_profile = _build_drive_provider(
            cfg=cfg,
            nq_total=int(stage_result.nq_total),
            ordered_labels_exyz=stage_result.ordered_labels_exyz,
        )

    macro_time = float(cfg.dynamics.t_final) / float(cfg.dynamics.trotter_steps)
    prep_plus_initial = replay_circuit if replay_circuit is not None else adapt_circuit
    empty_initial = _empty_qiskit_circuit(int(stage_result.nq_total))
    report_methods = tuple(
        method
        for method in cfg.dynamics.methods
        if str(method).strip().lower() == "suzuki2" or is_cfqm_dynamics_method(str(method))
    ) or ("suzuki2", "cfqm4")
    for method in report_methods:
        method_norm = str(method).strip().lower()
        circuitization_reason = time_dynamics_circuitization_reason(
            method=str(method_norm),
            cfqm_stage_exp=str(cfg.dynamics.cfqm_stage_exp),
        )
        notes = [
            "Circuit shows one representative macro-step only; repeat_count gives the full unrolled count.",
            "Expanded view decomposes only the PauliEvolutionGate layers, not the full repeated trajectory.",
        ]
        if circuitization_reason is None:
            warn_time_dynamics_circuit_semantics(
                method=str(method_norm),
                cfqm_stage_exp=str(cfg.dynamics.cfqm_stage_exp),
                drive_time_sampling=str(cfg.dynamics.drive_time_sampling),
            )
            macro_circuit = build_time_dynamics_circuit(
                method=str(method_norm),
                initial_circuit=(prep_plus_initial if prep_plus_initial is not None else empty_initial),
                ordered_labels_exyz=list(ordered_for_run),
                static_coeff_map_exyz=dict(stage_result.coeff_map_exyz),
                drive_provider_exyz=drive_provider,
                time_value=float(macro_time),
                trotter_steps=1,
                drive_t0=float(cfg.dynamics.drive_t0),
                drive_time_sampling=str(cfg.dynamics.drive_time_sampling),
                cfqm_stage_exp=str(cfg.dynamics.cfqm_stage_exp),
                cfqm_coeff_drop_abs_tol=float(cfg.dynamics.cfqm_coeff_drop_abs_tol),
            )
            dynamics_only_circuit = build_time_dynamics_circuit(
                method=str(method_norm),
                initial_circuit=empty_initial,
                ordered_labels_exyz=list(ordered_for_run),
                static_coeff_map_exyz=dict(stage_result.coeff_map_exyz),
                drive_provider_exyz=drive_provider,
                time_value=float(cfg.dynamics.t_final),
                trotter_steps=int(cfg.dynamics.trotter_steps),
                drive_t0=float(cfg.dynamics.drive_t0),
                drive_time_sampling=str(cfg.dynamics.drive_time_sampling),
                cfqm_stage_exp=str(cfg.dynamics.cfqm_stage_exp),
                cfqm_coeff_drop_abs_tol=float(cfg.dynamics.cfqm_coeff_drop_abs_tol),
            )
            prep_plus_circuit = (
                None
                if prep_plus_initial is None
                else build_time_dynamics_circuit(
                    method=str(method_norm),
                    initial_circuit=prep_plus_initial,
                    ordered_labels_exyz=list(ordered_for_run),
                    static_coeff_map_exyz=dict(stage_result.coeff_map_exyz),
                    drive_provider_exyz=drive_provider,
                    time_value=float(cfg.dynamics.t_final),
                    trotter_steps=int(cfg.dynamics.trotter_steps),
                    drive_t0=float(cfg.dynamics.drive_t0),
                    drive_time_sampling=str(cfg.dynamics.drive_time_sampling),
                    cfqm_stage_exp=str(cfg.dynamics.cfqm_stage_exp),
                    cfqm_coeff_drop_abs_tol=float(cfg.dynamics.cfqm_coeff_drop_abs_tol),
                )
            )
            proxy_total = compute_time_dynamics_proxy_cost(
                method=str(method_norm),
                t_final=float(cfg.dynamics.t_final),
                trotter_steps=int(cfg.dynamics.trotter_steps),
                drive_t0=float(cfg.dynamics.drive_t0),
                drive_time_sampling=str(cfg.dynamics.drive_time_sampling),
                ordered_labels_exyz=list(ordered_for_run),
                static_coeff_map_exyz=dict(stage_result.coeff_map_exyz),
                drive_provider_exyz=drive_provider,
                coeff_drop_abs_tol=float(cfg.dynamics.cfqm_coeff_drop_abs_tol),
                cfqm_stage_exp=str(cfg.dynamics.cfqm_stage_exp),
            )
            proxy_macro = compute_time_dynamics_proxy_cost(
                method=str(method_norm),
                t_final=float(macro_time),
                trotter_steps=1,
                drive_t0=float(cfg.dynamics.drive_t0),
                drive_time_sampling=str(cfg.dynamics.drive_time_sampling),
                ordered_labels_exyz=list(ordered_for_run),
                static_coeff_map_exyz=dict(stage_result.coeff_map_exyz),
                drive_provider_exyz=drive_provider,
                coeff_drop_abs_tol=float(cfg.dynamics.cfqm_coeff_drop_abs_tol),
                cfqm_stage_exp=str(cfg.dynamics.cfqm_stage_exp),
            )
            dynamics_metrics = {
                "macro_step": _transpile_metrics_or_error(cfg, circuit=macro_circuit),
                "dynamics_only": _transpile_metrics_or_error(cfg, circuit=dynamics_only_circuit),
                "prep_plus_dynamics": _transpile_metrics_or_error(cfg, circuit=prep_plus_circuit),
            }
        else:
            macro_circuit = None
            dynamics_only_circuit = None
            prep_plus_circuit = None
            proxy_total = {"skipped": True, "reason": str(circuitization_reason)}
            proxy_macro = {"skipped": True, "reason": str(circuitization_reason)}
            skip_metrics = {
                "target": _transpile_target_metadata(cfg),
                "skipped": True,
                "reason": str(circuitization_reason),
            }
            dynamics_metrics = {
                "macro_step": dict(skip_metrics),
                "dynamics_only": dict(skip_metrics),
                "prep_plus_dynamics": dict(skip_metrics),
            }
            notes.append(f"Circuit artifacts skipped: {str(circuitization_reason)}")
        dynamics_bundles[str(method)] = {
            "title": f"L={int(cfg.physics.L)} {str(method).upper()} dynamics macro-step",
            "circuit": macro_circuit,
            "metadata": {
                "repeat_count": int(cfg.dynamics.trotter_steps),
                "macro_step_time": float(macro_time),
                "t_final": float(cfg.dynamics.t_final),
                "drive_enabled": bool(cfg.dynamics.enable_drive),
                "drive_profile": drive_profile,
                "cfqm_stage_exp": str(cfg.dynamics.cfqm_stage_exp),
                "circuitization": {
                    "supported": bool(circuitization_reason is None),
                    "reason": (None if circuitization_reason is None else str(circuitization_reason)),
                    "cfqm_stage_exp": str(cfg.dynamics.cfqm_stage_exp),
                },
                "transpile_target": _transpile_target_metadata(cfg),
                "proxy_macro": dict(proxy_macro),
                "proxy_total": dict(proxy_total),
                "trajectory_circuit_metrics": dynamics_metrics,
            },
            "notes": notes,
        }

    return {
        "transpile_target": _transpile_target_metadata(cfg),
        "stages": stage_bundles,
        "dynamics": dynamics_bundles,
    }


def write_hh_staged_circuit_report_section(
    pdf: Any,
    *,
    cfg: StagedHHConfig,
    stage_result: StageExecutionResult,
    run_command: str | None = None,
) -> None:
    report_payload = build_stage_circuit_report_artifacts(stage_result, cfg)
    stage_summary = _stage_summary(stage_result, cfg)
    render_parameter_manifest(
        pdf,
        model="Hubbard-Holstein (HH)",
        ansatz=_staged_ansatz_manifest(cfg),
        drive_enabled=bool(cfg.dynamics.enable_drive),
        t=float(cfg.physics.t),
        U=float(cfg.physics.u),
        dv=float(cfg.physics.dv),
        extra={
            "L": int(cfg.physics.L),
            "omega0": float(cfg.physics.omega0),
            "g_ep": float(cfg.physics.g_ep),
            "n_ph_max": int(cfg.physics.n_ph_max),
            "boundary": str(cfg.physics.boundary),
            "ordering": str(cfg.physics.ordering),
            "run_replay": bool(cfg.replay.enabled),
            "run_dynamics": bool(cfg.dynamics.enabled),
            "warm_reps": int(cfg.warm_start.reps),
            "seed_refine_family": (None if cfg.seed_refine.family is None else str(cfg.seed_refine.family)),
            "seed_refine_reps": int(cfg.seed_refine.reps),
            "adapt_max_depth": int(cfg.adapt.max_depth),
            "replay_reps": int(cfg.replay.reps),
            "t_final": float(cfg.dynamics.t_final),
            "trotter_steps": int(cfg.dynamics.trotter_steps),
            "num_times": int(cfg.dynamics.num_times),
        },
        command=str(run_command) if run_command is not None else None,
    )
    summary_lines = [
        f"HH staged circuit report, L={int(cfg.physics.L)}",
        "",
    ]
    fixed_import = stage_summary.get("fixed_final_state_import", {})
    if isinstance(fixed_import, Mapping):
        summary_lines.extend(
            [
                f"Fixed seed import: {fixed_import.get('source_json', '')}",
                f"Imported E={float(fixed_import.get('energy', float('nan'))):.12g} "
                f"exact={float(fixed_import.get('exact_energy', float('nan'))):.12g} "
                f"|dE|={float(fixed_import.get('delta_abs', float('nan'))):.6e}",
                f"Metadata strict={bool(fixed_import.get('strict_match', False))} "
                f"mismatches={len(list(fixed_import.get('mismatches', [])))}",
            ]
        )
    warm_summary = stage_summary["warm_start"]
    seed_refine_summary = stage_summary.get("seed_refine", {})
    adapt_summary = stage_summary["adapt_vqe"]
    replay_summary = stage_summary["conventional_replay"]
    if bool(warm_summary.get("skipped", False)):
        summary_lines.append(f"Warm: skipped ({warm_summary.get('skip_reason', '')})")
    else:
        summary_lines.append(
            f"Warm: E={warm_summary['energy']:.12g} "
            f"exact={warm_summary['exact_energy']:.12g} "
            f"|dE|={warm_summary['delta_abs']:.6e}"
        )
    if isinstance(seed_refine_summary, Mapping) and seed_refine_summary:
        summary_lines.append(
            f"Seed refine: family={seed_refine_summary.get('family', '')} "
            f"|dE|={float(seed_refine_summary.get('delta_abs', float('nan'))):.6e} "
            f"stop={seed_refine_summary.get('stop_reason', '')}"
        )
    if bool(adapt_summary.get("skipped", False)):
        summary_lines.append(f"ADAPT: skipped ({adapt_summary.get('skip_reason', '')})")
    else:
        summary_lines.append(
            f"ADAPT: depth={adapt_summary['depth']} "
            f"pool={adapt_summary['pool_type']} "
            f"stop={adapt_summary['stop_reason']} "
            f"|dE|={adapt_summary['delta_abs']:.6e}"
        )
    if bool(replay_summary.get("skipped", False)):
        summary_lines.append(f"Replay: skipped ({replay_summary.get('skip_reason', '')})")
    else:
        summary_lines.append(
            f"Replay: E={replay_summary['energy']:.12g} "
            f"exact={replay_summary['exact_energy']:.12g} "
            f"|dE|={replay_summary['delta_abs']:.6e}"
        )
    summary_lines.extend(
        [
        "",
        "Section semantics",
        "- Representative view keeps high-level PauliEvolutionGate blocks.",
        "- Expanded view performs one decomposition pass to expose term-level layers.",
        (
            "- Dynamics pages show one macro-step only; proxy totals summarize the full repeat count."
            if bool(cfg.dynamics.enabled)
            else "- Dynamics pages omitted: run_dynamics=false."
        ),
        ]
    )
    render_text_page(pdf, summary_lines, fontsize=10, line_spacing=0.03, max_line_width=110)

    for key in ("warm_start", "adapt_vqe", "conventional_replay"):
        bundle = report_payload["stages"].get(key)
        if not isinstance(bundle, Mapping):
            render_text_page(
                pdf,
                [f"L={int(cfg.physics.L)} {key}", "", "Circuit context unavailable for this stage."],
                fontsize=10,
                line_spacing=0.03,
            )
            continue
        circuit = bundle["circuit"]
        title = str(bundle["title"])
        notes = [str(x) for x in bundle.get("notes", [])]
        render_circuit_summary_page(
            pdf,
            title=title,
            circuit=circuit,
            metadata=dict(bundle.get("metadata", {})),
            notes=notes,
        )
        render_circuit_page(pdf, circuit=circuit, title=title, subtitle="Representative view", notes=notes)
        render_circuit_page(
            pdf,
            circuit=circuit,
            title=title,
            subtitle="Expanded view",
            notes=notes,
            expand_evolution=True,
        )

    for method, bundle in report_payload["dynamics"].items():
        circuit = bundle["circuit"]
        title = str(bundle["title"])
        notes = [str(x) for x in bundle.get("notes", [])]
        render_circuit_summary_page(
            pdf,
            title=title,
            circuit=circuit,
            metadata=dict(bundle.get("metadata", {})),
            notes=notes,
        )
        if circuit is None:
            continue
        render_circuit_page(pdf, circuit=circuit, title=title, subtitle="Representative macro-step", notes=notes)
        render_circuit_page(
            pdf,
            circuit=circuit,
            title=title,
            subtitle="Expanded macro-step",
            notes=notes,
            expand_evolution=True,
        )


def _stage_delta(payload: Mapping[str, Any], *, energy_key: str, exact_key: str) -> float:
    return float(abs(float(payload.get(energy_key, float("nan"))) - float(payload.get(exact_key, float("nan")))) )


def _stage_summary(stage_result: StageExecutionResult, cfg: StagedHHConfig) -> dict[str, Any]:
    warm_energy = float(stage_result.warm_payload.get("energy", float("nan")))
    warm_exact = float(stage_result.warm_payload.get("exact_filtered_energy", float("nan")))
    adapt_energy = float(stage_result.adapt_payload.get("energy", float("nan")))
    adapt_exact = float(stage_result.adapt_payload.get("exact_gs_energy", float("nan")))
    replay_vqe = stage_result.replay_payload.get("vqe", {})
    replay_exact = stage_result.replay_payload.get("exact", {})
    final_energy = float(replay_vqe.get("energy", float("nan")))
    final_exact = float(replay_exact.get("E_exact_sector", float("nan")))
    warm_delta = float(abs(warm_energy - warm_exact))
    adapt_delta = float(abs(adapt_energy - adapt_exact))
    final_delta = float(abs(final_energy - final_exact))
    replay_skipped = bool(stage_result.replay_payload.get("skipped", False))
    summary = {
        "hf_reference": {
            "state_kind": "reference_state",
            "nq_total": int(stage_result.nq_total),
            "sector_n_up": int(cfg.physics.sector_n_up),
            "sector_n_dn": int(cfg.physics.sector_n_dn),
        },
        "warm_start": {
            "ansatz": str(stage_result.warm_payload.get("ansatz", cfg.warm_start.ansatz_name)),
            "energy": float(warm_energy),
            "exact_energy": float(warm_exact),
            "delta_abs": float(warm_delta),
            "ecut_1": {"threshold": float(cfg.gates.ecut_1), "pass": bool(warm_delta <= float(cfg.gates.ecut_1))},
            "optimizer_method": str(stage_result.warm_payload.get("optimizer_method", cfg.warm_start.method)),
            "reps": int(cfg.warm_start.reps),
            "restarts": int(cfg.warm_start.restarts),
            "maxiter": int(cfg.warm_start.maxiter),
            "message": str(stage_result.warm_payload.get("message", "")),
            "checkpoint_json_latest": str(
                stage_result.warm_payload.get("checkpoint_json_latest", cfg.artifacts.warm_checkpoint_json)
            ),
            "checkpoint_json_used": str(
                stage_result.warm_payload.get("checkpoint_json_used", cfg.artifacts.warm_cutover_json)
            ),
            "cutoff_triggered": bool(stage_result.warm_payload.get("cutoff_triggered", False)),
            "cutoff_reason": stage_result.warm_payload.get("cutoff_reason"),
            "resumed_from_checkpoint": bool(stage_result.warm_payload.get("resumed_from_checkpoint", False)),
            "skipped": bool(stage_result.warm_payload.get("skipped", False)),
            "skip_reason": stage_result.warm_payload.get("skip_reason"),
        },
        "adapt_vqe": {
            "energy": float(adapt_energy),
            "exact_energy": float(adapt_exact),
            "delta_abs": float(adapt_delta),
            "depth": int(stage_result.adapt_payload.get("ansatz_depth", 0)),
            "pool_type": str(stage_result.adapt_payload.get("pool_type", cfg.adapt.pool or cfg.adapt.continuation_mode)),
            "continuation_mode": str(stage_result.adapt_payload.get("continuation_mode", cfg.adapt.continuation_mode)),
            "stop_reason": str(stage_result.adapt_payload.get("stop_reason", "")),
            "handoff_json": str(cfg.artifacts.handoff_json),
            "adapt_ref_json": str(
                stage_result.adapt_payload.get("adapt_ref_json", stage_result.warm_payload.get("checkpoint_json_used", ""))
            ),
            "initial_state_source": str(stage_result.adapt_payload.get("initial_state_source", "adapt_ref_json")),
            "skipped": bool(stage_result.adapt_payload.get("skipped", False)),
            "skip_reason": stage_result.adapt_payload.get("skip_reason"),
        },
        "conventional_replay": {
            "energy": float(final_energy),
            "exact_energy": float(final_exact),
            "delta_abs": float(final_delta),
            "ecut_2": {
                "threshold": float(cfg.gates.ecut_2),
                "pass": (None if replay_skipped else bool(final_delta <= float(cfg.gates.ecut_2))),
                "evaluated": bool(not replay_skipped),
            },
            "generator_family": dict(stage_result.replay_payload.get("generator_family", {})),
            "seed_baseline": dict(stage_result.replay_payload.get("seed_baseline", {})),
            "stop_reason": str(replay_vqe.get("stop_reason", replay_vqe.get("message", ""))),
            "replay_continuation_mode": str(stage_result.replay_payload.get("replay_contract", {}).get("continuation_mode", cfg.replay.continuation_mode)),
            "replay_output_json": str(cfg.artifacts.replay_output_json),
            "skipped": replay_skipped,
            "skip_reason": stage_result.replay_payload.get("skip_reason"),
        },
    }
    if isinstance(stage_result.seed_refine_payload, Mapping):
        refine_vqe = stage_result.seed_refine_payload.get("vqe", {})
        refine_exact = stage_result.seed_refine_payload.get("exact", {})
        refine_pool = stage_result.seed_refine_payload.get("pool", {})
        summary["seed_refine"] = {
            "family": str(stage_result.seed_refine_payload.get("generator_family", {}).get("resolved", cfg.seed_refine.family or "")),
            "family_kind": str(refine_pool.get("family_kind", "explicit_family")) if isinstance(refine_pool, Mapping) else "explicit_family",
            "energy": float(refine_vqe.get("energy", float("nan"))),
            "exact_energy": float(refine_exact.get("E_exact_sector", float("nan"))),
            "delta_abs": float(abs(float(refine_vqe.get("energy", float("nan"))) - float(refine_exact.get("E_exact_sector", float("nan"))))),
            "optimizer_method": str(refine_vqe.get("method", cfg.seed_refine.optimizer)),
            "reps": int(cfg.seed_refine.reps),
            "maxiter": int(cfg.seed_refine.maxiter),
            "stop_reason": str(refine_vqe.get("stop_reason", refine_vqe.get("message", ""))),
            "state_json": str(stage_result.seed_refine_payload.get("state_json", _seed_refine_state_json_path(cfg))),
            "seed_baseline": dict(stage_result.seed_refine_payload.get("seed_baseline", {})),
            "pool": dict(refine_pool) if isinstance(refine_pool, Mapping) else {},
        }
    if isinstance(stage_result.fixed_final_state_import, Mapping):
        summary["fixed_final_state_import"] = dict(stage_result.fixed_final_state_import)
    return summary


def _compute_comparisons(payload: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "noiseless_vs_ground_state": {},
        "noiseless_vs_reference": {},
        "stage_gates": {},
    }
    stage_pipeline = payload.get("stage_pipeline", {})
    if isinstance(stage_pipeline, Mapping):
        warm = stage_pipeline.get("warm_start", {})
        final = stage_pipeline.get("conventional_replay", {})
        if isinstance(warm, Mapping):
            out["stage_gates"]["ecut_1"] = dict(warm.get("ecut_1", {}))
        if isinstance(final, Mapping):
            out["stage_gates"]["ecut_2"] = dict(final.get("ecut_2", {}))

    dynamics = payload.get("dynamics_noiseless", {})
    if isinstance(dynamics, Mapping):
        for profile_name, profile_payload in dynamics.get("profiles", {}).items():
            if not isinstance(profile_payload, Mapping):
                continue
            ground_state_cmp: dict[str, Any] = {}
            reference_cmp: dict[str, Any] = {}
            ground_state_ref = profile_payload.get("ground_state_reference", {})
            ground_state_energy = (
                float(ground_state_ref.get("energy", float("nan")))
                if isinstance(ground_state_ref, Mapping)
                else float("nan")
            )
            for method_name, method_payload in profile_payload.get("methods", {}).items():
                if not isinstance(method_payload, Mapping):
                    continue
                final = method_payload.get("final", {})
                ground_state_cmp[str(method_name)] = {
                    "ground_state_reference_energy": float(ground_state_energy),
                    "final_abs_energy_error": float(final.get("abs_energy_error_vs_ground_state", float("nan"))),
                }
                reference_cmp[str(method_name)] = {
                    "final_abs_energy_total_error": float(
                        final.get(
                            "abs_energy_total_error_vs_reference",
                            final.get("abs_energy_total_error", float("nan")),
                        )
                    ),
                    "final_fidelity": float(final.get("fidelity", float("nan"))),
                }
            out["noiseless_vs_ground_state"][str(profile_name)] = ground_state_cmp
            out["noiseless_vs_reference"][str(profile_name)] = reference_cmp
    return out


def _payload_artifacts(cfg: StagedHHConfig) -> dict[str, Any]:
    artifacts = {
        "workflow": {
            "output_json": str(cfg.artifacts.output_json),
            "output_pdf": str(cfg.artifacts.output_pdf),
        },
        "intermediate": {
            "warm_checkpoint_json": str(cfg.artifacts.warm_checkpoint_json),
            "warm_cutover_json": str(cfg.artifacts.warm_cutover_json),
            "adapt_handoff_json": str(cfg.artifacts.handoff_json),
            "replay_output_json": str(cfg.artifacts.replay_output_json),
            "replay_output_csv": str(cfg.artifacts.replay_output_csv),
            "replay_output_md": str(cfg.artifacts.replay_output_md),
            "replay_output_log": str(cfg.artifacts.replay_output_log),
            "workflow_log": str(cfg.artifacts.workflow_log),
        },
    }
    if cfg.fixed_final_state is not None:
        artifacts["intermediate"]["fixed_final_state_json"] = str(cfg.fixed_final_state.json_path)
    return artifacts


def assemble_payload(
    *,
    cfg: StagedHHConfig,
    stage_result: StageExecutionResult,
    dynamics_noiseless: Mapping[str, Any],
    circuit_report: Mapping[str, Any] | None = None,
    run_command: str,
) -> dict[str, Any]:
    fixed_mode = bool(stage_result.fixed_final_state_import)
    dynamics_enabled = bool(cfg.dynamics.enabled)
    replay_enabled = bool(cfg.replay.enabled)
    payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_staged_noiseless",
        "workflow_contract": {
            "stage_chain": _workflow_stage_chain(cfg, fixed_mode=fixed_mode),
            "conventional_vqe_definition": (
                "non-ADAPT matched-family replay from ADAPT handoff"
                if replay_enabled
                else "disabled (run_replay=false)"
            ),
            "drive_default": "opt_in",
            "noiseless_energy_metric": (
                "|E_method(t) - E_exact_sector_terminal| with terminal prepared-state exact energy as baseline"
                if dynamics_enabled
                else "not_run (run_dynamics=false)"
            ),
            "noiseless_fidelity_metric": (
                "fidelity(method(t), exact-propagated psi_final)"
                if dynamics_enabled
                else "not_run (run_dynamics=false)"
            ),
        },
        "settings": _jsonable(asdict(cfg)),
        "default_provenance": dict(cfg.default_provenance),
        "artifacts": _payload_artifacts(cfg),
        "command": str(run_command),
        "stage_pipeline": _stage_summary(stage_result, cfg),
        "dynamics_noiseless": dict(dynamics_noiseless),
        "circuit_metrics": (
            None if circuit_report is None else _strip_circuit_objects(dict(circuit_report))
        ),
    }
    payload["comparisons"] = _compute_comparisons(payload)
    return payload


def _profile_plot_page(pdf: Any, profile_name: str, profile_payload: Mapping[str, Any]) -> None:
    require_matplotlib()
    plt = get_plt()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True)
    times = [float(x) for x in profile_payload.get("times", [])]
    methods = profile_payload.get("methods", {})
    for method_name, method_payload in methods.items():
        if not isinstance(method_payload, Mapping):
            continue
        rows = method_payload.get("trajectory", [])
        if not rows:
            continue
        energy_err = [float(r.get("abs_energy_error_vs_ground_state", float("nan"))) for r in rows]
        fidelity = [float(r["fidelity"]) for r in rows]
        axes[0].plot(times, energy_err, label=str(method_name))
        axes[1].plot(times, fidelity, label=str(method_name))
    axes[0].set_title(f"{profile_name}: |E_method - E_GS|")
    axes[0].set_ylabel("abs energy error vs GS")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title(f"{profile_name}: fidelity to seeded exact reference")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("fidelity")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    pdf.savefig(fig)
    plt.close(fig)


def write_staged_hh_pdf(payload: Mapping[str, Any], cfg: StagedHHConfig, run_command: str) -> None:
    if bool(cfg.artifacts.skip_pdf):
        return
    require_matplotlib()
    pdf_path = Path(cfg.artifacts.output_pdf)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    PdfPages = get_PdfPages()
    plt = get_plt()

    stage_pipeline = payload.get("stage_pipeline", {})
    warm = stage_pipeline.get("warm_start", {}) if isinstance(stage_pipeline, Mapping) else {}
    adapt = stage_pipeline.get("adapt_vqe", {}) if isinstance(stage_pipeline, Mapping) else {}
    replay = stage_pipeline.get("conventional_replay", {}) if isinstance(stage_pipeline, Mapping) else {}

    with PdfPages(pdf_path) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein",
            ansatz=_staged_ansatz_manifest(cfg),
            drive_enabled=bool(cfg.dynamics.enable_drive),
            t=float(cfg.physics.t),
            U=float(cfg.physics.u),
            dv=float(cfg.physics.dv),
            extra={
                "L": int(cfg.physics.L),
                "omega0": float(cfg.physics.omega0),
                "g_ep": float(cfg.physics.g_ep),
                "n_ph_max": int(cfg.physics.n_ph_max),
                "boundary": str(cfg.physics.boundary),
                "ordering": str(cfg.physics.ordering),
                "run_replay": bool(cfg.replay.enabled),
                "run_dynamics": bool(cfg.dynamics.enabled),
                "warm_reps": int(cfg.warm_start.reps),
                "seed_refine_family": (None if cfg.seed_refine.family is None else str(cfg.seed_refine.family)),
                "seed_refine_reps": int(cfg.seed_refine.reps),
                "adapt_mode": str(cfg.adapt.continuation_mode),
                "replay_mode": str(cfg.replay.continuation_mode),
                "methods": ",".join(cfg.dynamics.methods),
                "t_final": float(cfg.dynamics.t_final),
                "trotter_steps": int(cfg.dynamics.trotter_steps),
                "num_times": int(cfg.dynamics.num_times),
            },
            command=str(run_command),
        )
        summary_lines = [
            "HH staged noiseless workflow summary",
            "",
        ]
        fixed_import = stage_pipeline.get("fixed_final_state_import", {}) if isinstance(stage_pipeline, Mapping) else {}
        if isinstance(fixed_import, Mapping) and fixed_import:
            summary_lines.extend(
                [
                    f"Fixed seed import: {fixed_import.get('source_json', '')}",
                    f"Imported energy={fixed_import.get('energy')} exact={fixed_import.get('exact_energy')} delta={fixed_import.get('delta_abs')}",
                ]
            )
        seed_refine = stage_pipeline.get("seed_refine", {}) if isinstance(stage_pipeline, Mapping) else {}
        summary_lines.extend(
            [
                (
                    f"Warm-start: E={warm.get('energy')} exact={warm.get('exact_energy')} "
                    f"delta={warm.get('delta_abs')} ecut_1={warm.get('ecut_1')}"
                    if not bool(warm.get("skipped", False))
                    else f"Warm-start: skipped ({warm.get('skip_reason', '')})"
                ),
                (
                    f"Seed refine: family={seed_refine.get('family')} delta={seed_refine.get('delta_abs')} "
                    f"stop={seed_refine.get('stop_reason')}"
                    if isinstance(seed_refine, Mapping) and seed_refine
                    else "Seed refine: disabled"
                ),
                (
                    f"ADAPT: depth={adapt.get('depth')} pool={adapt.get('pool_type')} "
                    f"delta={adapt.get('delta_abs')} stop={adapt.get('stop_reason')}"
                    if not bool(adapt.get("skipped", False))
                    else f"ADAPT: skipped ({adapt.get('skip_reason', '')})"
                ),
                (
                    f"Replay: E={replay.get('energy')} exact={replay.get('exact_energy')} "
                    f"delta={replay.get('delta_abs')} ecut_2={replay.get('ecut_2')}"
                    if not bool(replay.get("skipped", False))
                    else f"Replay: skipped ({replay.get('skip_reason', '')})"
                ),
            (
                "Dynamics metrics: energy uses the terminal-stage exact-sector GS baseline; "
                "fidelity uses exact propagation from psi_final."
                if bool(cfg.dynamics.enabled)
                else "Dynamics: skipped (run_dynamics=false)"
            ),
            "",
            "Artifacts",
            f"- workflow_json: {cfg.artifacts.output_json}",
            f"- workflow_pdf: {cfg.artifacts.output_pdf}",
            f"- adapt_handoff_json: {cfg.artifacts.handoff_json}",
            ]
        )
        if bool(cfg.replay.enabled):
            summary_lines.extend(
                [
                    f"- replay_json: {cfg.artifacts.replay_output_json}",
                    f"- replay_csv: {cfg.artifacts.replay_output_csv}",
                    f"- replay_md: {cfg.artifacts.replay_output_md}",
                    f"- replay_log: {cfg.artifacts.replay_output_log}",
                ]
            )
        render_text_page(pdf, summary_lines, fontsize=10, line_spacing=0.03)

        fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5))
        render_compact_table(
            axes[0],
            title="Stage metrics",
            col_labels=["Stage", "Energy", "Exact", "|ΔE|", "Gate/stop"],
            rows=[
                ["Warm", f"{warm.get('energy', float('nan')):.8f}", f"{warm.get('exact_energy', float('nan')):.8f}", f"{warm.get('delta_abs', float('nan')):.3e}", str(warm.get('ecut_1', {}))],
                ["ADAPT", f"{adapt.get('energy', float('nan')):.8f}", f"{adapt.get('exact_energy', float('nan')):.8f}", f"{adapt.get('delta_abs', float('nan')):.3e}", str(adapt.get('stop_reason', ''))],
                ["Replay", f"{replay.get('energy', float('nan')):.8f}", f"{replay.get('exact_energy', float('nan')):.8f}", f"{replay.get('delta_abs', float('nan')):.3e}", str(replay.get('ecut_2', {}))],
            ],
            fontsize=8,
        )
        cmp_rows: list[list[str]] = []
        gs_cmp = payload.get("comparisons", {}).get("noiseless_vs_ground_state", {})
        ref_cmp = payload.get("comparisons", {}).get("noiseless_vs_reference", {})
        for profile_name, profile_cmp in gs_cmp.items():
            reference_methods = ref_cmp.get(profile_name, {}) if isinstance(ref_cmp, Mapping) else {}
            for method_name, rec in profile_cmp.items():
                ref_rec = reference_methods.get(method_name, {}) if isinstance(reference_methods, Mapping) else {}
                cmp_rows.append([
                    str(profile_name),
                    str(method_name),
                    f"{float(rec.get('final_abs_energy_error', float('nan'))):.3e}",
                    f"{float(ref_rec.get('final_fidelity', float('nan'))):.6f}",
                ])
        if not cmp_rows:
            cmp_rows = [["(none)", "(none)", "nan", "nan"]]
        render_compact_table(
            axes[1],
            title="Noiseless dynamics: GS error + seeded-reference fidelity",
            col_labels=["Profile", "Method", "Final |E-E_GS|", "Final fidelity"],
            rows=cmp_rows,
            fontsize=8,
        )
        pdf.savefig(fig)
        plt.close(fig)

        for profile_name, profile_payload in payload.get("dynamics_noiseless", {}).get("profiles", {}).items():
            if isinstance(profile_payload, Mapping):
                _profile_plot_page(pdf, str(profile_name), profile_payload)

        render_command_page(
            pdf,
            str(run_command),
            script_name="pipelines/hardcoded/hh_staged_noiseless.py",
            extra_header_lines=(
                [
                    f"workflow_json: {cfg.artifacts.output_json}",
                    f"replay_json: {cfg.artifacts.replay_output_json}",
                ]
                if bool(cfg.replay.enabled)
                else [f"workflow_json: {cfg.artifacts.output_json}"]
            ),
        )


def run_staged_hh_noiseless(cfg: StagedHHConfig, *, run_command: str | None = None) -> dict[str, Any]:
    run_command_str = current_command_string() if run_command is None else str(run_command)
    cfg.artifacts.output_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.handoff_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.warm_checkpoint_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.warm_cutover_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.replay_output_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.replay_output_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.replay_output_md.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.replay_output_log.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.workflow_log.parent.mkdir(parents=True, exist_ok=True)
    _append_workflow_log(
        cfg,
        "workflow_run_start",
        command=str(run_command_str),
        output_json=str(cfg.artifacts.output_json),
        warm_checkpoint_json=str(cfg.artifacts.warm_checkpoint_json),
        warm_cutover_json=str(cfg.artifacts.warm_cutover_json),
    )

    stage_result = run_stage_pipeline(cfg)
    if bool(cfg.dynamics.enabled):
        dynamics_noiseless = run_noiseless_profiles(stage_result, cfg)
    else:
        dynamics_noiseless = {"profiles": {}, "skipped": True, "skip_reason": "run_dynamics_false"}
        _append_workflow_log(
            cfg,
            "dynamics_noiseless_skipped",
            reason="run_dynamics_false",
            adapt_handoff_json=str(cfg.artifacts.handoff_json),
        )
    circuit_report = build_stage_circuit_report_artifacts(stage_result, cfg)
    payload = assemble_payload(
        cfg=cfg,
        stage_result=stage_result,
        dynamics_noiseless=dynamics_noiseless,
        circuit_report=circuit_report,
        run_command=run_command_str,
    )
    _write_json(cfg.artifacts.output_json, payload)
    _append_workflow_log(
        cfg,
        "workflow_json_written",
        output_json=str(cfg.artifacts.output_json),
        replay_json=str(cfg.artifacts.replay_output_json),
        ecut_2_pass=payload.get("stage_pipeline", {}).get("conventional_replay", {}).get("ecut_2", {}).get("pass"),
    )
    if not bool(cfg.artifacts.skip_pdf):
        write_staged_hh_pdf(payload, cfg, run_command_str)
    _append_workflow_log(
        cfg,
        "workflow_run_complete",
        output_json=str(cfg.artifacts.output_json),
        replay_json=str(cfg.artifacts.replay_output_json),
        adapt_handoff_json=str(cfg.artifacts.handoff_json),
    )
    return payload
