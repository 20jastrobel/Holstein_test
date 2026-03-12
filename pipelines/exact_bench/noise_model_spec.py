#!/usr/bin/env python3
"""Typed normalized noise-model spec helpers for exact-bench wrappers.

This module stays in wrapper/benchmark space. It does not modify production
statevector paths; it only normalizes requested noisy-execution intent and
stores serializable metadata snapshots/hashes for report payloads.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence


_ALLOWED_EXECUTORS = {"statevector", "aer", "runtime_qpu"}
_ALLOWED_NOISE_KINDS = {
    "none",
    "shots_only",
    "backend_basic",
    "backend_scheduled",
    "patch_snapshot",
    "qpu_raw",
    "qpu_suppressed",
    "qpu_layer_learned",
}
_ALLOWED_BACKEND_PROFILES = {
    "not_applicable",
    "generic_seeded",
    "fake_snapshot",
    "live_backend",
    "frozen_snapshot_json",
}
_ALLOWED_MITIGATION_BUNDLES = {
    "none",
    "readout_only",
    "light_counts",
    "runtime_suppressed",
    "runtime_layer_learned",
    "report_heavy",
}
_ALLOWED_LAYOUT_POLICIES = {"auto_then_lock", "fixed_patch", "frozen_layout"}
_ALLOWED_SCHEDULE_POLICIES = {"none", "asap"}
_ALLOWED_RUNTIME_TWIRLING_STRATEGIES = {
    "active",
    "active-circuit",
    "active-accum",
    "all",
}


def _get_value(raw: Any, key: str, default: Any = None) -> Any:
    if isinstance(raw, Mapping):
        return raw.get(key, default)
    return getattr(raw, key, default)


def _coerce_int_list(raw: Any) -> list[int] | None:
    if raw is None or raw == "" or raw == () or raw == []:
        return None
    if isinstance(raw, str):
        tokens = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
        return [int(tok) for tok in tokens] if tokens else None
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        vals = [int(v) for v in raw]
        return vals if vals else None
    return [int(raw)]


def _coerce_edge_list(raw: Any) -> list[list[int]] | None:
    if raw is None or raw == "" or raw == () or raw == []:
        return None
    if isinstance(raw, str):
        edges: list[list[int]] = []
        for token in [tok.strip() for tok in raw.split(",") if tok.strip()]:
            token = token.replace("-", ":")
            left, right = token.split(":", 1)
            edges.append([int(left), int(right)])
        return edges or None
    out: list[list[int]] = []
    for edge in raw:
        if isinstance(edge, Sequence) and not isinstance(edge, (str, bytes, bytearray)) and len(edge) == 2:
            out.append([int(edge[0]), int(edge[1])])
        else:
            raise ValueError(f"Invalid fixed coupler edge {edge!r}; expected pair-like entries.")
    return out or None


def _canonical_int_list(raw: Any) -> list[int] | None:
    if raw is None or raw == "" or raw == () or raw == []:
        return None
    return [int(v) for v in list(raw)]


def _canonical_edge_list(raw: Any) -> list[list[int]] | None:
    if raw is None or raw == "" or raw == () or raw == []:
        return None
    cleaned = [[int(edge[0]), int(edge[1])] for edge in list(raw)]
    cleaned.sort(key=lambda edge: (int(edge[0]), int(edge[1])))
    return cleaned or None


def _canonical_str_list(raw: Any) -> list[str]:
    if raw is None or raw == "" or raw == () or raw == []:
        return []
    return sorted(str(v) for v in list(raw))


def _mitigation_has_layer_noise_model(raw_mitigation: Any) -> bool:
    if isinstance(raw_mitigation, Mapping):
        return (
            raw_mitigation.get("layer_noise_model", None) is not None
            or raw_mitigation.get(
                "layer_noise_model_json",
                raw_mitigation.get("layerNoiseModelJson", None),
            )
            not in {None, "", "none"}
        )
    return False


def normalize_runtime_twirling_config(raw: Any) -> dict[str, Any]:
    nested = _get_value(raw, "runtime_twirling", None)
    nested_empty = nested is None or nested == "" or nested == () or nested == []
    source = raw if nested_empty else nested

    enable_gates = bool(
        _get_value(
            source,
            "enable_gates",
            _get_value(raw, "runtime_enable_gate_twirling", False),
        )
    )
    enable_measure = bool(
        _get_value(
            source,
            "enable_measure",
            _get_value(raw, "runtime_enable_measure_twirling", False),
        )
    )
    num_randomizations_raw = _get_value(
        source,
        "num_randomizations",
        _get_value(raw, "runtime_twirling_num_randomizations", None),
    )
    strategy_raw = _get_value(
        source,
        "strategy",
        _get_value(raw, "runtime_twirling_strategy", None),
    )

    num_randomizations = (
        None if num_randomizations_raw in {None, "", "none"} else int(num_randomizations_raw)
    )
    strategy = None if strategy_raw in {None, "", "none"} else str(strategy_raw).strip().lower()

    if num_randomizations is not None and int(num_randomizations) <= 0:
        raise ValueError("runtime twirling num_randomizations must be >= 1.")
    if strategy is not None and strategy not in _ALLOWED_RUNTIME_TWIRLING_STRATEGIES:
        raise ValueError(
            "Unsupported runtime twirling strategy {!r}; expected one of {}.".format(
                strategy, sorted(_ALLOWED_RUNTIME_TWIRLING_STRATEGIES)
            )
        )
    if (num_randomizations is not None or strategy is not None) and not (
        enable_gates or enable_measure
    ):
        raise ValueError(
            "runtime twirling parameters require gate or measure twirling to be enabled."
        )

    return {
        "enable_gates": bool(enable_gates),
        "enable_measure": bool(enable_measure),
        "num_randomizations": num_randomizations,
        "strategy": strategy,
    }


@dataclass(frozen=True)
class ResolvedNoiseSpec:
    executor: str
    noise_kind: str
    backend_profile_kind: str
    mitigation_bundle: str
    layout_policy: str
    schedule_policy: str
    shots: int | None
    seed_transpiler: int | None
    seed_simulator: int | None
    allow_noisy_fallback: bool
    snapshot_path: str | None = None
    fixed_physical_patch: list[int] | None = None
    fixed_couplers: list[list[int]] | None = None
    notes: list[str] = field(default_factory=list)
    labels: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationQubitRecord:
    physical_qubit: int
    T1_s: float | None
    T2_s: float | None
    readout_error: float | None
    readout_p01: float | None
    readout_p10: float | None
    measure_duration_s: float | None
    frequency: float | None


@dataclass(frozen=True)
class CalibrationGateRecord:
    gate_name: str
    qubits: list[int]
    error: float | None
    duration_s: float | None


@dataclass(frozen=True)
class CalibrationSnapshot:
    source_kind: str
    backend_name: str | None
    backend_version: str | None
    processor_family: str | None
    retrieved_at_utc: str
    calibration_time_utc: str | None
    basis_gates: list[str]
    coupling_map: list[list[int]]
    dt: float | None
    per_qubit: list[CalibrationQubitRecord] = field(default_factory=list)
    per_gate: list[CalibrationGateRecord] = field(default_factory=list)
    median_1q_error: float | None = None
    median_2q_error: float | None = None
    median_readout_error: float | None = None
    median_T1: float | None = None
    median_T2: float | None = None
    snapshot_hash: str = ""


@dataclass(frozen=True)
class TranspileSnapshot:
    backend_name: str | None
    optimization_level: int
    layout_policy: str
    schedule_policy: str
    seed_transpiler: int | None
    initial_layout: list[int] | None
    final_layout: list[int] | None
    used_physical_qubits: list[int]
    used_physical_edges: list[list[int]]
    depth: int
    count_1q: int
    count_2q: int
    count_measure: int
    scheduled_duration_total: float | None
    idle_duration_total: float | None
    transpile_hash: str = ""


@dataclass
class NoiseArtifact:
    resolved_spec: ResolvedNoiseSpec
    calibration_snapshot: CalibrationSnapshot | None
    transpile_snapshot: TranspileSnapshot | None
    qiskit_backend: Any | None
    qiskit_noise_model: Any | None
    transpiled_circuit: Any | None
    scheduled_circuit_or_none: Any | None
    warnings: list[str] = field(default_factory=list)
    omitted_channels: list[str] = field(default_factory=list)
    noise_artifact_hash: str = ""
    mapped_observable: Any | None = None
    circuit_structure_hash: str | None = None
    layout_hash: str | None = None
    fixed_couplers_status: dict[str, Any] | None = None
    patch_selection_summary: dict[str, Any] | None = None



def jsonable_noise_value(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable_noise_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): jsonable_noise_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_noise_value(v) for v in value]
    if isinstance(value, set):
        return [jsonable_noise_value(v) for v in sorted(value)]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value



def canonical_noise_json(payload: Any) -> str:
    return json.dumps(jsonable_noise_value(payload), sort_keys=True, separators=(",", ":"))



def stable_noise_hash(payload: Any, *, prefix: str | None = None, length: int = 16) -> str:
    digest = hashlib.sha256(canonical_noise_json(payload).encode("utf-8")).hexdigest()
    if prefix:
        return f"{prefix}_{digest[:int(length)]}"
    return digest[:int(length)]



def resolved_noise_spec_to_dict(spec: ResolvedNoiseSpec) -> dict[str, Any]:
    return jsonable_noise_value(spec)



def calibration_snapshot_to_dict(snapshot: CalibrationSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return jsonable_noise_value(snapshot)



def transpile_snapshot_to_dict(snapshot: TranspileSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return jsonable_noise_value(snapshot)


def canonical_calibration_snapshot_payload(snapshot: CalibrationSnapshot | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    data = (
        calibration_snapshot_to_dict(snapshot)
        if isinstance(snapshot, CalibrationSnapshot)
        else jsonable_noise_value(snapshot)
    )
    if data is None:
        return None
    out = dict(data)
    out.pop("snapshot_hash", None)
    out.pop("retrieved_at_utc", None)
    out["basis_gates"] = _canonical_str_list(out.get("basis_gates", []))
    out["coupling_map"] = _canonical_edge_list(out.get("coupling_map", [])) or []
    per_qubit = [dict(rec) for rec in list(out.get("per_qubit", []))]
    per_qubit.sort(key=lambda rec: int(rec.get("physical_qubit", -1)))
    out["per_qubit"] = per_qubit
    per_gate = [dict(rec) for rec in list(out.get("per_gate", []))]
    for rec in per_gate:
        rec["qubits"] = _canonical_int_list(rec.get("qubits", [])) or []
    per_gate.sort(
        key=lambda rec: (
            str(rec.get("gate_name", "")),
            tuple(int(q) for q in list(rec.get("qubits", []))),
        )
    )
    out["per_gate"] = per_gate
    return out


def canonical_transpile_snapshot_payload(snapshot: TranspileSnapshot | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    data = (
        transpile_snapshot_to_dict(snapshot)
        if isinstance(snapshot, TranspileSnapshot)
        else jsonable_noise_value(snapshot)
    )
    if data is None:
        return None
    out = dict(data)
    out.pop("transpile_hash", None)
    out["initial_layout"] = _canonical_int_list(out.get("initial_layout", None))
    out["final_layout"] = _canonical_int_list(out.get("final_layout", None))
    out["used_physical_qubits"] = _canonical_int_list(out.get("used_physical_qubits", [])) or []
    out["used_physical_edges"] = _canonical_edge_list(out.get("used_physical_edges", [])) or []
    return out



def noise_artifact_metadata(artifact: NoiseArtifact) -> dict[str, Any]:
    return {
        "resolved_spec": resolved_noise_spec_to_dict(artifact.resolved_spec),
        "calibration_snapshot": calibration_snapshot_to_dict(artifact.calibration_snapshot),
        "transpile_snapshot": transpile_snapshot_to_dict(artifact.transpile_snapshot),
        "warnings": [str(x) for x in artifact.warnings],
        "omitted_channels": [str(x) for x in artifact.omitted_channels],
        "noise_artifact_hash": str(artifact.noise_artifact_hash),
        "circuit_structure_hash": artifact.circuit_structure_hash,
        "layout_hash": artifact.layout_hash,
        "fixed_couplers_status": jsonable_noise_value(artifact.fixed_couplers_status),
        "patch_selection_summary": jsonable_noise_value(artifact.patch_selection_summary),
    }



def _normalize_mitigation_bundle(
    raw_mitigation: Any,
    executor: str,
    noise_kind: str,
    runtime_twirling: Mapping[str, Any] | None = None,
) -> str:
    mode = "none"
    if isinstance(raw_mitigation, Mapping):
        mode = str(raw_mitigation.get("mode", raw_mitigation.get("mitigation", "none"))).strip().lower() or "none"
    elif raw_mitigation is not None:
        mode = str(raw_mitigation).strip().lower() or "none"
    runtime_twirling_enabled = bool(
        isinstance(runtime_twirling, Mapping)
        and (
            bool(runtime_twirling.get("enable_gates", False))
            or bool(runtime_twirling.get("enable_measure", False))
        )
    )

    if executor == "statevector":
        return "none"
    if executor == "runtime_qpu":
        if noise_kind == "qpu_layer_learned":
            return "runtime_layer_learned" if mode == "zne" else "none"
        return "runtime_suppressed" if mode in {"zne", "dd", "readout"} or runtime_twirling_enabled else "none"
    if noise_kind == "shots_only":
        return "none"
    if mode == "readout":
        return "readout_only"
    if mode in {"none", ""}:
        return "light_counts" if noise_kind in {"backend_basic", "backend_scheduled", "patch_snapshot"} else "none"
    return "light_counts"



def normalize_to_resolved_noise_spec(raw: Any) -> ResolvedNoiseSpec:
    requested_mode = str(_get_value(raw, "noise_mode", "ideal")).strip().lower() or "ideal"
    aer_noise_kind = str(_get_value(raw, "aer_noise_kind", "scheduled")).strip().lower() or "scheduled"
    backend_profile_raw = _get_value(raw, "backend_profile", None)
    backend_profile = (
        None
        if backend_profile_raw in {None, "", "none"}
        else str(backend_profile_raw).strip().lower()
    )
    snapshot_path_raw = _get_value(raw, "noise_snapshot_json", _get_value(raw, "snapshot_path", None))
    snapshot_path = None if snapshot_path_raw in {None, "", "none"} else str(snapshot_path_raw)
    fixed_patch = _coerce_int_list(_get_value(raw, "fixed_physical_patch", None))
    fixed_couplers = _coerce_edge_list(_get_value(raw, "fixed_couplers", None))
    allow_noisy_fallback = bool(
        _get_value(raw, "allow_noisy_fallback", _get_value(raw, "allow_aer_fallback", False))
    )
    seed_base = _get_value(raw, "seed", 7)
    seed_transpiler = _get_value(raw, "seed_transpiler", seed_base)
    seed_simulator = _get_value(raw, "seed_simulator", seed_base)
    shots_raw = _get_value(raw, "shots", None)
    shots = None if shots_raw is None else int(shots_raw)
    layout_lock_key = _get_value(raw, "layout_lock_key", None)
    mitigation_raw = _get_value(raw, "mitigation", "none")
    layer_noise_model_supplied = _mitigation_has_layer_noise_model(mitigation_raw)
    runtime_twirling = normalize_runtime_twirling_config(raw)
    runtime_twirling_enabled = bool(
        runtime_twirling.get("enable_gates", False) or runtime_twirling.get("enable_measure", False)
    )

    if requested_mode == "ideal":
        executor = "statevector"
        noise_kind = "none"
    elif requested_mode == "shots":
        executor = "aer"
        noise_kind = "shots_only"
    elif requested_mode == "aer_noise":
        executor = "aer"
        noise_kind = {
            "basic": "backend_basic",
            "scheduled": "backend_scheduled",
            "patch_snapshot": "patch_snapshot",
        }.get(aer_noise_kind, "backend_scheduled")
    elif requested_mode == "backend_basic":
        executor = "aer"
        noise_kind = "backend_basic"
    elif requested_mode == "backend_scheduled":
        executor = "aer"
        noise_kind = "backend_scheduled"
    elif requested_mode == "patch_snapshot":
        executor = "aer"
        noise_kind = "patch_snapshot"
    elif requested_mode == "runtime":
        executor = "runtime_qpu"
        if isinstance(mitigation_raw, Mapping):
            mitigation_mode = str(
                mitigation_raw.get("mode", mitigation_raw.get("mitigation", "none"))
            ).strip().lower()
        else:
            mitigation_mode = str(mitigation_raw).strip().lower()
        noise_kind = (
            "qpu_suppressed"
            if mitigation_mode in {"zne", "dd", "readout"} or runtime_twirling_enabled
            else "qpu_raw"
        )
    elif requested_mode in {"qpu_raw", "qpu_suppressed", "qpu_layer_learned"}:
        executor = "runtime_qpu"
        noise_kind = str(requested_mode)
    else:
        raise ValueError(
            "Unsupported noise mode {!r}; expected ideal/shots/aer_noise/runtime or advanced internal values.".format(
                requested_mode
            )
        )

    if backend_profile is None:
        if executor == "statevector":
            backend_profile = "not_applicable"
        elif snapshot_path is not None:
            backend_profile = "frozen_snapshot_json"
        elif bool(_get_value(raw, "use_fake_backend", False)):
            backend_profile = "fake_snapshot"
        elif executor == "aer" and noise_kind == "shots_only" and _get_value(raw, "backend_name", None) in {None, "", "none"}:
            backend_profile = "generic_seeded"
        elif executor == "aer" and noise_kind in {"backend_basic", "backend_scheduled", "patch_snapshot"} and _get_value(raw, "backend_name", None) in {None, "", "none"}:
            backend_profile = "fake_snapshot"
        elif executor == "runtime_qpu":
            backend_profile = "live_backend"
        else:
            backend_profile = "live_backend"

    if fixed_patch:
        layout_policy = "fixed_patch"
    else:
        layout_policy = str(_get_value(raw, "layout_policy", None) or ("frozen_layout" if noise_kind == "patch_snapshot" else "auto_then_lock")).strip().lower()
    schedule_policy = str(
        _get_value(raw, "schedule_policy", None)
        or ("asap" if noise_kind == "backend_scheduled" else "none")
    ).strip().lower()
    mitigation_bundle = _normalize_mitigation_bundle(
        mitigation_raw,
        executor,
        noise_kind,
        runtime_twirling=runtime_twirling,
    )

    spec = ResolvedNoiseSpec(
        executor=str(executor),
        noise_kind=str(noise_kind),
        backend_profile_kind=str(backend_profile),
        mitigation_bundle=str(mitigation_bundle),
        layout_policy=str(layout_policy),
        schedule_policy=str(schedule_policy),
        shots=shots,
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        seed_simulator=(None if seed_simulator is None else int(seed_simulator)),
        allow_noisy_fallback=bool(allow_noisy_fallback),
        snapshot_path=snapshot_path,
        fixed_physical_patch=fixed_patch,
        fixed_couplers=_canonical_edge_list(fixed_couplers),
        notes=[],
        labels={
            "requested_noise_mode": str(requested_mode),
            "backend_name": (None if _get_value(raw, "backend_name", None) in {None, "", "none"} else str(_get_value(raw, "backend_name"))),
            "layout_lock_key": (None if layout_lock_key in {None, "", "none"} else str(layout_lock_key)),
            "runtime_twirling": dict(runtime_twirling),
            "layer_noise_model_supplied": bool(layer_noise_model_supplied),
        },
    )

    if spec.executor not in _ALLOWED_EXECUTORS:
        raise ValueError(f"Unsupported executor {spec.executor!r}.")
    if spec.noise_kind not in _ALLOWED_NOISE_KINDS:
        raise ValueError(f"Unsupported noise_kind {spec.noise_kind!r}.")
    if spec.backend_profile_kind not in _ALLOWED_BACKEND_PROFILES:
        raise ValueError(f"Unsupported backend_profile_kind {spec.backend_profile_kind!r}.")
    if spec.mitigation_bundle not in _ALLOWED_MITIGATION_BUNDLES:
        raise ValueError(f"Unsupported mitigation_bundle {spec.mitigation_bundle!r}.")
    if spec.layout_policy not in _ALLOWED_LAYOUT_POLICIES:
        raise ValueError(f"Unsupported layout_policy {spec.layout_policy!r}.")
    if spec.schedule_policy not in _ALLOWED_SCHEDULE_POLICIES:
        raise ValueError(f"Unsupported schedule_policy {spec.schedule_policy!r}.")
    if spec.executor == "statevector":
        if spec.backend_profile_kind != "not_applicable":
            raise ValueError("ideal/statevector mode must normalize to backend_profile_kind='not_applicable'.")
        if spec.allow_noisy_fallback:
            raise ValueError("allow_noisy_fallback is only valid for noisy Aer execution.")
        if spec.fixed_physical_patch is not None or spec.fixed_couplers is not None:
            raise ValueError("ideal/statevector mode does not accept fixed physical patch or couplers.")
    if spec.executor != "aer" and spec.allow_noisy_fallback:
        raise ValueError("allow_noisy_fallback is only valid for local Aer execution.")
    if spec.backend_profile_kind == "frozen_snapshot_json" and spec.snapshot_path is None:
        raise ValueError("backend_profile_kind='frozen_snapshot_json' requires noise_snapshot_json/snapshot_path.")
    if spec.executor == "runtime_qpu" and spec.backend_profile_kind != "live_backend":
        raise ValueError("runtime_qpu mode requires backend_profile_kind='live_backend'.")
    if requested_mode == "qpu_raw" and spec.mitigation_bundle != "none":
        raise ValueError(
            "qpu_raw requires mitigation='none' and runtime twirling disabled so the normalized runtime bundle remains raw."
        )
    if requested_mode == "qpu_suppressed" and spec.mitigation_bundle != "runtime_suppressed":
        raise ValueError(
            "qpu_suppressed requires mitigation {readout,zne,dd} or runtime twirling so the normalized runtime bundle is runtime_suppressed."
        )
    if layer_noise_model_supplied and requested_mode != "qpu_layer_learned":
        raise ValueError(
            "layer_noise_model / layer_noise_model_json are only supported on noise_mode='qpu_layer_learned' in the current runtime path."
        )
    if requested_mode == "qpu_layer_learned" and spec.mitigation_bundle != "runtime_layer_learned":
        raise ValueError(
            "qpu_layer_learned requires mitigation='zne' so the normalized runtime bundle is runtime_layer_learned."
        )
    if requested_mode == "qpu_layer_learned" and runtime_twirling_enabled:
        raise ValueError(
            "qpu_layer_learned uses the runtime layer-noise-backed PEA path and does not accept explicit runtime twirling settings."
        )
    if spec.noise_kind == "shots_only":
        if spec.shots is None or int(spec.shots) <= 0:
            raise ValueError("shots_only requires shots > 0.")
        if spec.mitigation_bundle != "none":
            raise ValueError("shots_only does not support mitigation bundles in the normalized spec.")
    if spec.noise_kind == "backend_basic" and spec.schedule_policy != "none":
        raise ValueError("backend_basic must use schedule_policy='none'; use backend_scheduled for schedule-aware runs.")
    if spec.noise_kind == "backend_scheduled" and spec.schedule_policy != "asap":
        raise ValueError("backend_scheduled must use schedule_policy='asap'.")
    if spec.noise_kind == "patch_snapshot" and spec.layout_policy != "frozen_layout":
        raise ValueError("patch_snapshot requires layout_policy='frozen_layout'.")
    if spec.fixed_couplers is not None and spec.fixed_physical_patch is None:
        raise ValueError("fixed_couplers requires fixed_physical_patch to be specified.")
    if spec.fixed_couplers is not None and spec.fixed_physical_patch is not None:
        patch_set = {int(q) for q in list(spec.fixed_physical_patch)}
        invalid_edges = [
            [int(edge[0]), int(edge[1])]
            for edge in list(spec.fixed_couplers)
            if int(edge[0]) not in patch_set or int(edge[1]) not in patch_set
        ]
        if invalid_edges:
            raise ValueError(
                "fixed_couplers must lie within fixed_physical_patch; invalid edges="
                f"{invalid_edges} patch={list(spec.fixed_physical_patch)}."
            )
    return spec



def resolved_noise_spec_hash(spec: ResolvedNoiseSpec) -> str:
    return stable_noise_hash(resolved_noise_spec_to_dict(spec), prefix="nspec")
