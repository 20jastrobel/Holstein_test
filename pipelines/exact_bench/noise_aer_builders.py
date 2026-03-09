#!/usr/bin/env python3
"""Local Aer noise-artifact builders for exact-bench HH/Hubbard wrappers."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from pipelines.exact_bench.noise_model_spec import (
    CalibrationSnapshot,
    NoiseArtifact,
    ResolvedNoiseSpec,
    TranspileSnapshot,
    calibration_snapshot_to_dict,
    canonical_transpile_snapshot_payload,
    resolved_noise_spec_hash,
    stable_noise_hash,
    transpile_snapshot_to_dict,
)


_LAYOUT_LOCK_REGISTRY: dict[str, dict[str, Any]] = {}
_TEMPLATE_CACHE: dict[str, dict[str, Any]] = {}
_NOISE_MODEL_CACHE: dict[str, Any] = {}
_LAYOUT_LOCK_CACHE_DIR = Path(tempfile.gettempdir()) / "holstein_noise_layout_locks"



def _numeric_like(value: Any) -> bool:
    return isinstance(value, (int, float, complex, np.number))



def _param_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        if abs(value.imag) <= 1e-15:
            return float(value.real)
        return complex(value)
    return float(value)



def _circuit_structure_template(circuit: QuantumCircuit) -> tuple[QuantumCircuit, list[Parameter], list[Any], str]:
    template = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
    ordered_params: list[Parameter] = []
    ordered_values: list[Any] = []
    signature_rows: list[dict[str, Any]] = []
    param_counter = 0
    for inst in circuit.data:
        op = inst.operation.to_mutable() if hasattr(inst.operation, "to_mutable") else inst.operation.copy()
        new_params: list[Any] = []
        param_kind_sig: list[str] = []
        for param in list(getattr(op, "params", [])):
            if _numeric_like(param):
                p = Parameter(f"__noise_tpl_{param_counter}")
                param_counter += 1
                ordered_params.append(p)
                ordered_values.append(_param_value(param))
                new_params.append(p)
                param_kind_sig.append("num")
            else:
                new_params.append(param)
                param_kind_sig.append(f"lit:{type(param).__name__}:{str(param)}")
        if hasattr(op, "params") and list(getattr(op, "params", [])) != new_params:
            op.params = new_params
        qidx = [int(circuit.find_bit(q).index) for q in inst.qubits]
        cidx = [int(circuit.find_bit(c).index) for c in inst.clbits]
        signature_rows.append(
            {
                "name": str(op.name),
                "label": getattr(op, "label", None),
                "qargs": qidx,
                "cargs": cidx,
                "params": param_kind_sig,
            }
        )
        template.append(op, inst.qubits, inst.clbits)
    structure_hash = stable_noise_hash({"num_qubits": int(circuit.num_qubits), "rows": signature_rows}, prefix="cstruct")
    return template, ordered_params, ordered_values, structure_hash



def _layout_lock_registry_key(
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
) -> str:
    requested = resolved_spec.labels.get("layout_lock_key", None)
    return stable_noise_hash(
        {
            "requested": requested,
            "backend_name": None if calibration_snapshot is None else calibration_snapshot.backend_name,
            "snapshot_hash": None if calibration_snapshot is None else calibration_snapshot.snapshot_hash,
            "layout_policy": resolved_spec.layout_policy,
            "schedule_policy": resolved_spec.schedule_policy,
            "fixed_physical_patch": resolved_spec.fixed_physical_patch,
        },
        prefix="llock",
    )



def _layout_lock_file(lock_key: str) -> Path:
    return _LAYOUT_LOCK_CACHE_DIR / f"{str(lock_key)}.json"



def _load_persisted_layout_lock(lock_key: str) -> dict[str, Any] | None:
    path = _layout_lock_file(lock_key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload



def _persist_layout_lock(lock_key: str, payload: Mapping[str, Any]) -> None:
    try:
        _LAYOUT_LOCK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _layout_lock_file(lock_key).write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    except Exception:
        return



def _resolve_initial_layout(
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
    logical_qubits: int,
) -> list[int] | None:
    if resolved_spec.fixed_physical_patch is not None:
        if len(list(resolved_spec.fixed_physical_patch)) < int(logical_qubits):
            raise ValueError(
                f"fixed physical patch {resolved_spec.fixed_physical_patch} smaller than logical qubit count {logical_qubits}."
            )
        return [int(x) for x in list(resolved_spec.fixed_physical_patch)[: int(logical_qubits)]]
    lock_key = _layout_lock_registry_key(resolved_spec, calibration_snapshot)
    rec = _LAYOUT_LOCK_REGISTRY.get(lock_key)
    if rec is None:
        rec = _load_persisted_layout_lock(lock_key)
        if rec is not None:
            _LAYOUT_LOCK_REGISTRY[lock_key] = dict(rec)
    if rec is None:
        return None
    final_layout = rec.get("final_layout", None)
    initial_layout = rec.get("initial_layout", None)
    chosen = final_layout if final_layout else initial_layout
    if chosen is None:
        return None
    return [int(x) for x in list(chosen)[: int(logical_qubits)]]



def _store_layout_lock(
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
    transpile_snapshot: TranspileSnapshot,
    layout_hash: str,
) -> None:
    lock_key = _layout_lock_registry_key(resolved_spec, calibration_snapshot)
    payload = {
        "initial_layout": list(transpile_snapshot.initial_layout or []),
        "final_layout": list(transpile_snapshot.final_layout or []),
        "used_physical_qubits": list(transpile_snapshot.used_physical_qubits),
        "used_physical_edges": [list(edge) for edge in transpile_snapshot.used_physical_edges],
        "layout_hash": str(layout_hash),
    }
    _LAYOUT_LOCK_REGISTRY[lock_key] = dict(payload)
    _persist_layout_lock(lock_key, payload)



def _extract_layout_lists(circuit: QuantumCircuit, logical_qubits: int) -> tuple[list[int] | None, list[int] | None]:
    layout = getattr(circuit, "layout", None)
    initial_layout: list[int] | None = None
    final_layout: list[int] | None = None
    if layout is not None:
        input_map = dict(getattr(layout, "input_qubit_mapping", {}) or {})
        initial_layout_obj = getattr(layout, "initial_layout", None)
        if initial_layout_obj is not None and hasattr(initial_layout_obj, "get_virtual_bits"):
            v2p = dict(initial_layout_obj.get_virtual_bits())
            keyed: dict[int, int] = {}
            for virt, logical_idx in input_map.items():
                if int(logical_idx) < int(logical_qubits) and virt in v2p:
                    keyed[int(logical_idx)] = int(v2p[virt])
            if keyed and len(keyed) >= int(logical_qubits):
                initial_layout = [int(keyed[i]) for i in range(int(logical_qubits))]
        try:
            final_idx = list(layout.final_index_layout())
            if final_idx:
                final_layout = [int(x) for x in final_idx[: int(logical_qubits)]]
        except Exception:
            final_layout = None
    return initial_layout, final_layout



def _physical_qubit_map(circuit: QuantumCircuit) -> dict[Any, int]:
    layout = getattr(circuit, "layout", None)
    output_qubits = list(getattr(layout, "_output_qubit_list", []) or [])
    if output_qubits:
        return {qubit: int(idx) for idx, qubit in enumerate(output_qubits)}
    return {qubit: int(circuit.find_bit(qubit).index) for qubit in circuit.qubits}



def _used_edges_and_counts(
    circuit: QuantumCircuit,
) -> tuple[list[list[int]], list[int], int, int, int]:
    physical_map = _physical_qubit_map(circuit)
    used_edges: set[tuple[int, int]] = set()
    used_qubits: set[int] = set()
    count_1q = 0
    count_2q = 0
    count_measure = 0
    for inst in circuit.data:
        qidx = [int(physical_map[q]) for q in inst.qubits]
        if str(inst.operation.name) == "delay":
            continue
        used_qubits.update(int(idx) for idx in qidx)
        if str(inst.operation.name) == "measure":
            count_measure += 1
            continue
        if len(qidx) == 1:
            count_1q += 1
        elif len(qidx) == 2:
            count_2q += 1
            used_edges.add((int(qidx[0]), int(qidx[1])))
    return (
        [[int(a), int(b)] for a, b in sorted(used_edges)],
        [int(x) for x in sorted(used_qubits)],
        int(count_1q),
        int(count_2q),
        int(count_measure),
    )



def _duration_seconds(duration: Any, unit: Any, dt: float | None) -> float | None:
    if duration is None:
        return None
    try:
        val = float(duration)
    except Exception:
        return None
    if unit == "s":
        return float(val)
    if unit == "dt":
        if dt is None:
            return None
        return float(val) * float(dt)
    return None



def _idle_duration_seconds(circuit: QuantumCircuit, dt: float | None) -> float | None:
    total = 0.0
    seen = False
    for inst in circuit.data:
        if str(inst.operation.name) != "delay":
            continue
        seen = True
        op = inst.operation
        dur = _duration_seconds(getattr(op, "duration", None), getattr(op, "unit", None), dt)
        if dur is not None:
            total += float(dur)
    return float(total) if seen else 0.0



def _build_transpile_snapshot(
    transpiled: QuantumCircuit,
    *,
    logical_qubits: int,
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
    optimization_level: int,
) -> tuple[TranspileSnapshot, str]:
    initial_layout, final_layout = _extract_layout_lists(transpiled, logical_qubits)
    used_edges, used_qubits_from_ops, count_1q, count_2q, count_measure = _used_edges_and_counts(
        transpiled
    )
    fallback_qubits = []
    if (not used_qubits_from_ops) and len(list(transpiled.data)) == 0:
        fallback_qubits = list(
            final_layout
            or initial_layout
            or [int(transpiled.find_bit(q).index) for q in list(transpiled.qubits)[:logical_qubits]]
        )
    used_physical_qubits = sorted(set(used_qubits_from_ops or fallback_qubits))
    scheduled_duration_total = _duration_seconds(
        getattr(transpiled, "duration", None),
        getattr(transpiled, "unit", None),
        None if calibration_snapshot is None else calibration_snapshot.dt,
    )
    idle_duration_total = _idle_duration_seconds(
        transpiled,
        None if calibration_snapshot is None else calibration_snapshot.dt,
    )
    if resolved_spec.schedule_policy == "asap" and scheduled_duration_total is None:
        raise RuntimeError(
            "backend_scheduled requires schedule metadata, but transpile did not produce a scheduled duration."
        )
    snapshot = TranspileSnapshot(
        backend_name=(None if calibration_snapshot is None else calibration_snapshot.backend_name),
        optimization_level=int(optimization_level),
        layout_policy=str(resolved_spec.layout_policy),
        schedule_policy=str(resolved_spec.schedule_policy),
        seed_transpiler=resolved_spec.seed_transpiler,
        initial_layout=initial_layout,
        final_layout=final_layout,
        used_physical_qubits=[int(x) for x in used_physical_qubits],
        used_physical_edges=[[int(a), int(b)] for a, b in used_edges],
        depth=int(transpiled.depth()),
        count_1q=int(count_1q),
        count_2q=int(count_2q),
        count_measure=int(count_measure),
        scheduled_duration_total=scheduled_duration_total,
        idle_duration_total=idle_duration_total,
        transpile_hash="",
    )
    layout_hash = stable_noise_hash(
        {
            "initial_layout": snapshot.initial_layout,
            "final_layout": snapshot.final_layout,
            "used_physical_qubits": snapshot.used_physical_qubits,
            "used_physical_edges": snapshot.used_physical_edges,
            "layout_policy": snapshot.layout_policy,
        },
        prefix="layout",
    )
    snapshot = replace(
        snapshot,
        transpile_hash=stable_noise_hash(
            canonical_transpile_snapshot_payload(snapshot),
            prefix="tx",
        ),
    )
    return snapshot, layout_hash



def _cache_key(
    *,
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
    circuit_structure_hash: str,
    parameter_value_hash: str,
    observable_hash: str | None,
    optimization_level: int,
) -> str:
    return stable_noise_hash(
        {
            "resolved_spec_hash": resolved_noise_spec_hash(resolved_spec),
            "snapshot_hash": None if calibration_snapshot is None else calibration_snapshot.snapshot_hash,
            "layout_lock_key": _layout_lock_registry_key(resolved_spec, calibration_snapshot),
            "optimization_level": int(optimization_level),
            "circuit_structure_hash": str(circuit_structure_hash),
            "parameter_value_hash": str(parameter_value_hash),
            "observable_hash": observable_hash,
        },
        prefix="artcache",
    )



def _observable_hash(observable: SparsePauliOp | None) -> str | None:
    if observable is None:
        return None
    return stable_noise_hash(
        {"num_qubits": int(observable.num_qubits), "terms": [(str(lbl), complex(coeff)) for lbl, coeff in observable.to_list()]},
        prefix="obs",
    )



def _bind_cached_template(record: dict[str, Any], values: list[Any]) -> QuantumCircuit:
    params = list(record.get("ordered_params", []))
    template = record["transpiled_template"]
    if len(params) != len(values):
        raise RuntimeError(
            f"Template parameter/value mismatch: {len(params)} cached params vs {len(values)} values."
        )
    if not params:
        return template.copy()
    return template.assign_parameters({param: val for param, val in zip(params, values)}, inplace=False)



def _transpile_and_cache(
    *,
    circuit: QuantumCircuit,
    resolved_spec: ResolvedNoiseSpec,
    resolved_backend: Any,
    calibration_snapshot: CalibrationSnapshot | None,
    optimization_level: int,
) -> tuple[QuantumCircuit, TranspileSnapshot, str, str]:
    template, ordered_params, ordered_values, circuit_structure_hash = _circuit_structure_template(circuit)
    parameter_value_hash = stable_noise_hash({"values": ordered_values}, prefix="pvals")
    cache_key = _cache_key(
        resolved_spec=resolved_spec,
        calibration_snapshot=calibration_snapshot,
        circuit_structure_hash=circuit_structure_hash,
        parameter_value_hash=parameter_value_hash,
        observable_hash=None,
        optimization_level=optimization_level,
    )
    if cache_key in _TEMPLATE_CACHE:
        cached = _TEMPLATE_CACHE[cache_key]
        return cached["transpiled_circuit"].copy(), cached["transpile_snapshot"], cached["layout_hash"], circuit_structure_hash

    initial_layout = _resolve_initial_layout(resolved_spec, calibration_snapshot, int(circuit.num_qubits))
    scheduling_method = "asap" if resolved_spec.schedule_policy == "asap" else None
    bound_circuit = _bind_cached_template(
        {
            "ordered_params": ordered_params,
            "transpiled_template": template,
        },
        ordered_values,
    )
    transpiled = transpile(
        bound_circuit,
        backend=resolved_backend,
        optimization_level=int(optimization_level),
        seed_transpiler=(None if resolved_spec.seed_transpiler is None else int(resolved_spec.seed_transpiler)),
        initial_layout=initial_layout,
        scheduling_method=scheduling_method,
    )
    transpile_snapshot, layout_hash = _build_transpile_snapshot(
        transpiled,
        logical_qubits=int(circuit.num_qubits),
        resolved_spec=resolved_spec,
        calibration_snapshot=calibration_snapshot,
        optimization_level=int(optimization_level),
    )
    _store_layout_lock(resolved_spec, calibration_snapshot, transpile_snapshot, layout_hash)
    _TEMPLATE_CACHE[cache_key] = {
        "transpiled_circuit": transpiled.copy(),
        "transpile_snapshot": transpile_snapshot,
        "layout_hash": layout_hash,
    }
    return transpiled, transpile_snapshot, layout_hash, circuit_structure_hash



def transpile_and_lock_patch(
    circuit: QuantumCircuit,
    observable: SparsePauliOp | None,
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
    *,
    qiskit_backend: Any,
    qiskit_noise_model: Any | None,
    omitted_channels: list[str],
    warnings: list[str],
    optimization_level: int = 1,
) -> NoiseArtifact:
    transpiled_circuit, transpile_snapshot, layout_hash, circuit_structure_hash = _transpile_and_cache(
        circuit=circuit,
        resolved_spec=resolved_spec,
        resolved_backend=qiskit_backend,
        calibration_snapshot=calibration_snapshot,
        optimization_level=int(optimization_level),
    )
    mapped_observable = None if observable is None else observable.apply_layout(
        getattr(transpiled_circuit, "layout", None),
        num_qubits=int(transpiled_circuit.num_qubits),
    )
    artifact = NoiseArtifact(
        resolved_spec=resolved_spec,
        calibration_snapshot=calibration_snapshot,
        transpile_snapshot=transpile_snapshot,
        qiskit_backend=qiskit_backend,
        qiskit_noise_model=qiskit_noise_model,
        transpiled_circuit=transpiled_circuit,
        scheduled_circuit_or_none=(transpiled_circuit if resolved_spec.schedule_policy == "asap" else None),
        warnings=[str(x) for x in warnings],
        omitted_channels=[str(x) for x in omitted_channels],
        mapped_observable=mapped_observable,
        circuit_structure_hash=str(circuit_structure_hash),
        layout_hash=str(layout_hash),
        noise_artifact_hash="",
    )
    return replace(
        artifact,
        noise_artifact_hash=stable_noise_hash(
            {
                "resolved_spec_hash": resolved_noise_spec_hash(resolved_spec),
                "snapshot_hash": None if calibration_snapshot is None else calibration_snapshot.snapshot_hash,
                "transpile_hash": None if transpile_snapshot is None else transpile_snapshot.transpile_hash,
                "omitted_channels": sorted(str(x) for x in list(omitted_channels)),
                "layout_hash": layout_hash,
            },
            prefix="nart",
        ),
    )



def _noise_model_from_backend(resolved_backend: Any, cache_key: str) -> Any:
    if cache_key in _NOISE_MODEL_CACHE:
        return _NOISE_MODEL_CACHE[cache_key]
    try:
        from qiskit_aer.noise import NoiseModel
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("Failed to import qiskit_aer.noise.NoiseModel.") from exc
    model = NoiseModel.from_backend(resolved_backend)
    _NOISE_MODEL_CACHE[cache_key] = model
    return model



def build_shots_only_artifact(
    *,
    circuit: QuantumCircuit,
    observable: SparsePauliOp | None,
    resolved_spec: ResolvedNoiseSpec,
    resolved_backend: Any,
    calibration_snapshot: CalibrationSnapshot | None,
) -> NoiseArtifact:
    warnings: list[str] = []
    if resolved_spec.schedule_policy == "asap":
        warnings.append("shots_only scheduled via requested schedule_policy for benchmarking metadata only.")
    return transpile_and_lock_patch(
        circuit,
        observable,
        resolved_spec,
        calibration_snapshot,
        qiskit_backend=resolved_backend,
        qiskit_noise_model=None,
        omitted_channels=[],
        warnings=warnings,
    )



def build_backend_basic_artifact(
    *,
    circuit: QuantumCircuit,
    observable: SparsePauliOp | None,
    resolved_spec: ResolvedNoiseSpec,
    resolved_backend: Any,
    calibration_snapshot: CalibrationSnapshot,
) -> NoiseArtifact:
    noise_model = _noise_model_from_backend(resolved_backend, f"basic:{calibration_snapshot.snapshot_hash}")
    unscheduled_spec = replace(resolved_spec, schedule_policy="none")
    return transpile_and_lock_patch(
        circuit,
        observable,
        unscheduled_spec,
        calibration_snapshot,
        qiskit_backend=resolved_backend,
        qiskit_noise_model=noise_model,
        omitted_channels=[
            "delay_relaxation_if_unscheduled",
            "crosstalk",
            "leakage",
            "drift",
            "coherent_overrotation",
        ],
        warnings=["backend_basic is a smoke/debug hardware-facing approximation; scheduling is not enforced."],
    )



def build_backend_scheduled_artifact(
    *,
    circuit: QuantumCircuit,
    observable: SparsePauliOp | None,
    resolved_spec: ResolvedNoiseSpec,
    resolved_backend: Any,
    calibration_snapshot: CalibrationSnapshot,
) -> NoiseArtifact:
    if str(resolved_spec.schedule_policy) != "asap":
        resolved_spec = replace(resolved_spec, schedule_policy="asap")
    noise_model = _noise_model_from_backend(resolved_backend, f"sched:{calibration_snapshot.snapshot_hash}")
    return transpile_and_lock_patch(
        circuit,
        observable,
        resolved_spec,
        calibration_snapshot,
        qiskit_backend=resolved_backend,
        qiskit_noise_model=noise_model,
        omitted_channels=[
            "crosstalk",
            "leakage",
            "non_markovian_drift",
            "coherent_overrotation",
        ],
        warnings=[],
    )



def build_patch_snapshot_artifact(
    *,
    circuit: QuantumCircuit,
    observable: SparsePauliOp | None,
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
) -> NoiseArtifact:
    raise NotImplementedError(
        "patch_snapshot is wired for phase-2 replay but is not implemented in phase 1. "
        "Persisted calibration snapshots are written/read now; Aer rehydration from frozen snapshots comes next."
    )
