#!/usr/bin/env python3
"""Local Aer noise-artifact builders for exact-bench HH/Hubbard wrappers."""

from __future__ import annotations

import io
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.providers.backend import BackendV2, QubitProperties
from qiskit.providers.options import Options
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap, InstructionProperties, Target

from pipelines.exact_bench.noise_model_spec import (
    CalibrationSnapshot,
    NoiseArtifact,
    ResolvedNoiseSpec,
    TranspileSnapshot,
    calibration_snapshot_to_dict,
    canonical_transpile_snapshot_payload,
    jsonable_noise_value,
    resolved_noise_spec_hash,
    stable_noise_hash,
    transpile_snapshot_to_dict,
)
from pipelines.exact_bench.noise_snapshot import (
    load_snapshot_layout_lock,
    write_snapshot_layout_lock,
)


_LAYOUT_LOCK_REGISTRY: dict[str, dict[str, Any]] = {}
_LAYOUT_LOCK_SOURCE_REGISTRY: dict[str, str] = {}
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



def _detached_circuit_copy(circuit: QuantumCircuit) -> QuantumCircuit:
    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    buffer.seek(0)
    loaded = qpy.load(buffer)
    if not loaded:
        raise RuntimeError("QPY circuit clone produced no circuits.")
    return loaded[0]



def _circuit_structure_template(circuit: QuantumCircuit) -> tuple[QuantumCircuit, list[Parameter], list[Any], str]:
    detached = _detached_circuit_copy(circuit)
    template = QuantumCircuit(*detached.qregs, *detached.cregs, name=detached.name)
    ordered_params: list[Parameter] = []
    ordered_values: list[Any] = []
    signature_rows: list[dict[str, Any]] = []
    param_counter = 0
    for inst in detached.data:
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
        qidx = [int(detached.find_bit(q).index) for q in inst.qubits]
        cidx = [int(detached.find_bit(c).index) for c in inst.clbits]
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
            "backend_name": (
                resolved_spec.labels.get("backend_name", None)
                if calibration_snapshot is None
                else calibration_snapshot.backend_name
            ),
            "snapshot_hash": None if calibration_snapshot is None else calibration_snapshot.snapshot_hash,
            "layout_policy": _layout_lock_policy_scope(resolved_spec.layout_policy),
            "schedule_policy": resolved_spec.schedule_policy,
            "fixed_physical_patch": resolved_spec.fixed_physical_patch,
        },
        prefix="llock",
    )



def _layout_lock_file(lock_key: str) -> Path:
    return _LAYOUT_LOCK_CACHE_DIR / f"{str(lock_key)}.json"



def _layout_lock_policy_scope(layout_policy: str | None) -> str:
    policy = str(layout_policy or "").strip().lower() or "auto_then_lock"
    if policy in {"auto_then_lock", "frozen_layout"}:
        return "locked_layout"
    return str(policy)



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


def _load_snapshot_bundled_layout_lock(
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
    lock_key: str,
) -> dict[str, Any] | None:
    if calibration_snapshot is None or resolved_spec.snapshot_path in {None, ""}:
        return None
    try:
        record = load_snapshot_layout_lock(str(resolved_spec.snapshot_path), str(lock_key))
    except Exception:
        return None
    if record is None:
        return None
    return dict(record)



def _load_layout_lock_record_with_source(
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
) -> tuple[dict[str, Any] | None, str | None]:
    lock_key = _layout_lock_registry_key(resolved_spec, calibration_snapshot)
    rec = _LAYOUT_LOCK_REGISTRY.get(lock_key)
    if rec is not None:
        return dict(rec), str(_LAYOUT_LOCK_SOURCE_REGISTRY.get(lock_key, "persisted_lock"))
    rec = _load_snapshot_bundled_layout_lock(resolved_spec, calibration_snapshot, lock_key)
    if rec is not None:
        _LAYOUT_LOCK_REGISTRY[lock_key] = dict(rec)
        _LAYOUT_LOCK_SOURCE_REGISTRY[lock_key] = "snapshot_bundle_lock"
        return dict(rec), "snapshot_bundle_lock"
    rec = _load_persisted_layout_lock(lock_key)
    if rec is not None:
        _LAYOUT_LOCK_REGISTRY[lock_key] = dict(rec)
        _LAYOUT_LOCK_SOURCE_REGISTRY[lock_key] = "persisted_lock"
        return dict(rec), "persisted_lock"
    return None, None



def _load_layout_lock_record(
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
) -> dict[str, Any] | None:
    rec, _source = _load_layout_lock_record_with_source(resolved_spec, calibration_snapshot)
    return rec



def describe_layout_anchor_source(
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
) -> str:
    if resolved_spec.fixed_physical_patch is not None:
        return "fixed_patch"
    rec, source = _load_layout_lock_record_with_source(resolved_spec, calibration_snapshot)
    if rec is not None and source is not None:
        return str(source)
    if str(resolved_spec.layout_policy) == "frozen_layout":
        return "missing_frozen_lock"
    return "fresh_auto_then_lock"



def _snapshot_num_qubits(snapshot: CalibrationSnapshot) -> int:
    max_index = -1
    if snapshot.per_qubit:
        max_index = max(max_index, int(len(snapshot.per_qubit) - 1))
    for edge in list(snapshot.coupling_map or []):
        if len(edge) >= 2:
            max_index = max(max_index, int(edge[0]), int(edge[1]))
    for rec in list(snapshot.per_gate or []):
        for qubit in list(rec.qubits or []):
            max_index = max(max_index, int(qubit))
    return max(1, int(max_index + 1))


def _snapshot_basis_gates(snapshot: CalibrationSnapshot) -> list[str]:
    basis: list[str] = []
    seen: set[str] = set()
    for raw_name in list(snapshot.basis_gates or []):
        gate_name = str(raw_name).strip()
        if not gate_name or gate_name in seen:
            continue
        seen.add(gate_name)
        basis.append(gate_name)
    for required in ("measure", "reset"):
        if required not in seen:
            basis.append(required)
            seen.add(required)
    if snapshot.dt is not None and "delay" not in seen:
        basis.append("delay")
    return basis


def _snapshot_gate_property_map(snapshot: CalibrationSnapshot) -> dict[tuple[str, tuple[int, ...]], Any]:
    out: dict[tuple[str, tuple[int, ...]], Any] = {}
    for rec in list(snapshot.per_gate or []):
        key = (str(rec.gate_name), tuple(int(q) for q in list(rec.qubits or [])))
        out[key] = rec
    return out


def _measure_instruction_property(record: Any) -> InstructionProperties:
    readout_error = None if record.readout_error is None else float(record.readout_error)
    p10 = getattr(record, "readout_p10", None)
    p01 = getattr(record, "readout_p01", None)
    if p10 is None:
        p10 = (0.5 * readout_error) if readout_error is not None else 0.0
    if p01 is None:
        p01 = (0.5 * readout_error) if readout_error is not None else 0.0
    prop = InstructionProperties(
        duration=(None if record.measure_duration_s is None else float(record.measure_duration_s)),
        error=(float(readout_error) if readout_error is not None else float(max(p10, p01))),
    )
    prop.prob_meas1_prep0 = float(max(0.0, min(1.0, p10)))
    prop.prob_meas0_prep1 = float(max(0.0, min(1.0, p01)))
    return prop


def _build_target_from_snapshot(
    snapshot: CalibrationSnapshot,
    *,
    num_qubits: int,
) -> tuple[Target, list[str]]:
    target = Target(
        num_qubits=int(num_qubits),
        dt=(None if snapshot.dt is None else float(snapshot.dt)),
        qubit_properties=[
            QubitProperties(
                t1=(None if rec.T1_s is None else float(rec.T1_s)),
                t2=(None if rec.T2_s is None else float(rec.T2_s)),
                frequency=(None if rec.frequency is None else float(rec.frequency)),
            )
            for rec in list(snapshot.per_qubit or [])[: int(num_qubits)]
        ]
        + [
            QubitProperties(t1=None, t2=None, frequency=None)
            for _ in range(max(0, int(num_qubits) - len(list(snapshot.per_qubit or []))))
        ],
    )
    gate_map = get_standard_gate_name_mapping()
    gate_props = _snapshot_gate_property_map(snapshot)
    edges = [tuple(int(q) for q in edge[:2]) for edge in list(snapshot.coupling_map or []) if len(edge) >= 2]
    unsupported: list[str] = []

    for gate_name in _snapshot_basis_gates(snapshot):
        instruction = gate_map.get(str(gate_name), None)
        if instruction is None:
            unsupported.append(str(gate_name))
            continue
        properties: dict[tuple[int, ...], InstructionProperties | None] = {}
        if int(instruction.num_qubits) == 1:
            for q in range(int(num_qubits)):
                if str(gate_name) == "measure":
                    if q < len(list(snapshot.per_qubit or [])):
                        properties[(int(q),)] = _measure_instruction_property(snapshot.per_qubit[int(q)])
                    else:
                        properties[(int(q),)] = None
                    continue
                rec = gate_props.get((str(gate_name), (int(q),)))
                properties[(int(q),)] = (
                    None
                    if rec is None
                    else InstructionProperties(
                        duration=(None if rec.duration_s is None else float(rec.duration_s)),
                        error=(None if rec.error is None else float(rec.error)),
                    )
                )
        elif int(instruction.num_qubits) == 2:
            all_edges = sorted(set(edges + [key[1] for key in gate_props.keys() if len(key[1]) == 2]))
            if not all_edges:
                continue
            for edge in all_edges:
                rec = gate_props.get((str(gate_name), tuple(int(q) for q in edge)))
                properties[tuple(int(q) for q in edge)] = (
                    None
                    if rec is None
                    else InstructionProperties(
                        duration=(None if rec.duration_s is None else float(rec.duration_s)),
                        error=(None if rec.error is None else float(rec.error)),
                    )
                )
        else:
            continue
        target.add_instruction(instruction, properties, name=str(gate_name))
    return target, unsupported


class _SnapshotBackendV2(BackendV2):
    def __init__(self, snapshot: CalibrationSnapshot, *, backend_name_override: str | None = None):
        self._snapshot = snapshot
        self._num_qubits = _snapshot_num_qubits(snapshot)
        self._target, self._unsupported_basis_gates = _build_target_from_snapshot(
            snapshot,
            num_qubits=int(self._num_qubits),
        )
        self._coupling_map = (
            None
            if not list(snapshot.coupling_map or [])
            else CouplingMap([[int(edge[0]), int(edge[1])] for edge in list(snapshot.coupling_map or [])])
        )
        super().__init__(
            name=str(backend_name_override or snapshot.backend_name or "frozen_snapshot_json"),
            backend_version=str(snapshot.backend_version or "snapshot"),
        )

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> int | None:
        return None

    @property
    def coupling_map(self) -> CouplingMap | None:
        return self._coupling_map

    @property
    def dt(self) -> float | None:
        return None if self._snapshot.dt is None else float(self._snapshot.dt)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("_SnapshotBackendV2 is transpile/noise-model only and cannot execute circuits.")


def build_backend_from_calibration_snapshot(
    snapshot: CalibrationSnapshot,
    *,
    backend_name_override: str | None = None,
) -> tuple[Any, list[str]]:
    backend = _SnapshotBackendV2(
        snapshot,
        backend_name_override=(None if backend_name_override in {None, ""} else str(backend_name_override)),
    )
    warnings: list[str] = []
    unsupported = getattr(backend, "_unsupported_basis_gates", [])
    if unsupported:
        warnings.append(
            "snapshot_backend_ignored_unsupported_basis_gates="
            + ",".join(sorted(str(x) for x in list(unsupported)))
        )
    return backend, warnings



def _canonical_edge_tuples(raw: Any) -> set[tuple[int, int]]:
    return {
        tuple(int(q) for q in list(edge)[:2])
        for edge in list(raw or [])
        if len(list(edge)) >= 2
    }



def _canonical_edge_payload(raw: Any) -> list[list[int]]:
    return [[int(a), int(b)] for a, b in sorted(_canonical_edge_tuples(raw))]



def _circuit_has_two_qubit_ops(circuit: QuantumCircuit) -> bool:
    return any(len(list(inst.qubits)) == 2 for inst in list(circuit.data))



def _filter_target_to_subgraph(
    base_target: Target,
    *,
    allowed_qubits: set[int],
    allowed_edges: set[tuple[int, int]],
) -> tuple[Target, set[tuple[int, int]]]:
    num_qubits = int(base_target.num_qubits)
    qprops = list(getattr(base_target, "qubit_properties", []) or [])
    if qprops and len(qprops) < num_qubits:
        qprops = list(qprops) + [None] * max(0, int(num_qubits - len(qprops)))
    filtered = Target(
        num_qubits=num_qubits,
        dt=getattr(base_target, "dt", None),
        qubit_properties=(qprops if qprops else None),
    )
    supported_twoq_edges: set[tuple[int, int]] = set()
    for name in list(base_target.operation_names):
        instruction = base_target.operation_from_name(name)
        prop_map = dict(base_target[str(name)])
        properties: dict[tuple[int, ...], InstructionProperties | None] = {}
        for qargs in list(base_target.qargs_for_operation_name(name)):
            qtuple = tuple(int(q) for q in qargs)
            if int(instruction.num_qubits) == 1:
                if int(qtuple[0]) not in allowed_qubits:
                    continue
            elif int(instruction.num_qubits) == 2:
                supported_twoq_edges.add(qtuple)
                if qtuple not in allowed_edges:
                    continue
            elif any(int(q) not in allowed_qubits for q in qtuple):
                continue
            properties[qtuple] = prop_map.get(qtuple, None)
        if properties:
            filtered.add_instruction(instruction, properties, name=str(name))
    return filtered, supported_twoq_edges



class _RestrictedBackendV2(BackendV2):
    def __init__(
        self,
        base_backend: Any,
        *,
        target: Target,
        coupling_map: CouplingMap | None,
    ):
        self._base_backend = base_backend
        self._target = target
        self._coupling_map = coupling_map
        self._dt = getattr(base_backend, "dt", None)
        super().__init__(
            name=str(getattr(base_backend, "name", getattr(base_backend, "backend_name", "restricted_backend"))),
            backend_version=str(getattr(base_backend, "backend_version", "restricted")),
        )

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> int | None:
        return getattr(self._base_backend, "max_circuits", None)

    @property
    def coupling_map(self) -> CouplingMap | None:
        return self._coupling_map

    @property
    def dt(self) -> float | None:
        if self._dt is None:
            return None
        return float(self._dt)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("_RestrictedBackendV2 is transpile/noise-model only and cannot execute circuits.")



def _restrict_backend_to_fixed_couplers(
    *,
    circuit: QuantumCircuit,
    resolved_spec: ResolvedNoiseSpec,
    resolved_backend: Any,
    initial_layout: list[int] | None,
) -> tuple[Any, dict[str, Any] | None]:
    requested_edges = _canonical_edge_tuples(resolved_spec.fixed_couplers)
    if not requested_edges:
        return resolved_backend, None
    if initial_layout is None:
        raise RuntimeError(
            "fixed_couplers requires a concrete initial layout, but the current path did not resolve one."
        )
    allowed_qubits = {int(q) for q in list(initial_layout)}
    outside_layout = sorted(
        [list(edge) for edge in requested_edges if int(edge[0]) not in allowed_qubits or int(edge[1]) not in allowed_qubits]
    )
    if outside_layout:
        raise RuntimeError(
            "fixed_couplers requires all declared edges to lie within the concrete initial layout "
            f"{list(initial_layout)} for this circuit; unsupported edges={outside_layout}. "
            "The current repo does not include a separate coupler/layout planner to choose a different sub-layout."
        )
    base_target = getattr(resolved_backend, "target", None)
    if base_target is None:
        raise RuntimeError(
            "fixed_couplers enforcement requires backend target metadata on the resolved backend."
        )
    filtered_target, supported_twoq_edges = _filter_target_to_subgraph(
        base_target,
        allowed_qubits=allowed_qubits,
        allowed_edges=requested_edges,
    )
    missing_edges = requested_edges.difference(supported_twoq_edges)
    if missing_edges:
        raise RuntimeError(
            "fixed_couplers requested edges that are unsupported by the resolved backend target: "
            f"{_canonical_edge_payload(missing_edges)}."
        )
    if _circuit_has_two_qubit_ops(circuit) and not any(
        len(list(qargs)) == 2 for _instruction, qargs in list(filtered_target.instructions)
    ):
        raise RuntimeError(
            "fixed_couplers removed all available two-qubit backend edges for a circuit that requires two-qubit operations."
        )
    filtered_coupling_map = (
        CouplingMap([[int(a), int(b)] for a, b in sorted(requested_edges)])
        if requested_edges
        else None
    )
    status = {
        "requested": _canonical_edge_payload(requested_edges),
        "enforced": True,
        "enforcement_scope": "transpile_target_subgraph",
        "compatibility_mode": "subset",
        "allowed_physical_qubits": [int(q) for q in list(initial_layout)],
        "verified_used_edges_subset": None,
        "verified_used_qubits_subset": None,
    }
    return (
        _RestrictedBackendV2(
            resolved_backend,
            target=filtered_target,
            coupling_map=filtered_coupling_map,
        ),
        status,
    )



def _verify_fixed_couplers_status(
    *,
    transpile_snapshot: TranspileSnapshot,
    status: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if status is None:
        return None
    allowed_edges = _canonical_edge_tuples(status.get("requested", []))
    allowed_qubits = {int(q) for q in list(status.get("allowed_physical_qubits", []))}
    used_edges = _canonical_edge_tuples(transpile_snapshot.used_physical_edges)
    used_qubits = {int(q) for q in list(transpile_snapshot.used_physical_qubits)}
    if not used_edges.issubset(allowed_edges):
        raise RuntimeError(
            "fixed_couplers verification failed: transpiled circuit used physical edges outside the declared set. "
            f"used={_canonical_edge_payload(used_edges)} requested={_canonical_edge_payload(allowed_edges)}."
        )
    if not used_qubits.issubset(allowed_qubits):
        raise RuntimeError(
            "fixed_couplers verification failed: transpiled circuit used physical qubits outside the constrained layout. "
            f"used={sorted(used_qubits)} allowed={sorted(allowed_qubits)}."
        )
    out = dict(status)
    out["verified_used_edges_subset"] = True
    out["verified_used_qubits_subset"] = True
    return out



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
    rec = _load_layout_lock_record(resolved_spec, calibration_snapshot)
    if rec is None:
        if str(resolved_spec.layout_policy) == "frozen_layout":
            raise RuntimeError(
                "layout_policy='frozen_layout' requires an existing persisted layout lock or replay artifact; "
                "no prior locked layout was found for this backend/snapshot/key."
            )
        return None
    final_layout = rec.get("final_layout", None)
    initial_layout = rec.get("initial_layout", None)
    chosen = final_layout if final_layout else initial_layout
    if chosen is None:
        if str(resolved_spec.layout_policy) == "frozen_layout":
            raise RuntimeError(
                "layout_policy='frozen_layout' found a persisted lock record, but it does not contain an initial/final layout to replay."
            )
        return None
    chosen_list = [int(x) for x in list(chosen)]
    if len(chosen_list) < int(logical_qubits):
        raise RuntimeError(
            f"Persisted frozen layout {chosen_list} is smaller than logical qubit count {int(logical_qubits)}."
        )
    return chosen_list[: int(logical_qubits)]



def _store_layout_lock(
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
    transpile_snapshot: TranspileSnapshot,
    layout_hash: str,
    *,
    extra_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    lock_key = _layout_lock_registry_key(resolved_spec, calibration_snapshot)
    payload = {
        "initial_layout": list(transpile_snapshot.initial_layout or []),
        "final_layout": list(transpile_snapshot.final_layout or []),
        "used_physical_qubits": list(transpile_snapshot.used_physical_qubits),
        "used_physical_edges": [list(edge) for edge in transpile_snapshot.used_physical_edges],
        "layout_hash": str(layout_hash),
    }
    if extra_payload:
        payload.update(jsonable_noise_value(dict(extra_payload)))
    _LAYOUT_LOCK_REGISTRY[lock_key] = dict(payload)
    _LAYOUT_LOCK_SOURCE_REGISTRY[lock_key] = "persisted_lock"
    _persist_layout_lock(lock_key, payload)
    if calibration_snapshot is not None and resolved_spec.snapshot_path not in {None, ""}:
        try:
            write_snapshot_layout_lock(
                str(resolved_spec.snapshot_path),
                calibration_snapshot,
                lock_key=str(lock_key),
                layout_lock=payload,
            )
        except Exception:
            pass
    return dict(payload)



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
            "layout_policy": _layout_lock_policy_scope(snapshot.layout_policy),
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
    store_layout_lock: bool = True,
) -> tuple[QuantumCircuit, TranspileSnapshot, str, str, dict[str, Any] | None]:
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
        return (
            cached["transpiled_circuit"].copy(),
            cached["transpile_snapshot"],
            cached["layout_hash"],
            circuit_structure_hash,
            None if cached.get("fixed_couplers_status", None) is None else dict(cached["fixed_couplers_status"]),
        )

    initial_layout = _resolve_initial_layout(resolved_spec, calibration_snapshot, int(circuit.num_qubits))
    resolved_backend_for_transpile, fixed_couplers_status = _restrict_backend_to_fixed_couplers(
        circuit=circuit,
        resolved_spec=resolved_spec,
        resolved_backend=resolved_backend,
        initial_layout=initial_layout,
    )
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
        backend=resolved_backend_for_transpile,
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
    fixed_couplers_status = _verify_fixed_couplers_status(
        transpile_snapshot=transpile_snapshot,
        status=fixed_couplers_status,
    )
    if bool(store_layout_lock):
        _store_layout_lock(resolved_spec, calibration_snapshot, transpile_snapshot, layout_hash)
    _TEMPLATE_CACHE[cache_key] = {
        "transpiled_circuit": transpiled.copy(),
        "transpile_snapshot": transpile_snapshot,
        "layout_hash": layout_hash,
        "fixed_couplers_status": (
            None if fixed_couplers_status is None else dict(fixed_couplers_status)
        ),
    }
    return transpiled, transpile_snapshot, layout_hash, circuit_structure_hash, fixed_couplers_status



def select_representative_patch(
    *,
    representative_circuits: Sequence[QuantumCircuit],
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot,
    optimization_level: int = 1,
) -> dict[str, Any]:
    """Choose and persist one fixed backend patch/layout from representative circuits."""
    from pipelines.exact_bench.noise_patch_selection import select_canonical_patch

    selection = select_canonical_patch(
        calibration_snapshot,
        representative_circuits,
        schedule_policy=str(resolved_spec.schedule_policy),
        seed_transpiler=resolved_spec.seed_transpiler,
        optimization_level=int(optimization_level),
    )
    payload = _store_layout_lock(
        resolved_spec,
        calibration_snapshot,
        selection.anchor_transpile_snapshot,
        selection.anchor_layout_hash,
        extra_payload=selection.to_layout_lock_payload(),
    )
    summary = dict(selection.to_summary())
    summary["layout_lock_key"] = _layout_lock_registry_key(resolved_spec, calibration_snapshot)
    summary["persisted_layout_lock"] = dict(payload)
    return summary



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
    store_layout_lock: bool = True,
    patch_selection_summary: Mapping[str, Any] | None = None,
) -> NoiseArtifact:
    if patch_selection_summary is None:
        lock_record = _load_layout_lock_record(resolved_spec, calibration_snapshot)
        if isinstance(lock_record, Mapping):
            patch_selection_summary = lock_record.get("patch_selection_summary", None)
    transpiled_circuit, transpile_snapshot, layout_hash, circuit_structure_hash, fixed_couplers_status = _transpile_and_cache(
        circuit=circuit,
        resolved_spec=resolved_spec,
        resolved_backend=qiskit_backend,
        calibration_snapshot=calibration_snapshot,
        optimization_level=int(optimization_level),
        store_layout_lock=bool(store_layout_lock),
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
        fixed_couplers_status=(None if fixed_couplers_status is None else dict(fixed_couplers_status)),
        patch_selection_summary=(None if patch_selection_summary is None else dict(jsonable_noise_value(dict(patch_selection_summary)))),
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
                "fixed_couplers_status": (
                    None if fixed_couplers_status is None else dict(fixed_couplers_status)
                ),
                "patch_selection_summary": (
                    None
                    if patch_selection_summary is None
                    else dict(jsonable_noise_value(dict(patch_selection_summary)))
                ),
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



def build_runtime_submission_artifact(
    *,
    circuit: QuantumCircuit,
    observable: SparsePauliOp | None,
    resolved_spec: ResolvedNoiseSpec,
    resolved_backend: Any,
    calibration_snapshot: CalibrationSnapshot | None,
) -> NoiseArtifact:
    return transpile_and_lock_patch(
        circuit,
        observable,
        resolved_spec,
        calibration_snapshot,
        qiskit_backend=resolved_backend,
        qiskit_noise_model=None,
        omitted_channels=[],
        warnings=[],
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



def _canonical_int_payload(raw: Any) -> list[int]:
    return [int(x) for x in list(raw or [])]



def _required_patch_replay_payload(
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
    *,
    logical_qubits: int,
) -> dict[str, Any]:
    if calibration_snapshot is None:
        raise RuntimeError("patch_snapshot requires a calibration snapshot.")
    rec = _load_layout_lock_record(resolved_spec, calibration_snapshot)
    if rec is None:
        raise RuntimeError(
            "patch_snapshot requires persisted patch/layout replay data, but no prior locked layout was found for this backend/snapshot/key."
        )
    chosen = rec.get("final_layout", None) or rec.get("initial_layout", None)
    if chosen is None:
        raise RuntimeError(
            "patch_snapshot requires persisted initial/final layout data; the stored layout lock is incomplete."
        )
    chosen_list = [int(x) for x in list(chosen)]
    if len(chosen_list) < int(logical_qubits):
        raise RuntimeError(
            f"Persisted patch replay layout {chosen_list} is smaller than logical qubit count {int(logical_qubits)}."
        )
    payload = dict(rec)
    payload["chosen_layout"] = chosen_list
    payload["selected_patch_qubits"] = _canonical_int_payload(rec.get("selected_patch_qubits", []))
    payload["selected_patch_edges"] = _canonical_edge_payload(rec.get("selected_patch_edges", []))
    return payload



def _assert_patch_replay_matches_expected(
    transpile_snapshot: TranspileSnapshot,
    *,
    expected_payload: Mapping[str, Any],
    actual_layout_hash: str | None = None,
) -> None:
    expected_initial = _canonical_int_payload(expected_payload.get("chosen_layout", []))
    actual_initial = _canonical_int_payload(transpile_snapshot.initial_layout or [])
    if expected_initial and not actual_initial:
        raise RuntimeError(
            "patch_snapshot replay did not expose an initial layout in the regenerated transpile metadata."
        )
    if expected_initial and actual_initial and actual_initial[: len(expected_initial)] != expected_initial:
        raise RuntimeError(
            "patch_snapshot replay changed the frozen initial layout; refusing to silently choose a different patch/layout."
        )
    expected_layout_hash = expected_payload.get("layout_hash", None)
    if expected_layout_hash not in {None, ""} and actual_layout_hash not in {None, str(expected_layout_hash)}:
        raise RuntimeError(
            "patch_snapshot replay changed the frozen layout hash; refusing to silently choose a different patch/layout."
        )



def build_patch_snapshot_artifact(
    *,
    circuit: QuantumCircuit,
    observable: SparsePauliOp | None,
    resolved_spec: ResolvedNoiseSpec,
    calibration_snapshot: CalibrationSnapshot | None,
    resolved_backend: Any,
    qiskit_noise_model: Any | None,
) -> NoiseArtifact:
    if calibration_snapshot is None:
        raise RuntimeError("patch_snapshot requires a calibration snapshot.")
    if str(resolved_spec.layout_policy) != "frozen_layout":
        raise RuntimeError("patch_snapshot requires layout_policy='frozen_layout'.")
    if resolved_backend is None:
        raise RuntimeError("patch_snapshot requires a backend-like target reconstructed from the calibration snapshot.")
    expected_payload = _required_patch_replay_payload(
        resolved_spec,
        calibration_snapshot,
        logical_qubits=int(circuit.num_qubits),
    )
    if qiskit_noise_model is None:
        qiskit_noise_model = _noise_model_from_backend(
            resolved_backend,
            f"patch:{str(resolved_spec.schedule_policy)}:{calibration_snapshot.snapshot_hash}",
        )
    omitted_channels = (
        [
            "crosstalk",
            "leakage",
            "non_markovian_drift",
            "coherent_overrotation",
        ]
        if str(resolved_spec.schedule_policy) == "asap"
        else [
            "delay_relaxation_if_unscheduled",
            "crosstalk",
            "leakage",
            "drift",
            "coherent_overrotation",
        ]
    )
    artifact = transpile_and_lock_patch(
        circuit,
        observable,
        resolved_spec,
        calibration_snapshot,
        qiskit_backend=resolved_backend,
        qiskit_noise_model=qiskit_noise_model,
        omitted_channels=omitted_channels,
        warnings=[
            "patch_snapshot replays frozen patch/layout from a persisted layout lock; per-circuit transpile metadata is regenerated and recorded during replay."
        ],
        store_layout_lock=False,
        patch_selection_summary=expected_payload.get("patch_selection_summary", None),
    )
    if artifact.transpile_snapshot is None:
        raise RuntimeError("patch_snapshot replay did not produce a transpile snapshot.")
    _assert_patch_replay_matches_expected(
        artifact.transpile_snapshot,
        expected_payload=expected_payload,
        actual_layout_hash=artifact.layout_hash,
    )
    return artifact
