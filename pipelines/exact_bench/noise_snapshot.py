#!/usr/bin/env python3
"""Backend calibration snapshot freeze/load helpers for exact-bench noise wrappers."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from pipelines.exact_bench.noise_model_spec import (
    CalibrationGateRecord,
    CalibrationQubitRecord,
    CalibrationSnapshot,
    calibration_snapshot_to_dict,
    canonical_calibration_snapshot_payload,
    jsonable_noise_value,
    stable_noise_hash,
)

_HOLSTEIN_BUNDLE_KEY = "holstein_bundle"
_LAYOUT_LOCKS_KEY = "layout_locks"
_BUNDLE_SCHEMA_VERSION = 1



def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")



def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out



def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None



def _median_or_none(values: list[float | None]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return float(median(cleaned))



def _backend_name(backend: Any) -> str | None:
    raw = getattr(backend, "name", None)
    if callable(raw):
        try:
            raw = raw()
        except Exception:
            raw = None
    override = getattr(backend, "_hh_noise_backend_name_override", None)
    if override not in {None, ""}:
        raw = override
    return _safe_str(raw)



def _basis_gates(backend: Any) -> list[str]:
    names = getattr(backend, "operation_names", None)
    if names is not None:
        try:
            return [str(x) for x in list(names)]
        except Exception:
            pass
    target = getattr(backend, "target", None)
    if target is not None:
        try:
            return [str(x) for x in list(target.operation_names)]
        except Exception:
            pass
    cfg = None
    try:
        cfg = backend.configuration()
    except Exception:
        cfg = None
    if cfg is not None and getattr(cfg, "basis_gates", None) is not None:
        return [str(x) for x in list(cfg.basis_gates)]
    return []



def _coupling_map(backend: Any) -> list[list[int]]:
    cmap = getattr(backend, "coupling_map", None)
    if cmap is None:
        return []
    if hasattr(cmap, "get_edges"):
        try:
            return [[int(a), int(b)] for a, b in list(cmap.get_edges())]
        except Exception:
            return []
    try:
        return [[int(edge[0]), int(edge[1])] for edge in list(cmap)]
    except Exception:
        return []



def _processor_family(backend: Any) -> str | None:
    for attr in ("processor_type", "processor_family"):
        raw = getattr(backend, attr, None)
        if raw is None:
            continue
        if isinstance(raw, dict):
            for key in ("family", "name"):
                if raw.get(key) not in {None, ""}:
                    return str(raw.get(key))
        if hasattr(raw, "family"):
            try:
                return _safe_str(getattr(raw, "family"))
            except Exception:
                pass
        if raw not in {None, ""}:
            return str(raw)
    return None



def _qubit_records(backend: Any, num_qubits: int) -> list[CalibrationQubitRecord]:
    props = None
    try:
        props = backend.properties()
    except Exception:
        props = None
    target = getattr(backend, "target", None)
    target_qubit_props = list(getattr(target, "qubit_properties", []) or []) if target is not None else []
    target_measure_props = {}
    if target is not None:
        try:
            target_measure_props = dict(target["measure"])
        except Exception:
            target_measure_props = {}
    out: list[CalibrationQubitRecord] = []
    for q in range(int(num_qubits)):
        qprops = {}
        if props is not None:
            try:
                qprops = dict(props.qubit_property(int(q)))
            except Exception:
                qprops = {}
        target_qprops = target_qubit_props[int(q)] if int(q) < len(target_qubit_props) else None
        measure_props = target_measure_props.get((int(q),), None)
        out.append(
            CalibrationQubitRecord(
                physical_qubit=int(q),
                T1_s=_safe_float(qprops.get("T1", getattr(target_qprops, "t1", None))),
                T2_s=_safe_float(qprops.get("T2", getattr(target_qprops, "t2", None))),
                readout_error=_safe_float(qprops.get("readout_error", getattr(measure_props, "error", None))),
                readout_p01=_safe_float(qprops.get("prob_meas0_prep1", getattr(measure_props, "prob_meas0_prep1", None))),
                readout_p10=_safe_float(qprops.get("prob_meas1_prep0", getattr(measure_props, "prob_meas1_prep0", None))),
                measure_duration_s=_safe_float(qprops.get("readout_length", getattr(measure_props, "duration", None))),
                frequency=_safe_float(qprops.get("frequency", getattr(target_qprops, "frequency", None))),
            )
        )
    return out



def _gate_records(backend: Any) -> list[CalibrationGateRecord]:
    props = None
    try:
        props = backend.properties()
    except Exception:
        props = None
    out: list[CalibrationGateRecord] = []
    if props is not None and getattr(props, "gates", None) is not None:
        for gate_rec in list(props.gates):
            name = str(getattr(gate_rec, "gate", getattr(gate_rec, "name", "unknown")))
            qubits = [int(q) for q in list(getattr(gate_rec, "qubits", []))]
            err = None
            dur = None
            try:
                err = _safe_float(props.gate_error(name, qubits if len(qubits) != 1 else int(qubits[0])))
            except Exception:
                err = None
            try:
                dur = _safe_float(props.gate_length(name, qubits if len(qubits) != 1 else int(qubits[0])))
            except Exception:
                dur = None
            out.append(
                CalibrationGateRecord(
                    gate_name=name,
                    qubits=qubits,
                    error=err,
                    duration_s=dur,
                )
            )
        if out:
            return out
    target = getattr(backend, "target", None)
    if target is None:
        return []
    for gate_name in list(getattr(target, "operation_names", []) or []):
        if str(gate_name) in {"measure", "delay", "reset", "barrier"}:
            continue
        try:
            properties_map = dict(target[str(gate_name)])
        except Exception:
            continue
        for qargs, props_rec in list(properties_map.items()):
            qubits = [int(q) for q in list(qargs or [])]
            out.append(
                CalibrationGateRecord(
                    gate_name=str(gate_name),
                    qubits=qubits,
                    error=_safe_float(getattr(props_rec, "error", None)),
                    duration_s=_safe_float(getattr(props_rec, "duration", None)),
                )
            )
    return out



def freeze_backend_snapshot(resolved_backend: Any) -> CalibrationSnapshot:
    backend = resolved_backend
    source_kind = _safe_str(getattr(backend, "_hh_noise_source_kind", None)) or (
        "fake_snapshot" if str(type(backend).__name__).startswith("Fake") else "live_backend"
    )
    backend_name = _backend_name(backend)
    backend_version = _safe_str(getattr(backend, "backend_version", None))
    calibration_time_utc = None
    try:
        props = backend.properties()
    except Exception:
        props = None
    if props is not None:
        calibration_time_utc = _safe_str(getattr(props, "last_update_date", None))
    num_qubits = int(getattr(backend, "num_qubits", 0) or 0)
    per_qubit = _qubit_records(backend, num_qubits)
    per_gate = _gate_records(backend)
    snapshot = CalibrationSnapshot(
        source_kind=str(source_kind),
        backend_name=backend_name,
        backend_version=backend_version,
        processor_family=_processor_family(backend),
        retrieved_at_utc=_now_utc(),
        calibration_time_utc=calibration_time_utc,
        basis_gates=[str(x) for x in _basis_gates(backend)],
        coupling_map=_coupling_map(backend),
        dt=_safe_float(getattr(backend, "dt", None)),
        per_qubit=per_qubit,
        per_gate=per_gate,
        median_1q_error=_median_or_none([
            rec.error for rec in per_gate if len(rec.qubits) == 1
        ]),
        median_2q_error=_median_or_none([
            rec.error for rec in per_gate if len(rec.qubits) == 2
        ]),
        median_readout_error=_median_or_none([rec.readout_error for rec in per_qubit]),
        median_T1=_median_or_none([rec.T1_s for rec in per_qubit]),
        median_T2=_median_or_none([rec.T2_s for rec in per_qubit]),
        snapshot_hash="",
    )
    return replace(
        snapshot,
        snapshot_hash=stable_noise_hash(
            canonical_calibration_snapshot_payload(snapshot),
            prefix="csnap",
        ),
    )



def _snapshot_from_dict(payload: dict[str, Any]) -> CalibrationSnapshot:
    per_qubit = [CalibrationQubitRecord(**dict(rec)) for rec in list(payload.get("per_qubit", []))]
    per_gate = [CalibrationGateRecord(**dict(rec)) for rec in list(payload.get("per_gate", []))]
    snapshot = CalibrationSnapshot(
        source_kind=str(payload.get("source_kind")),
        backend_name=payload.get("backend_name"),
        backend_version=payload.get("backend_version"),
        processor_family=payload.get("processor_family"),
        retrieved_at_utc=str(payload.get("retrieved_at_utc")),
        calibration_time_utc=payload.get("calibration_time_utc"),
        basis_gates=[str(x) for x in list(payload.get("basis_gates", []))],
        coupling_map=[[int(edge[0]), int(edge[1])] for edge in list(payload.get("coupling_map", []))],
        dt=_safe_float(payload.get("dt", None)),
        per_qubit=per_qubit,
        per_gate=per_gate,
        median_1q_error=_safe_float(payload.get("median_1q_error", None)),
        median_2q_error=_safe_float(payload.get("median_2q_error", None)),
        median_readout_error=_safe_float(payload.get("median_readout_error", None)),
        median_T1=_safe_float(payload.get("median_T1", None)),
        median_T2=_safe_float(payload.get("median_T2", None)),
        snapshot_hash=str(payload.get("snapshot_hash", "")),
    )
    return replace(
        snapshot,
        snapshot_hash=stable_noise_hash(
            canonical_calibration_snapshot_payload(snapshot),
            prefix="csnap",
        ),
    )



def _load_snapshot_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Frozen calibration snapshot JSON must contain a top-level object.")
    return dict(payload)



def _normalize_snapshot_bundle(raw: Any) -> dict[str, Any]:
    payload = dict(raw) if isinstance(raw, Mapping) else {}
    out: dict[str, Any] = {
        "schema_version": int(payload.get("schema_version", _BUNDLE_SCHEMA_VERSION)),
        _LAYOUT_LOCKS_KEY: {},
    }
    layout_locks = payload.get(_LAYOUT_LOCKS_KEY, {})
    if isinstance(layout_locks, Mapping):
        for key, value in sorted(layout_locks.items(), key=lambda item: str(item[0])):
            if isinstance(value, Mapping):
                out[_LAYOUT_LOCKS_KEY][str(key)] = jsonable_noise_value(dict(value))
    for key, value in sorted(payload.items(), key=lambda item: str(item[0])):
        if str(key) in {"schema_version", _LAYOUT_LOCKS_KEY}:
            continue
        out[str(key)] = jsonable_noise_value(value)
    return out



def load_snapshot_bundle(path: str | Path) -> dict[str, Any]:
    payload = _load_snapshot_payload(path)
    return _normalize_snapshot_bundle(payload.get(_HOLSTEIN_BUNDLE_KEY, {}))



def load_snapshot_layout_lock(path: str | Path, lock_key: str) -> dict[str, Any] | None:
    bundle = load_snapshot_bundle(path)
    record = bundle.get(_LAYOUT_LOCKS_KEY, {}).get(str(lock_key))
    if not isinstance(record, Mapping):
        return None
    return dict(record)



def write_snapshot_layout_lock(
    path: str | Path,
    snapshot: CalibrationSnapshot,
    *,
    lock_key: str,
    layout_lock: Mapping[str, Any],
) -> None:
    bundle = load_snapshot_bundle(path) if Path(path).exists() else _normalize_snapshot_bundle({})
    layout_locks = dict(bundle.get(_LAYOUT_LOCKS_KEY, {}))
    layout_locks[str(lock_key)] = jsonable_noise_value(dict(layout_lock))
    bundle[_LAYOUT_LOCKS_KEY] = layout_locks
    write_calibration_snapshot(path, snapshot, bundle=bundle)



def load_calibration_snapshot(path: str | Path) -> CalibrationSnapshot:
    payload = _load_snapshot_payload(path)
    return _snapshot_from_dict(dict(payload))



def write_calibration_snapshot(
    path: str | Path,
    snapshot: CalibrationSnapshot,
    *,
    bundle: Mapping[str, Any] | None = None,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preserved_bundle = (
        load_snapshot_bundle(out_path)
        if bundle is None and out_path.exists()
        else _normalize_snapshot_bundle(bundle)
    )
    payload = dict(calibration_snapshot_to_dict(snapshot))
    if preserved_bundle.get(_LAYOUT_LOCKS_KEY):
        payload[_HOLSTEIN_BUNDLE_KEY] = preserved_bundle
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
