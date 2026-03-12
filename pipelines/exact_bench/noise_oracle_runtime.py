#!/usr/bin/env python3
"""Noise/runtime expectation oracle utilities for HH/Hubbard validation.

This module stays in wrapper/benchmark space. It does not modify core operator
algebra modules and only adapts existing PauliPolynomial + ansatz objects to
Qiskit primitives at the boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter

from docs.reports.qiskit_circuit_report import (
    ansatz_to_circuit as _shared_ansatz_to_circuit,
    append_reference_state as _shared_append_reference_state,
    pauli_poly_to_sparse_pauli_op as _shared_pauli_poly_to_sparse_pauli_op,
)
from pipelines.exact_bench.noise_aer_builders import (
    build_backend_basic_artifact,
    build_backend_from_calibration_snapshot,
    build_backend_scheduled_artifact,
    build_patch_snapshot_artifact,
    build_runtime_submission_artifact,
    build_shots_only_artifact,
    describe_layout_anchor_source,
)
from pipelines.exact_bench.noise_model_spec import (
    NoiseArtifact,
    ResolvedNoiseSpec,
    calibration_snapshot_to_dict,
    noise_artifact_metadata,
    normalize_to_resolved_noise_spec,
    normalize_runtime_twirling_config,
    resolved_noise_spec_hash,
    resolved_noise_spec_to_dict,
    transpile_snapshot_to_dict,
)
from pipelines.exact_bench.noise_snapshot import (
    freeze_backend_snapshot,
    load_calibration_snapshot,
    write_calibration_snapshot,
)


@dataclass(frozen=True)
class OracleConfig:
    noise_mode: str = "ideal"  # ideal | shots | aer_noise | runtime | advanced internal values
    shots: int = 2048
    seed: int = 7
    oracle_repeats: int = 1
    oracle_aggregate: str = "mean"  # mean | median
    backend_name: str | None = None
    use_fake_backend: bool = False
    backend_profile: str | None = None
    aer_noise_kind: str = "scheduled"
    schedule_policy: str | None = None
    layout_policy: str | None = None
    noise_snapshot_json: str | None = None
    fixed_physical_patch: str | list[int] | tuple[int, ...] | None = None
    fixed_couplers: str | list[list[int]] | tuple[tuple[int, int], ...] | None = None
    layout_lock_key: str | None = None
    seed_transpiler: int | None = None
    seed_simulator: int | None = None
    approximation: bool = False
    abelian_grouping: bool = True
    allow_aer_fallback: bool = False
    allow_noisy_fallback: bool = False
    aer_fallback_mode: str = "sampler_shots"
    omp_shm_workaround: bool = True
    mitigation: dict[str, Any] | str = "none"
    runtime_twirling: dict[str, Any] | None = None
    symmetry_mitigation: dict[str, Any] | str = "off"


@dataclass(frozen=True)
class MitigationConfig:
    mode: str = "none"  # none | readout | zne | dd
    zne_scales: tuple[float, ...] = ()
    dd_sequence: str | None = None


@dataclass(frozen=True)
class OracleEstimate:
    mean: float
    std: float
    stdev: float
    stderr: float
    n_samples: int
    raw_values: list[float]
    aggregate: str


@dataclass(frozen=True)
class SymmetryMitigationConfig:
    mode: str = "off"  # off | verify_only | postselect_diag_v1 | projector_renorm_v1
    num_sites: int | None = None
    ordering: str = "blocked"
    sector_n_up: int | None = None
    sector_n_dn: int | None = None


class SymmetryMitigationDowngrade(RuntimeError):
    """Semantic symmetry-mitigation ineligibility that may downgrade to verify_only."""


_MITIGATION_MODES = {"none", "readout", "zne", "dd"}
_SYMMETRY_MITIGATION_MODES = {"off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"}


def _parse_zne_scales(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, str):
        tokens = [tok.strip() for tok in str(raw).split(",")]
        vals = [tok for tok in tokens if tok]
    elif isinstance(raw, Sequence):
        vals = [str(v).strip() for v in list(raw)]
        vals = [tok for tok in vals if tok]
    else:
        vals = [str(raw).strip()]
    out: list[float] = []
    for tok in vals:
        value = float(tok)
        if (not np.isfinite(value)) or (value <= 0.0):
            raise ValueError(f"Invalid mitigation zne scale {tok!r}; expected finite > 0.")
        out.append(float(value))
    return out


def normalize_mitigation_config(mitigation: Any) -> dict[str, Any]:
    mode = "none"
    zne_scales: list[float] = []
    dd_sequence: str | None = None
    layer_noise_model_json: str | None = None

    if mitigation is None:
        pass
    elif isinstance(mitigation, MitigationConfig):
        mode = str(mitigation.mode).strip().lower() or "none"
        zne_scales = _parse_zne_scales(list(mitigation.zne_scales))
        dd_sequence = None if mitigation.dd_sequence is None else str(mitigation.dd_sequence)
    elif isinstance(mitigation, str):
        mode = str(mitigation).strip().lower() or "none"
    elif isinstance(mitigation, Mapping):
        if mitigation.get("layer_noise_model", None) is not None and mitigation.get(
            "layer_noise_model_json",
            mitigation.get("layerNoiseModelJson", None),
        ) not in {None, "", "none"}:
            raise ValueError(
                "Specify only one of mitigation.layer_noise_model or mitigation.layer_noise_model_json."
            )
        mode = str(mitigation.get("mode", mitigation.get("mitigation", "none"))).strip().lower() or "none"
        zne_raw = mitigation.get("zne_scales", mitigation.get("zneScales", []))
        zne_scales = _parse_zne_scales(zne_raw)
        dd_raw = mitigation.get("dd_sequence", mitigation.get("ddSequence", None))
        dd_sequence = None if dd_raw is None else str(dd_raw)
        layer_noise_model_json_raw = mitigation.get(
            "layer_noise_model_json",
            mitigation.get("layerNoiseModelJson", None),
        )
        layer_noise_model_json = (
            None
            if layer_noise_model_json_raw in {None, "", "none"}
            else os.fspath(layer_noise_model_json_raw)
        )
    else:
        raise ValueError(
            "Unsupported mitigation config type; expected str, dict, MitigationConfig, or None."
        )

    if mode not in _MITIGATION_MODES:
        raise ValueError(
            f"Unsupported mitigation mode {mode!r}; expected one of {sorted(_MITIGATION_MODES)}."
        )
    if mode != "zne" and zne_scales:
        raise ValueError("zne_scales require mitigation mode 'zne'.")
    if mode == "zne" and not zne_scales:
        raise ValueError("mitigation mode 'zne' requires non-empty zne_scales.")
    if mode != "dd" and dd_sequence not in {None, ""}:
        raise ValueError("dd_sequence requires mitigation mode 'dd'.")
    if mode == "dd" and dd_sequence in {None, ""}:
        raise ValueError("mitigation mode 'dd' requires dd_sequence.")

    out = {
        "mode": str(mode),
        "zne_scales": [float(x) for x in zne_scales],
        "dd_sequence": dd_sequence,
    }
    if layer_noise_model_json is not None:
        out["layer_noise_model_json"] = str(layer_noise_model_json)
    return out


def _coerce_runtime_layer_noise_model(
    raw_model: Any,
    *,
    source: str,
    fingerprint: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    try:
        from qiskit_ibm_runtime.utils.noise_learner_result import LayerError, NoiseLearnerResult
    except Exception as exc:
        raise RuntimeError(
            "External layer_noise_model support requires qiskit-ibm-runtime noise learner types in this environment."
        ) from exc

    if isinstance(raw_model, NoiseLearnerResult):
        entries = list(getattr(raw_model, "data", []) or [])
        if not entries:
            raise ValueError(
                "mitigation layer_noise_model must be a non-empty NoiseLearnerResult or Sequence[LayerError]."
            )
        return raw_model, {
            "supplied": True,
            "kind": "NoiseLearnerResult",
            "source": str(source),
            "entry_count": int(len(entries)),
            "fingerprint": fingerprint,
        }

    if isinstance(raw_model, Sequence) and not isinstance(
        raw_model, (str, bytes, bytearray, Mapping)
    ):
        entries = list(raw_model)
        if not entries:
            raise ValueError(
                "mitigation layer_noise_model must be a non-empty NoiseLearnerResult or Sequence[LayerError]."
            )
        bad_entry = next((entry for entry in entries if not isinstance(entry, LayerError)), None)
        if bad_entry is not None:
            raise ValueError(
                "mitigation layer_noise_model must be a NoiseLearnerResult or Sequence[LayerError]. "
                f"Unsupported entry type {type(bad_entry)!r}."
            )
        return list(entries), {
            "supplied": True,
            "kind": "Sequence[LayerError]",
            "source": str(source),
            "entry_count": int(len(entries)),
            "fingerprint": fingerprint,
        }

    raise ValueError(
        "mitigation layer_noise_model must be a NoiseLearnerResult or Sequence[LayerError]."
    )


def _load_runtime_layer_noise_model_json(raw_path: Any) -> tuple[Any, dict[str, Any]]:
    try:
        path_str = os.fspath(raw_path)
    except TypeError as exc:
        raise ValueError(
            "mitigation layer_noise_model_json must be a filesystem path to a RuntimeEncoder JSON artifact."
        ) from exc
    if path_str in {"", "none"}:
        raise ValueError(
            "mitigation layer_noise_model_json must be a filesystem path to a RuntimeEncoder JSON artifact."
        )
    try:
        payload_bytes = open(path_str, "rb").read()
    except OSError as exc:
        raise ValueError(f"Failed to read mitigation layer_noise_model_json file {path_str!r}.") from exc
    try:
        from qiskit_ibm_runtime.utils.json import RuntimeDecoder

        decoded = json.loads(payload_bytes.decode("utf-8"), cls=RuntimeDecoder)
    except Exception as exc:
        raise ValueError(
            "Failed to decode mitigation layer_noise_model_json; expected JSON encoded with qiskit_ibm_runtime.utils.json.RuntimeEncoder containing a NoiseLearnerResult or Sequence[LayerError]."
        ) from exc
    return _coerce_runtime_layer_noise_model(
        decoded,
        source="file_backed_json",
        fingerprint=f"sha256:{hashlib.sha256(payload_bytes).hexdigest()}",
    )


def _extract_runtime_layer_noise_model(mitigation: Any) -> tuple[Any | None, dict[str, Any]]:
    raw_model = None
    raw_model_json = None
    if isinstance(mitigation, Mapping):
        raw_model = mitigation.get("layer_noise_model", None)
        raw_model_json = mitigation.get(
            "layer_noise_model_json",
            mitigation.get("layerNoiseModelJson", None),
        )
    if raw_model is not None and raw_model_json not in {None, "", "none"}:
        raise ValueError(
            "Specify only one of mitigation.layer_noise_model or mitigation.layer_noise_model_json."
        )
    if raw_model_json not in {None, "", "none"}:
        return _load_runtime_layer_noise_model_json(raw_model_json)
    if raw_model is None:
        return None, {
            "supplied": False,
            "kind": None,
            "source": None,
            "entry_count": None,
            "fingerprint": None,
        }
    return _coerce_runtime_layer_noise_model(
        raw_model,
        source="programmatic_object",
        fingerprint=None,
    )


def normalize_symmetry_mitigation_config(symmetry_mitigation: Any) -> dict[str, Any]:
    mode = "off"
    num_sites: int | None = None
    ordering = "blocked"
    sector_n_up: int | None = None
    sector_n_dn: int | None = None

    if symmetry_mitigation is None:
        pass
    elif isinstance(symmetry_mitigation, SymmetryMitigationConfig):
        mode = str(symmetry_mitigation.mode).strip().lower() or "off"
        num_sites = (
            None if symmetry_mitigation.num_sites is None else int(symmetry_mitigation.num_sites)
        )
        ordering = str(symmetry_mitigation.ordering).strip().lower() or "blocked"
        sector_n_up = (
            None if symmetry_mitigation.sector_n_up is None else int(symmetry_mitigation.sector_n_up)
        )
        sector_n_dn = (
            None if symmetry_mitigation.sector_n_dn is None else int(symmetry_mitigation.sector_n_dn)
        )
    elif isinstance(symmetry_mitigation, str):
        mode = str(symmetry_mitigation).strip().lower() or "off"
    elif isinstance(symmetry_mitigation, Mapping):
        mode = str(
            symmetry_mitigation.get("mode", symmetry_mitigation.get("symmetry_mitigation", "off"))
        ).strip().lower() or "off"
        num_sites_raw = symmetry_mitigation.get("num_sites", symmetry_mitigation.get("L", None))
        ordering_raw = symmetry_mitigation.get("ordering", "blocked")
        n_up_raw = symmetry_mitigation.get("sector_n_up", symmetry_mitigation.get("n_up", None))
        n_dn_raw = symmetry_mitigation.get("sector_n_dn", symmetry_mitigation.get("n_dn", None))
        num_sites = None if num_sites_raw is None else int(num_sites_raw)
        ordering = str(ordering_raw).strip().lower() or "blocked"
        sector_n_up = None if n_up_raw is None else int(n_up_raw)
        sector_n_dn = None if n_dn_raw is None else int(n_dn_raw)
    else:
        raise ValueError(
            "Unsupported symmetry mitigation config type; expected str, dict, SymmetryMitigationConfig, or None."
        )

    if mode not in _SYMMETRY_MITIGATION_MODES:
        raise ValueError(
            f"Unsupported symmetry mitigation mode {mode!r}; expected one of {sorted(_SYMMETRY_MITIGATION_MODES)}."
        )
    if ordering not in {"blocked", "interleaved"}:
        raise ValueError("symmetry mitigation ordering must be 'blocked' or 'interleaved'.")
    if mode in {"postselect_diag_v1", "projector_renorm_v1"}:
        missing = [
            name
            for name, value in (
                ("num_sites", num_sites),
                ("sector_n_up", sector_n_up),
                ("sector_n_dn", sector_n_dn),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "symmetry mitigation mode {!r} requires {}.".format(mode, ", ".join(missing))
            )

    return {
        "mode": str(mode),
        "num_sites": (None if num_sites is None else int(num_sites)),
        "ordering": str(ordering),
        "sector_n_up": (None if sector_n_up is None else int(sector_n_up)),
        "sector_n_dn": (None if sector_n_dn is None else int(sector_n_dn)),
    }


def normalize_ideal_reference_symmetry_mitigation(
    symmetry_mitigation: Any,
    *,
    noise_mode: str,
) -> dict[str, Any]:
    cfg = normalize_symmetry_mitigation_config(symmetry_mitigation)
    runtime_like_modes = {"runtime", "qpu_raw", "qpu_suppressed", "qpu_layer_learned"}
    if str(noise_mode).strip().lower() in runtime_like_modes and str(cfg.get("mode", "off")) not in {"off", "verify_only"}:
        downgraded = dict(cfg)
        downgraded["mode"] = "verify_only"
        return downgraded
    return cfg


def _ensure_mitigation_execution_supported(
    resolved_spec: ResolvedNoiseSpec,
    mitigation_cfg: Mapping[str, Any],
) -> None:
    mitigation_mode = str(mitigation_cfg.get("mode", "none"))
    if mitigation_mode == "none":
        return
    if str(resolved_spec.executor) == "runtime_qpu":
        return
    raise RuntimeError(
        f"mitigation mode {mitigation_mode!r} is only executable on runtime_qpu in the current repo; "
        f"executor={resolved_spec.executor!r} noise_kind={resolved_spec.noise_kind!r} does not apply readout/zne/dd. "
        "Use mitigation='none' for statevector/local Aer, or use symmetry_mitigation for the existing diagonal postselection path."
    )


def _runtime_twirling_enabled(runtime_twirling_cfg: Mapping[str, Any]) -> bool:
    return bool(
        runtime_twirling_cfg.get("enable_gates", False)
        or runtime_twirling_cfg.get("enable_measure", False)
    )


def _ensure_runtime_twirling_execution_supported(
    resolved_spec: ResolvedNoiseSpec,
    runtime_twirling_cfg: Mapping[str, Any],
) -> None:
    if not _runtime_twirling_enabled(runtime_twirling_cfg):
        return
    if str(resolved_spec.executor) == "runtime_qpu":
        return
    raise RuntimeError(
        "Runtime twirling is only executable on runtime_qpu in the current repo; "
        f"executor={resolved_spec.executor!r} noise_kind={resolved_spec.noise_kind!r} does not apply runtime twirling on local Aer/statevector paths."
    )


def _ensure_runtime_twirling_compatibility(
    mitigation_cfg: Mapping[str, Any],
    runtime_twirling_cfg: Mapping[str, Any],
) -> None:
    if not _runtime_twirling_enabled(runtime_twirling_cfg):
        return
    mitigation_mode = str(mitigation_cfg.get("mode", "none"))
    if bool(runtime_twirling_cfg.get("enable_measure", False)) and mitigation_mode != "readout":
        raise RuntimeError(
            "Measurement twirling is only wired with mitigation='readout' in the current repo so the Runtime bundle remains TREX-like and truthfully distinct from readout-only."
        )


def _set_runtime_option(
    options: Any,
    path: str,
    value: Any,
    *,
    mitigation_mode: str,
) -> None:
    target = options
    parts = [str(part) for part in str(path).split(".") if str(part)]
    if not parts:
        raise RuntimeError("Internal error: empty runtime option path.")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise RuntimeError(
                f"Runtime Estimator options in this environment do not expose '{path}' required for mitigation mode {mitigation_mode!r}."
            )
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise RuntimeError(
            f"Runtime Estimator options in this environment do not expose '{path}' required for mitigation mode {mitigation_mode!r}."
        )
    setattr(target, leaf, value)



def _ensure_runtime_mode_request_supported(
    cfg: OracleConfig,
    resolved_spec: ResolvedNoiseSpec,
    mitigation_cfg: Mapping[str, Any],
) -> None:
    if str(resolved_spec.executor) != "runtime_qpu":
        return
    requested_mode = str(getattr(cfg, "noise_mode", "runtime")).strip().lower()
    mitigation_mode = str(mitigation_cfg.get("mode", "none"))
    mitigation_bundle = str(resolved_spec.mitigation_bundle)
    if requested_mode == "runtime":
        return
    if requested_mode == "qpu_raw" and mitigation_mode == "none":
        return
    if requested_mode == "qpu_suppressed" and mitigation_bundle == "runtime_suppressed":
        return
    if requested_mode == "qpu_layer_learned" and mitigation_bundle == "runtime_layer_learned":
        return
    if requested_mode == "qpu_suppressed":
        raise RuntimeError(
            "noise_mode='qpu_suppressed' requires a supported runtime suppression bundle in the current repo. "
            "Use mitigation {readout,zne,dd} or runtime twirling so the mode resolves to mitigation_bundle='runtime_suppressed', "
            "or use noise_mode='qpu_raw' / noise_mode='runtime' for the existing explicit paths."
        )
    if requested_mode == "qpu_layer_learned":
        raise RuntimeError(
            "noise_mode='qpu_layer_learned' requires the explicit runtime layer-noise-backed bundle in the current repo. "
            "Use mitigation='zne' without explicit runtime twirling so the mode maps to PEA-backed ZNE with Runtime-managed or externally supplied layer-noise data."
        )
    raise RuntimeError(
        f"Advanced runtime noise_mode {requested_mode!r} is not wired to explicit Runtime options in the current repo. "
        "Use noise_mode='runtime' with mitigation {none,readout,zne,dd} for the supported path, or extend this owner with an explicit mapping first."
    )



def _runtime_execution_bundle_details(
    cfg: OracleConfig,
    resolved_spec: ResolvedNoiseSpec,
    mitigation_cfg: Mapping[str, Any],
    runtime_twirling_cfg: Mapping[str, Any],
    layer_noise_model_meta: Mapping[str, Any],
) -> dict[str, Any]:
    suppression_components: list[str] = []
    mitigation_mode = str(mitigation_cfg.get("mode", "none"))
    layer_learned = str(resolved_spec.noise_kind) == "qpu_layer_learned"
    zne_amplifier = "pea" if layer_learned and mitigation_mode == "zne" else None
    layer_noise_model_supplied = bool(layer_noise_model_meta.get("supplied", False))
    layer_noise_learning_requested = bool(
        layer_learned and mitigation_mode == "zne" and not layer_noise_model_supplied
    )
    layer_noise_model_source = (
        layer_noise_model_meta.get("source", None)
        if layer_noise_model_supplied
        else ("runtime_service_learning" if layer_noise_learning_requested else None)
    )
    if mitigation_mode == "readout":
        suppression_components.append("measure_mitigation")
    elif mitigation_mode == "zne":
        suppression_components.append("zne")
    elif mitigation_mode == "dd":
        suppression_components.append("dynamical_decoupling")
    if zne_amplifier == "pea":
        suppression_components.append("pea_amplifier")
        if layer_noise_learning_requested:
            suppression_components.append("layer_noise_learning")
        if layer_noise_model_supplied:
            suppression_components.append("external_layer_noise_model")
        suppression_components.append("implicit_gate_twirling_via_pea")
    if bool(runtime_twirling_cfg.get("enable_gates", False)):
        suppression_components.append("gate_twirling")
    if bool(runtime_twirling_cfg.get("enable_measure", False)):
        suppression_components.append("measure_twirling")
        if mitigation_mode == "readout":
            suppression_components.append("trex_like_readout")
    return {
        "requested_noise_mode": str(getattr(cfg, "noise_mode", "runtime")).strip().lower(),
        "resolved_noise_kind": str(resolved_spec.noise_kind),
        "mitigation_bundle": str(resolved_spec.mitigation_bundle),
        "mitigation_mode": mitigation_mode,
        "twirling_enable_gates": bool(runtime_twirling_cfg.get("enable_gates", False)),
        "twirling_enable_measure": bool(runtime_twirling_cfg.get("enable_measure", False)),
        "twirling_num_randomizations": runtime_twirling_cfg.get("num_randomizations", None),
        "twirling_strategy": runtime_twirling_cfg.get("strategy", None),
        "trex_like_measure_suppression": bool(
            runtime_twirling_cfg.get("enable_measure", False) and mitigation_mode == "readout"
        ),
        "zne_amplifier": zne_amplifier,
        "layer_noise_learning_requested": layer_noise_learning_requested,
        "layer_noise_model_supplied": layer_noise_model_supplied,
        "layer_noise_model_source": layer_noise_model_source,
        "layer_noise_model_kind": layer_noise_model_meta.get("kind", None),
        "layer_noise_model_entry_count": layer_noise_model_meta.get("entry_count", None),
        "layer_noise_model_fingerprint": layer_noise_model_meta.get("fingerprint", None),
        "implicit_gate_twirling_via_pea": bool(zne_amplifier == "pea"),
        "suppression_components": suppression_components,
    }



def _derive_provenance_summary(details: Mapping[str, Any]) -> dict[str, Any]:
    resolved_spec = dict(details.get("resolved_noise_spec", {}) or {})
    executor = str(resolved_spec.get("executor", "")).strip().lower()
    noise_kind = str(resolved_spec.get("noise_kind", "")).strip().lower()
    backend_profile_kind = str(resolved_spec.get("backend_profile_kind", "")).strip().lower()
    runtime_bundle = dict(details.get("runtime_execution_bundle", {}) or {})
    fixed_couplers_status = dict(details.get("fixed_couplers_status", {}) or {})
    layout_anchor_source = details.get("layout_anchor_source", None)
    snapshot_hash = details.get("snapshot_hash", None)
    layout_hash = details.get("layout_hash", None)
    transpile_hash = details.get("transpile_hash", None)
    circuit_structure_hash = details.get("circuit_structure_hash", None)
    runtime_submission_recorded = bool(
        executor == "runtime_qpu" and layout_hash and transpile_hash and circuit_structure_hash
    )
    runtime_layer_learned = bool(
        runtime_bundle
        and (
            str(runtime_bundle.get("mitigation_bundle", "")).strip().lower()
            == "runtime_layer_learned"
            or str(runtime_bundle.get("resolved_noise_kind", "")).strip().lower()
            == "qpu_layer_learned"
            or bool(runtime_bundle.get("layer_noise_learning_requested", False))
        )
    )
    runtime_suppressed = bool(
        runtime_bundle
        and (
            str(runtime_bundle.get("mitigation_bundle", "")).strip().lower() == "runtime_suppressed"
            or str(runtime_bundle.get("resolved_noise_kind", "")).strip().lower() == "qpu_suppressed"
        )
    )
    if executor == "runtime_qpu":
        if runtime_layer_learned:
            classification = "runtime_submitted_layer_learned"
        elif runtime_suppressed:
            classification = "runtime_submitted_suppressed"
        elif runtime_bundle or runtime_submission_recorded:
            classification = "runtime_submitted_raw"
        else:
            classification = "runtime_submission_unresolved"
    elif executor == "aer":
        if noise_kind == "patch_snapshot":
            classification = "local_patch_frozen_replay"
        elif backend_profile_kind == "frozen_snapshot_json" and noise_kind in {"backend_basic", "backend_scheduled"}:
            classification = "local_snapshot_replay"
        else:
            classification = "local_generic_aer_execution"
    elif executor == "statevector" or noise_kind == "none":
        classification = "ideal_statevector_control"
    else:
        classification = "unresolved_execution_provenance"
    return {
        "classification": str(classification),
        "uses_backend_snapshot": bool(snapshot_hash),
        "uses_frozen_snapshot_json": bool(backend_profile_kind == "frozen_snapshot_json"),
        "local_replay": bool(classification in {"local_snapshot_replay", "local_patch_frozen_replay"}),
        "patch_frozen_replay": bool(classification == "local_patch_frozen_replay"),
        "runtime_submission_recorded": bool(runtime_submission_recorded),
        "runtime_suppressed": bool(runtime_suppressed),
        "runtime_layer_noise_learning": bool(
            runtime_bundle.get("layer_noise_learning_requested", False)
        ),
        "runtime_layer_noise_model_supplied": bool(
            runtime_bundle.get("layer_noise_model_supplied", False)
        ),
        "runtime_layer_noise_model_source": runtime_bundle.get(
            "layer_noise_model_source", None
        ),
        "runtime_layer_noise_model_kind": runtime_bundle.get("layer_noise_model_kind", None),
        "runtime_layer_noise_model_fingerprint": runtime_bundle.get(
            "layer_noise_model_fingerprint", None
        ),
        "runtime_zne_amplifier": runtime_bundle.get("zne_amplifier", None),
        "runtime_gate_twirling": bool(runtime_bundle.get("twirling_enable_gates", False)),
        "runtime_measure_twirling": bool(runtime_bundle.get("twirling_enable_measure", False)),
        "runtime_trex_like_measure_suppression": bool(
            runtime_bundle.get("trex_like_measure_suppression", False)
        ),
        "layout_anchor_source": layout_anchor_source,
        "layout_anchor_reused": bool(layout_anchor_source in {"persisted_lock", "snapshot_bundle_lock", "fixed_patch"}),
        "runtime_execution_bundle": (None if not runtime_bundle else dict(runtime_bundle)),
        "fixed_couplers_requested": bool(resolved_spec.get("fixed_couplers", None)),
        "fixed_couplers_verified": bool(
            fixed_couplers_status.get("verified_used_edges_subset", False)
            and fixed_couplers_status.get("verified_used_qubits_subset", False)
        ),
    }



def _with_provenance_summary(details: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(details)
    out["provenance_summary"] = _derive_provenance_summary(out)
    return out



def _build_runtime_mitigation_options(
    resolved_spec: ResolvedNoiseSpec,
    mitigation_cfg: Mapping[str, Any],
    runtime_twirling_cfg: Mapping[str, Any],
    layer_noise_model: Any | None,
    layer_noise_model_meta: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from qiskit_ibm_runtime.options import EstimatorOptions

    mitigation_mode = str(mitigation_cfg.get("mode", "none"))
    layer_learned = str(resolved_spec.noise_kind) == "qpu_layer_learned"
    options = EstimatorOptions()
    applied = {
        "mode": mitigation_mode,
        "resilience_level": None,
        "measure_mitigation": False,
        "zne_mitigation": False,
        "zne_noise_factors": [],
        "zne_amplifier": None,
        "dynamical_decoupling_enable": False,
        "dynamical_decoupling_sequence_type": None,
        "layer_noise_learning_requested": False,
        "layer_noise_model_supplied": bool(layer_noise_model_meta.get("supplied", False)),
        "layer_noise_model_source": None,
        "layer_noise_model_kind": layer_noise_model_meta.get("kind", None),
        "layer_noise_model_entry_count": layer_noise_model_meta.get("entry_count", None),
        "layer_noise_model_fingerprint": layer_noise_model_meta.get("fingerprint", None),
        "implicit_gate_twirling_via_pea": False,
        "twirling_enable_gates": bool(runtime_twirling_cfg.get("enable_gates", False)),
        "twirling_enable_measure": bool(runtime_twirling_cfg.get("enable_measure", False)),
        "twirling_num_randomizations": runtime_twirling_cfg.get("num_randomizations", None),
        "twirling_strategy": runtime_twirling_cfg.get("strategy", None),
        "trex_like_measure_suppression": bool(
            runtime_twirling_cfg.get("enable_measure", False) and mitigation_mode == "readout"
        ),
    }

    _set_runtime_option(
        options,
        "twirling.enable_gates",
        bool(runtime_twirling_cfg.get("enable_gates", False)),
        mitigation_mode=mitigation_mode,
    )
    _set_runtime_option(
        options,
        "twirling.enable_measure",
        bool(runtime_twirling_cfg.get("enable_measure", False)),
        mitigation_mode=mitigation_mode,
    )
    if runtime_twirling_cfg.get("num_randomizations", None) is not None:
        _set_runtime_option(
            options,
            "twirling.num_randomizations",
            int(runtime_twirling_cfg.get("num_randomizations")),
            mitigation_mode=mitigation_mode,
        )
    if runtime_twirling_cfg.get("strategy", None) is not None:
        _set_runtime_option(
            options,
            "twirling.strategy",
            str(runtime_twirling_cfg.get("strategy")),
            mitigation_mode=mitigation_mode,
        )

    if mitigation_mode == "none":
        _set_runtime_option(options, "resilience_level", 0, mitigation_mode=mitigation_mode)
        _set_runtime_option(options, "resilience.measure_mitigation", False, mitigation_mode=mitigation_mode)
        _set_runtime_option(options, "resilience.zne_mitigation", False, mitigation_mode=mitigation_mode)
        _set_runtime_option(options, "dynamical_decoupling.enable", False, mitigation_mode=mitigation_mode)
        applied["resilience_level"] = 0
    elif mitigation_mode == "readout":
        _set_runtime_option(options, "resilience.measure_mitigation", True, mitigation_mode=mitigation_mode)
        _set_runtime_option(options, "resilience.zne_mitigation", False, mitigation_mode=mitigation_mode)
        _set_runtime_option(options, "dynamical_decoupling.enable", False, mitigation_mode=mitigation_mode)
        applied["measure_mitigation"] = True
    elif mitigation_mode == "zne":
        _set_runtime_option(options, "resilience.measure_mitigation", False, mitigation_mode=mitigation_mode)
        _set_runtime_option(options, "resilience.zne_mitigation", True, mitigation_mode=mitigation_mode)
        _set_runtime_option(options, "resilience.zne.noise_factors", [float(x) for x in mitigation_cfg.get("zne_scales", [])], mitigation_mode=mitigation_mode)
        _set_runtime_option(options, "dynamical_decoupling.enable", False, mitigation_mode=mitigation_mode)
        applied["zne_mitigation"] = True
        applied["zne_noise_factors"] = [float(x) for x in mitigation_cfg.get("zne_scales", [])]
        if layer_learned:
            _set_runtime_option(
                options,
                "resilience.zne.amplifier",
                "pea",
                mitigation_mode=mitigation_mode,
            )
            applied["zne_amplifier"] = "pea"
            applied["implicit_gate_twirling_via_pea"] = True
            if bool(layer_noise_model_meta.get("supplied", False)):
                _set_runtime_option(
                    options,
                    "resilience.layer_noise_model",
                    layer_noise_model,
                    mitigation_mode=mitigation_mode,
                )
                applied["layer_noise_learning_requested"] = False
                applied["layer_noise_model_supplied"] = True
                applied["layer_noise_model_source"] = layer_noise_model_meta.get(
                    "source", None
                )
            else:
                _set_runtime_option(
                    options,
                    "resilience.layer_noise_learning",
                    {},
                    mitigation_mode=mitigation_mode,
                )
                applied["layer_noise_learning_requested"] = True
                applied["layer_noise_model_supplied"] = False
                applied["layer_noise_model_source"] = "runtime_service_learning"
    elif mitigation_mode == "dd":
        _set_runtime_option(options, "resilience.measure_mitigation", False, mitigation_mode=mitigation_mode)
        _set_runtime_option(options, "resilience.zne_mitigation", False, mitigation_mode=mitigation_mode)
        _set_runtime_option(options, "dynamical_decoupling.enable", True, mitigation_mode=mitigation_mode)
        _set_runtime_option(options, "dynamical_decoupling.sequence_type", str(mitigation_cfg.get("dd_sequence")), mitigation_mode=mitigation_mode)
        applied["dynamical_decoupling_enable"] = True
        applied["dynamical_decoupling_sequence_type"] = str(mitigation_cfg.get("dd_sequence"))
    else:
        raise RuntimeError(f"Unsupported mitigation mode {mitigation_mode!r} for runtime option wiring.")

    return options, applied


@dataclass(frozen=True)
class NoiseBackendInfo:
    noise_mode: str
    estimator_kind: str
    backend_name: str | None = None
    using_fake_backend: bool = False
    details: dict[str, Any] = field(default_factory=dict)


_SYMMETRY_RESULT_KEYS = {
    "requested_mode",
    "applied_mode",
    "executed",
    "eligible",
    "fallback_reason",
    "retained_fraction_mean",
    "retained_fraction_samples",
    "sector_probability_mean",
    "sector_probability_samples",
    "sector_values",
    "estimator_form",
}


@dataclass(frozen=True)
class OracleExecutionRecord:
    noise_mode: str
    estimator_kind: str
    backend_name: str | None = None
    using_fake_backend: bool = False
    shots: int | None = None
    mitigation: dict[str, Any] = field(default_factory=dict)
    runtime_twirling: dict[str, Any] = field(default_factory=dict)
    symmetry_mitigation_config: dict[str, Any] = field(default_factory=dict)
    symmetry_mitigation_result: dict[str, Any] | None = None
    resolved_noise_spec: dict[str, Any] = field(default_factory=dict)
    resolved_noise_spec_hash: str | None = None
    calibration_snapshot: dict[str, Any] | None = None
    transpile_snapshot: dict[str, Any] | None = None
    noise_artifact_hash: str | None = None
    snapshot_hash: str | None = None
    layout_hash: str | None = None
    transpile_hash: str | None = None
    circuit_structure_hash: str | None = None
    layout_anchor_source: str | None = None
    runtime_mitigation_options: dict[str, Any] | None = None
    runtime_execution_bundle: dict[str, Any] | None = None
    fixed_couplers_status: dict[str, Any] | None = None
    patch_selection_summary: dict[str, Any] | None = None
    source_kind: str | None = None
    used_physical_qubits: list[int] = field(default_factory=list)
    used_physical_edges: list[list[int]] = field(default_factory=list)
    scheduled_duration_total: int | float | None = None
    idle_duration_total: int | float | None = None
    seed_transpiler: int | None = None
    seed_simulator: int | None = None
    warnings: list[str] = field(default_factory=list)
    omitted_channels: list[str] = field(default_factory=list)
    noise_model_basis_gates: list[str] = field(default_factory=list)
    aer_failed: bool = False
    fallback_used: bool = False
    fallback_mode: str | None = None
    fallback_reason: str = ""
    env_workaround_applied: bool = False
    provenance_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def symmetry_mitigation(self) -> dict[str, Any]:
        return dict(self.symmetry_mitigation_result or self.symmetry_mitigation_config)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_legacy_details(self) -> dict[str, Any]:
        return _execution_record_to_backend_details(self)


@dataclass(frozen=True)
class OracleEvaluationResult:
    estimate: OracleEstimate
    execution: OracleExecutionRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": asdict(self.estimate),
            "execution": self.execution.to_dict(),
        }


def _mapping_copy(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return deepcopy(dict(value))


def _maybe_mapping_copy(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return _mapping_copy(value)


def _list_copy(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if isinstance(value, Sequence):
        return deepcopy(list(value))
    return [value]


def _looks_like_symmetry_mitigation_result(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(set(value.keys()) & _SYMMETRY_RESULT_KEYS)


def _copy_oracle_estimate(estimate: OracleEstimate) -> OracleEstimate:
    return OracleEstimate(
        mean=float(estimate.mean),
        std=float(estimate.std),
        stdev=float(estimate.stdev),
        stderr=float(estimate.stderr),
        n_samples=int(estimate.n_samples),
        raw_values=[float(x) for x in list(estimate.raw_values)],
        aggregate=str(estimate.aggregate),
    )


def _copy_execution_record(record: OracleExecutionRecord) -> OracleExecutionRecord:
    return OracleExecutionRecord(**deepcopy(record.to_dict()))


def _copy_evaluation_result(result: OracleEvaluationResult) -> OracleEvaluationResult:
    return OracleEvaluationResult(
        estimate=_copy_oracle_estimate(result.estimate),
        execution=_copy_execution_record(result.execution),
    )


def _execution_record_from_backend_details(
    details: Mapping[str, Any] | None,
    *,
    noise_mode: str,
    estimator_kind: str,
    backend_name: str | None,
    using_fake_backend: bool,
    config: OracleConfig | None = None,
    resolved_spec: ResolvedNoiseSpec | Mapping[str, Any] | None = None,
) -> OracleExecutionRecord:
    raw = dict(details or {})
    mitigation_cfg = (
        normalize_mitigation_config(getattr(config, "mitigation", "none"))
        if config is not None
        else {}
    )
    runtime_twirling_cfg = (
        normalize_runtime_twirling_config(getattr(config, "runtime_twirling", None))
        if config is not None
        else {}
    )
    symmetry_cfg = (
        normalize_symmetry_mitigation_config(getattr(config, "symmetry_mitigation", "off"))
        if config is not None
        else {}
    )
    if isinstance(resolved_spec, ResolvedNoiseSpec):
        resolved_spec_dict_default = resolved_noise_spec_to_dict(resolved_spec)
        resolved_spec_hash_default = resolved_noise_spec_hash(resolved_spec)
        executor_default = str(resolved_spec.executor)
    elif isinstance(resolved_spec, Mapping):
        resolved_spec_dict_default = dict(resolved_spec)
        resolved_spec_hash_default = raw.get("resolved_noise_spec_hash", None)
        executor_default = str(resolved_spec_dict_default.get("executor", ""))
    else:
        resolved_spec_dict_default = {}
        resolved_spec_hash_default = raw.get("resolved_noise_spec_hash", None)
        executor_default = ""

    legacy_symmetry = _mapping_copy(raw.get("symmetry_mitigation", {}))
    symmetry_mitigation_config = _maybe_mapping_copy(raw.get("symmetry_mitigation_config", None))
    symmetry_mitigation_result = _maybe_mapping_copy(raw.get("symmetry_mitigation_result", None))
    if not symmetry_mitigation_result and _looks_like_symmetry_mitigation_result(legacy_symmetry):
        symmetry_mitigation_result = legacy_symmetry
    if not symmetry_mitigation_config:
        if legacy_symmetry and not _looks_like_symmetry_mitigation_result(legacy_symmetry):
            symmetry_mitigation_config = legacy_symmetry
        else:
            symmetry_mitigation_config = dict(symmetry_cfg)

    provenance_seed = dict(raw)
    if not provenance_seed.get("resolved_noise_spec", None) and resolved_spec_dict_default:
        provenance_seed["resolved_noise_spec"] = deepcopy(resolved_spec_dict_default)
    if not provenance_seed.get("resolved_noise_spec_hash", None) and resolved_spec_hash_default is not None:
        provenance_seed["resolved_noise_spec_hash"] = resolved_spec_hash_default
    provenance_summary = _mapping_copy(raw.get("provenance_summary", {}))
    if not provenance_summary:
        provenance_summary = _derive_provenance_summary(provenance_seed)

    default_shots = None if executor_default == "statevector" else (
        None if config is None else int(getattr(config, "shots", 0))
    )
    shots_raw = raw.get("shots", default_shots)
    shots_value = None if shots_raw is None else int(shots_raw)

    return OracleExecutionRecord(
        noise_mode=str(noise_mode),
        estimator_kind=str(estimator_kind),
        backend_name=backend_name,
        using_fake_backend=bool(using_fake_backend),
        shots=shots_value,
        mitigation=_mapping_copy(raw.get("mitigation", mitigation_cfg) or mitigation_cfg),
        runtime_twirling=_mapping_copy(
            raw.get("runtime_twirling", runtime_twirling_cfg) or runtime_twirling_cfg
        ),
        symmetry_mitigation_config=dict(symmetry_mitigation_config or {}),
        symmetry_mitigation_result=(
            None if not symmetry_mitigation_result else dict(symmetry_mitigation_result)
        ),
        resolved_noise_spec=_mapping_copy(
            raw.get("resolved_noise_spec", resolved_spec_dict_default) or resolved_spec_dict_default
        ),
        resolved_noise_spec_hash=(
            raw.get("resolved_noise_spec_hash", None) or resolved_spec_hash_default
        ),
        calibration_snapshot=_maybe_mapping_copy(raw.get("calibration_snapshot", None)),
        transpile_snapshot=_maybe_mapping_copy(raw.get("transpile_snapshot", None)),
        noise_artifact_hash=raw.get("noise_artifact_hash", None),
        snapshot_hash=raw.get("snapshot_hash", None),
        layout_hash=raw.get("layout_hash", None),
        transpile_hash=raw.get("transpile_hash", None),
        circuit_structure_hash=raw.get("circuit_structure_hash", None),
        layout_anchor_source=raw.get("layout_anchor_source", None),
        runtime_mitigation_options=_maybe_mapping_copy(
            raw.get("runtime_mitigation_options", None)
        ),
        runtime_execution_bundle=_maybe_mapping_copy(raw.get("runtime_execution_bundle", None)),
        fixed_couplers_status=_maybe_mapping_copy(raw.get("fixed_couplers_status", None)),
        patch_selection_summary=_maybe_mapping_copy(raw.get("patch_selection_summary", None)),
        source_kind=raw.get("source_kind", None),
        used_physical_qubits=[int(x) for x in _list_copy(raw.get("used_physical_qubits", []))],
        used_physical_edges=[
            [int(q) for q in list(edge)]
            for edge in _list_copy(raw.get("used_physical_edges", []))
            if isinstance(edge, Sequence) and not isinstance(edge, (str, bytes, bytearray))
        ],
        scheduled_duration_total=raw.get("scheduled_duration_total", None),
        idle_duration_total=raw.get("idle_duration_total", None),
        seed_transpiler=(
            None
            if raw.get("seed_transpiler", None) is None
            else int(raw.get("seed_transpiler"))
        ),
        seed_simulator=(
            None
            if raw.get("seed_simulator", None) is None
            else int(raw.get("seed_simulator"))
        ),
        warnings=[str(x) for x in _list_copy(raw.get("warnings", []))],
        omitted_channels=[str(x) for x in _list_copy(raw.get("omitted_channels", []))],
        noise_model_basis_gates=[str(x) for x in _list_copy(raw.get("noise_model_basis_gates", []))],
        aer_failed=bool(raw.get("aer_failed", False)),
        fallback_used=bool(raw.get("fallback_used", False)),
        fallback_mode=raw.get("fallback_mode", None),
        fallback_reason=str(raw.get("fallback_reason", "") or ""),
        env_workaround_applied=bool(raw.get("env_workaround_applied", False)),
        provenance_summary=dict(provenance_summary),
    )


def _execution_record_from_backend_info(
    backend_info: NoiseBackendInfo,
    *,
    config: OracleConfig | None = None,
    resolved_spec: ResolvedNoiseSpec | Mapping[str, Any] | None = None,
) -> OracleExecutionRecord:
    return _execution_record_from_backend_details(
        getattr(backend_info, "details", {}),
        noise_mode=str(getattr(backend_info, "noise_mode", "")),
        estimator_kind=str(getattr(backend_info, "estimator_kind", "")),
        backend_name=getattr(backend_info, "backend_name", None),
        using_fake_backend=bool(getattr(backend_info, "using_fake_backend", False)),
        config=config,
        resolved_spec=resolved_spec,
    )


def _execution_record_to_backend_details(record: OracleExecutionRecord) -> dict[str, Any]:
    details: dict[str, Any] = {
        "shots": record.shots,
        "mitigation": _mapping_copy(record.mitigation),
        "runtime_twirling": _mapping_copy(record.runtime_twirling),
        "symmetry_mitigation": _mapping_copy(record.symmetry_mitigation),
        "symmetry_mitigation_config": _mapping_copy(record.symmetry_mitigation_config),
        "symmetry_mitigation_result": _maybe_mapping_copy(record.symmetry_mitigation_result),
        "resolved_noise_spec": _mapping_copy(record.resolved_noise_spec),
        "resolved_noise_spec_hash": record.resolved_noise_spec_hash,
        "calibration_snapshot": _maybe_mapping_copy(record.calibration_snapshot),
        "transpile_snapshot": _maybe_mapping_copy(record.transpile_snapshot),
        "noise_artifact_hash": record.noise_artifact_hash,
        "snapshot_hash": record.snapshot_hash,
        "layout_hash": record.layout_hash,
        "transpile_hash": record.transpile_hash,
        "circuit_structure_hash": record.circuit_structure_hash,
        "layout_anchor_source": record.layout_anchor_source,
        "runtime_mitigation_options": _maybe_mapping_copy(record.runtime_mitigation_options),
        "runtime_execution_bundle": _maybe_mapping_copy(record.runtime_execution_bundle),
        "fixed_couplers_status": _maybe_mapping_copy(record.fixed_couplers_status),
        "patch_selection_summary": _maybe_mapping_copy(record.patch_selection_summary),
        "source_kind": record.source_kind,
        "used_physical_qubits": [int(x) for x in list(record.used_physical_qubits)],
        "used_physical_edges": [
            [int(q) for q in list(edge)] for edge in list(record.used_physical_edges)
        ],
        "scheduled_duration_total": record.scheduled_duration_total,
        "idle_duration_total": record.idle_duration_total,
        "seed_transpiler": record.seed_transpiler,
        "seed_simulator": record.seed_simulator,
        "warnings": [str(x) for x in list(record.warnings)],
        "omitted_channels": [str(x) for x in list(record.omitted_channels)],
        "noise_model_basis_gates": [str(x) for x in list(record.noise_model_basis_gates)],
        "aer_failed": bool(record.aer_failed),
        "fallback_used": bool(record.fallback_used),
        "fallback_mode": record.fallback_mode,
        "fallback_reason": str(record.fallback_reason),
        "env_workaround_applied": bool(record.env_workaround_applied),
    }
    provenance_summary = _mapping_copy(record.provenance_summary)
    if not provenance_summary:
        provenance_summary = _derive_provenance_summary(details)
    details["provenance_summary"] = provenance_summary
    return details


def oracle_execution_dict(
    execution: OracleExecutionRecord | Mapping[str, Any] | None = None,
    *,
    backend_info: NoiseBackendInfo | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(execution, OracleExecutionRecord):
        return execution.to_dict()
    if isinstance(execution, Mapping):
        return deepcopy(dict(execution))
    if isinstance(backend_info, NoiseBackendInfo):
        return _execution_record_from_backend_info(backend_info).to_dict()
    if isinstance(backend_info, Mapping):
        record = _execution_record_from_backend_details(
            backend_info.get("details", {}),
            noise_mode=str(backend_info.get("noise_mode", "")),
            estimator_kind=str(backend_info.get("estimator_kind", "")),
            backend_name=backend_info.get("backend_name", None),
            using_fake_backend=bool(backend_info.get("using_fake_backend", False)),
        )
        return record.to_dict()
    return {}


def _to_ixyz(label_exyz: str) -> str:
    return (
        str(label_exyz)
        .replace("e", "I")
        .replace("x", "X")
        .replace("y", "Y")
        .replace("z", "Z")
    )


_PAULI_POLY_TO_QOP_MATH = "H = sum_j c_j P_j  ->  SparsePauliOp([(P_j, c_j)])"


def _pauli_poly_to_sparse_pauli_op(poly: Any, tol: float = 1e-12) -> SparsePauliOp:
    """Convert repo PauliPolynomial (exyz labels) into SparsePauliOp."""
    return _shared_pauli_poly_to_sparse_pauli_op(poly, tol=float(tol))


def _ansatz_terms_with_parameters(ansatz: Any, theta: np.ndarray) -> list[tuple[Any, float]]:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    num_parameters = int(getattr(ansatz, "num_parameters", -1))
    if num_parameters < 0:
        raise ValueError("ansatz is missing num_parameters")
    if int(theta.size) != num_parameters:
        raise ValueError(f"theta length {int(theta.size)} does not match ansatz.num_parameters={num_parameters}")

    reps = int(getattr(ansatz, "reps", 1))
    out: list[tuple[Any, float]] = []
    k = 0

    layer_term_groups = getattr(ansatz, "layer_term_groups", None)
    if isinstance(layer_term_groups, list) and layer_term_groups:
        for _ in range(reps):
            for _name, terms in layer_term_groups:
                val = float(theta[k])
                for term in terms:
                    out.append((term.polynomial, val))
                k += 1
    else:
        base_terms = list(getattr(ansatz, "base_terms", []))
        if not base_terms:
            raise ValueError("ansatz has no base_terms/layer_term_groups")
        for _ in range(reps):
            for term in base_terms:
                out.append((term.polynomial, float(theta[k])))
                k += 1

    if k != int(theta.size):
        raise RuntimeError(
            f"ansatz parameter traversal consumed {k}, expected {int(theta.size)}"
        )
    return out


def _append_reference_state(circuit: QuantumCircuit, reference_state: np.ndarray) -> None:
    _shared_append_reference_state(circuit, reference_state)


def _ansatz_to_circuit(
    ansatz: Any,
    theta: np.ndarray,
    *,
    num_qubits: int,
    reference_state: np.ndarray | None = None,
    coefficient_tolerance: float = 1e-12,
) -> QuantumCircuit:
    """Convert existing hardcoded ansatz object into a Qiskit circuit."""
    return _shared_ansatz_to_circuit(
        ansatz,
        theta,
        num_qubits=int(num_qubits),
        reference_state=reference_state,
        coefficient_tolerance=float(coefficient_tolerance),
    )


def _load_fake_backend(name: str | None) -> tuple[Any, str]:
    try:
        from qiskit_ibm_runtime import fake_provider
    except Exception as exc:
        raise RuntimeError(
            "Unable to import qiskit_ibm_runtime.fake_provider; install qiskit-ibm-runtime."
        ) from exc

    class_name = str(name).strip() if name is not None else "FakeManilaV2"
    if class_name and not class_name.startswith("Fake"):
        class_name = f"Fake{class_name.replace('-', '_').replace(' ', '').title().replace('_', '')}V2"
    backend_cls = getattr(fake_provider, class_name, None)
    if backend_cls is None:
        available = sorted([x for x in dir(fake_provider) if x.startswith("Fake") and x.endswith("V2")])
        sample = ", ".join(available[:8])
        raise ValueError(
            f"Unknown fake backend '{class_name}'. Available examples: {sample}"
        )
    backend = backend_cls()
    setattr(backend, "_hh_noise_source_kind", "fake_snapshot")
    setattr(backend, "_hh_noise_backend_name_override", str(class_name))
    return backend, class_name


def _make_generic_seeded_backend(
    *,
    logical_qubits: int | None,
    seed: int | None,
    backend_name: str | None,
) -> tuple[Any, str]:
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
    except Exception as exc:
        raise RuntimeError("GenericBackendV2 is unavailable in this Qiskit install.") from exc

    nq = max(2, int(logical_qubits or 2))
    edges: list[list[int]] = []
    for q in range(nq - 1):
        edges.append([int(q), int(q + 1)])
        edges.append([int(q + 1), int(q)])
    name = str(backend_name or f"generic_seeded_{nq}q")
    backend = GenericBackendV2(
        int(nq),
        basis_gates=["id", "rz", "sx", "x", "cx", "measure", "delay", "reset"],
        coupling_map=edges,
        dt=2.2222222222222221e-10,
        seed=(None if seed is None else int(seed)),
        noise_info=True,
    )
    setattr(backend, "_hh_noise_source_kind", "generic_seeded")
    setattr(backend, "_hh_noise_backend_name_override", name)
    return backend, name


def _resolve_noise_backend(
    cfg: OracleConfig,
    resolved_spec: ResolvedNoiseSpec,
    *,
    logical_qubits: int | None = None,
) -> tuple[Any | None, str | None, bool]:
    profile = str(resolved_spec.backend_profile_kind)
    if profile == "frozen_snapshot_json":
        return None, str(cfg.backend_name or "frozen_snapshot_json"), False
    if profile == "generic_seeded":
        backend, name = _make_generic_seeded_backend(
            logical_qubits=logical_qubits,
            seed=(resolved_spec.seed_transpiler if resolved_spec.seed_transpiler is not None else cfg.seed),
            backend_name=cfg.backend_name,
        )
        return backend, name, False
    if profile == "fake_snapshot" or bool(cfg.use_fake_backend):
        backend, name = _load_fake_backend(cfg.backend_name)
        return backend, name, True

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:
        raise RuntimeError(
            "qiskit_ibm_runtime is required for live backend lookup. "
            "Use --backend-profile fake_snapshot/generic_seeded or install/configure qiskit-ibm-runtime."
        ) from exc

    if cfg.backend_name is None:
        raise RuntimeError("A live backend profile requires --backend-name <ibm_backend>.")
    try:
        service = QiskitRuntimeService()
        backend = service.backend(str(cfg.backend_name))
        setattr(backend, "_hh_noise_source_kind", "live_backend")
        return backend, str(cfg.backend_name), False
    except Exception as exc:
        raise RuntimeError(
            f"Unable to resolve runtime backend '{cfg.backend_name}'. "
            "Check IBM Runtime credentials, backend name, or use a fake/generic backend profile."
        ) from exc


_OMP_SHM_MARKERS = (
    "OMP: Error #178",
    "Can't open SHM2",
    "Function Can't open SHM2 failed",
    "OMP: System error",
)
_AER_PREFLIGHT_OK_CACHE: set[tuple[str, int, int | None, bool, bool]] = set()
_BACKEND_SNAPSHOT_CACHE: dict[str, Any] = {}


def _tail_text(text: str, max_chars: int = 2400) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return "<no output captured>"
    if len(cleaned) <= int(max_chars):
        return cleaned
    return "..." + cleaned[-int(max_chars):]


def _looks_like_openmp_shm_abort(text: str) -> bool:
    lowered = str(text).lower()
    if not lowered:
        return False
    return any(str(marker).lower() in lowered for marker in _OMP_SHM_MARKERS)


def _apply_omp_env_workaround(cfg: OracleConfig) -> bool:
    if not bool(cfg.omp_shm_workaround):
        return False
    changed = False
    if os.environ.get("KMP_USE_SHM") != "0":
        os.environ["KMP_USE_SHM"] = "0"
        changed = True
    if os.environ.get("OMP_NUM_THREADS") != "1":
        os.environ["OMP_NUM_THREADS"] = "1"
        changed = True
    return changed


def _preflight_aer_environment(cfg: OracleConfig, *, with_noise_model: bool) -> None:
    mode = "aer_noise" if bool(with_noise_model) else "shots"
    key = (
        str(mode),
        int(cfg.shots),
        (None if cfg.seed is None else int(cfg.seed)),
        bool(cfg.approximation),
        bool(cfg.abelian_grouping),
    )
    if key in _AER_PREFLIGHT_OK_CACHE:
        return

    payload = {
        "mode": str(mode),
        "shots": int(cfg.shots),
        "seed": (None if cfg.seed is None else int(cfg.seed)),
        "approximation": bool(cfg.approximation),
        "abelian_grouping": bool(cfg.abelian_grouping),
    }
    script = r"""
import json
import sys

cfg = json.loads(sys.argv[1])
mode = str(cfg.get("mode", "shots")).strip().lower()

from qiskit_aer.primitives import Estimator as AerEstimator

backend_options = {}
if mode == "aer_noise":
    from qiskit_aer.noise import NoiseModel
    backend_options["noise_model"] = NoiseModel()

run_options = {"shots": int(cfg["shots"])}
seed = cfg.get("seed", None)
if seed is not None:
    run_options["seed"] = int(seed)
    run_options["seed_simulator"] = int(seed)

_ = AerEstimator(
    backend_options=backend_options if backend_options else None,
    run_options=run_options,
    approximation=bool(cfg.get("approximation", False)),
    abelian_grouping=bool(cfg.get("abelian_grouping", True)),
)
print("AER_PREFLIGHT_OK")
"""
    env = None
    if bool(cfg.omp_shm_workaround):
        env = dict(os.environ)
        env["KMP_USE_SHM"] = "0"
        env["OMP_NUM_THREADS"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(payload, sort_keys=True)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if int(result.returncode) != 0:
        combined = f"{result.stdout}\n{result.stderr}"
        detail_tail = _tail_text(combined)
        if _looks_like_openmp_shm_abort(combined):
            raise RuntimeError(
                "Aer preflight failed due to OpenMP shared-memory restrictions in this environment "
                "(detected OMP/SHM2 failure). This is an environment-level crash path, not a script logic "
                "error. Local Aer modes are offline and do not require IBM Runtime credentials. "
                f"Preflight stderr/stdout tail:\n{detail_tail}"
            )
        raise RuntimeError(
            "Aer preflight failed before noisy execution started. "
            f"Preflight stderr/stdout tail:\n{detail_tail}"
        )

    _AER_PREFLIGHT_OK_CACHE.add(key)


def _freeze_snapshot_cached(backend_obj: Any) -> Any:
    cache_key = f"{getattr(backend_obj, '_hh_noise_source_kind', 'unknown')}::{getattr(backend_obj, '_hh_noise_backend_name_override', getattr(backend_obj, 'name', None))}::{getattr(backend_obj, 'backend_version', None)}"
    if cache_key not in _BACKEND_SNAPSHOT_CACHE:
        _BACKEND_SNAPSHOT_CACHE[cache_key] = freeze_backend_snapshot(backend_obj)
    return _BACKEND_SNAPSHOT_CACHE[cache_key]


def _build_estimator(
    cfg: OracleConfig,
    *,
    logical_qubits: int | None = None,
) -> tuple[Any, Any | None, NoiseBackendInfo, dict[str, Any] | None]:
    mode = str(cfg.noise_mode).strip().lower()
    mitigation_cfg = normalize_mitigation_config(getattr(cfg, "mitigation", "none"))
    runtime_twirling_cfg = normalize_runtime_twirling_config(getattr(cfg, "runtime_twirling", None))
    symmetry_cfg = normalize_symmetry_mitigation_config(getattr(cfg, "symmetry_mitigation", "off"))
    resolved_spec = normalize_to_resolved_noise_spec(cfg)
    _ensure_mitigation_execution_supported(resolved_spec, mitigation_cfg)
    _ensure_runtime_twirling_execution_supported(resolved_spec, runtime_twirling_cfg)
    _ensure_runtime_twirling_compatibility(mitigation_cfg, runtime_twirling_cfg)
    _ensure_runtime_mode_request_supported(cfg, resolved_spec, mitigation_cfg)
    layer_noise_model, layer_noise_model_meta = _extract_runtime_layer_noise_model(
        getattr(cfg, "mitigation", "none")
    )

    if resolved_spec.executor == "statevector":
        try:
            from qiskit.primitives import StatevectorEstimator
        except Exception as exc:
            raise RuntimeError(
                "Failed to import StatevectorEstimator. Ensure qiskit primitives are available."
            ) from exc
        estimator = StatevectorEstimator()
        details = _with_provenance_summary(
            {
                "shots": None,
                "mitigation": dict(mitigation_cfg),
                "runtime_twirling": dict(runtime_twirling_cfg),
                "symmetry_mitigation": dict(symmetry_cfg),
                "resolved_noise_spec": resolved_noise_spec_to_dict(resolved_spec),
                "resolved_noise_spec_hash": resolved_noise_spec_hash(resolved_spec),
                "source_kind": resolved_spec.backend_profile_kind,
                "warnings": [],
                "omitted_channels": [],
            }
        )
        info = NoiseBackendInfo(
            noise_mode=mode,
            estimator_kind="qiskit.primitives.StatevectorEstimator",
            backend_name="statevector_simulator",
            using_fake_backend=False,
            details=details,
        )
        return estimator, None, info, None

    if resolved_spec.executor == "aer":
        if str(os.environ.get("HH_FORCE_SAMPLER_FALLBACK", "0")).strip() == "1":
            raise RuntimeError(
                "OMP: Error #178: Forced sampler fallback via HH_FORCE_SAMPLER_FALLBACK=1."
            )
        env_workaround_applied = _apply_omp_env_workaround(cfg)
        _preflight_aer_environment(
            cfg,
            with_noise_model=bool(resolved_spec.noise_kind in {"backend_basic", "backend_scheduled", "patch_snapshot"}),
        )
        try:
            from qiskit_aer.primitives import Estimator as AerEstimator
            from qiskit_aer.noise import NoiseModel
        except Exception as exc:
            raise RuntimeError(
                "Failed to import qiskit-aer Estimator/NoiseModel. Install qiskit-aer."
            ) from exc

        backend_obj, backend_name, using_fake = _resolve_noise_backend(
            cfg,
            resolved_spec,
            logical_qubits=logical_qubits,
        )
        calibration_snapshot = None
        warnings: list[str] = []
        if resolved_spec.backend_profile_kind == "frozen_snapshot_json":
            if resolved_spec.snapshot_path is None:
                raise RuntimeError("frozen_snapshot_json requires --noise-snapshot-json PATH.")
            calibration_snapshot = load_calibration_snapshot(resolved_spec.snapshot_path)
            backend_name = str(calibration_snapshot.backend_name or backend_name or "frozen_snapshot_json")
            if backend_obj is None and resolved_spec.noise_kind in {"backend_basic", "backend_scheduled", "patch_snapshot"}:
                backend_obj, snapshot_warnings = build_backend_from_calibration_snapshot(
                    calibration_snapshot,
                    backend_name_override=backend_name,
                )
                warnings.extend(list(snapshot_warnings))
            elif backend_obj is None:
                raise RuntimeError(
                    "Local Aer replay from a frozen snapshot JSON is only implemented for backend_basic/backend_scheduled/patch_snapshot in the current repo. "
                    f"Unsupported frozen-snapshot noise_kind={resolved_spec.noise_kind!r}. "
                    "Use --backend-profile fake_snapshot/live_backend/generic_seeded for other local modes."
                )
        elif backend_obj is not None:
            calibration_snapshot = _freeze_snapshot_cached(backend_obj)
            if resolved_spec.snapshot_path is not None:
                write_calibration_snapshot(resolved_spec.snapshot_path, calibration_snapshot)

        noise_model = None
        backend_options: dict[str, Any] = {}
        omitted_channels: list[str] = []
        if resolved_spec.noise_kind == "backend_basic":
            noise_model = NoiseModel.from_backend(backend_obj)
            backend_options["noise_model"] = noise_model
            omitted_channels = [
                "delay_relaxation_if_unscheduled",
                "crosstalk",
                "leakage",
                "drift",
                "coherent_overrotation",
            ]
            warnings.append("backend_basic is a smoke/debug hardware-facing approximation; scheduling is not enforced.")
        elif resolved_spec.noise_kind == "backend_scheduled":
            noise_model = NoiseModel.from_backend(backend_obj)
            backend_options["noise_model"] = noise_model
            omitted_channels = [
                "crosstalk",
                "leakage",
                "non_markovian_drift",
                "coherent_overrotation",
            ]
        elif resolved_spec.noise_kind == "patch_snapshot":
            if backend_obj is None or calibration_snapshot is None:
                raise RuntimeError("patch_snapshot requires a resolved backend-like target and calibration snapshot.")
            noise_model = NoiseModel.from_backend(backend_obj)
            backend_options["noise_model"] = noise_model
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
            warnings.append(
                "patch_snapshot will replay a frozen patch/layout from the persisted layout lock; transpile metadata is regenerated during replay."
            )

        run_options: dict[str, Any] = {"shots": int(cfg.shots)}
        sim_seed = resolved_spec.seed_simulator if resolved_spec.seed_simulator is not None else cfg.seed
        if sim_seed is not None:
            run_options["seed"] = int(sim_seed)
            run_options["seed_simulator"] = int(sim_seed)
        estimator = AerEstimator(
            backend_options=backend_options if backend_options else None,
            run_options=run_options,
            approximation=bool(cfg.approximation),
            abelian_grouping=bool(cfg.abelian_grouping),
        )
        details: dict[str, Any] = _with_provenance_summary(
            {
                "shots": int(cfg.shots),
                "aer_failed": False,
                "fallback_used": False,
                "fallback_mode": str(cfg.aer_fallback_mode),
                "fallback_reason": "",
                "env_workaround_applied": bool(cfg.omp_shm_workaround or env_workaround_applied),
                "mitigation": dict(mitigation_cfg),
                "runtime_twirling": dict(runtime_twirling_cfg),
                "symmetry_mitigation": dict(symmetry_cfg),
                "resolved_noise_spec": resolved_noise_spec_to_dict(resolved_spec),
                "resolved_noise_spec_hash": resolved_noise_spec_hash(resolved_spec),
                "calibration_snapshot": calibration_snapshot_to_dict(calibration_snapshot),
                "snapshot_hash": (None if calibration_snapshot is None else calibration_snapshot.snapshot_hash),
                "source_kind": resolved_spec.backend_profile_kind,
                "seed_transpiler": resolved_spec.seed_transpiler,
                "seed_simulator": resolved_spec.seed_simulator,
                "warnings": list(warnings),
                "omitted_channels": list(omitted_channels),
            }
        )
        if noise_model is not None:
            details["noise_model_basis_gates"] = list(getattr(noise_model, "basis_gates", []))
        info = NoiseBackendInfo(
            noise_mode=mode,
            estimator_kind="qiskit_aer.primitives.Estimator",
            backend_name=backend_name,
            using_fake_backend=using_fake,
            details=details,
        )
        local_context = {
            "resolved_spec": resolved_spec,
            "resolved_backend": backend_obj,
            "calibration_snapshot": calibration_snapshot,
            "noise_model": noise_model,
            "warnings": warnings,
            "omitted_channels": omitted_channels,
        }
        return estimator, None, info, local_context

    # runtime_qpu
    try:
        from qiskit_ibm_runtime import (
            QiskitRuntimeService,
            Session,
            EstimatorV2 as RuntimeEstimatorV2,
        )
    except Exception as exc:
        raise RuntimeError(
            "runtime mode requires qiskit-ibm-runtime. Install and configure IBM Runtime."
        ) from exc

    if cfg.backend_name is None:
        raise RuntimeError(
            "runtime mode requires --backend-name <ibm_backend>."
        )

    try:
        service = QiskitRuntimeService()
        backend = service.backend(str(cfg.backend_name))
        setattr(backend, "_hh_noise_source_kind", "live_backend")
        calibration_snapshot = None
        warnings: list[str] = []
        try:
            calibration_snapshot = _freeze_snapshot_cached(backend)
        except Exception as exc:
            warnings.append(f"runtime_backend_snapshot_unavailable:{type(exc).__name__}:{exc}")
        session = Session(service=service, backend=backend)
        runtime_options, runtime_mitigation_options = _build_runtime_mitigation_options(
            resolved_spec,
            mitigation_cfg,
            runtime_twirling_cfg,
            layer_noise_model,
            layer_noise_model_meta,
        )
        runtime_execution_bundle = _runtime_execution_bundle_details(
            cfg,
            resolved_spec,
            mitigation_cfg,
            runtime_twirling_cfg,
            layer_noise_model_meta,
        )
        estimator = RuntimeEstimatorV2(mode=session, options=runtime_options)
        details = _with_provenance_summary(
            {
                "shots": int(cfg.shots),
                "mitigation": dict(mitigation_cfg),
                "runtime_twirling": dict(runtime_twirling_cfg),
                "symmetry_mitigation": dict(symmetry_cfg),
                "resolved_noise_spec": resolved_noise_spec_to_dict(resolved_spec),
                "resolved_noise_spec_hash": resolved_noise_spec_hash(resolved_spec),
                "calibration_snapshot": calibration_snapshot_to_dict(calibration_snapshot),
                "snapshot_hash": (None if calibration_snapshot is None else calibration_snapshot.snapshot_hash),
                "source_kind": (
                    resolved_spec.backend_profile_kind
                    if calibration_snapshot is None
                    else calibration_snapshot.source_kind
                ),
                "warnings": list(warnings),
                "omitted_channels": [],
                "seed_transpiler": resolved_spec.seed_transpiler,
                "seed_simulator": resolved_spec.seed_simulator,
                "runtime_mitigation_options": dict(runtime_mitigation_options),
                "runtime_execution_bundle": dict(runtime_execution_bundle),
            }
        )
        info = NoiseBackendInfo(
            noise_mode=mode,
            estimator_kind="qiskit_ibm_runtime.EstimatorV2",
            backend_name=str(cfg.backend_name),
            using_fake_backend=False,
            details=details,
        )
        runtime_context = {
            "resolved_spec": resolved_spec,
            "resolved_backend": backend,
            "calibration_snapshot": calibration_snapshot,
            "noise_model": None,
            "warnings": list(warnings),
            "omitted_channels": [],
        }
        return estimator, session, info, runtime_context
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize IBM Runtime Estimator. "
            "Verify IBM credentials (`QISKIT_IBM_TOKEN`), backend availability, and account access."
        ) from exc


def _extract_expectation_value(result: Any) -> float:
    if hasattr(result, "values"):
        vals = np.asarray(getattr(result, "values"), dtype=float).reshape(-1)
        if vals.size > 0:
            return float(vals[0])

    try:
        first = result[0]
    except Exception:
        first = None

    if first is not None:
        data = getattr(first, "data", None)
        if data is not None and hasattr(data, "evs"):
            evs = np.asarray(getattr(data, "evs"), dtype=float).reshape(-1)
            if evs.size > 0:
                return float(evs[0])
        if hasattr(first, "value"):
            return float(np.real(getattr(first, "value")))
        if hasattr(first, "evs"):
            evs = np.asarray(getattr(first, "evs"), dtype=float).reshape(-1)
            if evs.size > 0:
                return float(evs[0])

    if hasattr(result, "evs"):
        evs = np.asarray(getattr(result, "evs"), dtype=float).reshape(-1)
        if evs.size > 0:
            return float(evs[0])

    raise RuntimeError(
        f"Unable to extract expectation value from estimator result type: {type(result)!r}"
    )


def _run_estimator_job(estimator: Any, circuit: QuantumCircuit, observable: SparsePauliOp) -> float:
    errors: list[Exception] = []

    # V2-style tuple(pub) invocation
    for pub in (
        [(circuit, observable)],
        [(circuit, [observable])],
    ):
        try:
            job = estimator.run(pub)
            result = job.result()
            return float(np.real(_extract_expectation_value(result)))
        except Exception as exc:
            errors.append(exc)

    # V1-style invocation
    try:
        job = estimator.run([circuit], [observable])
        result = job.result()
        return float(np.real(_extract_expectation_value(result)))
    except Exception as exc:
        errors.append(exc)

    msg = "; ".join(f"{type(e).__name__}: {e}" for e in errors)
    raise RuntimeError(f"Estimator execution failed across known call paths. Details: {msg}")


def _term_measurement_circuit(base: QuantumCircuit, pauli_label_ixyz: str) -> QuantumCircuit:
    """Rotate into Pauli measurement basis and measure all qubits."""
    label = str(pauli_label_ixyz).upper()
    n = int(base.num_qubits)
    if len(label) != n:
        raise ValueError(f"Pauli label length {len(label)} does not match circuit qubits {n}")

    qc = base.copy()
    for q in range(n):
        op = label[n - 1 - q]  # left-to-right is q_(n-1)..q_0; q0 rightmost
        if op == "X":
            qc.h(q)
        elif op == "Y":
            qc.sdg(q)
            qc.h(q)
        elif op in {"I", "Z"}:
            continue
        else:
            raise ValueError(f"Unsupported Pauli op '{op}' in '{label}'")
    qc.measure_all()
    return qc


def _pauli_parity_from_bitstring(bitstr_raw: str, pauli_label_ixyz: str, n_qubits: int) -> float:
    label = str(pauli_label_ixyz).upper()
    if len(label) != int(n_qubits):
        raise ValueError(f"Pauli label length {len(label)} does not match n_qubits={n_qubits}")
    bitstr = str(bitstr_raw).replace(" ", "")
    if len(bitstr) < int(n_qubits):
        bitstr = bitstr.zfill(int(n_qubits))
    parity = 1.0
    for q in range(int(n_qubits)):
        if label[int(n_qubits) - 1 - q] == "I":
            continue
        bit = bitstr[-1 - int(q)]
        parity *= (-1.0 if bit == "1" else 1.0)
    return float(parity)


def _pauli_expectation_from_counts(counts: dict[str, int], pauli_label_ixyz: str, n_qubits: int) -> float:
    label = str(pauli_label_ixyz).upper()
    if len(label) != int(n_qubits):
        raise ValueError(f"Pauli label length {len(label)} does not match n_qubits={n_qubits}")
    active_q = [q for q in range(int(n_qubits)) if label[int(n_qubits) - 1 - q] != "I"]
    if not active_q:
        return 1.0
    shots = int(sum(int(v) for v in counts.values()))
    if shots <= 0:
        raise RuntimeError("Sampler returned zero total shots.")

    acc = 0.0
    for bitstr_raw, ct in counts.items():
        acc += _pauli_parity_from_bitstring(str(bitstr_raw), label, int(n_qubits)) * float(ct)
    return float(acc / float(shots))


def _observable_is_diagonal(observable: SparsePauliOp) -> bool:
    for label, _coeff in observable.to_list():
        if any(ch not in {"I", "Z"} for ch in str(label).upper()):
            return False
    return True


def _spin_orbital_index_sets(num_sites: int, ordering: str) -> tuple[list[int], list[int]]:
    if str(ordering).strip().lower() == "interleaved":
        return [2 * i for i in range(int(num_sites))], [2 * i + 1 for i in range(int(num_sites))]
    return list(range(int(num_sites))), list(range(int(num_sites), 2 * int(num_sites)))


def _bitstring_passes_sector(
    bitstr_raw: str,
    *,
    n_qubits: int,
    num_sites: int,
    ordering: str,
    sector_n_up: int,
    sector_n_dn: int,
) -> bool:
    bitstr = str(bitstr_raw).replace(" ", "")
    if len(bitstr) < int(n_qubits):
        bitstr = bitstr.zfill(int(n_qubits))
    alpha_indices, beta_indices = _spin_orbital_index_sets(int(num_sites), str(ordering))
    n_up = sum(1 for idx in alpha_indices if bitstr[-1 - int(idx)] == "1")
    n_dn = sum(1 for idx in beta_indices if bitstr[-1 - int(idx)] == "1")
    return int(n_up) == int(sector_n_up) and int(n_dn) == int(sector_n_dn)


def _diagonal_expectation_from_counts(counts: dict[str, int], observable: SparsePauliOp) -> float:
    total = 0.0 + 0.0j
    n = int(observable.num_qubits)
    for label, coeff in observable.to_list():
        total += complex(coeff) * complex(_pauli_expectation_from_counts(counts, str(label), n), 0.0)
    return float(np.real(total))


def _exact_postselected_diagonal_expectation(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    symmetry_cfg: Mapping[str, Any],
) -> tuple[float, float]:
    psi = np.asarray(Statevector.from_instruction(circuit).data, dtype=complex).reshape(-1)
    n_qubits = int(circuit.num_qubits)
    kept_prob = 0.0
    total = 0.0 + 0.0j
    for idx, amp in enumerate(psi):
        prob = float(abs(complex(amp)) ** 2)
        if prob <= 1e-18:
            continue
        bitstr = format(int(idx), f"0{n_qubits}b")
        if not _bitstring_passes_sector(
            bitstr,
            n_qubits=int(n_qubits),
            num_sites=int(symmetry_cfg.get("num_sites", 0)),
            ordering=str(symmetry_cfg.get("ordering", "blocked")),
            sector_n_up=int(symmetry_cfg.get("sector_n_up", 0)),
            sector_n_dn=int(symmetry_cfg.get("sector_n_dn", 0)),
        ):
            continue
        kept_prob += prob
        for label, coeff in observable.to_list():
            total += complex(coeff) * complex(
                _pauli_parity_from_bitstring(bitstr, str(label), int(n_qubits)) * prob,
                0.0,
            )
    if kept_prob <= 0.0:
        raise SymmetryMitigationDowngrade("Symmetry postselection retained zero probability mass.")
    return float(np.real(total) / kept_prob), float(kept_prob)


def _exact_projector_renorm_diagonal_expectation(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    symmetry_cfg: Mapping[str, Any],
) -> tuple[float, float]:
    psi = np.asarray(Statevector.from_instruction(circuit).data, dtype=complex).reshape(-1)
    n_qubits = int(circuit.num_qubits)
    sector_prob = 0.0
    numerator = 0.0 + 0.0j
    for idx, amp in enumerate(psi):
        prob = float(abs(complex(amp)) ** 2)
        if prob <= 1e-18:
            continue
        bitstr = format(int(idx), f"0{n_qubits}b")
        in_sector = _bitstring_passes_sector(
            bitstr,
            n_qubits=int(n_qubits),
            num_sites=int(symmetry_cfg.get("num_sites", 0)),
            ordering=str(symmetry_cfg.get("ordering", "blocked")),
            sector_n_up=int(symmetry_cfg.get("sector_n_up", 0)),
            sector_n_dn=int(symmetry_cfg.get("sector_n_dn", 0)),
        )
        if not in_sector:
            continue
        sector_prob += prob
        for label, coeff in observable.to_list():
            numerator += complex(coeff) * complex(
                _pauli_parity_from_bitstring(bitstr, str(label), int(n_qubits)) * prob,
                0.0,
            )
    if sector_prob <= 0.0:
        raise SymmetryMitigationDowngrade(
            "Projector renormalization retained zero probability mass."
        )
    return float(np.real(numerator) / sector_prob), float(sector_prob)


def _sample_measurement_counts(
    circuit: QuantumCircuit,
    cfg: OracleConfig,
    *,
    repeat_idx: int,
) -> dict[str, int]:
    measured = circuit.copy()
    measured.measure_all()
    mode = str(cfg.noise_mode).strip().lower()
    if mode == "shots" or mode == "ideal":
        from qiskit.primitives import StatevectorSampler

        sampler = StatevectorSampler(
            default_shots=int(cfg.shots),
            seed=int(cfg.seed) + int(repeat_idx),
        )
        job = sampler.run([measured])
        result = job.result()
        return dict(result[0].join_data().get_counts())

    resolved_spec = normalize_to_resolved_noise_spec(cfg)
    if resolved_spec.executor == "aer":
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel

        backend_obj, _backend_name, _using_fake = _resolve_noise_backend(
            cfg,
            resolved_spec,
            logical_qubits=int(circuit.num_qubits),
        )
        sim_kwargs: dict[str, Any] = {
            "seed_simulator": int(
                resolved_spec.seed_simulator if resolved_spec.seed_simulator is not None else cfg.seed
            )
            + int(repeat_idx)
        }
        if backend_obj is not None and resolved_spec.noise_kind in {"backend_basic", "backend_scheduled"}:
            sim_kwargs["noise_model"] = NoiseModel.from_backend(backend_obj)
        sim = AerSimulator(**sim_kwargs)
        compiled = transpile(measured, backend_obj or sim, optimization_level=0)
        result = sim.run(compiled, shots=int(cfg.shots)).result()
        counts = result.get_counts()
        if isinstance(counts, list):
            counts = counts[0]
        return dict(counts)

    raise RuntimeError(f"Counts-based symmetry mitigation is unavailable for noise_mode={mode!r}.")


def _postselected_counts_and_fraction(
    counts: Mapping[str, int],
    *,
    n_qubits: int,
    symmetry_cfg: Mapping[str, Any],
) -> tuple[dict[str, int], float]:
    kept: dict[str, int] = {}
    total = int(sum(int(v) for v in counts.values()))
    if total <= 0:
        raise RuntimeError("Counts-based symmetry mitigation received zero total shots.")
    kept_total = 0
    for bitstr_raw, ct_raw in counts.items():
        ct = int(ct_raw)
        if ct <= 0:
            continue
        if _bitstring_passes_sector(
            str(bitstr_raw),
            n_qubits=int(n_qubits),
            num_sites=int(symmetry_cfg.get("num_sites", 0)),
            ordering=str(symmetry_cfg.get("ordering", "blocked")),
            sector_n_up=int(symmetry_cfg.get("sector_n_up", 0)),
            sector_n_dn=int(symmetry_cfg.get("sector_n_dn", 0)),
        ):
            kept[str(bitstr_raw)] = kept.get(str(bitstr_raw), 0) + int(ct)
            kept_total += int(ct)
    if kept_total <= 0:
        raise SymmetryMitigationDowngrade("Symmetry postselection retained zero shots.")
    return kept, float(kept_total) / float(total)


def _projector_renorm_diagonal_expectation_from_counts(
    counts: Mapping[str, int],
    observable: SparsePauliOp,
    *,
    n_qubits: int,
    symmetry_cfg: Mapping[str, Any],
) -> tuple[float, float]:
    total_shots = int(sum(int(v) for v in counts.values()))
    if total_shots <= 0:
        raise RuntimeError("Counts-based projector renormalization received zero total shots.")
    sector_shots = 0
    numerator = 0.0 + 0.0j
    for bitstr_raw, ct_raw in counts.items():
        ct = int(ct_raw)
        if ct <= 0:
            continue
        if not _bitstring_passes_sector(
            str(bitstr_raw),
            n_qubits=int(n_qubits),
            num_sites=int(symmetry_cfg.get("num_sites", 0)),
            ordering=str(symmetry_cfg.get("ordering", "blocked")),
            sector_n_up=int(symmetry_cfg.get("sector_n_up", 0)),
            sector_n_dn=int(symmetry_cfg.get("sector_n_dn", 0)),
        ):
            continue
        sector_shots += int(ct)
        for label, coeff in observable.to_list():
            numerator += complex(coeff) * complex(
                _pauli_parity_from_bitstring(str(bitstr_raw), str(label), int(n_qubits)) * float(ct),
                0.0,
            )
    if sector_shots <= 0:
        raise SymmetryMitigationDowngrade("Projector renormalization retained zero shots.")
    sector_prob = float(sector_shots) / float(total_shots)
    numerator_expectation = numerator / float(total_shots)
    return float(np.real(numerator_expectation) / sector_prob), float(sector_prob)


def _run_sampler_fallback_job(
    sampler: Any,
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
) -> float:
    total = 0.0 + 0.0j
    n = int(observable.num_qubits)
    for label, coeff in observable.to_list():
        lbl = str(label).upper()
        meas_qc = _term_measurement_circuit(circuit, lbl)
        job = sampler.run([meas_qc])
        result = job.result()
        counts = result[0].join_data().get_counts()
        exp_lbl = _pauli_expectation_from_counts(counts, lbl, n)
        total += complex(coeff) * complex(exp_lbl, 0.0)
    return float(np.real(total))


class ExpectationOracle:
    """Shared expectation-value oracle for ideal/noisy/runtime execution."""

    def __init__(self, config: OracleConfig):
        self.config = OracleConfig(
            noise_mode=str(config.noise_mode).strip().lower(),
            shots=int(config.shots),
            seed=int(config.seed),
            oracle_repeats=max(1, int(config.oracle_repeats)),
            oracle_aggregate=str(config.oracle_aggregate).strip().lower(),
            backend_name=(None if config.backend_name is None else str(config.backend_name)),
            use_fake_backend=bool(config.use_fake_backend),
            backend_profile=(None if getattr(config, "backend_profile", None) in {None, "", "none"} else str(getattr(config, "backend_profile"))),
            aer_noise_kind=str(getattr(config, "aer_noise_kind", "scheduled")).strip().lower(),
            schedule_policy=(None if getattr(config, "schedule_policy", None) in {None, "", "none"} else str(getattr(config, "schedule_policy")).strip().lower()),
            layout_policy=(None if getattr(config, "layout_policy", None) in {None, "", "none"} else str(getattr(config, "layout_policy")).strip().lower()),
            noise_snapshot_json=(None if getattr(config, "noise_snapshot_json", None) in {None, "", "none"} else str(getattr(config, "noise_snapshot_json"))),
            fixed_physical_patch=getattr(config, "fixed_physical_patch", None),
            fixed_couplers=getattr(config, "fixed_couplers", None),
            layout_lock_key=(None if getattr(config, "layout_lock_key", None) in {None, "", "none"} else str(getattr(config, "layout_lock_key"))),
            seed_transpiler=(None if getattr(config, "seed_transpiler", None) is None else int(getattr(config, "seed_transpiler"))),
            seed_simulator=(None if getattr(config, "seed_simulator", None) is None else int(getattr(config, "seed_simulator"))),
            approximation=bool(config.approximation),
            abelian_grouping=bool(config.abelian_grouping),
            allow_aer_fallback=bool(getattr(config, "allow_aer_fallback", getattr(config, "allow_noisy_fallback", False))),
            allow_noisy_fallback=bool(getattr(config, "allow_noisy_fallback", getattr(config, "allow_aer_fallback", False))),
            aer_fallback_mode=str(config.aer_fallback_mode).strip().lower(),
            omp_shm_workaround=bool(config.omp_shm_workaround),
            mitigation=normalize_mitigation_config(getattr(config, "mitigation", "none")),
            runtime_twirling=normalize_runtime_twirling_config(getattr(config, "runtime_twirling", None)),
            symmetry_mitigation=normalize_symmetry_mitigation_config(
                getattr(config, "symmetry_mitigation", "off")
            ),
        )
        if self.config.oracle_aggregate not in {"mean", "median"}:
            raise ValueError(
                f"Unsupported oracle_aggregate={self.config.oracle_aggregate}; use mean or median."
            )
        if self.config.aer_fallback_mode not in {"sampler_shots"}:
            raise ValueError(
                f"Unsupported aer_fallback_mode={self.config.aer_fallback_mode}; use sampler_shots."
            )

        self.resolved_noise_spec: ResolvedNoiseSpec = normalize_to_resolved_noise_spec(self.config)
        self._sampler_fallback = None
        self._fallback_reason = ""
        self._estimator = None
        self._session = None
        self._local_context: dict[str, Any] | None = None
        self.backend_info = NoiseBackendInfo(
            noise_mode=str(self.config.noise_mode),
            estimator_kind="unknown",
            backend_name=None,
            using_fake_backend=bool(self.config.use_fake_backend),
            details={},
        )
        self._current_execution = _execution_record_from_backend_info(
            self.backend_info,
            config=self.config,
            resolved_spec=self.resolved_noise_spec,
        )
        self._last_result: OracleEvaluationResult | None = None

        try:
            build_out = _build_estimator(self.config)
            self._estimator, self._session, self.backend_info, self._local_context = build_out
            self._sync_execution_from_backend_info()
        except Exception as exc:
            if self._can_fallback_from_error(exc):
                self._activate_sampler_fallback(reason=str(exc), aer_failed=True)
            else:
                raise
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
        self._closed = True

    def __enter__(self) -> "ExpectationOracle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _local_mode_active(self) -> bool:
        return bool(self.resolved_noise_spec.executor == "aer")

    def _runtime_mode_active(self) -> bool:
        return bool(self.resolved_noise_spec.executor == "runtime_qpu")

    def _ensure_estimator_ready(self, *, logical_qubits: int | None = None) -> None:
        backend_obj = None if self._local_context is None else self._local_context.get("resolved_backend", None)
        need_rebuild = self._estimator is None and self._sampler_fallback is None
        if (
            not need_rebuild
            and self._local_mode_active()
            and backend_obj is not None
            and getattr(backend_obj, "num_qubits", None) is not None
            and logical_qubits is not None
            and int(logical_qubits) > int(getattr(backend_obj, "num_qubits"))
            and str(self.resolved_noise_spec.backend_profile_kind) == "generic_seeded"
        ):
            need_rebuild = True
        if not need_rebuild:
            return
        build_out = _build_estimator(self.config, logical_qubits=logical_qubits)
        self._estimator, self._session, self.backend_info, self._local_context = build_out
        self._sync_execution_from_backend_info()

    @property
    def current_execution(self) -> OracleExecutionRecord:
        return _copy_execution_record(self._current_execution)

    @property
    def last_result(self) -> OracleEvaluationResult | None:
        return None if self._last_result is None else _copy_evaluation_result(self._last_result)

    def _sync_execution_from_backend_info(self) -> None:
        self._current_execution = _execution_record_from_backend_info(
            self.backend_info,
            config=self.config,
            resolved_spec=self.resolved_noise_spec,
        )
        self._sync_backend_info_from_execution()

    def _sync_backend_info_from_execution(self) -> None:
        self.backend_info = NoiseBackendInfo(
            noise_mode=str(self._current_execution.noise_mode),
            estimator_kind=str(self._current_execution.estimator_kind),
            backend_name=self._current_execution.backend_name,
            using_fake_backend=bool(self._current_execution.using_fake_backend),
            details=self._current_execution.to_legacy_details(),
        )

    def _update_backend_details(self, **updates: Any) -> None:
        details = self._current_execution.to_legacy_details()
        details.update(updates)
        if details:
            details = _with_provenance_summary(details)
        self._current_execution = _execution_record_from_backend_details(
            details,
            noise_mode=str(self._current_execution.noise_mode),
            estimator_kind=str(self._current_execution.estimator_kind),
            backend_name=self._current_execution.backend_name,
            using_fake_backend=bool(self._current_execution.using_fake_backend),
            config=self.config,
            resolved_spec=self.resolved_noise_spec,
        )
        self._sync_backend_info_from_execution()

    def _set_symmetry_mitigation_details(self, details_map: Mapping[str, Any]) -> None:
        self._update_backend_details(symmetry_mitigation_result=dict(details_map))

    def _record_noise_artifact(
        self,
        artifact: NoiseArtifact,
        *,
        layout_anchor_source: str | None = None,
    ) -> None:
        metadata = noise_artifact_metadata(artifact)
        transpile_snapshot = metadata.get("transpile_snapshot", None) or {}
        calibration_snapshot = metadata.get("calibration_snapshot", None) or {}
        existing_warnings = list(getattr(self.backend_info, "details", {}).get("warnings", []))
        merged_warnings = list(dict.fromkeys(existing_warnings + list(metadata.get("warnings", []))))
        updates = {
            "resolved_noise_spec": metadata.get("resolved_spec", {}),
            "resolved_noise_spec_hash": resolved_noise_spec_hash(artifact.resolved_spec),
            "calibration_snapshot": calibration_snapshot,
            "transpile_snapshot": transpile_snapshot,
            "noise_artifact_hash": metadata.get("noise_artifact_hash"),
            "snapshot_hash": calibration_snapshot.get("snapshot_hash", None),
            "layout_hash": metadata.get("layout_hash"),
            "transpile_hash": transpile_snapshot.get("transpile_hash", None),
            "circuit_structure_hash": metadata.get("circuit_structure_hash", None),
            "omitted_channels": list(metadata.get("omitted_channels", [])),
            "warnings": merged_warnings,
            "source_kind": calibration_snapshot.get("source_kind", self.resolved_noise_spec.backend_profile_kind),
            "used_physical_qubits": list(transpile_snapshot.get("used_physical_qubits", [])),
            "used_physical_edges": list(transpile_snapshot.get("used_physical_edges", [])),
            "scheduled_duration_total": transpile_snapshot.get("scheduled_duration_total", None),
            "idle_duration_total": transpile_snapshot.get("idle_duration_total", None),
            "seed_transpiler": artifact.resolved_spec.seed_transpiler,
            "seed_simulator": artifact.resolved_spec.seed_simulator,
            "fixed_couplers_status": metadata.get("fixed_couplers_status", None),
            "patch_selection_summary": metadata.get("patch_selection_summary", None),
        }
        if layout_anchor_source is not None:
            updates["layout_anchor_source"] = str(layout_anchor_source)
        self._update_backend_details(**updates)

    def _handle_symmetry_downgrade(self, details: dict[str, Any], reason: str) -> OracleEstimate | None:
        details["applied_mode"] = "verify_only"
        details["fallback_reason"] = str(reason)
        warnings = list(getattr(self.backend_info, "details", {}).get("warnings", []))
        downgrade_warning = f"symmetry_mitigation_downgraded:{str(reason)}"
        if downgrade_warning not in warnings:
            warnings.append(downgrade_warning)
        if self.resolved_noise_spec.executor in {"statevector", "runtime_qpu"}:
            self._set_symmetry_mitigation_details(details)
            self._update_backend_details(warnings=warnings)
            return None
        if not self._fallback_allowed_for_mode():
            raise RuntimeError(
                "Symmetry mitigation requested but could not be executed: {}. "
                "Re-run with --allow-noisy-fallback to permit verify_only downgrade.".format(
                    str(reason)
                )
            )
        self._set_symmetry_mitigation_details(details)
        self._update_backend_details(warnings=warnings)
        return None

    def _build_local_artifact(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp | None,
    ) -> NoiseArtifact:
        self._ensure_estimator_ready(logical_qubits=int(circuit.num_qubits))
        if self._local_context is None:
            raise RuntimeError("Local Aer context was not initialized.")
        resolved_spec = self._local_context["resolved_spec"]
        backend_obj = self._local_context.get("resolved_backend", None)
        calibration_snapshot = self._local_context.get("calibration_snapshot", None)
        layout_anchor_source = describe_layout_anchor_source(resolved_spec, calibration_snapshot)
        if resolved_spec.noise_kind == "shots_only":
            if backend_obj is None:
                backend_obj, _name, _using_fake = _resolve_noise_backend(
                    self.config,
                    resolved_spec,
                    logical_qubits=int(circuit.num_qubits),
                )
            artifact = build_shots_only_artifact(
                circuit=circuit,
                observable=observable,
                resolved_spec=resolved_spec,
                resolved_backend=backend_obj,
                calibration_snapshot=calibration_snapshot,
            )
        elif resolved_spec.noise_kind == "backend_basic":
            if backend_obj is None or calibration_snapshot is None:
                raise RuntimeError("backend_basic requires a resolved backend and calibration snapshot.")
            artifact = build_backend_basic_artifact(
                circuit=circuit,
                observable=observable,
                resolved_spec=resolved_spec,
                resolved_backend=backend_obj,
                calibration_snapshot=calibration_snapshot,
            )
        elif resolved_spec.noise_kind == "backend_scheduled":
            if backend_obj is None or calibration_snapshot is None:
                raise RuntimeError("backend_scheduled requires a resolved backend and calibration snapshot.")
            artifact = build_backend_scheduled_artifact(
                circuit=circuit,
                observable=observable,
                resolved_spec=resolved_spec,
                resolved_backend=backend_obj,
                calibration_snapshot=calibration_snapshot,
            )
        elif resolved_spec.noise_kind == "patch_snapshot":
            artifact = build_patch_snapshot_artifact(
                circuit=circuit,
                observable=observable,
                resolved_spec=resolved_spec,
                calibration_snapshot=calibration_snapshot,
                resolved_backend=backend_obj,
                qiskit_noise_model=self._local_context.get("noise_model", None),
            )
        else:
            raise RuntimeError(f"Unsupported local Aer noise kind {resolved_spec.noise_kind!r}.")
        self._record_noise_artifact(artifact, layout_anchor_source=layout_anchor_source)
        return artifact

    def _build_runtime_artifact(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp | None,
    ) -> NoiseArtifact:
        self._ensure_estimator_ready(logical_qubits=int(circuit.num_qubits))
        if self._local_context is None:
            raise RuntimeError("Runtime submission context was not initialized.")
        resolved_spec = self._local_context["resolved_spec"]
        backend_obj = self._local_context.get("resolved_backend", None)
        calibration_snapshot = self._local_context.get("calibration_snapshot", None)
        if backend_obj is None:
            raise RuntimeError("Runtime submission requires a resolved backend target.")
        layout_anchor_source = describe_layout_anchor_source(resolved_spec, calibration_snapshot)
        artifact = build_runtime_submission_artifact(
            circuit=circuit,
            observable=observable,
            resolved_spec=resolved_spec,
            resolved_backend=backend_obj,
            calibration_snapshot=calibration_snapshot,
        )
        self._record_noise_artifact(artifact, layout_anchor_source=layout_anchor_source)
        return artifact

    def prime_layout(self, circuit: QuantumCircuit) -> None:
        if self._sampler_fallback is not None:
            return
        if self._local_mode_active():
            _ = self._build_local_artifact(circuit, None)
            return
        if self._runtime_mode_active():
            _ = self._build_runtime_artifact(circuit, None)

    def _run_local_estimator(self, circuit: QuantumCircuit, observable: SparsePauliOp) -> float:
        artifact = self._build_local_artifact(circuit, observable)
        mapped_observable = artifact.mapped_observable
        if mapped_observable is None:
            raise RuntimeError("Local noise artifact did not provide a mapped observable.")
        submitted_circuit = artifact.scheduled_circuit_or_none or artifact.transpiled_circuit
        return float(np.real(_run_estimator_job(self._estimator, submitted_circuit, mapped_observable)))

    def _run_runtime_estimator(self, circuit: QuantumCircuit, observable: SparsePauliOp) -> float:
        artifact = self._build_runtime_artifact(circuit, observable)
        mapped_observable = artifact.mapped_observable
        if mapped_observable is None:
            raise RuntimeError("Runtime submission artifact did not provide a mapped observable.")
        submitted_circuit = artifact.scheduled_circuit_or_none or artifact.transpiled_circuit
        return float(np.real(_run_estimator_job(self._estimator, submitted_circuit, mapped_observable)))

    def _run_local_measurement_counts(
        self,
        circuit: QuantumCircuit,
        *,
        repeat_idx: int,
    ) -> dict[str, int]:
        try:
            from qiskit_aer import AerSimulator
        except Exception as exc:
            raise RuntimeError("Counts-based local Aer measurement requires qiskit-aer.") from exc
        measured = circuit.copy()
        measured.measure_all()
        artifact = self._build_local_artifact(measured, None)
        seed = (
            self.resolved_noise_spec.seed_simulator
            if self.resolved_noise_spec.seed_simulator is not None
            else self.config.seed
        )
        sim_kwargs: dict[str, Any] = {
            "seed_simulator": int(seed) + int(repeat_idx),
        }
        if artifact.qiskit_noise_model is not None:
            sim_kwargs["noise_model"] = artifact.qiskit_noise_model
        sim = AerSimulator(**sim_kwargs)
        compiled = artifact.scheduled_circuit_or_none or artifact.transpiled_circuit
        result = sim.run(compiled, shots=int(self.config.shots)).result()
        counts = result.get_counts()
        if isinstance(counts, list):
            counts = counts[0]
        return dict(counts)

    def _maybe_evaluate_symmetry_mitigated(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
    ) -> OracleEstimate | None:
        symmetry_cfg = normalize_symmetry_mitigation_config(
            getattr(self.config, "symmetry_mitigation", "off")
        )
        requested_mode = str(symmetry_cfg.get("mode", "off"))
        details: dict[str, Any] = {
            "requested_mode": str(requested_mode),
            "applied_mode": str(requested_mode),
            "executed": False,
            "eligible": False,
            "fallback_reason": "",
            "retained_fraction_mean": None,
            "retained_fraction_samples": [],
            "sector_probability_mean": None,
            "sector_probability_samples": [],
            "sector_values": {
                "sector_n_up": symmetry_cfg.get("sector_n_up", None),
                "sector_n_dn": symmetry_cfg.get("sector_n_dn", None),
            },
            "estimator_form": "none",
        }
        if requested_mode in {"off", "verify_only"}:
            self._set_symmetry_mitigation_details(details)
            return None
        if not _observable_is_diagonal(observable):
            return self._handle_symmetry_downgrade(details, "observable_not_diagonal")
        if self.resolved_noise_spec.executor == "runtime_qpu":
            return self._handle_symmetry_downgrade(details, "runtime_counts_path_unavailable")
        required_keys = ("num_sites", "sector_n_up", "sector_n_dn")
        if any(symmetry_cfg.get(key, None) is None for key in required_keys):
            return self._handle_symmetry_downgrade(details, "incomplete_sector_config")
        vals: list[float] = []
        retained: list[float] = []
        repeats = max(1, int(self.config.oracle_repeats))
        try:
            for rep in range(repeats):
                if str(self.config.noise_mode) == "ideal":
                    if requested_mode == "projector_renorm_v1":
                        val, retained_fraction = _exact_projector_renorm_diagonal_expectation(
                            circuit,
                            observable,
                            symmetry_cfg,
                        )
                    else:
                        val, retained_fraction = _exact_postselected_diagonal_expectation(
                            circuit,
                            observable,
                            symmetry_cfg,
                        )
                else:
                    if self._local_mode_active():
                        counts = self._run_local_measurement_counts(circuit, repeat_idx=int(rep))
                    else:
                        counts = _sample_measurement_counts(circuit, self.config, repeat_idx=int(rep))
                    if requested_mode == "projector_renorm_v1":
                        val, retained_fraction = _projector_renorm_diagonal_expectation_from_counts(
                            counts,
                            observable,
                            n_qubits=int(circuit.num_qubits),
                            symmetry_cfg=symmetry_cfg,
                        )
                    else:
                        kept_counts, retained_fraction = _postselected_counts_and_fraction(
                            counts,
                            n_qubits=int(circuit.num_qubits),
                            symmetry_cfg=symmetry_cfg,
                        )
                        val = _diagonal_expectation_from_counts(kept_counts, observable)
                vals.append(float(val))
                retained.append(float(retained_fraction))
        except SymmetryMitigationDowngrade as exc:
            return self._handle_symmetry_downgrade(details, str(exc))
        except Exception as exc:
            if self._can_fallback_from_error(exc):
                self._activate_sampler_fallback(reason=str(exc), aer_failed=True)
                return self._handle_symmetry_downgrade(details, str(exc))
            raise
        arr = np.asarray(vals, dtype=float)
        stdev = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        stderr = float(stdev / np.sqrt(float(arr.size))) if arr.size > 0 else float("nan")
        agg = float(np.median(arr)) if self.config.oracle_aggregate == "median" else float(np.mean(arr))
        details.update(
            {
                "applied_mode": str(requested_mode),
                "executed": True,
                "eligible": True,
                "fallback_reason": "",
                "retained_fraction_mean": (float(np.mean(retained)) if retained else None),
                "retained_fraction_samples": [float(x) for x in retained],
                "sector_probability_mean": (float(np.mean(retained)) if retained else None),
                "sector_probability_samples": [float(x) for x in retained],
                "estimator_form": (
                    "postselected_bitstring_average"
                    if requested_mode == "postselect_diag_v1"
                    else "projector_ratio_diag_v1"
                ),
            }
        )
        self._set_symmetry_mitigation_details(details)
        return OracleEstimate(
            mean=agg,
            std=stdev,
            stdev=stdev,
            stderr=stderr,
            n_samples=int(arr.size),
            raw_values=[float(x) for x in arr.tolist()],
            aggregate=self.config.oracle_aggregate,
        )

    def _fallback_allowed_for_mode(self) -> bool:
        return (
            bool(self.resolved_noise_spec.executor == "aer")
            and bool(self.resolved_noise_spec.allow_noisy_fallback)
            and str(self.config.aer_fallback_mode) == "sampler_shots"
        )

    def _can_fallback_from_error(self, exc: Exception) -> bool:
        if not self._fallback_allowed_for_mode():
            return False
        return _looks_like_openmp_shm_abort(str(exc))

    def _activate_sampler_fallback(self, *, reason: str, aer_failed: bool) -> None:
        if self._sampler_fallback is None:
            try:
                from qiskit.primitives import StatevectorSampler
            except Exception as exc:
                raise RuntimeError(
                    "Failed to activate sampler fallback (`StatevectorSampler` unavailable)."
                ) from exc
            self._sampler_fallback = StatevectorSampler(
                default_shots=int(self.config.shots),
                seed=int(self.config.seed),
            )
        self._fallback_reason = str(reason)
        old = self.backend_info
        calibration_snapshot = None
        if self._local_context is not None:
            calibration_snapshot = self._local_context.get("calibration_snapshot", None)
        details = {
            "shots": (
                None
                if self.resolved_noise_spec.executor == "statevector"
                else int(self.config.shots)
            ),
            "mitigation": normalize_mitigation_config(getattr(self.config, "mitigation", "none")),
            "runtime_twirling": normalize_runtime_twirling_config(
                getattr(self.config, "runtime_twirling", None)
            ),
            "symmetry_mitigation": normalize_symmetry_mitigation_config(
                getattr(self.config, "symmetry_mitigation", "off")
            ),
            "resolved_noise_spec": resolved_noise_spec_to_dict(self.resolved_noise_spec),
            "resolved_noise_spec_hash": resolved_noise_spec_hash(self.resolved_noise_spec),
            "calibration_snapshot": calibration_snapshot_to_dict(calibration_snapshot),
            "snapshot_hash": (
                None if calibration_snapshot is None else calibration_snapshot.snapshot_hash
            ),
            "source_kind": (
                None if calibration_snapshot is None else calibration_snapshot.source_kind
            )
            or self.resolved_noise_spec.backend_profile_kind,
            "seed_transpiler": self.resolved_noise_spec.seed_transpiler,
            "seed_simulator": self.resolved_noise_spec.seed_simulator,
            "warnings": [],
            "omitted_channels": [],
        }
        details.update(dict(getattr(old, "details", {})))
        details["aer_failed"] = bool(aer_failed)
        details["fallback_used"] = True
        details["fallback_mode"] = str(self.config.aer_fallback_mode)
        details["fallback_reason"] = str(reason)
        details["env_workaround_applied"] = bool(self.config.omp_shm_workaround)
        details["warnings"] = list(details.get("warnings", [])) + [f"fallback:{str(reason)}"]
        self.backend_info = NoiseBackendInfo(
            noise_mode=str(self.config.noise_mode),
            estimator_kind="qiskit.primitives.StatevectorSampler(fallback)",
            backend_name=(
                old.backend_name
                or self.resolved_noise_spec.labels.get("backend_name")
                or "statevector_sampler_fallback"
            ),
            using_fake_backend=bool(
                old.using_fake_backend
                or self.resolved_noise_spec.backend_profile_kind == "fake_snapshot"
            ),
            details=details,
        )
        self._sync_execution_from_backend_info()
        self._estimator = None

    def evaluate_result(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
    ) -> OracleEvaluationResult:
        if self._closed:
            raise RuntimeError("ExpectationOracle is closed.")

        symmetry_est = self._maybe_evaluate_symmetry_mitigated(circuit, observable)
        if symmetry_est is not None:
            stored = OracleEvaluationResult(
                estimate=_copy_oracle_estimate(symmetry_est),
                execution=_copy_execution_record(self._current_execution),
            )
            self._last_result = stored
            return _copy_evaluation_result(stored)

        vals: list[float] = []
        repeats = max(1, int(self.config.oracle_repeats))
        for _ in range(repeats):
            if self._sampler_fallback is not None:
                val = _run_sampler_fallback_job(self._sampler_fallback, circuit, observable)
                vals.append(float(np.real(val)))
                continue
            try:
                if self._local_mode_active():
                    val = self._run_local_estimator(circuit, observable)
                elif self._runtime_mode_active():
                    val = self._run_runtime_estimator(circuit, observable)
                else:
                    val = _run_estimator_job(self._estimator, circuit, observable)
                vals.append(float(np.real(val)))
            except Exception as exc:
                if self._can_fallback_from_error(exc):
                    self._activate_sampler_fallback(reason=str(exc), aer_failed=True)
                    val = _run_sampler_fallback_job(self._sampler_fallback, circuit, observable)
                    vals.append(float(np.real(val)))
                else:
                    raise

        arr = np.asarray(vals, dtype=float)
        stdev = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        stderr = float(stdev / np.sqrt(float(arr.size))) if arr.size > 0 else float("nan")
        if self.config.oracle_aggregate == "median":
            agg = float(np.median(arr))
        else:
            agg = float(np.mean(arr))

        stored = OracleEvaluationResult(
            estimate=OracleEstimate(
                mean=agg,
                std=stdev,
                stdev=stdev,
                stderr=stderr,
                n_samples=int(arr.size),
                raw_values=[float(x) for x in arr.tolist()],
                aggregate=self.config.oracle_aggregate,
            ),
            execution=_copy_execution_record(self._current_execution),
        )
        self._last_result = stored
        return _copy_evaluation_result(stored)

    def evaluate(self, circuit: QuantumCircuit, observable: SparsePauliOp) -> OracleEstimate:
        return self.evaluate_result(circuit, observable).estimate


_NUMBER_OPERATOR_MATH = "n_p = (I - Z_p) / 2"


def _number_operator_qop(num_qubits: int, index: int) -> SparsePauliOp:
    if index < 0 or index >= int(num_qubits):
        raise ValueError(f"index {index} out of range for num_qubits={num_qubits}")
    chars = ["I"] * int(num_qubits)
    chars[int(num_qubits) - 1 - int(index)] = "Z"
    z_label = "".join(chars)
    return SparsePauliOp.from_list(
        [
            ("I" * int(num_qubits), 0.5),
            (z_label, -0.5),
        ]
    ).simplify(atol=1e-12)


_DOUBLON_OPERATOR_MATH = "D_i = n_{i,up} n_{i,dn} = (I - Z_up - Z_dn + Z_up Z_dn) / 4"


def _doublon_site_qop(num_qubits: int, up_index: int, dn_index: int) -> SparsePauliOp:
    if up_index == dn_index:
        raise ValueError("up_index and dn_index must differ")
    chars_up = ["I"] * int(num_qubits)
    chars_dn = ["I"] * int(num_qubits)
    chars_both = ["I"] * int(num_qubits)
    chars_up[int(num_qubits) - 1 - int(up_index)] = "Z"
    chars_dn[int(num_qubits) - 1 - int(dn_index)] = "Z"
    chars_both[int(num_qubits) - 1 - int(up_index)] = "Z"
    chars_both[int(num_qubits) - 1 - int(dn_index)] = "Z"
    return SparsePauliOp.from_list(
        [
            ("I" * int(num_qubits), 0.25),
            ("".join(chars_up), -0.25),
            ("".join(chars_dn), -0.25),
            ("".join(chars_both), 0.25),
        ]
    ).simplify(atol=1e-12)


def _ordered_qop_from_exyz(
    ordered_labels_exyz: Sequence[str],
    coeff_map_exyz: dict[str, complex],
    *,
    tol: float = 1e-12,
) -> SparsePauliOp:
    terms: list[tuple[str, complex]] = []
    for lbl in ordered_labels_exyz:
        coeff = complex(coeff_map_exyz[lbl])
        if abs(coeff) <= float(tol):
            continue
        terms.append((_to_ixyz(lbl), coeff))
    if not terms:
        nq = len(ordered_labels_exyz[0]) if ordered_labels_exyz else 1
        terms = [("I" * nq, 0.0 + 0.0j)]
    return SparsePauliOp.from_list(terms).simplify(atol=float(tol))
