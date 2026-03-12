#!/usr/bin/env python3
"""Deterministic backend-patch selection for offline exact-bench noise replay."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from pipelines.exact_bench import noise_aer_builders as nab
from pipelines.exact_bench.noise_model_spec import (
    CalibrationSnapshot,
    ResolvedNoiseSpec,
    TranspileSnapshot,
    stable_noise_hash,
)

_SELECTOR_VERSION = "canonical_patch_v1"
_PATCH_SELECTION_OBJECTIVE = [
    "min_extra_2q_count",
    "min_used_edge_2q_error_exposure",
    "min_relaxation_exposure",
    "min_readout_burden",
]


@dataclass(frozen=True)
class PatchSelectionScore:
    extra_2q_count: int
    used_edge_2q_error_exposure: float
    relaxation_exposure: float
    readout_burden: float

    def lexicographic_key(self) -> tuple[int, float, float, float]:
        return (
            int(self.extra_2q_count),
            round(float(self.used_edge_2q_error_exposure), 15),
            round(float(self.relaxation_exposure), 15),
            round(float(self.readout_burden), 15),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "extra_2q_count": int(self.extra_2q_count),
            "used_edge_2q_error_exposure": float(self.used_edge_2q_error_exposure),
            "relaxation_exposure": float(self.relaxation_exposure),
            "readout_burden": float(self.readout_burden),
        }


@dataclass(frozen=True)
class PatchSelectionResult:
    selector_version: str
    logical_qubits: int
    selected_physical_qubits: list[int]
    selected_edges: list[list[int]]
    score: PatchSelectionScore
    candidate_count: int
    anchor_circuit_index: int
    anchor_circuit_hash: str
    representative_circuit_hashes: list[str]
    anchor_circuit_structure_hash: str
    anchor_layout_hash: str
    anchor_transpile_snapshot: TranspileSnapshot
    schedule_policy: str
    optimization_level: int
    seed_transpiler: int | None
    snapshot_hash: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "selector_version": str(self.selector_version),
            "patch_selection_objective": list(_PATCH_SELECTION_OBJECTIVE),
            "logical_qubits": int(self.logical_qubits),
            "candidate_count": int(self.candidate_count),
            "selected_physical_qubits": [int(q) for q in list(self.selected_physical_qubits)],
            "selected_edges": [[int(a), int(b)] for a, b in [tuple(edge) for edge in list(self.selected_edges)]],
            "patch_score_summary": self.score.to_dict(),
            "anchor_circuit_index": int(self.anchor_circuit_index),
            "anchor_circuit_hash": str(self.anchor_circuit_hash),
            "representative_circuit_hashes": [str(x) for x in list(self.representative_circuit_hashes)],
            "anchor_circuit_structure_hash": str(self.anchor_circuit_structure_hash),
            "anchor_layout_hash": str(self.anchor_layout_hash),
            "anchor_transpile_hash": str(self.anchor_transpile_snapshot.transpile_hash),
            "anchor_used_physical_qubits": [int(q) for q in list(self.anchor_transpile_snapshot.used_physical_qubits)],
            "anchor_used_physical_edges": [
                [int(edge[0]), int(edge[1])] for edge in list(self.anchor_transpile_snapshot.used_physical_edges)
            ],
            "anchor_scheduled_duration_total": self.anchor_transpile_snapshot.scheduled_duration_total,
            "anchor_idle_duration_total": self.anchor_transpile_snapshot.idle_duration_total,
            "schedule_policy": str(self.schedule_policy),
            "optimization_level": int(self.optimization_level),
            "seed_transpiler": self.seed_transpiler,
            "snapshot_hash": str(self.snapshot_hash),
        }

    def to_layout_lock_payload(self) -> dict[str, Any]:
        summary = self.to_summary()
        return {
            "selector_version": str(self.selector_version),
            "selected_patch_qubits": [int(q) for q in list(self.selected_physical_qubits)],
            "selected_patch_edges": [[int(edge[0]), int(edge[1])] for edge in list(self.selected_edges)],
            "anchor_circuit_hash": str(self.anchor_circuit_hash),
            "representative_circuit_hashes": [str(x) for x in list(self.representative_circuit_hashes)],
            "anchor_circuit_structure_hash": str(self.anchor_circuit_structure_hash),
            "anchor_layout_hash": str(self.anchor_layout_hash),
            "anchor_transpile_hash": str(self.anchor_transpile_snapshot.transpile_hash),
            "patch_selection_summary": summary,
        }


@dataclass(frozen=True)
class _CircuitCandidateMetrics:
    circuit_index: int
    circuit_hash: str
    circuit_structure_hash: str
    original_twoq_count: int
    transpile_snapshot: TranspileSnapshot
    layout_hash: str
    extra_2q_count: int
    used_edge_2q_error_exposure: float
    relaxation_exposure: float
    readout_burden: float


r"""P = {S \subseteq V : |S| = n, G[S] connected}."""
def enumerate_connected_patch_candidates(
    calibration_snapshot: CalibrationSnapshot,
    logical_qubits: int,
) -> list[list[int]]:
    logical = int(logical_qubits)
    if logical <= 0:
        raise ValueError("logical_qubits must be >= 1 for patch selection.")
    adjacency = _undirected_adjacency(calibration_snapshot)
    nodes = sorted(adjacency)
    if logical > len(nodes):
        raise ValueError(
            f"logical_qubits={logical} exceeds available snapshot qubits={len(nodes)} for patch selection."
        )
    if logical == 1:
        return [[int(node)] for node in nodes]

    seen: set[tuple[int, ...]] = set()
    candidates: list[list[int]] = []

    def _expand(current: tuple[int, ...], frontier: set[int]) -> None:
        if len(current) == logical:
            if current not in seen:
                seen.add(current)
                candidates.append([int(q) for q in current])
            return
        for nxt in sorted(frontier):
            new_current_set = set(current)
            new_current_set.add(int(nxt))
            new_current = tuple(sorted(new_current_set))
            new_frontier = set(frontier)
            new_frontier.discard(int(nxt))
            new_frontier.update(
                int(neighbor)
                for neighbor in adjacency.get(int(nxt), set())
                if int(neighbor) not in new_current_set
            )
            _expand(new_current, new_frontier)

    for seed in nodes:
        _expand((int(seed),), set(int(n) for n in adjacency.get(int(seed), set())))

    candidates.sort()
    return candidates


r"""score(S) = (Δ2Q(S), ε_2Q(S), ρ_relax(S), ε_ro(S))."""
def score_patch_candidate(
    calibration_snapshot: CalibrationSnapshot,
    representative_circuits: Sequence[QuantumCircuit],
    physical_qubits: Sequence[int],
    *,
    schedule_policy: str = "asap",
    seed_transpiler: int | None = None,
    optimization_level: int = 1,
) -> PatchSelectionResult:
    circuits = _normalized_circuit_list(representative_circuits)
    logical_qubits = _shared_logical_qubits(circuits)
    candidate_qubits = [int(q) for q in list(physical_qubits)]
    if len(candidate_qubits) != logical_qubits:
        raise ValueError(
            f"Candidate patch size {len(candidate_qubits)} must equal logical qubit count {logical_qubits}."
        )

    restricted_backend, selected_edges = _build_candidate_backend(calibration_snapshot, candidate_qubits)
    resolved_spec = _selection_resolved_spec(
        schedule_policy=str(schedule_policy),
        seed_transpiler=seed_transpiler,
    )
    exact_gate_errors, edge_fallback_errors = _twoq_error_maps(calibration_snapshot)
    qubit_records = {
        int(rec.physical_qubit): rec for rec in list(calibration_snapshot.per_qubit or [])
    }

    circuit_metrics: list[_CircuitCandidateMetrics] = []
    total_extra_2q = 0
    total_used_edge_error = 0.0
    total_relaxation = 0.0
    total_readout = 0.0

    for index, circuit in enumerate(circuits):
        transpiled = transpile(
            circuit,
            backend=restricted_backend,
            optimization_level=int(optimization_level),
            seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
            scheduling_method=("asap" if str(schedule_policy) == "asap" else None),
        )
        transpile_snapshot, layout_hash = nab._build_transpile_snapshot(
            transpiled,
            logical_qubits=int(circuit.num_qubits),
            resolved_spec=resolved_spec,
            calibration_snapshot=calibration_snapshot,
            optimization_level=int(optimization_level),
        )
        original_twoq_count = _count_twoq_ops(circuit)
        extra_2q_count = max(0, int(transpile_snapshot.count_2q) - int(original_twoq_count))
        used_edge_error = _used_edge_2q_error_exposure(
            transpiled,
            exact_gate_errors=exact_gate_errors,
            edge_fallback_errors=edge_fallback_errors,
            median_2q_error=calibration_snapshot.median_2q_error,
        )
        relaxation_exposure = _relaxation_exposure(
            transpile_snapshot,
            qubit_records=qubit_records,
            median_t1=calibration_snapshot.median_T1,
            median_t2=calibration_snapshot.median_T2,
        )
        readout_burden = _readout_burden(
            transpiled,
            qubit_records=qubit_records,
            median_readout_error=calibration_snapshot.median_readout_error,
        )
        circuit_hash = _circuit_hash(circuit)
        circuit_structure_hash = nab._circuit_structure_template(circuit)[3]
        metrics = _CircuitCandidateMetrics(
            circuit_index=int(index),
            circuit_hash=str(circuit_hash),
            circuit_structure_hash=str(circuit_structure_hash),
            original_twoq_count=int(original_twoq_count),
            transpile_snapshot=transpile_snapshot,
            layout_hash=str(layout_hash),
            extra_2q_count=int(extra_2q_count),
            used_edge_2q_error_exposure=float(used_edge_error),
            relaxation_exposure=float(relaxation_exposure),
            readout_burden=float(readout_burden),
        )
        circuit_metrics.append(metrics)
        total_extra_2q += int(extra_2q_count)
        total_used_edge_error += float(used_edge_error)
        total_relaxation += float(relaxation_exposure)
        total_readout += float(readout_burden)

    anchor_metrics = max(
        circuit_metrics,
        key=lambda record: (
            int(record.original_twoq_count),
            int(record.transpile_snapshot.count_2q),
            int(record.transpile_snapshot.depth),
            -int(record.circuit_index),
        ),
    )
    score = PatchSelectionScore(
        extra_2q_count=int(total_extra_2q),
        used_edge_2q_error_exposure=float(total_used_edge_error),
        relaxation_exposure=float(total_relaxation),
        readout_burden=float(total_readout),
    )
    return PatchSelectionResult(
        selector_version=_SELECTOR_VERSION,
        logical_qubits=int(logical_qubits),
        selected_physical_qubits=[int(q) for q in candidate_qubits],
        selected_edges=[[int(edge[0]), int(edge[1])] for edge in list(selected_edges)],
        score=score,
        candidate_count=1,
        anchor_circuit_index=int(anchor_metrics.circuit_index),
        anchor_circuit_hash=str(anchor_metrics.circuit_hash),
        representative_circuit_hashes=[str(record.circuit_hash) for record in circuit_metrics],
        anchor_circuit_structure_hash=str(anchor_metrics.circuit_structure_hash),
        anchor_layout_hash=str(anchor_metrics.layout_hash),
        anchor_transpile_snapshot=anchor_metrics.transpile_snapshot,
        schedule_policy=str(schedule_policy),
        optimization_level=int(optimization_level),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        snapshot_hash=str(calibration_snapshot.snapshot_hash),
    )


r"""S* = argmin_{S in P} score(S)."""
def select_canonical_patch(
    calibration_snapshot: CalibrationSnapshot,
    representative_circuits: Sequence[QuantumCircuit],
    *,
    schedule_policy: str = "asap",
    seed_transpiler: int | None = None,
    optimization_level: int = 1,
    candidate_patches: Sequence[Sequence[int]] | None = None,
) -> PatchSelectionResult:
    circuits = _normalized_circuit_list(representative_circuits)
    logical_qubits = _shared_logical_qubits(circuits)
    candidates = (
        [[int(q) for q in list(candidate)] for candidate in list(candidate_patches)]
        if candidate_patches is not None
        else enumerate_connected_patch_candidates(calibration_snapshot, logical_qubits)
    )
    if not candidates:
        raise RuntimeError("Patch selection found no connected candidate patches.")

    evaluations: list[PatchSelectionResult] = []
    for candidate in candidates:
        evaluation = score_patch_candidate(
            calibration_snapshot,
            circuits,
            candidate,
            schedule_policy=str(schedule_policy),
            seed_transpiler=seed_transpiler,
            optimization_level=int(optimization_level),
        )
        evaluations.append(evaluation)

    winner = min(
        evaluations,
        key=lambda result: (
            result.score.lexicographic_key(),
            tuple(int(q) for q in list(result.selected_physical_qubits)),
            tuple(tuple(int(v) for v in list(edge)) for edge in list(result.selected_edges)),
            str(result.anchor_circuit_hash),
        ),
    )
    return PatchSelectionResult(
        selector_version=str(winner.selector_version),
        logical_qubits=int(winner.logical_qubits),
        selected_physical_qubits=[int(q) for q in list(winner.selected_physical_qubits)],
        selected_edges=[[int(edge[0]), int(edge[1])] for edge in list(winner.selected_edges)],
        score=winner.score,
        candidate_count=int(len(evaluations)),
        anchor_circuit_index=int(winner.anchor_circuit_index),
        anchor_circuit_hash=str(winner.anchor_circuit_hash),
        representative_circuit_hashes=[str(x) for x in list(winner.representative_circuit_hashes)],
        anchor_circuit_structure_hash=str(winner.anchor_circuit_structure_hash),
        anchor_layout_hash=str(winner.anchor_layout_hash),
        anchor_transpile_snapshot=winner.anchor_transpile_snapshot,
        schedule_policy=str(winner.schedule_policy),
        optimization_level=int(winner.optimization_level),
        seed_transpiler=winner.seed_transpiler,
        snapshot_hash=str(winner.snapshot_hash),
    )


r"""R = {selector, patch, score, anchor, provenance}."""
def build_patch_selection_summary(result: PatchSelectionResult) -> dict[str, Any]:
    return result.to_summary()



def _normalized_circuit_list(representative_circuits: Sequence[QuantumCircuit]) -> list[QuantumCircuit]:
    circuits = [circuit for circuit in list(representative_circuits) if circuit is not None]
    if not circuits:
        raise ValueError("Patch selection requires at least one representative circuit.")
    keyed = [(_circuit_hash(circuit), idx, circuit) for idx, circuit in enumerate(circuits)]
    keyed.sort(key=lambda row: (str(row[0]), int(row[1])))
    return [circuit for _hash, _idx, circuit in keyed]



def _shared_logical_qubits(circuits: Sequence[QuantumCircuit]) -> int:
    counts = {int(circuit.num_qubits) for circuit in list(circuits)}
    if len(counts) != 1:
        raise ValueError("Representative circuits must share the same logical qubit count for patch selection.")
    return int(next(iter(counts)))



def _selection_resolved_spec(
    *,
    schedule_policy: str,
    seed_transpiler: int | None,
) -> ResolvedNoiseSpec:
    return ResolvedNoiseSpec(
        executor="aer",
        noise_kind=("backend_scheduled" if str(schedule_policy) == "asap" else "backend_basic"),
        backend_profile_kind="frozen_snapshot_json",
        mitigation_bundle="none",
        layout_policy="auto_then_lock",
        schedule_policy=str(schedule_policy),
        shots=None,
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        seed_simulator=None,
        allow_noisy_fallback=False,
    )



def _undirected_adjacency(calibration_snapshot: CalibrationSnapshot) -> dict[int, set[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    if calibration_snapshot.per_qubit:
        for record in list(calibration_snapshot.per_qubit):
            out[int(record.physical_qubit)]
    for edge in list(calibration_snapshot.coupling_map or []):
        if len(list(edge)) < 2:
            continue
        left = int(edge[0])
        right = int(edge[1])
        out[left].add(right)
        out[right].add(left)
    for record in list(calibration_snapshot.per_gate or []):
        qubits = [int(q) for q in list(record.qubits or [])]
        if len(qubits) != 2:
            continue
        out[int(qubits[0])].add(int(qubits[1]))
        out[int(qubits[1])].add(int(qubits[0]))
    return {int(key): set(int(v) for v in value) for key, value in out.items()}



def _candidate_edges(
    calibration_snapshot: CalibrationSnapshot,
    physical_qubits: Sequence[int],
) -> list[list[int]]:
    allowed = {int(q) for q in list(physical_qubits)}
    raw_edges: list[list[int]] = [
        [int(edge[0]), int(edge[1])]
        for edge in list(calibration_snapshot.coupling_map or [])
        if len(list(edge)) >= 2 and int(edge[0]) in allowed and int(edge[1]) in allowed
    ]
    raw_edges.extend(
        [int(qubits[0]), int(qubits[1])]
        for qubits in (
            [int(q) for q in list(record.qubits or [])]
            for record in list(calibration_snapshot.per_gate or [])
        )
        if len(qubits) == 2 and int(qubits[0]) in allowed and int(qubits[1]) in allowed
    )
    return nab._canonical_edge_payload(raw_edges)



def _build_candidate_backend(
    calibration_snapshot: CalibrationSnapshot,
    physical_qubits: Sequence[int],
):
    base_backend, _warnings = nab.build_backend_from_calibration_snapshot(calibration_snapshot)
    base_target = getattr(base_backend, "target", None)
    if base_target is None:
        raise RuntimeError("Patch selection requires a backend-like target reconstructed from the calibration snapshot.")
    allowed_qubits = {int(q) for q in list(physical_qubits)}
    selected_edges = _candidate_edges(calibration_snapshot, physical_qubits)
    filtered_target, supported_twoq_edges = nab._filter_target_to_subgraph(
        base_target,
        allowed_qubits=allowed_qubits,
        allowed_edges={tuple(int(v) for v in list(edge)[:2]) for edge in list(selected_edges)},
    )
    filtered_coupling_map = (
        CouplingMap([[int(a), int(b)] for a, b in sorted(supported_twoq_edges)])
        if supported_twoq_edges
        else None
    )
    return (
        nab._RestrictedBackendV2(
            base_backend,
            target=filtered_target,
            coupling_map=filtered_coupling_map,
        ),
        [[int(a), int(b)] for a, b in sorted(supported_twoq_edges)],
    )



def _count_twoq_ops(circuit: QuantumCircuit) -> int:
    total = 0
    for inst in list(circuit.data):
        if str(inst.operation.name) in {"measure", "delay", "barrier"}:
            continue
        if len(list(inst.qubits)) == 2:
            total += 1
    return int(total)



def _circuit_hash(circuit: QuantumCircuit) -> str:
    return stable_noise_hash(
        {
            "num_qubits": int(circuit.num_qubits),
            "circuit_structure_hash": nab._circuit_structure_template(circuit)[3],
        },
        prefix="pcirc",
    )



def _twoq_error_maps(
    calibration_snapshot: CalibrationSnapshot,
) -> tuple[dict[tuple[str, tuple[int, int]], float], dict[tuple[int, int], float]]:
    exact: dict[tuple[str, tuple[int, int]], float] = {}
    by_edge: dict[tuple[int, int], float] = {}
    for record in list(calibration_snapshot.per_gate or []):
        qubits = tuple(int(q) for q in list(record.qubits or []))
        if len(qubits) != 2 or record.error is None:
            continue
        exact[(str(record.gate_name), qubits)] = float(record.error)
        by_edge[qubits] = float(record.error)
    return exact, by_edge



def _physical_edge_counts(circuit: QuantumCircuit) -> dict[tuple[str, tuple[int, int]], int]:
    physical_map = nab._physical_qubit_map(circuit)
    counts: dict[tuple[str, tuple[int, int]], int] = defaultdict(int)
    for inst in list(circuit.data):
        if str(inst.operation.name) in {"measure", "delay", "barrier"}:
            continue
        qidx = [int(physical_map[q]) for q in list(inst.qubits)]
        if len(qidx) != 2:
            continue
        counts[(str(inst.operation.name), (int(qidx[0]), int(qidx[1])))] += 1
    return dict(counts)



def _used_edge_2q_error_exposure(
    circuit: QuantumCircuit,
    *,
    exact_gate_errors: dict[tuple[str, tuple[int, int]], float],
    edge_fallback_errors: dict[tuple[int, int], float],
    median_2q_error: float | None,
) -> float:
    total = 0.0
    for (gate_name, edge), count in _physical_edge_counts(circuit).items():
        error = exact_gate_errors.get((str(gate_name), tuple(edge)))
        if error is None:
            reversed_edge = (int(edge[1]), int(edge[0]))
            error = exact_gate_errors.get((str(gate_name), reversed_edge))
        if error is None:
            error = edge_fallback_errors.get(tuple(edge))
        if error is None:
            reversed_edge = (int(edge[1]), int(edge[0]))
            error = edge_fallback_errors.get(reversed_edge)
        if error is None:
            error = median_2q_error
        if error is not None:
            total += float(count) * float(error)
    return float(total)



def _safe_ratio(numerator: float, denominator: float | None) -> float:
    if denominator in {None, 0.0}:
        return 0.0
    return float(numerator) / float(denominator)



def _relaxation_exposure(
    transpile_snapshot: TranspileSnapshot,
    *,
    qubit_records: dict[int, Any],
    median_t1: float | None,
    median_t2: float | None,
) -> float:
    scheduled_total = float(transpile_snapshot.scheduled_duration_total or 0.0)
    idle_total = float(transpile_snapshot.idle_duration_total or 0.0)
    total = 0.0
    for qubit in list(transpile_snapshot.used_physical_qubits or []):
        record = qubit_records.get(int(qubit))
        t1 = getattr(record, "T1_s", None) if record is not None else None
        t2 = getattr(record, "T2_s", None) if record is not None else None
        total += _safe_ratio(scheduled_total, t1 if t1 not in {None, 0.0} else median_t1)
        total += _safe_ratio(idle_total, t2 if t2 not in {None, 0.0} else median_t2)
    return float(total)



def _readout_burden(
    circuit: QuantumCircuit,
    *,
    qubit_records: dict[int, Any],
    median_readout_error: float | None,
) -> float:
    physical_map = nab._physical_qubit_map(circuit)
    total = 0.0
    for inst in list(circuit.data):
        if str(inst.operation.name) != "measure":
            continue
        qubit = int(physical_map[inst.qubits[0]])
        record = qubit_records.get(int(qubit))
        error = getattr(record, "readout_error", None) if record is not None else None
        if error is None:
            error = median_readout_error
        if error is not None:
            total += float(error)
    return float(total)
