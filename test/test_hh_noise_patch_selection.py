from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench import noise_aer_builders as nab
from pipelines.exact_bench.noise_aer_builders import (
    build_backend_from_calibration_snapshot,
    build_patch_snapshot_artifact,
    describe_layout_anchor_source,
    select_representative_patch,
)
from pipelines.exact_bench.noise_model_spec import (
    CalibrationGateRecord,
    CalibrationQubitRecord,
    CalibrationSnapshot,
    canonical_calibration_snapshot_payload,
    normalize_to_resolved_noise_spec,
    stable_noise_hash,
)
from pipelines.exact_bench.noise_patch_selection import (
    build_patch_selection_summary,
    enumerate_connected_patch_candidates,
    score_patch_candidate,
    select_canonical_patch,
)
from pipelines.exact_bench.noise_snapshot import (
    load_calibration_snapshot,
    load_snapshot_layout_lock,
    write_calibration_snapshot,
)


def _snapshot_hash(snapshot: CalibrationSnapshot) -> str:
    return stable_noise_hash(canonical_calibration_snapshot_payload(snapshot), prefix="csnap")



def _make_snapshot(
    *,
    num_qubits: int,
    undirected_edges: list[tuple[int, int]],
    twoq_errors: dict[tuple[int, int], float],
    readout_errors: dict[int, float],
    t1s: dict[int, float],
    t2s: dict[int, float],
) -> CalibrationSnapshot:
    coupling_map: list[list[int]] = []
    for left, right in undirected_edges:
        coupling_map.append([int(left), int(right)])
        coupling_map.append([int(right), int(left)])

    per_qubit = [
        CalibrationQubitRecord(
            physical_qubit=int(q),
            T1_s=float(t1s.get(q, 120e-6)),
            T2_s=float(t2s.get(q, 90e-6)),
            readout_error=float(readout_errors.get(q, 0.02)),
            readout_p01=None,
            readout_p10=None,
            measure_duration_s=3.0e-7,
            frequency=5.0e9,
        )
        for q in range(int(num_qubits))
    ]
    per_gate: list[CalibrationGateRecord] = []
    for q in range(int(num_qubits)):
        per_gate.extend(
            [
                CalibrationGateRecord("rz", [int(q)], 1.0e-4, 2.0e-8),
                CalibrationGateRecord("sx", [int(q)], 1.2e-4, 4.0e-8),
                CalibrationGateRecord("x", [int(q)], 1.4e-4, 5.0e-8),
            ]
        )
    for left, right in coupling_map:
        per_gate.append(
            CalibrationGateRecord(
                "cx",
                [int(left), int(right)],
                float(twoq_errors.get((int(left), int(right)), 0.02)),
                3.5e-7,
            )
        )
    snapshot = CalibrationSnapshot(
        source_kind="frozen_snapshot_json",
        backend_name="test_patch_backend",
        backend_version="1.0",
        processor_family="test_family",
        retrieved_at_utc="2026-03-11T00:00:00Z",
        calibration_time_utc="2026-03-10T00:00:00Z",
        basis_gates=["rz", "sx", "x", "cx", "measure", "delay", "reset"],
        coupling_map=coupling_map,
        dt=1.0e-9,
        per_qubit=per_qubit,
        per_gate=per_gate,
        median_1q_error=1.2e-4,
        median_2q_error=0.02,
        median_readout_error=0.02,
        median_T1=120e-6,
        median_T2=90e-6,
        snapshot_hash="",
    )
    return replace(snapshot, snapshot_hash=_snapshot_hash(snapshot))



def test_connected_candidate_patches_are_enumerated_deterministically() -> None:
    snapshot = _make_snapshot(
        num_qubits=4,
        undirected_edges=[(0, 1), (1, 2), (2, 3)],
        twoq_errors={},
        readout_errors={},
        t1s={},
        t2s={},
    )

    first = enumerate_connected_patch_candidates(snapshot, logical_qubits=2)
    second = enumerate_connected_patch_candidates(snapshot, logical_qubits=2)

    assert first == [[0, 1], [1, 2], [2, 3]]
    assert second == first



def test_selector_uses_per_gate_connectivity_when_coupling_map_is_empty() -> None:
    snapshot = _make_snapshot(
        num_qubits=2,
        undirected_edges=[(0, 1)],
        twoq_errors={(0, 1): 0.01, (1, 0): 0.01},
        readout_errors={},
        t1s={},
        t2s={},
    )
    per_gate_only = replace(snapshot, coupling_map=[], snapshot_hash="")
    per_gate_only = replace(per_gate_only, snapshot_hash=_snapshot_hash(per_gate_only))

    assert enumerate_connected_patch_candidates(per_gate_only, logical_qubits=2) == [[0, 1]]



def test_selector_prefers_zero_routing_patch_over_lower_error_routed_candidate() -> None:
    snapshot = _make_snapshot(
        num_qubits=4,
        undirected_edges=[(0, 1), (1, 2), (0, 2), (1, 2), (2, 3)],
        twoq_errors={
            (0, 1): 0.08,
            (1, 0): 0.08,
            (1, 2): 0.08,
            (2, 1): 0.08,
            (0, 2): 0.08,
            (2, 0): 0.08,
            (2, 3): 1.0e-4,
            (3, 2): 1.0e-4,
        },
        readout_errors={},
        t1s={},
        t2s={},
    )
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 2)
    qc.measure_all()

    triangle = score_patch_candidate(snapshot, [qc], [0, 1, 2], schedule_policy="asap")
    routed = score_patch_candidate(snapshot, [qc], [1, 2, 3], schedule_policy="asap")
    selected = select_canonical_patch(
        snapshot,
        [qc],
        schedule_policy="asap",
        candidate_patches=[[0, 1, 2], [1, 2, 3]],
    )

    assert triangle.score.extra_2q_count == 0
    assert routed.score.extra_2q_count > triangle.score.extra_2q_count
    assert selected.selected_physical_qubits == [0, 1, 2]



def test_selector_prefers_lower_2q_error_among_routing_equivalent_patches() -> None:
    snapshot = _make_snapshot(
        num_qubits=4,
        undirected_edges=[(0, 1), (2, 3)],
        twoq_errors={
            (0, 1): 0.07,
            (1, 0): 0.07,
            (2, 3): 0.002,
            (3, 2): 0.002,
        },
        readout_errors={},
        t1s={},
        t2s={},
    )
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.measure_all()

    high_error = score_patch_candidate(snapshot, [qc], [0, 1], schedule_policy="asap")
    low_error = score_patch_candidate(snapshot, [qc], [2, 3], schedule_policy="asap")
    selected = select_canonical_patch(
        snapshot,
        [qc],
        schedule_policy="asap",
        candidate_patches=[[0, 1], [2, 3]],
    )

    assert high_error.score.extra_2q_count == low_error.score.extra_2q_count
    assert low_error.score.used_edge_2q_error_exposure < high_error.score.used_edge_2q_error_exposure
    assert selected.selected_physical_qubits == [2, 3]



def test_selector_uses_relaxation_tiebreak_before_readout() -> None:
    snapshot = _make_snapshot(
        num_qubits=4,
        undirected_edges=[(0, 1), (2, 3)],
        twoq_errors={(0, 1): 0.01, (1, 0): 0.01, (2, 3): 0.01, (3, 2): 0.01},
        readout_errors={0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01},
        t1s={0: 20e-6, 1: 20e-6, 2: 250e-6, 3: 250e-6},
        t2s={0: 15e-6, 1: 15e-6, 2: 200e-6, 3: 200e-6},
    )
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.measure_all()

    selected = select_canonical_patch(
        snapshot,
        [qc],
        schedule_policy="asap",
        candidate_patches=[[0, 1], [2, 3]],
    )

    assert selected.selected_physical_qubits == [2, 3]



def test_selector_uses_readout_burden_as_final_tiebreaker() -> None:
    snapshot = _make_snapshot(
        num_qubits=4,
        undirected_edges=[(0, 1), (2, 3)],
        twoq_errors={(0, 1): 0.01, (1, 0): 0.01, (2, 3): 0.01, (3, 2): 0.01},
        readout_errors={0: 0.08, 1: 0.08, 2: 0.01, 3: 0.01},
        t1s={0: 200e-6, 1: 200e-6, 2: 200e-6, 3: 200e-6},
        t2s={0: 160e-6, 1: 160e-6, 2: 160e-6, 3: 160e-6},
    )
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.measure_all()

    selected = select_canonical_patch(
        snapshot,
        [qc],
        schedule_policy="asap",
        candidate_patches=[[0, 1], [2, 3]],
    )

    assert selected.selected_physical_qubits == [2, 3]



def test_selector_is_order_independent_for_representative_circuit_sets() -> None:
    snapshot = _make_snapshot(
        num_qubits=2,
        undirected_edges=[(0, 1)],
        twoq_errors={(0, 1): 0.01, (1, 0): 0.01},
        readout_errors={},
        t1s={},
        t2s={},
    )
    qc_one = QuantumCircuit(2)
    qc_one.cx(0, 1)
    qc_one.measure_all()
    qc_two = QuantumCircuit(2)
    qc_two.x(0)
    qc_two.cx(0, 1)
    qc_two.measure_all()

    ordered = select_canonical_patch(snapshot, [qc_one, qc_two], candidate_patches=[[0, 1]])
    reversed_order = select_canonical_patch(snapshot, [qc_two, qc_one], candidate_patches=[[0, 1]])

    assert ordered.anchor_circuit_hash == reversed_order.anchor_circuit_hash
    assert ordered.anchor_layout_hash == reversed_order.anchor_layout_hash
    assert ordered.representative_circuit_hashes == reversed_order.representative_circuit_hashes



def test_persisted_selection_summary_contains_replay_provenance(tmp_path: Path) -> None:
    snapshot = _make_snapshot(
        num_qubits=2,
        undirected_edges=[(0, 1)],
        twoq_errors={(0, 1): 0.01, (1, 0): 0.01},
        readout_errors={},
        t1s={},
        t2s={},
    )
    snapshot_path = tmp_path / "selection_bundle_snapshot.json"
    write_calibration_snapshot(snapshot_path, snapshot)

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.measure_all()
    spec = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "backend_scheduled",
            "backend_profile": "frozen_snapshot_json",
            "noise_snapshot_json": str(snapshot_path),
            "schedule_policy": "asap",
            "layout_lock_key": "patch_selector_bundle_lock",
            "shots": 128,
        }
    )

    summary = select_representative_patch(
        representative_circuits=[qc],
        resolved_spec=spec,
        calibration_snapshot=snapshot,
    )
    lock_key = nab._layout_lock_registry_key(spec, snapshot)
    bundled = load_snapshot_layout_lock(snapshot_path, lock_key)

    assert summary["patch_selection_objective"][0] == "min_extra_2q_count"
    assert summary["selected_physical_qubits"] == [0, 1]
    assert "extra_2q_count" in summary["patch_score_summary"]
    assert bundled is not None
    assert bundled["selected_patch_qubits"] == [0, 1]
    assert bundled["selected_patch_edges"] == [[0, 1], [1, 0]]
    assert bundled["patch_selection_summary"]["anchor_layout_hash"] == summary["anchor_layout_hash"]



def test_patch_snapshot_artifact_replays_from_snapshot_bundle_without_tempdir_lock(tmp_path: Path) -> None:
    pytest.importorskip("qiskit_aer")

    snapshot = _make_snapshot(
        num_qubits=2,
        undirected_edges=[(0, 1)],
        twoq_errors={(0, 1): 0.01, (1, 0): 0.01},
        readout_errors={},
        t1s={},
        t2s={},
    )
    snapshot_path = tmp_path / "patch_bundle_replay.json"
    write_calibration_snapshot(snapshot_path, snapshot)

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.measure_all()
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    capture_spec = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "backend_scheduled",
            "backend_profile": "frozen_snapshot_json",
            "noise_snapshot_json": str(snapshot_path),
            "schedule_policy": "asap",
            "layout_lock_key": "patch_bundle_replay_lock",
            "shots": 128,
        }
    )
    replay_spec = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "patch_snapshot",
            "backend_profile": "frozen_snapshot_json",
            "noise_snapshot_json": str(snapshot_path),
            "schedule_policy": "asap",
            "layout_lock_key": "patch_bundle_replay_lock",
            "shots": 128,
        }
    )

    summary = select_representative_patch(
        representative_circuits=[qc],
        resolved_spec=capture_spec,
        calibration_snapshot=snapshot,
    )
    lock_key = nab._layout_lock_registry_key(capture_spec, snapshot)
    nab._LAYOUT_LOCK_REGISTRY.pop(lock_key, None)
    lock_file = nab._layout_lock_file(lock_key)
    if lock_file.exists():
        lock_file.unlink()

    frozen_snapshot = load_calibration_snapshot(snapshot_path)
    nab._LAYOUT_LOCK_SOURCE_REGISTRY.pop(lock_key, None)
    assert describe_layout_anchor_source(replay_spec, frozen_snapshot) == "snapshot_bundle_lock"
    assert describe_layout_anchor_source(replay_spec, frozen_snapshot) == "snapshot_bundle_lock"

    backend, _warnings = build_backend_from_calibration_snapshot(frozen_snapshot)
    artifact = build_patch_snapshot_artifact(
        circuit=qc,
        observable=obs,
        resolved_spec=replay_spec,
        calibration_snapshot=frozen_snapshot,
        resolved_backend=backend,
        qiskit_noise_model=None,
    )

    assert artifact.patch_selection_summary is not None
    assert artifact.patch_selection_summary["selected_physical_qubits"] == [0, 1]
    assert artifact.patch_selection_summary["selected_edges"] == [[0, 1], [1, 0]]
    assert artifact.patch_selection_summary["snapshot_hash"] == frozen_snapshot.snapshot_hash
    assert artifact.layout_hash == summary["anchor_layout_hash"]
    assert artifact.transpile_snapshot is not None
    assert artifact.transpile_snapshot.used_physical_qubits == summary["anchor_used_physical_qubits"]
    assert artifact.transpile_snapshot.used_physical_edges == summary["anchor_used_physical_edges"]
