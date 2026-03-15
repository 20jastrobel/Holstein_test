from __future__ import annotations

from typing import Any

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

import pipelines.exact_bench.hh_noise_hardware_validation as hhv


class _StubValidationOracle:
    def __init__(self, config: Any):
        self.config = config
        self.backend_info = hhv.NoiseBackendInfo(
            noise_mode=str(config.noise_mode),
            estimator_kind="stub",
            backend_name=str(getattr(config, "backend_name", "ibm_fake_runtime") or "ibm_fake_runtime"),
            using_fake_backend=False,
            details={},
        )
        self.current_execution = type(
            "_StubExecution",
            (),
            {"to_dict": lambda inner_self: hhv.oracle_execution_dict(backend_info=self.backend_info)},
        )()

    def __enter__(self) -> "_StubValidationOracle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def evaluate(self, circuit: QuantumCircuit, observable: SparsePauliOp) -> hhv.OracleEstimate:
        mode = str(self.config.noise_mode)
        if mode == "patch_snapshot" and str(self.config.layout_policy) == "frozen_layout":
            raise RuntimeError(
                "layout_policy='frozen_layout' requires an existing persisted layout lock or replay artifact; no prior locked layout was found for this backend/snapshot/key."
            )
        twirling_cfg = dict(getattr(self.config, "runtime_twirling", {}) or {})
        runtime_bundle = None
        executor = "aer"
        noise_kind = mode
        anchor_source = "persisted_lock"
        if mode == "backend_scheduled":
            noise_kind = "backend_scheduled"
            anchor_source = "fresh_auto_then_lock"
        elif mode == "qpu_raw":
            executor = "runtime_qpu"
            noise_kind = "qpu_raw"
            runtime_bundle = {
                "requested_noise_mode": "qpu_raw",
                "resolved_noise_kind": "qpu_raw",
                "mitigation_bundle": "none",
                "mitigation_mode": "none",
                "twirling_enable_gates": False,
                "twirling_enable_measure": False,
                "twirling_num_randomizations": None,
                "twirling_strategy": None,
            }
        elif mode == "qpu_suppressed":
            executor = "runtime_qpu"
            noise_kind = "qpu_suppressed"
            runtime_bundle = {
                "requested_noise_mode": "qpu_suppressed",
                "resolved_noise_kind": "qpu_suppressed",
                "mitigation_bundle": "runtime_suppressed",
                "mitigation_mode": str(self.config.mitigation.get("mode")),
                "twirling_enable_gates": bool(twirling_cfg.get("enable_gates", False)),
                "twirling_enable_measure": bool(twirling_cfg.get("enable_measure", False)),
                "twirling_num_randomizations": twirling_cfg.get("num_randomizations", None),
                "twirling_strategy": twirling_cfg.get("strategy", None),
            }
        elif mode == "patch_snapshot":
            noise_kind = "patch_snapshot"
        provenance_class = {
            "backend_scheduled": "local_generic_aer_execution",
            "patch_snapshot": "local_patch_frozen_replay",
            "qpu_raw": "runtime_submitted_raw",
            "qpu_suppressed": "runtime_submitted_suppressed",
        }.get(mode, "unresolved_execution_provenance")
        details = {
            "resolved_noise_spec": {"executor": executor, "noise_kind": noise_kind},
            "layout_hash": "layout:shared",
            "transpile_hash": "tx:shared",
            "snapshot_hash": "snap:shared",
            "used_physical_qubits": [0, 1],
            "used_physical_edges": [[0, 1]],
            "circuit_structure_hash": "cstruct:shared",
            "layout_anchor_source": anchor_source,
            "runtime_execution_bundle": runtime_bundle,
            "provenance_summary": {
                "classification": provenance_class,
                "layout_anchor_reused": bool(anchor_source in {"persisted_lock", "fixed_patch"}),
            },
            "source_kind": "live_backend",
        }
        self.backend_info = hhv.NoiseBackendInfo(
            noise_mode=str(self.config.noise_mode),
            estimator_kind="stub",
            backend_name=str(getattr(self.config, "backend_name", "ibm_fake_runtime") or "ibm_fake_runtime"),
            using_fake_backend=False,
            details=details,
        )
        self.current_execution = type(
            "_StubExecution",
            (),
            {"to_dict": lambda inner_self: hhv.oracle_execution_dict(backend_info=self.backend_info)},
        )()
        means = {
            "backend_scheduled": 0.11,
            "patch_snapshot": 0.11,
            "qpu_raw": 0.12,
            "qpu_suppressed": 0.10,
        }
        mean = float(means.get(mode, 0.0))
        return hhv.OracleEstimate(
            mean=mean,
            std=0.0,
            stdev=0.0,
            stderr=0.0,
            n_samples=1,
            raw_values=[mean],
            aggregate="mean",
        )

    def evaluate_result(self, circuit: QuantumCircuit, observable: SparsePauliOp) -> Any:
        estimate = self.evaluate(circuit, observable)
        return type(
            "_StubResult",
            (),
            {"estimate": estimate, "execution": self.current_execution.to_dict()},
        )()



def test_build_validation_dynamics_circuit_accepts_cfqm_with_pauli_suzuki2() -> None:
    qc = hhv._build_validation_dynamics_circuit(
        initial_circuit=QuantumCircuit(1),
        ordered_labels_exyz=["z"],
        static_coeff_map_exyz={"z": 0.5 + 0.0j},
        method="cfqm4",
        time_value=0.2,
        trotter_steps=1,
        suzuki_order=2,
        cfqm_stage_exp="pauli_suzuki2",
        cfqm_coeff_drop_abs_tol=0.0,
    )
    assert qc.num_qubits == 1


def test_build_validation_dynamics_circuit_rejects_numerical_only_cfqm_stage_exp() -> None:
    with pytest.raises(ValueError, match="pauli_suzuki2"):
        hhv._build_validation_dynamics_circuit(
            initial_circuit=QuantumCircuit(1),
            ordered_labels_exyz=["z"],
            static_coeff_map_exyz={"z": 0.5 + 0.0j},
            method="cfqm4",
            time_value=0.2,
            trotter_steps=1,
            suzuki_order=2,
            cfqm_stage_exp="dense_expm",
            cfqm_coeff_drop_abs_tol=0.0,
        )


def test_paired_anchor_validation_reports_same_anchor_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hhv, "ExpectationOracle", _StubValidationOracle)
    args = hhv.parse_args(
        [
            "--problem",
            "hh",
            "--ansatz",
            "hh_hva_ptw",
            "--L",
            "2",
            "--noise-mode",
            "runtime",
            "--backend-name",
            "ibm_fake_runtime",
            "--mitigation",
            "readout",
            "--paired-anchor-validation",
        ]
    )
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    payload = hhv._run_paired_anchor_validation(args=args, circuit=qc, observable=obs)

    assert payload["layout_lock_key"] == "validation:L2:hh:hh_hva_ptw:local_aer_locked_patch"
    assert payload["lock_family_token"] == "local_aer_locked_patch"
    assert payload["local_mode"] == "backend_scheduled"
    assert [row["label"] for row in payload["rows"]] == [
        "local_patch_anchor",
        "runtime_raw",
        "runtime_suppressed",
    ]
    local_row, raw_row, suppressed_row = payload["rows"]
    assert local_row["layout_anchor_source"] == "fresh_auto_then_lock"
    assert raw_row["layout_anchor_source"] == "persisted_lock"
    assert suppressed_row["layout_anchor_source"] == "persisted_lock"
    assert local_row["provenance_classification"] == "local_generic_aer_execution"
    assert raw_row["provenance_classification"] == "runtime_submitted_raw"
    assert suppressed_row["provenance_classification"] == "runtime_submitted_suppressed"
    assert local_row["layout_hash"] == raw_row["layout_hash"] == suppressed_row["layout_hash"]
    assert local_row["used_physical_qubits"] == raw_row["used_physical_qubits"] == suppressed_row["used_physical_qubits"]
    assert local_row["used_physical_edges"] == raw_row["used_physical_edges"] == suppressed_row["used_physical_edges"]
    assert local_row["circuit_structure_hash"] == raw_row["circuit_structure_hash"] == suppressed_row["circuit_structure_hash"]
    assert raw_row["runtime_execution_bundle"]["mitigation_mode"] == "none"
    assert suppressed_row["runtime_execution_bundle"]["mitigation_mode"] == "readout"
    cmp = payload["comparability"]
    assert cmp["same_lock_context"] is True
    assert cmp["same_backend_name"] is True
    assert cmp["same_layout_hash"] is True
    assert cmp["local_vs_runtime_same_anchor_evidence_status"] == "fully_evidenced"
    assert cmp["runtime_pair_same_submitted_evidence_status"] == "fully_evidenced"
    assert cmp["exact_anchor_identity_evidence_status"] == "fully_evidenced"
    assert cmp["local_vs_runtime_same_anchor_provenance"] is True
    assert cmp["runtime_pair_same_submitted_provenance"] is True
    assert cmp["runtime_execution_bundle_differs"] is True
    assert cmp["runtime_pair_differs_only_in_execution_bundle"] is True
    assert cmp["exact_anchor_identity_across_rows"] is True



def test_paired_anchor_validation_projects_runtime_twirling_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hhv, "ExpectationOracle", _StubValidationOracle)
    args = hhv.parse_args(
        [
            "--problem",
            "hh",
            "--ansatz",
            "hh_hva_ptw",
            "--L",
            "2",
            "--noise-mode",
            "runtime",
            "--backend-name",
            "ibm_fake_runtime",
            "--mitigation",
            "readout",
            "--runtime-enable-measure-twirling",
            "--runtime-twirling-num-randomizations",
            "8",
            "--runtime-twirling-strategy",
            "active",
            "--paired-anchor-validation",
        ]
    )
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    payload = hhv._run_paired_anchor_validation(args=args, circuit=qc, observable=obs)

    suppressed_row = next(row for row in payload["rows"] if row["label"] == "runtime_suppressed")
    assert payload["runtime_twirling_config"] == {
        "enable_gates": False,
        "enable_measure": True,
        "num_randomizations": 8,
        "strategy": "active",
    }
    assert suppressed_row["runtime_execution_bundle"]["twirling_enable_measure"] is True
    assert suppressed_row["runtime_execution_bundle"]["twirling_num_randomizations"] == 8
    assert suppressed_row["runtime_execution_bundle"]["twirling_strategy"] == "active"


def test_paired_anchor_row_preserves_layer_noise_model_runtime_bundle_fields() -> None:
    cfg = hhv.OracleConfig(noise_mode="qpu_layer_learned", layout_lock_key="shared")
    estimate = hhv.OracleEstimate(
        mean=0.1,
        std=0.0,
        stdev=0.0,
        stderr=0.0,
        n_samples=1,
        raw_values=[0.1],
        aggregate="mean",
    )
    backend_info = hhv.NoiseBackendInfo(
        noise_mode="qpu_layer_learned",
        estimator_kind="stub",
        backend_name="ibm_fake_runtime",
        using_fake_backend=False,
        details={
            "resolved_noise_spec": {"executor": "runtime_qpu", "noise_kind": "qpu_layer_learned"},
            "layout_hash": "layout:shared",
            "transpile_hash": "tx:shared",
            "snapshot_hash": "snap:shared",
            "used_physical_qubits": [0, 1],
            "used_physical_edges": [[0, 1]],
            "circuit_structure_hash": "cstruct:shared",
            "layout_anchor_source": "persisted_lock",
            "runtime_execution_bundle": {
                "requested_noise_mode": "qpu_layer_learned",
                "resolved_noise_kind": "qpu_layer_learned",
                "mitigation_bundle": "runtime_layer_learned",
                "mitigation_mode": "zne",
                "zne_amplifier": "pea",
                "layer_noise_learning_requested": False,
                "layer_noise_model_supplied": True,
                "layer_noise_model_source": "programmatic_object",
                "layer_noise_model_kind": "Sequence[LayerError]",
                "layer_noise_model_entry_count": 1,
            },
            "provenance_summary": {
                "classification": "runtime_submitted_layer_learned",
                "layout_anchor_reused": True,
                "runtime_layer_noise_learning": False,
                "runtime_layer_noise_model_supplied": True,
                "runtime_layer_noise_model_source": "programmatic_object",
            },
            "source_kind": "live_backend",
        },
    )

    row = hhv._paired_anchor_row(
        label="runtime_layer_learned",
        cfg=cfg,
        estimate=estimate,
        backend_info=backend_info,
    )

    assert row["provenance_classification"] == "runtime_submitted_layer_learned"
    assert row["runtime_execution_bundle"]["layer_noise_model_supplied"] is True
    assert row["runtime_execution_bundle"]["layer_noise_model_source"] == "programmatic_object"
    assert row["runtime_execution_bundle"]["layer_noise_model_kind"] == "Sequence[LayerError]"
    assert row["runtime_execution_bundle"]["layer_noise_model_entry_count"] == 1


def test_paired_anchor_row_preserves_file_backed_layer_noise_model_runtime_bundle_fields() -> None:
    cfg = hhv.OracleConfig(noise_mode="qpu_layer_learned", layout_lock_key="shared")
    estimate = hhv.OracleEstimate(
        mean=0.1,
        std=0.0,
        stdev=0.0,
        stderr=0.0,
        n_samples=1,
        raw_values=[0.1],
        aggregate="mean",
    )
    backend_info = hhv.NoiseBackendInfo(
        noise_mode="qpu_layer_learned",
        estimator_kind="stub",
        backend_name="ibm_fake_runtime",
        using_fake_backend=False,
        details={
            "resolved_noise_spec": {"executor": "runtime_qpu", "noise_kind": "qpu_layer_learned"},
            "layout_hash": "layout:shared",
            "transpile_hash": "tx:shared",
            "snapshot_hash": "snap:shared",
            "used_physical_qubits": [0, 1],
            "used_physical_edges": [[0, 1]],
            "circuit_structure_hash": "cstruct:shared",
            "layout_anchor_source": "persisted_lock",
            "runtime_execution_bundle": {
                "requested_noise_mode": "qpu_layer_learned",
                "resolved_noise_kind": "qpu_layer_learned",
                "mitigation_bundle": "runtime_layer_learned",
                "mitigation_mode": "zne",
                "zne_amplifier": "pea",
                "layer_noise_learning_requested": False,
                "layer_noise_model_supplied": True,
                "layer_noise_model_source": "file_backed_json",
                "layer_noise_model_kind": "NoiseLearnerResult",
                "layer_noise_model_entry_count": 1,
                "layer_noise_model_fingerprint": "sha256:test",
            },
            "provenance_summary": {
                "classification": "runtime_submitted_layer_learned",
                "layout_anchor_reused": True,
                "runtime_layer_noise_learning": False,
                "runtime_layer_noise_model_supplied": True,
                "runtime_layer_noise_model_source": "file_backed_json",
                "runtime_layer_noise_model_kind": "NoiseLearnerResult",
                "runtime_layer_noise_model_fingerprint": "sha256:test",
            },
            "source_kind": "live_backend",
        },
    )

    row = hhv._paired_anchor_row(
        label="runtime_layer_learned",
        cfg=cfg,
        estimate=estimate,
        backend_info=backend_info,
    )

    assert row["runtime_execution_bundle"]["layer_noise_model_source"] == "file_backed_json"
    assert row["runtime_execution_bundle"]["layer_noise_model_kind"] == "NoiseLearnerResult"
    assert row["runtime_execution_bundle"]["layer_noise_model_fingerprint"] == "sha256:test"


def test_paired_anchor_validation_frozen_layout_missing_anchor_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hhv, "ExpectationOracle", _StubValidationOracle)
    args = hhv.parse_args(
        [
            "--problem",
            "hh",
            "--ansatz",
            "hh_hva_ptw",
            "--L",
            "2",
            "--noise-mode",
            "runtime",
            "--backend-name",
            "ibm_fake_runtime",
            "--mitigation",
            "readout",
            "--paired-anchor-validation",
            "--layout-policy",
            "frozen_layout",
            "--noise-snapshot-json",
            "artifacts/json/frozen_snapshot.json",
        ]
    )
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    with pytest.raises(RuntimeError, match="layout_policy='frozen_layout' requires an existing persisted layout lock"):
        hhv._run_paired_anchor_validation(args=args, circuit=qc, observable=obs)


def test_paired_anchor_comparability_does_not_overclaim_bundle_only_difference() -> None:
    rows = [
        {
            "label": "runtime_raw",
            "layout_lock_key": "shared",
            "lock_family_token": "local_aer_locked_patch",
            "backend_name": "ibm_fake_runtime",
            "snapshot_hash": "snap:shared",
            "layout_hash": "layout:a",
            "transpile_hash": "tx:a",
            "used_physical_qubits": [0, 1],
            "used_physical_edges": [[0, 1]],
            "circuit_structure_hash": "cstruct:a",
            "runtime_execution_bundle": {"mitigation_mode": "none"},
        },
        {
            "label": "runtime_suppressed",
            "layout_lock_key": "shared",
            "lock_family_token": "local_aer_locked_patch",
            "backend_name": "ibm_fake_runtime",
            "snapshot_hash": "snap:shared",
            "layout_hash": "layout:b",
            "transpile_hash": "tx:b",
            "used_physical_qubits": [0, 1],
            "used_physical_edges": [[0, 1]],
            "circuit_structure_hash": "cstruct:b",
            "runtime_execution_bundle": {"mitigation_mode": "readout"},
        },
    ]

    cmp = hhv._paired_anchor_comparability(rows)

    assert cmp["runtime_execution_bundle_differs"] is True
    assert cmp["runtime_pair_same_submitted_evidence_status"] == "not_evidenced"
    assert cmp["runtime_pair_same_submitted_provenance"] is False
    assert cmp["runtime_pair_differs_only_in_execution_bundle"] is False


def test_paired_anchor_comparability_marks_partial_evidence_when_backend_identity_is_missing() -> None:
    rows = [
        {
            "label": "runtime_raw",
            "layout_lock_key": "shared",
            "lock_family_token": "local_aer_locked_patch",
            "backend_name": "ibm_fake_runtime",
            "snapshot_hash": None,
            "layout_hash": "layout:a",
            "transpile_hash": "tx:a",
            "used_physical_qubits": [0, 1],
            "used_physical_edges": [[0, 1]],
            "circuit_structure_hash": "cstruct:a",
            "runtime_execution_bundle": {"mitigation_mode": "none"},
        },
        {
            "label": "runtime_suppressed",
            "layout_lock_key": "shared",
            "lock_family_token": "local_aer_locked_patch",
            "backend_name": "ibm_fake_runtime",
            "snapshot_hash": None,
            "layout_hash": "layout:a",
            "transpile_hash": "tx:a",
            "used_physical_qubits": [0, 1],
            "used_physical_edges": [[0, 1]],
            "circuit_structure_hash": "cstruct:a",
            "runtime_execution_bundle": {"mitigation_mode": "readout"},
        },
    ]

    cmp = hhv._paired_anchor_comparability(rows)

    assert cmp["runtime_pair_same_submitted_evidence_status"] == "partially_evidenced"
    assert cmp["runtime_pair_same_submitted_provenance"] is False
    assert cmp["runtime_pair_differs_only_in_execution_bundle"] is False
