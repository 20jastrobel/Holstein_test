from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.exact_bench.hh_l2_stage_unit_audit_workflow as wf
from pipelines.hardcoded.hh_staged_workflow import StageExecutionResult


class _DummyPolyTerm:
    def __init__(self, label_exyz: str, coeff: complex = 1.0 + 0.0j) -> None:
        self._label_exyz = str(label_exyz)
        self.p_coeff = complex(coeff)

    def pw2strng(self) -> str:
        return str(self._label_exyz)


class _DummyPoly:
    def __init__(self, label_exyz: str, matrix: np.ndarray | None = None) -> None:
        self._label_exyz = str(label_exyz)
        self.matrix = np.asarray(matrix if matrix is not None else np.eye(2), dtype=complex)

    def return_polynomial(self) -> list[_DummyPolyTerm]:
        return [_DummyPolyTerm(self._label_exyz)]


class _DummyAnsatzTerm:
    def __init__(self, label: str, polynomial: _DummyPoly) -> None:
        self.label = str(label)
        self.polynomial = polynomial


class _DummyTermwiseAnsatz:
    def __init__(self, *, reps: int, base_terms: list[_DummyAnsatzTerm]) -> None:
        self.reps = int(reps)
        self.base_terms = list(base_terms)
        self.num_parameters = int(self.reps * len(self.base_terms))


class _DummyLayerwiseAnsatz:
    def __init__(self, *, reps: int, layer_term_groups: list[tuple[str, list[_DummyAnsatzTerm]]]) -> None:
        self.reps = int(reps)
        self.layer_term_groups = list(layer_term_groups)
        self.num_parameters = int(self.reps * len(self.layer_term_groups))


class _DummyReplayAnsatz(_DummyTermwiseAnsatz):
    pass


def _basis(dim: int, idx: int) -> np.ndarray:
    out = np.zeros(int(dim), dtype=complex)
    out[int(idx)] = 1.0
    return out


def _stage_result(
    *,
    hmat: np.ndarray | None = None,
    psi_hf: np.ndarray | None = None,
    psi_warm: np.ndarray | None = None,
    psi_adapt: np.ndarray | None = None,
    psi_final: np.ndarray | None = None,
    warm_ctx: dict[str, object] | None = None,
    adapt_ctx: dict[str, object] | None = None,
    replay_ctx: dict[str, object] | None = None,
    warm_payload: dict[str, object] | None = None,
    adapt_payload: dict[str, object] | None = None,
    replay_payload: dict[str, object] | None = None,
) -> StageExecutionResult:
    hmat_use = np.asarray(hmat if hmat is not None else np.diag([0.0, 1.0]), dtype=complex)
    psi0 = np.asarray(psi_hf if psi_hf is not None else _basis(2, 1), dtype=complex)
    return StageExecutionResult(
        h_poly=object(),
        hmat=hmat_use,
        ordered_labels_exyz=[],
        coeff_map_exyz={},
        nq_total=1,
        psi_hf=psi0,
        psi_warm=np.asarray(psi_warm if psi_warm is not None else psi0, dtype=complex),
        psi_adapt=np.asarray(psi_adapt if psi_adapt is not None else psi0, dtype=complex),
        psi_final=np.asarray(psi_final if psi_final is not None else psi0, dtype=complex),
        warm_payload=dict(warm_payload or {}),
        adapt_payload=dict(adapt_payload or {}),
        replay_payload=dict(replay_payload or {}),
        warm_circuit_context=warm_ctx,
        adapt_circuit_context=adapt_ctx,
        replay_circuit_context=replay_ctx,
    )


def test_build_locked_staged_hh_audit_config_pins_l2_nph2_repo_minimums() -> None:
    cfg = wf.build_locked_staged_hh_audit_config(wf.AuditWorkflowConfig())

    assert int(cfg.physics.L) == 2
    assert int(cfg.physics.n_ph_max) == 2
    assert str(cfg.warm_start.method) == "SPSA"
    assert int(cfg.warm_start.reps) == 3
    assert int(cfg.warm_start.restarts) == 4
    assert int(cfg.warm_start.maxiter) == 1500
    assert str(cfg.replay.method) == "SPSA"
    assert int(cfg.replay.reps) == 3
    assert int(cfg.replay.restarts) == 4
    assert int(cfg.replay.maxiter) == 1500
    assert int(cfg.dynamics.trotter_steps) == 128
    assert bool(cfg.dynamics.enable_drive) is False
    assert str(cfg.physics.ordering) == "blocked"
    assert str(cfg.physics.boundary) == "open"
    assert str(cfg.adapt.pool) == "paop_lf_std"
    assert bool(cfg.adapt.phase1_prune_enabled) is False
    assert int(cfg.adapt.max_depth) == 80
    assert int(cfg.adapt.maxiter) == 2222
    assert cfg.default_provenance["audit_locked_profile"] == "AGENTS.hh_L2_nph2.audit_locked_profile"



def test_extract_warm_termwise_units_in_layer_major_order() -> None:
    ansatz = _DummyTermwiseAnsatz(
        reps=2,
        base_terms=[
            _DummyAnsatzTerm("hop", _DummyPoly("x")),
            _DummyAnsatzTerm("eph", _DummyPoly("z")),
        ],
    )
    stage_result = _stage_result(
        warm_ctx={
            "ansatz": ansatz,
            "theta": np.asarray([0.1, 0.2, 0.3, 0.4]),
            "reference_state": _basis(2, 1),
            "ansatz_name": "hh_hva_ptw",
        },
        psi_warm=_basis(2, 0),
    )

    spec = wf._warm_stage_spec(stage_result)

    assert [unit.unit_label for unit in spec.units_in_acceptance_order] == [
        "layer1:hop",
        "layer1:eph",
        "layer2:hop",
        "layer2:eph",
    ]
    assert [unit.unit_kind for unit in spec.units_in_acceptance_order] == [
        "logical_block",
        "logical_block",
        "logical_block",
        "logical_block",
    ]
    assert [len(unit.polynomials) for unit in spec.units_in_acceptance_order] == [1, 1, 1, 1]



def test_extract_warm_layerwise_units_in_layer_group_order() -> None:
    ansatz = _DummyLayerwiseAnsatz(
        reps=2,
        layer_term_groups=[
            (
                "hop_layer",
                [
                    _DummyAnsatzTerm("hop_term_0", _DummyPoly("xx")),
                    _DummyAnsatzTerm("hop_term_1", _DummyPoly("yy")),
                ],
            ),
            (
                "eph_layer",
                [_DummyAnsatzTerm("eph_term_0", _DummyPoly("z"))],
            ),
        ],
    )
    stage_result = _stage_result(
        warm_ctx={
            "ansatz": ansatz,
            "theta": np.asarray([0.1, 0.2, 0.3, 0.4]),
            "reference_state": _basis(2, 1),
            "ansatz_name": "hh_hva",
        },
        psi_warm=_basis(2, 0),
    )

    spec = wf._warm_stage_spec(stage_result)

    assert [unit.unit_label for unit in spec.units_in_acceptance_order] == [
        "layer1:hop_layer",
        "layer1:eph_layer",
        "layer2:hop_layer",
        "layer2:eph_layer",
    ]
    assert [unit.unit_kind for unit in spec.units_in_acceptance_order] == [
        "layer_sub_block",
        "layer_sub_block",
        "layer_sub_block",
        "layer_sub_block",
    ]
    assert [len(unit.polynomials) for unit in spec.units_in_acceptance_order] == [2, 1, 2, 1]



def test_extract_adapt_units_uses_acceptance_order_and_insertion_history() -> None:
    final_selected_ops = [
        _DummyAnsatzTerm("B", _DummyPoly("x")),
        _DummyAnsatzTerm("A", _DummyPoly("z")),
    ]
    stage_result = _stage_result(
        adapt_ctx={
            "selected_ops": final_selected_ops,
            "theta": np.asarray([0.2, 0.1]),
            "reference_state": _basis(2, 0),
            "pool_type": "paop_lf_std",
            "continuation_mode": "phase3_v1",
        },
        adapt_payload={
            "history": [
                {"selected_op": "A", "selected_position": 0},
                {"selected_op": "B", "selected_position": 0},
            ]
        },
        psi_adapt=_basis(2, 0),
    )

    spec = wf._adapt_stage_spec(stage_result)
    units_by_id = {unit.unit_id: unit for unit in spec.units_in_acceptance_order}

    assert [unit.base_label for unit in spec.units_in_acceptance_order] == ["A", "B"]
    assert [unit.theta_value for unit in spec.units_in_acceptance_order] == pytest.approx([0.1, 0.2])
    assert [units_by_id[unit_id].base_label for unit_id in spec.full_order_ids] == ["B", "A"]
    assert [
        [units_by_id[unit_id].base_label for unit_id in order_ids]
        for order_ids in spec.prefix_order_ids
    ] == [["A"], ["B", "A"]]



def test_compute_stage_audit_rows_prefix_and_removal_penalties(monkeypatch: pytest.MonkeyPatch) -> None:
    x_poly = _DummyPoly("x", matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
    i_poly = _DummyPoly("e", matrix=np.eye(2, dtype=complex))

    def _fake_apply(psi: np.ndarray, polynomial: _DummyPoly, theta_value: float) -> np.ndarray:
        _ = theta_value
        return np.asarray(polynomial.matrix @ np.asarray(psi, dtype=complex).reshape(-1), dtype=complex)

    monkeypatch.setattr(wf, "_apply_single_polynomial", _fake_apply)

    unit_1 = wf._make_unit(
        stage="warm_start",
        unit_index=1,
        unit_kind="logical_block",
        unit_label="layer1:flip",
        base_label="flip",
        theta_value=0.1,
        polynomials=[x_poly],
        insertion_position=0,
        final_order_index=0,
    )
    unit_2 = wf._make_unit(
        stage="warm_start",
        unit_index=2,
        unit_kind="logical_block",
        unit_label="layer1:stay",
        base_label="stay",
        theta_value=0.2,
        polynomials=[i_poly],
        insertion_position=1,
        final_order_index=1,
    )
    spec = wf.StageAuditSpec(
        stage="warm_start",
        reference_state=_basis(2, 1),
        expected_full_state=_basis(2, 0),
        units_in_acceptance_order=(unit_1, unit_2),
        full_order_ids=(unit_1.unit_id, unit_2.unit_id),
        prefix_order_ids=((unit_1.unit_id,), (unit_1.unit_id, unit_2.unit_id)),
        reference_energy=1.0,
        stage_metadata={},
    )

    rows, stage_summary = wf.compute_stage_audit_rows(spec, np.diag([0.0, 1.0]))

    assert rows[0]["energy_prefix"] == pytest.approx(0.0)
    assert rows[0]["delta_energy_from_previous"] == pytest.approx(1.0)
    assert rows[0]["removal_penalty"] == pytest.approx(1.0)
    assert rows[1]["energy_prefix"] == pytest.approx(0.0)
    assert rows[1]["delta_energy_from_previous"] == pytest.approx(0.0)
    assert rows[1]["removal_penalty"] == pytest.approx(0.0)
    assert stage_summary["reconstruction_error_global_phase_aligned"] == pytest.approx(0.0)



def test_run_audit_emits_csv_json_and_uses_no_qiskit_core_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    x_poly = _DummyPoly("x", matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
    i_poly = _DummyPoly("e", matrix=np.eye(2, dtype=complex))

    def _fake_apply(psi: np.ndarray, polynomial: _DummyPoly, theta_value: float) -> np.ndarray:
        _ = theta_value
        return np.asarray(polynomial.matrix @ np.asarray(psi, dtype=complex).reshape(-1), dtype=complex)

    monkeypatch.setattr(wf, "_apply_single_polynomial", _fake_apply)

    psi_hf = _basis(2, 1)
    psi_zero = _basis(2, 0)
    warm_ansatz = _DummyTermwiseAnsatz(reps=1, base_terms=[_DummyAnsatzTerm("flip", x_poly)])
    adapt_ops = [_DummyAnsatzTerm("stay", i_poly)]
    replay_ansatz = _DummyReplayAnsatz(reps=1, base_terms=[_DummyAnsatzTerm("stay", i_poly)])
    stage_result = _stage_result(
        psi_hf=psi_hf,
        psi_warm=psi_zero,
        psi_adapt=psi_zero,
        psi_final=psi_zero,
        warm_ctx={
            "ansatz": warm_ansatz,
            "theta": np.asarray([0.1]),
            "reference_state": psi_hf,
            "ansatz_name": "hh_hva_ptw",
        },
        adapt_ctx={
            "selected_ops": adapt_ops,
            "theta": np.asarray([0.2]),
            "reference_state": psi_zero,
            "pool_type": "paop_lf_std",
            "continuation_mode": "phase3_v1",
        },
        replay_ctx={
            "ansatz": replay_ansatz,
            "theta": np.asarray([0.3]),
            "reference_state": psi_zero,
            "family_info": {"resolved": "paop_lf_std"},
            "resolved_seed_policy": "residual_only",
        },
        adapt_payload={"history": [{"selected_op": "stay", "selected_position": 0}]},
    )
    workflow_cfg = wf.AuditWorkflowConfig(
        output_json=tmp_path / "hh_l2_stage_unit_audit.json",
        output_csv=tmp_path / "hh_l2_stage_unit_audit.csv",
    )

    payload = wf.run_hh_l2_stage_unit_audit(workflow_cfg, stage_result=stage_result)

    assert payload["settings"]["physics"]["n_ph_max"] == 2
    assert payload["audit_scope"]["patch_selection_enabled"] is False
    assert payload["audit_scope"]["noise_enabled"] is False
    assert {row["stage"] for row in payload["rows"]} == {
        "warm_start",
        "adapt_vqe",
        "conventional_replay",
    }
    assert workflow_cfg.output_json.exists()
    assert workflow_cfg.output_csv.exists()

    payload_json = json.loads(workflow_cfg.output_json.read_text(encoding="utf-8"))
    assert payload_json["rows"][0]["unit_label"]
    assert payload_json["summary"]["smallest_removal_penalty"]

    with workflow_cfg.output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 3
    required_fields = {
        "stage",
        "unit_index",
        "unit_kind",
        "unit_label",
        "sequence_order",
        "energy_prefix",
        "delta_energy_from_previous",
        "energy_full",
        "energy_full_minus_unit",
        "removal_penalty",
        "parameter_count",
        "logical_2q_count",
        "logical_depth",
        "circuit_hash",
    }
    assert required_fields.issubset(set(rows[0].keys()))
