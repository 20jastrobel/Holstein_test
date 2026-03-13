from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.exact_bench.hh_l2_logical_screen_workflow as wf
from pipelines.exact_bench.hh_l2_stage_unit_audit_workflow import AuditWorkflowConfig


def _screen_cfg(
    tmp_path: Path,
    *,
    points: tuple[wf.HamiltonianPoint, ...] | None = None,
    seed_count: int = 3,
    include_prefix_50: bool = False,
    warm_vqe_reps_override: int | None = None,
    warm_vqe_restarts_override: int | None = None,
    warm_vqe_maxiter_override: int | None = None,
    adapt_max_depth_override: int | None = None,
    adapt_drop_min_depth_override: int | None = None,
    adapt_maxiter_override: int | None = None,
    final_vqe_reps_override: int | None = None,
    final_vqe_restarts_override: int | None = None,
    final_vqe_maxiter_override: int | None = None,
) -> wf.LogicalScreenConfig:
    if points is None:
        points = (wf.HamiltonianPoint(t=1.0, u=4.0, dv=0.0, omega0=1.0, g_ep=1.0),)
    return wf.LogicalScreenConfig(
        output_json=tmp_path / "screen.json",
        output_csv=tmp_path / "screen.csv",
        run_root=tmp_path / "runs",
        screen_tag="screen_test",
        points=tuple(points),
        seed_count=int(seed_count),
        last_k=3,
        stress_point_count=1,
        warm_ansatz="hh_hva_ptw",
        adapt_pool="paop_lf_std",
        adapt_continuation_mode="phase3_v1",
        include_prefix_50=bool(include_prefix_50),
        warm_vqe_reps_override=warm_vqe_reps_override,
        warm_vqe_restarts_override=warm_vqe_restarts_override,
        warm_vqe_maxiter_override=warm_vqe_maxiter_override,
        adapt_max_depth_override=adapt_max_depth_override,
        adapt_drop_min_depth_override=adapt_drop_min_depth_override,
        adapt_maxiter_override=adapt_maxiter_override,
        final_vqe_reps_override=final_vqe_reps_override,
        final_vqe_restarts_override=final_vqe_restarts_override,
        final_vqe_maxiter_override=final_vqe_maxiter_override,
    )


def _baseline_record(point: wf.HamiltonianPoint, *, seed_index: int, delta: float) -> wf.BaselineRunRecord:
    seeds = wf.SeedTriple(seed_index=seed_index, warm_seed=7 + 100 * (seed_index - 1), adapt_seed=11 + 100 * (seed_index - 1), final_seed=19 + 100 * (seed_index - 1))
    return wf.BaselineRunRecord(
        point=point,
        seeds=seeds,
        staged_cfg=None,
        audit_cfg=AuditWorkflowConfig(
            output_json=Path(f"audit_{seed_index}.json"),
            output_csv=Path(f"audit_{seed_index}.csv"),
            stage_tag=f"stage_{seed_index}",
            t=point.t,
            u=point.u,
            dv=point.dv,
            omega0=point.omega0,
            g_ep=point.g_ep,
            warm_ansatz="hh_hva_ptw",
            adapt_pool="paop_lf_std",
            adapt_continuation_mode="phase3_v1",
        ),
        run_dir=Path(f"run_{seed_index}"),
        handoff_json=Path(f"handoff_{seed_index}.json"),
        baseline_payload={},
        baseline_row={"seed_index": seed_index, "baseline_replay_delta_abs": delta},
        baseline_delta_abs=float(delta),
        weakest_adapt_unit=None,
        audit_extrema={},
    )


def test_resolve_screen_points_representative_and_custom_override() -> None:
    representative = wf.resolve_screen_points(point_preset="representative_6", raw_points=None, t=1.0, dv=0.0)
    assert len(representative) == 6
    assert representative[0] == wf.HamiltonianPoint(t=1.0, u=0.5, dv=0.0, g_ep=0.25, omega0=0.5)
    custom = wf.resolve_screen_points(
        point_preset="full_18",
        raw_points="0.5:0.25:2.0;8:2.0:0.5",
        t=1.0,
        dv=0.2,
    )
    assert custom == (
        wf.HamiltonianPoint(t=1.0, u=0.5, dv=0.2, g_ep=0.25, omega0=2.0),
        wf.HamiltonianPoint(t=1.0, u=8.0, dv=0.2, g_ep=2.0, omega0=0.5),
    )


def test_parse_cli_args_rejects_even_seed_count() -> None:
    with pytest.raises(ValueError, match="positive odd integer"):
        wf.parse_cli_args(["--seed-count", "4"])


def test_build_replay_ablation_plans_uses_final_order_index_and_dedups() -> None:
    handoff_payload = {
        "adapt_vqe": {
            "operators": ["op0", "op1", "op2", "op3", "op4"],
            "optimal_point": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    }
    plans = wf._build_replay_ablation_plans(
        handoff_payload=handoff_payload,
        weakest_adapt_unit={"final_order_index": 1, "unit_index": 99},
        include_prefix_50=True,
    )
    by_id = {plan.ablation_id: plan for plan in plans}
    assert set(by_id) == {
        "full_replay_baseline",
        "prefix_75",
        "prefix_50",
        "tail_drop_1",
        "drop_weakest_accepted",
    }
    assert by_id["drop_weakest_accepted"].removed_operator_indices == (1,)
    assert by_id["drop_weakest_accepted"].removed_label == "op1"

    deduped = wf._build_replay_ablation_plans(
        handoff_payload={
            "adapt_vqe": {
                "operators": ["op0", "op1", "op2", "op3"],
                "optimal_point": [0.1, 0.2, 0.3, 0.4],
            }
        },
        weakest_adapt_unit={"final_order_index": 3, "unit_index": 7},
        include_prefix_50=False,
    )
    deduped_ids = [plan.ablation_id for plan in deduped]
    assert deduped_ids == ["full_replay_baseline", "prefix_75"]


def test_select_median_records_is_deterministic_under_ties() -> None:
    point = wf.HamiltonianPoint(t=1.0, u=4.0, dv=0.0, omega0=1.0, g_ep=1.0)
    records = (
        _baseline_record(point, seed_index=3, delta=0.2),
        _baseline_record(point, seed_index=1, delta=0.1),
        _baseline_record(point, seed_index=2, delta=0.2),
    )
    selected = wf._select_median_records(records)
    assert len(selected) == 1
    assert selected[0].seeds.seed_index == 2
    assert selected[0].baseline_delta_abs == pytest.approx(0.2)



def test_build_screen_staged_cfg_applies_optional_budget_overrides(tmp_path: Path) -> None:
    screen_cfg = _screen_cfg(
        tmp_path,
        warm_vqe_reps_override=5,
        warm_vqe_restarts_override=6,
        warm_vqe_maxiter_override=2500,
        adapt_max_depth_override=120,
        adapt_drop_min_depth_override=30,
        adapt_maxiter_override=3333,
        final_vqe_reps_override=4,
        final_vqe_restarts_override=6,
        final_vqe_maxiter_override=2800,
    )
    point = screen_cfg.points[0]

    staged_cfg, _audit_cfg, seeds, _run_dir = wf.build_screen_staged_cfg(screen_cfg, point=point, seed_index=0)

    assert seeds.seed_index == 1
    assert staged_cfg.warm_start.reps == 5
    assert staged_cfg.warm_start.restarts == 6
    assert staged_cfg.warm_start.maxiter == 2500
    assert staged_cfg.adapt.max_depth == 120
    assert staged_cfg.adapt.drop_min_depth == 30
    assert staged_cfg.adapt.maxiter == 3333
    assert staged_cfg.replay.reps == 4
    assert staged_cfg.replay.restarts == 6
    assert staged_cfg.replay.maxiter == 2800


def test_run_screen_emits_baselines_and_median_seed_ablations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    screen_cfg = _screen_cfg(tmp_path)

    def _fake_run_stage_pipeline(staged_cfg):
        seed_index = 1 + int((int(staged_cfg.warm_start.seed) - 7) / 100)
        baseline_abs_delta = {1: 0.30, 2: 0.10, 3: 0.20}[seed_index]
        return SimpleNamespace(
            hmat=np.eye(2),
            ordered_labels_exyz=["ze"],
            coeff_map_exyz={"ze": 1.0},
            nq_total=1,
            psi_hf=np.array([1.0, 0.0], dtype=complex),
            psi_warm=np.array([1.0, 0.0], dtype=complex),
            psi_adapt=np.array([1.0, 0.0], dtype=complex),
            psi_final=np.array([1.0, 0.0], dtype=complex),
            warm_payload={"energy": -0.50, "exact_filtered_energy": -0.60},
            adapt_payload={
                "energy": -0.80,
                "exact_gs_energy": -0.90,
                "operators": ["op0", "op1", "op2", "op3", "op4"],
                "history": [
                    {"delta_abs_drop_from_prev": 0.12, "optimizer_memory_reused": True},
                    {"delta_abs_drop_from_prev": 0.04},
                    {"delta_abs_drop_from_prev": -0.01, "drop_low_signal": True},
                ],
                "continuation": {"rescue_history": [{"kind": "rescue"}]},
            },
            replay_payload={
                "vqe": {
                    "energy": -1.20 - 0.01 * seed_index,
                    "abs_delta_e": baseline_abs_delta,
                    "relative_error_abs": baseline_abs_delta,
                    "num_parameters": 5,
                    "runtime_s": 1.5,
                    "gate_pass_1e2": True,
                    "stop_reason": "ok",
                },
                "exact": {"E_exact_sector": -1.50},
            },
            warm_circuit_context=None,
            adapt_circuit_context=None,
            replay_circuit_context=None,
        )

    def _fake_build_audit_context(*, stage_result, staged_cfg, audit_cfg):
        wf._write_json(audit_cfg.output_json, {"rows": []})
        wf._write_csv(audit_cfg.output_csv, [{"stage": "adapt_vqe", "unit_index": 1}])
        return {
            "audit_json": Path(audit_cfg.output_json),
            "audit_csv": Path(audit_cfg.output_csv),
            "weakest_adapt_unit": {
                "unit_index": 7,
                "unit_label": "adapt_op_2",
                "base_label": "op1",
                "final_order_index": 1,
                "insertion_position": 1,
                "removal_penalty": 0.002,
                "delta_energy_from_previous": 0.003,
            },
            "accepted_insertion_count": 3,
            "seed_prefix_depth": 2,
            "final_adapt_depth": 5,
            "audit_extrema": {
                "warm": {"min_delta_energy_from_previous": 0.05, "min_removal_penalty": 0.06},
                "adapt": {"min_delta_energy_from_previous": 0.003, "min_removal_penalty": 0.002},
                "replay": {"min_delta_energy_from_previous": 0.004, "min_removal_penalty": 0.005},
            },
        }

    def _fake_replay_run(cfg):
        handoff_payload = json.loads(Path(cfg.adapt_input_json).read_text(encoding="utf-8"))
        ablation = handoff_payload["meta"]["logical_screen_ablation"]
        ablation_id = str(ablation["ablation_id"])
        op_count = len(handoff_payload["adapt_vqe"]["operators"])
        abs_delta = {
            "prefix_75": 0.12,
            "tail_drop_1": 0.11,
            "drop_weakest_accepted": 0.105,
        }[ablation_id]
        payload = {
            "vqe": {
                "energy": -1.10 - 0.01 * op_count,
                "abs_delta_e": abs_delta,
                "relative_error_abs": abs_delta,
                "num_parameters": op_count,
                "runtime_s": 0.25,
                "gate_pass_1e2": True,
                "stop_reason": "ok",
            },
            "exact": {"E_exact_sector": -1.50},
        }
        Path(cfg.output_json).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _fake_load_json(path: Path):
        return {
            "adapt_vqe": {
                "operators": ["op0", "op1", "op2", "op3", "op4"],
                "optimal_point": [0.1, 0.2, 0.3, 0.4, 0.5],
                "ansatz_depth": 5,
                "num_parameters": 5,
            },
            "continuation": {
                "selected_generator_metadata": [{"id": i} for i in range(5)],
                "replay_contract": {"reps": 1, "adapt_depth": 5, "derived_num_parameters": 5},
            },
            "meta": {},
        }

    monkeypatch.setattr(wf, "run_stage_pipeline", _fake_run_stage_pipeline)
    monkeypatch.setattr(wf, "_build_audit_context", _fake_build_audit_context)
    monkeypatch.setattr(wf.replay_mod, "run", _fake_replay_run)
    monkeypatch.setattr(wf, "_load_json", _fake_load_json)

    payload = wf.run_hh_l2_logical_screen(screen_cfg)

    assert Path(screen_cfg.output_json).exists()
    assert Path(screen_cfg.output_csv).exists()
    assert payload["screen_scope"]["patch_selection_enabled"] is False
    assert payload["screen_scope"]["median_seed_ablation_only"] is True

    rows = payload["rows"]
    baseline_rows = [row for row in rows if row["ablation_id"] == "full_replay_baseline"]
    ablation_rows = [row for row in rows if row["ablation_id"] != "full_replay_baseline"]

    assert len(baseline_rows) == 3
    assert len(ablation_rows) == 3
    assert {row["seed_role"] for row in baseline_rows} == {"best", "median", "worst"}
    assert {row["seed_index"] for row in ablation_rows} == {3}
    assert {row["seed_role"] for row in ablation_rows} == {"median"}
    assert {row["ablation_id"] for row in ablation_rows} == {
        "prefix_75",
        "tail_drop_1",
        "drop_weakest_accepted",
    }
    assert {row["final_adapt_depth"] for row in rows} == {5}
    assert {row["accepted_insertion_count"] for row in rows} == {3}
    assert {row["accepted_operator_count"] for row in rows} == {3}
    assert {row["seed_prefix_depth"] for row in rows} == {2}

    drop_row = next(row for row in ablation_rows if row["ablation_id"] == "drop_weakest_accepted")
    drop_payload = json.loads(Path(drop_row["handoff_input_json"]).read_text(encoding="utf-8"))
    assert drop_payload["adapt_vqe"]["operators"] == ["op0", "op2", "op3", "op4"]
    assert drop_payload["adapt_vqe"]["optimal_point"] == [0.1, 0.3, 0.4, 0.5]
    assert drop_payload["continuation"]["selected_generator_metadata"] == [{"id": 0}, {"id": 2}, {"id": 3}, {"id": 4}]
    assert drop_payload["continuation"]["replay_contract"]["adapt_depth"] == 4
    assert drop_payload["meta"]["logical_screen_ablation"]["removed_operator_indices"] == [1]

    with Path(screen_cfg.output_csv).open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 6
    assert payload["summary"]["stress_points"][0]["point_id"] == screen_cfg.points[0].point_id()
