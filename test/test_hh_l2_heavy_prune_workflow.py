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

import pipelines.exact_bench.hh_l2_heavy_prune_workflow as wf


def test_build_heavy_prune_staged_cfg_respects_seed_and_boundary_overrides(tmp_path: Path) -> None:
    cfg = wf.HeavyPruneConfig(
        output_json=tmp_path / "heavy.json",
        output_csv=tmp_path / "heavy.csv",
        run_root=tmp_path / "runs",
        tag="heavy_case",
        ordering="interleaved",
        boundary="open",
        warm_seed=101,
        adapt_seed=103,
        final_seed=107,
        adapt_max_depth_override=120,
    )

    staged_cfg, audit_cfg, run_dir = wf.build_heavy_prune_staged_cfg(cfg)

    assert run_dir == tmp_path / "runs" / "heavy_case"
    assert staged_cfg.physics.ordering == "interleaved"
    assert staged_cfg.physics.boundary == "open"
    assert staged_cfg.warm_start.seed == 101
    assert staged_cfg.adapt.seed == 103
    assert staged_cfg.replay.seed == 107
    assert staged_cfg.adapt.max_depth == 120
    assert audit_cfg.ordering == "interleaved"
    assert audit_cfg.boundary == "open"


def test_build_ranked_prune_plans_preserves_seed_prefix_and_ranking() -> None:
    handoff_payload = {
        "adapt_vqe": {
            "operators": ["seed0", "seed1", "acc0", "acc1", "acc2"],
            "optimal_point": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    }
    ranked_adapt_units = [
        {"unit_index": 2, "base_label": "acc1", "final_order_index": 3, "removal_penalty": 0.01, "delta_energy_from_previous": 0.02},
        {"unit_index": 3, "base_label": "acc2", "final_order_index": 4, "removal_penalty": 0.02, "delta_energy_from_previous": 0.03},
        {"unit_index": 1, "base_label": "acc0", "final_order_index": 2, "removal_penalty": 0.05, "delta_energy_from_previous": 0.06},
    ]

    plans = wf._build_ranked_prune_plans(
        handoff_payload=handoff_payload,
        ranked_adapt_units=ranked_adapt_units,
        seed_prefix_depth=2,
        include_prefix_50=True,
        weakest_single_count=3,
        weakest_cumulative_count=3,
    )
    by_id = {plan.ablation_id: plan for plan in plans}

    assert "accepted_prefix_75" not in by_id
    assert "accepted_prefix_50" not in by_id
    assert "tail_drop_1" not in by_id
    assert by_id["drop_weakest_accepted"].removed_operator_indices == (3,)
    assert by_id["drop_ranked_weakest_02"].removed_operator_indices == (4,)
    assert by_id["drop_ranked_cumulative_02"].removed_operator_indices == (3, 4)
    assert by_id["drop_ranked_cumulative_03"].removed_operator_indices == (2, 3, 4)
    assert [entry["base_label"] for entry in by_id["drop_ranked_weakest_02"].ranking_entries] == ["acc2"]
    assert [entry["base_label"] for entry in by_id["drop_ranked_cumulative_02"].ranking_entries] == ["acc1", "acc2"]


def test_run_heavy_prune_emits_rows_and_pareto(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = wf.HeavyPruneConfig(
        output_json=tmp_path / "heavy.json",
        output_csv=tmp_path / "heavy.csv",
        run_root=tmp_path / "runs",
        tag="heavy_test",
        weakest_single_count=2,
        weakest_cumulative_count=2,
        include_prefix_50=False,
    )

    def _fake_run_stage_pipeline(staged_cfg):
        return SimpleNamespace(
            hmat=np.eye(2),
            ordered_labels_exyz=["ze"],
            coeff_map_exyz={"ze": 1.0},
            nq_total=1,
            psi_hf=np.array([1.0, 0.0], dtype=complex),
            psi_warm=np.array([1.0, 0.0], dtype=complex),
            psi_adapt=np.array([1.0, 0.0], dtype=complex),
            psi_final=np.array([1.0, 0.0], dtype=complex),
            warm_payload={"energy": -0.5, "exact_filtered_energy": -0.6},
            adapt_payload={
                "energy": -0.9,
                "exact_gs_energy": -1.0,
                "operators": ["seed0", "seed1", "acc0", "acc1", "acc2"],
                "history": [
                    {"delta_abs_drop_from_prev": 0.12},
                    {"delta_abs_drop_from_prev": 0.05},
                    {"delta_abs_drop_from_prev": 0.02},
                ],
                "continuation": {"rescue_history": []},
            },
            replay_payload={
                "vqe": {
                    "energy": -1.2,
                    "abs_delta_e": 0.05,
                    "relative_error_abs": 0.05,
                    "num_parameters": 5,
                    "runtime_s": 1.2,
                    "gate_pass_1e2": False,
                    "stop_reason": "baseline",
                },
                "exact": {"E_exact_sector": -1.25},
            },
            warm_circuit_context=None,
            adapt_circuit_context=None,
            replay_circuit_context=None,
        )

    def _fake_build_audit_context(*, stage_result, staged_cfg, audit_cfg):
        Path(audit_cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(audit_cfg.output_json).write_text(json.dumps({"rows": []}), encoding="utf-8")
        with Path(audit_cfg.output_csv).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stage"])
            writer.writeheader()
            writer.writerow({"stage": "adapt_vqe"})
        return {
            "audit_json": Path(audit_cfg.output_json),
            "audit_csv": Path(audit_cfg.output_csv),
            "rows_by_stage": {"adapt_vqe": []},
            "ranked_adapt_units": [
                {"unit_index": 2, "unit_label": "acc1", "base_label": "acc1", "final_order_index": 3, "insertion_position": 3, "removal_penalty": 0.01, "delta_energy_from_previous": 0.02},
                {"unit_index": 3, "unit_label": "acc2", "base_label": "acc2", "final_order_index": 4, "insertion_position": 4, "removal_penalty": 0.02, "delta_energy_from_previous": 0.03},
                {"unit_index": 1, "unit_label": "acc0", "base_label": "acc0", "final_order_index": 2, "insertion_position": 2, "removal_penalty": 0.05, "delta_energy_from_previous": 0.06},
            ],
            "weakest_adapt_unit": {"unit_index": 2, "base_label": "acc1", "final_order_index": 3, "removal_penalty": 0.01, "delta_energy_from_previous": 0.02},
            "adapt_stage_metadata": {"seed_prefix_depth": 2},
            "seed_prefix_depth": 2,
            "accepted_insertion_count": 3,
            "final_adapt_depth": 5,
            "audit_extrema": {},
        }

    def _fake_load_json(path: Path):
        return {
            "adapt_vqe": {
                "operators": ["seed0", "seed1", "acc0", "acc1", "acc2"],
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

    def _fake_build_stage_circuit_costs(stage_result, cfg_local):
        return {
            "adapt_vqe": {"transpiled": {"depth": 120, "size": 180, "cx_count": 30, "one_q_count": 90}},
            "conventional_replay": {"transpiled": {"depth": 150, "size": 200, "cx_count": 40, "one_q_count": 100}},
        }

    def _fake_replay_run(run_cfg, diagnostics_out=None):
        payload_in = json.loads(Path(run_cfg.adapt_input_json).read_text(encoding="utf-8"))
        ablation = payload_in["meta"]["heavy_prune_ablation"]
        ablation_id = str(ablation["ablation_id"])
        mapping = {
            "drop_weakest_accepted": (0.045, 28, 120),
            "drop_ranked_weakest_02": (0.052, 30, 128),
            "drop_ranked_cumulative_02": (0.070, 24, 110),
        }
        delta_abs, cx_count, depth = mapping[ablation_id]
        if diagnostics_out is not None:
            diagnostics_out.update({"ablation_id": ablation_id, "cx_count": cx_count, "depth": depth})
        payload = {
            "vqe": {
                "energy": -1.25 + delta_abs,
                "abs_delta_e": delta_abs,
                "relative_error_abs": delta_abs,
                "num_parameters": len(payload_in["adapt_vqe"]["operators"]),
                "runtime_s": 0.25,
                "gate_pass_1e2": bool(delta_abs <= 1e-2),
                "stop_reason": "ok",
            },
            "exact": {"E_exact_sector": -1.25},
        }
        Path(run_cfg.output_json).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _fake_replay_cost(diag, *, basis_gates, optimization_level):
        _ = basis_gates, optimization_level
        return {
            "transpiled": {
                "depth": int(diag["depth"]),
                "size": int(diag["depth"] + diag["cx_count"]),
                "cx_count": int(diag["cx_count"]),
                "one_q_count": 100,
            }
        }

    monkeypatch.setattr(wf, "run_stage_pipeline", _fake_run_stage_pipeline)
    monkeypatch.setattr(wf.logical_wf, "_build_audit_context", _fake_build_audit_context)
    monkeypatch.setattr(wf.logical_wf, "_load_json", _fake_load_json)
    monkeypatch.setattr(wf, "_build_stage_circuit_costs", _fake_build_stage_circuit_costs)
    monkeypatch.setattr(wf.replay_mod, "run", _fake_replay_run)
    monkeypatch.setattr(wf, "_replay_circuit_cost_from_diagnostics", _fake_replay_cost)

    payload = wf.run_hh_l2_heavy_prune(cfg)

    assert Path(cfg.output_json).exists()
    assert Path(cfg.output_csv).exists()
    assert payload["scope"]["pruning_level"] == "accepted_adapt_units_only"
    assert payload["baseline"]["seed_prefix_depth"] == 2
    assert len(payload["rows"]) == 4
    assert {row["ablation_id"] for row in payload["rows"]} == {
        "full_replay_baseline",
        "drop_weakest_accepted",
        "drop_ranked_weakest_02",
        "drop_ranked_cumulative_02",
    }
    best = payload["summary"]["best_delta_abs_row"]
    assert best["ablation_id"] == "drop_weakest_accepted"
    pareto_ids = {row["ablation_id"] for row in payload["summary"]["pareto_front"]}
    assert "drop_weakest_accepted" in pareto_ids

    cumulative_row = next(row for row in payload["rows"] if row["ablation_id"] == "drop_ranked_cumulative_02")
    assert cumulative_row["removed_operator_indices"] == [3, 4]
    handoff_payload = json.loads(Path(cumulative_row["handoff_input_json"]).read_text(encoding="utf-8"))
    assert handoff_payload["adapt_vqe"]["operators"] == ["seed0", "seed1", "acc0"]
