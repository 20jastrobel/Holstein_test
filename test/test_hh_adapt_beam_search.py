from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.adapt_pipeline import _BeamBranchState, _run_hardcoded_adapt_vqe
from pipelines.hardcoded.hh_continuation_scoring import MeasurementCacheAudit
from pipelines.hardcoded.hh_continuation_stage_control import StageController, StageControllerConfig
from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian


def _make_branch() -> _BeamBranchState:
    stage = StageController(StageControllerConfig())
    stage.start_with_seed()
    measure_cache = MeasurementCacheAudit(nominal_shots_per_group=4)
    measure_cache.commit(["parent_group"])
    return _BeamBranchState(
        branch_id=11,
        parent_branch_id=None,
        depth_local=2,
        terminated=False,
        stop_reason=None,
        selected_ops=[SimpleNamespace(label="op_a"), SimpleNamespace(label="op_b")],
        theta=np.array([0.1, 0.2], dtype=float),
        energy_current=-1.25,
        available_indices={1, 2, 3},
        selection_counts=np.array([0, 1, 0, 2], dtype=np.int64),
        history=[{"depth": 1, "nested": {"path": ["parent"]}}],
        phase1_stage=stage,
        phase1_residual_opened=False,
        phase1_last_probe_reason="append_only",
        phase1_last_positions_considered=[2],
        phase1_last_trough_detected=False,
        phase1_last_trough_probe_triggered=False,
        phase1_last_selected_score=0.5,
        phase1_features_history=[{"nested": {"scores": [1.0]}}],
        phase1_stage_events=[{"meta": {"reason": "seed_complete"}}],
        phase1_measure_cache=measure_cache,
        phase2_optimizer_memory={"nested": {"weights": [1.0, 2.0]}, "slots": {"0": {"mean": 0.1}}},
        phase2_last_shortlist_records=[{"ids": [1], "feature": {"simple_score": 1.0}}],
        phase2_last_batch_selected=False,
        phase2_last_batch_penalty_total=0.0,
        phase2_last_optimizer_memory_reused=False,
        phase2_last_optimizer_memory_source="parent",
        phase2_last_shortlist_eval_records=[{"eval": {"kept": True}}],
        drop_prev_delta_abs=0.25,
        drop_plateau_hits=1,
        eps_energy_low_streak=0,
        phase3_split_events=[{"meta": {"selected": ["x"]}}],
        phase3_runtime_split_summary={"counts": {"selected": 1}, "selected_child_labels": ["x"]},
        phase3_motif_usage={"used": {"motif_a": 1}},
        phase3_rescue_history=[{"rescue": {"winner": "x"}}],
        nfev_total_local=17,
    )


def _hh_h() -> object:
    return build_hubbard_holstein_hamiltonian(
        dims=2,
        J=1.0,
        U=2.0,
        omega0=1.0,
        g=1.0,
        n_ph_max=1,
        boson_encoding="binary",
        v_t=None,
        v0=0.0,
        t_eval=None,
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
    )


def test_beam_branch_clone_for_child_isolates_branch_owned_state() -> None:
    parent = _make_branch()
    child = parent.clone_for_child(branch_id=12)

    child.available_indices.remove(2)
    child.selection_counts[1] = 99
    child.theta[0] = 9.0
    child.history[0]["nested"]["path"].append("child")
    child.phase1_features_history[0]["nested"]["scores"].append(2.0)
    child.phase1_stage_events[0]["meta"]["reason"] = "child_only"
    child.phase1_measure_cache.commit(["child_group"])
    child.phase2_optimizer_memory["nested"]["weights"].append(3.0)
    child.phase2_last_shortlist_records[0]["ids"].append(2)
    child.phase2_last_shortlist_eval_records[0]["eval"]["kept"] = False
    child.phase3_split_events[0]["meta"]["selected"].append("y")
    child.phase3_runtime_split_summary["counts"]["selected"] = 7
    child.phase3_motif_usage["used"]["motif_a"] = 9
    child.phase3_rescue_history[0]["rescue"]["winner"] = "child"
    child.phase1_stage.begin_core()

    assert child.branch_id == 12
    assert child.parent_branch_id == parent.branch_id
    assert child.selected_ops is not parent.selected_ops
    assert child.selected_ops[0] is parent.selected_ops[0]
    assert parent.available_indices == {1, 2, 3}
    assert parent.selection_counts.tolist() == [0, 1, 0, 2]
    assert parent.theta.tolist() == [0.1, 0.2]
    assert parent.history[0]["nested"]["path"] == ["parent"]
    assert parent.phase1_features_history[0]["nested"]["scores"] == [1.0]
    assert parent.phase1_stage_events[0]["meta"]["reason"] == "seed_complete"
    assert parent.phase1_measure_cache.summary()["groups_known"] == 1.0
    assert parent.phase2_optimizer_memory["nested"]["weights"] == [1.0, 2.0]
    assert parent.phase2_last_shortlist_records[0]["ids"] == [1]
    assert parent.phase2_last_shortlist_eval_records[0]["eval"]["kept"] is True
    assert parent.phase3_split_events[0]["meta"]["selected"] == ["x"]
    assert parent.phase3_runtime_split_summary["counts"]["selected"] == 1
    assert parent.phase3_motif_usage["used"]["motif_a"] == 1
    assert parent.phase3_rescue_history[0]["rescue"]["winner"] == "x"
    assert parent.phase1_stage.stage_name == "seed"


def test_beam_branch_sibling_clones_diverge_independently() -> None:
    parent = _make_branch()
    child_a = parent.clone_for_child(branch_id=21)
    child_b = parent.clone_for_child(branch_id=22)

    child_a.available_indices.remove(1)
    child_b.available_indices.remove(3)
    child_a.selection_counts[0] = 5
    child_b.selection_counts[0] = 8
    child_a.phase2_optimizer_memory["nested"]["weights"].append(10.0)
    child_b.phase2_optimizer_memory["nested"]["weights"].append(20.0)
    child_a.phase1_measure_cache.commit(["a_only"])
    child_b.phase1_measure_cache.commit(["b_only"])
    child_a.phase1_stage.begin_core()

    assert parent.available_indices == {1, 2, 3}
    assert child_a.available_indices == {2, 3}
    assert child_b.available_indices == {1, 2}
    assert parent.selection_counts.tolist() == [0, 1, 0, 2]
    assert child_a.selection_counts.tolist() == [5, 1, 0, 2]
    assert child_b.selection_counts.tolist() == [8, 1, 0, 2]
    assert parent.phase2_optimizer_memory["nested"]["weights"] == [1.0, 2.0]
    assert child_a.phase2_optimizer_memory["nested"]["weights"] == [1.0, 2.0, 10.0]
    assert child_b.phase2_optimizer_memory["nested"]["weights"] == [1.0, 2.0, 20.0]
    assert parent.phase1_measure_cache.summary()["groups_known"] == 1.0
    assert child_a.phase1_measure_cache.summary()["groups_known"] == 2.0
    assert child_b.phase1_measure_cache.summary()["groups_known"] == 2.0
    assert parent.phase1_stage.stage_name == "seed"
    assert child_a.phase1_stage.stage_name == "core"
    assert child_b.phase1_stage.stage_name == "seed"


def test_true_beam_defaults_are_exposed_and_winner_history_stays_singleton() -> None:
    diagnostics: dict[str, object] = {}
    payload, _psi = _run_hardcoded_adapt_vqe(
        h_poly=_hh_h(),
        num_sites=2,
        ordering="blocked",
        problem="hh",
        adapt_pool="paop_lf_std",
        t=1.0,
        u=2.0,
        dv=0.0,
        boundary="open",
        omega0=1.0,
        g_ep=1.0,
        n_ph_max=1,
        boson_encoding="binary",
        max_depth=1,
        eps_grad=1e-2,
        eps_energy=1e-6,
        maxiter=5,
        seed=7,
        allow_repeats=True,
        finite_angle_fallback=False,
        finite_angle=0.1,
        finite_angle_min_improvement=0.0,
        adapt_continuation_mode="phase3_v1",
        adapt_beam_live_branches=3,
        diagnostics_out=diagnostics,
    )

    beam_policy = diagnostics["beam_policy"]
    beam_search = diagnostics["beam_search"]
    assert beam_policy["beam_enabled"] is True
    assert int(beam_policy["live_branches_effective"]) == 3
    assert int(beam_policy["children_per_parent_effective"]) == 2
    assert int(beam_policy["terminated_keep_effective"]) == 3
    assert beam_search["beam_enabled"] is True
    assert beam_search["finalist_count"] >= 1
    assert isinstance(beam_search["rounds"], list)
    assert len(beam_search["rounds"]) >= 1
    round0 = beam_search["rounds"][0]
    assert int(round0["frontier_input_count"]) == 1
    assert int(round0["parents_expanded_count"]) == 1
    assert int(round0["children_materialized_count"]) == int(round0["active_children_raw_count"]) + int(
        round0["round_terminals_raw_count"]
    )
    assert int(round0["active_children_unique_count"]) <= int(round0["active_children_raw_count"])
    assert int(round0["frontier_kept_count"]) <= int(beam_policy["live_branches_effective"])
    assert int(round0["terminal_kept_count"]) <= int(beam_policy["terminated_keep_effective"])
    assert payload["adapt_beam_enabled"] is True
    assert int(payload["adapt_beam_live_branches"]) == 3
    assert int(payload["adapt_beam_children_per_parent"]) == 2
    assert int(payload["adapt_beam_terminated_keep"]) == 3
    assert isinstance(payload["operators"], list)
    assert isinstance(payload["optimal_point"], list)
    assert len(payload["optimal_point"]) == len(payload["operators"])
    for row in payload.get("history", []):
        assert row["selected_positions"] == [row["selected_position"]]
        assert bool(row["batch_selected"]) is False
