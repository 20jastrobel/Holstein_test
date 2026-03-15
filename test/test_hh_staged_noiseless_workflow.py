from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.hardcoded.hh_staged_workflow as wf
from pipelines.hardcoded.hh_staged_noiseless import parse_args
from pipelines.hardcoded.hh_staged_workflow import resolve_staged_hh_config


def _basis(dim: int, idx: int) -> np.ndarray:
    out = np.zeros(dim, dtype=complex)
    out[int(idx)] = 1.0
    return out


def _amplitudes_qn_to_q0(psi: np.ndarray) -> dict[str, dict[str, float]]:
    nq = int(round(np.log2(int(np.asarray(psi).size))))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(np.asarray(psi, dtype=complex).reshape(-1)):
        if abs(amp) <= 1e-14:
            continue
        out[format(idx, f"0{nq}b")] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


def test_resolve_staged_defaults_from_run_guide_formulae() -> None:
    args = parse_args(["--L", "3", "--skip-pdf"])
    cfg = resolve_staged_hh_config(args)

    assert str(cfg.warm_start.ansatz_name) == "hh_hva_ptw"
    assert int(cfg.warm_start.reps) == 3
    assert int(cfg.warm_start.restarts) == 5
    assert int(cfg.warm_start.maxiter) == 4000
    assert int(cfg.adapt.max_depth) == 120
    assert int(cfg.adapt.maxiter) == 5000
    assert float(cfg.adapt.eps_grad) == pytest.approx(5e-7)
    assert float(cfg.adapt.eps_energy) == pytest.approx(1e-9)
    assert cfg.adapt.drop_floor is None
    assert cfg.adapt.drop_patience is None
    assert cfg.adapt.drop_min_depth is None
    assert cfg.adapt.grad_floor is None
    assert int(cfg.replay.reps) == 3
    assert int(cfg.replay.restarts) == 5
    assert int(cfg.replay.maxiter) == 4000
    assert bool(cfg.replay.enabled) is False
    assert cfg.seed_refine.family is None
    assert int(cfg.seed_refine.reps) == 3
    assert int(cfg.seed_refine.maxiter) == 4000
    assert str(cfg.seed_refine.optimizer) == "SPSA"
    assert str(cfg.replay.continuation_mode) == "phase3_v1"
    assert bool(cfg.dynamics.enabled) is False
    assert int(cfg.dynamics.t_final) == 15
    assert int(cfg.dynamics.trotter_steps) == 192
    assert int(cfg.dynamics.num_times) == 201
    assert int(cfg.dynamics.exact_steps_multiplier) == 2
    assert float(cfg.gates.ecut_1) == pytest.approx(1e-1)
    assert float(cfg.gates.ecut_2) == pytest.approx(1e-4)
    assert bool(cfg.dynamics.enable_drive) is False
    assert cfg.default_provenance["warm_ansatz"] == "workflow.warm_ansatz.default=hh_hva_ptw"
    assert cfg.default_provenance["warm_reps"] == "run_guide.ws_reps(L)=L"
    assert cfg.default_provenance["replay_continuation_mode"] == "workflow.replay_mode := adapt_continuation_mode"


def test_replay_and_dynamics_flags_roundtrip() -> None:
    cfg = resolve_staged_hh_config(
        parse_args(["--L", "2", "--skip-pdf", "--run-replay", "--run-dynamics"])
    )

    assert bool(cfg.replay.enabled) is True
    assert bool(cfg.dynamics.enabled) is True


def test_warm_ansatz_override_is_resolved_and_retagged() -> None:
    cfg_default = resolve_staged_hh_config(parse_args(["--L", "2", "--skip-pdf"]))
    cfg_layerwise = resolve_staged_hh_config(parse_args(["--L", "2", "--warm-ansatz", "hh_hva", "--skip-pdf"]))

    assert str(cfg_layerwise.warm_start.ansatz_name) == "hh_hva"
    assert str(cfg_default.warm_start.ansatz_name) == "hh_hva_ptw"
    assert str(cfg_layerwise.artifacts.tag) != str(cfg_default.artifacts.tag)
    assert "warmhh_hva" in str(cfg_layerwise.artifacts.tag)


def test_seed_refine_cli_fields_roundtrip() -> None:
    cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--seed-refine-family",
                "uccsd_otimes_paop_lf_std",
                "--seed-refine-reps",
                "3",
                "--seed-refine-maxiter",
                "2000",
                "--seed-refine-optimizer",
                "SPSA",
            ]
        )
    )

    assert str(cfg.seed_refine.family) == "uccsd_otimes_paop_lf_std"
    assert int(cfg.seed_refine.reps) == 3
    assert int(cfg.seed_refine.maxiter) == 2000
    assert str(cfg.seed_refine.optimizer) == "SPSA"
    assert "refineuccsd_otimes_paop_lf_std" in str(cfg.artifacts.tag)


def test_adapt_beam_capacity_cli_fields_roundtrip() -> None:
    cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--adapt-beam-live-branches",
                "4",
                "--adapt-beam-children-per-parent",
                "3",
                "--adapt-beam-terminated-keep",
                "5",
            ]
        )
    )

    assert int(cfg.adapt.beam_live_branches) == 4
    assert int(cfg.adapt.beam_children_per_parent) == 3
    assert int(cfg.adapt.beam_terminated_keep) == 5


def test_adapt_drop_policy_overrides_roundtrip() -> None:
    cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--adapt-drop-floor",
                "1e-3",
                "--adapt-drop-patience",
                "5",
                "--adapt-drop-min-depth",
                "20",
                "--adapt-grad-floor",
                "-1",
                "--skip-pdf",
            ]
        )
    )

    assert cfg.adapt.drop_floor == pytest.approx(1e-3)
    assert cfg.adapt.drop_patience == 5
    assert cfg.adapt.drop_min_depth == 20
    assert cfg.adapt.grad_floor == pytest.approx(-1.0)


def test_warm_checkpoint_cli_fields_roundtrip(tmp_path: Path) -> None:
    resume_json = tmp_path / "resume.json"
    handoff_json = tmp_path / "handoff.json"
    with pytest.raises(ValueError, match="Use either --resume-from-warm-checkpoint or --handoff-from-warm-checkpoint"):
        resolve_staged_hh_config(
            parse_args(
                [
                    "--L",
                    "2",
                    "--skip-pdf",
                    "--warm-stop-energy",
                    "-0.9",
                    "--warm-stop-delta-abs",
                    "0.1",
                    "--state-export-dir",
                    str(tmp_path),
                    "--state-export-prefix",
                    "warm_case",
                    "--resume-from-warm-checkpoint",
                    str(resume_json),
                    "--handoff-from-warm-checkpoint",
                    str(handoff_json),
                ]
            )
        )

    cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--warm-stop-energy",
                "-0.9",
                "--warm-stop-delta-abs",
                "0.1",
                "--state-export-dir",
                str(tmp_path),
                "--state-export-prefix",
                "warm_case",
                "--handoff-from-warm-checkpoint",
                str(handoff_json),
            ]
        )
    )

    assert cfg.warm_checkpoint.stop_energy == pytest.approx(-0.9)
    assert cfg.warm_checkpoint.stop_delta_abs == pytest.approx(0.1)
    assert cfg.warm_checkpoint.state_export_dir == tmp_path
    assert cfg.warm_checkpoint.state_export_prefix == "warm_case"
    assert cfg.warm_checkpoint.resume_from_warm_checkpoint is None
    assert cfg.warm_checkpoint.handoff_from_warm_checkpoint == handoff_json
    assert cfg.artifacts.warm_checkpoint_json == tmp_path / "warm_case_warm_checkpoint_state.json"
    assert cfg.artifacts.warm_cutover_json == tmp_path / "warm_case_warm_cutover_state.json"


def test_fixed_final_state_cli_fields_roundtrip(tmp_path: Path) -> None:
    fixed_json = tmp_path / "fixed_seed.json"
    cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--fixed-final-state-json",
                str(fixed_json),
                "--circuit-backend-name",
                "FakeGuadalupeV2",
                "--circuit-use-fake-backend",
                "--circuit-transpile-optimization-level",
                "2",
                "--circuit-seed-transpiler",
                "99",
            ]
        )
    )

    assert cfg.fixed_final_state is not None
    assert cfg.fixed_final_state.json_path == fixed_json
    assert cfg.fixed_final_state.strict_match is True
    assert cfg.circuit_metrics.backend_name == "FakeGuadalupeV2"
    assert cfg.circuit_metrics.use_fake_backend is True
    assert cfg.circuit_metrics.optimization_level == 2
    assert cfg.circuit_metrics.seed_transpiler == 99


def test_nondefault_sector_override_rejected_cleanly() -> None:
    args = parse_args(["--L", "2", "--sector-n-up", "2", "--skip-pdf"])
    with pytest.raises(ValueError, match="half-filled sector"):
        resolve_staged_hh_config(args)



def test_staged_hh_parse_rejects_unsupported_warm_and_final_methods() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--L", "2", "--warm-method", "COBYLA", "--skip-pdf"])
    with pytest.raises(SystemExit):
        parse_args(["--L", "2", "--final-method", "COBYLA", "--skip-pdf"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--L",
                "2",
                "--seed-refine-family",
                "uccsd_otimes_paop_lf_std",
                "--seed-refine-optimizer",
                "COBYLA",
                "--skip-pdf",
            ]
        )


def test_staged_hh_parse_accepts_powell_methods() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--warm-method",
            "Powell",
            "--seed-refine-family",
            "uccsd_otimes_paop_lf_std",
            "--seed-refine-optimizer",
            "Powell",
            "--adapt-inner-optimizer",
            "Powell",
            "--final-method",
            "Powell",
            "--skip-pdf",
        ]
    )
    assert str(args.warm_method) == "Powell"
    assert str(args.seed_refine_optimizer) == "Powell"
    assert str(args.adapt_inner_optimizer) == "Powell"
    assert str(args.final_method) == "Powell"


def test_underparameterized_override_rejected_without_smoke_flag() -> None:
    args = parse_args([
        "--L",
        "2",
        "--warm-reps",
        "1",
        "--skip-pdf",
    ])
    with pytest.raises(ValueError, match="Under-parameterized staged HH run rejected"):
        resolve_staged_hh_config(args)

    args = parse_args([
        "--L",
        "2",
        "--warm-reps",
        "1",
        "--skip-pdf",
        "--smoke-test-intentionally-weak",
    ])
    cfg = resolve_staged_hh_config(args)
    assert int(cfg.warm_start.reps) == 1


def test_warm_stage_checkpoint_cutover_and_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dim = 1 << int(wf._hh_nq_total(2, 1, "binary"))
    psi_hf = _basis(dim, 0)
    psi_warm = _basis(dim, 1)

    class _FakeAnsatz:
        def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
            return np.array(psi_warm, copy=True)

    def _fake_run_hardcoded_vqe(**kwargs):
        observer = kwargs.get("progress_observer")
        if observer is not None:
            observer(
                {
                    "event": "new_best",
                    "restart_index": 1,
                    "restarts_total": 1,
                    "nfev_so_far": 7,
                    "energy_best_global": -0.9,
                    "theta_restart_best": [0.1],
                }
            )
        return {
            "success": True,
            "ansatz": str(kwargs["ansatz_name"]),
            "optimizer_method": str(kwargs["method"]),
            "energy": -0.9,
            "exact_filtered_energy": -1.0,
            "message": "early_stop_checker_returning_best_restart",
            "optimal_point": [0.1],
        }, np.array(psi_warm, copy=True)

    monkeypatch.setattr(wf, "_build_hh_warm_ansatz", lambda _cfg: _FakeAnsatz())
    monkeypatch.setattr(wf, "exact_ground_energy_sector_hh", lambda *args, **kwargs: -1.0)
    monkeypatch.setattr(wf.hc_pipeline, "_run_hardcoded_vqe", _fake_run_hardcoded_vqe)

    cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--warm-stop-energy",
                "-0.9",
                "--state-export-dir",
                str(tmp_path),
                "--state-export-prefix",
                "warm_resume",
            ]
        )
    )
    warm_payload, psi_out, checkpoint_json = wf._run_warm_start_stage(cfg, h_poly=object(), psi_hf=psi_hf)

    assert np.allclose(psi_out, psi_warm)
    assert checkpoint_json == cfg.artifacts.warm_cutover_json
    assert warm_payload["cutoff_triggered"] is True
    assert Path(warm_payload["checkpoint_json_latest"]).exists()
    assert Path(warm_payload["checkpoint_json_used"]).exists()
    log_text = cfg.artifacts.workflow_log.read_text(encoding="utf-8")
    assert "warm_new_best_checkpoint" in log_text
    assert "warm_cutoff_triggered" in log_text

    cfg_resume = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--warm-stop-energy",
                "-0.9",
                "--resume-from-warm-checkpoint",
                str(checkpoint_json),
                "--state-export-dir",
                str(tmp_path),
                "--state-export-prefix",
                "warm_resume",
            ]
        )
    )
    resumed_payload, resumed_psi, resumed_json = wf._run_warm_start_stage(
        cfg_resume,
        h_poly=object(),
        psi_hf=psi_hf,
    )

    assert resumed_payload["resumed_from_checkpoint"] is True
    assert resumed_payload["cutoff_triggered"] is True
    assert resumed_json == checkpoint_json
    assert np.allclose(resumed_psi, psi_warm)


def test_warm_stage_resume_from_precutoff_checkpoint_continues_optimization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dim = 1 << int(wf._hh_nq_total(2, 1, "binary"))
    psi_hf = _basis(dim, 0)
    psi_seed = _basis(dim, 1)
    psi_warm = _basis(dim, 2)
    calls: dict[str, object] = {}

    class _FakeAnsatz:
        def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
            return np.array(psi_warm, copy=True)

    def _fake_run_hardcoded_vqe(**kwargs):
        calls["initial_point"] = kwargs.get("initial_point")
        observer = kwargs.get("progress_observer")
        if observer is not None:
            observer(
                {
                    "event": "new_best",
                    "restart_index": 1,
                    "restarts_total": 6,
                    "nfev_so_far": 12,
                    "energy_best_global": -0.92,
                    "theta_restart_best": [0.2],
                }
            )
        return {
            "success": True,
            "ansatz": str(kwargs["ansatz_name"]),
            "optimizer_method": str(kwargs["method"]),
            "energy": -0.92,
            "exact_filtered_energy": -1.0,
            "message": "early_stop_checker_returning_best_restart",
            "optimal_point": [0.2],
        }, np.array(psi_warm, copy=True)

    seed_cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--warm-stop-energy",
                "-0.9",
                "--state-export-dir",
                str(tmp_path),
                "--state-export-prefix",
                "seed",
            ]
        )
    )
    wf._write_warm_checkpoint_bundle(
        seed_cfg,
        path=seed_cfg.artifacts.warm_checkpoint_json,
        psi_state=psi_seed,
        energy=-0.8,
        exact_filtered_energy=-1.0,
        theta=[0.1],
        role="warm_checkpoint",
        cutoff_status=wf._warm_stop_status(seed_cfg, energy=-0.8, exact_filtered_energy=-1.0),
    )

    monkeypatch.setattr(wf, "_build_hh_warm_ansatz", lambda _cfg: _FakeAnsatz())
    monkeypatch.setattr(wf.hc_pipeline, "_run_hardcoded_vqe", _fake_run_hardcoded_vqe)

    resume_cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--warm-stop-energy",
                "-0.9",
                "--resume-from-warm-checkpoint",
                str(seed_cfg.artifacts.warm_checkpoint_json),
                "--state-export-dir",
                str(tmp_path),
                "--state-export-prefix",
                "resume",
            ]
        )
    )
    warm_payload, psi_out, checkpoint_json = wf._run_warm_start_stage(
        resume_cfg,
        h_poly=object(),
        psi_hf=psi_hf,
    )

    assert list(calls["initial_point"]) == pytest.approx([0.1])
    assert warm_payload["resumed_from_checkpoint"] is True
    assert warm_payload["resume_checkpoint_json"] == str(seed_cfg.artifacts.warm_checkpoint_json)
    assert warm_payload["checkpoint_json_latest"] == str(resume_cfg.artifacts.warm_checkpoint_json)
    assert warm_payload["checkpoint_json_used"] == str(resume_cfg.artifacts.warm_cutover_json)
    assert checkpoint_json == resume_cfg.artifacts.warm_cutover_json
    assert np.allclose(psi_out, psi_warm)
    assert Path(resume_cfg.artifacts.warm_checkpoint_json).exists()
    assert Path(resume_cfg.artifacts.warm_cutover_json).exists()
    log_text = resume_cfg.artifacts.workflow_log.read_text(encoding="utf-8")
    assert "warm_resume_checkpoint_continue" in log_text
    assert "warm_cutoff_triggered" in log_text


def test_warm_stage_handoff_from_checkpoint_skips_warm_optimization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dim = 1 << int(wf._hh_nq_total(2, 1, "binary"))
    psi_hf = _basis(dim, 0)
    psi_seed = _basis(dim, 1)
    call_count = {"warm_vqe": 0}

    def _unexpected_warm_run(**kwargs):
        call_count["warm_vqe"] += 1
        raise AssertionError("warm VQE should not run when handoffing from a checkpoint")

    seed_cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--warm-stop-energy",
                "-0.9",
                "--state-export-dir",
                str(tmp_path),
                "--state-export-prefix",
                "seed_handoff",
            ]
        )
    )
    wf._write_warm_checkpoint_bundle(
        seed_cfg,
        path=seed_cfg.artifacts.warm_checkpoint_json,
        psi_state=psi_seed,
        energy=-0.8,
        exact_filtered_energy=-1.0,
        theta=[0.1],
        role="warm_checkpoint",
        cutoff_status=wf._warm_stop_status(seed_cfg, energy=-0.8, exact_filtered_energy=-1.0),
    )
    monkeypatch.setattr(wf.hc_pipeline, "_run_hardcoded_vqe", _unexpected_warm_run)

    handoff_cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--warm-stop-energy",
                "-0.9",
                "--handoff-from-warm-checkpoint",
                str(seed_cfg.artifacts.warm_checkpoint_json),
                "--state-export-dir",
                str(tmp_path),
                "--state-export-prefix",
                "handoff_only",
            ]
        )
    )
    warm_payload, psi_out, checkpoint_json = wf._run_warm_start_stage(
        handoff_cfg,
        h_poly=object(),
        psi_hf=psi_hf,
    )

    assert call_count["warm_vqe"] == 0
    assert warm_payload["handoff_from_checkpoint"] is True
    assert warm_payload["resumed_from_checkpoint"] is False
    assert warm_payload["handoff_checkpoint_json"] == str(seed_cfg.artifacts.warm_checkpoint_json)
    assert warm_payload["checkpoint_json_used"] == str(handoff_cfg.artifacts.warm_cutover_json)
    assert checkpoint_json == handoff_cfg.artifacts.warm_cutover_json
    assert np.allclose(psi_out, psi_seed)
    log_text = handoff_cfg.artifacts.workflow_log.read_text(encoding="utf-8")
    assert "warm_handoff_checkpoint_loaded" in log_text
    assert "warm_stage_complete" in log_text


def test_warm_stage_below_cutoff_still_writes_cutover_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dim = 1 << int(wf._hh_nq_total(2, 1, "binary"))
    psi_hf = _basis(dim, 0)
    psi_warm = _basis(dim, 1)

    class _FakeAnsatz:
        def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
            return np.array(psi_warm, copy=True)

    def _fake_run_hardcoded_vqe(**kwargs):
        return {
            "success": True,
            "ansatz": str(kwargs["ansatz_name"]),
            "optimizer_method": str(kwargs["method"]),
            "energy": -0.8,
            "exact_filtered_energy": -1.0,
            "message": "warm_completed_below_cutoff",
            "optimal_point": [0.3],
        }, np.array(psi_warm, copy=True)

    monkeypatch.setattr(wf, "_build_hh_warm_ansatz", lambda _cfg: _FakeAnsatz())
    monkeypatch.setattr(wf.hc_pipeline, "_run_hardcoded_vqe", _fake_run_hardcoded_vqe)
    monkeypatch.setattr(wf, "exact_ground_energy_sector_hh", lambda *args, **kwargs: -1.0)

    cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--warm-stop-energy",
                "-0.9",
                "--state-export-dir",
                str(tmp_path),
                "--state-export-prefix",
                "below_cutoff",
            ]
        )
    )
    warm_payload, psi_out, checkpoint_json = wf._run_warm_start_stage(
        cfg,
        h_poly=object(),
        psi_hf=psi_hf,
    )

    assert warm_payload["cutoff_triggered"] is False
    assert warm_payload["checkpoint_json_used"] == str(cfg.artifacts.warm_cutover_json)
    assert checkpoint_json == cfg.artifacts.warm_cutover_json
    assert np.allclose(psi_out, psi_warm)
    assert Path(cfg.artifacts.warm_cutover_json).exists()
    log_text = cfg.artifacts.workflow_log.read_text(encoding="utf-8")
    assert "warm_cutoff_not_reached_continue" in log_text
    assert "warm_stage_complete" in log_text


def test_run_stage_pipeline_uses_fixed_final_state_and_skips_prep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nq_total = int(wf._hh_nq_total(2, 1, "binary"))
    dim = 1 << nq_total
    psi_hf = _basis(dim, 0)
    psi_seed = _basis(dim, 3)
    fixed_json = tmp_path / "fixed_seed.json"
    fixed_json.write_text(
        json.dumps(
            {
                "settings": {
                    "L": 2,
                    "problem": "hh",
                    "ordering": "blocked",
                    "boundary": "open",
                    "t": 1.0,
                    "u": 2.0,
                    "dv": 0.0,
                    "omega0": 1.0,
                    "g_ep": 1.0,
                    "n_ph_max": 1,
                    "boson_encoding": "binary",
                },
                "initial_state": {
                    "source": "adapt_vqe",
                    "amplitudes_qn_to_q0": _amplitudes_qn_to_q0(psi_seed),
                },
                "adapt_vqe": {
                    "energy": -1.01,
                    "exact_gs_energy": -1.02,
                },
                "ground_state": {
                    "exact_energy_filtered": -1.02,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        wf,
        "_build_hh_context",
        lambda _cfg: (
            object(),
            np.eye(dim, dtype=complex),
            ["eeeeee"],
            {"eeeeee": 1.0 + 0.0j},
            np.array(psi_hf, copy=True),
        ),
    )
    monkeypatch.setattr(wf, "_run_warm_start_stage", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("warm stage should be skipped")))
    monkeypatch.setattr(wf.adapt_mod, "_run_hardcoded_adapt_vqe", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ADAPT should be skipped")))
    monkeypatch.setattr(wf.replay_mod, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay should be skipped")))

    cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--fixed-final-state-json",
                str(fixed_json),
                "--tag",
                "fixed_seed_test",
                "--output-json",
                str(tmp_path / "workflow.json"),
                "--output-pdf",
                str(tmp_path / "workflow.pdf"),
            ]
        )
    )
    cfg = replace(
        cfg,
        artifacts=replace(
            cfg.artifacts,
            handoff_json=tmp_path / "handoff.json",
            replay_output_json=tmp_path / "replay.json",
        ),
    )
    stage_result = wf.run_stage_pipeline(cfg)

    assert np.allclose(stage_result.psi_final, psi_seed)
    assert np.allclose(stage_result.psi_warm, psi_seed)
    assert np.allclose(stage_result.psi_adapt, psi_seed)
    assert stage_result.fixed_final_state_import is not None
    assert stage_result.fixed_final_state_import["source_json"] == str(fixed_json)
    assert stage_result.warm_payload["skipped"] is True
    assert stage_result.adapt_payload["skipped"] is True
    assert stage_result.replay_payload["skipped"] is True
    assert cfg.artifacts.handoff_json.exists()
    assert cfg.artifacts.replay_output_json.exists()


def test_run_stage_pipeline_resolves_fixed_final_state_from_workflow_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nq_total = int(wf._hh_nq_total(2, 1, "binary"))
    dim = 1 << nq_total
    psi_hf = _basis(dim, 0)
    psi_seed = _basis(dim, 5)
    handoff_json = tmp_path / "handoff_seed.json"
    handoff_json.write_text(
        json.dumps(
            {
                "settings": {
                    "L": 2,
                    "problem": "hh",
                    "ordering": "blocked",
                    "boundary": "open",
                    "t": 1.0,
                    "u": 2.0,
                    "dv": 0.0,
                    "omega0": 1.0,
                    "g_ep": 1.0,
                    "n_ph_max": 1,
                    "boson_encoding": "binary",
                },
                "initial_state": {
                    "source": "fixed_final_state_import",
                    "amplitudes_qn_to_q0": _amplitudes_qn_to_q0(psi_seed),
                },
                "adapt_vqe": {
                    "energy": -1.015,
                    "exact_gs_energy": -1.02,
                },
                "ground_state": {
                    "exact_energy_filtered": -1.02,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    workflow_json = tmp_path / "workflow.json"
    workflow_json.write_text(
        json.dumps(
            {
                "artifacts": {
                    "intermediate": {
                        "adapt_handoff_json": str(handoff_json),
                    }
                },
                "stage_pipeline": {
                    "adapt_vqe": {
                        "handoff_json": str(handoff_json),
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        wf,
        "_build_hh_context",
        lambda _cfg: (
            object(),
            np.eye(dim, dtype=complex),
            ["eeeeee"],
            {"eeeeee": 1.0 + 0.0j},
            np.array(psi_hf, copy=True),
        ),
    )
    monkeypatch.setattr(wf, "_run_warm_start_stage", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("warm stage should be skipped")))
    monkeypatch.setattr(wf.adapt_mod, "_run_hardcoded_adapt_vqe", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ADAPT should be skipped")))
    monkeypatch.setattr(wf.replay_mod, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay should be skipped")))

    cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--fixed-final-state-json",
                str(workflow_json),
                "--tag",
                "fixed_seed_workflow_test",
                "--output-json",
                str(tmp_path / "workflow_out.json"),
                "--output-pdf",
                str(tmp_path / "workflow_out.pdf"),
            ]
        )
    )
    cfg = replace(
        cfg,
        artifacts=replace(
            cfg.artifacts,
            handoff_json=tmp_path / "handoff_out.json",
            replay_output_json=tmp_path / "replay_out.json",
        ),
    )
    stage_result = wf.run_stage_pipeline(cfg)

    assert np.allclose(stage_result.psi_final, psi_seed)
    assert stage_result.fixed_final_state_import is not None
    assert stage_result.fixed_final_state_import["source_json"] == str(workflow_json)
    assert stage_result.fixed_final_state_import["resolved_json"] == str(handoff_json)
    assert (
        stage_result.fixed_final_state_import["resolved_via"]
        == "artifacts.intermediate.adapt_handoff_json"
    )
    assert stage_result.warm_payload["skipped"] is True
    assert stage_result.adapt_payload["skipped"] is True
    assert stage_result.replay_payload["skipped"] is True


def test_run_stage_pipeline_inserts_seed_refine_before_adapt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dim = 1 << int(wf._hh_nq_total(2, 1, "binary"))
    psi_hf = _basis(dim, 0)
    psi_warm = _basis(dim, 1)
    psi_refine = _basis(dim, 2)
    psi_adapt = _basis(dim, 3)
    psi_final = _basis(dim, 4)
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        wf,
        "_build_hh_context",
        lambda _cfg: (
            object(),
            np.eye(dim, dtype=complex),
            ["eeeeee"],
            {"eeeeee": 1.0 + 0.0j},
            np.array(psi_hf, copy=True),
        ),
    )
    monkeypatch.setattr(
        wf,
        "_run_warm_start_stage",
        lambda *_args, **_kwargs: (
            {
                "ansatz": "hh_hva_ptw",
                "energy": -1.0,
                "exact_filtered_energy": -1.05,
                "optimizer_method": "SPSA",
                "message": "warm_ok",
                "checkpoint_json_used": str(tmp_path / "warm_cutover.json"),
                "cutoff_triggered": False,
                "cutoff_reason": None,
            },
            np.array(psi_warm, copy=True),
            tmp_path / "warm_cutover.json",
        ),
    )

    refine_state_json = tmp_path / "seed_refine_state.json"
    refine_state_json.write_text("{}", encoding="utf-8")

    def _fake_seed_refine(_cfg, *, h_poly, psi_ref, exact_filtered_energy):
        calls["seed_refine_h_poly"] = h_poly
        calls["seed_refine_psi_ref"] = np.array(psi_ref, copy=True)
        calls["seed_refine_exact"] = float(exact_filtered_energy)
        return (
            {
                "generator_family": {"requested": "uccsd_otimes_paop_lf_std", "resolved": "uccsd_otimes_paop_lf_std"},
                "pool": {"family_kind": "uccsd_paop_product", "motif_family": "paop_lf_std"},
                "seed_baseline": {"theta_policy": "all_zero"},
                "exact": {"E_exact_sector": -1.05},
                "vqe": {"energy": -1.02, "message": "ok", "stop_reason": "converged", "method": "SPSA"},
                "state_json": str(refine_state_json),
            },
            np.array(psi_refine, copy=True),
            refine_state_json,
        )

    def _fake_run_adapt(**kwargs):
        calls["adapt_kwargs"] = kwargs
        return {
            "success": True,
            "energy": -1.03,
            "exact_gs_energy": -1.05,
            "abs_delta_e": 0.02,
            "ansatz_depth": 1,
            "pool_type": "paop_lf_std",
            "continuation_mode": str(kwargs["adapt_continuation_mode"]),
            "stop_reason": "eps_grad",
            "operators": ["op_1"],
            "optimal_point": [0.1],
            "continuation": {
                "selected_generator_metadata": [{"generator_id": "g1", "family_id": "paop_lf_std"}],
            },
        }, np.array(psi_adapt, copy=True)

    def _fake_replay_run(cfg, diagnostics_out=None):
        calls["replay_cfg"] = cfg
        if diagnostics_out is not None:
            diagnostics_out.clear()
        return {
            "generator_family": {
                "requested": "match_adapt",
                "resolved": "paop_lf_std",
                "resolution_source": "adapt_vqe.pool_type",
            },
            "seed_baseline": {"theta_policy": "auto", "abs_delta_e": 0.005},
            "exact": {"E_exact_sector": -1.05},
            "vqe": {"energy": -1.049, "stop_reason": "converged"},
            "replay_contract": {"continuation_mode": str(cfg.replay_continuation_mode)},
            "best_state": {"amplitudes_qn_to_q0": _amplitudes_qn_to_q0(psi_final)},
        }

    monkeypatch.setattr(wf, "_run_seed_refine_stage", _fake_seed_refine)
    monkeypatch.setattr(wf.adapt_mod, "_run_hardcoded_adapt_vqe", _fake_run_adapt)
    monkeypatch.setattr(wf.replay_mod, "run", _fake_replay_run)

    cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--state-export-dir",
                str(tmp_path),
                "--state-export-prefix",
                "seed_refine_case",
                "--output-json",
                str(tmp_path / "workflow.json"),
                "--output-pdf",
                str(tmp_path / "workflow.pdf"),
                "--seed-refine-family",
                "uccsd_otimes_paop_lf_std",
            ]
        )
    )
    cfg = replace(
        cfg,
        artifacts=replace(
            cfg.artifacts,
            handoff_json=tmp_path / "handoff.json",
            replay_output_json=tmp_path / "replay.json",
            replay_output_csv=tmp_path / "replay.csv",
            replay_output_md=tmp_path / "replay.md",
            replay_output_log=tmp_path / "replay.log",
        ),
    )
    stage_result = wf.run_stage_pipeline(cfg)

    adapt_kwargs = calls["adapt_kwargs"]
    handoff = json.loads(cfg.artifacts.handoff_json.read_text(encoding="utf-8"))

    assert np.allclose(calls["seed_refine_psi_ref"], psi_warm)
    assert calls["seed_refine_exact"] == pytest.approx(-1.05)
    assert str(adapt_kwargs["adapt_ref_json"]) == str(refine_state_json)
    assert np.allclose(stage_result.psi_seed_refine, psi_refine)
    assert stage_result.seed_refine_payload is not None
    assert stage_result.seed_refine_payload["generator_family"]["resolved"] == "uccsd_otimes_paop_lf_std"
    assert handoff["seed_provenance"]["warm_ansatz"] == "hh_hva_ptw"
    assert handoff["seed_provenance"]["refine_family"] == "uccsd_otimes_paop_lf_std"
    assert handoff["seed_provenance"]["refine_family_kind"] == "uccsd_paop_product"
    assert handoff["seed_provenance"]["refine_paop_motif_families"] == ["paop_lf_std"]
    assert "seed_refine_vqe" in handoff["meta"]["stage_chain"]


def test_seed_refine_failure_aborts_before_adapt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dim = 1 << int(wf._hh_nq_total(2, 1, "binary"))
    psi_hf = _basis(dim, 0)
    psi_warm = _basis(dim, 1)
    calls = {"adapt": 0, "replay": 0}

    monkeypatch.setattr(
        wf,
        "_build_hh_context",
        lambda _cfg: (
            object(),
            np.eye(dim, dtype=complex),
            ["eeeeee"],
            {"eeeeee": 1.0 + 0.0j},
            np.array(psi_hf, copy=True),
        ),
    )
    monkeypatch.setattr(
        wf,
        "_run_warm_start_stage",
        lambda *_args, **_kwargs: (
            {
                "ansatz": "hh_hva_ptw",
                "energy": -1.0,
                "exact_filtered_energy": -1.05,
                "optimizer_method": "SPSA",
                "message": "warm_ok",
                "checkpoint_json_used": str(tmp_path / "warm_cutover.json"),
                "cutoff_triggered": False,
                "cutoff_reason": None,
            },
            np.array(psi_warm, copy=True),
            tmp_path / "warm_cutover.json",
        ),
    )
    monkeypatch.setattr(wf, "_run_seed_refine_stage", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("seed refine boom")))
    monkeypatch.setattr(wf.adapt_mod, "_run_hardcoded_adapt_vqe", lambda **_kwargs: calls.__setitem__("adapt", calls["adapt"] + 1))
    monkeypatch.setattr(wf.replay_mod, "run", lambda *_args, **_kwargs: calls.__setitem__("replay", calls["replay"] + 1))

    cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--seed-refine-family",
                "uccsd_otimes_paop_lf_std",
                "--output-json",
                str(tmp_path / "workflow.json"),
                "--output-pdf",
                str(tmp_path / "workflow.pdf"),
            ]
        )
    )

    with pytest.raises(RuntimeError, match="seed refine boom"):
        wf.run_stage_pipeline(cfg)
    assert calls["adapt"] == 0
    assert calls["replay"] == 0


def test_workflow_runs_matched_family_replay_and_static_plus_drive_profiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dim = 1 << int(wf._hh_nq_total(2, 1, "binary"))
    psi_hf = _basis(dim, 0)
    psi_warm = _basis(dim, 1)
    psi_adapt = _basis(dim, 2)
    psi_final = _basis(dim, 3)
    calls: dict[str, object] = {}

    monkeypatch.setattr(wf, "build_hubbard_holstein_hamiltonian", lambda **kwargs: object())
    monkeypatch.setattr(wf, "hubbard_holstein_reference_state", lambda **kwargs: np.array(psi_hf, copy=True))
    monkeypatch.setattr(wf.hc_pipeline, "_collect_hardcoded_terms_exyz", lambda h: (["eeeeee"], {"eeeeee": 1.0 + 0.0j}))
    monkeypatch.setattr(wf.hc_pipeline, "_build_hamiltonian_matrix", lambda coeff: np.eye(dim, dtype=complex))

    def _fake_run_hardcoded_vqe(**kwargs):
        calls["warm_kwargs"] = kwargs
        return {
            "success": True,
            "ansatz": str(kwargs["ansatz_name"]),
            "optimizer_method": str(kwargs["method"]),
            "energy": -1.00,
            "exact_filtered_energy": -1.02,
            "message": "warm_ok",
        }, np.array(psi_warm, copy=True)

    def _fake_run_adapt(**kwargs):
        calls["adapt_kwargs"] = kwargs
        return {
            "success": True,
            "energy": -1.03,
            "exact_gs_energy": -1.04,
            "abs_delta_e": 0.01,
            "ansatz_depth": 2,
            "pool_type": "phase1_v1",
            "continuation_mode": str(kwargs["adapt_continuation_mode"]),
            "stop_reason": "eps_grad",
            "operators": ["op_1", "op_2"],
                "optimal_point": [0.1, 0.2],
                "continuation": {
                    "optimizer_memory": {"cached": True},
                    "selected_generator_metadata": [{"generator_id": "g1", "family_id": "paop_lf_std"}],
                },
            }, np.array(psi_adapt, copy=True)

    def _fake_replay_run(cfg):
        calls["replay_cfg"] = cfg
        return {
            "generator_family": {
                "requested": "match_adapt",
                "resolved": "paop_lf_std",
                "resolution_source": "adapt_vqe.pool_type",
            },
            "seed_baseline": {"theta_policy": "auto", "abs_delta_e": 0.005},
            "exact": {"E_exact_sector": -1.05},
            "vqe": {"energy": -1.049, "stop_reason": "converged"},
            "replay_contract": {"continuation_mode": str(cfg.replay_continuation_mode)},
            "best_state": {"amplitudes_qn_to_q0": _amplitudes_qn_to_q0(psi_final)},
        }

    def _fake_simulate_trajectory(**kwargs):
        calls.setdefault("propagators", []).append(str(kwargs["propagator"]))
        rows = [
            {
                "time": 0.0,
                "fidelity": 1.0,
                "energy_total_trotter": -1.049,
                "energy_total_exact": -1.049,
                "doublon_trotter": 0.1,
                "doublon_exact": 0.1,
            },
            {
                "time": 1.0,
                "fidelity": 0.99,
                "energy_total_trotter": -1.045,
                "energy_total_exact": -1.049,
                "doublon_trotter": 0.11,
                "doublon_exact": 0.10,
            },
        ]
        return rows, []

    class _FakeDriveTemplate:
        def labels_exyz(self, include_identity: bool = False):
            return ["zeeeee"]

    class _FakeDrive:
        def __init__(self):
            self.include_identity = False
            self.template = _FakeDriveTemplate()
            self.coeff_map_exyz = lambda _t: {"zeeeee": 0.1 + 0.0j}

    monkeypatch.setattr(wf.hc_pipeline, "_run_hardcoded_vqe", _fake_run_hardcoded_vqe)
    monkeypatch.setattr(wf.adapt_mod, "_run_hardcoded_adapt_vqe", _fake_run_adapt)
    monkeypatch.setattr(wf.replay_mod, "run", _fake_replay_run)
    monkeypatch.setattr(wf.hc_pipeline, "_simulate_trajectory", _fake_simulate_trajectory)
    monkeypatch.setattr(wf, "build_gaussian_sinusoid_density_drive", lambda **kwargs: _FakeDrive())
    monkeypatch.setattr(wf, "exact_ground_energy_sector_hh", lambda *args, **kwargs: -1.02)

    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--run-replay",
            "--run-dynamics",
            "--enable-drive",
            "--output-json",
            str(tmp_path / "hh_staged.json"),
            "--output-pdf",
            str(tmp_path / "hh_staged.pdf"),
            "--adapt-beam-live-branches",
            "3",
            "--adapt-beam-children-per-parent",
            "2",
            "--adapt-beam-terminated-keep",
            "4",
        ]
    )
    cfg = resolve_staged_hh_config(args)
    payload = wf.run_staged_hh_noiseless(cfg, run_command="python pipelines/hardcoded/hh_staged_noiseless.py --L 2")

    warm_kwargs = calls["warm_kwargs"]
    adapt_kwargs = calls["adapt_kwargs"]
    replay_cfg = calls["replay_cfg"]
    adapt_handoff = json.loads(cfg.artifacts.handoff_json.read_text(encoding="utf-8"))

    assert warm_kwargs["ansatz_name"] == "hh_hva_ptw"
    assert str(adapt_kwargs["adapt_ref_json"]) == str(cfg.artifacts.warm_cutover_json)
    assert Path(adapt_kwargs["adapt_ref_json"]).exists()
    assert int(adapt_kwargs["adapt_beam_live_branches"]) == 3
    assert int(adapt_kwargs["adapt_beam_children_per_parent"]) == 2
    assert int(adapt_kwargs["adapt_beam_terminated_keep"]) == 4
    assert adapt_handoff["adapt_vqe"]["operators"] == ["op_1", "op_2"]
    assert adapt_handoff["adapt_vqe"]["optimal_point"] == [0.1, 0.2]
    assert isinstance(adapt_handoff["adapt_vqe"]["operators"], list)
    assert isinstance(adapt_handoff["adapt_vqe"]["optimal_point"], list)
    assert adapt_handoff["initial_state"]["handoff_state_kind"] == "prepared_state"
    assert adapt_handoff["settings"]["adapt_pool"] == "paop_lf_std"
    assert adapt_handoff["continuation"]["replay_contract"]["generator_family"]["requested"] == "match_adapt"
    assert adapt_handoff["continuation"]["replay_contract"]["generator_family"]["resolved"] == "paop_lf_std"
    assert adapt_handoff["continuation"]["replay_contract"]["seed_policy_requested"] == "auto"
    assert adapt_handoff["continuation"]["replay_contract"]["seed_policy_resolved"] == "residual_only"
    assert replay_cfg.generator_family == "match_adapt"
    assert replay_cfg.replay_continuation_mode == "phase3_v1"
    assert payload["stage_pipeline"]["conventional_replay"]["generator_family"]["requested"] == "match_adapt"
    assert payload["stage_pipeline"]["warm_start"]["checkpoint_json_used"] == str(cfg.artifacts.warm_cutover_json)
    assert payload["stage_pipeline"]["adapt_vqe"]["adapt_ref_json"] == str(cfg.artifacts.warm_cutover_json)
    assert "seed_provenance" not in adapt_handoff
    assert "seed_refine_vqe" not in adapt_handoff["meta"]["stage_chain"]
    assert set(payload["dynamics_noiseless"]["profiles"].keys()) == {"static", "drive"}
    static_profile = payload["dynamics_noiseless"]["profiles"]["static"]
    static_rows = static_profile["methods"]["suzuki2"]["trajectory"]
    assert static_profile["ground_state_reference"]["energy"] == pytest.approx(-1.05)
    assert abs(static_rows[0]["energy_total_trotter"] - static_rows[0]["energy_total_exact"]) == pytest.approx(0.0)
    assert static_rows[0]["abs_energy_error_vs_ground_state"] == pytest.approx(1e-3)
    assert payload["comparisons"]["noiseless_vs_ground_state"]["static"]["suzuki2"]["final_abs_energy_error"] == pytest.approx(5e-3)
    assert payload["comparisons"]["noiseless_vs_reference"]["static"]["suzuki2"]["final_fidelity"] == pytest.approx(0.99)
    assert "noiseless_vs_exact" not in payload["comparisons"]
    assert payload["workflow_contract"]["noiseless_energy_metric"].startswith("|E_method(t) - E_exact_sector_terminal|")
    assert calls["propagators"] == ["suzuki2", "cfqm4", "suzuki2", "cfqm4"]
    assert Path(cfg.artifacts.output_json).exists()


def test_workflow_skips_replay_and_dynamics_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dim = 1 << int(wf._hh_nq_total(2, 1, "binary"))
    psi_hf = _basis(dim, 0)
    psi_warm = _basis(dim, 1)
    psi_adapt = _basis(dim, 2)

    monkeypatch.setattr(wf, "build_hubbard_holstein_hamiltonian", lambda **kwargs: object())
    monkeypatch.setattr(wf, "hubbard_holstein_reference_state", lambda **kwargs: np.array(psi_hf, copy=True))
    monkeypatch.setattr(wf.hc_pipeline, "_collect_hardcoded_terms_exyz", lambda h: (["eeeeee"], {"eeeeee": 1.0 + 0.0j}))
    monkeypatch.setattr(wf.hc_pipeline, "_build_hamiltonian_matrix", lambda coeff: np.eye(dim, dtype=complex))
    monkeypatch.setattr(
        wf.hc_pipeline,
        "_run_hardcoded_vqe",
        lambda **kwargs: (
            {
                "success": True,
                "ansatz": str(kwargs["ansatz_name"]),
                "optimizer_method": str(kwargs["method"]),
                "energy": -1.00,
                "exact_filtered_energy": -1.02,
                "message": "warm_ok",
            },
            np.array(psi_warm, copy=True),
        ),
    )
    monkeypatch.setattr(
        wf.adapt_mod,
        "_run_hardcoded_adapt_vqe",
        lambda **kwargs: (
            {
                "success": True,
                "energy": -1.03,
                "exact_gs_energy": -1.04,
                "abs_delta_e": 0.01,
                "ansatz_depth": 2,
                "pool_type": "paop_lf_std",
                "continuation_mode": str(kwargs["adapt_continuation_mode"]),
                "stop_reason": "eps_grad",
                "operators": ["op_1", "op_2"],
                "optimal_point": [0.1, 0.2],
                "continuation": {
                    "selected_generator_metadata": [{"generator_id": "g1", "family_id": "paop_lf_std"}],
                },
            },
            np.array(psi_adapt, copy=True),
        ),
    )
    monkeypatch.setattr(wf.replay_mod, "run", lambda *args, **kwargs: pytest.fail("replay should be skipped"))
    monkeypatch.setattr(
        wf.hc_pipeline,
        "_simulate_trajectory",
        lambda **kwargs: pytest.fail("dynamics should be skipped"),
    )
    monkeypatch.setattr(wf, "exact_ground_energy_sector_hh", lambda *args, **kwargs: -1.02)

    cfg = resolve_staged_hh_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--output-json",
                str(tmp_path / "hh_staged.json"),
                "--output-pdf",
                str(tmp_path / "hh_staged.pdf"),
            ]
        )
    )
    payload = wf.run_staged_hh_noiseless(cfg, run_command="python pipelines/hardcoded/hh_staged_noiseless.py --L 2")

    assert payload["workflow_contract"]["stage_chain"] == ["hf_reference", "warm_start_hva", "adapt_vqe"]
    assert payload["workflow_contract"]["conventional_vqe_definition"] == "disabled (run_replay=false)"
    assert payload["workflow_contract"]["noiseless_energy_metric"] == "not_run (run_dynamics=false)"
    assert payload["stage_pipeline"]["conventional_replay"]["skipped"] is True
    assert payload["stage_pipeline"]["conventional_replay"]["skip_reason"] == "run_replay_false"
    assert payload["stage_pipeline"]["conventional_replay"]["ecut_2"]["pass"] is None
    assert payload["dynamics_noiseless"]["skipped"] is True
    assert payload["dynamics_noiseless"]["skip_reason"] == "run_dynamics_false"
    assert Path(cfg.artifacts.handoff_json).exists()
    assert not Path(cfg.artifacts.replay_output_json).exists()


def test_infer_handoff_adapt_pool_prefers_selected_and_rejects_mixed() -> None:
    cfg = resolve_staged_hh_config(parse_args(["--L", "2", "--skip-pdf"]))

    family, source = wf._infer_handoff_adapt_pool(
        cfg,
        {
            "continuation": {
                "selected_generator_metadata": [
                    {"family_id": "paop_lf_std"},
                    {"family_id": "full_meta"},
                ],
                "motif_library": {
                    "records": [{"family_id": "paop_lf_std"}]
                },
            },
            "pool_type": "paop_lf_std",
        },
    )
    assert family is None
    assert source == "continuation.selected_generator_metadata.family_id(mixed)"


def test_infer_handoff_adapt_pool_rejects_mixed_operator_labels_without_metadata() -> None:
    cfg = resolve_staged_hh_config(parse_args(["--L", "2", "--skip-pdf"]))

    family, source = wf._infer_handoff_adapt_pool(
        cfg,
        {
            "operators": [
                "hh_termwise_ham_quadrature_term(yezeee)",
                "paop_lf_std:paop_disp(site=1)",
            ],
            "pool_type": "paop_lf_std",
        },
    )

    assert family is None
    assert source == "adapt_payload.operators(mixed)"


def test_infer_handoff_adapt_pool_rejects_mixed_split_termwise_operator_labels() -> None:
    cfg = resolve_staged_hh_config(parse_args(["--L", "2", "--skip-pdf"]))

    family, source = wf._infer_handoff_adapt_pool(
        cfg,
        {
            "operators": [
                "hh_termwise_ham_quadrature_term(yezeee)::split[0]::yezeee",
                "paop_lf_std:paop_disp(site=1)",
            ],
            "pool_type": "paop_lf_std",
        },
    )

    assert family is None
    assert source == "adapt_payload.operators(mixed)"


def test_infer_handoff_adapt_pool_prefers_operator_labels_over_motif_family() -> None:
    cfg = resolve_staged_hh_config(parse_args(["--L", "2", "--skip-pdf"]))

    family, source = wf._infer_handoff_adapt_pool(
        cfg,
        {
            "operators": [
                "hh_termwise_ham_quadrature_term(yezeee)",
                "paop_lf_std:paop_disp(site=1)",
            ],
            "continuation": {
                "motif_library": {
                    "records": [{"family_id": "paop_lf_std"}, {"family_id": "paop_lf_std"}],
                }
            },
            "pool_type": "paop_lf_std",
        },
    )

    assert family is None
    assert source == "adapt_payload.operators(mixed)"


def test_infer_handoff_adapt_pool_uses_motif_family_when_selected_missing() -> None:
    cfg = resolve_staged_hh_config(parse_args(["--L", "2", "--skip-pdf"]))

    family, source = wf._infer_handoff_adapt_pool(
        cfg,
        {
            "continuation": {
                "motif_library": {
                    "records": [{"family_id": "paop_lf_std"}, {"family_id": "paop_lf_std"}],
                }
            }
        },
    )
    assert family == "paop_lf_std"
    assert source == "continuation.motif_library.records"
