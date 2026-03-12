from __future__ import annotations

import json
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
    assert str(cfg.replay.continuation_mode) == "phase3_v1"
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


def test_warm_ansatz_override_is_resolved_and_retagged() -> None:
    cfg_default = resolve_staged_hh_config(parse_args(["--L", "2", "--skip-pdf"]))
    cfg_layerwise = resolve_staged_hh_config(parse_args(["--L", "2", "--warm-ansatz", "hh_hva", "--skip-pdf"]))

    assert str(cfg_layerwise.warm_start.ansatz_name) == "hh_hva"
    assert str(cfg_default.warm_start.ansatz_name) == "hh_hva_ptw"
    assert str(cfg_layerwise.artifacts.tag) != str(cfg_default.artifacts.tag)
    assert "warmhh_hva" in str(cfg_layerwise.artifacts.tag)


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


def test_nondefault_sector_override_rejected_cleanly() -> None:
    args = parse_args(["--L", "2", "--sector-n-up", "2", "--skip-pdf"])
    with pytest.raises(ValueError, match="half-filled sector"):
        resolve_staged_hh_config(args)



def test_staged_hh_parse_rejects_non_spsa_methods() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--L", "2", "--warm-method", "COBYLA", "--skip-pdf"])
    with pytest.raises(SystemExit):
        parse_args(["--L", "2", "--final-method", "COBYLA", "--skip-pdf"])


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
            "--enable-drive",
            "--output-json",
            str(tmp_path / "hh_staged.json"),
            "--output-pdf",
            str(tmp_path / "hh_staged.pdf"),
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
    assert set(payload["dynamics_noiseless"]["profiles"].keys()) == {"static", "drive"}
    static_profile = payload["dynamics_noiseless"]["profiles"]["static"]
    static_rows = static_profile["methods"]["suzuki2"]["trajectory"]
    assert static_profile["ground_state_reference"]["energy"] == pytest.approx(-1.05)
    assert abs(static_rows[0]["energy_total_trotter"] - static_rows[0]["energy_total_exact"]) == pytest.approx(0.0)
    assert static_rows[0]["abs_energy_error_vs_ground_state"] == pytest.approx(1e-3)
    assert payload["comparisons"]["noiseless_vs_ground_state"]["static"]["suzuki2"]["final_abs_energy_error"] == pytest.approx(5e-3)
    assert payload["comparisons"]["noiseless_vs_reference"]["static"]["suzuki2"]["final_fidelity"] == pytest.approx(0.99)
    assert "noiseless_vs_exact" not in payload["comparisons"]
    assert payload["workflow_contract"]["noiseless_energy_metric"].startswith("|E_method(t) - E_exact_sector_replay|")
    assert calls["propagators"] == ["suzuki2", "cfqm4", "suzuki2", "cfqm4"]
    assert Path(cfg.artifacts.output_json).exists()


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
