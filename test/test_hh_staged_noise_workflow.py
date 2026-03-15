from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.hardcoded.hh_staged_noise as noise_cli
import pipelines.hardcoded.hh_staged_noise_workflow as noise_wf
import pipelines.hardcoded.hh_staged_workflow as base_wf
from pipelines.hardcoded.hh_staged_noise import parse_args


def _basis(dim: int, idx: int) -> np.ndarray:
    out = np.zeros(dim, dtype=complex)
    out[int(idx)] = 1.0
    return out


def _stage_result(dim: int = 8) -> base_wf.StageExecutionResult:
    psi0 = _basis(dim, 0)
    psi1 = _basis(dim, 1)
    psi2 = _basis(dim, 2)
    psi3 = _basis(dim, 3)
    return base_wf.StageExecutionResult(
        h_poly=object(),
        hmat=np.eye(dim, dtype=complex),
        ordered_labels_exyz=["eee"],
        coeff_map_exyz={"eee": 1.0 + 0.0j},
        nq_total=int(round(np.log2(dim))),
        psi_hf=np.array(psi0, copy=True),
        psi_warm=np.array(psi1, copy=True),
        psi_adapt=np.array(psi2, copy=True),
        psi_final=np.array(psi3, copy=True),
        warm_payload={"energy": -1.0, "exact_filtered_energy": -1.1},
        adapt_payload={"energy": -1.05, "exact_gs_energy": -1.1, "stop_reason": "eps_grad"},
        replay_payload={"vqe": {"energy": -1.09}, "exact": {"E_exact_sector": -1.1}},
    )


def test_resolve_noise_defaults_and_retagged_artifacts() -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(parse_args(["--L", "2", "--skip-pdf"]))

    assert cfg.noise.methods == ("cfqm4", "suzuki2")
    assert cfg.noise.modes == ("ideal", "shots", "aer_noise")
    assert int(cfg.noise.shots) == 2048
    assert int(cfg.noise.oracle_repeats) == 4
    assert str(cfg.noise.oracle_aggregate) == "mean"
    assert cfg.noise.backend_profile is None
    assert str(cfg.noise.aer_noise_kind) == "scheduled"
    assert cfg.noise.schedule_policy is None
    assert cfg.noise.layout_policy is None
    assert cfg.noise.noise_snapshot_json is None
    assert cfg.noise.fixed_physical_patch is None
    assert bool(cfg.noise.allow_noisy_fallback) is False
    assert cfg.noise.mitigation_config == {"mode": "none", "zne_scales": [], "dd_sequence": None}
    assert cfg.noise.runtime_twirling_config == {
        "enable_gates": False,
        "enable_measure": False,
        "num_randomizations": None,
        "strategy": None,
    }
    assert cfg.noise.symmetry_mitigation_config == {
        "mode": "off",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }
    assert bool(cfg.noise.include_final_audit) is False
    assert bool(cfg.noise.paired_anchor_comparison) is False
    assert str(cfg.staged.artifacts.tag).startswith("hh_staged_noise_")
    assert Path(cfg.staged.artifacts.output_json).name == f"{cfg.staged.artifacts.tag}.json"
    assert Path(cfg.staged.artifacts.replay_output_json).name == f"{cfg.staged.artifacts.tag}_replay.json"


def test_explicit_noise_tag_is_preserved() -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(["--L", "2", "--skip-pdf", "--tag", "custom_noise_tag"])
    )

    assert str(cfg.staged.artifacts.tag) == "custom_noise_tag"
    assert Path(cfg.staged.artifacts.output_json).name == "custom_noise_tag.json"


def test_resolve_noise_config_preserves_runtime_layer_noise_model_json() -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--noise-modes",
                "qpu_layer_learned",
                "--mitigation",
                "zne",
                "--zne-scales",
                "1.0,2.0",
                "--runtime-layer-noise-model-json",
                "artifacts/json/layer_noise_model.json",
            ]
        )
    )

    assert cfg.noise.mitigation_config == {
        "mode": "zne",
        "zne_scales": [1.0, 2.0],
        "dd_sequence": None,
        "layer_noise_model_json": "artifacts/json/layer_noise_model.json",
    }


def test_local_aer_modes_share_layout_lock_key_for_patch_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mode_calls: list[dict[str, object]] = []

    def _fake_run_noisy_mode_isolated(*, kwargs: dict[str, object], timeout_s: int) -> dict[str, object]:
        mode_calls.append({"kwargs": kwargs, "timeout_s": timeout_s})
        return {
            "success": True,
            "trajectory": [],
            "delta_uncertainty": {},
            "benchmark_cost": {},
            "benchmark_runtime": {},
        }

    monkeypatch.setattr(noise_wf.noise_report, "_run_noisy_mode_isolated", _fake_run_noisy_mode_isolated)
    monkeypatch.setattr(noise_wf.noise_report, "_run_noisy_audit_mode_isolated", lambda **_kwargs: {"success": True})
    monkeypatch.setattr(noise_wf.noise_report, "_collect_noisy_benchmark_rows", lambda dynamics_noisy: [])

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--noise-modes",
                "backend_scheduled,patch_snapshot",
                "--noisy-methods",
                "cfqm4",
                "--backend-profile",
                "frozen_snapshot_json",
                "--noise-snapshot-json",
                "artifacts/json/frozen_snapshot.json",
                "--schedule-policy",
                "asap",
            ]
        )
    )
    stage_result = _stage_result()
    noise_wf.run_noisy_profiles(stage_result, cfg)

    by_mode = {str(rec["kwargs"]["noise_mode"]): str(rec["kwargs"]["layout_lock_key"]) for rec in mode_calls}
    assert by_mode["backend_scheduled"] == by_mode["patch_snapshot"]



def test_run_noisy_profiles_uses_final_state_and_optional_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mode_calls: list[dict[str, object]] = []
    audit_calls: list[dict[str, object]] = []

    def _fake_run_noisy_mode_isolated(*, kwargs: dict[str, object], timeout_s: int) -> dict[str, object]:
        mode_calls.append({"kwargs": kwargs, "timeout_s": timeout_s})
        return {
            "success": True,
            "trajectory": [
                {
                    "time": 0.0,
                    "energy_total_noisy": -1.0,
                    "doublon_noisy": 0.1,
                    "energy_total_delta_noisy_minus_ideal": 0.0,
                    "doublon_delta_noisy_minus_ideal": 0.0,
                }
            ],
            "delta_uncertainty": {
                "energy_total": {
                    "max_abs_delta": 0.0,
                    "max_abs_delta_over_stderr": 0.0,
                    "mean_abs_delta_over_stderr": 0.0,
                }
            },
            "benchmark_cost": {
                "term_exp_count_total": 1,
                "pauli_rot_count_total": 1,
                "cx_proxy_total": 0,
                "sq_proxy_total": 0,
                "depth_proxy_total": 1,
            },
            "benchmark_runtime": {
                "wall_total_s": 0.1,
                "oracle_eval_s_total": 0.05,
                "oracle_calls_total": 1,
            },
        }

    def _fake_run_noisy_audit_mode_isolated(*, kwargs: dict[str, object], timeout_s: int) -> dict[str, object]:
        audit_calls.append({"kwargs": kwargs, "timeout_s": timeout_s})
        return {
            "success": True,
            "final_observables": {
                "energy_total": {"noisy_mean": -1.0, "delta_mean": 0.0, "delta_stderr": 0.0},
                "doublon": {"noisy_mean": 0.1, "delta_mean": 0.0, "delta_stderr": 0.0},
            },
        }

    monkeypatch.setattr(noise_wf.noise_report, "_run_noisy_mode_isolated", _fake_run_noisy_mode_isolated)
    monkeypatch.setattr(noise_wf.noise_report, "_run_noisy_audit_mode_isolated", _fake_run_noisy_audit_mode_isolated)
    monkeypatch.setattr(
        noise_wf.noise_report,
        "_collect_noisy_benchmark_rows",
        lambda dynamics_noisy: [{"profile": "static", "method": "cfqm4", "mode": "ideal"}],
    )

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--enable-drive",
                "--include-final-audit",
                "--noise-modes",
                "ideal,shots",
                "--noisy-methods",
                "cfqm4",
                "--backend-profile",
                "generic_seeded",
                "--aer-noise-kind",
                "scheduled",
                "--allow-noisy-fallback",
                "--cfqm-stage-exp",
                "pauli_suzuki2",
            ]
        )
    )
    stage_result = _stage_result()
    dynamics_noisy, noisy_final_audit, dynamics_benchmarks = noise_wf.run_noisy_profiles(stage_result, cfg)

    assert set(dynamics_noisy["profiles"].keys()) == {"static", "drive"}
    assert set(noisy_final_audit["profiles"].keys()) == {"static", "drive"}
    assert dynamics_benchmarks["rows"] == [{"profile": "static", "method": "cfqm4", "mode": "ideal"}]
    assert len(mode_calls) == 4
    assert len(audit_calls) == 4
    assert all(np.allclose(rec["kwargs"]["psi_seed"], stage_result.psi_final) for rec in mode_calls)
    assert all(np.allclose(rec["kwargs"]["psi_seed"], stage_result.psi_final) for rec in audit_calls)
    assert {str(rec["kwargs"]["method"]) for rec in mode_calls} == {"cfqm4"}
    assert {str(rec["kwargs"]["noise_mode"]) for rec in mode_calls} == {"ideal", "shots"}
    assert {str(rec["kwargs"]["cfqm_stage_exp"]) for rec in mode_calls} == {"pauli_suzuki2"}


def test_resolve_noise_runtime_twirling_flags_propagate() -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--runtime-enable-measure-twirling",
                "--runtime-twirling-num-randomizations",
                "8",
                "--runtime-twirling-strategy",
                "active",
            ]
        )
    )

    assert cfg.noise.runtime_twirling_config == {
        "enable_gates": False,
        "enable_measure": True,
        "num_randomizations": 8,
        "strategy": "active",
    }


def test_run_noisy_profiles_emits_paired_anchor_comparison_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        noise_wf.noise_report,
        "_run_noisy_mode_isolated",
        lambda **_kwargs: {
            "success": True,
            "trajectory": [],
            "delta_uncertainty": {},
            "benchmark_cost": {},
            "benchmark_runtime": {},
        },
    )
    monkeypatch.setattr(noise_wf.noise_report, "_run_noisy_audit_mode_isolated", lambda **_kwargs: {"success": True})
    monkeypatch.setattr(noise_wf.noise_report, "_collect_noisy_benchmark_rows", lambda dynamics_noisy: [])

    def _fake_paired_helper(**kwargs: object) -> dict[str, object]:
        helper_calls.append(dict(kwargs))
        return {
            "enabled": True,
            "executed": True,
            "profile": str(kwargs["profile_name"]),
            "method": str(kwargs["method"]),
            "layout_lock_key": str(kwargs["layout_lock_key"]),
            "lock_family_token": "local_aer_locked_patch",
            "local_mode": "backend_scheduled",
            "rows": [],
            "comparability": {
                "local_vs_runtime_same_anchor_evidence_status": "fully_evidenced",
                "runtime_pair_same_submitted_evidence_status": "fully_evidenced",
                "exact_anchor_identity_evidence_status": "fully_evidenced",
                "runtime_pair_differs_only_in_execution_bundle": True,
            },
        }

    monkeypatch.setattr(
        noise_wf.noise_report,
        "_run_paired_anchor_method_trajectory_set",
        _fake_paired_helper,
    )

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--paired-anchor-comparison",
                "--noise-modes",
                "ideal",
                "--noisy-methods",
                "cfqm4",
                "--runtime-enable-gate-twirling",
                "--runtime-twirling-num-randomizations",
                "16",
                "--runtime-twirling-strategy",
                "active",
                "--cfqm-stage-exp",
                "pauli_suzuki2",
            ]
        )
    )
    stage_result = _stage_result()
    _, _, dynamics_benchmarks = noise_wf.run_noisy_profiles(stage_result, cfg)

    paired = dynamics_benchmarks["paired_anchor_comparisons"]
    assert paired["profiles"]["static"]["methods"]["cfqm4"]["local_mode"] == "backend_scheduled"
    assert len(helper_calls) == 1
    assert helper_calls[0]["profile_name"] == "static"
    assert helper_calls[0]["method"] == "cfqm4"
    assert helper_calls[0]["runtime_twirling_config"] == {
        "enable_gates": True,
        "enable_measure": False,
        "num_randomizations": 16,
        "strategy": "active",
    }
    assert helper_calls[0]["cfqm_stage_exp"] == "pauli_suzuki2"
    assert helper_calls[0]["layout_lock_key"] == (
        f"staged_noise_paired:{cfg.staged.artifacts.tag}:static:cfqm4:local_aer_locked_patch"
    )


def test_run_staged_hh_noise_merges_base_payload_and_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--noise-modes",
                "ideal",
                "--noisy-methods",
                "cfqm4",
                "--output-json",
                str(tmp_path / "hh_staged_noise.json"),
                "--output-pdf",
                str(tmp_path / "hh_staged_noise.pdf"),
            ]
        )
    )
    stage_result = _stage_result()
    calls: dict[str, object] = {}

    def _fake_run_stage_pipeline(staged_cfg: base_wf.StagedHHConfig) -> base_wf.StageExecutionResult:
        calls["stage_cfg"] = staged_cfg
        return stage_result

    monkeypatch.setattr(noise_wf.base_wf, "run_stage_pipeline", _fake_run_stage_pipeline)
    monkeypatch.setattr(
        noise_wf.base_wf,
        "run_noiseless_profiles",
        lambda stage_result_arg, staged_cfg: {
            "profiles": {
                "static": {
                    "methods": {
                        "cfqm4": {
                            "trajectory": [
                                {
                                    "time": 0.0,
                                    "energy_total_trotter": -1.0,
                                    "doublon_trotter": 0.1,
                                    "fidelity": 1.0,
                                }
                            ]
                        }
                    }
                }
            }
        },
    )
    monkeypatch.setattr(
        noise_wf,
        "run_noisy_profiles",
        lambda stage_result_arg, cfg_arg: (
            {
                "profiles": {
                    "static": {
                        "methods": {
                            "cfqm4": {
                                "modes": {
                                    "ideal": {
                                        "success": True,
                                        "trajectory": [
                                            {
                                                "time": 0.0,
                                                "energy_total_noisy": -1.0,
                                                "doublon_noisy": 0.1,
                                                "energy_total_delta_noisy_minus_ideal": 0.0,
                                                "doublon_delta_noisy_minus_ideal": 0.0,
                                            }
                                        ],
                                    }
                                }
                            }
                        }
                    }
                }
            },
            {"profiles": {}},
            {"rows": [{"profile": "static", "method": "cfqm4", "mode": "ideal"}]},
        ),
    )
    monkeypatch.setattr(
        noise_wf.base_wf,
        "assemble_payload",
        lambda **kwargs: {
            "pipeline": "hh_staged_noiseless",
            "workflow_contract": {},
            "settings": {},
            "artifacts": {
                "workflow": {
                    "output_json": str(cfg.staged.artifacts.output_json),
                    "output_pdf": str(cfg.staged.artifacts.output_pdf),
                },
                "intermediate": {
                    "adapt_handoff_json": str(cfg.staged.artifacts.handoff_json),
                    "replay_output_json": str(cfg.staged.artifacts.replay_output_json),
                },
            },
            "stage_pipeline": {
                "warm_start": {"delta_abs": 1e-1},
                "adapt_vqe": {"delta_abs": 1e-2},
                "conventional_replay": {"delta_abs": 1e-3},
            },
            "dynamics_noiseless": kwargs["dynamics_noiseless"],
        },
    )
    monkeypatch.setattr(noise_wf.base_wf, "_compute_comparisons", lambda payload: {"base_compare": True})
    monkeypatch.setattr(noise_wf.noise_report, "_compute_comparisons", lambda payload: {"noise_compare": True})

    writes: dict[str, object] = {}

    def _fake_write_json(path: Path, payload: dict[str, object]) -> None:
        writes["json_path"] = Path(path)
        writes["payload"] = payload

    monkeypatch.setattr(noise_wf.base_wf, "_write_json", _fake_write_json)
    monkeypatch.setattr(
        noise_wf,
        "write_staged_hh_noise_pdf",
        lambda payload, cfg_arg, run_command: writes.setdefault("pdf_called", True),
    )

    payload = noise_wf.run_staged_hh_noise(cfg, run_command="python pipelines/hardcoded/hh_staged_noise.py --L 2")

    assert calls["stage_cfg"] is cfg.staged
    assert payload["pipeline"] == "hh_staged_noise"
    assert payload["workflow_contract"]["noise_extension"] == "final_only_noisy_dynamics"
    assert payload["settings"]["noise"]["methods"] == ["cfqm4"]
    assert payload["dynamics_benchmarks"]["rows"] == [{"profile": "static", "method": "cfqm4", "mode": "ideal"}]
    assert payload["comparisons"] == {"base_compare": True, "noise_compare": True}
    assert writes["json_path"] == cfg.staged.artifacts.output_json
    assert bool(writes["pdf_called"]) is True


def test_run_staged_hh_noise_reuses_fixed_state_from_workflow_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nq_total = int(base_wf._hh_nq_total(2, 1, "binary"))
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
                    "amplitudes_qn_to_q0": {
                        format(5, f"0{nq_total}b"): {"re": 1.0, "im": 0.0},
                    },
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
    workflow_json = tmp_path / "staged_workflow.json"
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
        noise_wf.base_wf,
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
        noise_wf.base_wf,
        "_run_warm_start_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("warm stage should be skipped")),
    )
    monkeypatch.setattr(
        noise_wf.base_wf.adapt_mod,
        "_run_hardcoded_adapt_vqe",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ADAPT should be skipped")),
    )
    monkeypatch.setattr(
        noise_wf.base_wf.replay_mod,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay should be skipped")),
    )
    monkeypatch.setattr(noise_wf.base_wf, "run_noiseless_profiles", lambda stage_result, staged_cfg: {"profiles": {}})
    monkeypatch.setattr(noise_wf.base_wf, "build_stage_circuit_report_artifacts", lambda stage_result, staged_cfg: None)
    monkeypatch.setattr(noise_wf.base_wf, "_write_json", lambda path, payload: None)
    monkeypatch.setattr(noise_wf, "write_staged_hh_noise_pdf", lambda payload, cfg_arg, run_command: None)

    captured: dict[str, object] = {}

    def _fake_run_noisy_profiles(
        stage_result: base_wf.StageExecutionResult,
        cfg_arg: noise_wf.StagedHHNoiseConfig,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        captured["psi_final"] = np.array(stage_result.psi_final, copy=True)
        captured["fixed_import"] = dict(stage_result.fixed_final_state_import or {})
        return {"profiles": {}}, {"profiles": {}}, {"rows": []}

    monkeypatch.setattr(noise_wf, "run_noisy_profiles", _fake_run_noisy_profiles)

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--fixed-final-state-json",
                str(workflow_json),
                "--noise-modes",
                "ideal",
                "--noisy-methods",
                "cfqm4",
                "--output-json",
                str(tmp_path / "hh_staged_noise.json"),
                "--output-pdf",
                str(tmp_path / "hh_staged_noise.pdf"),
            ]
        )
    )

    payload = noise_wf.run_staged_hh_noise(
        cfg,
        run_command="python pipelines/hardcoded/hh_staged_noise.py --fixed-final-state-json staged_workflow.json",
    )

    assert payload["pipeline"] == "hh_staged_noise"
    assert np.allclose(captured["psi_final"], psi_seed)
    fixed_import = payload["stage_pipeline"]["fixed_final_state_import"]
    assert fixed_import["source_json"] == str(workflow_json)
    assert fixed_import["resolved_json"] == str(handoff_json)
    assert fixed_import["resolved_via"] == "artifacts.intermediate.adapt_handoff_json"


def test_noise_cli_main_print_contract(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake_cfg = SimpleNamespace(staged=SimpleNamespace(artifacts=SimpleNamespace(skip_pdf=True)))
    fake_payload = {
        "artifacts": {
            "workflow": {"output_json": "artifacts/json/hh_staged_noise.json", "output_pdf": "artifacts/pdf/hh_staged_noise.pdf"},
            "intermediate": {
                "adapt_handoff_json": "artifacts/json/hh_staged_noise_adapt_handoff.json",
                "replay_output_json": "artifacts/json/hh_staged_noise_replay.json",
            },
        }
    }

    monkeypatch.setattr(noise_cli, "resolve_staged_hh_noise_config", lambda args: fake_cfg)
    monkeypatch.setattr(noise_cli, "run_staged_hh_noise", lambda cfg: fake_payload)

    noise_cli.main(["--skip-pdf"])
    lines = capsys.readouterr().out.strip().splitlines()

    assert lines == [
        "workflow_json=artifacts/json/hh_staged_noise.json",
        "adapt_handoff_json=artifacts/json/hh_staged_noise_adapt_handoff.json",
    ]
