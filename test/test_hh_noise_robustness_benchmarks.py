from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipelines.exact_bench.hh_noise_robustness_seq_report import (
    _build_mitigation_config,
    _build_summary,
    _build_symmetry_mitigation_config,
    _collect_noisy_benchmark_rows,
    _compute_time_dynamics_proxy_cost,
    _disabled_hardcoded_superset_meta,
    _enforce_defaults_and_minimums,
    _layout_lock_mode_token,
    _noise_config_caption,
    _noise_style_legend_lines,
    _normalize_display_string_list,
    _paired_anchor_summary_rows,
    _parse_noisy_methods_csv,
    _run_noisy_method_trajectory,
    _run_paired_anchor_method_trajectory_set,
    _validate_pool_b_strict_composition,
    parse_args,
)
from pipelines.exact_bench.noise_snapshot import freeze_backend_snapshot, write_calibration_snapshot


def test_parse_args_noisy_benchmark_flags() -> None:
    args = parse_args(
        [
            "--noisy-methods",
            "suzuki2,cfqm4",
            "--benchmark-active-coeff-tol",
            "1e-9",
            "--aer-noise-kind",
            "scheduled",
            "--backend-profile",
            "generic_seeded",
            "--schedule-policy",
            "asap",
            "--layout-policy",
            "auto_then_lock",
            "--fixed-physical-patch",
            "0,1",
        ]
    )
    assert str(args.noisy_methods) == "suzuki2,cfqm4"
    assert float(args.benchmark_active_coeff_tol) == 1e-9
    assert str(args.aer_noise_kind) == "scheduled"
    assert str(args.backend_profile) == "generic_seeded"
    assert str(args.schedule_policy) == "asap"
    assert str(args.layout_policy) == "auto_then_lock"
    assert str(args.fixed_physical_patch) == "0,1"
    assert bool(args.disable_time_dynamics) is False


def test_parse_args_mitigation_defaults_and_values() -> None:
    args = parse_args([])
    assert str(args.mitigation) == "none"
    assert str(args.symmetry_mitigation_mode) == "off"
    assert args.zne_scales is None
    assert args.dd_sequence is None

    args = parse_args(
        [
            "--mitigation",
            "dd",
            "--symmetry-mitigation-mode",
            "projector_renorm_v1",
            "--zne-scales",
            "1.0,2.0",
            "--dd-sequence",
            "XY4",
        ]
    )
    assert str(args.mitigation) == "dd"
    assert str(args.symmetry_mitigation_mode) == "projector_renorm_v1"
    assert str(args.zne_scales) == "1.0,2.0"
    assert str(args.dd_sequence) == "XY4"


def test_parse_args_disable_time_dynamics_flag() -> None:
    args = parse_args(["--disable-time-dynamics"])
    assert bool(args.disable_time_dynamics) is True



def test_parse_args_paired_anchor_comparison_flag() -> None:
    args = parse_args(["--paired-anchor-comparison"])
    assert bool(args.paired_anchor_comparison) is True


def test_parse_args_runtime_twirling_flags() -> None:
    args = parse_args(
        [
            "--runtime-enable-gate-twirling",
            "--runtime-enable-measure-twirling",
            "--runtime-twirling-num-randomizations",
            "16",
            "--runtime-twirling-strategy",
            "active",
        ]
    )
    assert bool(args.runtime_enable_gate_twirling) is True
    assert bool(args.runtime_enable_measure_twirling) is True
    assert int(args.runtime_twirling_num_randomizations) == 16
    assert str(args.runtime_twirling_strategy) == "active"


def test_parse_args_runtime_layer_noise_model_json_flag() -> None:
    args = parse_args(["--runtime-layer-noise-model-json", "artifacts/json/layer_noise_model.json"])
    assert str(args.runtime_layer_noise_model_json) == "artifacts/json/layer_noise_model.json"



def test_hh_robustness_defaults_to_spsa_methods() -> None:
    args = _enforce_defaults_and_minimums(parse_args([]))
    assert str(args.warm_method) == "SPSA"
    assert str(args.final_method) == "SPSA"



def test_hh_robustness_rejects_non_spsa_methods_at_parse_time() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--warm-method", "COBYLA"])
    with pytest.raises(SystemExit):
        parse_args(["--final-method", "COBYLA"])


def test_mitigation_schema_defaults_and_caption() -> None:
    mit = _build_mitigation_config(
        mitigation="none",
        zne_scales=None,
        dd_sequence=None,
        layer_noise_model_json=None,
    )
    layer_noise_mit = _build_mitigation_config(
        mitigation="zne",
        zne_scales="1.0,2.0",
        dd_sequence=None,
        layer_noise_model_json="artifacts/json/layer_noise_model.json",
    )
    sym = _build_symmetry_mitigation_config(mode="postselect_diag_v1", L=2, ordering="blocked")
    assert mit == {"mode": "none", "zne_scales": [], "dd_sequence": None}
    assert layer_noise_mit == {
        "mode": "zne",
        "zne_scales": [1.0, 2.0],
        "dd_sequence": None,
        "layer_noise_model_json": "artifacts/json/layer_noise_model.json",
    }
    assert sym == {
        "mode": "postselect_diag_v1",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }

    caption = _noise_config_caption(
        {
            "shots": 2048,
            "oracle_repeats": 4,
            "oracle_aggregate": "mean",
            "mitigation_config": mit,
            "runtime_twirling_config": {
                "enable_gates": True,
                "enable_measure": False,
                "num_randomizations": 8,
                "strategy": "active",
            },
            "symmetry_mitigation_config": sym,
        },
        "shots",
    )
    assert "mitigation=none" in caption
    assert "runtime_twirling={'enable_gates': True" in caption
    assert "symmetry=postselect_diag_v1" in caption


def test_parse_noisy_methods_csv_validation() -> None:
    assert _parse_noisy_methods_csv("suzuki2,cfqm4,suzuki2") == ["suzuki2", "cfqm4"]


def test_normalize_display_string_list_uses_defaults_when_missing() -> None:
    assert _normalize_display_string_list(None, default=["cfqm4", "suzuki2"]) == ["cfqm4", "suzuki2"]
    assert _normalize_display_string_list([], default=["ideal", "shots", "aer_noise"]) == ["ideal", "shots", "aer_noise"]
    assert _normalize_display_string_list(["shots", "aer_noise"], default=["ideal"]) == ["shots", "aer_noise"]
    assert _normalize_display_string_list("cfqm4,suzuki2", default=["ideal"]) == ["cfqm4", "suzuki2"]


def test_pool_b_enforcement_passes_for_exact_family_set() -> None:
    audit = _validate_pool_b_strict_composition(
        {
            "raw_sizes": {"uccsd": 4, "hva": 6, "paop_full": 3},
            "dedup_source_presence_counts": {"uccsd": 4, "hva": 5, "paop_full": 3},
        }
    )
    assert bool(audit["passed"]) is True
    assert list(audit["required_families"]) == ["uccsd_lifted", "hva", "paop_full"]


def test_pool_b_enforcement_fails_on_missing_family() -> None:
    try:
        _validate_pool_b_strict_composition(
            {
                "raw_sizes": {"uccsd": 4, "hva": 6},
                "dedup_source_presence_counts": {"uccsd": 4, "hva": 5},
            }
        )
        raise AssertionError("Expected ValueError for missing Pool B family.")
    except ValueError as exc:
        assert "Pool B composition mismatch" in str(exc)


def test_cfqm_proxy_cost_is_deterministic() -> None:
    kwargs = dict(
        method="cfqm4",
        t_final=1.0,
        trotter_steps=4,
        drive_t0=0.0,
        drive_time_sampling="midpoint",
        ordered_labels_exyz=["ee", "xz", "yy"],
        static_coeff_map_exyz={"ee": 1.0 + 0.0j, "xz": 0.2 + 0.0j, "yy": -0.1 + 0.0j},
        drive_provider_exyz=None,
        active_coeff_tol=1e-12,
        coeff_drop_abs_tol=0.0,
    )
    c1 = _compute_time_dynamics_proxy_cost(**kwargs)
    c2 = _compute_time_dynamics_proxy_cost(**kwargs)
    assert c1 == c2
    assert int(c1["cx_proxy_total"]) >= 0
    assert int(c1["term_exp_count_total"]) > 0


def test_suzuki_and_cfqm_proxy_cost_sanity() -> None:
    base_kwargs = dict(
        t_final=1.0,
        trotter_steps=4,
        drive_t0=0.0,
        drive_time_sampling="midpoint",
        ordered_labels_exyz=["ee", "xz", "yy"],
        static_coeff_map_exyz={"ee": 1.0 + 0.0j, "xz": 0.2 + 0.0j, "yy": -0.1 + 0.0j},
        drive_provider_exyz=None,
        active_coeff_tol=1e-12,
        coeff_drop_abs_tol=0.0,
    )
    suz = _compute_time_dynamics_proxy_cost(method="suzuki2", **base_kwargs)
    cfq = _compute_time_dynamics_proxy_cost(method="cfqm4", **base_kwargs)
    for rec in (suz, cfq):
        assert set(rec.keys()) == {
            "term_exp_count_total",
            "pauli_rot_count_total",
            "cx_proxy_total",
            "sq_proxy_total",
            "depth_proxy_total",
        }
        assert int(rec["term_exp_count_total"]) >= 0
        assert int(rec["depth_proxy_total"]) == int(rec["pauli_rot_count_total"])


def _generic_backend_2q():
    from qiskit.providers.fake_provider import GenericBackendV2

    return GenericBackendV2(
        2,
        basis_gates=["id", "rz", "sx", "x", "cx", "measure", "delay", "reset"],
        coupling_map=[[0, 1], [1, 0]],
        dt=2.2222222222222221e-10,
        seed=7,
        noise_info=True,
    )



def test_layout_lock_mode_token_shares_patch_replay_family() -> None:
    shared_token = _layout_lock_mode_token("patch_snapshot")
    assert _layout_lock_mode_token("backend_scheduled") == shared_token
    assert _layout_lock_mode_token("runtime") == shared_token
    assert _layout_lock_mode_token("qpu_raw") == shared_token
    assert _layout_lock_mode_token("qpu_suppressed") == shared_token



def test_patch_snapshot_replay_is_executable_through_noisy_method_trajectory(tmp_path: Path) -> None:
    pytest.importorskip("qiskit_aer")

    backend = _generic_backend_2q()
    snapshot = freeze_backend_snapshot(backend)
    snapshot_path = tmp_path / "patch_replay_snapshot.json"
    write_calibration_snapshot(snapshot_path, snapshot)

    common_kwargs = {
        "L": 1,
        "ordering": "blocked",
        "psi_seed": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=complex),
        "ordered_labels_exyz": ["zz"],
        "static_coeff_map_exyz": {"zz": 0.25 + 0.0j},
        "t_final": 0.2,
        "num_times": 1,
        "trotter_steps": 1,
        "drive_profile": None,
        "shots": 64,
        "seed": 21,
        "oracle_repeats": 1,
        "oracle_aggregate": "mean",
        "mitigation_config": {"mode": "none", "zne_scales": [], "dd_sequence": None},
        "runtime_twirling_config": {
            "enable_gates": False,
            "enable_measure": False,
            "num_randomizations": None,
            "strategy": None,
        },
        "symmetry_mitigation_config": {
            "mode": "off",
            "num_sites": 1,
            "ordering": "blocked",
            "sector_n_up": 1,
            "sector_n_dn": 0,
        },
        "backend_name": None,
        "use_fake_backend": False,
        "backend_profile": "frozen_snapshot_json",
        "aer_noise_kind": "scheduled",
        "schedule_policy": "asap",
        "noise_snapshot_json": str(snapshot_path),
        "fixed_physical_patch": None,
        "allow_noisy_fallback": False,
        "omp_shm_workaround": True,
        "layout_lock_key": "bench_patch_replay_shared",
        "method": "suzuki2",
        "benchmark_active_coeff_tol": 1e-12,
        "cfqm_coeff_drop_abs_tol": 0.0,
    }

    captured = _run_noisy_method_trajectory(
        noise_mode="backend_scheduled",
        layout_policy="auto_then_lock",
        **common_kwargs,
    )
    replayed = _run_noisy_method_trajectory(
        noise_mode="patch_snapshot",
        layout_policy="frozen_layout",
        **common_kwargs,
    )

    captured_details = dict(captured.get("backend_info", {}).get("details", {}))
    replayed_details = dict(replayed.get("backend_info", {}).get("details", {}))

    assert bool(replayed.get("success", False)) is True
    assert captured_details.get("provenance_summary", {}).get("classification") == "local_snapshot_replay"
    assert replayed_details.get("resolved_noise_spec", {}).get("noise_kind") == "patch_snapshot"
    assert replayed_details.get("provenance_summary", {}).get("classification") == "local_patch_frozen_replay"
    assert replayed_details.get("snapshot_hash") == snapshot.snapshot_hash
    assert replayed_details.get("used_physical_qubits") == captured_details.get("used_physical_qubits")
    assert replayed_details.get("used_physical_edges") == captured_details.get("used_physical_edges")
    assert replayed_details.get("layout_hash") == captured_details.get("layout_hash")



def test_collect_noisy_benchmark_rows_schema_and_values() -> None:
    dyn_noisy = {
        "profiles": {
            "static": {
                "methods": {
                    "suzuki2": {
                        "modes": {
                            "shots": {
                                "success": True,
                                "backend_info": {
                                    "details": {
                                        "source_kind": "fake_snapshot",
                                        "provenance_summary": {"classification": "local_generic_aer_execution"},
                                        "snapshot_hash": "snap_a",
                                        "layout_hash": "layout_a",
                                        "omitted_channels": ["crosstalk"],
                                        "scheduled_duration_total": 1.2,
                                        "idle_duration_total": 0.1,
                                        "used_physical_qubits": [0, 1],
                                        "used_physical_edges": [[0, 1]],
                                    }
                                },
                                "delta_uncertainty": {
                                    "energy_total": {
                                        "max_abs_delta": 0.02,
                                        "max_abs_delta_over_stderr": 4.0,
                                        "mean_abs_delta_over_stderr": 2.5,
                                    }
                                },
                                "benchmark_cost": {
                                    "term_exp_count_total": 100,
                                    "pauli_rot_count_total": 100,
                                    "cx_proxy_total": 220,
                                    "sq_proxy_total": 340,
                                    "depth_proxy_total": 100,
                                },
                                "benchmark_runtime": {
                                    "wall_total_s": 1.25,
                                    "oracle_eval_s_total": 0.55,
                                    "oracle_calls_total": 120,
                                },
                            }
                        }
                    },
                    "cfqm4": {
                        "modes": {
                            "shots": {
                                "success": True,
                                "backend_info": {
                                    "details": {
                                        "source_kind": "fake_snapshot",
                                        "provenance_summary": {"classification": "local_snapshot_replay"},
                                        "snapshot_hash": "snap_b",
                                        "layout_hash": "layout_b",
                                        "omitted_channels": ["crosstalk"],
                                        "scheduled_duration_total": 1.0,
                                        "idle_duration_total": 0.05,
                                        "used_physical_qubits": [0, 1],
                                        "used_physical_edges": [[0, 1]],
                                    }
                                },
                                "delta_uncertainty": {
                                    "energy_total": {
                                        "max_abs_delta": 0.015,
                                        "max_abs_delta_over_stderr": 3.0,
                                        "mean_abs_delta_over_stderr": 2.0,
                                    }
                                },
                                "benchmark_cost": {
                                    "term_exp_count_total": 88,
                                    "pauli_rot_count_total": 88,
                                    "cx_proxy_total": 180,
                                    "sq_proxy_total": 300,
                                    "depth_proxy_total": 88,
                                },
                                "benchmark_runtime": {
                                    "wall_total_s": 1.10,
                                    "oracle_eval_s_total": 0.48,
                                    "oracle_calls_total": 120,
                                },
                            }
                        }
                    },
                },
                "modes": {},
            }
        }
    }
    rows = _collect_noisy_benchmark_rows(dyn_noisy)
    assert len(rows) == 2
    methods = {str(r["method"]) for r in rows}
    assert methods == {"suzuki2", "cfqm4"}
    for row in rows:
        assert set(row.keys()) == {
            "profile",
            "method",
            "mode",
            "benchmark_cell_id",
            "source_kind",
            "provenance_classification",
            "snapshot_hash",
            "layout_hash",
            "transpile_hash",
            "circuit_structure_hash",
            "layout_anchor_source",
            "runtime_execution_bundle",
            "fixed_couplers_status",
            "omitted_channels",
            "term_exp_count_total",
            "pauli_rot_count_total",
            "cx_proxy_total",
            "sq_proxy_total",
            "depth_proxy_total",
            "wall_total_s",
            "oracle_eval_s_total",
            "oracle_calls_total",
            "scheduled_duration_total",
            "idle_duration_total",
            "used_physical_qubits",
            "used_physical_edges",
            "max_abs_delta",
            "max_abs_delta_over_stderr",
            "mean_abs_delta_over_stderr",
        }
    classes = {str(row["provenance_classification"]) for row in rows}
    assert classes == {"local_generic_aer_execution", "local_snapshot_replay"}


def test_disabled_hardcoded_superset_metadata_and_summary_shape() -> None:
    hardcoded = _disabled_hardcoded_superset_meta()
    assert bool(hardcoded["disabled"]) is True
    assert hardcoded["profiles"] == {}
    assert "final-only dynamics" in str(hardcoded.get("reason", ""))

    payload = {
        "stage_pipeline": {
            "warm_start": {"delta_abs": 0.1, "stop_reason": "warm_done"},
            "adapt_pool_b": {"delta_abs": 0.01, "stop_reason": "adapt_done"},
            "conventional_vqe": {"delta_abs": 0.001, "stop_reason": "final_done"},
        },
        "hardcoded_superset": hardcoded,
        "dynamics_noisy": {
            "profiles": {
                "static": {
                    "modes": {},
                    "methods": {
                        "suzuki2": {
                            "modes": {
                                "shots": {
                                    "success": True,
                                    "delta_uncertainty": {
                                        "energy_total": {
                                            "max_abs_delta": 0.01,
                                            "max_abs_delta_over_stderr": 5.0,
                                            "mean_abs_delta_over_stderr": 3.5,
                                        }
                                    },
                                }
                            }
                        }
                    },
                }
            }
        },
        "dynamics_benchmarks": {"rows": [{"profile": "static", "method": "suzuki2", "mode": "shots"}]},
    }
    summary = _build_summary(payload)
    assert int(summary["noisy_method_modes_total"]) == 1
    assert int(summary["noisy_method_modes_completed"]) == 1
    assert int(summary["dynamics_benchmark_rows"]) == 1
    assert float(summary["max_abs_delta"]) == 0.01
    assert float(summary["max_abs_delta_over_stderr"]) == 5.0
    assert float(summary["mean_abs_delta_over_stderr"]) == 3.5


def test_summary_uses_noisy_final_audit_when_dynamics_disabled() -> None:
    payload = {
        "stage_pipeline": {
            "warm_start": {"delta_abs": 0.1, "stop_reason": "warm_done"},
            "adapt_pool_b": {"delta_abs": 0.01, "stop_reason": "adapt_done"},
            "conventional_vqe": {"delta_abs": 0.001, "stop_reason": "final_done"},
        },
        "dynamics_noisy": {"profiles": {}},
        "noisy_final_audit": {
            "profiles": {
                "static": {
                    "modes": {
                        "shots": {
                            "success": True,
                            "delta_uncertainty": {
                                "energy_total": {
                                    "max_abs_delta": 0.02,
                                    "max_abs_delta_over_stderr": 4.5,
                                    "mean_abs_delta_over_stderr": 2.8,
                                }
                            },
                        },
                        "aer_noise": {
                            "success": False,
                            "reason": "env_blocked",
                        },
                    }
                }
            }
        },
        "dynamics_benchmarks": {"rows": []},
    }
    summary = _build_summary(payload)
    assert int(summary["noisy_audit_modes_total"]) == 2
    assert int(summary["noisy_audit_modes_completed"]) == 1
    assert float(summary["noisy_audit_max_abs_delta"]) == 0.02
    assert float(summary["noisy_audit_max_abs_delta_over_stderr"]) == 4.5
    assert float(summary["noisy_audit_mean_abs_delta_over_stderr"]) == 2.8
    # Combined summary fields should still be populated even with no trajectories.
    assert float(summary["max_abs_delta"]) == 0.02


def test_noise_style_legend_semantics_tokens_present() -> None:
    text = "\n".join(_noise_style_legend_lines())
    assert "Δ(noisy-ideal)" in text
    assert "noiseless (final-seed Suzuki-2)" in text



def _paired_mode_result(
    *,
    mode: str,
    mitigation_mode: str | None = None,
    runtime_twirling: dict[str, object] | None = None,
    anchor_source: str = "persisted_lock",
) -> dict[str, object]:
    twirling_cfg = dict(runtime_twirling or {})
    runtime_bundle = None
    executor = "aer"
    provenance = "local_snapshot_replay"
    if mode == "qpu_raw":
        executor = "runtime_qpu"
        provenance = "runtime_submitted_raw"
        runtime_bundle = {
            "requested_noise_mode": "qpu_raw",
            "resolved_noise_kind": "qpu_raw",
            "mitigation_bundle": "none",
            "mitigation_mode": "none",
            "twirling_enable_gates": False,
            "twirling_enable_measure": False,
            "twirling_num_randomizations": None,
            "twirling_strategy": None,
            "trex_like_measure_suppression": False,
            "suppression_components": [],
        }
    elif mode == "qpu_suppressed":
        executor = "runtime_qpu"
        provenance = "runtime_submitted_suppressed"
        suppression_components: list[str] = []
        if str(mitigation_mode) == "readout":
            suppression_components.append("measure_mitigation")
        if bool(twirling_cfg.get("enable_gates", False)):
            suppression_components.append("gate_twirling")
        if bool(twirling_cfg.get("enable_measure", False)):
            suppression_components.append("measure_twirling")
            if str(mitigation_mode) == "readout":
                suppression_components.append("trex_like_readout")
        runtime_bundle = {
            "requested_noise_mode": "qpu_suppressed",
            "resolved_noise_kind": "qpu_suppressed",
            "mitigation_bundle": "runtime_suppressed",
            "mitigation_mode": str(mitigation_mode),
            "twirling_enable_gates": bool(twirling_cfg.get("enable_gates", False)),
            "twirling_enable_measure": bool(twirling_cfg.get("enable_measure", False)),
            "twirling_num_randomizations": twirling_cfg.get("num_randomizations", None),
            "twirling_strategy": twirling_cfg.get("strategy", None),
            "trex_like_measure_suppression": bool(
                twirling_cfg.get("enable_measure", False) and str(mitigation_mode) == "readout"
            ),
            "suppression_components": suppression_components,
        }
    return {
        "success": True,
        "noise_mode": str(mode),
        "trajectory": [
            {
                "time": 0.0,
                "energy_total_noisy": 0.1,
                "energy_total_delta_noisy_minus_ideal": 0.01,
            }
        ],
        "backend_info": {
            "backend_name": "ibm_fake_runtime",
            "details": {
                "resolved_noise_spec": {"executor": executor, "noise_kind": str(mode)},
                "layout_hash": "layout:shared",
                "transpile_hash": "tx:shared",
                "snapshot_hash": "snap:shared",
                "used_physical_qubits": [0, 1],
                "used_physical_edges": [[0, 1]],
                "circuit_structure_hash": "cstruct:shared",
                "layout_anchor_source": anchor_source,
                "runtime_execution_bundle": runtime_bundle,
                "provenance_summary": {"classification": provenance},
                "fixed_couplers_status": {"requested": False, "enforced": False, "verified": False},
            },
        },
    }



def test_run_paired_anchor_method_trajectory_set_reuses_same_lock_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mode_calls: list[dict[str, object]] = []

    def _fake_run_noisy_mode_isolated(*, kwargs: dict[str, object], timeout_s: int) -> dict[str, object]:
        mode_calls.append({"kwargs": kwargs, "timeout_s": timeout_s})
        mode = str(kwargs["noise_mode"])
        anchor_source = "fresh_auto_then_lock" if mode == "backend_scheduled" else "persisted_lock"
        mitigation_mode = str(dict(kwargs["mitigation_config"]).get("mode"))
        return _paired_mode_result(
            mode=mode,
            mitigation_mode=mitigation_mode,
            runtime_twirling=dict(kwargs.get("runtime_twirling_config", {})),
            anchor_source=anchor_source,
        )

    monkeypatch.setattr(
        "pipelines.exact_bench.hh_noise_robustness_seq_report._run_noisy_mode_isolated",
        _fake_run_noisy_mode_isolated,
    )

    payload = _run_paired_anchor_method_trajectory_set(
        L=1,
        ordering="blocked",
        psi_seed=np.asarray([1.0, 0.0], dtype=complex),
        ordered_labels_exyz=["z"],
        static_coeff_map_exyz={"z": 0.5 + 0.0j},
        t_final=0.2,
        num_times=2,
        trotter_steps=1,
        drive_profile=None,
        shots=64,
        seed=7,
        oracle_repeats=1,
        oracle_aggregate="mean",
        mitigation_config={"mode": "readout", "zne_scales": [], "dd_sequence": None},
        runtime_twirling_config={
            "enable_gates": True,
            "enable_measure": False,
            "num_randomizations": 16,
            "strategy": "active",
        },
        symmetry_mitigation_config={"mode": "off"},
        backend_name="ibm_fake_runtime",
        use_fake_backend=False,
        backend_profile="live_backend",
        aer_noise_kind="scheduled",
        schedule_policy="asap",
        layout_policy="auto_then_lock",
        noise_snapshot_json=None,
        fixed_physical_patch=None,
        allow_noisy_fallback=False,
        omp_shm_workaround=True,
        layout_lock_key="paired:shared",
        method="cfqm4",
        benchmark_active_coeff_tol=1e-12,
        cfqm_coeff_drop_abs_tol=0.0,
        noisy_mode_timeout_s=120,
        profile_name="static",
    )

    assert payload["local_mode"] == "backend_scheduled"
    assert [row["label"] for row in payload["rows"]] == [
        "local_patch_anchor",
        "runtime_raw",
        "runtime_suppressed",
    ]
    assert {str(rec["kwargs"]["layout_lock_key"]) for rec in mode_calls} == {"paired:shared"}
    assert {str(rec["kwargs"]["noise_mode"]) for rec in mode_calls} == {
        "backend_scheduled",
        "qpu_raw",
        "qpu_suppressed",
    }
    raw_row = next(row for row in payload["rows"] if row["label"] == "runtime_raw")
    suppressed_row = next(row for row in payload["rows"] if row["label"] == "runtime_suppressed")
    assert raw_row["runtime_execution_bundle"]["mitigation_mode"] == "none"
    assert suppressed_row["runtime_execution_bundle"]["mitigation_mode"] == "readout"
    assert payload["runtime_twirling_config"] == {
        "enable_gates": True,
        "enable_measure": False,
        "num_randomizations": 16,
        "strategy": "active",
    }
    suppressed_call = next(rec for rec in mode_calls if str(rec["kwargs"]["noise_mode"]) == "qpu_suppressed")
    assert suppressed_call["kwargs"]["runtime_twirling_config"] == payload["runtime_twirling_config"]
    assert suppressed_row["runtime_execution_bundle"]["twirling_enable_gates"] is True
    assert suppressed_row["runtime_execution_bundle"]["twirling_num_randomizations"] == 16
    assert suppressed_row["runtime_execution_bundle"]["suppression_components"] == [
        "measure_mitigation",
        "gate_twirling",
    ]
    cmp = payload["comparability"]
    assert cmp["same_lock_context"] is True
    assert cmp["local_vs_runtime_same_anchor_evidence_status"] == "fully_evidenced"
    assert cmp["runtime_pair_same_submitted_evidence_status"] == "fully_evidenced"
    assert cmp["runtime_pair_differs_only_in_execution_bundle"] is True



def test_run_paired_anchor_method_trajectory_set_fails_explicitly_for_missing_frozen_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run_noisy_mode_isolated(*, kwargs: dict[str, object], timeout_s: int) -> dict[str, object]:
        if str(kwargs["noise_mode"]) == "patch_snapshot":
            return {
                "success": False,
                "reason": "worker_exception",
                "error": (
                    "RuntimeError: layout_policy='frozen_layout' requires an existing persisted layout lock "
                    "or replay artifact; no prior locked layout was found for this backend/snapshot/key."
                ),
            }
        return _paired_mode_result(mode=str(kwargs["noise_mode"]), mitigation_mode="readout")

    monkeypatch.setattr(
        "pipelines.exact_bench.hh_noise_robustness_seq_report._run_noisy_mode_isolated",
        _fake_run_noisy_mode_isolated,
    )

    with pytest.raises(RuntimeError, match="layout_policy='frozen_layout' requires an existing persisted layout lock"):
        _run_paired_anchor_method_trajectory_set(
            L=1,
            ordering="blocked",
            psi_seed=np.asarray([1.0, 0.0], dtype=complex),
            ordered_labels_exyz=["z"],
            static_coeff_map_exyz={"z": 0.5 + 0.0j},
            t_final=0.2,
            num_times=2,
            trotter_steps=1,
            drive_profile=None,
            shots=64,
            seed=7,
            oracle_repeats=1,
            oracle_aggregate="mean",
            mitigation_config={"mode": "readout", "zne_scales": [], "dd_sequence": None},
            runtime_twirling_config={
                "enable_gates": False,
                "enable_measure": False,
                "num_randomizations": None,
                "strategy": None,
            },
            symmetry_mitigation_config={"mode": "off"},
            backend_name="ibm_fake_runtime",
            use_fake_backend=False,
            backend_profile="frozen_snapshot_json",
            aer_noise_kind="scheduled",
            schedule_policy="asap",
            layout_policy="frozen_layout",
            noise_snapshot_json="artifacts/json/frozen_snapshot.json",
            fixed_physical_patch=None,
            allow_noisy_fallback=False,
            omp_shm_workaround=True,
            layout_lock_key="paired:frozen",
            method="cfqm4",
            benchmark_active_coeff_tol=1e-12,
            cfqm_coeff_drop_abs_tol=0.0,
            noisy_mode_timeout_s=120,
            profile_name="static",
        )



def test_paired_anchor_summary_rows_encode_evidence_statuses() -> None:
    rows = _paired_anchor_summary_rows(
        {
            "profiles": {
                "static": {
                    "methods": {
                        "cfqm4": {
                            "executed": True,
                            "local_mode": "backend_scheduled",
                            "reason": "",
                            "comparability": {
                                "local_vs_runtime_same_anchor_evidence_status": "fully_evidenced",
                                "runtime_pair_same_submitted_evidence_status": "fully_evidenced",
                                "exact_anchor_identity_evidence_status": "fully_evidenced",
                                "runtime_pair_differs_only_in_execution_bundle": True,
                            },
                        }
                    }
                }
            }
        }
    )
    assert rows == [[
        "static",
        "cfqm4",
        "backend_scheduled",
        "True",
        "fully_evidenced",
        "fully_evidenced",
        "fully_evidenced",
        "True",
        "",
    ]]
