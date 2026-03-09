from __future__ import annotations

import json

import pytest

from pipelines.exact_bench.hh_noise_hardware_validation import (
    _build_mitigation_config_from_args,
    _build_symmetry_mitigation_config_from_args,
    _apply_defaults_and_minimums,
    _default_validation_ansatz,
    _effective_layout_lock_key,
    _ground_state_report_fields,
    _load_imported_vqe_parameters,
    _mode_honesty_statement,
    _theta_hash,
    _vqe_energy_block_rows,
    parse_args,
)


def test_hh_defaults_applied_from_minimum_table_l2_nph1() -> None:
    args = parse_args(["--L", "2"])
    args = _apply_defaults_and_minimums(args)

    assert str(args.ansatz) == "hh_hva_ptw"
    assert int(args.vqe_reps) == 2
    assert int(args.vqe_restarts) == 3
    assert int(args.vqe_maxiter) == 800
    assert int(args.trotter_steps) == 64
    assert str(args.vqe_method) == "SPSA"


@pytest.mark.parametrize("method", ["COBYLA", "SLSQP"])
def test_hh_non_spsa_method_rejected(method: str) -> None:
    args = parse_args(["--L", "2", "--vqe-method", method])
    with pytest.raises(ValueError, match="HH validation is SPSA-only for --vqe-method"):
        _apply_defaults_and_minimums(args)



def test_hh_under_minimum_rejected_without_smoke_flag() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--vqe-reps",
            "1",
            "--vqe-restarts",
            "1",
            "--vqe-maxiter",
            "100",
            "--trotter-steps",
            "8",
        ]
    )
    with pytest.raises(ValueError):
        _apply_defaults_and_minimums(args)


def test_hh_under_minimum_allowed_with_smoke_flag() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--vqe-reps",
            "1",
            "--vqe-restarts",
            "1",
            "--vqe-maxiter",
            "100",
            "--trotter-steps",
            "8",
            "--smoke-test-intentionally-weak",
        ]
    )
    args = _apply_defaults_and_minimums(args)

    assert int(args.vqe_reps) == 1
    assert int(args.vqe_restarts) == 1
    assert int(args.vqe_maxiter) == 100
    assert int(args.trotter_steps) == 8


def test_problem_aware_default_validation_ansatz() -> None:
    assert _default_validation_ansatz("hh") == "hh_hva_ptw"
    assert _default_validation_ansatz("hubbard") == "hva"



def test_hubbard_defaults_use_problem_aware_ansatz() -> None:
    args = parse_args(["--problem", "hubbard", "--L", "2"])
    args = _apply_defaults_and_minimums(args)

    assert str(args.ansatz) == "hva"
    assert str(args.vqe_method) == "COBYLA"



def test_hubbard_under_minimum_rejected_without_smoke_flag() -> None:
    args = parse_args(
        [
            "--problem",
            "hubbard",
            "--ansatz",
            "hva",
            "--L",
            "4",
            "--vqe-reps",
            "1",
            "--vqe-restarts",
            "1",
            "--vqe-maxiter",
            "100",
            "--trotter-steps",
            "16",
        ]
    )
    with pytest.raises(ValueError):
        _apply_defaults_and_minimums(args)


def test_cli_parses_fallback_flags() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--no-allow-noisy-fallback",
            "--no-omp-shm-workaround",
        ]
    )
    assert bool(args.allow_noisy_fallback) is False
    assert bool(args.omp_shm_workaround) is False


def test_cli_parses_new_noise_plumbing_flags() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--aer-noise-kind",
            "basic",
            "--backend-profile",
            "fake_snapshot",
            "--schedule-policy",
            "asap",
            "--layout-policy",
            "fixed_patch",
            "--noise-snapshot-json",
            "artifacts/json/frozen_snapshot.json",
            "--fixed-physical-patch",
            "1,3",
            "--allow-noisy-fallback",
        ]
    )
    assert str(args.aer_noise_kind) == "basic"
    assert str(args.backend_profile) == "fake_snapshot"
    assert str(args.schedule_policy) == "asap"
    assert str(args.layout_policy) == "fixed_patch"
    assert str(args.noise_snapshot_json).endswith("artifacts/json/frozen_snapshot.json")
    assert str(args.fixed_physical_patch) == "1,3"
    assert bool(args.allow_noisy_fallback) is True


def test_cli_parses_layout_lock_and_imported_theta_flags() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--layout-lock-key",
            "shared_l2_hh_trio",
            "--vqe-parameter-source",
            "imported_json",
            "--vqe-parameter-json",
            "artifacts/json/l2_hh_ideal.json",
        ]
    )
    assert str(args.layout_lock_key) == "shared_l2_hh_trio"
    assert str(args.vqe_parameter_source) == "imported_json"
    assert str(args.vqe_parameter_json).endswith("artifacts/json/l2_hh_ideal.json")


def test_effective_layout_lock_key_uses_override_or_stable_default() -> None:
    assert _effective_layout_lock_key(
        layout_lock_key="shared_l2_hh_trio",
        L=2,
        problem="hh",
        ansatz="hh_hva_ptw",
        noise_mode="backend_scheduled",
    ) == "shared_l2_hh_trio"
    assert _effective_layout_lock_key(
        layout_lock_key=None,
        L=2,
        problem="hh",
        ansatz="hh_hva_ptw",
        noise_mode="backend_scheduled",
    ) == "validation:L2:hh:hh_hva_ptw:backend_scheduled"


def test_cli_parses_legacy_parity_flags() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--legacy-reference-json",
            "artifacts/json/hc_hh_L2_static_t1.0_U2.0_g1.0_nph1.json",
            "--legacy-parity-tol",
            "1e-10",
            "--output-compare-plot",
            "artifacts/pdf/hh_noise_cmp.png",
            "--compare-observables",
            "energy_static_trotter,doublon_trotter",
        ]
    )
    assert str(args.legacy_reference_json).endswith(
        "artifacts/json/hc_hh_L2_static_t1.0_U2.0_g1.0_nph1.json"
    )
    assert float(args.legacy_parity_tol) == pytest.approx(1e-10)
    assert str(args.output_compare_plot).endswith("artifacts/pdf/hh_noise_cmp.png")
    assert str(args.compare_observables) == "energy_static_trotter,doublon_trotter"


def test_cli_parses_mitigation_flags_defaults_and_values() -> None:
    args = parse_args(["--L", "2"])
    assert str(args.mitigation) == "none"
    assert args.zne_scales is None
    assert args.dd_sequence is None
    assert _build_mitigation_config_from_args(args) == {
        "mode": "none",
        "zne_scales": [],
        "dd_sequence": None,
    }

    args = parse_args(
        [
            "--L",
            "2",
            "--mitigation",
            "zne",
            "--zne-scales",
            "1.0,2.0,3.0",
            "--dd-sequence",
            "XY4",
        ]
    )
    assert str(args.mitigation) == "zne"
    assert str(args.zne_scales) == "1.0,2.0,3.0"
    assert str(args.dd_sequence) == "XY4"


def test_cli_parses_symmetry_mitigation_flags_defaults_and_values() -> None:
    args = parse_args(["--L", "2"])
    assert str(args.symmetry_mitigation_mode) == "off"
    assert _build_symmetry_mitigation_config_from_args(args) == {
        "mode": "off",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }

    args = parse_args(
        [
            "--L",
            "2",
            "--symmetry-mitigation-mode",
            "postselect_diag_v1",
        ]
    )
    assert str(args.symmetry_mitigation_mode) == "postselect_diag_v1"
    assert _build_symmetry_mitigation_config_from_args(args) == {
        "mode": "postselect_diag_v1",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }


def test_ground_state_report_fields_include_exact_filtered_energy() -> None:
    rows = _ground_state_report_fields(
        {
            "exact_energy_filtered": 0.1586679041257264,
            "exact_energy_full_hilbert": 0.123,
            "filtered_sector": {"n_up": 1, "n_dn": 1},
        }
    )
    assert rows[0] == ("Exact ground-state (filtered sector)", 0.1586679041257264)
    assert rows[1] == ("Filtered sector", "n_up=1, n_dn=1")
    assert rows[2] == ("Exact ground-state (full Hilbert)", 0.123)


def test_report_helpers_expose_mode_honesty_and_energy_block_rows() -> None:
    assert _mode_honesty_statement({"noise_kind": "backend_basic", "executor": "aer"}).startswith(
        "backend_basic:"
    )
    rows = _vqe_energy_block_rows(
        {
            "energy_ideal_reference": -1.2,
            "energy_noisy": -1.1,
            "delta_ansatz": 0.05,
            "delta_total": 0.15,
            "delta_noisy_minus_ideal": 0.1,
            "energy_qiskit_ideal_control": -1.2,
            "delta_qiskit_ideal_control_minus_ideal_reference": 0.0,
        },
        {"exact_energy_filtered": -1.25},
    )
    assert rows[0] == ("E_exact_filtered", -1.25)
    assert rows[1] == ("E_ideal_ref", -1.2)
    assert rows[2] == ("E_noisy", -1.1)


def test_load_imported_vqe_parameters_reads_theta_and_metadata(tmp_path) -> None:
    src = tmp_path / "baseline.json"
    src.write_text(
        json.dumps(
            {
                "settings": {"problem": "hh", "ansatz": "hh_hva_ptw", "L": 2},
                "vqe": {"optimal_point": [0.1, -0.2, 0.3]},
            }
        ),
        encoding="utf-8",
    )
    theta, meta = _load_imported_vqe_parameters(
        src,
        current_problem="hh",
        current_ansatz="hh_hva_ptw",
        current_L=2,
    )
    assert list(theta.tolist()) == pytest.approx([0.1, -0.2, 0.3])
    assert meta["kind"] == "imported_json"
    assert meta["path"] == str(src)
    assert meta["theta_hash"] == _theta_hash(theta)


def test_load_imported_vqe_parameters_rejects_mismatch(tmp_path) -> None:
    src = tmp_path / "baseline_bad.json"
    src.write_text(
        json.dumps(
            {
                "settings": {"problem": "hh", "ansatz": "hh_hva", "L": 2},
                "vqe": {"optimal_point": [0.1, -0.2]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Imported VQE parameter JSON mismatch"):
        _load_imported_vqe_parameters(
            src,
            current_problem="hh",
            current_ansatz="hh_hva_ptw",
            current_L=2,
        )


def test_load_imported_vqe_parameters_rejects_missing_settings_metadata(tmp_path) -> None:
    src = tmp_path / "baseline_missing_meta.json"
    src.write_text(
        json.dumps(
            {
                "settings": {"problem": "hh", "L": 2},
                "vqe": {"optimal_point": [0.1, -0.2, 0.3]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required settings fields"):
        _load_imported_vqe_parameters(
            src,
            current_problem="hh",
            current_ansatz="hh_hva_ptw",
            current_L=2,
        )


def test_load_imported_vqe_parameters_rejects_nonfinite_theta(tmp_path) -> None:
    src = tmp_path / "baseline_nonfinite.json"
    src.write_text(
        json.dumps(
            {
                "settings": {"problem": "hh", "ansatz": "hh_hva_ptw", "L": 2},
                "vqe": {"optimal_point": [0.1, "NaN", 0.3]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite theta"):
        _load_imported_vqe_parameters(
            src,
            current_problem="hh",
            current_ansatz="hh_hva_ptw",
            current_L=2,
        )
