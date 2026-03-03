from __future__ import annotations

import numpy as np
import pytest

from pipelines.exact_bench.hh_noise_hardware_validation import (
    _compute_legacy_parity,
    _write_legacy_comparison_plot,
    _build_adapt_pool,
    _build_hamiltonian,
    _half_filled_particles,
    parse_args,
)


def test_parse_args_phase2_adapt_flags_present() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--run-adapt",
            "--no-run-vqe",
            "--initial-state-source",
            "adapt",
            "--adapt-pool",
            "hva",
            "--adapt-max-depth",
            "7",
            "--adapt-gradient-step",
            "0.2",
            "--no-allow-aer-fallback",
        ]
    )
    assert bool(args.run_adapt) is True
    assert bool(args.run_vqe) is False
    assert str(args.initial_state_source) == "adapt"
    assert str(args.adapt_pool) == "hva"
    assert int(args.adapt_max_depth) == 7
    assert float(args.adapt_gradient_step) == pytest.approx(0.2)
    assert bool(args.allow_aer_fallback) is False


def test_build_adapt_pool_hh_hva_nonempty() -> None:
    args = parse_args(["--L", "2", "--problem", "hh", "--adapt-pool", "hva"])
    num_particles = _half_filled_particles(int(args.L))
    h_poly = _build_hamiltonian(args)
    pool = _build_adapt_pool(args=args, h_poly=h_poly, num_particles=num_particles)
    assert len(pool) > 0


def test_build_adapt_pool_hubbard_uccsd_nonempty() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--problem",
            "hubbard",
            "--ansatz",
            "hva",
            "--adapt-pool",
            "uccsd",
        ]
    )
    num_particles = _half_filled_particles(int(args.L))
    h_poly = _build_hamiltonian(args)
    pool = _build_adapt_pool(args=args, h_poly=h_poly, num_particles=num_particles)
    assert len(pool) > 0


def _legacy_ref_fixture() -> dict[str, object]:
    return {
        "reference_json": "legacy.json",
        "times": np.asarray([0.0, 1.0], dtype=float),
        "series": {
            "energy_static_trotter": np.asarray([1.0, 2.0], dtype=float),
            "doublon_trotter": np.asarray([0.2, 0.3], dtype=float),
        },
    }


def _new_traj_fixture(*, energy_last: float = 2.0, t_last: float = 1.0) -> list[dict[str, float]]:
    return [
        {
            "time": 0.0,
            "energy_static_trotter_noisy": 1.1,
            "energy_static_trotter_ideal": 1.0,
            "doublon_trotter_noisy": 0.21,
            "doublon_trotter_ideal": 0.2,
        },
        {
            "time": float(t_last),
            "energy_static_trotter_noisy": 2.1,
            "energy_static_trotter_ideal": float(energy_last),
            "doublon_trotter_noisy": 0.31,
            "doublon_trotter_ideal": 0.3,
        },
    ]


def test_legacy_parity_exact_match_passes_strict_tol() -> None:
    res = _compute_legacy_parity(
        legacy_ref=_legacy_ref_fixture(),
        new_trajectory=_new_traj_fixture(),
        observables=["energy_static_trotter", "doublon_trotter"],
        tolerance=1e-10,
    )
    assert bool(res["time_grid_match"]) is True
    assert bool(res["passed_all"]) is True
    assert bool(res["per_observable"]["energy_static_trotter"]["passed"]) is True
    assert bool(res["per_observable"]["doublon_trotter"]["passed"]) is True


def test_legacy_parity_tiny_mismatch_fails_strict_tol() -> None:
    res = _compute_legacy_parity(
        legacy_ref=_legacy_ref_fixture(),
        new_trajectory=_new_traj_fixture(energy_last=2.0 + 1e-9),
        observables=["energy_static_trotter", "doublon_trotter"],
        tolerance=1e-10,
    )
    assert bool(res["time_grid_match"]) is True
    assert bool(res["passed_all"]) is False
    assert bool(res["per_observable"]["energy_static_trotter"]["passed"]) is False
    assert float(res["per_observable"]["energy_static_trotter"]["max_abs_delta"]) > 1e-10


def test_legacy_parity_time_grid_mismatch_fails_with_reason() -> None:
    res = _compute_legacy_parity(
        legacy_ref=_legacy_ref_fixture(),
        new_trajectory=_new_traj_fixture(t_last=1.5),
        observables=["energy_static_trotter", "doublon_trotter"],
        tolerance=1e-10,
    )
    assert bool(res["time_grid_match"]) is False
    assert bool(res["passed_all"]) is False
    assert "time-grid mismatch" in str(res["reason"])


def test_legacy_comparison_plot_helper_smoke(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    payload = {"trajectory": _new_traj_fixture()}
    plot_path = tmp_path / "legacy_cmp.png"
    _write_legacy_comparison_plot(
        plot_path=plot_path,
        payload=payload,
        legacy_ref=_legacy_ref_fixture(),
    )
    assert plot_path.exists()
