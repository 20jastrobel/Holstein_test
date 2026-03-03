from __future__ import annotations

import pytest

from pipelines.exact_bench.hh_noise_hardware_validation import (
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
