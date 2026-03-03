from __future__ import annotations

import pytest

from pipelines.exact_bench.hh_noise_hardware_validation import (
    _apply_defaults_and_minimums,
    parse_args,
)


def test_hh_defaults_applied_from_minimum_table_l2_nph1() -> None:
    args = parse_args(["--L", "2"])
    args = _apply_defaults_and_minimums(args)

    assert int(args.vqe_reps) == 2
    assert int(args.vqe_restarts) == 3
    assert int(args.vqe_maxiter) == 800
    assert int(args.trotter_steps) == 64


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
            "--no-allow-aer-fallback",
            "--no-omp-shm-workaround",
        ]
    )
    assert bool(args.allow_aer_fallback) is False
    assert bool(args.omp_shm_workaround) is False
