from __future__ import annotations

import pytest

from pipelines.exact_bench.cross_check_suite import parse_args


def test_hh_cross_check_rejects_non_spsa_method() -> None:
    with pytest.raises(ValueError, match="HH cross-check is SPSA-only for --vqe-method"):
        parse_args(["--L", "2", "--problem", "hh", "--vqe-method", "COBYLA"])


def test_hh_cross_check_accepts_spsa_method() -> None:
    args = parse_args(["--L", "2", "--problem", "hh", "--vqe-method", "SPSA"])
    assert str(args.vqe_method) == "SPSA"


def test_legacy_hubbard_cross_check_still_accepts_cobyla() -> None:
    args = parse_args(["--L", "2", "--problem", "hubbard", "--vqe-method", "COBYLA"])
    assert str(args.vqe_method) == "COBYLA"
