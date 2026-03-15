from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipelines.exact_bench.cross_check_suite as cross_check_mod
from pipelines.exact_bench.cross_check_suite import parse_args


def test_hh_cross_check_rejects_non_spsa_method() -> None:
    with pytest.raises(ValueError, match="HH cross-check is SPSA-only for --vqe-method"):
        parse_args(["--L", "2", "--problem", "hh", "--vqe-method", "COBYLA"])


def test_hh_cross_check_accepts_spsa_method() -> None:
    args = parse_args(["--L", "2", "--problem", "hh", "--vqe-method", "SPSA"])
    assert str(args.vqe_method) == "SPSA"


def test_hh_cross_check_accepts_seed_refine_surface_flag() -> None:
    args = parse_args(["--L", "2", "--problem", "hh", "--hh-seed-refine-surface"])
    assert bool(args.hh_seed_refine_surface) is True


def test_hh_seed_benchmark_preset_does_not_require_L() -> None:
    args = parse_args(["--problem", "hh", "--hh-seed-benchmark-preset", "mini4"])
    assert args.L is None
    assert str(args.hh_seed_benchmark_preset) == "mini4"


def test_hh_seed_surface_rejects_non_hh_problem() -> None:
    with pytest.raises(ValueError, match="HH-only"):
        parse_args(["--L", "2", "--problem", "hubbard", "--hh-seed-refine-surface"])


def test_hh_seed_benchmark_preset_runs_all_points(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[int, float, bool, str]] = []

    def _fake_run_cross_check(args):
        point_id = f"L{int(args.L)}_g{float(args.g_ep):.1f}"
        calls.append((int(args.L), float(args.g_ep), bool(args.hh_seed_refine_surface), str(args.output_dir)))
        return {
            "trials": [
                {
                    "run_id": "HH-PhysicalTermwise",
                    "method_id": "HH-PhysicalTermwise",
                    "method_kind": "conventional_vqe",
                    "ansatz_name": "HH-PhysicalTermwise",
                    "pool_name": "hh_hva_ptw",
                    "delta_E_abs": 0.20,
                    "seed_surface_role": "warm_only",
                    "seed_surface_warm_ansatz": "hh_hva_ptw",
                    "benchmark_point_id": point_id,
                    "cx_proxy": 20,
                    "sq_proxy": 10,
                    "depth_proxy": 40,
                },
                {
                    "run_id": "HH-PhysicalTermwise + uccsd_otimes_paop_lf_std",
                    "method_id": "HH-PhysicalTermwise + uccsd_otimes_paop_lf_std",
                    "method_kind": "conventional_vqe",
                    "ansatz_name": "HH-PhysicalTermwise + uccsd_otimes_paop_lf_std",
                    "pool_name": "uccsd_otimes_paop_lf_std",
                    "delta_E_abs": 0.10,
                    "seed_surface_role": "product_family",
                    "seed_surface_warm_ansatz": "hh_hva_ptw",
                    "seed_surface_refine_family": "uccsd_otimes_paop_lf_std",
                    "benchmark_point_id": point_id,
                    "cx_proxy": 30,
                    "sq_proxy": 12,
                    "depth_proxy": 50,
                },
            ],
            "seed_surface": {
                "comparisons": [
                    {
                        "warm_ansatz": "hh_hva_ptw",
                        "refine_family": "uccsd_otimes_paop_lf_std",
                        "improvement_per_added_cx_proxy": 0.01,
                    }
                ]
            },
        }

    def _fake_write_proxy_sidecars(rows, output_dir, **kwargs):
        del rows, kwargs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "csv": output_dir / "metrics_proxy_runs.csv",
            "jsonl": output_dir / "metrics_proxy_runs.jsonl",
            "summary_json": output_dir / "metrics_proxy_summary.json",
        }
        paths["csv"].write_text("csv\n", encoding="utf-8")
        paths["jsonl"].write_text("{}\n", encoding="utf-8")
        paths["summary_json"].write_text(json.dumps({"ok": True}), encoding="utf-8")
        return paths

    monkeypatch.setattr(cross_check_mod, "run_cross_check", _fake_run_cross_check)
    monkeypatch.setattr(cross_check_mod, "write_proxy_sidecars", _fake_write_proxy_sidecars)

    args = parse_args(
        [
            "--problem",
            "hh",
            "--hh-seed-benchmark-preset",
            "mini4",
            "--output-dir",
            str(tmp_path),
        ]
    )
    payload = cross_check_mod.run_cross_check_preset(args)

    assert len(calls) == 4
    assert all(flag is True for _L, _g, flag, _out in calls)
    assert payload["decision_metrics"]["pre_adapt_best_family_by_median_cx_proxy"] == "uccsd_otimes_paop_lf_std"
    assert Path(payload["artifacts"]["summary_json"]).exists()


def test_legacy_hubbard_cross_check_still_accepts_cobyla() -> None:
    args = parse_args(["--L", "2", "--problem", "hubbard", "--vqe-method", "COBYLA"])
    assert str(args.vqe_method) == "COBYLA"
