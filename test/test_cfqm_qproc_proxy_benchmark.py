from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.cfqm_vs_suzuki_qproc_proxy_benchmark import (
    BenchmarkConfig,
    _build_cost_matched_pairs,
    _build_stage_map_for_proxy,
    _compute_cfqm_proxy_cost,
    _compute_sweep_proxy_cost,
    _cx_proxy_term,
    _pauli_weight,
    _pauli_xy_count,
    _sq_proxy_term,
    run_benchmark,
)


def _mk_payload(
    *,
    method: str,
    steps: int,
    num_times: int,
    final_energy: float,
    drive_enabled: bool,
) -> dict:
    static_terms = [
        {
            "label_exyz": "ze",
            "coeff": {"re": 1.0, "im": 0.0},
        },
        {
            "label_exyz": "ez",
            "coeff": {"re": 0.5, "im": 0.0},
        },
    ]

    traj = []
    for i in range(int(num_times)):
        frac = 0.0 if int(num_times) <= 1 else float(i) / float(int(num_times) - 1)
        traj.append({"time": float(frac), "energy_total_trotter": float(frac * final_energy)})

    settings = {
        "L": 2,
        "problem": "hubbard",
        "ordering": "blocked",
        "t_final": 1.0,
        "num_times": int(num_times),
        "trotter_steps": int(steps),
    }
    if method != "suzuki2":
        settings["propagator"] = str(method)
    if drive_enabled:
        settings["drive"] = {
            "enabled": True,
            "A": 0.2,
            "omega": 1.0,
            "tbar": 1.0,
            "phi": 0.0,
            "t0": 0.0,
            "pattern": "staggered",
            "custom_s": None,
            "include_identity": False,
            "time_sampling": "midpoint",
        }

    return {
        "settings": settings,
        "hamiltonian": {"coefficients_exyz": static_terms},
        "trajectory": traj,
    }


def test_pauli_proxy_primitives() -> None:
    assert _pauli_weight("eeee") == 0
    assert _pauli_weight("xeyz") == 3
    assert _pauli_xy_count("xeyz") == 2
    assert _cx_proxy_term("eeee") == 0
    assert _cx_proxy_term("xeee") == 0
    assert _cx_proxy_term("xyez") == 4
    assert _sq_proxy_term("eeee") == 1
    assert _sq_proxy_term("xyez") == 5


def test_sweep_proxy_arithmetic() -> None:
    cost = _compute_sweep_proxy_cost(["z", "x", "xy"])
    assert cost.term_exp_count == 6
    # cx per term: z->0, x->0, xy->2 => sum 2, sweep doubles => 4
    assert cost.cx_proxy == 4
    # sq per term: z->1, x->3, xy->5 => sum 9, sweep doubles => 18
    assert cost.sq_proxy == 18


def test_stage_map_unknown_label_ignore_and_zero_insertion_guard() -> None:
    stage = _build_stage_map_for_proxy(
        ordered_labels=["z"],
        static_coeff_map={"z": 0.0 + 0.0j},
        drive_maps=[{"unknown": 0.75 + 0.0j, "z": 0.0 + 0.0j}],
        a_row=[1.0],
        s_static=0.0,
        coeff_drop_abs_tol=0.0,
    )
    assert stage == {}


def test_cfqm_proxy_cost_deterministic() -> None:
    def drive(t: float) -> dict[str, float]:
        return {"z": 0.1 * t}

    kwargs = {
        "method": "cfqm4",
        "T": 1.0,
        "n_steps": 8,
        "t0": 0.0,
        "static_coeff_map": {"z": 0.7 + 0.0j},
        "drive_provider": drive,
        "ordered_labels": ["z"],
        "active_coeff_tol": 1e-14,
        "coeff_drop_abs_tol": 0.0,
    }
    c1 = _compute_cfqm_proxy_cost(**kwargs)
    c2 = _compute_cfqm_proxy_cost(**kwargs)
    assert c1 == c2


def test_cost_match_pairs_exact_tie() -> None:
    rows = [
        {
            "method": "suzuki2",
            "trotter_steps": 16,
            "cx_proxy_total": 100,
            "final_abs_energy_error": 0.10,
            "run_runtime_s": 0.01,
        },
        {
            "method": "cfqm4",
            "trotter_steps": 16,
            "cx_proxy_total": 100,
            "final_abs_energy_error": 0.08,
            "run_runtime_s": 0.02,
        },
        {
            "method": "cfqm6",
            "trotter_steps": 16,
            "cx_proxy_total": 150,
            "final_abs_energy_error": 0.05,
            "run_runtime_s": 0.03,
        },
    ]
    pairs = _build_cost_matched_pairs(rows, metric="cx_proxy_total", tolerance=0.0)
    assert len(pairs) == 2
    tied = {float(p["target_metric"]) for p in pairs}
    assert tied == {100.0, 150.0}
    first_100 = next(p for p in pairs if float(p["target_metric"]) == 100.0)
    tied_rows = [m for m in first_100["matched_rows"] if isinstance(m, dict) and m["method"] in {"suzuki2", "cfqm4"}]
    assert all(bool(r["exact_match"]) for r in tied_rows)
    assert all(float(r["metric_delta"]) == 0.0 for r in tied_rows)


def test_cost_match_pairs_missing_tie_with_tolerance() -> None:
    rows = [
        {
            "method": "suzuki2",
            "trotter_steps": 16,
            "cx_proxy_total": 100,
            "final_abs_energy_error": 0.20,
            "run_runtime_s": 0.02,
        },
        {
            "method": "cfqm4",
            "trotter_steps": 16,
            "cx_proxy_total": 130,
            "final_abs_energy_error": 0.10,
            "run_runtime_s": 0.03,
        },
    ]
    pairs = _build_cost_matched_pairs(rows, metric="cx_proxy_total", tolerance=10.0)
    assert len(pairs) == 2
    target_100 = next(p for p in pairs if float(p["target_metric"]) == 100.0)
    cfqm4_row = next(m for m in target_100["matched_rows"] if isinstance(m, dict) and m["method"] == "cfqm4")
    assert float(cfqm4_row["matched_metric"]) == 130.0
    assert cfqm4_row["exact_match"] is False


def test_run_benchmark_smoke_with_mock_runner(tmp_path: Path) -> None:
    runs_json = tmp_path / "runs.json"
    runs_csv = tmp_path / "runs.csv"

    cfg = BenchmarkConfig(
        problem="hubbard",
        L=2,
        t=1.0,
        u=4.0,
        dv=0.0,
        boundary="periodic",
        ordering="blocked",
        t_final=1.0,
        num_times=5,
        methods=("suzuki2", "cfqm4", "cfqm6"),
        steps_grid=(8, 16),
        reference_steps=64,
        active_coeff_tol=1e-14,
        drive_enabled=False,
        drive_A=0.2,
        drive_omega=1.0,
        drive_tbar=1.0,
        drive_phi=0.0,
        drive_t0=0.0,
        drive_pattern="staggered",
        drive_custom_s=None,
        drive_include_identity=False,
        drive_time_sampling="midpoint",
        initial_state_source="exact",
        vqe_ansatz="uccsd",
        vqe_reps=1,
        vqe_restarts=1,
        vqe_maxiter=1,
        vqe_method="COBYLA",
        adapt_pool="uccsd",
        adapt_max_depth=1,
        adapt_maxiter=1,
        calibrate_transpile=False,
        compare_policy="sweep_only",
        cost_match_metric="cx_proxy_total",
        cost_match_tolerance=0.0,
        output_json=runs_json,
        output_csv=runs_csv,
        output_pdf=tmp_path / "plots.pdf",
        output_summary=tmp_path / "summary.md",
        skip_pdf=True,
    )

    def fake_runner(cmd: list[str], _cwd: Path) -> dict[str, object]:
        prop = "suzuki2"
        steps = 64
        for i, tok in enumerate(cmd):
            if tok == "--propagator":
                prop = str(cmd[i + 1])
            if tok == "--trotter-steps":
                steps = int(cmd[i + 1])
        if prop == "piecewise_exact":
            payload = _mk_payload(
                method="piecewise_exact",
                steps=steps,
                num_times=cfg.num_times,
                final_energy=0.0,
                drive_enabled=False,
            )
        else:
            if prop == "suzuki2":
                err = 1.0 / float(steps)
            elif prop == "cfqm4":
                err = 1.0 / float(steps**2)
            else:
                err = 1.0 / float(steps**3)
            payload = _mk_payload(
                method=prop,
                steps=steps,
                num_times=cfg.num_times,
                final_energy=err,
                drive_enabled=False,
            )
        payload["_run_runtime_s"] = 0.01
        return payload

    res = run_benchmark(cfg, run_pipeline=fake_runner)
    out = res["payload"]

    assert runs_json.exists()
    assert runs_csv.exists()
    assert Path(res["summary_json"]).exists()

    assert out["schema"] == "cfqm_qproc_proxy_v1"
    assert len(out["runs"]) == 6
    assert len(out["summary"]["pareto_front"]) >= 1

    parsed_json = json.loads(runs_json.read_text(encoding="utf-8"))
    assert parsed_json["summary"]["pareto_front"]


def test_cost_match_summary_written(tmp_path: Path) -> None:
    runs_json = tmp_path / "runs_cost_match.json"
    runs_csv = tmp_path / "runs_cost_match.csv"

    cfg = BenchmarkConfig(
        problem="hubbard",
        L=2,
        t=1.0,
        u=4.0,
        dv=0.0,
        boundary="periodic",
        ordering="blocked",
        t_final=1.0,
        num_times=5,
        methods=("suzuki2", "cfqm4"),
        steps_grid=(8, 16),
        reference_steps=32,
        active_coeff_tol=1e-14,
        drive_enabled=False,
        drive_A=0.2,
        drive_omega=1.0,
        drive_tbar=1.0,
        drive_phi=0.0,
        drive_t0=0.0,
        drive_pattern="staggered",
        drive_custom_s=None,
        drive_include_identity=False,
        drive_time_sampling="midpoint",
        initial_state_source="exact",
        vqe_ansatz="uccsd",
        vqe_reps=1,
        vqe_restarts=1,
        vqe_maxiter=1,
        vqe_method="COBYLA",
        adapt_pool="uccsd",
        adapt_max_depth=1,
        adapt_maxiter=1,
        calibrate_transpile=False,
        compare_policy="cost_match",
        cost_match_metric="cx_proxy_total",
        cost_match_tolerance=0.0,
        output_json=runs_json,
        output_csv=runs_csv,
        output_pdf=tmp_path / "plots_cost_match.pdf",
        output_summary=tmp_path / "summary_cost_match.md",
        skip_pdf=True,
    )

    def fake_runner(cmd: list[str], _cwd: Path) -> dict[str, object]:
        prop = "suzuki2"
        steps = 8
        for i, tok in enumerate(cmd):
            if tok == "--propagator":
                prop = str(cmd[i + 1])
            if tok == "--trotter-steps":
                steps = int(cmd[i + 1])
        if prop == "piecewise_exact":
            payload = _mk_payload(
                method="piecewise_exact",
                steps=steps,
                num_times=cfg.num_times,
                final_energy=0.0,
                drive_enabled=False,
            )
        else:
            payload = _mk_payload(
                method=prop,
                steps=steps,
                num_times=cfg.num_times,
                final_energy=0.02,
                drive_enabled=False,
            )
        payload["_run_runtime_s"] = 0.01
        return payload

    out = run_benchmark(cfg, run_pipeline=fake_runner)["payload"]
    cost_match = out["summary"]["cost_match"]
    assert cost_match["enabled"] is True
    assert cost_match["metric"] == "cx_proxy_total"
    assert isinstance(cost_match["pairs"], list)
    assert cost_match["pairs"]
