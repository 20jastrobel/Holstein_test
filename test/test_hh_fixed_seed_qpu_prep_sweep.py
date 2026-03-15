from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.exact_bench.hh_fixed_seed_qpu_prep_sweep as wf


def _seed_json(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "settings": {
                    "L": 2,
                    "t": 1.0,
                    "u": 4.0,
                    "dv": 0.0,
                    "omega0": 1.0,
                    "g_ep": 0.5,
                    "n_ph_max": 2,
                    "boson_encoding": "binary",
                    "ordering": "blocked",
                    "boundary": "open",
                    "adapt_continuation_mode": "phase3_v1",
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_parse_args_defaults_to_fake_backend_mode(tmp_path: Path) -> None:
    cfg = wf.parse_args(
        [
            "--fixed-final-state-json",
            str(_seed_json(tmp_path / "seed.json")),
        ]
    )

    assert cfg.fixed_final_state_json == tmp_path / "seed.json"
    assert cfg.use_fake_backend is True
    assert cfg.backend_name == "FakeGuadalupeV2"
    assert cfg.budget_mode == "full_trajectory"
    assert cfg.suzuki_steps == (16, 32, 48, 64, 96, 128)
    assert cfg.cfqm_steps == (8, 16, 24, 32, 48, 64)


def test_parse_args_allows_empty_cfqm_grid(tmp_path: Path) -> None:
    cfg = wf.parse_args(
        [
            "--fixed-final-state-json",
            str(_seed_json(tmp_path / "seed.json")),
            "--cfqm-steps",
            "",
        ]
    )

    assert cfg.cfqm_steps == ()
    assert cfg.suzuki_steps


def test_run_sweep_writes_summary_artifacts(monkeypatch, tmp_path: Path) -> None:
    seed_json = _seed_json(tmp_path / "seed.json")
    summary_pdf = tmp_path / "summary.pdf"
    cfg = wf.parse_args(
        [
            "--fixed-final-state-json",
            str(seed_json),
            "--output-json",
            str(tmp_path / "summary.json"),
            "--output-csv",
            str(tmp_path / "summary.csv"),
            "--output-pdf",
            str(summary_pdf),
            "--run-root",
            str(tmp_path / "runs"),
            "--suzuki-steps",
            "16",
            "--cfqm-steps",
            "8",
        ]
    )

    monkeypatch.setattr(wf, "resolve_staged_hh_config", lambda args: args)

    def _fake_run_staged_hh_noiseless(staged_cfg, *, run_command=None):
        method = str(staged_cfg.noiseless_methods).split(",")[0]
        steps = int(staged_cfg.trotter_steps)
        Path(staged_cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(staged_cfg.output_json).write_text("{}", encoding="utf-8")
        return {
            "dynamics_noiseless": {
                "profiles": {
                    "drive": {
                        "methods": {
                            method: {
                                "trajectory": [
                                    {
                                        "time": 0.0,
                                        "fidelity": 1.0,
                                        "energy_total_trotter": -1.0,
                                        "energy_total_exact": -1.0,
                                    },
                                    {
                                        "time": 1.0,
                                        "fidelity": 0.99 if method == "suzuki2" else 0.985,
                                        "energy_total_trotter": -1.0 + 0.001 * steps,
                                        "energy_total_exact": -1.0,
                                    },
                                ],
                                "final": {
                                    "fidelity": 0.99 if method == "suzuki2" else 0.985,
                                    "abs_energy_total_error": 0.001 * steps,
                                    "abs_energy_error_vs_ground_state": 0.001 * steps,
                                },
                            }
                        }
                    }
                }
            },
            "circuit_metrics": {
                "transpile_target": {"backend_name": str(staged_cfg.circuit_backend_name)},
                "dynamics": {
                    method: {
                        "metadata": {
                            "proxy_total": {
                                "cx_proxy_total": 10 * steps,
                                "depth_proxy_total": 5 * steps,
                            },
                            "trajectory_circuit_metrics": {
                                "dynamics_only": {
                                    "transpiled": {
                                        "count_2q": steps,
                                        "cx_count": steps,
                                        "depth": 2 * steps,
                                    }
                                },
                                "prep_plus_dynamics": {
                                    "skipped": True,
                                    "reason": "no_replay_prep",
                                },
                            },
                        }
                    }
                },
            },
        }

    monkeypatch.setattr(wf, "run_staged_hh_noiseless", _fake_run_staged_hh_noiseless)
    monkeypatch.setattr(
        wf,
        "_write_summary_pdf",
        lambda cfg, **kwargs: Path(cfg.output_pdf).write_text("pdf stub", encoding="utf-8"),
    )

    payload = wf.run_sweep(cfg, run_command="python sweep.py")

    assert cfg.output_json.exists()
    assert cfg.output_csv.exists()
    assert summary_pdf.exists()
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["transpile_backend"] == "FakeGuadalupeV2"
    assert payload["rows"][0]["budget_mode"] == "full_trajectory"
    assert payload["rows"][0]["budget_cx_count"] == payload["rows"][0]["dynamics_only_cx_count"]
    assert payload["pareto_shortlist"]

    with cfg.output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["method"] for row in rows} == {"suzuki2", "cfqm4"}


def test_run_sweep_supports_suzuki_only(monkeypatch, tmp_path: Path) -> None:
    seed_json = _seed_json(tmp_path / "seed.json")
    cfg = wf.parse_args(
        [
            "--fixed-final-state-json",
            str(seed_json),
            "--output-json",
            str(tmp_path / "summary.json"),
            "--output-csv",
            str(tmp_path / "summary.csv"),
            "--output-pdf",
            str(tmp_path / "summary.pdf"),
            "--run-root",
            str(tmp_path / "runs"),
            "--suzuki-steps",
            "64",
            "--cfqm-steps",
            "",
        ]
    )

    monkeypatch.setattr(wf, "resolve_staged_hh_config", lambda args: args)
    monkeypatch.setattr(
        wf,
        "run_staged_hh_noiseless",
        lambda staged_cfg, **kwargs: {
            "dynamics_noiseless": {
                "profiles": {
                    "drive": {
                        "methods": {
                            "suzuki2": {
                                "trajectory": [
                                    {"time": 0.0, "fidelity": 1.0, "energy_total_trotter": -1.0, "energy_total_exact": -1.0},
                                    {"time": 1.0, "fidelity": 0.999, "energy_total_trotter": -0.99, "energy_total_exact": -1.0},
                                ],
                                "final": {
                                    "fidelity": 0.999,
                                    "abs_energy_total_error": 0.01,
                                    "abs_energy_error_vs_ground_state": 0.01,
                                },
                            }
                        }
                    }
                }
            },
            "circuit_metrics": {
                "transpile_target": {"backend_name": "FakeGuadalupeV2"},
                "dynamics": {
                    "suzuki2": {
                        "metadata": {
                            "proxy_total": {"cx_proxy_total": 10, "depth_proxy_total": 5},
                            "trajectory_circuit_metrics": {
                                "dynamics_only": {"transpiled": {"count_2q": 10, "cx_count": 10, "depth": 20}},
                                "prep_plus_dynamics": {"skipped": True, "reason": "no_replay_prep"},
                            },
                        }
                    }
                },
            },
        },
    )
    monkeypatch.setattr(
        wf,
        "_write_summary_pdf",
        lambda cfg, **kwargs: Path(cfg.output_pdf).write_text("pdf stub", encoding="utf-8"),
    )

    payload = wf.run_sweep(cfg, run_command="python sweep.py")

    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["method"] == "suzuki2"


def test_run_sweep_snapshot_budget_uses_snapshot_metrics(monkeypatch, tmp_path: Path) -> None:
    seed_json = _seed_json(tmp_path / "seed.json")
    cfg = wf.parse_args(
        [
            "--fixed-final-state-json",
            str(seed_json),
            "--output-json",
            str(tmp_path / "summary.json"),
            "--output-csv",
            str(tmp_path / "summary.csv"),
            "--output-pdf",
            str(tmp_path / "summary.pdf"),
            "--run-root",
            str(tmp_path / "runs"),
            "--suzuki-steps",
            "2",
            "--cfqm-steps",
            "",
            "--budget-mode",
            "snapshot",
        ]
    )

    monkeypatch.setattr(wf, "resolve_staged_hh_config", lambda args: args)
    monkeypatch.setattr(
        wf,
        "_build_snapshot_budget_context",
        lambda cfg, *, seed_settings: {"stub": True},
    )
    monkeypatch.setattr(
        wf,
        "_snapshot_budget_details",
        lambda **kwargs: {
            "transpile_rows": [
                {"time_index": 0, "time": 0.0, "count_2q": 0, "cx_count": 0, "depth": 0, "size": 0},
                {"time_index": 1, "time": 1.0, "count_2q": 12, "cx_count": 12, "depth": 24, "size": 30},
            ],
            "max": {"time_index": 1, "time": 1.0, "count_2q": 12, "cx_count": 12, "depth": 24, "size": 30},
            "final": {"time_index": 1, "time": 1.0, "count_2q": 12, "cx_count": 12, "depth": 24, "size": 30},
        },
    )

    monkeypatch.setattr(
        wf,
        "run_staged_hh_noiseless",
        lambda staged_cfg, **kwargs: {
            "dynamics_noiseless": {
                "profiles": {
                    "drive": {
                        "methods": {
                            "suzuki2": {
                                "trajectory": [
                                    {"time": 0.0, "fidelity": 1.0, "energy_total_trotter": -1.0, "energy_total_exact": -1.0},
                                    {"time": 1.0, "fidelity": 0.99, "energy_total_trotter": -0.99, "energy_total_exact": -1.0},
                                ],
                                "final": {
                                    "fidelity": 0.99,
                                    "abs_energy_total_error": 0.01,
                                    "abs_energy_error_vs_ground_state": 0.01,
                                },
                            }
                        }
                    }
                }
            },
            "circuit_metrics": {
                "transpile_target": {"backend_name": "FakeGuadalupeV2"},
                "dynamics": {
                    "suzuki2": {
                        "metadata": {
                            "proxy_total": {"cx_proxy_total": 20, "depth_proxy_total": 10},
                            "trajectory_circuit_metrics": {
                                "dynamics_only": {"transpiled": {"count_2q": 20, "cx_count": 20, "depth": 40}},
                                "prep_plus_dynamics": {"skipped": True, "reason": "no_replay_prep"},
                            },
                        }
                    }
                },
            },
        },
    )
    monkeypatch.setattr(
        wf,
        "_write_summary_pdf",
        lambda cfg, **kwargs: Path(cfg.output_pdf).write_text("pdf stub", encoding="utf-8"),
    )

    payload = wf.run_sweep(cfg, run_command="python sweep.py")

    row = payload["rows"][0]
    assert row["budget_mode"] == "snapshot"
    assert row["budget_source"] == "snapshot_max_dynamics_only"
    assert row["budget_count_2q"] == 12
    assert row["budget_cx_count"] == 12
    assert row["budget_depth"] == 24
    assert row["dynamics_only_cx_count"] == 20
    assert Path(row["snapshot_metrics_json"]).exists()
