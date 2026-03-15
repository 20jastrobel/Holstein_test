from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.exact_bench.hh_fixed_seed_budgeted_projected_dynamics as wf
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _basis(dim: int, idx: int) -> np.ndarray:
    out = np.zeros(dim, dtype=complex)
    out[int(idx)] = 1.0
    return out


def _poly(label: str, *, coeff: complex = 1.0) -> PauliPolynomial:
    return PauliPolynomial("JW", [PauliTerm(2, ps=str(label), pc=complex(coeff))])


def _term(label: str) -> AnsatzTerm:
    return AnsatzTerm(label=f"term_{label}", polynomial=_poly(label))


def test_run_sweep_writes_summary_outputs(monkeypatch, tmp_path: Path) -> None:
    fake_replay = {
        "payload": {
            "settings": {
                "L": 1,
                "t": 1.0,
                "u": 2.0,
                "dv": 0.0,
                "omega0": 1.0,
                "g_ep": 0.5,
                "n_ph_max": 1,
                "boson_encoding": "binary",
                "ordering": "blocked",
                "boundary": "open",
            }
        },
        "initial_state": _basis(4, 0),
        "h_poly": _poly("ze"),
        "family_info": {"resolved": "full_meta", "resolution_source": "fallback_family_missing_labels"},
        "family_resolved": "full_meta",
        "replay_terms": [_term("xe"), _term("ex")],
        "pool_meta": {"selection_mode": "sparse_label_lookup"},
        "nq": 2,
    }

    def _fake_transpile_theta_history(*, terms, theta_history, num_qubits, cfg):
        row = {
            "time_index": 0,
            "count_2q": 10 * int(len(terms)),
            "count_1q": 20 * int(len(terms)),
            "depth": 15 * int(len(terms)),
            "size": 30 * int(len(terms)),
        }
        rows = [dict(row) for _ in range(int(np.asarray(theta_history).shape[0]))]
        return rows, dict(row), dict(row)

    monkeypatch.setattr(wf, "build_replay_sequence_from_input_json", lambda path: fake_replay)
    monkeypatch.setattr(wf, "_transpile_theta_history", _fake_transpile_theta_history)
    monkeypatch.setattr(wf, "write_summary_pdf", lambda **kwargs: None)

    cfg = wf.SweepConfig(
        fixed_seed_json=tmp_path / "seed.json",
        output_json=tmp_path / "out.json",
        output_csv=tmp_path / "out.csv",
        output_pdf=tmp_path / "out.pdf",
        run_root=tmp_path / "runs",
        tag="budgeted_test",
        backend_name="FakeGuadalupeV2",
        use_fake_backend=True,
        circuit_optimization_level=3,
        circuit_seed_transpiler=11,
        max_cx_budget=300,
        t_final=0.25,
        num_times=5,
        reference_steps=16,
        ode_substeps=2,
        tangent_eps=1e-6,
        lambda_reg=1e-8,
        svd_rcond=1e-12,
        drive_A=1.0,
        drive_omega=1.0,
        drive_tbar=5.0,
        drive_phi=0.0,
        drive_pattern="staggered",
        drive_t0=0.0,
        drive_time_sampling="midpoint",
        prefix_limit=2,
        representative_prefixes=(1, 2),
        skip_pdf=True,
    )

    payload = wf.run_sweep(cfg)

    assert Path(cfg.output_json).exists()
    assert Path(cfg.output_csv).exists()
    assert len(payload["rows"]) == 2
    assert payload["summary"]["feasible_count"] == 2
    assert payload["summary"]["best_by_cx_under_budget"]["prefix_k"] == 1
    assert payload["summary"]["best_by_error_under_budget"]["prefix_k"] in {1, 2}
    assert payload["representative_prefixes"] == [1, 2]

    candidate_payload = json.loads((cfg.run_root / "K01" / "candidate.json").read_text(encoding="utf-8"))
    assert candidate_payload["row"]["prefix_k"] == 1
    assert len(candidate_payload["trajectory"]) == 5
