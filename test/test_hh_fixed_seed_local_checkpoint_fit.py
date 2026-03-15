from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.exact_bench.hh_fixed_seed_local_checkpoint_fit as suite
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.time_propagation.local_checkpoint_fit import (
    LocalPauliAnsatzSpec,
    build_local_pauli_ansatz_terms,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor


def test_run_sweep_writes_checkpoint_fit_artifacts(tmp_path: Path, monkeypatch) -> None:
    psi_ref = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=complex)
    terms = build_local_pauli_ansatz_terms(
        LocalPauliAnsatzSpec(num_qubits=2, reps=1, single_axes=("y",), entangler_axes=())
    )
    executor = CompiledAnsatzExecutor(list(terms))
    psi_t1 = executor.prepare_state(np.asarray([0.15, -0.05], dtype=float), psi_ref)
    fake_h_poly = PauliPolynomial("JW", [PauliTerm(2, ps="ee", pc=0.0)])

    def _fake_seed_loader(_path: Path) -> dict[str, object]:
        return {
            "payload": {
                "settings": {
                    "t": 1.0,
                    "u": 4.0,
                    "dv": 0.0,
                    "g_ep": 0.5,
                    "omega0": 1.0,
                    "n_ph_max": 0,
                    "L": 1,
                    "ordering": "blocked",
                }
            },
            "initial_state": psi_ref,
            "h_poly": fake_h_poly,
            "replay_terms": (),
            "family_info": {"resolution_source": "test"},
            "family_resolved": "test",
            "pool_meta": {},
            "nq": 2,
        }

    def _fake_drive_provider(**_kwargs):
        return (lambda _time: {}), {"A": 0.0, "omega": 0.0, "tbar": 1.0, "phi": 0.0, "pattern": "test"}

    def _fake_exact_reference(*_args, **_kwargs):
        return SimpleNamespace(
            times=np.asarray([0.0, 0.2], dtype=float),
            states=(np.asarray(psi_ref, dtype=complex), np.asarray(psi_t1, dtype=complex)),
            energies_total=np.asarray([0.0, 0.0], dtype=float),
            trajectory_rows=(
                {"time": 0.0, "energy_total": 0.0},
                {"time": 0.2, "energy_total": 0.0},
            ),
        )

    monkeypatch.setattr(suite, "build_replay_sequence_from_input_json", _fake_seed_loader)
    monkeypatch.setattr(suite, "_build_drive_provider", _fake_drive_provider)
    monkeypatch.setattr(suite, "run_exact_driven_reference", _fake_exact_reference)
    monkeypatch.setattr(suite, "write_summary_pdf", lambda **_kwargs: None)

    cfg = suite.SweepConfig(
        fixed_seed_json=tmp_path / "seed.json",
        output_json=tmp_path / "summary.json",
        output_csv=tmp_path / "summary.csv",
        output_pdf=tmp_path / "summary.pdf",
        run_root=tmp_path / "runs",
        tag="test",
        backend_name="FakeGuadalupeV2",
        use_fake_backend=True,
        circuit_optimization_level=3,
        circuit_seed_transpiler=11,
        max_cx_budget=100,
        t_final=0.2,
        num_times=2,
        reference_steps=4,
        single_axes=("y",),
        entangler_axes=(),
        reps_list=(1, 2),
        optimizer_method="L-BFGS-B",
        optimizer_maxiter=30,
        optimizer_gtol=1e-8,
        optimizer_ftol=1e-12,
        angle_bound=float(np.pi),
        param_shift=float(np.pi / 2.0),
        drive_A=0.0,
        drive_omega=0.0,
        drive_tbar=1.0,
        drive_phi=0.0,
        drive_pattern="staggered",
        drive_t0=0.0,
        drive_time_sampling="midpoint",
        skip_pdf=True,
    )

    payload = suite.run_sweep(cfg)

    assert cfg.output_json.exists()
    assert cfg.output_csv.exists()
    assert len(payload["rows"]) == 2
    assert int(payload["summary"]["feasible_count"]) == 2
    assert (cfg.run_root / "local_y_none_reps1" / "candidate.json").exists()
