from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.hardcoded.hh_staged_circuit_report as report_cli
import pipelines.hardcoded.hh_staged_workflow as wf
from pipelines.hardcoded.hh_staged_noiseless import parse_args as parse_staged_args
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _basis(dim: int, idx: int) -> np.ndarray:
    out = np.zeros(dim, dtype=complex)
    out[int(idx)] = 1.0
    return out


def _simple_term() -> AnsatzTerm:
    poly = PauliPolynomial(2)
    poly.add_term(PauliTerm(2, ps="ze", pc=1.0))
    return AnsatzTerm(label="z_drive_like", polynomial=poly)


def _simple_ansatz(term: AnsatzTerm) -> SimpleNamespace:
    return SimpleNamespace(base_terms=[term], reps=1, num_parameters=1)


def _simple_stage_result() -> wf.StageExecutionResult:
    psi0 = _basis(4, 0)
    psi1 = _basis(4, 1)
    term = _simple_term()
    ansatz = _simple_ansatz(term)
    return wf.StageExecutionResult(
        h_poly=object(),
        hmat=np.eye(4, dtype=complex),
        ordered_labels_exyz=["ze"],
        coeff_map_exyz={"ze": 1.0 + 0.0j},
        nq_total=2,
        psi_hf=np.array(psi0, copy=True),
        psi_warm=np.array(psi1, copy=True),
        psi_adapt=np.array(psi1, copy=True),
        psi_final=np.array(psi1, copy=True),
        warm_payload={
            "energy": -1.0,
            "exact_filtered_energy": -1.1,
            "optimal_point": [0.1],
            "ansatz": "hh_hva_ptw",
        },
        adapt_payload={
            "energy": -1.05,
            "exact_gs_energy": -1.1,
            "ansatz_depth": 1,
            "pool_type": "paop_lf_std",
            "continuation_mode": "phase1_v1",
            "stop_reason": "eps_grad",
        },
        replay_payload={
            "generator_family": {"requested": "match_adapt", "resolved": "paop_lf_std"},
            "vqe": {"energy": -1.09, "stop_reason": "converged"},
            "exact": {"E_exact_sector": -1.1},
        },
        warm_circuit_context={
            "ansatz": ansatz,
            "theta": np.array([0.1], dtype=float),
            "reference_state": np.array(psi0, copy=True),
            "num_qubits": 2,
            "ansatz_name": "hh_hva_ptw",
        },
        adapt_circuit_context={
            "selected_ops": [term],
            "theta": np.array([0.2], dtype=float),
            "reference_state": np.array(psi0, copy=True),
            "num_qubits": 2,
            "pool_type": "paop_lf_std",
            "continuation_mode": "phase1_v1",
        },
        replay_circuit_context={
            "ansatz": ansatz,
            "theta": np.array([0.3], dtype=float),
            "seed_theta": np.array([0.2], dtype=float),
            "reference_state": np.array(psi0, copy=True),
            "num_qubits": 2,
            "family_info": {"requested": "match_adapt", "resolved": "paop_lf_std"},
            "handoff_state_kind": "prepared_state",
            "provenance_source": "explicit",
            "resolved_seed_policy": "auto",
        },
    )


def test_combined_report_runs_l2_l3_and_writes_pdf(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stage_calls: list[int] = []
    section_calls: list[int] = []
    manifest_calls: list[dict[str, object]] = []

    monkeypatch.setattr(report_cli, "run_stage_pipeline", lambda cfg: stage_calls.append(int(cfg.physics.L)) or _simple_stage_result())

    def _fake_manifest(pdf, **kwargs):
        manifest_calls.append(dict(kwargs))
        report_cli.render_text_page(pdf, ["combined manifest"], fontsize=10, line_spacing=0.03)

    def _fake_section(pdf, *, cfg, stage_result, run_command=None):
        section_calls.append(int(cfg.physics.L))
        report_cli.render_text_page(pdf, [f"section L={int(cfg.physics.L)}"], fontsize=10, line_spacing=0.03)

    monkeypatch.setattr(report_cli, "render_parameter_manifest", _fake_manifest)
    monkeypatch.setattr(report_cli, "write_hh_staged_circuit_report_section", _fake_section)

    output_pdf = tmp_path / "hh_staged_circuit_report.pdf"
    args = report_cli.parse_args(["--output-pdf", str(output_pdf)])
    result = report_cli.run(args)

    assert result == output_pdf
    assert output_pdf.exists()
    assert stage_calls == [2, 3]
    assert section_calls == [2, 3]
    assert manifest_calls
    assert manifest_calls[0]["model"] == "Hubbard-Holstein (HH)"
    assert manifest_calls[0]["drive_enabled"] is False
    assert manifest_calls[0]["t"] == 1.0
    assert manifest_calls[0]["U"] == 2.0
    assert manifest_calls[0]["dv"] == 0.0


def test_section_writer_emits_representative_and_expanded_views(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = wf.resolve_staged_hh_config(
        parse_staged_args(["--L", "2", "--skip-pdf", "--run-replay", "--run-dynamics"])
    )
    stage_result = _simple_stage_result()
    manifest_calls: list[dict[str, object]] = []
    page_calls: list[tuple[str, str, bool]] = []
    summary_titles: list[str] = []

    def _fake_manifest(pdf, **kwargs):
        manifest_calls.append(dict(kwargs))
        wf.render_text_page(pdf, ["manifest"], fontsize=10, line_spacing=0.03)

    def _fake_summary(pdf, *, title, circuit=None, metadata=None, notes=None):
        summary_titles.append(str(title))
        wf.render_text_page(pdf, [str(title)], fontsize=10, line_spacing=0.03)

    def _fake_page(
        pdf,
        *,
        circuit,
        title,
        subtitle=None,
        notes=None,
        fold=30,
        scale=0.65,
        idle_wires=False,
        expand_evolution=False,
    ):
        page_calls.append((str(title), str(subtitle), bool(expand_evolution)))
        wf.render_text_page(pdf, [str(title), str(subtitle)], fontsize=10, line_spacing=0.03)

    monkeypatch.setattr(wf, "render_parameter_manifest", _fake_manifest)
    monkeypatch.setattr(wf, "render_circuit_summary_page", _fake_summary)
    monkeypatch.setattr(wf, "render_circuit_page", _fake_page)

    output_pdf = tmp_path / "section.pdf"
    PdfPages = wf.get_PdfPages()
    with PdfPages(output_pdf) as pdf:
        wf.write_hh_staged_circuit_report_section(
            pdf,
            cfg=cfg,
            stage_result=stage_result,
            run_command="python pipelines/hardcoded/hh_staged_circuit_report.py",
        )

    assert output_pdf.exists()
    assert manifest_calls
    assert manifest_calls[0]["model"] == "Hubbard-Holstein (HH)"
    assert manifest_calls[0]["drive_enabled"] is False
    assert manifest_calls[0]["t"] == 1.0
    assert manifest_calls[0]["U"] == 2.0
    assert manifest_calls[0]["dv"] == 0.0
    assert any(expand is False for _, _, expand in page_calls)
    assert any(expand is True for _, _, expand in page_calls)
    joined_titles = " ".join(summary_titles + [title for title, _, _ in page_calls])
    assert "warm HH-HVA" in joined_titles
    assert "ADAPT-VQE" in joined_titles
    assert "matched-family replay" in joined_titles
    assert "SUZUKI2 dynamics macro-step" in joined_titles
    assert "CFQM4 dynamics macro-step" in joined_titles


def test_build_stage_circuit_report_artifacts_skip_cfqm_when_stage_exp_is_numerical_only(monkeypatch) -> None:
    cfg = wf.resolve_staged_hh_config(
        parse_staged_args(["--L", "2", "--skip-pdf", "--run-replay", "--run-dynamics"])
    )
    stage_result = _simple_stage_result()

    monkeypatch.setattr(
        wf,
        "transpile_circuit_metrics",
        lambda circuit, **kwargs: {
            "target": {"backend_name": kwargs.get("backend_name")},
            "raw": {"depth": 1, "count_2q": 0, "cx_count": 0},
            "expanded_once": {"depth": 2, "count_2q": 0, "cx_count": 0},
            "transpiled": {"depth": 3, "count_2q": 0, "cx_count": 0},
        },
    )
    report_payload = wf.build_stage_circuit_report_artifacts(stage_result, cfg)

    cfqm_bundle = report_payload["dynamics"]["cfqm4"]
    assert cfqm_bundle["circuit"] is None
    assert cfqm_bundle["metadata"]["circuitization"]["supported"] is False
    assert "pauli_suzuki2" in str(cfqm_bundle["metadata"]["circuitization"]["reason"])
    assert cfqm_bundle["metadata"]["trajectory_circuit_metrics"]["dynamics_only"]["skipped"] is True


def test_build_stage_circuit_report_artifacts_include_trajectory_transpile_metrics(monkeypatch) -> None:
    cfg = wf.resolve_staged_hh_config(
        parse_staged_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--run-replay",
                "--run-dynamics",
                "--circuit-backend-name",
                "FakeGuadalupeV2",
                "--circuit-use-fake-backend",
                "--cfqm-stage-exp",
                "pauli_suzuki2",
            ]
        )
    )
    stage_result = _simple_stage_result()

    def _fake_transpile(circuit, **kwargs):
        return {
            "target": {"backend_name": kwargs["backend_name"]},
            "raw": {"depth": 1, "count_2q": 0, "cx_count": 0},
            "expanded_once": {"depth": 2, "count_2q": 1, "cx_count": 1},
            "transpiled": {"depth": 7, "count_2q": 3, "cx_count": 3},
        }

    monkeypatch.setattr(wf, "transpile_circuit_metrics", _fake_transpile)
    report_payload = wf.build_stage_circuit_report_artifacts(stage_result, cfg)

    assert report_payload["transpile_target"]["backend_name"] == "FakeGuadalupeV2"
    suz_metrics = report_payload["dynamics"]["suzuki2"]["metadata"]["trajectory_circuit_metrics"]["dynamics_only"]
    cfqm_metrics = report_payload["dynamics"]["cfqm4"]["metadata"]["trajectory_circuit_metrics"]["dynamics_only"]
    assert suz_metrics["transpiled"]["count_2q"] == 3
    assert suz_metrics["transpiled"]["depth"] == 7
    assert cfqm_metrics["transpiled"]["cx_count"] == 3
    assert report_payload["stages"]["warm_start"]["metadata"]["transpile_metrics"]["transpiled"]["depth"] == 7
