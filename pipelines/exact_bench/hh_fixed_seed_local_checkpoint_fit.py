"""Fixed-seed HH local checkpoint-fit screen under a hard dynamics-only CX budget."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    get_plt,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)
from docs.reports.qiskit_circuit_report import ops_to_circuit, transpile_circuit_metrics
from pipelines.hardcoded.hh_vqe_from_adapt_family import build_replay_sequence_from_input_json
from src.quantum.time_propagation import (
    CheckpointFitConfig,
    LocalPauliAnsatzSpec,
    build_local_pauli_ansatz_terms,
    default_chain_edges,
    expectation_total_hamiltonian,
    fit_checkpoint_trajectory,
    run_exact_driven_reference,
)
from src.quantum.drives_time_potential import build_gaussian_sinusoid_density_drive


_DEFAULT_FIXED_SEED_JSON = Path("artifacts/json/l2_hh_open_direct_adapt_phase3_paoplf_u4_g05_nph2.json")
_DEFAULT_OUTPUT_JSON = Path("artifacts/json/hh_fixed_seed_local_checkpoint_fit.json")
_DEFAULT_OUTPUT_CSV = Path("artifacts/json/hh_fixed_seed_local_checkpoint_fit.csv")
_DEFAULT_OUTPUT_PDF = Path("artifacts/pdf/hh_fixed_seed_local_checkpoint_fit.pdf")
_DEFAULT_RUN_ROOT = Path("artifacts/json/hh_fixed_seed_local_checkpoint_fit_runs")
_DEFAULT_BACKEND_NAME = "FakeGuadalupeV2"
_DEFAULT_REPS = (1, 2, 3)


@dataclass(frozen=True)
class SweepConfig:
    fixed_seed_json: Path
    output_json: Path
    output_csv: Path
    output_pdf: Path
    run_root: Path
    tag: str
    backend_name: str
    use_fake_backend: bool
    circuit_optimization_level: int
    circuit_seed_transpiler: int
    max_cx_budget: int
    t_final: float
    num_times: int
    reference_steps: int
    single_axes: tuple[str, ...]
    entangler_axes: tuple[str, ...]
    reps_list: tuple[int, ...]
    optimizer_method: str
    optimizer_maxiter: int
    optimizer_gtol: float
    optimizer_ftol: float
    angle_bound: float
    param_shift: float
    drive_A: float
    drive_omega: float
    drive_tbar: float
    drive_phi: float
    drive_pattern: str
    drive_t0: float
    drive_time_sampling: str
    skip_pdf: bool = False


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded: dict[str, Any] = {}
            for key in fieldnames:
                raw = _jsonable(row.get(key))
                encoded[key] = json.dumps(raw, sort_keys=True) if isinstance(raw, (dict, list)) else raw
            writer.writerow(encoded)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    text = str(raw).strip()
    if text == "":
        return ()
    seen: set[int] = set()
    out: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if item == "":
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("lists must contain positive integers.")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _parse_axis_tuple(raw: str, *, allowed: set[str]) -> tuple[str, ...]:
    text = str(raw).strip()
    if text == "":
        return ()
    seen: set[str] = set()
    out: list[str] = []
    for part in text.split(","):
        axis = str(part).strip().lower()
        if axis == "":
            continue
        if axis not in allowed:
            raise ValueError(f"Unsupported axis {axis!r}; allowed={sorted(allowed)}")
        if axis in seen:
            continue
        seen.add(axis)
        out.append(axis)
    return tuple(out)


def _collect_hardcoded_terms_exyz(h_poly: Any) -> tuple[list[str], dict[str, complex]]:
    coeff_map: dict[str, complex] = {}
    native_order: list[str] = []
    for term in h_poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if label not in coeff_map:
            native_order.append(label)
            coeff_map[label] = coeff
        else:
            coeff_map[label] = coeff_map[label] + coeff
    return native_order, coeff_map


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = {
        "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    }
    out = mats[str(label)[0]]
    for ch in str(label)[1:]:
        out = np.kron(out, mats[ch])
    return out


def _build_hamiltonian_matrix(coeff_map_exyz: Mapping[str, complex]) -> np.ndarray:
    if not coeff_map_exyz:
        return np.zeros((1, 1), dtype=complex)
    nq = len(next(iter(coeff_map_exyz)))
    dim = 1 << int(nq)
    hmat = np.zeros((dim, dim), dtype=complex)
    for label, coeff in coeff_map_exyz.items():
        hmat += complex(coeff) * _pauli_matrix_exyz(str(label))
    return hmat


def _build_drive_provider(
    *,
    num_sites: int,
    nq_total: int,
    ordering: str,
    cfg: SweepConfig,
) -> tuple[Any, dict[str, Any]]:
    drive = build_gaussian_sinusoid_density_drive(
        n_sites=int(num_sites),
        nq_total=int(nq_total),
        indexing=str(ordering),
        A=float(cfg.drive_A),
        omega=float(cfg.drive_omega),
        tbar=float(cfg.drive_tbar),
        phi=float(cfg.drive_phi),
        pattern_mode=str(cfg.drive_pattern),
        include_identity=False,
        coeff_tol=0.0,
    )
    return drive.coeff_map_exyz, {
        "A": float(cfg.drive_A),
        "omega": float(cfg.drive_omega),
        "tbar": float(cfg.drive_tbar),
        "phi": float(cfg.drive_phi),
        "pattern": str(cfg.drive_pattern),
        "t0": float(cfg.drive_t0),
        "time_sampling": str(cfg.drive_time_sampling),
    }


def _transpile_theta_history(
    *,
    terms: Sequence[Any],
    theta_history: np.ndarray,
    num_qubits: int,
    cfg: SweepConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_row: dict[str, Any] | None = None
    final_row: dict[str, Any] | None = None
    for time_idx, theta in enumerate(np.asarray(theta_history, dtype=float)):
        metrics = transpile_circuit_metrics(
            ops_to_circuit(
                terms,
                np.asarray(theta, dtype=float),
                num_qubits=int(num_qubits),
            ),
            backend_name=str(cfg.backend_name),
            use_fake_backend=bool(cfg.use_fake_backend),
            optimization_level=int(cfg.circuit_optimization_level),
            seed_transpiler=int(cfg.circuit_seed_transpiler),
        )
        tx = metrics.get("transpiled", {})
        row = {
            "time_index": int(time_idx),
            "count_2q": int(tx.get("count_2q", 0)),
            "count_1q": int(tx.get("count_1q", 0)),
            "depth": int(tx.get("depth", 0)),
            "size": int(tx.get("size", 0)),
        }
        rows.append(row)
        final_row = dict(row)
        if max_row is None or (
            int(row["count_2q"]),
            int(row["depth"]),
            int(row["time_index"]),
        ) > (
            int(max_row["count_2q"]),
            int(max_row["depth"]),
            int(max_row["time_index"]),
        ):
            max_row = dict(row)
    return rows, (max_row or {}), (final_row or {})


def _best_row(rows: Sequence[Mapping[str, Any]], *, key: str) -> dict[str, Any] | None:
    valid = [dict(row) for row in rows if row.get("budget_pass", False)]
    if not valid:
        return None
    if str(key) == "error":
        return dict(
            min(
                valid,
                key=lambda row: (
                    float(row.get("max_abs_energy_total_error", float("inf"))),
                    int(row.get("max_transpiled_count_2q", 10**9)),
                    int(row.get("reps", 10**9)),
                ),
            )
        )
    return dict(
        min(
            valid,
            key=lambda row: (
                int(row.get("max_transpiled_count_2q", 10**9)),
                float(row.get("max_abs_energy_total_error", float("inf"))),
                int(row.get("reps", 10**9)),
            ),
        )
    )


def _candidate_row(
    *,
    reps: int,
    label: str,
    times: np.ndarray,
    exact_states: Sequence[np.ndarray],
    exact_energies: np.ndarray,
    fit_result: Any,
    terms: Sequence[Any],
    hmat_static: np.ndarray,
    drive_provider: Any,
    cfg: SweepConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory: list[dict[str, Any]] = []
    energy_errors: list[float] = []
    fidelities: list[float] = []
    for idx, time_val in enumerate(np.asarray(times, dtype=float)):
        psi_fit = np.asarray(fit_result.states[idx], dtype=complex).reshape(-1)
        psi_exact = np.asarray(exact_states[idx], dtype=complex).reshape(-1)
        energy_fit = expectation_total_hamiltonian(
            psi_fit,
            hmat_static,
            drive_coeff_provider_exyz=drive_provider,
            t_physical=float(cfg.drive_t0) + float(time_val),
        )
        fidelity = float(abs(np.vdot(psi_exact / np.linalg.norm(psi_exact), psi_fit / np.linalg.norm(psi_fit))) ** 2)
        energy_error = float(abs(float(energy_fit) - float(exact_energies[idx])))
        energy_errors.append(float(energy_error))
        fidelities.append(float(fidelity))
        trajectory.append(
            {
                "time": float(time_val),
                "energy_total_exact": float(exact_energies[idx]),
                "energy_total_fit": float(energy_fit),
                "abs_energy_total_error": float(energy_error),
                "fidelity": float(fidelity),
                "theta_norm": float(np.linalg.norm(np.asarray(fit_result.theta_history[idx], dtype=float))),
            }
        )
    tx_rows, tx_max, tx_final = _transpile_theta_history(
        terms=terms,
        theta_history=np.asarray(fit_result.theta_history, dtype=float),
        num_qubits=int(np.asarray(fit_result.states[0]).size.bit_length() - 1),
        cfg=cfg,
    )
    row = {
        "ansatz_label": str(label),
        "reps": int(reps),
        "num_parameters": int(len(terms)),
        "max_transpiled_count_2q": int(tx_max.get("count_2q", 0)),
        "max_transpiled_depth": int(tx_max.get("depth", 0)),
        "final_transpiled_count_2q": int(tx_final.get("count_2q", 0)),
        "final_transpiled_depth": int(tx_final.get("depth", 0)),
        "final_fidelity": float(fidelities[-1]),
        "min_fidelity": float(min(fidelities)),
        "max_abs_energy_total_error": float(max(energy_errors)),
        "final_abs_energy_total_error": float(energy_errors[-1]),
        "budget_pass": bool(int(tx_max.get("count_2q", 0)) <= int(cfg.max_cx_budget)),
    }
    return row, {
        "ansatz_label": str(label),
        "reps": int(reps),
        "row": dict(row),
        "trajectory": trajectory,
        "solver_rows": [dict(item) for item in fit_result.solver_rows],
        "transpile_rows": tx_rows,
        "term_labels": [str(term.label) for term in terms],
    }


def write_summary_pdf(
    *,
    cfg: SweepConfig,
    payload: Mapping[str, Any],
    candidate_details: Mapping[str, Mapping[str, Any]],
    seed_payload: Mapping[str, Any],
) -> None:
    if bool(cfg.skip_pdf):
        return
    require_matplotlib()
    plt = get_plt()
    PdfPages = get_PdfPages()
    command = current_command_string()
    rows = [dict(row) for row in payload.get("rows", [])]

    with PdfPages(Path(cfg.output_pdf)) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein (HH)",
            ansatz="Fixed imported state; local checkpoint-fit Pauli ansatz on top of seed",
            drive_enabled=True,
            t=float(seed_payload["settings"]["t"]),
            U=float(seed_payload["settings"]["u"]),
            dv=float(seed_payload["settings"]["dv"]),
            extra={
                "fixed_seed_json": str(cfg.fixed_seed_json),
                "g_ep": float(seed_payload["settings"]["g_ep"]),
                "omega0": float(seed_payload["settings"]["omega0"]),
                "n_ph_max": int(seed_payload["settings"]["n_ph_max"]),
                "budget_scope": "dynamics_only",
                "surrogate_method": "checkpoint_local_fit",
                "single_axes": ", ".join(cfg.single_axes) or "none",
                "entangler_axes": ", ".join(cfg.entangler_axes) or "none",
                "reps_list": ", ".join(str(x) for x in cfg.reps_list),
                "max_cx_budget": int(cfg.max_cx_budget),
                "t_final": float(cfg.t_final),
                "num_times": int(cfg.num_times),
                "reference_steps": int(cfg.reference_steps),
                "backend_name": str(cfg.backend_name),
            },
            command=command,
        )

        fig = plt.figure(figsize=(11.0, 8.5))
        ax = fig.add_subplot(111)
        table_rows = [
            [
                str(int(row["reps"])),
                str(int(row["num_parameters"])),
                str(int(row["max_transpiled_count_2q"])),
                str(int(row["max_transpiled_depth"])),
                f"{float(row['final_fidelity']):.6f}",
                f"{float(row['min_fidelity']):.6f}",
                f"{float(row['max_abs_energy_total_error']):.6e}",
            ]
            for row in rows
        ]
        render_compact_table(
            ax,
            title="Checkpoint-fit scoreboard",
            col_labels=["reps", "params", "Max 2Q", "Max depth", "Final F", "Min F", "Max |dE|"],
            rows=table_rows or [["(none)", "", "", "", "", "", ""]],
            fontsize=7,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        labels = [str(row["ansatz_label"]) for row in rows]
        for key, ylabel, title in (
            ("energy_total_fit", "energy", "Checkpoint-fit vs exact total energy"),
            ("fidelity", "fidelity", "Checkpoint-fit fidelity to exact trajectory"),
        ):
            fig, ax = plt.subplots(figsize=(11.0, 8.5))
            first_label = labels[0] if labels else None
            exact_detail = candidate_details.get(first_label, None)
            if exact_detail is not None and key == "energy_total_fit":
                times = [float(item["time"]) for item in exact_detail["trajectory"]]
                exact_vals = [float(item["energy_total_exact"]) for item in exact_detail["trajectory"]]
                ax.plot(times, exact_vals, color="black", linewidth=2.0, label="exact")
            for label in labels:
                detail = candidate_details.get(str(label), None)
                if detail is None:
                    continue
                times = [float(item["time"]) for item in detail["trajectory"]]
                values = [float(item[key]) for item in detail["trajectory"]]
                ax.plot(times, values, linewidth=1.5, label=str(label))
            ax.set_title(title)
            ax.set_xlabel("time")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        summary_lines = [
            "Checkpoint-local fit summary",
            "",
            "This is not a sequential propagator. Each time checkpoint is fit independently",
            "to the exact driven state using the same fixed imported seed as the external reference state.",
            "",
            f"fixed_seed_json: {cfg.fixed_seed_json}",
            f"backend_name: {cfg.backend_name}",
            f"max_cx_budget: {cfg.max_cx_budget}",
            f"best_by_error_under_budget: {(summary.get('best_by_error_under_budget') or {}).get('ansatz_label', 'n/a')}",
            f"best_by_cx_under_budget: {(summary.get('best_by_cx_under_budget') or {}).get('ansatz_label', 'n/a')}",
        ]
        render_text_page(pdf, summary_lines, fontsize=10, line_spacing=0.03, max_line_width=110)


def run_sweep(cfg: SweepConfig) -> dict[str, Any]:
    replay_data = build_replay_sequence_from_input_json(Path(cfg.fixed_seed_json))
    seed_payload = replay_data["payload"]
    initial_state = np.asarray(replay_data["initial_state"], dtype=complex).reshape(-1)
    h_poly = replay_data["h_poly"]
    num_qubits = int(replay_data["nq"])
    seed_settings = seed_payload["settings"]
    _ordered_labels_exyz, coeff_map_exyz = _collect_hardcoded_terms_exyz(h_poly)
    hmat_static = _build_hamiltonian_matrix(coeff_map_exyz)
    drive_provider, drive_profile = _build_drive_provider(
        num_sites=int(seed_settings["L"]),
        nq_total=int(num_qubits),
        ordering=str(seed_settings["ordering"]),
        cfg=cfg,
    )
    exact_ref = run_exact_driven_reference(
        initial_state,
        hmat_static,
        t_final=float(cfg.t_final),
        num_times=int(cfg.num_times),
        reference_steps=int(cfg.reference_steps),
        drive_coeff_provider_exyz=drive_provider,
        drive_t0=float(cfg.drive_t0),
        time_sampling=str(cfg.drive_time_sampling),
    )

    fit_cfg = CheckpointFitConfig(
        optimizer_method=str(cfg.optimizer_method),
        maxiter=int(cfg.optimizer_maxiter),
        gtol=float(cfg.optimizer_gtol),
        ftol=float(cfg.optimizer_ftol),
        angle_bound=float(cfg.angle_bound),
        param_shift=float(cfg.param_shift),
    )

    rows: list[dict[str, Any]] = []
    candidate_details: dict[str, dict[str, Any]] = {}
    for reps in cfg.reps_list:
        spec = LocalPauliAnsatzSpec(
            num_qubits=int(num_qubits),
            reps=int(reps),
            single_axes=tuple(cfg.single_axes),
            entangler_axes=tuple(cfg.entangler_axes),
        )
        terms = build_local_pauli_ansatz_terms(spec)
        fit_result = fit_checkpoint_trajectory(
            initial_state,
            exact_ref.states,
            exact_ref.times,
            terms,
            config=fit_cfg,
        )
        label = (
            f"local_{'-'.join(cfg.single_axes) or 'none'}_"
            f"{'-'.join(cfg.entangler_axes) or 'none'}_reps{int(reps)}"
        )
        row, detail = _candidate_row(
            reps=int(reps),
            label=str(label),
            times=np.asarray(exact_ref.times, dtype=float),
            exact_states=exact_ref.states,
            exact_energies=np.asarray(exact_ref.energies_total, dtype=float),
            fit_result=fit_result,
            terms=terms,
            hmat_static=hmat_static,
            drive_provider=drive_provider,
            cfg=cfg,
        )
        rows.append(dict(row))
        run_dir = Path(cfg.run_root) / str(label)
        run_dir.mkdir(parents=True, exist_ok=True)
        detail_out = {**dict(detail), "run_dir": str(run_dir)}
        candidate_details[str(label)] = detail_out
        _write_json(run_dir / "candidate.json", detail_out)

    feasible = [dict(row) for row in rows if bool(row.get("budget_pass", False))]
    payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_fixed_seed_local_checkpoint_fit",
        "model_family": "Hubbard-Holstein (HH)",
        "scope": {
            "local_only": True,
            "noiseless_only": True,
            "budget_scope": "dynamics_only",
            "surrogate_method": "checkpoint_local_fit",
        },
        "settings": asdict(cfg),
        "seed": {
            "fixed_seed_json": str(cfg.fixed_seed_json),
            "family_info": dict(replay_data["family_info"]),
            "family_resolved": str(replay_data["family_resolved"]),
            "term_count": int(len(replay_data["replay_terms"])),
            "pool_meta": dict(replay_data["pool_meta"]),
        },
        "ansatz_family": {
            "single_axes": list(cfg.single_axes),
            "entangler_axes": list(cfg.entangler_axes),
            "logical_entangler_edges": [[int(q0), int(q1)] for q0, q1 in default_chain_edges(int(num_qubits))],
        },
        "drive_profile": dict(drive_profile),
        "artifacts": {
            "output_json": str(cfg.output_json),
            "output_csv": str(cfg.output_csv),
            "output_pdf": str(cfg.output_pdf),
            "run_root": str(cfg.run_root),
        },
        "exact_reference": {
            "reference_steps": int(cfg.reference_steps),
            "time_sampling": str(cfg.drive_time_sampling),
            "trajectory": [dict(row) for row in exact_ref.trajectory_rows],
        },
        "rows": rows,
        "summary": {
            "feasible_count": int(len(feasible)),
            "best_by_error_under_budget": _best_row(rows, key="error"),
            "best_by_cx_under_budget": _best_row(rows, key="cx"),
        },
    }
    _write_json(Path(cfg.output_json), payload)
    _write_csv(Path(cfg.output_csv), rows)
    write_summary_pdf(
        cfg=cfg,
        payload=payload,
        candidate_details=candidate_details,
        seed_payload=seed_payload,
    )
    return payload


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed-seed HH local checkpoint-fit sweep over shallow local-Pauli "
            "dynamics circuits under a hard dynamics-only CX budget."
        )
    )
    parser.add_argument("--fixed-seed-json", type=Path, default=_DEFAULT_FIXED_SEED_JSON)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=_DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-pdf", type=Path, default=_DEFAULT_OUTPUT_PDF)
    parser.add_argument("--run-root", type=Path, default=_DEFAULT_RUN_ROOT)
    parser.add_argument("--tag", type=str, default="local_checkpoint_fit")
    parser.add_argument("--backend-name", type=str, default=_DEFAULT_BACKEND_NAME)
    parser.add_argument("--circuit-optimization-level", type=int, default=3)
    parser.add_argument("--circuit-seed-transpiler", type=int, default=11)
    parser.add_argument("--max-cx-budget", type=int, default=100)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--num-times", type=int, default=41)
    parser.add_argument("--reference-steps", type=int, default=256)
    parser.add_argument("--single-axes", type=str, default="y")
    parser.add_argument("--entangler-axes", type=str, default="xx,zz")
    parser.add_argument("--reps-list", type=str, default=",".join(str(x) for x in _DEFAULT_REPS))
    parser.add_argument("--optimizer-method", type=str, default="L-BFGS-B")
    parser.add_argument("--optimizer-maxiter", type=int, default=60)
    parser.add_argument("--optimizer-gtol", type=float, default=1e-8)
    parser.add_argument("--optimizer-ftol", type=float, default=1e-12)
    parser.add_argument("--angle-bound", type=float, default=float(math.pi))
    parser.add_argument("--param-shift", type=float, default=float(math.pi / 2.0))
    parser.add_argument("--drive-A", type=float, default=1.0)
    parser.add_argument("--drive-omega", type=float, default=1.0)
    parser.add_argument("--drive-tbar", type=float, default=5.0)
    parser.add_argument("--drive-phi", type=float, default=0.0)
    parser.add_argument("--drive-pattern", type=str, default="staggered")
    parser.add_argument("--drive-t0", type=float, default=0.0)
    parser.add_argument("--drive-time-sampling", type=str, default="midpoint")
    parser.add_argument("--skip-pdf", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> SweepConfig:
    args = build_cli_parser().parse_args(list(argv) if argv is not None else None)
    if int(args.circuit_optimization_level) < 0 or int(args.circuit_optimization_level) > 3:
        raise ValueError("--circuit-optimization-level must be in {0,1,2,3}.")
    if int(args.max_cx_budget) <= 0:
        raise ValueError("--max-cx-budget must be positive.")
    if int(args.num_times) < 2:
        raise ValueError("--num-times must be >= 2.")
    if int(args.reference_steps) < 1:
        raise ValueError("--reference-steps must be >= 1.")
    if int(args.optimizer_maxiter) < 1:
        raise ValueError("--optimizer-maxiter must be >= 1.")
    single_axes = _parse_axis_tuple(str(args.single_axes), allowed={"x", "y", "z"})
    entangler_axes = _parse_axis_tuple(str(args.entangler_axes), allowed={"xx", "yy", "zz"})
    reps_list = _parse_int_tuple(str(args.reps_list))
    if not single_axes and not entangler_axes:
        raise ValueError("Need at least one single axis or entangler axis.")
    if not reps_list:
        raise ValueError("Need at least one reps value.")
    return SweepConfig(
        fixed_seed_json=Path(args.fixed_seed_json),
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
        output_pdf=Path(args.output_pdf),
        run_root=Path(args.run_root),
        tag=str(args.tag),
        backend_name=str(args.backend_name),
        use_fake_backend=True,
        circuit_optimization_level=int(args.circuit_optimization_level),
        circuit_seed_transpiler=int(args.circuit_seed_transpiler),
        max_cx_budget=int(args.max_cx_budget),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        reference_steps=int(args.reference_steps),
        single_axes=single_axes,
        entangler_axes=entangler_axes,
        reps_list=reps_list,
        optimizer_method=str(args.optimizer_method),
        optimizer_maxiter=int(args.optimizer_maxiter),
        optimizer_gtol=float(args.optimizer_gtol),
        optimizer_ftol=float(args.optimizer_ftol),
        angle_bound=float(args.angle_bound),
        param_shift=float(args.param_shift),
        drive_A=float(args.drive_A),
        drive_omega=float(args.drive_omega),
        drive_tbar=float(args.drive_tbar),
        drive_phi=float(args.drive_phi),
        drive_pattern=str(args.drive_pattern),
        drive_t0=float(args.drive_t0),
        drive_time_sampling=str(args.drive_time_sampling),
        skip_pdf=bool(args.skip_pdf),
    )


def main(argv: Sequence[str] | None = None) -> int:
    cfg = parse_args(argv)
    run_sweep(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
