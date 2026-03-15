#!/usr/bin/env python3
"""Fixed-seed HH noiseless dynamics sweep with local transpile metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from docs.reports.qiskit_circuit_report import build_time_dynamics_circuit, transpile_circuit_metrics
from pipelines.hardcoded.hh_staged_noiseless import parse_args as parse_staged_args
from pipelines.hardcoded.hh_staged_workflow import resolve_staged_hh_config, run_staged_hh_noiseless

_DEFAULT_FIXED_FINAL_STATE_JSON = REPO_ROOT / "artifacts" / "json" / "l2_hh_open_direct_adapt_phase3_paoplf_u4_g05_nph2.json"
_DEFAULT_OUTPUT_JSON = REPO_ROOT / "artifacts" / "json" / "hh_fixed_seed_qpu_prep_sweep.json"
_DEFAULT_OUTPUT_CSV = REPO_ROOT / "artifacts" / "json" / "hh_fixed_seed_qpu_prep_sweep.csv"
_DEFAULT_OUTPUT_PDF = REPO_ROOT / "artifacts" / "pdf" / "hh_fixed_seed_qpu_prep_sweep.pdf"
_DEFAULT_RUN_ROOT = REPO_ROOT / "artifacts" / "json" / "hh_fixed_seed_qpu_prep_sweep_runs"
_DEFAULT_TAG = "hh_fixed_seed_qpu_prep_sweep"
_DEFAULT_SUZUKI_STEPS = (16, 32, 48, 64, 96, 128)
_DEFAULT_CFQM_STEPS = (8, 16, 24, 32, 48, 64)
_DEFAULT_BACKEND_NAME = "FakeGuadalupeV2"
_DEFAULT_BUDGET_MODE = "full_trajectory"


@dataclass(frozen=True)
class SweepConfig:
    fixed_final_state_json: Path
    output_json: Path
    output_csv: Path
    output_pdf: Path
    run_root: Path
    tag: str
    backend_name: str
    use_fake_backend: bool
    circuit_optimization_level: int
    circuit_seed_transpiler: int
    suzuki_steps: tuple[int, ...]
    cfqm_steps: tuple[int, ...]
    t_final: float
    num_times: int
    exact_steps_multiplier: int
    drive_A: float
    drive_omega: float
    drive_tbar: float
    drive_phi: float
    drive_pattern: str
    drive_t0: float
    drive_time_sampling: str
    budget_mode: str
    cfqm_stage_exp: str
    cfqm_coeff_drop_abs_tol: float
    cfqm_normalize: bool


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if str(key) not in fieldnames:
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


def _parse_steps(raw: str) -> tuple[int, ...]:
    seen: set[int] = set()
    out: list[int] = []
    text_raw = str(raw).strip()
    if text_raw == "":
        return ()
    for part in text_raw.split(","):
        text = part.strip()
        if text == "":
            continue
        value = int(text)
        if value < 1:
            raise ValueError("step grids must contain positive integers")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _load_seed_settings(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    settings = raw.get("settings", {})
    if not isinstance(settings, Mapping):
        raise ValueError(f"Seed JSON {path} is missing a settings block.")
    required = ("L", "t", "u", "dv", "omega0", "g_ep", "n_ph_max", "boson_encoding", "ordering", "boundary")
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(f"Seed JSON {path} is missing required settings fields: {missing}")
    return dict(settings)


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


def _build_drive_provider(
    *,
    num_sites: int,
    nq_total: int,
    ordering: str,
    cfg: SweepConfig,
) -> Any:
    from src.quantum.drives_time_potential import build_gaussian_sinusoid_density_drive

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
    return drive.coeff_map_exyz


def _build_snapshot_budget_context(cfg: SweepConfig, *, seed_settings: Mapping[str, Any]) -> dict[str, Any]:
    from pipelines.hardcoded.hh_vqe_from_adapt_family import build_replay_sequence_from_input_json

    replay_ctx = build_replay_sequence_from_input_json(cfg.fixed_final_state_json)
    ordered_labels_exyz, static_coeff_map_exyz = _collect_hardcoded_terms_exyz(replay_ctx["h_poly"])
    drive_provider_exyz = _build_drive_provider(
        num_sites=int(seed_settings["L"]),
        nq_total=int(replay_ctx["nq"]),
        ordering=str(seed_settings["ordering"]),
        cfg=cfg,
    )
    return {
        "num_qubits": int(replay_ctx["nq"]),
        "ordered_labels_exyz": list(ordered_labels_exyz),
        "static_coeff_map_exyz": dict(static_coeff_map_exyz),
        "drive_provider_exyz": drive_provider_exyz,
    }


def _build_candidate_args(
    *,
    cfg: SweepConfig,
    seed_settings: Mapping[str, Any],
    method: str,
    trotter_steps: int,
    run_dir: Path,
) -> list[str]:
    args = [
        "--L",
        str(int(seed_settings["L"])),
        "--t",
        str(float(seed_settings["t"])),
        "--u",
        str(float(seed_settings["u"])),
        "--dv",
        str(float(seed_settings["dv"])),
        "--omega0",
        str(float(seed_settings["omega0"])),
        "--g-ep",
        str(float(seed_settings["g_ep"])),
        "--n-ph-max",
        str(int(seed_settings["n_ph_max"])),
        "--boson-encoding",
        str(seed_settings["boson_encoding"]),
        "--ordering",
        str(seed_settings["ordering"]),
        "--boundary",
        str(seed_settings["boundary"]),
        "--adapt-continuation-mode",
        str(seed_settings.get("adapt_continuation_mode", "phase3_v1")),
        "--fixed-final-state-json",
        str(cfg.fixed_final_state_json),
        "--enable-drive",
        "--drive-A",
        str(float(cfg.drive_A)),
        "--drive-omega",
        str(float(cfg.drive_omega)),
        "--drive-tbar",
        str(float(cfg.drive_tbar)),
        "--drive-phi",
        str(float(cfg.drive_phi)),
        "--drive-pattern",
        str(cfg.drive_pattern),
        "--drive-t0",
        str(float(cfg.drive_t0)),
        "--drive-time-sampling",
        str(cfg.drive_time_sampling),
        "--noiseless-methods",
        str(method),
        "--t-final",
        str(float(cfg.t_final)),
        "--num-times",
        str(int(cfg.num_times)),
        "--trotter-steps",
        str(int(trotter_steps)),
        "--exact-steps-multiplier",
        str(int(cfg.exact_steps_multiplier)),
        "--cfqm-stage-exp",
        str(cfg.cfqm_stage_exp),
        "--cfqm-coeff-drop-abs-tol",
        str(float(cfg.cfqm_coeff_drop_abs_tol)),
        "--circuit-backend-name",
        str(cfg.backend_name),
        "--circuit-use-fake-backend",
        "--circuit-transpile-optimization-level",
        str(int(cfg.circuit_optimization_level)),
        "--circuit-seed-transpiler",
        str(int(cfg.circuit_seed_transpiler)),
        "--output-json",
        str(run_dir / "workflow.json"),
        "--output-pdf",
        str(run_dir / "workflow.pdf"),
        "--tag",
        f"{cfg.tag}_{method}_S{int(trotter_steps)}",
        "--skip-pdf",
    ]
    if bool(cfg.cfqm_normalize):
        args.append("--cfqm-normalize")
    if int(trotter_steps) < 128:
        args.append("--smoke-test-intentionally-weak")
    return args


def _extract_drive_profile(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    profiles = payload.get("dynamics_noiseless", {}).get("profiles", {})
    if not isinstance(profiles, Mapping) or "drive" not in profiles:
        raise ValueError("Expected drive profile in fixed-seed sweep payload.")
    drive_profile = profiles["drive"]
    if not isinstance(drive_profile, Mapping):
        raise ValueError("Drive profile payload is malformed.")
    return drive_profile


def _snapshot_budget_details(
    *,
    cfg: SweepConfig,
    snapshot_ctx: Mapping[str, Any],
    method: str,
    trotter_steps: int,
    times: Sequence[float],
) -> dict[str, Any]:
    from qiskit import QuantumCircuit

    rows: list[dict[str, Any]] = []
    max_row: dict[str, Any] | None = None
    final_row: dict[str, Any] | None = None
    for time_idx, time_value in enumerate(times):
        qc = build_time_dynamics_circuit(
            method=str(method),
            initial_circuit=QuantumCircuit(int(snapshot_ctx["num_qubits"])),
            ordered_labels_exyz=list(snapshot_ctx["ordered_labels_exyz"]),
            static_coeff_map_exyz=dict(snapshot_ctx["static_coeff_map_exyz"]),
            drive_provider_exyz=snapshot_ctx["drive_provider_exyz"],
            time_value=float(time_value),
            trotter_steps=int(trotter_steps),
            drive_t0=float(cfg.drive_t0),
            drive_time_sampling=str(cfg.drive_time_sampling),
            cfqm_stage_exp=str(cfg.cfqm_stage_exp),
            cfqm_coeff_drop_abs_tol=float(cfg.cfqm_coeff_drop_abs_tol),
        )
        metrics = transpile_circuit_metrics(
            qc,
            backend_name=str(cfg.backend_name),
            use_fake_backend=bool(cfg.use_fake_backend),
            optimization_level=int(cfg.circuit_optimization_level),
            seed_transpiler=int(cfg.circuit_seed_transpiler),
        )
        tx = metrics.get("transpiled", {})
        row = {
            "time_index": int(time_idx),
            "time": float(time_value),
            "count_2q": int(tx.get("count_2q", 0)),
            "cx_count": int(tx.get("cx_count", 0)),
            "depth": int(tx.get("depth", 0)),
            "size": int(tx.get("size", 0)),
        }
        rows.append(dict(row))
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
    return {
        "transpile_rows": rows,
        "max": (max_row or {}),
        "final": (final_row or {}),
    }


def _candidate_row(
    payload: Mapping[str, Any],
    *,
    method: str,
    trotter_steps: int,
    run_dir: Path,
    cfg: SweepConfig,
    snapshot_budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    drive_profile = _extract_drive_profile(payload)
    method_payload = drive_profile.get("methods", {}).get(str(method), {})
    if not isinstance(method_payload, Mapping):
        raise ValueError(f"Missing drive payload for method {method}.")
    trajectory = method_payload.get("trajectory", [])
    if not isinstance(trajectory, Sequence) or not trajectory:
        raise ValueError(f"Drive trajectory for method {method} is empty.")
    final = method_payload.get("final", {})
    circuit_payload = payload.get("circuit_metrics", {})
    if not isinstance(circuit_payload, Mapping):
        circuit_payload = {}
    circuit_metrics = (
        circuit_payload.get("dynamics", {})
        .get(str(method), {})
        .get("metadata", {})
    )
    if not isinstance(circuit_metrics, Mapping):
        circuit_metrics = {}
    trajectory_metrics = circuit_metrics.get("trajectory_circuit_metrics", {})
    if not isinstance(trajectory_metrics, Mapping):
        trajectory_metrics = {}
    dyn_tx = trajectory_metrics.get("dynamics_only", {})
    prep_tx = trajectory_metrics.get("prep_plus_dynamics", {})
    dyn_transpiled = dyn_tx.get("transpiled", {}) if isinstance(dyn_tx, Mapping) else {}
    prep_transpiled = prep_tx.get("transpiled", {}) if isinstance(prep_tx, Mapping) else {}
    proxy_total = circuit_metrics.get("proxy_total", {}) if isinstance(circuit_metrics.get("proxy_total", {}), Mapping) else {}
    fidelities = [float(row.get("fidelity", float("nan"))) for row in trajectory]
    max_abs_energy_total_error = max(
        abs(float(row.get("energy_total_trotter", float("nan"))) - float(row.get("energy_total_exact", float("nan"))))
        for row in trajectory
    )
    budget_mode = str(cfg.budget_mode)
    if budget_mode == "snapshot":
        snap_max = dict((snapshot_budget or {}).get("max", {}))
        snap_final = dict((snapshot_budget or {}).get("final", {}))
        budget_source = "snapshot_max_dynamics_only"
        budget_count_2q = snap_max.get("count_2q")
        budget_cx_count = snap_max.get("cx_count")
        budget_depth = snap_max.get("depth")
    else:
        snap_max = {}
        snap_final = {}
        budget_source = "full_trajectory_dynamics_only"
        budget_count_2q = dyn_transpiled.get("count_2q")
        budget_cx_count = dyn_transpiled.get("cx_count")
        budget_depth = dyn_transpiled.get("depth")
    return {
        "method": str(method),
        "trotter_steps": int(trotter_steps),
        "hardware_diagnostic_only": bool(int(trotter_steps) < 128),
        "budget_mode": budget_mode,
        "budget_scope": "dynamics_only",
        "budget_source": str(budget_source),
        "budget_count_2q": budget_count_2q,
        "budget_cx_count": budget_cx_count,
        "budget_depth": budget_depth,
        "run_dir": str(run_dir),
        "workflow_json": str(run_dir / "workflow.json"),
        "workflow_pdf": str(run_dir / "workflow.pdf"),
        "snapshot_metrics_json": (str(run_dir / "snapshot_metrics.json") if snapshot_budget is not None else None),
        "transpile_backend": circuit_payload.get("transpile_target", {}).get("backend_name"),
        "dynamics_only_count_2q": dyn_transpiled.get("count_2q"),
        "dynamics_only_cx_count": dyn_transpiled.get("cx_count"),
        "dynamics_only_depth": dyn_transpiled.get("depth"),
        "prep_plus_dynamics_count_2q": prep_transpiled.get("count_2q") if isinstance(prep_tx, Mapping) and "skipped" not in prep_tx else None,
        "prep_plus_dynamics_cx_count": prep_transpiled.get("cx_count") if isinstance(prep_tx, Mapping) and "skipped" not in prep_tx else None,
        "prep_plus_dynamics_depth": prep_transpiled.get("depth") if isinstance(prep_tx, Mapping) and "skipped" not in prep_tx else None,
        "snapshot_max_count_2q": snap_max.get("count_2q"),
        "snapshot_max_cx_count": snap_max.get("cx_count"),
        "snapshot_max_depth": snap_max.get("depth"),
        "snapshot_final_count_2q": snap_final.get("count_2q"),
        "snapshot_final_cx_count": snap_final.get("cx_count"),
        "snapshot_final_depth": snap_final.get("depth"),
        "cx_proxy_total": proxy_total.get("cx_proxy_total"),
        "depth_proxy_total": proxy_total.get("depth_proxy_total"),
        "final_fidelity": float(final.get("fidelity", float("nan"))),
        "min_fidelity": float(min(fidelities)),
        "max_abs_energy_total_error": float(max_abs_energy_total_error),
        "final_abs_energy_total_error": float(final.get("abs_energy_total_error", float("nan"))),
        "final_abs_energy_error_vs_ground_state": float(final.get("abs_energy_error_vs_ground_state", float("nan"))),
        "transpile_status": (
            dyn_tx.get("error")
            if isinstance(dyn_tx, Mapping) and dyn_tx.get("error") is not None
            else dyn_tx.get("reason")
            if isinstance(dyn_tx, Mapping) and dyn_tx.get("skipped")
            else "ok"
        ),
    }


def _is_dominated(row_i: Mapping[str, Any], row_j: Mapping[str, Any]) -> bool:
    if row_i.get("budget_count_2q") is None or row_i.get("budget_depth") is None:
        return False
    if row_j.get("budget_count_2q") is None or row_j.get("budget_depth") is None:
        return False
    pair_i = (int(row_i["budget_count_2q"]), int(row_i["budget_depth"]))
    pair_j = (int(row_j["budget_count_2q"]), int(row_j["budget_depth"]))
    return pair_j[0] <= pair_i[0] and pair_j[1] <= pair_i[1] and pair_j != pair_i


def _pareto_shortlist(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    shortlist: list[dict[str, Any]] = []
    for row in rows:
        if any(_is_dominated(row, other) for other in rows if other is not row):
            continue
        shortlist.append(dict(row))
    shortlist.sort(
        key=lambda rec: (
            int(rec.get("budget_count_2q", 10**9) or 10**9),
            int(rec.get("budget_depth", 10**9) or 10**9),
            float(rec.get("max_abs_energy_total_error", float("inf"))),
        )
    )
    return shortlist


def _plot_energy_page(pdf: Any, *, method: str, candidates: Sequence[Mapping[str, Any]], title_suffix: str) -> None:
    require_matplotlib()
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    exact_drawn = False
    for candidate in candidates:
        drive_profile = _extract_drive_profile(candidate["payload"])
        method_payload = drive_profile["methods"][str(method)]
        rows = method_payload["trajectory"]
        times = [float(row["time"]) for row in rows]
        energies = [float(row["energy_total_trotter"]) for row in rows]
        ax.plot(times, energies, label=f"S={int(candidate['trotter_steps'])}")
        if not exact_drawn:
            exact = [float(row["energy_total_exact"]) for row in rows]
            ax.plot(times, exact, color="black", linewidth=2.0, linestyle="--", label="exact")
            exact_drawn = True
    ax.set_title(f"{method.upper()} driven energy dynamics ({title_suffix})")
    ax.set_xlabel("time")
    ax.set_ylabel("energy_total")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    pdf.savefig(fig)
    plt.close(fig)


def _plot_fidelity_page(pdf: Any, *, method: str, candidates: Sequence[Mapping[str, Any]], title_suffix: str) -> None:
    require_matplotlib()
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    for candidate in candidates:
        drive_profile = _extract_drive_profile(candidate["payload"])
        method_payload = drive_profile["methods"][str(method)]
        rows = method_payload["trajectory"]
        times = [float(row["time"]) for row in rows]
        fidelities = [float(row["fidelity"]) for row in rows]
        ax.plot(times, fidelities, label=f"S={int(candidate['trotter_steps'])}")
    ax.set_title(f"{method.upper()} driven fidelity ({title_suffix})")
    ax.set_xlabel("time")
    ax.set_ylabel("fidelity")
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    pdf.savefig(fig)
    plt.close(fig)


def _write_summary_pdf(
    cfg: SweepConfig,
    *,
    seed_settings: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    pareto_shortlist: Sequence[Mapping[str, Any]],
    run_command: str,
) -> None:
    require_matplotlib()
    cfg.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    PdfPages = get_PdfPages()
    plt = get_plt()
    with PdfPages(cfg.output_pdf) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein (HH)",
            ansatz="fixed imported state; driven Suzuki2 / CFQM4 dynamics sweep",
            drive_enabled=True,
            t=float(seed_settings["t"]),
            U=float(seed_settings["u"]),
            dv=float(seed_settings["dv"]),
            extra={
                "L": int(seed_settings["L"]),
                "omega0": float(seed_settings["omega0"]),
                "g_ep": float(seed_settings["g_ep"]),
                "n_ph_max": int(seed_settings["n_ph_max"]),
                "ordering": str(seed_settings["ordering"]),
                "boundary": str(seed_settings["boundary"]),
                "fixed_final_state_json": str(cfg.fixed_final_state_json),
                "circuit_backend_name": str(cfg.backend_name),
                "suzuki_steps": ",".join(str(x) for x in cfg.suzuki_steps),
                "cfqm_steps": ",".join(str(x) for x in cfg.cfqm_steps),
                "t_final": float(cfg.t_final),
                "num_times": int(cfg.num_times),
                "exact_steps_multiplier": int(cfg.exact_steps_multiplier),
                "drive_A": float(cfg.drive_A),
                "drive_omega": float(cfg.drive_omega),
                "drive_tbar": float(cfg.drive_tbar),
                "drive_phi": float(cfg.drive_phi),
                "drive_pattern": str(cfg.drive_pattern),
                "budget_mode": str(cfg.budget_mode),
                "cfqm_stage_exp": str(cfg.cfqm_stage_exp),
            },
            command=run_command,
        )
        budget_line = (
            "Pareto shortlist is computed on the max per-snapshot transpiled dynamics circuit "
            "(count_2q/depth over sampled times). Full-trajectory dynamics-only counts remain in CSV/JSON."
            if str(cfg.budget_mode) == "snapshot"
            else "Pareto shortlist is computed on the full-trajectory dynamics-only transpiled circuit "
            "(single circuit to t_final)."
        )
        render_text_page(
            pdf,
            [
                "Fixed-seed HH QPU-prep sweep",
                "",
                f"fixed_final_state_json: {cfg.fixed_final_state_json}",
                f"run_root: {cfg.run_root}",
                f"transpile_target: {cfg.backend_name} (fake_backend={cfg.use_fake_backend})",
                f"budget_mode: {cfg.budget_mode}",
                "",
                budget_line,
                "Accuracy selection remains visual: inspect the driven energy overlays and fidelity overlays.",
            ],
            fontsize=10,
            line_spacing=0.03,
            max_line_width=110,
        )
        fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5))
        scoreboard_rows = [
            [
                str(row.get("method", "")),
                str(row.get("trotter_steps", "")),
                str(row.get("budget_cx_count", "")),
                str(row.get("budget_depth", "")),
                f"{float(row.get('final_fidelity', float('nan'))):.6f}",
                f"{float(row.get('min_fidelity', float('nan'))):.6f}",
                f"{float(row.get('max_abs_energy_total_error', float('nan'))):.3e}",
            ]
            for row in rows
        ]
        render_compact_table(
            axes[0],
            title="Driven scoreboard",
            col_labels=["Method", "S", "Budget CX", "Budget depth", "Final F", "Min F", "Max |dE|"],
            rows=scoreboard_rows or [["(none)", "", "", "", "", "", ""]],
            fontsize=8,
        )
        pareto_rows = [
            [
                str(row.get("method", "")),
                str(row.get("trotter_steps", "")),
                str(row.get("budget_cx_count", "")),
                str(row.get("budget_depth", "")),
                f"{float(row.get('max_abs_energy_total_error', float('nan'))):.3e}",
            ]
            for row in pareto_shortlist
        ]
        render_compact_table(
            axes[1],
            title="Pareto shortlist",
            col_labels=["Method", "S", "Budget CX", "Budget depth", "Max |dE|"],
            rows=pareto_rows or [["(none)", "", "", "", ""]],
            fontsize=8,
        )
        pdf.savefig(fig)
        plt.close(fig)
        for method in ("suzuki2", "cfqm4"):
            subset = [candidate for candidate in candidates if str(candidate["method"]) == method]
            if not subset:
                continue
            _plot_energy_page(pdf, method=method, candidates=subset, title_suffix=f"backend={cfg.backend_name}")
            _plot_fidelity_page(pdf, method=method, candidates=subset, title_suffix=f"backend={cfg.backend_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-seed HH driven dynamics sweep with local transpile metrics.")
    parser.add_argument("--fixed-final-state-json", type=Path, default=_DEFAULT_FIXED_FINAL_STATE_JSON)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=_DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-pdf", type=Path, default=_DEFAULT_OUTPUT_PDF)
    parser.add_argument("--run-root", type=Path, default=_DEFAULT_RUN_ROOT)
    parser.add_argument("--tag", type=str, default=_DEFAULT_TAG)
    parser.add_argument("--circuit-backend-name", type=str, default=_DEFAULT_BACKEND_NAME)
    parser.set_defaults(circuit_use_fake_backend=True)
    parser.add_argument("--circuit-use-fake-backend", dest="circuit_use_fake_backend", action="store_true")
    parser.add_argument("--no-circuit-use-fake-backend", dest="circuit_use_fake_backend", action="store_false")
    parser.add_argument("--circuit-transpile-optimization-level", type=int, default=3)
    parser.add_argument("--circuit-seed-transpiler", type=int, default=7)
    parser.add_argument("--suzuki-steps", type=str, default=",".join(str(x) for x in _DEFAULT_SUZUKI_STEPS))
    parser.add_argument("--cfqm-steps", type=str, default=",".join(str(x) for x in _DEFAULT_CFQM_STEPS))
    parser.add_argument("--t-final", type=float, default=10.0)
    parser.add_argument("--num-times", type=int, default=201)
    parser.add_argument("--exact-steps-multiplier", type=int, default=4)
    parser.add_argument("--drive-A", type=float, default=1.0)
    parser.add_argument("--drive-omega", type=float, default=1.0)
    parser.add_argument("--drive-tbar", type=float, default=5.0)
    parser.add_argument("--drive-phi", type=float, default=0.0)
    parser.add_argument("--drive-pattern", choices=["staggered", "dimer_bias", "custom"], default="staggered")
    parser.add_argument("--drive-t0", type=float, default=0.0)
    parser.add_argument("--drive-time-sampling", choices=["midpoint", "left", "right"], default="midpoint")
    parser.add_argument(
        "--budget-mode",
        choices=["full_trajectory", "snapshot"],
        default=_DEFAULT_BUDGET_MODE,
        help=(
            "Budget surface for Pareto/scoreboard fields: "
            "full_trajectory = one dynamics circuit to t_final; "
            "snapshot = max per-sampled-time dynamics circuit when each sampled time is a separate job."
        ),
    )
    parser.add_argument("--cfqm-stage-exp", choices=["pauli_suzuki2"], default="pauli_suzuki2")
    parser.add_argument("--cfqm-coeff-drop-abs-tol", type=float, default=0.0)
    parser.add_argument("--cfqm-normalize", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> SweepConfig:
    args = build_parser().parse_args(argv)
    return SweepConfig(
        fixed_final_state_json=Path(args.fixed_final_state_json),
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
        output_pdf=Path(args.output_pdf),
        run_root=Path(args.run_root),
        tag=str(args.tag),
        backend_name=str(args.circuit_backend_name),
        use_fake_backend=bool(args.circuit_use_fake_backend),
        circuit_optimization_level=int(args.circuit_transpile_optimization_level),
        circuit_seed_transpiler=int(args.circuit_seed_transpiler),
        suzuki_steps=_parse_steps(str(args.suzuki_steps)),
        cfqm_steps=_parse_steps(str(args.cfqm_steps)),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        exact_steps_multiplier=int(args.exact_steps_multiplier),
        drive_A=float(args.drive_A),
        drive_omega=float(args.drive_omega),
        drive_tbar=float(args.drive_tbar),
        drive_phi=float(args.drive_phi),
        drive_pattern=str(args.drive_pattern),
        drive_t0=float(args.drive_t0),
        drive_time_sampling=str(args.drive_time_sampling),
        budget_mode=str(args.budget_mode),
        cfqm_stage_exp=str(args.cfqm_stage_exp),
        cfqm_coeff_drop_abs_tol=float(args.cfqm_coeff_drop_abs_tol),
        cfqm_normalize=bool(args.cfqm_normalize),
    )


def run_sweep(cfg: SweepConfig, *, run_command: str | None = None) -> dict[str, Any]:
    run_command_str = current_command_string() if run_command is None else str(run_command)
    seed_settings = _load_seed_settings(cfg.fixed_final_state_json)
    cfg.run_root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    snapshot_ctx = (
        _build_snapshot_budget_context(cfg, seed_settings=seed_settings)
        if str(cfg.budget_mode) == "snapshot"
        else None
    )
    method_grids = [(method, steps_grid) for method, steps_grid in (("suzuki2", cfg.suzuki_steps), ("cfqm4", cfg.cfqm_steps)) if steps_grid]
    if not method_grids:
        raise ValueError("At least one non-empty step grid is required.")
    for method, steps_grid in method_grids:
        for trotter_steps in steps_grid:
            run_dir = cfg.run_root / f"{method}_S{int(trotter_steps)}"
            staged_args = parse_staged_args(
                _build_candidate_args(
                    cfg=cfg,
                    seed_settings=seed_settings,
                    method=method,
                    trotter_steps=int(trotter_steps),
                    run_dir=run_dir,
                )
            )
            staged_cfg = resolve_staged_hh_config(staged_args)
            payload = run_staged_hh_noiseless(
                staged_cfg,
                run_command=f"{run_command_str}::{method}_S{int(trotter_steps)}",
            )
            snapshot_budget = None
            if snapshot_ctx is not None:
                drive_profile = _extract_drive_profile(payload)
                method_payload = drive_profile.get("methods", {}).get(str(method), {})
                trajectory = method_payload.get("trajectory", []) if isinstance(method_payload, Mapping) else []
                if not isinstance(trajectory, Sequence) or not trajectory:
                    raise ValueError(f"Drive trajectory for method {method} is empty.")
                times = [float(item.get("time", 0.0)) for item in trajectory]
                snapshot_budget = _snapshot_budget_details(
                    cfg=cfg,
                    snapshot_ctx=snapshot_ctx,
                    method=method,
                    trotter_steps=int(trotter_steps),
                    times=times,
                )
                _write_json(
                    run_dir / "snapshot_metrics.json",
                    {
                        "budget_mode": str(cfg.budget_mode),
                        "method": str(method),
                        "trotter_steps": int(trotter_steps),
                        **dict(snapshot_budget),
                    },
                )
            row = _candidate_row(
                payload,
                method=method,
                trotter_steps=int(trotter_steps),
                run_dir=run_dir,
                cfg=cfg,
                snapshot_budget=snapshot_budget,
            )
            rows.append(dict(row))
            candidates.append(
                {
                    "method": str(method),
                    "trotter_steps": int(trotter_steps),
                    "payload": payload,
                    "row": dict(row),
                }
            )
    pareto_shortlist = _pareto_shortlist(rows)
    _write_csv(cfg.output_csv, rows)
    _write_summary_pdf(
        cfg,
        seed_settings=seed_settings,
        rows=rows,
        candidates=candidates,
        pareto_shortlist=pareto_shortlist,
        run_command=run_command_str,
    )
    payload = {
        "generated_utc": _now_utc(),
        "config": asdict(cfg),
        "seed_settings": dict(seed_settings),
        "rows": rows,
        "pareto_shortlist": pareto_shortlist,
        "artifacts": {
            "output_json": str(cfg.output_json),
            "output_csv": str(cfg.output_csv),
            "output_pdf": str(cfg.output_pdf),
            "run_root": str(cfg.run_root),
        },
    }
    _write_json(cfg.output_json, payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    payload = run_sweep(cfg)
    print(f"summary_json={payload['artifacts']['output_json']}")
    print(f"summary_csv={payload['artifacts']['output_csv']}")
    print(f"summary_pdf={payload['artifacts']['output_pdf']}")


if __name__ == "__main__":
    main()
