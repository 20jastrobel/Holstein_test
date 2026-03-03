#!/usr/bin/env python3
"""Quantum-processor-oriented CFQM vs Suzuki proxy benchmark.

This wrapper script compares propagators using hardware-cost proxies rather
than local wall-clock runtime. It runs the hardcoded Hubbard pipeline as an
external process, then scores each run by:

- final absolute energy error vs a fine piecewise-exact reference
- proxy two-qubit/single-qubit/term-exponential costs

No core propagator code is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import warnings
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_text_page,
    require_matplotlib,
)
from src.quantum.drives_time_potential import build_gaussian_sinusoid_density_drive
from src.quantum.time_propagation.cfqm_schemes import get_cfqm_scheme


_CX_TERM_PROXY_MATH = "cx_proxy_term(p) = 2 * max(weight(p)-1, 0)"
_SQ_TERM_PROXY_MATH = "sq_proxy_term(p) = 2 * xy_count(p) + 1"
_SUZUKI_SWEEP_PROXY_MATH = (
    "sweep(C): term_exp=2*|A(C)|, "
    "cx=2*sum_{p in A(C)} cx_proxy_term(p), "
    "sq=2*sum_{p in A(C)} sq_proxy_term(p)"
)


@dataclass(frozen=True)
class ProxyCost:
    term_exp_count: int
    cx_proxy: int
    sq_proxy: int


@dataclass(frozen=True)
class BenchmarkConfig:
    problem: str
    L: int
    t: float
    u: float
    dv: float
    boundary: str
    ordering: str
    t_final: float
    num_times: int
    methods: tuple[str, ...]
    steps_grid: tuple[int, ...]
    reference_steps: int
    active_coeff_tol: float
    drive_enabled: bool
    drive_A: float
    drive_omega: float
    drive_tbar: float
    drive_phi: float
    drive_t0: float
    drive_pattern: str
    drive_custom_s: str | None
    drive_include_identity: bool
    drive_time_sampling: str
    initial_state_source: str
    vqe_ansatz: str
    vqe_reps: int
    vqe_restarts: int
    vqe_maxiter: int
    vqe_method: str
    adapt_pool: str
    adapt_max_depth: int
    adapt_maxiter: int
    calibrate_transpile: bool
    compare_policy: str
    cost_match_metric: str
    cost_match_tolerance: float
    output_json: Path
    output_csv: Path
    output_pdf: Path
    output_summary: Path
    skip_pdf: bool


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    vals: list[int] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        raise ValueError("Expected at least one integer in CSV list.")
    return tuple(vals)


def _parse_csv_methods(raw: str) -> tuple[str, ...]:
    allowed = {"suzuki2", "cfqm4", "cfqm6"}
    vals: list[str] = []
    for tok in str(raw).split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t not in allowed:
            raise ValueError(f"Unsupported method '{t}'. Allowed: {sorted(allowed)}")
        vals.append(t)
    if not vals:
        raise ValueError("Expected at least one method in CSV list.")
    dedup: list[str] = []
    for m in vals:
        if m not in dedup:
            dedup.append(m)
    return tuple(dedup)


def _parse_custom_s(raw: str | None, n_sites: int) -> list[float] | None:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    if txt.startswith("["):
        arr = json.loads(txt)
        if not isinstance(arr, list):
            raise ValueError("--drive-custom-s JSON must decode to a list.")
        vals = [float(x) for x in arr]
    else:
        vals = [float(x.strip()) for x in txt.split(",") if x.strip()]
    if len(vals) != int(n_sites):
        raise ValueError(
            f"--drive-custom-s length must be L={n_sites}, got {len(vals)} values."
        )
    return vals


def _pauli_weight(label: str) -> int:
    return int(sum(1 for ch in str(label) if ch in {"x", "y", "z"}))


def _pauli_xy_count(label: str) -> int:
    return int(sum(1 for ch in str(label) if ch in {"x", "y"}))


def _cx_proxy_term(label: str) -> int:
    return int(2 * max(_pauli_weight(label) - 1, 0))


def _sq_proxy_term(label: str) -> int:
    return int(2 * _pauli_xy_count(label) + 1)


def _active_labels(
    coeff_map: Mapping[str, complex],
    ordered_labels: list[str],
    tol: float,
) -> list[str]:
    out: list[str] = []
    thr = float(max(0.0, tol))
    for lbl in ordered_labels:
        coeff = complex(coeff_map.get(lbl, 0.0 + 0.0j))
        if abs(coeff) > thr:
            out.append(lbl)
    return out


def _compute_sweep_proxy_cost(active_labels: list[str]) -> ProxyCost:
    term_exp_count = int(2 * len(active_labels))
    cx_proxy = int(2 * sum(_cx_proxy_term(lbl) for lbl in active_labels))
    sq_proxy = int(2 * sum(_sq_proxy_term(lbl) for lbl in active_labels))
    return ProxyCost(term_exp_count=term_exp_count, cx_proxy=cx_proxy, sq_proxy=sq_proxy)


def _sum_cost(lhs: ProxyCost, rhs: ProxyCost) -> ProxyCost:
    return ProxyCost(
        term_exp_count=int(lhs.term_exp_count + rhs.term_exp_count),
        cx_proxy=int(lhs.cx_proxy + rhs.cx_proxy),
        sq_proxy=int(lhs.sq_proxy + rhs.sq_proxy),
    )


def _extract_ordered_static_maps(payload: Mapping[str, Any]) -> tuple[list[str], dict[str, complex]]:
    h_terms = payload.get("hamiltonian", {}).get("coefficients_exyz", [])
    if not isinstance(h_terms, list) or not h_terms:
        raise ValueError("Payload missing hamiltonian.coefficients_exyz list.")

    ordered_labels: list[str] = []
    static_map: dict[str, complex] = {}
    for row in h_terms:
        if not isinstance(row, Mapping):
            continue
        lbl = str(row.get("label_exyz", ""))
        if not lbl:
            continue
        coeff = row.get("coeff", {})
        if not isinstance(coeff, Mapping):
            coeff = {}
        c = complex(float(coeff.get("re", 0.0)), float(coeff.get("im", 0.0)))
        ordered_labels.append(lbl)
        static_map[lbl] = c

    if not ordered_labels:
        raise ValueError("Failed to parse ordered labels from payload.")
    return ordered_labels, static_map


def _build_drive_provider_from_settings(
    settings: Mapping[str, Any],
    L: int,
    nq_total: int,
):
    drive = settings.get("drive")
    if not isinstance(drive, Mapping) or not bool(drive.get("enabled", False)):
        return None

    pattern = str(drive.get("pattern", "staggered"))
    raw_custom = drive.get("custom_s")
    custom_s = _parse_custom_s(None if raw_custom is None else str(raw_custom), int(L))

    drive_obj = build_gaussian_sinusoid_density_drive(
        n_sites=int(L),
        nq_total=int(nq_total),
        indexing=str(settings.get("ordering", "blocked")),
        A=float(drive.get("A", 0.0)),
        omega=float(drive.get("omega", 1.0)),
        tbar=float(drive.get("tbar", 1.0)),
        phi=float(drive.get("phi", 0.0)),
        pattern_mode=pattern,
        custom_weights=custom_s,
        include_identity=bool(drive.get("include_identity", False)),
    )
    return drive_obj.coeff_map_exyz


def _suzuki_sample_time(step_idx: int, dt: float, t0: float, sampling: str) -> float:
    samp = str(sampling).strip().lower()
    if samp == "midpoint":
        return float(t0) + (float(step_idx) + 0.5) * float(dt)
    if samp == "left":
        return float(t0) + float(step_idx) * float(dt)
    if samp == "right":
        return float(t0) + (float(step_idx) + 1.0) * float(dt)
    raise ValueError("time_sampling must be one of {'midpoint','left','right'}")


def _compute_suzuki_proxy_cost(
    *,
    T: float,
    n_steps: int,
    t0: float,
    sampling: str,
    static_coeff_map: Mapping[str, complex],
    drive_provider: Callable[[float], Mapping[str, float]] | None,
    ordered_labels: list[str],
    active_coeff_tol: float,
) -> ProxyCost:
    if int(n_steps) < 1:
        raise ValueError("n_steps must be >= 1")
    if float(T) < 0.0:
        raise ValueError("T must be >= 0")

    dt = float(T) / float(n_steps)
    total = ProxyCost(term_exp_count=0, cx_proxy=0, sq_proxy=0)

    for k in range(int(n_steps)):
        t_sample = _suzuki_sample_time(k, dt, t0, sampling)
        drive_map_raw = {} if drive_provider is None else dict(drive_provider(float(t_sample)))
        merged: dict[str, complex] = {}
        for lbl in ordered_labels:
            merged[lbl] = complex(static_coeff_map.get(lbl, 0.0 + 0.0j)) + complex(
                drive_map_raw.get(lbl, 0.0)
            )
        active = _active_labels(merged, ordered_labels, active_coeff_tol)
        total = _sum_cost(total, _compute_sweep_proxy_cost(active))

    return total


def _build_stage_map_for_proxy(
    *,
    ordered_labels: list[str],
    static_coeff_map: Mapping[str, complex],
    drive_maps: list[Mapping[str, complex]],
    a_row: list[float],
    s_static: float,
    coeff_drop_abs_tol: float,
) -> dict[str, complex]:
    ordered_set = set(ordered_labels)
    stage_map: dict[str, complex] = {}

    for lbl in ordered_labels:
        coeff0 = static_coeff_map.get(lbl)
        if coeff0 is None:
            continue
        scaled = complex(float(s_static)) * complex(coeff0)
        if scaled != 0.0:
            stage_map[lbl] = scaled

    for j, drive_map in enumerate(drive_maps):
        w = float(a_row[j])
        if w == 0.0:
            continue
        for lbl, coeff_drive in drive_map.items():
            if lbl not in ordered_set:
                # Unknown labels are ignored by policy.
                continue
            incr = complex(w) * complex(coeff_drive)
            # Zero-increment insertion guard.
            if incr == 0.0 and lbl not in stage_map:
                continue
            stage_map[lbl] = stage_map.get(lbl, 0.0 + 0.0j) + incr

    drop_tol = float(max(0.0, coeff_drop_abs_tol))
    if drop_tol > 0.0:
        for lbl in list(stage_map):
            if abs(stage_map[lbl]) < drop_tol:
                del stage_map[lbl]

    return stage_map


def _compute_cfqm_proxy_cost(
    *,
    method: str,
    T: float,
    n_steps: int,
    t0: float,
    static_coeff_map: Mapping[str, complex],
    drive_provider: Callable[[float], Mapping[str, float]] | None,
    ordered_labels: list[str],
    active_coeff_tol: float,
    coeff_drop_abs_tol: float,
) -> ProxyCost:
    if int(n_steps) < 1:
        raise ValueError("n_steps must be >= 1")
    if float(T) < 0.0:
        raise ValueError("T must be >= 0")

    scheme = get_cfqm_scheme(str(method))
    c_nodes = [float(x) for x in scheme["c"]]
    a_rows = [[float(v) for v in row] for row in scheme["a"]]
    s_static = [float(v) for v in scheme["s_static"]]

    dt = float(T) / float(n_steps)
    total = ProxyCost(term_exp_count=0, cx_proxy=0, sq_proxy=0)

    for step_idx in range(int(n_steps)):
        t_abs = float(t0) + float(step_idx) * dt

        drive_maps: list[dict[str, complex]] = []
        for c_j in c_nodes:
            t_node = float(t_abs) + float(c_j) * dt
            raw = {} if drive_provider is None else dict(drive_provider(float(t_node)))
            drive_maps.append({str(k): complex(v) for k, v in raw.items()})

        for k, a_row in enumerate(a_rows):
            stage_map = _build_stage_map_for_proxy(
                ordered_labels=ordered_labels,
                static_coeff_map=static_coeff_map,
                drive_maps=drive_maps,
                a_row=a_row,
                s_static=float(s_static[k]),
                coeff_drop_abs_tol=float(coeff_drop_abs_tol),
            )
            active = _active_labels(stage_map, ordered_labels, active_coeff_tol)
            total = _sum_cost(total, _compute_sweep_proxy_cost(active))

    return total


def _run_pipeline_subprocess(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = float(time.perf_counter() - started)
    if proc.returncode != 0:
        tail_out = (proc.stdout or "")[-2500:]
        tail_err = (proc.stderr or "")[-2500:]
        raise RuntimeError(
            "Pipeline run failed. "
            f"cmd={' '.join(cmd)}\n"
            f"returncode={proc.returncode}\n"
            f"stdout_tail=\n{tail_out}\n"
            f"stderr_tail=\n{tail_err}\n"
        )

    output_json: Path | None = None
    for i, tok in enumerate(cmd):
        if tok == "--output-json" and (i + 1) < len(cmd):
            output_json = Path(cmd[i + 1])
            break
    if output_json is None:
        raise RuntimeError("Internal error: --output-json missing in command.")

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    payload["_run_runtime_s"] = elapsed
    payload["_run_cmd"] = cmd
    return payload


def _extract_error_metrics(
    payload: Mapping[str, Any],
    ref_payload: Mapping[str, Any],
) -> tuple[float, float]:
    traj = payload.get("trajectory", [])
    ref = ref_payload.get("trajectory", [])
    if not isinstance(traj, list) or not isinstance(ref, list) or not traj or not ref:
        raise ValueError("Both payload and reference must include non-empty trajectory lists.")
    if len(traj) != len(ref):
        raise ValueError(
            f"Trajectory length mismatch: run={len(traj)} reference={len(ref)}"
        )

    abs_deltas: list[float] = []
    for row, row_ref in zip(traj, ref):
        e = float(row["energy_total_trotter"])
        e_ref = float(row_ref["energy_total_trotter"])
        abs_deltas.append(abs(e - e_ref))

    return float(abs_deltas[-1]), float(max(abs_deltas))


def _format_metric(value: float) -> str:
    if value == 0.0:
        return "0.000e+00"
    return f"{float(value):.6e}"


def _extract_energy_series(payload: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    traj = payload.get("trajectory")
    if not isinstance(traj, list) or not traj:
        return ([], [])
    times: list[float] = []
    energies: list[float] = []
    for row in traj:
        if not isinstance(row, Mapping):
            continue
        t_raw = row.get("time")
        e_raw = row.get("energy_total_trotter")
        if t_raw is None or e_raw is None:
            continue
        times.append(float(t_raw))
        energies.append(float(e_raw))
    return times, energies


def _align_to_reference(
    times: list[float],
    energies: list[float],
    ref_times: list[float],
    ref_energies: list[float],
) -> list[float]:
    if not times or not ref_times:
        return []
    if len(times) == len(ref_times) and all(abs(a - b) <= 1e-12 for a, b in zip(times, ref_times)):
        return list(ref_energies)
    if not ref_times:
        return []
    out: list[float] = []
    for t in times:
        idx = min(range(len(ref_times)), key=lambda j: abs(ref_times[j] - t))
        out.append(float(ref_energies[idx]))
    return out


def _write_energy_plots(
    *,
    output_pdf: Path,
    cfg: BenchmarkConfig | Any,
    reference_series: tuple[list[float], list[float]],
    run_series: list[tuple[str, int, list[float], list[float]]],
    rows: list[dict[str, Any]],
    best_summary: dict[str, Any],
) -> None:
    require_matplotlib()
    plt = get_plt()
    pdf = get_PdfPages()

    ref_times, ref_energies = reference_series
    if not ref_times or not ref_energies:
        return

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with pdf(str(output_pdf)) as pdf_out:
        render_command_page(
            pdf_out,
            command=current_command_string(),
            script_name="pipelines/exact_bench/cfqm_vs_suzuki_qproc_proxy_benchmark.py",
            extra_header_lines=[
                f"Benchmark problem: L={cfg.L}, t={cfg.t}, U={cfg.u}, dv={cfg.dv}",
                f"Methods: {','.join(cfg.methods)}",
                f"Steps grid: {','.join(str(s) for s in cfg.steps_grid)}",
            ],
        )

        fig = plt.figure(figsize=(12, 6.8))
        ax = fig.add_subplot(111)
        col_labels = ["meth", "S", "final_err", "max_err", "cx", "term_exp", "runtime_s"]
        table_rows: list[list[str]] = []
        for method, steps, final_err, max_err, cx, terms, runtime in sorted(
            [
                (
                    r.get("method"),
                    int(r.get("trotter_steps", 0)),
                    float(r.get("final_abs_energy_error", 0.0)),
                    float(r.get("max_abs_energy_error", 0.0)),
                    int(r.get("cx_proxy_total", 0)),
                    int(r.get("term_exp_count_total", 0)),
                    float(r.get("run_runtime_s", float("nan"))),
                )
                for r in rows
                if isinstance(r, Mapping)
            ],
            key=lambda x: (str(x[0]), int(x[1])),
        ):
            table_rows.append(
                [
                    str(method),
                    f"{steps:>3d}",
                    _format_metric(final_err),
                    _format_metric(max_err),
                    f"{cx:>6d}",
                    f"{terms:>7d}",
                    f"{runtime:>7.2f}",
                ]
            )
        render_compact_table(
            ax,
            title="Metrics (vs piecewise_exact reference)",
            col_labels=col_labels,
            rows=table_rows,
            fontsize=8,
        )
        pdf_out.savefig(fig)
        plt.close(fig)

        cost_match = best_summary.get("cost_match", {})
        if isinstance(cost_match, Mapping) and bool(cost_match.get("enabled")):
            pairs = cost_match.get("pairs", [])
            metric_name = str(cost_match.get("metric", ""))
            tol = float(cost_match.get("tolerance", 0.0))
            if isinstance(pairs, list) and pairs:
                fig = plt.figure(figsize=(12, 7))
                ax = fig.add_subplot(111)
                headers = [
                    "metric",
                    "target",
                    "meth",
                    "S",
                    "match",
                    "err",
                    "delta",
                    "exact",
                ]
                eq_rows: list[list[str]] = []
                for pair in pairs:
                    if not isinstance(pair, Mapping):
                        continue
                    target = float(pair.get("target_metric", 0.0))
                    matched_rows = pair.get("matched_rows", [])
                    if not isinstance(matched_rows, list):
                        matched_rows = []
                    for m in matched_rows:
                        if not isinstance(m, Mapping):
                            continue
                        eq_rows.append(
                            [
                                str(metric_name),
                                _format_metric(target),
                                str(m.get("method", "")),
                                f"{int(m.get('matched_trotter_steps', 0)):>4d}",
                                _format_metric(float(m.get("matched_metric", 0.0))),
                                _format_metric(float(m.get("final_abs_energy_error", 0.0))),
                                _format_metric(float(m.get("metric_delta", 0.0))),
                                "Y" if bool(m.get("exact_match")) else "N",
                            ]
                        )
                if eq_rows:
                    render_compact_table(
                        ax,
                        title=f"Equal-cost rows (CX / term-exp matched, metric={metric_name}, tolerance={tol:g})",
                        col_labels=headers,
                        rows=eq_rows,
                        fontsize=7,
                    )
                    pdf_out.savefig(fig)
                plt.close(fig)

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        ax.plot(ref_times, ref_energies, "--", color="#2ca02c", linewidth=2.0, label="exact (piecewise)")
        for method, steps, times, energies in run_series:
            if not times or not energies:
                continue
            label = f"{method} (S={steps})"
            ax.plot(
                times,
                energies,
                label=label,
                linewidth=1.3,
                alpha=0.9,
            )
        ax.set_title("Energy evolution vs exact")
        ax.set_xlabel("time")
        ax.set_ylabel("total_trotter energy")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        pdf_out.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        for method, steps, times, energies in run_series:
            if not times or not energies:
                continue
            aligned_ref = _align_to_reference(times, energies, ref_times, ref_energies)
            abs_err = [abs(e_m - e_ref) for e_m, e_ref in zip(energies, aligned_ref)]
            label = f"{method} (S={steps})"
            ax.plot(
                times,
                abs_err,
                label=label,
                linewidth=1.2,
                alpha=0.9,
            )
        ax.set_title("Absolute energy error vs exact")
        ax.set_ylabel(r"$|E_{\mathrm{method}} - E_{\mathrm{exact}}|$")
        ax.set_xlabel("time")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        pdf_out.savefig(fig)
        plt.close(fig)

        summary_lines = [
            "CFQM-vs-Suzuki proxy benchmark",
            "",
            f"best_by_error_then_cx={best_summary.get('ranking_by_error_then_cx', [])[:3]}",
        ]
        render_text_page(
            pdf_out,
            summary_lines,
            fontsize=9,
            line_spacing=0.03,
        )


def _format_markdown_table_row(
    method: str,
    steps: int,
    final_err: float,
    max_err: float,
    cx: int,
    terms: int,
    runtime: float,
) -> str:
    return (
        f"| {method:<6} | {steps:>3} | {_format_metric(final_err):>10} "
        f"| {_format_metric(max_err):>10} | {cx:>6} | {terms:>7} | {runtime:>6.2f} |"
    )


def _write_markdown_summary(
    *,
    payload: Mapping[str, Any],
    output_path: Path,
) -> None:
    settings = payload.get("settings")
    reference = payload.get("reference", {})
    rows = payload.get("runs", [])
    summary = payload.get("summary", {})
    if not isinstance(settings, Mapping) or not isinstance(rows, list):
        raise ValueError("Invalid payload for summary generation.")

    out_lines: list[str] = []
    out_lines.append("# CFQM vs Suzuki proxy benchmark")
    out_lines.append("")
    out_lines.append("## Run metadata")
    out_lines.append("")
    lines_meta = [
        f"problem={settings.get('problem')}",
        f"L={settings.get('L')}",
        f"t={settings.get('t')}",
        f"u={settings.get('u')}",
        f"dv={settings.get('dv')}",
        f"t_final={settings.get('t_final')}",
        f"reference_steps={reference.get('trotter_steps')}",
        f"methods={','.join(str(m) for m in settings.get('methods', []))}",
        f"steps_grid={','.join(str(s) for s in settings.get('steps_grid', []))}",
    ]
    for ln in lines_meta:
        out_lines.append(f"- {ln}")
    out_lines.append("")

    out_lines.append("## Metrics (vs piecewise_exact reference)")
    out_lines.append("")
    out_lines.append("- `S` is the macro-step count (`trotter_steps`), not a cost axis.")
    out_lines.append("| meth | S | final_err | max_err | cx | term_exp | runtime_s |")
    out_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        out_lines.append(
            _format_markdown_table_row(
                method=str(row.get("method", "")),
                steps=int(row.get("trotter_steps", 0)),
                final_err=float(row.get("final_abs_energy_error", 0.0)),
                max_err=float(row.get("max_abs_energy_error", 0.0)),
                cx=int(row.get("cx_proxy_total", 0)),
                terms=int(row.get("term_exp_count_total", 0)),
                runtime=float(row.get("run_runtime_s", float("nan"))),
            )
        )
    out_lines.append("")

    out_lines.append("## Ordering")
    out_lines.append("")
    out_lines.append("- Pareto front is reported by increasing CX cost; each entry keeps best-so-far error.")
    pareto = summary.get("pareto_front")
    if isinstance(pareto, list):
        for item in pareto[:6]:
            if not isinstance(item, Mapping):
                continue
            out_lines.append(
                f"- {item.get('method')}-{item.get('trotter_steps')} "
                f"err={_format_metric(float(item.get('final_abs_energy_error', 0.0)))} "
                f"cx={int(item.get('cx_proxy_total', 0))}"
            )

    out_lines.append("")
    cost_match = summary.get("cost_match")
    if isinstance(cost_match, Mapping) and cost_match.get("enabled", False):
        out_lines.append("## Equal-cost comparison")
        out_lines.append("")
        out_lines.append(
            f"- Metric: {cost_match.get('metric')} (tolerance={cost_match.get('tolerance')})"
        )
        pairs = cost_match.get("pairs")
        if isinstance(pairs, list) and pairs:
            for idx, pair in enumerate(pairs, start=1):
                if not isinstance(pair, Mapping):
                    continue
                metric = pair.get("metric", "")
                target = pair.get("target_metric", "")
                out_lines.append(
                    f"- target[{idx}] {metric}={_format_metric(float(target) if isinstance(target, int | float) else 0.0)}"
                )
                out_lines.append("| meth | matched_S | matched_metric | final_err | delta | exact_match |")
                out_lines.append("|---|---:|---:|---:|---:|---|")
                matched_rows = pair.get("matched_rows", [])
                if not isinstance(matched_rows, list):
                    matched_rows = []
                for m in matched_rows:
                    if not isinstance(m, Mapping):
                        continue
                    out_lines.append(
                        "| "
                        f"{str(m.get('method', '')):<4} | "
                        f"{int(m.get('matched_trotter_steps', 0)):>9} | "
                        f"{_format_metric(float(m.get('matched_metric', 0.0)))} | "
                        f"{_format_metric(float(m.get('final_abs_energy_error', 0.0)))} | "
                        f"{_format_metric(float(m.get('metric_delta', 0.0)))} | "
                        f"{bool(m.get('exact_match', False))} |"
                    )
            out_lines.append("")
        else:
            out_lines.append("- No equal-cost pair rows were generated.")
        out_lines.append("")

    out_lines.append("## Note")
    out_lines.append("")
    wrapped = textwrap.wrap(
        "Formatting rule: compact markdown rows use fixed-width numeric fields "
        "to prevent visual spillover in rendered artifacts.",
        width=64,
    )
    out_lines.extend(f"- {w}" for w in wrapped)

    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _replot_from_output_json(*, artifact_json: Path, output_pdf: Path) -> None:
    if not artifact_json.exists():
        raise FileNotFoundError(f"Missing benchmark JSON artifact: {artifact_json}")

    payload = json.loads(artifact_json.read_text(encoding="utf-8"))
    settings = payload.get("settings")
    runs = payload.get("runs")
    summary = payload.get("summary", {})
    if not isinstance(settings, Mapping) or not isinstance(runs, list):
        raise ValueError("Invalid benchmark JSON: missing 'settings' or 'runs'.")

    out_dir = artifact_json.parent
    ref_payload_path = out_dir / "_reference_piecewise_exact.json"
    if not ref_payload_path.exists():
        raise FileNotFoundError(
            "Missing sidecar reference file: "
            f"{ref_payload_path.name}. Re-run benchmark with --output-pdf."
        )
    ref_payload = json.loads(ref_payload_path.read_text(encoding="utf-8"))
    ref_series = _extract_energy_series(ref_payload)
    if not ref_series[0] or not ref_series[1]:
        raise ValueError(
            "Reference trajectory missing in _reference_piecewise_exact.json; "
            "cannot build energy plots."
        )

    plot_series: list[tuple[str, int, list[float], list[float]]] = []
    for row in runs:
        if not isinstance(row, Mapping):
            continue
        method = str(row.get("method", ""))
        steps_raw = row.get("trotter_steps")
        if not method or steps_raw is None:
            continue
        steps = int(steps_raw)
        run_payload_path = out_dir / f"_run_{method}_S{steps}.json"
        if not run_payload_path.exists():
            continue
        run_payload = json.loads(run_payload_path.read_text(encoding="utf-8"))
        times, energies = _extract_energy_series(run_payload)
        plot_series.append((method, steps, times, energies))

    if not plot_series:
        raise ValueError("No run trajectory artifacts found for replot mode.")

    cfg = SimpleNamespace(
        L=int(settings.get("L", 0)),
        t=float(settings.get("t", 0.0)),
        u=float(settings.get("u", 0.0)),
        dv=float(settings.get("dv", 0.0)),
        methods=tuple(str(m) for m in settings.get("methods", [])),
        steps_grid=tuple(int(s) for s in settings.get("steps_grid", [])),
    )
    _write_energy_plots(
        output_pdf=output_pdf,
        cfg=cfg,
        reference_series=ref_series,
        run_series=plot_series,
        rows=[row for row in runs if isinstance(row, Mapping)],
        best_summary=summary if isinstance(summary, Mapping) else {},
    )


def _default_artifact_paths() -> tuple[Path, Path, Path, Path, Path]:
    out_dir = REPO_ROOT / "artifacts" / "cfqm_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_json = out_dir / "cfqm_vs_suzuki_proxy_runs.json"
    runs_csv = out_dir / "cfqm_vs_suzuki_proxy_runs.csv"
    summary_json = out_dir / "cfqm_vs_suzuki_proxy_summary.json"
    plots_pdf = out_dir / "cfqm_vs_suzuki_proxy_energy_comparison.pdf"
    summary_md = out_dir / "cfqm_vs_suzuki_proxy_summary.md"
    return runs_json, runs_csv, summary_json, plots_pdf, summary_md


def _build_pipeline_cmd(
    *,
    cfg: BenchmarkConfig,
    propagator: str,
    trotter_steps: int,
    output_json_path: Path,
    cfqm_stage_exp: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "pipelines/hardcoded/hubbard_pipeline.py",
        "--problem",
        str(cfg.problem),
        "--L",
        str(int(cfg.L)),
        "--t",
        str(float(cfg.t)),
        "--u",
        str(float(cfg.u)),
        "--dv",
        str(float(cfg.dv)),
        "--boundary",
        str(cfg.boundary),
        "--ordering",
        str(cfg.ordering),
        "--t-final",
        str(float(cfg.t_final)),
        "--num-times",
        str(int(cfg.num_times)),
        "--suzuki-order",
        "2",
        "--trotter-steps",
        str(int(trotter_steps)),
        "--propagator",
        str(propagator),
        "--initial-state-source",
        str(cfg.initial_state_source),
        "--vqe-ansatz",
        str(cfg.vqe_ansatz),
        "--vqe-reps",
        str(int(cfg.vqe_reps)),
        "--vqe-restarts",
        str(int(cfg.vqe_restarts)),
        "--vqe-maxiter",
        str(int(cfg.vqe_maxiter)),
        "--vqe-method",
        str(cfg.vqe_method),
        "--adapt-pool",
        str(cfg.adapt_pool),
        "--adapt-max-depth",
        str(int(cfg.adapt_max_depth)),
        "--adapt-maxiter",
        str(int(cfg.adapt_maxiter)),
        "--skip-qpe",
        "--output-json",
        str(output_json_path),
    ]

    if bool(cfg.skip_pdf):
        cmd.append("--skip-pdf")

    if bool(cfg.drive_enabled):
        cmd.extend(
            [
                "--enable-drive",
                "--drive-A",
                str(float(cfg.drive_A)),
                "--drive-omega",
                str(float(cfg.drive_omega)),
                "--drive-tbar",
                str(float(cfg.drive_tbar)),
                "--drive-phi",
                str(float(cfg.drive_phi)),
                "--drive-t0",
                str(float(cfg.drive_t0)),
                "--drive-pattern",
                str(cfg.drive_pattern),
                "--drive-time-sampling",
                str(cfg.drive_time_sampling),
                "--exact-steps-multiplier",
                "1",
            ]
        )
        if cfg.drive_include_identity:
            cmd.append("--drive-include-identity")
        if cfg.drive_pattern == "custom" and cfg.drive_custom_s is not None:
            cmd.extend(["--drive-custom-s", str(cfg.drive_custom_s)])

    if cfqm_stage_exp is not None:
        cmd.extend(
            [
                "--cfqm-stage-exp",
                str(cfqm_stage_exp),
                "--cfqm-coeff-drop-abs-tol",
                "0.0",
            ]
        )

    return cmd


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write CSV: no rows.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_cost_matched_pairs(
    rows: list[dict[str, Any]],
    metric: str,
    tolerance: float,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    metric_key = str(metric)
    tol = float(max(0.0, tolerance))

    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        method = str(row.get("method", ""))
        if not method:
            continue
        if metric_key not in row:
            continue
        by_method.setdefault(method, []).append(row)
    if not by_method:
        return []

    def _metric_value(candidate: Mapping[str, Any]) -> float:
        return float(candidate[metric_key])

    target_values = sorted({float(_metric_value(r)) for rs in by_method.values() for r in rs})
    if not target_values:
        return []

    out: list[dict[str, Any]] = []
    for target in target_values:
        matches: list[dict[str, Any]] = []
        for method in sorted(by_method):
            rows_for_method = by_method[method]
            if not rows_for_method:
                continue
            best = min(
                rows_for_method,
                key=lambda item: abs(float(item[metric_key]) - target),
            )
            best_metric = float(best[metric_key])
            delta = abs(best_metric - target)
            matches.append(
                {
                    "method": method,
                    "target_metric": target,
                    "metric_delta": delta,
                    "exact_match": delta <= tol,
                    "matched_trotter_steps": int(best.get("trotter_steps", 0)),
                    "matched_metric": best_metric,
                    "final_abs_energy_error": float(best.get("final_abs_energy_error", float("nan"))),
                    "run_runtime_s": float(best.get("run_runtime_s", float("nan"))),
                }
            )
        out.append(
            {
                "metric": metric_key,
                "target_metric": target,
                "tolerance": tol,
                "matched_rows": matches,
            }
        )
    return out


def _summarize_pareto(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "pareto_front": [],
            "ranking_by_error_then_cx": [],
            "best_by_budget": [],
        }

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            float(r["cx_proxy_total"]),
            float(r["final_abs_energy_error"]),
            str(r["method"]),
            int(r["trotter_steps"]),
        ),
    )

    pareto: list[dict[str, Any]] = []
    best_err = float("inf")
    for row in sorted_rows:
        err = float(row["final_abs_energy_error"])
        if err <= best_err + 1e-18:
            pareto.append(row)
            best_err = min(best_err, err)

    ranking = sorted(
        rows,
        key=lambda r: (
            float(r["final_abs_energy_error"]),
            float(r["cx_proxy_total"]),
            float(r["term_exp_count_total"]),
        ),
    )

    budgets = sorted({int(r["cx_proxy_total"]) for r in rows})
    best_by_budget: list[dict[str, Any]] = []
    for budget in budgets:
        feas = [r for r in rows if int(r["cx_proxy_total"]) <= budget]
        if not feas:
            continue
        best = min(
            feas,
            key=lambda r: (
                float(r["final_abs_energy_error"]),
                float(r["cx_proxy_total"]),
            ),
        )
        best_by_budget.append(
            {
                "cx_budget": int(budget),
                "best_method": str(best["method"]),
                "best_steps": int(best["trotter_steps"]),
                "best_final_abs_energy_error": float(best["final_abs_energy_error"]),
                "best_cx_proxy_total": int(best["cx_proxy_total"]),
            }
        )

    return {
        "pareto_front": [
            {
                "method": str(r["method"]),
                "trotter_steps": int(r["trotter_steps"]),
                "cx_proxy_total": int(r["cx_proxy_total"]),
                "term_exp_count_total": int(r["term_exp_count_total"]),
                "final_abs_energy_error": float(r["final_abs_energy_error"]),
            }
            for r in pareto
        ],
        "ranking_by_error_then_cx": [
            {
                "method": str(r["method"]),
                "trotter_steps": int(r["trotter_steps"]),
                "cx_proxy_total": int(r["cx_proxy_total"]),
                "term_exp_count_total": int(r["term_exp_count_total"]),
                "final_abs_energy_error": float(r["final_abs_energy_error"]),
            }
            for r in ranking
        ],
        "best_by_budget": best_by_budget,
    }


def _maybe_calibrate_transpile(
    *,
    rows: list[dict[str, Any]],
    cfg: BenchmarkConfig,
) -> dict[str, Any]:
    if not bool(cfg.calibrate_transpile):
        return {
            "enabled": False,
            "status": "skipped",
            "reason": "--calibrate-transpile not enabled",
            "samples": [],
        }

    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.synthesis import SuzukiTrotter
        from qiskit_ibm_runtime import fake_provider
    except Exception as exc:
        warnings.warn(
            "Calibration transpile skipped: Qiskit fake-backend stack unavailable "
            f"({exc}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "enabled": True,
            "status": "skipped",
            "reason": f"import_failed: {exc}",
            "samples": [],
        }

    backend_cls = getattr(fake_provider, "FakeManilaV2", None)
    if backend_cls is None:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "FakeManilaV2 unavailable",
            "samples": [],
        }

    backend = backend_cls()
    out_samples: list[dict[str, Any]] = []
    for row in rows:
        # Calibration uses a representative static-only evolution gate at T.
        static = row.get("static_coeff_map", {})
        ordered = row.get("ordered_labels", [])
        if not isinstance(static, Mapping) or not isinstance(ordered, list) or not ordered:
            continue

        terms: list[tuple[str, complex]] = []
        for lbl in ordered:
            c = complex(static.get(lbl, 0.0 + 0.0j))
            if abs(c) <= cfg.active_coeff_tol:
                continue
            terms.append((str(lbl).replace("e", "I").upper(), c))

        if not terms:
            continue

        qop = SparsePauliOp.from_list(terms)
        nq = int(qop.num_qubits)
        qc = QuantumCircuit(nq)
        qc.append(
            PauliEvolutionGate(
                qop,
                time=float(cfg.t_final),
                synthesis=SuzukiTrotter(order=2, reps=1, preserve_order=True),
            ),
            list(range(nq)),
        )
        tqc = transpile(qc, backend=backend, optimization_level=1)
        counts = tqc.count_ops()
        out_samples.append(
            {
                "method": str(row["method"]),
                "trotter_steps": int(row["trotter_steps"]),
                "transpiled_depth": int(tqc.depth()),
                "transpiled_size": int(tqc.size()),
                "transpiled_2q_count": int(
                    sum(int(v) for k, v in counts.items() if str(k) in {"cx", "ecr", "cz"})
                ),
                "proxy_cx_total": int(row["cx_proxy_total"]),
            }
        )

    return {
        "enabled": True,
        "status": "ok",
        "backend": "FakeManilaV2",
        "samples": out_samples,
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    run_pipeline: Callable[[list[str], Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run_fn = run_pipeline
    if run_fn is None:
        run_fn = lambda cmd, cwd: _run_pipeline_subprocess(cmd, cwd=cwd)

    out_dir = config.output_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_run_json = out_dir / "_reference_piecewise_exact.json"
    ref_cmd = _build_pipeline_cmd(
        cfg=config,
        propagator="piecewise_exact",
        trotter_steps=int(config.reference_steps),
        output_json_path=ref_run_json,
        cfqm_stage_exp=None,
    )
    ref_payload = run_fn(ref_cmd, REPO_ROOT)

    ref_final = float(ref_payload["trajectory"][-1]["energy_total_trotter"])
    ref_series = _extract_energy_series(ref_payload)

    rows: list[dict[str, Any]] = []
    plot_series: list[tuple[str, int, list[float], list[float]]] = []
    for method in config.methods:
        for steps in config.steps_grid:
            run_json = out_dir / f"_run_{method}_S{int(steps)}.json"
            cfqm_stage_exp = "pauli_suzuki2" if method in {"cfqm4", "cfqm6"} else None
            cmd = _build_pipeline_cmd(
                cfg=config,
                propagator=str(method),
                trotter_steps=int(steps),
                output_json_path=run_json,
                cfqm_stage_exp=cfqm_stage_exp,
            )
            payload = run_fn(cmd, REPO_ROOT)

            final_err, max_err = _extract_error_metrics(payload, ref_payload)
            ordered_labels, static_map = _extract_ordered_static_maps(payload)
            settings = payload.get("settings", {})
            if not isinstance(settings, Mapping):
                settings = {}

            drive_provider = _build_drive_provider_from_settings(
                settings,
                int(config.L),
                len(ordered_labels[0]),
            )
            t0 = float(settings.get("drive", {}).get("t0", 0.0)) if isinstance(settings.get("drive"), Mapping) else 0.0
            sampling = (
                str(settings.get("drive", {}).get("time_sampling", config.drive_time_sampling))
                if isinstance(settings.get("drive"), Mapping)
                else str(config.drive_time_sampling)
            )

            if method == "suzuki2":
                cost = _compute_suzuki_proxy_cost(
                    T=float(config.t_final),
                    n_steps=int(steps),
                    t0=float(t0),
                    sampling=str(sampling),
                    static_coeff_map=static_map,
                    drive_provider=drive_provider,
                    ordered_labels=ordered_labels,
                    active_coeff_tol=float(config.active_coeff_tol),
                )
            else:
                cost = _compute_cfqm_proxy_cost(
                    method=str(method),
                    T=float(config.t_final),
                    n_steps=int(steps),
                    t0=float(t0),
                    static_coeff_map=static_map,
                    drive_provider=drive_provider,
                    ordered_labels=ordered_labels,
                    active_coeff_tol=float(config.active_coeff_tol),
                    coeff_drop_abs_tol=0.0,
                )

            row = {
                "method": str(method),
                "trotter_steps": int(steps),
                "final_abs_energy_error": float(final_err),
                "max_abs_energy_error": float(max_err),
                "final_energy_total_trotter": float(payload["trajectory"][-1]["energy_total_trotter"]),
                "reference_final_energy_total_trotter": float(ref_final),
                "term_exp_count_total": int(cost.term_exp_count),
                "cx_proxy_total": int(cost.cx_proxy),
                "sq_proxy_total": int(cost.sq_proxy),
                "run_runtime_s": float(payload.get("_run_runtime_s", float("nan"))),
                "propagator": str(settings.get("propagator", "suzuki2")),
                "drive_enabled": bool(isinstance(settings.get("drive"), Mapping) and settings.get("drive", {}).get("enabled", False)),
                # Include internals for optional calibration; stripped in summary export.
                "ordered_labels": ordered_labels,
                "static_coeff_map": {lbl: static_map[lbl] for lbl in ordered_labels},
            }
            rows.append(row)
            method_times, method_energies = _extract_energy_series(payload)
            plot_series.append((str(method), int(steps), method_times, method_energies))

    public_rows: list[dict[str, Any]] = []
    for row in rows:
        cleaned = {k: v for k, v in row.items() if k not in {"ordered_labels", "static_coeff_map"}}
        public_rows.append(cleaned)

    pareto_summary = _summarize_pareto(public_rows)
    if str(config.compare_policy).strip().lower() == "cost_match":
        metric = str(config.cost_match_metric)
        cost_match = {
            "enabled": True,
            "metric": metric,
            "tolerance": float(config.cost_match_tolerance),
            "pairs": _build_cost_matched_pairs(
                rows=public_rows,
                metric=metric,
                tolerance=float(config.cost_match_tolerance),
            ),
        }
    else:
        cost_match = {
            "enabled": False,
            "metric": str(config.cost_match_metric),
            "tolerance": float(config.cost_match_tolerance),
            "pairs": [],
        }

    calibration = _maybe_calibrate_transpile(rows=rows, cfg=config)

    payload_out = {
        "schema": "cfqm_qproc_proxy_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "problem": str(config.problem),
            "L": int(config.L),
            "t": float(config.t),
            "u": float(config.u),
            "dv": float(config.dv),
            "boundary": str(config.boundary),
            "ordering": str(config.ordering),
            "t_final": float(config.t_final),
            "num_times": int(config.num_times),
            "methods": [str(m) for m in config.methods],
            "steps_grid": [int(s) for s in config.steps_grid],
            "reference_steps": int(config.reference_steps),
            "active_coeff_tol": float(config.active_coeff_tol),
            "drive_enabled": bool(config.drive_enabled),
            "drive": {
                "A": float(config.drive_A),
                "omega": float(config.drive_omega),
                "tbar": float(config.drive_tbar),
                "phi": float(config.drive_phi),
                "t0": float(config.drive_t0),
                "pattern": str(config.drive_pattern),
                "custom_s": config.drive_custom_s,
                "include_identity": bool(config.drive_include_identity),
                "time_sampling": str(config.drive_time_sampling),
            },
            "cfqm_stage_exp_profile": "pauli_suzuki2",
            "note": (
                "CFQM runs use pauli_suzuki2 stage exponentials for hardware-comparable "
                "termwise gate proxy accounting."
            ),
            "compare_policy": str(config.compare_policy),
            "cost_match_metric": str(config.cost_match_metric),
            "cost_match_tolerance": float(config.cost_match_tolerance),
            "output_pdf": str(config.output_pdf),
            "skip_pdf": bool(config.skip_pdf),
        },
        "reference": {
            "propagator": "piecewise_exact",
            "trotter_steps": int(config.reference_steps),
            "final_energy_total_trotter": float(ref_final),
        },
        "runs": public_rows,
        "summary": {
            **pareto_summary,
            "cost_match": cost_match,
        },
        "calibration": calibration,
    }

    config.output_json.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    _write_csv(public_rows, config.output_csv)

    summary_path = config.output_json.parent / "cfqm_vs_suzuki_proxy_summary.json"
    summary_path.write_text(json.dumps(payload_out["summary"], indent=2), encoding="utf-8")
    _write_markdown_summary(payload=payload_out, output_path=config.output_summary)

    if not bool(config.skip_pdf):
        _write_energy_plots(
            output_pdf=config.output_pdf,
            cfg=config,
            reference_series=ref_series,
            run_series=plot_series,
            rows=public_rows,
            best_summary=payload_out["summary"],
        )
        print(f"WROTE {config.output_pdf}")

    return {
        "payload": payload_out,
        "output_json": config.output_json,
        "output_csv": config.output_csv,
        "summary_json": summary_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_runs_json, default_runs_csv, _default_summary, default_output_pdf, default_output_summary = (
        _default_artifact_paths()
    )

    p = argparse.ArgumentParser(
        description="Quantum-processor-oriented CFQM vs Suzuki proxy benchmark (wrapper-level)."
    )
    p.add_argument("--problem", choices=["hubbard"], default="hubbard")
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=4.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")

    p.add_argument("--t-final", type=float, default=10.0)
    p.add_argument("--num-times", type=int, default=201)

    p.add_argument("--steps-grid", type=str, default="64,128,256,512")
    p.add_argument("--methods", type=str, default="suzuki2,cfqm4,cfqm6")
    p.add_argument("--reference-steps", type=int, default=2048)
    p.add_argument("--active-coeff-tol", type=float, default=1e-14)

    p.set_defaults(drive_enabled=True)
    p.add_argument("--drive-enabled", dest="drive_enabled", action="store_true")
    p.add_argument("--no-drive-enabled", dest="drive_enabled", action="store_false")

    p.add_argument("--drive-A", type=float, default=0.2)
    p.add_argument("--drive-omega", type=float, default=1.0)
    p.add_argument("--drive-tbar", type=float, default=1.0)
    p.add_argument("--drive-phi", type=float, default=0.0)
    p.add_argument("--drive-t0", type=float, default=0.0)
    p.add_argument("--drive-pattern", choices=["staggered", "dimer_bias", "custom"], default="staggered")
    p.add_argument("--drive-custom-s", type=str, default=None)
    p.add_argument("--drive-time-sampling", choices=["midpoint", "left", "right"], default="midpoint")
    p.add_argument("--drive-include-identity", action="store_true")

    p.add_argument("--initial-state-source", choices=["exact", "vqe", "hf", "adapt_json"], default="exact")
    p.add_argument("--vqe-ansatz", choices=["uccsd", "hva", "hh_hva", "hh_hva_tw", "hh_hva_ptw"], default="uccsd")
    p.add_argument("--vqe-reps", type=int, default=2)
    p.add_argument("--vqe-restarts", type=int, default=2)
    p.add_argument("--vqe-maxiter", type=int, default=600)
    p.add_argument("--vqe-method", choices=["SLSQP", "COBYLA", "L-BFGS-B", "Powell", "Nelder-Mead"], default="COBYLA")
    p.add_argument("--adapt-pool", type=str, default="uccsd")
    p.add_argument("--adapt-max-depth", type=int, default=2)
    p.add_argument("--adapt-maxiter", type=int, default=30)

    p.add_argument("--output-json", type=Path, default=default_runs_json)
    p.add_argument("--output-csv", type=Path, default=default_runs_csv)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument("--output-summary", type=Path, default=None)
    p.add_argument("--skip-pdf", action="store_true", default=True)
    p.add_argument("--replot-only", action="store_true", default=False)
    p.add_argument(
        "--compare-policy",
        type=str,
        choices=["sweep_only", "cost_match"],
        default="sweep_only",
    )
    p.add_argument(
        "--cost-match-metric",
        type=str,
        choices=["cx_proxy_total", "term_exp_count_total"],
        default="cx_proxy_total",
    )
    p.add_argument("--cost-match-tolerance", type=float, default=0.0)

    p.add_argument("--calibrate-transpile", action="store_true")
    return p.parse_args(argv)


def _to_config(args: argparse.Namespace) -> BenchmarkConfig:
    default_runs_json, default_runs_csv, _default_summary, default_output_pdf, default_output_summary = (
        _default_artifact_paths()
    )
    steps_grid = _parse_csv_ints(str(args.steps_grid))
    methods = _parse_csv_methods(str(args.methods))

    if int(args.L) != 2:
        raise ValueError("v1 benchmark is locked to L=2 for reproducible low-risk comparison.")
    if str(args.problem).strip().lower() != "hubbard":
        raise ValueError("v1 benchmark supports --problem hubbard only.")
    if int(args.reference_steps) < 1:
        raise ValueError("--reference-steps must be >= 1")

    if str(args.drive_pattern) == "custom" and args.drive_custom_s is None:
        raise ValueError("--drive-custom-s is required when --drive-pattern custom")

    effective_skip_pdf = bool(args.skip_pdf)
    if args.output_pdf is not None:
        effective_skip_pdf = False

    effective_summary = (
        Path(args.output_summary) if args.output_summary is not None else default_output_summary
    )

    return BenchmarkConfig(
        problem=str(args.problem),
        L=int(args.L),
        t=float(args.t),
        u=float(args.u),
        dv=float(args.dv),
        boundary=str(args.boundary),
        ordering=str(args.ordering),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        methods=methods,
        steps_grid=steps_grid,
        reference_steps=int(args.reference_steps),
        active_coeff_tol=float(args.active_coeff_tol),
        drive_enabled=bool(args.drive_enabled),
        drive_A=float(args.drive_A),
        drive_omega=float(args.drive_omega),
        drive_tbar=float(args.drive_tbar),
        drive_phi=float(args.drive_phi),
        drive_t0=float(args.drive_t0),
        drive_pattern=str(args.drive_pattern),
        drive_custom_s=(None if args.drive_custom_s is None else str(args.drive_custom_s)),
        drive_include_identity=bool(args.drive_include_identity),
        drive_time_sampling=str(args.drive_time_sampling),
        initial_state_source=str(args.initial_state_source),
        vqe_ansatz=str(args.vqe_ansatz),
        vqe_reps=int(args.vqe_reps),
        vqe_restarts=int(args.vqe_restarts),
        vqe_maxiter=int(args.vqe_maxiter),
        vqe_method=str(args.vqe_method),
        adapt_pool=str(args.adapt_pool),
        adapt_max_depth=int(args.adapt_max_depth),
        adapt_maxiter=int(args.adapt_maxiter),
        calibrate_transpile=bool(args.calibrate_transpile),
        compare_policy=str(args.compare_policy),
        cost_match_metric=str(args.cost_match_metric),
        cost_match_tolerance=float(args.cost_match_tolerance),
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
        output_pdf=(Path(args.output_pdf) if args.output_pdf is not None else default_output_pdf),
        output_summary=effective_summary,
        skip_pdf=effective_skip_pdf,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if bool(args.replot_only):
        output_pdf = (
            Path(args.output_pdf)
            if args.output_pdf is not None
            else (Path(args.output_json).parent / "cfqm_vs_suzuki_proxy_energy_comparison.pdf")
        )
        _replot_from_output_json(
            artifact_json=Path(args.output_json),
            output_pdf=output_pdf,
        )
        print(f"WROTE {output_pdf}")
        return

    cfg = _to_config(args)
    res = run_benchmark(cfg)
    out = res["payload"]

    print(f"WROTE {res['output_json']}")
    print(f"WROTE {res['output_csv']}")
    print(f"WROTE {res['summary_json']}")
    print(f"WROTE {cfg.output_summary}")

    top = out["summary"]["ranking_by_error_then_cx"]
    if top:
        best = top[0]
        print(
            "BEST "
            f"method={best['method']} steps={best['trotter_steps']} "
            f"final_abs_energy_error={best['final_abs_energy_error']:.6e} "
            f"cx_proxy_total={best['cx_proxy_total']}"
        )


if __name__ == "__main__":
    main()
