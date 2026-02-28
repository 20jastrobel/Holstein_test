#!/usr/bin/env python3
"""Generate a readable PDF report for HH ADAPT-VQE delta-e reproducibility.

The report includes:
- fixed parameters used for the runs
- compact pipeline commands to reproduce each artifact
- pytest command used in the validation suite
- readable result tables (no number overflow in cells)
- plots of energies and |delta_e| versus g for L=2 and L=3 artifacts
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


SUITE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SUITE_DIR.parent
ARTIFACT_DIR = SUITE_DIR / "artifacts"
OUTPUT_PDF = ARTIFACT_DIR / "hh_adapt_vqe_g_repro_report.pdf"

BASE_SETTINGS = {
    "problem": "hh",
    "t": 0.2,
    "u": 0.2,
    "omega0": 0.2,
    "n_ph_max": 1,
    "boson_encoding": "binary",
    "boundary": "open",
    "ordering": "blocked",
    "adapt_pool": "paop_full",
    "adapt_max_depth": 120,
    "adapt_maxiter": 1200,
    "adapt_eps_grad": 1e-12,
    "adapt_eps_energy": 1e-10,
    "adapt_seed": 7,
    "paop_r": 1,
    "initial_state_source": "hf",
    "dv": 0.0,
}

L2_G_POINTS = [0.5, 1.25, 2.0]
L3_ARTIFACT_FILES = ["hh_adapt_vqe_L3_seed7.json"]
L3_PROBE_ARTIFACT_G_POINTS = {
    0.5: "hh_adapt_vqe_L3_seed7_g0p5_probe.json",
    1.0: "hh_adapt_vqe_L3_seed7_g1p0_probe.json",
    1.5: "hh_adapt_vqe_L3_seed7_g1p5_probe.json",
    2.0: "hh_adapt_vqe_L3_seed7_g2p0_probe.json",
}

PROBE_SETTINGS = {
    "adapt_max_depth": 30,
    "adapt_maxiter": 25,
    "adapt_eps_grad": 1e-4,
    "adapt_eps_energy": 1e-4,
}


def _fmt_float(v: float, digits: int = 15) -> str:
    return f"{v:.{digits}f}"


def _l2_artifact_name(g: float) -> str:
    return f"hh_adapt_vqe_L2_seed7_g{str(g).replace('.', 'p')}.json"


def _pipeline_command(row: dict) -> str:
    settings = row["settings"]
    seed = settings.get("adapt_seed", BASE_SETTINGS["adapt_seed"])
    output_json = settings.get("output_json", f"hh_adapt_vqe_validation_suite/artifacts/{row['artifact']}")

    common_args = (
        f"--problem {settings['problem']} "
        f"--t {settings['t']} "
        f"--u {settings['u']} "
        f"--omega0 {settings['omega0']} "
        f"--n-ph-max {settings['n_ph_max']} "
        f"--boson-encoding {settings['boson_encoding']} "
        f"--boundary {settings['boundary']} "
        f"--ordering {settings['ordering']} "
        f"--adapt-pool {settings['adapt_pool']} "
        f"--adapt-max-depth {settings['adapt_max_depth']} "
        f"--adapt-maxiter {settings['adapt_maxiter']} "
        f"--adapt-eps-grad {settings['adapt_eps_grad']} "
        f"--adapt-eps-energy {settings['adapt_eps_energy']} "
        "--adapt-no-repeats --adapt-no-finite-angle-fallback "
        f"--adapt-seed {seed} "
        f"--paop-r {settings.get('paop_r', BASE_SETTINGS['paop_r'])} "
        f"--initial-state-source {settings.get('initial_state_source', BASE_SETTINGS['initial_state_source'])} "
        f"--skip-pdf --dv {settings.get('dv', BASE_SETTINGS['dv'])}"
    )
    return (
        "python pipelines/hardcoded_adapt_pipeline.py "
        f"--L {settings['L']} --g-ep {settings['g_ep']} {common_args} "
        f"--output-json {output_json}"
    )


def _row_profile(settings: dict, artifact: str) -> str:
    if "_probe" in artifact:
        return (
            "probe (max_depth=30, maxiter=25, eps_grad=1e-4, eps_energy=1e-4)"
        )
    if int(settings.get("L", 0)) == 2:
        return (
            "full (max_depth=1200, maxiter=1200, eps_grad=1e-12, eps_energy=1e-10)"
        )
    if artifact == "hh_adapt_vqe_L3_seed7.json":
        return "full (L3 single, same as above)"
    return "full"


def _load_artifact(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    file_settings = payload.get("settings", {})
    adapt = payload.get("adapt_vqe", {})
    artifact = path.name
    settings = dict(BASE_SETTINGS)
    settings.update({k: v for k, v in file_settings.items() if v is not None})

    if "_probe" in artifact:
        settings.update(PROBE_SETTINGS)

    settings["output_json"] = file_settings.get(
        "output_json",
        f"hh_adapt_vqe_validation_suite/artifacts/{artifact}",
    )
    settings["initial_state_source"] = file_settings.get(
        "initial_state_source",
        BASE_SETTINGS["initial_state_source"],
    )
    settings["paop_r"] = file_settings.get("paop_r", BASE_SETTINGS["paop_r"])
    settings["dv"] = file_settings.get("dv", BASE_SETTINGS["dv"])
    settings["t"] = file_settings.get("t", BASE_SETTINGS["t"])
    settings["u"] = file_settings.get("u", BASE_SETTINGS["u"])
    settings["omega0"] = file_settings.get("omega0", BASE_SETTINGS["omega0"])
    settings["n_ph_max"] = file_settings.get("n_ph_max", BASE_SETTINGS["n_ph_max"])
    settings["boson_encoding"] = file_settings.get("boson_encoding", BASE_SETTINGS["boson_encoding"])
    settings["boundary"] = file_settings.get("boundary", BASE_SETTINGS["boundary"])
    settings["ordering"] = file_settings.get("ordering", BASE_SETTINGS["ordering"])
    settings["adapt_pool"] = file_settings.get("adapt_pool", BASE_SETTINGS["adapt_pool"])
    settings["adapt_seed"] = file_settings.get("adapt_seed", BASE_SETTINGS["adapt_seed"])

    exact = float(adapt["exact_gs_energy"])
    energy = float(adapt["energy"])
    abs_delta = float(adapt.get("abs_delta_e", abs(energy - exact)))
    rel_delta = float(abs_delta / abs(exact)) if exact != 0 else 0.0

    return {
        "artifact": artifact,
        "L": int(settings["L"]),
        "g": float(settings["g_ep"]),
        "settings": settings,
        "profile": _row_profile(settings, artifact),
        "seed": settings.get("adapt_seed"),
        "exact": exact,
        "adapt": energy,
        "delta": energy - exact,
        "abs_delta": abs_delta,
        "rel_delta": rel_delta,
        "depth": int(adapt.get("ansatz_depth", 0)),
        "nfev": int(adapt.get("nfev_total", 0)),
        "stop": str(adapt.get("stop_reason", "n/a")),
    }


def _load_rows() -> list[dict]:
    rows: list[dict] = []

    for g_val in L2_G_POINTS:
        path = ARTIFACT_DIR / _l2_artifact_name(g_val)
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact file: {path}")
        rows.append(_load_artifact(path))

    for fname in L3_ARTIFACT_FILES:
        path = ARTIFACT_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact file: {path}")
        rows.append(_load_artifact(path))

    for g_val, fname in L3_PROBE_ARTIFACT_G_POINTS.items():
        path = ARTIFACT_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact file: {path}")
        rows.append(_load_artifact(path))

    rows.sort(key=lambda r: (r["L"], r["g"], r["artifact"]))
    return rows


def _parameter_lines() -> list[str]:
    return [
        "Shared settings used for all rows in this report:",
        f"Suite directory: {SUITE_DIR}",
        f"Pipeline script: {PIPELINE_DIR / 'pipelines' / 'hardcoded_adapt_pipeline.py'}",
        f"Test file: {SUITE_DIR / 'tests' / 'test_hh_adapt_vqe_ground_states.py'}",
        "",
        f"problem={BASE_SETTINGS['problem']}  t={BASE_SETTINGS['t']}  u={BASE_SETTINGS['u']}  omega0={BASE_SETTINGS['omega0']}",
        f"n_ph_max={BASE_SETTINGS['n_ph_max']}  boson_encoding={BASE_SETTINGS['boson_encoding']}  boundary={BASE_SETTINGS['boundary']}",
        f"ordering={BASE_SETTINGS['ordering']}  adapt_pool={BASE_SETTINGS['adapt_pool']}  paop_r={BASE_SETTINGS['paop_r']}",
        f"adapt_max_depth={BASE_SETTINGS['adapt_max_depth']}  adapt_maxiter={BASE_SETTINGS['adapt_maxiter']}",
        f"adapt_eps_grad={BASE_SETTINGS['adapt_eps_grad']}  adapt_eps_energy={BASE_SETTINGS['adapt_eps_energy']}",
        "adapt_no_repeats=True  adapt_no_finite_angle_fallback=True",
        f"initial_state_source={BASE_SETTINGS['initial_state_source']}  dv={BASE_SETTINGS['dv']}",
        "",
        "Included artifacts:",
        "- L=2 at g={0.5, 1.25, 2.0}",
        "- L=2 at g=0.5, 1.25, 2.0 (full settings)",
        "- L=3 at g=0.2 (hh_adapt_vqe_L3_seed7.json, full settings)",
        "- L=3 at g=0.5, 1.0, 1.5, 2.0 (probe settings)",
    ]


def _command_lines(rows: list[dict]) -> list[str]:
    lines: list[str] = [
        "Commands used / needed to reproduce the artifacts and delta-e values:",
        "cd Adapt-VQE-Pipeline",
        "",
    ]

    for row in rows:
        lines.append(f"# {row['artifact']}  (L={row['L']}, g={row['g']}, seed={row['seed']}, {row['profile']})")
        lines.append(_pipeline_command(row))
        lines.append("")

    lines.extend(
        [
            "cd hh_adapt_vqe_validation_suite",
            "python -m pytest -q tests/test_hh_adapt_vqe_ground_states.py -k energy_is_sensitive_to_g",
        ]
    )
    return lines


def _wrap_lines(lines: list[str], width: int = 118) -> str:
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(line, width=width, break_long_words=False))
    return "\n".join(wrapped)


def _render_inputs_page(rows: list[dict]) -> plt.Figure:
    fig = plt.figure(figsize=(11.0, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35])

    ax_params = fig.add_subplot(gs[0, 0])
    ax_cmds = fig.add_subplot(gs[1, 0])

    ax_params.axis("off")
    ax_cmds.axis("off")

    ax_params.text(
        0.01,
        0.98,
        "Delta-E Reproducibility Inputs (L=2 and L=3)",
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    ax_params.text(
        0.01,
        0.90,
        "\n".join(_parameter_lines()),
        fontsize=10,
        family="monospace",
        va="top",
    )

    ax_cmds.text(0.01, 0.98, "Pipeline and test commands", fontsize=13, fontweight="bold", va="top")
    ax_cmds.text(
        0.01,
        0.92,
        _wrap_lines(_command_lines(rows), width=122),
        fontsize=8.0,
        family="monospace",
        va="top",
    )

    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03, hspace=0.22)
    return fig


def _apply_table_geometry(table, widths: list[float], row_height: float, font_size: int) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    n_cols = len(widths)
    for (r_idx, c_idx), cell in table.get_celld().items():
        if c_idx < n_cols:
            cell.set_width(widths[c_idx])
        cell.set_height(row_height)
        cell.get_text().set_fontfamily("monospace")
        if r_idx == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e6f2ff")


def _render_tables_page(rows: list[dict]) -> plt.Figure:
    fig = plt.figure(figsize=(11.0, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0])

    ax_energy = fig.add_subplot(gs[0, 0])
    ax_diag = fig.add_subplot(gs[1, 0])

    ax_energy.axis("off")
    ax_diag.axis("off")

    energy_cols = ["L", "mode", "g", "exact_gs_energy", "adapt_energy", "delta_e", "abs_delta_e"]
    energy_rows = [
        [
            str(r["L"]),
            "probe" if "_probe" in r["artifact"] else "full",
            f"{r['g']:.3f}",
            _fmt_float(r["exact"]),
            _fmt_float(r["adapt"]),
            _fmt_float(r["delta"]),
            _fmt_float(r["abs_delta"]),
        ]
        for r in rows
    ]

    energy_table = ax_energy.table(
        cellText=energy_rows,
        colLabels=energy_cols,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    _apply_table_geometry(
        energy_table,
        widths=[0.06, 0.08, 0.08, 0.20, 0.20, 0.19, 0.19],
        row_height=0.14,
        font_size=9,
    )
    ax_energy.set_title("Energy Table (values sized to fit cells)", fontsize=13, pad=12)

    diag_cols = ["L", "mode", "g", "ansatz_depth", "nfev_total", "stop_reason"]
    diag_rows = [
        [
            str(r["L"]),
            "probe" if "_probe" in r["artifact"] else "full",
            f"{r['g']:.3f}",
            str(r["depth"]),
            str(r["nfev"]),
            r["stop"],
        ]
        for r in rows
    ]

    diag_table = ax_diag.table(
        cellText=diag_rows,
        colLabels=diag_cols,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    _apply_table_geometry(
        diag_table,
        widths=[0.10, 0.09, 0.09, 0.15, 0.15, 0.42],
        row_height=0.18,
        font_size=10,
    )
    ax_diag.set_title("ADAPT Diagnostics", fontsize=13, pad=12)

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])
    return fig


def _render_plots_page(rows: list[dict]) -> plt.Figure:
    fig = plt.figure(figsize=(11.0, 11.5))
    ax_energy = fig.add_subplot(3, 1, 1)
    ax_delta = fig.add_subplot(3, 1, 2)
    ax_rel = fig.add_subplot(3, 1, 3)

    l_values = sorted({int(r["L"]) for r in rows})
    for l_val in l_values:
        l_rows = sorted((r for r in rows if int(r["L"]) == l_val), key=lambda r: r["g"])
        xs = [r["g"] for r in l_rows]
        exact = [r["exact"] for r in l_rows]
        adapt = [r["adapt"] for r in l_rows]
        abs_delta = [r["abs_delta"] for r in l_rows]
        rel_delta = [r["rel_delta"] for r in l_rows]

        if len(xs) > 1:
            ax_energy.plot(xs, exact, marker="o", linewidth=2, label=f"L={l_val} exact")
            ax_energy.plot(xs, adapt, marker="s", linewidth=2, label=f"L={l_val} adapt")
            ax_delta.plot(xs, abs_delta, marker="d", linewidth=2, label=f"L={l_val} |delta_e|")
            ax_rel.plot(xs, rel_delta, marker="^", linewidth=2, label=f"L={l_val} rel_err")
        else:
            ax_energy.scatter(xs, exact, marker="o", s=70, label=f"L={l_val} exact (single)")
            ax_energy.scatter(xs, adapt, marker="s", s=70, label=f"L={l_val} adapt (single)")
            ax_delta.scatter(xs, abs_delta, marker="d", s=70, label=f"L={l_val} |delta_e| (single)")
            ax_rel.scatter(xs, rel_delta, marker="^", s=70, label=f"L={l_val} rel_err (single)")

    for row in rows:
        marker = "d" if "_probe" in row["artifact"] else "o"
        ax_energy.scatter(
            row["g"],
            row["exact"],
            marker=marker,
            s=30,
            alpha=0.7,
        )
        ax_energy.scatter(
            row["g"],
            row["adapt"],
            marker=marker,
            s=30,
            alpha=0.7,
        )
        ax_delta.scatter(
            row["g"],
            row["abs_delta"],
            marker=marker,
            s=30,
            alpha=0.7,
        )
        ax_rel.scatter(
            row["g"],
            row["rel_delta"],
            marker=marker,
            s=30,
            alpha=0.7,
        )

    ax_energy.set_title("Ground-state energies vs g")
    ax_energy.set_xlabel("g")
    ax_energy.set_ylabel("Energy")
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend(fontsize=9)

    ax_delta.set_title("Absolute error |delta_e| vs g")
    ax_delta.set_xlabel("g")
    ax_delta.set_ylabel("|delta_e|")
    ax_delta.grid(True, alpha=0.3)
    ax_delta.legend(fontsize=9)

    ax_rel.set_title("Relative error |delta_e| / |E_exact| vs g")
    ax_rel.set_xlabel("g")
    ax_rel.set_ylabel("Relative error")
    ax_rel.set_yscale("log")
    ax_rel.grid(True, alpha=0.3)

    handles, labels = ax_rel.get_legend_handles_labels()
    if handles:
        ax_rel.legend(handles[:8], labels[:8], fontsize=8, loc="upper left", ncol=2)

    fig.text(
        0.03,
        0.01,
        "Legend description: curves are plotted per L. Solid lines show exact/adapt/|delta_e|/relative-error trends. "
        "Open-point overlays indicate artifact type: circle = full run, diamond = probe run.",
        fontsize=8,
        va="bottom",
    )

    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.98])
    return fig


def generate_pdf() -> Path:
    rows = _load_rows()

    fig_inputs = _render_inputs_page(rows)
    fig_tables = _render_tables_page(rows)
    fig_plots = _render_plots_page(rows)

    with PdfPages(OUTPUT_PDF) as pdf:
        pdf.savefig(fig_inputs)
        pdf.savefig(fig_tables)
        pdf.savefig(fig_plots)

    plt.close(fig_inputs)
    plt.close(fig_tables)
    plt.close(fig_plots)

    return OUTPUT_PDF


if __name__ == "__main__":
    output = generate_pdf()
    print(f"Wrote PDF: {output}")
    print(f"Generated at UTC: {datetime.now(timezone.utc).isoformat()}")
