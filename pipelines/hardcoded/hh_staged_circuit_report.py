#!/usr/bin/env python3
"""Combined HH staged circuit report for selected system sizes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)
from pipelines.hardcoded.hh_staged_cli_args import build_staged_hh_parser
from pipelines.hardcoded.hh_staged_workflow import (
    resolve_staged_hh_config,
    run_stage_pipeline,
    write_hh_staged_circuit_report_section,
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_staged_hh_parser(
        description=(
            "Combined HH staged circuit report: runs the staged HH warm -> ADAPT -> "
            "matched-family replay chain for each requested L and writes one PDF with "
            "representative and expanded circuit views."
        )
    )
    parser.add_argument(
        "--l-values",
        type=str,
        default="2,3",
        help="Comma-separated L values to include in the combined circuit PDF.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _parse_l_values(raw: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for part in str(raw).split(","):
        text = part.strip()
        if text == "":
            continue
        value = int(text)
        if value <= 0:
            raise ValueError("--l-values entries must be positive integers.")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    if not out:
        raise ValueError("--l-values must contain at least one positive integer.")
    return out


def _combined_output_pdf(args: argparse.Namespace, l_values: list[int]) -> Path:
    if getattr(args, "output_pdf", None) is not None:
        return Path(args.output_pdf)
    suffix = "_".join(f"L{int(L)}" for L in l_values)
    return Path("artifacts/pdf") / f"hh_staged_circuit_report_{suffix}.pdf"


def _args_for_l(base_args: argparse.Namespace, L: int) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(base_args))
    ns.L = int(L)
    ns.skip_pdf = True
    ns.output_json = None
    ns.output_pdf = None
    if getattr(base_args, "tag", None):
        ns.tag = f"{base_args.tag}_L{int(L)}"
    return ns


def run(args: argparse.Namespace) -> Path:
    if bool(getattr(args, "skip_pdf", False)):
        raise ValueError("hh_staged_circuit_report.py always writes a PDF; omit --skip-pdf.")

    l_values = _parse_l_values(str(args.l_values))
    output_pdf = _combined_output_pdf(args, l_values)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    run_command = current_command_string()

    require_matplotlib()
    PdfPages = get_PdfPages()
    with PdfPages(output_pdf) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein (HH)",
            ansatz=(
                f"warm: {args.warm_ansatz}; "
                f"ADAPT: {args.adapt_continuation_mode}; "
                "final: matched-family replay"
            ),
            drive_enabled=bool(args.enable_drive),
            t=float(args.t),
            U=float(args.u),
            dv=float(args.dv),
            extra={
                "L_values": ",".join(str(x) for x in l_values),
                "omega0": float(args.omega0),
                "g_ep": float(args.g_ep),
                "n_ph_max": int(args.n_ph_max),
                "ordering": str(args.ordering),
                "boundary": str(args.boundary),
                "dynamics_pages": "suzuki2,cfqm4 macro-step only",
            },
            command=run_command,
        )
        render_text_page(
            pdf,
            [
                "Combined HH staged circuit report",
                "",
                f"L values: {', '.join(str(x) for x in l_values)}",
                "Each L section includes:",
                "- parameter manifest",
                "- stage summary",
                "- warm HH-HVA circuit",
                "- ADAPT circuit",
                "- matched-family replay circuit",
                "- Suzuki2 macro-step circuit",
                "- CFQM4 macro-step circuit",
                "",
                "Representative pages preserve PauliEvolutionGate blocks.",
                "Expanded pages decompose one level only for readability.",
            ],
            fontsize=10,
            line_spacing=0.03,
            max_line_width=110,
        )
        for L in l_values:
            cfg = resolve_staged_hh_config(_args_for_l(args, int(L)))
            stage_result = run_stage_pipeline(cfg)
            write_hh_staged_circuit_report_section(
                pdf,
                cfg=cfg,
                stage_result=stage_result,
                run_command=run_command,
            )

    return output_pdf


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_pdf = run(args)
    print(f"circuit_report_pdf={output_pdf}")


if __name__ == "__main__":
    main()
