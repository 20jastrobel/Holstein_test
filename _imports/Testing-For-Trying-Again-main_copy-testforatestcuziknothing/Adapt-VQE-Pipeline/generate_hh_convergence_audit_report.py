#!/usr/bin/env python3
"""Generate HH convergence audit report v2 (long, stratified, readable).

This report is artifact-first and separates:
1) Observed execution evidence from JSON artifacts.
2) Configured defaults/evidence parsed from suite scripts/docs/code.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIRS = [
    SCRIPT_DIR / "artifacts",
    SCRIPT_DIR / "hh_adapt_vqe_validation_suite" / "artifacts",
]

DEFAULT_OUTPUT_PDF = SCRIPT_DIR / "artifacts" / "hh_convergence_audit_report.pdf"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "artifacts" / "hh_convergence_audit_summary.csv"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "artifacts" / "hh_convergence_audit_summary.json"

THRESHOLDS = [1e-1, 1e-2, 1e-3, 1e-4]
ZERO_CASE_STRATA_ORDER = ["g=0 & omega=0", "exactly_one_zero", "both_nonzero", "missing_parameter"]
G_STRATA_ORDER = ["g=0", "0<g<=0.1", "0.1<g<=0.5", "0.5<g<=1.0", "1.0<g<=2.0", "g>2.0", "missing_parameter"]
OMEGA_STRATA_ORDER = ["omega=0", "0<omega<=0.2", "omega>0.2", "missing_parameter"]


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _is_finite_number(value: Any) -> bool:
    return _safe_float(value) is not None


def _fmt_float(value: Any, digits: int = 6) -> str:
    fv = _safe_float(value)
    if fv is None:
        return "n/a"
    return f"{fv:.{digits}g}"


def _fmt_float_fixed(value: Any, digits: int = 6) -> str:
    fv = _safe_float(value)
    if fv is None:
        return "n/a"
    return f"{fv:.{digits}f}"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _split_csv_tokens(raw: str) -> list[str]:
    out: list[str] = []
    for token in str(raw).split(","):
        tok = token.strip()
        if tok:
            out.append(tok)
    return out


def _resolve_artifact_dirs(raw: str) -> list[Path]:
    tokens = _split_csv_tokens(raw)
    if not tokens:
        tokens = [str(p) for p in DEFAULT_ARTIFACT_DIRS]
    out: list[Path] = []
    for token in tokens:
        path = Path(token)
        if not path.is_absolute():
            path = (SCRIPT_DIR / token).resolve()
        else:
            path = path.resolve()
        if path not in out:
            out.append(path)
    return out


def _relative_to_script(path: Path) -> str:
    try:
        return str(path.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


def _g_stratum(g_value: Any) -> str:
    g = _safe_float(g_value)
    if g is None:
        return "missing_parameter"
    if abs(g) <= 1e-15:
        return "g=0"
    if g <= 0.1:
        return "0<g<=0.1"
    if g <= 0.5:
        return "0.1<g<=0.5"
    if g <= 1.0:
        return "0.5<g<=1.0"
    if g <= 2.0:
        return "1.0<g<=2.0"
    return "g>2.0"


def _omega_stratum(omega_value: Any) -> str:
    omega = _safe_float(omega_value)
    if omega is None:
        return "missing_parameter"
    if abs(omega) <= 1e-15:
        return "omega=0"
    if omega <= 0.2:
        return "0<omega<=0.2"
    return "omega>0.2"


def _zero_case_stratum(g_value: Any, omega_value: Any) -> str:
    g = _safe_float(g_value)
    omega = _safe_float(omega_value)
    if g is None or omega is None:
        return "missing_parameter"
    g_zero = abs(g) <= 1e-15
    omega_zero = abs(omega) <= 1e-15
    if g_zero and omega_zero:
        return "g=0 & omega=0"
    if g_zero != omega_zero:
        return "exactly_one_zero"
    return "both_nonzero"


def _zero_case_from_strata(g_stratum: str, omega_stratum: str) -> str:
    if g_stratum == "missing_parameter" or omega_stratum == "missing_parameter":
        return "missing_parameter"
    if g_stratum == "g=0" and omega_stratum == "omega=0":
        return "g=0 & omega=0"
    if (g_stratum == "g=0") != (omega_stratum == "omega=0"):
        return "exactly_one_zero"
    return "both_nonzero"


def _physics_point_label(g_value: Any, omega_value: Any) -> str:
    g = _safe_float(g_value)
    omega = _safe_float(omega_value)
    if g is None or omega is None:
        return "g=?, omega0=?"
    return f"g={g:.6g}, omega0={omega:.6g}"


def _is_hh_record(payload: dict[str, Any], settings: dict[str, Any], path: Path, *, strict_hh_only: bool) -> bool:
    method = ""
    if isinstance(payload.get("adapt_vqe"), dict):
        method = str(payload["adapt_vqe"].get("method", ""))
    elif isinstance(payload.get("vqe"), dict):
        method = str(payload["vqe"].get("method", ""))

    problem = settings.get("problem")
    relpath = _relative_to_script(path)

    if strict_hh_only:
        if problem is not None and str(problem).lower() == "hh":
            return True
        hh_setting_keys = ("g_ep", "omega0", "n_ph_max", "boson_encoding")
        if any(settings.get(key) is not None for key in hh_setting_keys):
            return True
        if "hh_" in relpath:
            return True
        method_l = method.lower()
        return (
            "hh" in method_l
            or "_hva_" in method_l
            or method_l.startswith("hardcoded_hh")
            or method_l.startswith("hardcoded_adapt_vqe_paop")
        )

    # Relaxed mode keeps likely HH-related records if either HH markers or explicit
    # e-ph fields are present.
    if problem is not None and str(problem).lower() == "hh":
        return True
    if any(settings.get(key) is not None for key in ("g_ep", "omega0", "n_ph_max")):
        return True
    method_l = method.lower()
    return "hh" in method_l or "paop" in method_l or "hva" in method_l


def _to_category_label(kind: str, pool_value: str, method_value: str) -> str:
    pool = str(pool_value).strip().lower()
    method = str(method_value).strip().lower()
    combined = " ".join([pool, method])
    if any(token in combined for token in ("termwise", "term-wise", "componentwise", "component-wise")):
        return "HVA (term-wise)"

    if kind == "ADAPT":
        if pool.startswith("paop") or "paop" in method:
            return "Polaron PAOP"
        if pool == "hva":
            return "HVA-adapt"
        if pool == "full_hamiltonian":
            return "Full-Hamiltonian"
        if pool == "cse":
            return "CSE"
        if pool == "uccsd":
            return "UCCSD"
        return pool_value if pool_value else "unknown"

    if pool in {"hh_hva", "hva"}:
        return "HVA (layer-wise)"
    if pool == "uccsd":
        return "UCCSD"
    if "hva" in method:
        return "HVA (layer-wise)"
    if "uccsd" in method:
        return "UCCSD"
    return pool_value if pool_value else "unknown"


def _extract_run_record(
    *,
    kind: str,
    payload: dict[str, Any],
    settings: dict[str, Any],
    rec: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    g_value = settings.get("g_ep")
    if g_value is None:
        g_value = settings.get("g")
    omega0_value = settings.get("omega0")
    if omega0_value is None:
        omega0_value = settings.get("omega")

    if kind == "ADAPT":
        pool_value = str(rec.get("pool_type", settings.get("adapt_pool", "unknown")))
        method_value = str(rec.get("method", "unknown"))
        exact_value = _safe_float(rec.get("exact_gs_energy"))
        energy_value = _safe_float(rec.get("energy"))
        abs_delta_value = _safe_float(rec.get("abs_delta_e"))
    else:
        ansatz = rec.get("ansatz")
        if ansatz is None:
            ansatz = settings.get("vqe_ansatz")
        if ansatz is None:
            ansatz = settings.get("ansatz")
        pool_value = str(ansatz or "unknown")
        method_value = str(rec.get("method", "unknown"))
        exact_value = _safe_float(rec.get("exact_filtered_energy"))
        if exact_value is None:
            exact_value = _safe_float(rec.get("exact_energy"))
        energy_value = _safe_float(rec.get("energy"))
        abs_delta_value = _safe_float(rec.get("abs_delta_e"))

    if abs_delta_value is None and exact_value is not None and energy_value is not None:
        abs_delta_value = abs(float(energy_value) - float(exact_value))

    category = _to_category_label(kind, pool_value, method_value)
    g_stratum = _g_stratum(g_value)
    omega_stratum = _omega_stratum(omega0_value)
    zero_stratum = _zero_case_stratum(g_value, omega0_value)
    problem_setting = str(settings.get("problem", "")).strip()

    return {
        "kind": str(kind),
        "category": str(category),
        "pipeline": str(payload.get("pipeline", "unknown")),
        "method": str(method_value),
        "pool": str(pool_value),
        "path": _relative_to_script(path),
        "L": _safe_float(settings.get("L")),
        "boundary": str(settings.get("boundary", "")),
        "ordering": str(settings.get("ordering", "")),
        "t": _safe_float(settings.get("t")),
        "u": _safe_float(settings.get("u")),
        "dv": _safe_float(settings.get("dv")),
        "g": _safe_float(g_value),
        "omega0": _safe_float(omega0_value),
        "n_ph_max": _safe_float(settings.get("n_ph_max")),
        "boson_encoding": str(settings.get("boson_encoding", "")),
        "energy": energy_value,
        "exact": exact_value,
        "abs_delta": abs_delta_value,
        "success": bool(rec.get("success", False)),
        "problem_setting": problem_setting,
        "g_stratum": g_stratum,
        "omega_stratum": omega_stratum,
        "zero_case_stratum": zero_stratum,
        "physics_point_label": _physics_point_label(g_value, omega0_value),
    }


def _run_dedupe_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["kind"],
        record["path"],
        record["method"],
        record["pool"],
        record["energy"],
        record["exact"],
        record["abs_delta"],
    )


def _run_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    l_value = _safe_float(record.get("L"))
    g_value = _safe_float(record.get("g"))
    omega_value = _safe_float(record.get("omega0"))
    return (
        str(record.get("kind", "")),
        str(record.get("category", "")),
        l_value if l_value is not None else float("inf"),
        g_value if g_value is not None else float("inf"),
        omega_value if omega_value is not None else float("inf"),
        str(record.get("boundary", "")),
        str(record.get("ordering", "")),
        str(record.get("path", "")),
    )


def _extract_hh_runs(artifact_dirs: Sequence[Path], *, strict_hh_only: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    scanned_files = 0
    parsed_files = 0
    included_files = 0
    included_paths: set[str] = set()

    for base in artifact_dirs:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.json")):
            scanned_files += 1
            payload = _read_json(path)
            if not isinstance(payload, dict):
                continue
            parsed_files += 1

            settings = payload.get("settings", {})
            if not isinstance(settings, dict):
                settings = {}

            if not _is_hh_record(payload, settings, path, strict_hh_only=strict_hh_only):
                continue

            file_had_record = False

            if isinstance(payload.get("adapt_vqe"), dict):
                record = _extract_run_record(
                    kind="ADAPT",
                    payload=payload,
                    settings=settings,
                    rec=payload["adapt_vqe"],
                    path=path,
                )
                key = _run_dedupe_key(record)
                if key not in seen:
                    seen.add(key)
                    runs.append(record)
                    file_had_record = True

            if isinstance(payload.get("vqe"), dict):
                record = _extract_run_record(
                    kind="VQE",
                    payload=payload,
                    settings=settings,
                    rec=payload["vqe"],
                    path=path,
                )
                key = _run_dedupe_key(record)
                if key not in seen:
                    seen.add(key)
                    runs.append(record)
                    file_had_record = True

            if file_had_record:
                included_files += 1
                included_paths.add(_relative_to_script(path))

    runs.sort(key=_run_sort_key)
    provenance = {
        "artifact_dirs": [str(path) for path in artifact_dirs],
        "strict_hh_only": bool(strict_hh_only),
        "json_files_scanned": int(scanned_files),
        "json_files_parsed": int(parsed_files),
        "json_files_with_included_runs": int(included_files),
        "json_included_unique_paths": sorted(included_paths),
        "run_count": int(len(runs)),
    }
    return runs, provenance


def _stats_for_bucket(bucket: Sequence[dict[str, Any]], thresholds: Sequence[float]) -> dict[str, Any]:
    abs_values = [float(r["abs_delta"]) for r in bucket if _is_finite_number(r.get("abs_delta"))]
    out: dict[str, Any] = {
        "N": int(len(bucket)),
        "N_abs_delta": int(len(abs_values)),
        "mean_abs_delta": None,
        "median_abs_delta": None,
        "pass_counts": {},
    }
    if abs_values:
        out["mean_abs_delta"] = float(mean(abs_values))
        out["median_abs_delta"] = float(median(abs_values))
    for threshold in thresholds:
        count = sum(1 for value in abs_values if value <= float(threshold))
        out["pass_counts"][f"{threshold:.0e}"] = {
            "count": int(count),
            "denominator": int(len(abs_values)),
            "rate": float(count / len(abs_values)) if abs_values else None,
        }
    return out


def _group_stats_rows(
    runs: Sequence[dict[str, Any]],
    *,
    group_fields: Sequence[str],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[tuple(run.get(field) for field in group_fields)].append(run)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=lambda k: tuple(str(x) for x in k)):
        bucket = grouped[key]
        row: dict[str, Any] = {field: key[idx] for idx, field in enumerate(group_fields)}
        row.update(_stats_for_bucket(bucket, thresholds))
        rows.append(row)
    return rows


def _build_coverage(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    contexts = sorted(
        {
            (
                int(_safe_float(run.get("L"), default=-1) or -1),
                str(run.get("boundary", "")),
                str(run.get("ordering", "")),
            )
            for run in runs
        }
    )

    zero_counts = Counter()
    full_counts = Counter()
    for run in runs:
        ctx = (
            int(_safe_float(run.get("L"), default=-1) or -1),
            str(run.get("boundary", "")),
            str(run.get("ordering", "")),
        )
        z = str(run.get("zero_case_stratum"))
        g = str(run.get("g_stratum"))
        omega = str(run.get("omega_stratum"))
        zero_counts[(ctx, z)] += 1
        full_counts[(ctx, g, omega, z)] += 1

    zero_matrix: list[dict[str, Any]] = []
    for ctx in contexts:
        for z_stratum in ZERO_CASE_STRATA_ORDER:
            count = int(zero_counts.get((ctx, z_stratum), 0))
            zero_matrix.append(
                {
                    "L": int(ctx[0]),
                    "boundary": str(ctx[1]),
                    "ordering": str(ctx[2]),
                    "zero_case_stratum": z_stratum,
                    "count": count,
                    "status": "Observed" if count > 0 else "Not observed",
                }
            )

    # Keep this matrix compact/readable: observed combinations only.
    full_matrix: list[dict[str, Any]] = []
    for (ctx, g_stratum, omega_stratum, zero_stratum), count in sorted(
        full_counts.items(),
        key=lambda item: (
            item[0][0][0],
            item[0][0][1],
            item[0][0][2],
            str(item[0][1]),
            str(item[0][2]),
            str(item[0][3]),
        ),
    ):
        full_matrix.append(
            {
                "L": int(ctx[0]),
                "boundary": str(ctx[1]),
                "ordering": str(ctx[2]),
                "g_stratum": g_stratum,
                "omega_stratum": omega_stratum,
                "zero_case_stratum": zero_stratum,
                "count": int(count),
                "status": "Observed",
            }
        )

    return {
        "contexts": [
            {"L": int(ctx[0]), "boundary": str(ctx[1]), "ordering": str(ctx[2])}
            for ctx in contexts
        ],
        "zero_case_matrix": zero_matrix,
        "stratum_matrix": full_matrix,
        "stratum_matrix_mode": "observed_only",
    }


def _build_method_summary(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall_by_category": _group_stats_rows(runs, group_fields=["kind", "category"], thresholds=THRESHOLDS),
        "by_category_and_zero_case": _group_stats_rows(
            runs,
            group_fields=["kind", "category", "zero_case_stratum"],
            thresholds=THRESHOLDS,
        ),
        "by_category_and_g_stratum": _group_stats_rows(
            runs,
            group_fields=["kind", "category", "g_stratum"],
            thresholds=THRESHOLDS,
        ),
        "by_category_and_omega_stratum": _group_stats_rows(
            runs,
            group_fields=["kind", "category", "omega_stratum"],
            thresholds=THRESHOLDS,
        ),
    }


def _build_threshold_summary(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "thresholds": [f"{value:.0e}" for value in THRESHOLDS],
        "overall": _stats_for_bucket(runs, THRESHOLDS),
        "by_kind": _group_stats_rows(runs, group_fields=["kind"], thresholds=THRESHOLDS),
        "by_category": _group_stats_rows(runs, group_fields=["kind", "category"], thresholds=THRESHOLDS),
    }


def _build_stratified_distributions(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_kind": _group_stats_rows(runs, group_fields=["kind"], thresholds=THRESHOLDS),
        "by_zero_case_stratum": _group_stats_rows(runs, group_fields=["zero_case_stratum"], thresholds=THRESHOLDS),
        "by_g_stratum": _group_stats_rows(runs, group_fields=["g_stratum"], thresholds=THRESHOLDS),
        "by_omega_stratum": _group_stats_rows(runs, group_fields=["omega_stratum"], thresholds=THRESHOLDS),
        "by_kind_and_zero_case_stratum": _group_stats_rows(
            runs,
            group_fields=["kind", "zero_case_stratum"],
            thresholds=THRESHOLDS,
        ),
        "by_kind_and_g_stratum": _group_stats_rows(
            runs,
            group_fields=["kind", "g_stratum"],
            thresholds=THRESHOLDS,
        ),
        "by_kind_and_omega_stratum": _group_stats_rows(
            runs,
            group_fields=["kind", "omega_stratum"],
            thresholds=THRESHOLDS,
        ),
    }


def _extract_export_defaults(shell_text: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r'^\s*export\s+([A-Z0-9_]+)=["\']?\$\{[A-Z0-9_]+:-([^}]+)\}["\']?\s*$',
        re.MULTILINE,
    )
    out: list[tuple[str, str]] = []
    for match in pattern.finditer(shell_text):
        out.append((str(match.group(1)), str(match.group(2)).strip()))
    return out


def _extract_ast_constant_dicts(path: Path, names: Sequence[str]) -> dict[str, Any]:
    text = _read_text(path)
    if not text:
        return {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}

    values: dict[str, Any] = {}
    wanted = set(names)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        continue
    return values


def _extract_hardcoded_parser_defaults(path: Path) -> list[tuple[str, str]]:
    text = _read_text(path)
    if not text:
        return []

    options = [
        "--adapt-max-depth",
        "--adapt-eps-grad",
        "--adapt-eps-energy",
        "--adapt-maxiter",
        "--adapt-seed",
        "--adapt-finite-angle",
        "--adapt-finite-angle-min-improvement",
        "--paop-r",
        "--paop-prune-eps",
    ]
    out: list[tuple[str, str]] = []
    for option in options:
        pattern = re.compile(
            rf'add_argument\(\s*"{re.escape(option)}".*?default\s*=\s*([^,\n\)]+)',
            re.DOTALL,
        )
        match = pattern.search(text)
        if match:
            out.append((option, str(match.group(1)).strip()))
    return out


def _extract_configured_defaults(*, include: bool) -> dict[str, Any]:
    if not include:
        return {"enabled": False, "entries": [], "sources": []}

    entries: list[dict[str, Any]] = []
    sources: list[str] = []

    def add_entry(*, key: str, value: str, source: Path, section: str, note: str = "") -> None:
        entries.append(
            {
                "key": str(key),
                "value": str(value),
                "source": _relative_to_script(source),
                "section": str(section),
                "note": str(note),
            }
        )
        src = _relative_to_script(source)
        if src not in sources:
            sources.append(src)

    readme = SCRIPT_DIR / "hh_adapt_vqe_validation_suite" / "README.md"
    run_sh = SCRIPT_DIR / "hh_adapt_vqe_validation_suite" / "run_hh_adapt_vqe_validation.sh"
    stress_sh = SCRIPT_DIR / "hh_adapt_vqe_validation_suite" / "run_hh_adapt_vqe_validation_stress.sh"
    tests_py = SCRIPT_DIR / "hh_adapt_vqe_validation_suite" / "tests" / "test_hh_adapt_vqe_ground_states.py"
    hardcoded_py = SCRIPT_DIR / "pipelines" / "hardcoded_adapt_pipeline.py"

    readme_text = _read_text(readme)
    for match in re.finditer(r"-\s*`L=(\d+)`:\s*`abs_delta_e\s*<\s*([^`]+)`", readme_text):
        add_entry(
            key=f"suite.acceptance_abs_delta_e.L{match.group(1)}",
            value=str(match.group(2)).strip(),
            source=readme,
            section="README acceptance threshold",
        )

    for match in re.finditer(r"-\s*`([^`]+)`:\s*`([^`]+)`", readme_text):
        key = str(match.group(1)).strip()
        value = str(match.group(2)).strip()
        if key in {
            "adapt-max-depth",
            "adapt-maxiter",
            "adapt-eps-grad",
            "adapt-eps-energy",
            "t",
            "u",
            "omega0",
            "g-ep",
            "n-ph-max",
            "boson-encoding",
            "boundary",
            "ordering",
        }:
            add_entry(
                key=f"suite.default.{key}",
                value=value,
                source=readme,
                section="README defaults",
            )

    for script_path, section in [(run_sh, "validation shell defaults"), (stress_sh, "stress shell defaults")]:
        for key, value in _extract_export_defaults(_read_text(script_path)):
            add_entry(
                key=f"env.{key}",
                value=value,
                source=script_path,
                section=section,
            )

    constants = _extract_ast_constant_dicts(tests_py, names=["DEFAULT_ARGS", "ABS_DELTA_E_TOL"])
    default_args = constants.get("DEFAULT_ARGS", {})
    if isinstance(default_args, dict):
        for key in ["t", "u", "omega0", "g_ep", "n_ph_max", "boson_encoding", "boundary", "ordering", "adapt_pool"]:
            if key in default_args:
                add_entry(
                    key=f"tests.DEFAULT_ARGS.{key}",
                    value=str(default_args[key]),
                    source=tests_py,
                    section="pytest DEFAULT_ARGS",
                )
        for key in ["adapt_maxiter", "adapt_eps_grad", "adapt_eps_energy"]:
            value = default_args.get(key)
            if isinstance(value, dict):
                for lattice, val in sorted(value.items(), key=lambda item: str(item[0])):
                    add_entry(
                        key=f"tests.DEFAULT_ARGS.{key}.L{lattice}",
                        value=str(val),
                        source=tests_py,
                        section="pytest DEFAULT_ARGS per-L",
                    )
        if "adapt_max_depth" in default_args:
            add_entry(
                key="tests.DEFAULT_ARGS.adapt_max_depth",
                value=str(default_args["adapt_max_depth"]),
                source=tests_py,
                section="pytest DEFAULT_ARGS",
            )

    abs_tol = constants.get("ABS_DELTA_E_TOL", {})
    if isinstance(abs_tol, dict):
        for lattice, value in sorted(abs_tol.items(), key=lambda item: str(item[0])):
            add_entry(
                key=f"tests.ABS_DELTA_E_TOL.L{lattice}",
                value=str(value),
                source=tests_py,
                section="pytest acceptance threshold",
            )

    for option, value in _extract_hardcoded_parser_defaults(hardcoded_py):
        add_entry(
            key=f"hardcoded_adapt_cli.default.{option}",
            value=value,
            source=hardcoded_py,
            section="hardcoded_adapt_pipeline parser defaults",
        )

    # Deduplicate deterministic.
    dedupe_key = set()
    deduped: list[dict[str, Any]] = []
    for entry in sorted(
        entries,
        key=lambda item: (item["key"], item["value"], item["source"], item["section"]),
    ):
        key = (entry["key"], entry["value"], entry["source"], entry["section"])
        if key in dedupe_key:
            continue
        dedupe_key.add(key)
        deduped.append(entry)

    return {
        "enabled": True,
        "sources": sorted(sources),
        "entries": deduped,
    }


def _termwise_componentwise_evidence(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    markers = ["termwise", "term-wise", "componentwise", "component-wise"]
    observed: list[dict[str, Any]] = []
    for run in runs:
        haystack = " ".join(
            [
                str(run.get("method", "")),
                str(run.get("pool", "")),
                str(run.get("category", "")),
                str(run.get("path", "")),
            ]
        ).lower()
        if any(marker in haystack for marker in markers):
            observed.append(run)

    project_root = SCRIPT_DIR.parent
    implemented_checks = [
        (
            project_root / "hh_validation_suite" / "tests" / "test_level4_vqe_hh_ground_states.py",
            "term-wise",
        ),
        (
            project_root / "hh_validation_suite" / "scripts" / "run_termwise_benchmarks.py",
            "termwise_ansatz",
        ),
        (
            project_root / "pipelines" / "hh_termwise_zero_holstein_report.py",
            "termwise",
        ),
        (
            SCRIPT_DIR / "pipelines" / "hardcoded_adapt_pipeline.py",
            "_build_hh_termwise_augmented_pool",
        ),
    ]
    implemented_rows: list[dict[str, Any]] = []
    for path, marker in implemented_checks:
        text = _read_text(path)
        exists = path.exists()
        marker_present = bool(text and marker in text)
        implemented_rows.append(
            {
                "path": str(path),
                "exists": bool(exists),
                "marker": str(marker),
                "marker_present": bool(marker_present),
            }
        )

    observed_rows = [
        {
            "kind": run["kind"],
            "category": run["category"],
            "method": run["method"],
            "pool": run["pool"],
            "path": run["path"],
            "L": run["L"],
            "g": run["g"],
            "omega0": run["omega0"],
        }
        for run in observed
    ]

    return {
        "observed_count": int(len(observed_rows)),
        "observed_rows": observed_rows,
        "implemented_paths": implemented_rows,
    }


def _paop_usage_summary(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paop_runs = []
    for run in runs:
        if run["kind"] != "ADAPT":
            continue
        pool = str(run.get("pool", "")).lower()
        method = str(run.get("method", "")).lower()
        if "paop" in pool or "paop" in method or run.get("category") == "Polaron PAOP":
            paop_runs.append(run)

    variant_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in paop_runs:
        pool = str(run.get("pool", "")).lower()
        if pool.startswith("paop_full"):
            variant = "paop_full"
        elif pool.startswith("paop_std"):
            variant = "paop_std"
        elif pool.startswith("paop_min"):
            variant = "paop_min"
        elif pool.startswith("paop"):
            variant = "paop_other"
        else:
            variant = "paop_other"
        variant_groups[variant].append(run)

    rows: list[dict[str, Any]] = []
    for variant in sorted(variant_groups.keys()):
        bucket = variant_groups[variant]
        row = {"variant": variant}
        row.update(_stats_for_bucket(bucket, THRESHOLDS))
        rows.append(row)

    return {
        "count": int(len(paop_runs)),
        "variants": rows,
    }


def _build_parameter_grid_rows(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = (
            int(_safe_float(run.get("L"), default=-1) or -1),
            str(run.get("boundary", "")),
            str(run.get("ordering", "")),
            run.get("g"),
            run.get("omega0"),
        )
        grouped[key].append(run)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=lambda item: (item[0], item[1], item[2], str(item[3]), str(item[4]))):
        bucket = grouped[key]
        categories = sorted({str(run.get("category", "")) for run in bucket})
        methods = sorted({str(run.get("method", "")) for run in bucket})
        rows.append(
            {
                "L": int(key[0]),
                "boundary": str(key[1]),
                "ordering": str(key[2]),
                "g": key[3],
                "omega0": key[4],
                "count": int(len(bucket)),
                "categories": ", ".join(categories),
                "methods": ", ".join(methods),
            }
        )
    return rows


def _write_csv_summary(path: Path, runs: Sequence[dict[str, Any]]) -> None:
    headers = [
        "kind",
        "category",
        "pipeline",
        "method",
        "pool",
        "path",
        "L",
        "boundary",
        "ordering",
        "t",
        "u",
        "dv",
        "g",
        "omega0",
        "n_ph_max",
        "boson_encoding",
        "energy",
        "exact",
        "abs_delta",
        "success",
        "problem_setting",
        "model_target_inference",
        "g_stratum",
        "omega_stratum",
        "zero_case_stratum",
        "physics_point_label",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for run in runs:
            writer.writerow({key: run.get(key) for key in headers})


def _write_json_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _wrap_cell(value: Any, limit: int) -> str:
    text = str(value)
    if not text:
        return ""
    if len(text) <= int(limit):
        return text
    return textwrap.fill(
        text,
        width=int(limit),
        break_long_words=False,
        break_on_hyphens=True,
    )


def _table_pages(
    pdf: PdfPages,
    *,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    rows_per_page: int,
    max_cell_chars: int,
    col_widths: Sequence[float] | None = None,
    col_char_limits: Sequence[int] | None = None,
    fontsize: float = 8.0,
) -> None:
    if not rows:
        _render_text_pages(pdf, title, ["No rows available."])
        return

    limits = list(col_char_limits) if col_char_limits is not None else [int(max_cell_chars)] * len(headers)
    if len(limits) < len(headers):
        limits.extend([int(max_cell_chars)] * (len(headers) - len(limits)))
    wrapped_headers = [_wrap_cell(header, limits[idx]) for idx, header in enumerate(headers)]

    for page_start in range(0, len(rows), int(rows_per_page)):
        chunk = rows[page_start : page_start + int(rows_per_page)]
        wrapped_rows: list[list[str]] = []
        for row in chunk:
            wrapped_rows.append(
                [
                    _wrap_cell(value, limits[idx] if idx < len(limits) else int(max_cell_chars))
                    for idx, value in enumerate(row)
                ]
            )

        page_no = page_start // int(rows_per_page) + 1
        page_count = (len(rows) + int(rows_per_page) - 1) // int(rows_per_page)
        page_title = f"{title} (page {page_no}/{page_count})" if page_count > 1 else title

        fig, ax = plt.subplots(figsize=(18, 10))
        ax.axis("off")
        ax.set_title(page_title, fontsize=14, pad=12)
        table = ax.table(
            cellText=wrapped_rows,
            colLabels=wrapped_headers,
            colWidths=list(col_widths) if col_widths is not None else None,
            cellLoc="left",
            loc="upper center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(float(fontsize))

        # Dynamic row heights based on wrapped line count.
        for (row_idx, col_idx), cell in table.get_celld().items():
            txt = cell.get_text().get_text()
            n_lines = max(1, txt.count("\n") + 1)
            if row_idx == 0:
                cell.set_facecolor("#E8EEF7")
                cell.set_height(0.05 * n_lines)
                cell.set_text_props(weight="bold")
            else:
                cell.set_height(0.04 * n_lines)
            if col_idx < len(limits):
                cell.set_linewidth(0.6)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _render_text_pages(
    pdf: PdfPages,
    title: str,
    lines: Sequence[str],
    *,
    width: int = 145,
    fontsize: float = 10.0,
) -> None:
    expanded: list[str] = []
    for line in lines:
        raw = str(line)
        if not raw:
            expanded.append("")
            continue
        wrapped = textwrap.wrap(raw, width=width, break_long_words=False, break_on_hyphens=True)
        if not wrapped:
            expanded.append("")
        else:
            expanded.extend(wrapped)

    lines_per_page = 40
    for page_start in range(0, len(expanded), lines_per_page):
        chunk = expanded[page_start : page_start + lines_per_page]
        page_no = page_start // lines_per_page + 1
        page_count = (len(expanded) + lines_per_page - 1) // lines_per_page
        page_title = f"{title} (page {page_no}/{page_count})" if page_count > 1 else title

        fig, ax = plt.subplots(figsize=(15.5, 9))
        ax.axis("off")
        ax.set_title(page_title, fontsize=15, pad=10)
        y = 0.96
        for line in chunk:
            ax.text(
                0.01,
                y,
                line,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=float(fontsize),
                family="monospace",
            )
            y -= 0.023
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _rows_from_group_stats(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    headers = [
        "group",
        "N",
        "N_abs",
        "mean|dE|",
        "median|dE|",
        "pass@1e-1",
        "pass@1e-2",
        "pass@1e-3",
        "pass@1e-4",
    ]
    out_rows: list[list[str]] = []
    for row in rows:
        group_parts = []
        for key in ["kind", "category", "zero_case_stratum", "g_stratum", "omega_stratum"]:
            if key in row:
                group_parts.append(f"{key}={row[key]}")
        group_label = " | ".join(group_parts) if group_parts else "overall"
        pass_counts = row.get("pass_counts", {})
        out_rows.append(
            [
                group_label,
                str(row.get("N", 0)),
                str(row.get("N_abs_delta", 0)),
                _fmt_float(row.get("mean_abs_delta"), 3),
                _fmt_float(row.get("median_abs_delta"), 3),
                _pass_display(pass_counts, "1e-01"),
                _pass_display(pass_counts, "1e-02"),
                _pass_display(pass_counts, "1e-03"),
                _pass_display(pass_counts, "1e-04"),
            ]
        )
    return headers, out_rows


def _pass_display(pass_counts: dict[str, Any], key: str) -> str:
    entry = pass_counts.get(key, {})
    count = int(entry.get("count", 0))
    denominator = int(entry.get("denominator", 0))
    return f"{count}/{denominator}"


def _plot_abs_delta_hist_by_kind(pdf: PdfPages, runs: Sequence[dict[str, Any]]) -> None:
    kind_to_values: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        value = _safe_float(run.get("abs_delta"))
        if value is not None:
            kind_to_values[str(run.get("kind", ""))].append(float(value))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for axis, kind in zip(axes, ["ADAPT", "VQE"]):
        values = kind_to_values.get(kind, [])
        if not values:
            axis.axis("off")
            axis.text(0.5, 0.5, f"No finite |dE| values for {kind}", ha="center", va="center")
            continue
        bins = np.logspace(-8, 1, 28)
        axis.hist(values, bins=bins, color="#0072B2" if kind == "ADAPT" else "#D55E00", alpha=0.85)
        axis.set_xscale("log")
        axis.set_title(f"{kind}: |dE| distribution")
        axis.set_xlabel("|dE|")
        axis.set_ylabel("count")
        axis.grid(alpha=0.25, which="both")
    fig.suptitle("Distribution Analysis: |dE| Histograms by Kind", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_overview_boxplot(pdf: PdfPages, runs: Sequence[dict[str, Any]]) -> None:
    categories = ["Polaron PAOP", "HVA (layer-wise)", "HVA-adapt", "UCCSD", "Full-Hamiltonian"]
    data: list[list[float]] = []
    labels: list[str] = []
    for category in categories:
        values = [
            float(run["abs_delta"])
            for run in runs
            if str(run.get("category")) == category and _is_finite_number(run.get("abs_delta"))
        ]
        if values:
            data.append(values)
            labels.append(f"{category}\nN={len(values)}")

    if not data:
        _render_text_pages(pdf, "Overview: |dE| by category", ["No finite |dE| values available."])
        return

    fig, ax = plt.subplots(figsize=(14.5, 7.5))
    box = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=True)
    palette = ["#009E73", "#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(palette[idx % len(palette)])
        patch.set_alpha(0.6)
    ax.set_yscale("log")
    ax.set_title("Overview: |dE| distribution by method category")
    ax.set_ylabel("|dE| (log scale)")
    ax.grid(alpha=0.25, which="both")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_category_by_g(
    pdf: PdfPages,
    runs: Sequence[dict[str, Any]],
    *,
    metric: str,
    title: str,
    cmap: str,
) -> None:
    categories = ["Polaron PAOP", "HVA (layer-wise)", "HVA-adapt", "UCCSD", "Full-Hamiltonian"]
    g_strata = ["g=0", "0<g<=0.1", "0.1<g<=0.5", "0.5<g<=1.0", "1.0<g<=2.0", "g>2.0"]
    values = np.full((len(categories), len(g_strata)), np.nan, dtype=float)
    annotations = [["" for _ in g_strata] for _ in categories]

    for row_idx, category in enumerate(categories):
        for col_idx, g_stratum in enumerate(g_strata):
            bucket = [
                run
                for run in runs
                if str(run.get("category")) == category
                and str(run.get("g_stratum")) == g_stratum
                and _is_finite_number(run.get("abs_delta"))
            ]
            if not bucket:
                annotations[row_idx][col_idx] = "N=0"
                continue
            arr = np.asarray([float(run["abs_delta"]) for run in bucket], dtype=float)
            if metric == "median_abs_delta":
                values[row_idx, col_idx] = float(np.median(arr))
                annotations[row_idx][col_idx] = f"N={len(arr)}"
            elif metric == "pass_rate_1e3":
                passed = float(np.sum(arr <= 1e-3))
                rate = passed / float(len(arr))
                values[row_idx, col_idx] = float(rate)
                annotations[row_idx][col_idx] = f"{int(passed)}/{len(arr)}"
            else:
                raise ValueError(f"unsupported heatmap metric {metric}")

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    matrix = np.array(values, dtype=float)
    if metric == "median_abs_delta":
        # Use log10 transform for readability across orders of magnitude.
        transformed = np.full_like(matrix, np.nan, dtype=float)
        mask = np.isfinite(matrix) & (matrix > 0.0)
        transformed[mask] = np.log10(matrix[mask])
        image = ax.imshow(transformed, aspect="auto", cmap=cmap)
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("log10(median |dE|)")
    else:
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("pass rate @ 1e-3")

    ax.set_xticks(np.arange(len(g_strata)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(g_strata, rotation=25, ha="right")
    ax.set_yticklabels(categories)
    ax.set_title(title)
    ax.set_xlabel("g stratum")
    ax.set_ylabel("category")

    for row_idx in range(len(categories)):
        for col_idx in range(len(g_strata)):
            ann = annotations[row_idx][col_idx]
            ax.text(col_idx, row_idx, ann, ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _log_bins(values: Sequence[float], n_bins: int = 24) -> np.ndarray:
    finite = [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0]
    if not finite:
        return np.logspace(-8, 1, int(n_bins))
    v_min = max(min(finite), 1e-12)
    v_max = max(finite)
    if v_max <= v_min:
        v_max = v_min * 10.0
    return np.logspace(math.log10(v_min), math.log10(v_max), int(n_bins))


def _plot_histograms_by_category(pdf: PdfPages, runs: Sequence[dict[str, Any]]) -> None:
    categories = ["Polaron PAOP", "HVA (layer-wise)", "HVA-adapt", "UCCSD", "Full-Hamiltonian"]
    g_strata = ["g=0", "0<g<=0.1", "0.1<g<=0.5", "0.5<g<=1.0", "1.0<g<=2.0"]
    colors = ["#999999", "#56B4E9", "#009E73", "#E69F00", "#D55E00"]

    for category in categories:
        bucket = [
            run
            for run in runs
            if str(run.get("category")) == category and _is_finite_number(run.get("abs_delta"))
        ]
        all_values = [float(run["abs_delta"]) for run in bucket]
        bins = _log_bins(all_values, n_bins=22)

        fig, ax = plt.subplots(figsize=(14.5, 7.5))
        if not bucket:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No finite |dE| values for category: {category}", ha="center", va="center")
        else:
            stacked_values: list[list[float]] = []
            labels: list[str] = []
            stacked_colors: list[str] = []
            for idx, g_stratum in enumerate(g_strata):
                values = [float(run["abs_delta"]) for run in bucket if str(run.get("g_stratum")) == g_stratum]
                stacked_values.append(values)
                labels.append(f"{g_stratum} (N={len(values)})")
                stacked_colors.append(colors[idx % len(colors)])

            ax.hist(
                stacked_values,
                bins=bins,
                stacked=True,
                color=stacked_colors,
                alpha=0.90,
                label=labels,
            )
            ax.set_xscale("log")
            ax.set_xlabel("|dE| (log bins)")
            ax.set_ylabel("count")
            ax.grid(alpha=0.25, which="both")
            ax.legend(loc="best", fontsize=8)
            omega_observed = sorted({str(run.get("omega_stratum")) for run in bucket})
            ax.set_title(
                f"Histogram by category: {category}\n"
                f"condition: category={category}, stacked by g_stratum, omega_strata={omega_observed}"
            )
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _plot_histograms_by_g_stratum(pdf: PdfPages, runs: Sequence[dict[str, Any]]) -> None:
    g_strata = ["g=0", "0<g<=0.1", "0.1<g<=0.5", "0.5<g<=1.0", "1.0<g<=2.0"]
    categories = ["Polaron PAOP", "HVA (layer-wise)", "HVA-adapt", "UCCSD", "Full-Hamiltonian"]
    colors = ["#009E73", "#0072B2", "#E69F00", "#CC79A7", "#D55E00"]

    for g_stratum in g_strata:
        bucket = [
            run
            for run in runs
            if str(run.get("g_stratum")) == g_stratum and _is_finite_number(run.get("abs_delta"))
        ]
        all_values = [float(run["abs_delta"]) for run in bucket]
        bins = _log_bins(all_values, n_bins=22)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), sharey=True)
        for axis, kind in zip(axes, ["ADAPT", "VQE"]):
            kind_bucket = [run for run in bucket if str(run.get("kind")) == kind]
            if not kind_bucket:
                axis.axis("off")
                axis.text(0.5, 0.5, f"No finite |dE| values\nfor {kind} in {g_stratum}", ha="center", va="center")
                continue

            stacked_values: list[list[float]] = []
            labels: list[str] = []
            stacked_colors: list[str] = []
            for idx, category in enumerate(categories):
                values = [float(run["abs_delta"]) for run in kind_bucket if str(run.get("category")) == category]
                stacked_values.append(values)
                labels.append(f"{category} (N={len(values)})")
                stacked_colors.append(colors[idx % len(colors)])

            axis.hist(
                stacked_values,
                bins=bins,
                stacked=True,
                color=stacked_colors,
                alpha=0.9,
                label=labels,
            )
            axis.set_xscale("log")
            axis.set_xlabel("|dE| (log bins)")
            axis.set_title(f"{kind} | g_stratum={g_stratum}")
            axis.grid(alpha=0.25, which="both")
            axis.legend(loc="best", fontsize=7)

        axes[0].set_ylabel("count")
        omega_observed = sorted({str(run.get("omega_stratum")) for run in bucket})
        fig.suptitle(
            f"Histogram by g_stratum: {g_stratum}\n"
            f"condition: g_stratum={g_stratum}, split by kind, stacked by category, omega_strata={omega_observed}",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _infer_model_target_for_run(run: dict[str, Any]) -> str:
    problem = str(run.get("problem_setting", "")).lower().strip()
    method = str(run.get("method", "")).lower()
    pool = str(run.get("pool", "")).lower()
    if problem == "hh" or "hh" in method or "hh" in pool:
        return "hh_explicit"
    if "hubbard" in method:
        return "hubbard_named"
    if "uccsd" in method and "hh" not in method:
        return "generic_or_hubbard_uccsd"
    return "unknown"


def _plot_uccsd_diagnostics(pdf: PdfPages, runs: Sequence[dict[str, Any]]) -> None:
    uccsd_runs = [run for run in runs if str(run.get("category")) == "UCCSD"]
    for run in uccsd_runs:
        run["model_target_inference"] = _infer_model_target_for_run(run)

    if not uccsd_runs:
        _render_text_pages(pdf, "UCCSD Diagnostics", ["No UCCSD runs observed in current artifacts."])
        return

    # Page 1: abs_delta vs g, colored by inferred model target.
    fig, ax = plt.subplots(figsize=(13.5, 7))
    target_types = sorted({str(run.get("model_target_inference")) for run in uccsd_runs})
    color_map = {
        "hh_explicit": "#0072B2",
        "hubbard_named": "#D55E00",
        "generic_or_hubbard_uccsd": "#CC79A7",
        "unknown": "#666666",
    }
    for target in target_types:
        bucket = [
            run
            for run in uccsd_runs
            if str(run.get("model_target_inference")) == target and _is_finite_number(run.get("abs_delta"))
        ]
        if not bucket:
            continue
        x = np.asarray([float(run.get("g")) for run in bucket if _is_finite_number(run.get("g"))], dtype=float)
        y = np.asarray(
            [
                float(run.get("abs_delta"))
                for run in bucket
                if _is_finite_number(run.get("g")) and _is_finite_number(run.get("abs_delta"))
            ],
            dtype=float,
        )
        if x.size == 0 or y.size == 0:
            continue
        ax.scatter(x, y, s=50, alpha=0.85, label=f"{target} (N={len(y)})", color=color_map.get(target, "#666666"))

    ax.set_yscale("log")
    ax.set_xlabel("g")
    ax.set_ylabel("|dE|")
    ax.set_title("UCCSD diagnostics: |dE| vs g, colored by inferred model target")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Page 2: exact vs VQE energy with y=x reference.
    fig, ax = plt.subplots(figsize=(13.5, 7))
    points = [
        run
        for run in uccsd_runs
        if _is_finite_number(run.get("exact")) and _is_finite_number(run.get("energy"))
    ]
    if not points:
        ax.axis("off")
        ax.text(0.5, 0.5, "No finite exact/energy pairs for UCCSD.", ha="center", va="center")
    else:
        for target in target_types:
            bucket = [run for run in points if str(run.get("model_target_inference")) == target]
            if not bucket:
                continue
            x = np.asarray([float(run.get("exact")) for run in bucket], dtype=float)
            y = np.asarray([float(run.get("energy")) for run in bucket], dtype=float)
            ax.scatter(x, y, s=45, alpha=0.85, label=f"{target} (N={len(bucket)})", color=color_map.get(target, "#666666"))
        all_exact = np.asarray([float(run.get("exact")) for run in points], dtype=float)
        min_v = float(np.min(all_exact))
        max_v = float(np.max(all_exact))
        ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="#111111", linewidth=1.5, label="y=x")
        ax.set_xlabel("exact energy")
        ax.set_ylabel("VQE energy")
        ax.set_title("UCCSD diagnostics: VQE energy vs exact energy")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Page 3: compact textual interpretation.
    lines = [
        f"UCCSD observed run count: {len(uccsd_runs)}",
        "Model-target inference counts:",
    ]
    counts = Counter(str(run.get("model_target_inference")) for run in uccsd_runs)
    for key, value in sorted(counts.items(), key=lambda item: item[0]):
        lines.append(f"  - {key}: {value}")
    finite = [float(run["abs_delta"]) for run in uccsd_runs if _is_finite_number(run.get("abs_delta"))]
    lines.append("")
    lines.append(f"UCCSD finite |dE| count: {len(finite)}")
    lines.append(f"UCCSD median |dE|: {_fmt_float(median(finite) if finite else None, 4)}")
    lines.append(f"UCCSD pass@1e-3: {sum(1 for v in finite if v <= 1e-3)}/{len(finite)}")
    lines.append("")
    lines.append("Interpretation warning:")
    lines.append("  UCCSD results should be interpreted with model-target context from method metadata.")
    _render_text_pages(pdf, "UCCSD Diagnostics Summary", lines)


def _plot_g_stratum_performance(pdf: PdfPages, runs: Sequence[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    categories = sorted({str(run.get("category", "")) for run in runs})
    for run in runs:
        value = _safe_float(run.get("abs_delta"))
        if value is None:
            continue
        grouped[(str(run.get("category", "")), str(run.get("g_stratum", "")))].append(float(value))

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(G_STRATA_ORDER), dtype=float)
    for idx, category in enumerate(categories):
        y_values = []
        for g_stratum in G_STRATA_ORDER:
            vals = grouped.get((category, g_stratum), [])
            y_values.append(float(median(vals)) if vals else np.nan)
        ax.plot(x, y_values, marker="o", linewidth=1.8, label=category)

    ax.set_xticks(x)
    ax.set_xticklabels(G_STRATA_ORDER, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("median |dE|")
    ax.set_xlabel("g stratum")
    ax.set_title("Performance vs Coupling Regime: median |dE| by method category")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_paop_usage(pdf: PdfPages, paop_summary: dict[str, Any]) -> None:
    variants = paop_summary.get("variants", [])
    if not variants:
        _render_text_pages(pdf, "Polaron PAOP Usage", ["No PAOP runs observed in current artifacts."])
        return

    labels = [str(item["variant"]) for item in variants]
    counts = [int(item.get("N", 0)) for item in variants]
    medians = [
        _safe_float(item.get("median_abs_delta")) if _safe_float(item.get("median_abs_delta")) is not None else np.nan
        for item in variants
    ]
    x = np.arange(len(labels), dtype=float)

    fig, ax1 = plt.subplots(figsize=(14.5, 7))
    ax1.bar(x, counts, color="#009E73", alpha=0.85, label="N")
    ax1.set_ylabel("run count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_title("Polaron PAOP usage and median |dE| by variant")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, medians, color="#CC79A7", marker="o", linewidth=2.0, label="median |dE|")
    ax2.set_yscale("log")
    ax2.set_ylabel("median |dE|")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _coverage_table_rows(coverage: dict[str, Any]) -> tuple[list[str], list[list[str]], list[str], list[list[str]]]:
    zero_headers = ["L", "boundary", "ordering", "zero_case_stratum", "count", "status"]
    zero_rows: list[list[str]] = []
    for row in coverage.get("zero_case_matrix", []):
        zero_rows.append(
            [
                str(row["L"]),
                str(row["boundary"]),
                str(row["ordering"]),
                str(row["zero_case_stratum"]),
                str(row["count"]),
                str(row["status"]),
            ]
        )

    full_headers = ["L", "boundary", "ordering", "g_stratum", "omega_stratum", "zero_case_stratum", "count", "status"]
    full_rows: list[list[str]] = []
    for row in coverage.get("stratum_matrix", []):
        full_rows.append(
            [
                str(row["L"]),
                str(row["boundary"]),
                str(row["ordering"]),
                str(row["g_stratum"]),
                str(row["omega_stratum"]),
                str(row["zero_case_stratum"]),
                str(row["count"]),
                str(row["status"]),
            ]
        )
    return zero_headers, zero_rows, full_headers, full_rows


def _build_basic_summary(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    kind_counts = Counter(str(run.get("kind", "")) for run in runs)
    category_counts = Counter(str(run.get("category", "")) for run in runs)
    zero_counts = Counter(str(run.get("zero_case_stratum", "")) for run in runs)
    g_counts = Counter(str(run.get("g_stratum", "")) for run in runs)
    omega_counts = Counter(str(run.get("omega_stratum", "")) for run in runs)
    boundary_counts = Counter(str(run.get("boundary", "")) for run in runs)
    ordering_counts = Counter(str(run.get("ordering", "")) for run in runs)
    l_counts = Counter(int(_safe_float(run.get("L"), default=-1) or -1) for run in runs)
    abs_values = [float(run["abs_delta"]) for run in runs if _is_finite_number(run.get("abs_delta"))]

    return {
        "num_runs": int(len(runs)),
        "kind_counts": dict(kind_counts),
        "category_counts": dict(category_counts),
        "zero_case_counts": dict(zero_counts),
        "g_stratum_counts": dict(g_counts),
        "omega_stratum_counts": dict(omega_counts),
        "boundary_counts": dict(boundary_counts),
        "ordering_counts": dict(ordering_counts),
        "L_counts": dict(l_counts),
        "num_with_abs_delta": int(len(abs_values)),
        "mean_abs_delta": float(mean(abs_values)) if abs_values else None,
        "median_abs_delta": float(median(abs_values)) if abs_values else None,
        "max_abs_delta": float(max(abs_values)) if abs_values else None,
        "min_abs_delta": float(min(abs_values)) if abs_values else None,
    }


def _build_pdf_report(
    *,
    output_pdf: Path,
    runs: Sequence[dict[str, Any]],
    config: dict[str, Any],
    provenance: dict[str, Any],
    coverage: dict[str, Any],
    method_summary: dict[str, Any],
    threshold_summary: dict[str, Any],
    stratified_distributions: dict[str, Any],
    configured_defaults: dict[str, Any],
    termwise_evidence: dict[str, Any],
    paop_summary: dict[str, Any],
    parameter_grid_rows: Sequence[dict[str, Any]],
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    basic = _build_basic_summary(runs)

    with PdfPages(str(output_pdf)) as pdf:
        # 1) Executive summary (text)
        uccsd_runs = [run for run in runs if str(run.get("category")) == "UCCSD"]
        uccsd_finite = [float(run["abs_delta"]) for run in uccsd_runs if _is_finite_number(run.get("abs_delta"))]
        uccsd_pass_1e3 = sum(1 for value in uccsd_finite if value <= 1e-3)
        uccsd_target_counts = Counter(str(run.get("model_target_inference", "unknown")) for run in uccsd_runs)
        exec_lines = [
            f"Generated UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "",
            "Scope:",
            "  - Observed artifacts are primary evidence",
            "  - Configured defaults/evidence are reported separately",
            "  - Zero-parameter strata are explicit, including Not observed rows",
            "",
            f"Observed run count: {basic['num_runs']}",
            f"Kind counts: {basic['kind_counts']}",
            f"Category counts: {basic['category_counts']}",
            f"L coverage: {basic['L_counts']}",
            f"Boundary coverage: {basic['boundary_counts']}",
            f"Ordering coverage: {basic['ordering_counts']}",
            f"zero_case_stratum coverage: {basic['zero_case_counts']}",
            f"g_stratum coverage: {basic['g_stratum_counts']}",
            f"omega_stratum coverage: {basic['omega_stratum_counts']}",
            "",
            f"Finite |dE| count: {basic['num_with_abs_delta']}",
            f"mean |dE|: {_fmt_float(basic['mean_abs_delta'], 4)}",
            f"median |dE|: {_fmt_float(basic['median_abs_delta'], 4)}",
            f"min |dE|: {_fmt_float(basic['min_abs_delta'], 4)}",
            f"max |dE|: {_fmt_float(basic['max_abs_delta'], 4)}",
            "",
            "UCCSD diagnostic summary:",
            f"  - UCCSD runs: {len(uccsd_runs)}",
            f"  - UCCSD finite |dE|: {len(uccsd_finite)}",
            f"  - UCCSD pass@1e-3: {uccsd_pass_1e3}/{len(uccsd_finite)}",
            f"  - UCCSD inferred model targets: {dict(uccsd_target_counts)}",
            "",
            "Caveats:",
            "  - This report does not fabricate missing experiments.",
            "  - Zero-case strata are reported as missing when not present in artifacts.",
            "  - Method metadata can indicate model-target mismatch risk (see UCCSD diagnostics section).",
        ]
        _render_text_pages(pdf, "HH Convergence Audit Report v2: Executive Summary", exec_lines)

        # 2) Provenance + notation (text)
        notation_lines = [
            "Data provenance and inclusion rules",
            "",
            f"artifact_dirs: {provenance.get('artifact_dirs', [])}",
            f"strict_hh_only: {provenance.get('strict_hh_only')}",
            f"json_files_scanned: {provenance.get('json_files_scanned')}",
            f"json_files_parsed: {provenance.get('json_files_parsed')}",
            f"json_files_with_included_runs: {provenance.get('json_files_with_included_runs')}",
            f"run_count: {provenance.get('run_count')}",
            "",
            "Notation and definitions:",
            "  - N: total runs in group",
            "  - N_abs: runs with finite abs_delta values",
            "  - pass@X: count of finite abs_delta <= X over denominator N_abs",
            "  - zero_case_stratum in {g=0 & omega=0, exactly_one_zero, both_nonzero, missing_parameter}",
            "  - g_stratum in {g=0, 0<g<=0.1, 0.1<g<=0.5, 0.5<g<=1.0, 1.0<g<=2.0, g>2.0, missing_parameter}",
            "  - omega_stratum in {omega=0, 0<omega<=0.2, omega>0.2, missing_parameter}",
            "",
            "Stable run sorting:",
            "  kind, category, L, g, omega0, boundary, ordering, path",
        ]
        _render_text_pages(pdf, "Data Provenance and Inclusion Rules", notation_lines)

        # 3) Coverage matrix (one concise table focused on missing/observed zero-case strata)
        zero_headers, zero_rows, full_headers, full_rows = _coverage_table_rows(coverage)
        _table_pages(
            pdf,
            title="Coverage Matrix: zero_case_stratum by (L, boundary, ordering)",
            headers=zero_headers,
            rows=zero_rows,
            rows_per_page=max(12, int(config["rows_per_page"])),
            max_cell_chars=int(config["max_cell_chars"]),
            col_widths=[0.06, 0.11, 0.11, 0.26, 0.08, 0.10],
            col_char_limits=[4, 12, 12, 24, 6, 12],
            fontsize=8.4,
        )

        # 4) Convergence threshold definitions + configured defaults (text + one table)
        threshold_lines = [
            "Observed threshold summary uses finite abs_delta runs as denominator (N_abs).",
            "Threshold set: 1e-1, 1e-2, 1e-3, 1e-4.",
            "",
            "Configured defaults/evidence are separate and not mixed into observed run counts.",
            f"Configured defaults enabled: {configured_defaults.get('enabled')}",
            f"Configured sources: {configured_defaults.get('sources', [])}",
        ]
        _render_text_pages(pdf, "Convergence Threshold Definitions", threshold_lines)

        if configured_defaults.get("enabled"):
            cfg_rows = [
                [entry["key"], entry["value"], entry["section"], entry["source"], entry.get("note", "")]
                for entry in configured_defaults.get("entries", [])
            ]
            _table_pages(
                pdf,
                title="Configured Defaults/Evidence Extracted from Suite Docs/Scripts/Code",
                headers=["key", "value", "section", "source", "note"],
                rows=cfg_rows,
                rows_per_page=max(16, int(config["rows_per_page"])),
                max_cell_chars=int(config["max_cell_chars"]),
                col_widths=[0.31, 0.13, 0.18, 0.26, 0.08],
                col_char_limits=[30, 16, 22, 36, 20],
                fontsize=7.7,
            )

        # 5) Plot-first overview pages
        _plot_overview_boxplot(pdf, runs)
        _plot_abs_delta_hist_by_kind(pdf, runs)
        _plot_g_stratum_performance(pdf, runs)
        _plot_heatmap_category_by_g(
            pdf,
            runs,
            metric="median_abs_delta",
            title="Heatmap: median |dE| by category and g stratum",
            cmap="Blues",
        )
        _plot_heatmap_category_by_g(
            pdf,
            runs,
            metric="pass_rate_1e3",
            title="Heatmap: pass rate @1e-3 by category and g stratum",
            cmap="YlGn",
        )

        # 6) Ten histogram pages with explicit parameter sorting:
        #    5 by category + 5 by g_stratum.
        _plot_histograms_by_category(pdf, runs)
        _plot_histograms_by_g_stratum(pdf, runs)

        # 7) UCCSD mismatch diagnostics (focused section)
        _plot_uccsd_diagnostics(pdf, runs)

        # 8) Term-wise/component-wise evidence status (text only + compact implemented table)
        term_lines = [
            "Observed vs implemented term-wise/component-wise evidence:",
            f"  observed_artifact_markers: {termwise_evidence.get('observed_count', 0)}",
            "  Note: observed markers are string matches over method/pool/category/path fields.",
            "",
            "Implemented path checks are code-path evidence only (not execution evidence).",
        ]
        _render_text_pages(pdf, "Term-wise / Component-wise Evidence Status", term_lines)

        impl_rows = [
            [
                row.get("path"),
                row.get("exists"),
                row.get("marker"),
                row.get("marker_present"),
            ]
            for row in termwise_evidence.get("implemented_paths", [])
        ]
        _table_pages(
            pdf,
            title="Implemented code-path checks for term-wise/component-wise support",
            headers=["path", "exists", "marker", "marker_present"],
            rows=impl_rows,
            rows_per_page=int(config["rows_per_page"]),
            max_cell_chars=int(config["max_cell_chars"]),
            col_widths=[0.62, 0.08, 0.20, 0.10],
            col_char_limits=[90, 8, 30, 12],
            fontsize=8.3,
        )

        # 9) PAOP usage detail (one plot + compact table)
        _plot_paop_usage(pdf, paop_summary)
        paop_headers = [
            "variant",
            "N",
            "N_abs",
            "mean|dE|",
            "median|dE|",
            "pass@1e-1",
            "pass@1e-2",
            "pass@1e-3",
            "pass@1e-4",
        ]
        paop_rows: list[list[str]] = []
        for row in paop_summary.get("variants", []):
            pass_counts = row.get("pass_counts", {})
            paop_rows.append(
                [
                    str(row.get("variant", "unknown")),
                    str(row.get("N", 0)),
                    str(row.get("N_abs_delta", 0)),
                    _fmt_float(row.get("mean_abs_delta"), 3),
                    _fmt_float(row.get("median_abs_delta"), 3),
                    _pass_display(pass_counts, "1e-01"),
                    _pass_display(pass_counts, "1e-02"),
                    _pass_display(pass_counts, "1e-03"),
                    _pass_display(pass_counts, "1e-04"),
                ]
            )
        _table_pages(
            pdf,
            title="Polaron PAOP variants: counts and threshold performance",
            headers=paop_headers,
            rows=paop_rows,
            rows_per_page=int(config["rows_per_page"]),
            max_cell_chars=int(config["max_cell_chars"]),
            col_widths=[0.39, 0.07, 0.08, 0.11, 0.11, 0.09, 0.09, 0.09, 0.09],
            col_char_limits=[46, 6, 6, 10, 10, 9, 9, 9, 9],
            fontsize=8.1,
        )

        # 10) Appendix: parameter grid detail (single compact table)
        grid_rows = [
            [
                row["L"],
                row["boundary"],
                row["ordering"],
                _fmt_float(row["g"], 6),
                _fmt_float(row["omega0"], 6),
                row["count"],
                row["categories"],
                row["methods"],
            ]
            for row in parameter_grid_rows
        ]
        _table_pages(
            pdf,
            title="Appendix: Unique parameter grid and repetition counts",
            headers=["L", "boundary", "ordering", "g", "omega0", "N", "categories", "methods"],
            rows=grid_rows,
            rows_per_page=int(config["rows_per_page"]),
            max_cell_chars=int(config["max_cell_chars"]),
            col_widths=[0.05, 0.10, 0.10, 0.07, 0.08, 0.05, 0.20, 0.30],
            col_char_limits=[3, 10, 10, 8, 8, 5, 28, 44],
            fontsize=7.9,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HH convergence audit report v2.")
    parser.add_argument(
        "--artifact-dirs",
        type=str,
        default="artifacts,hh_adapt_vqe_validation_suite/artifacts",
        help="Comma-separated artifact directories (relative to script dir or absolute).",
    )
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--rows-per-page", type=int, default=14)
    parser.add_argument("--max-cell-chars", type=int, default=28)
    parser.add_argument("--include-config-evidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-hh-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact_dirs = _resolve_artifact_dirs(args.artifact_dirs)

    runs, provenance = _extract_hh_runs(
        artifact_dirs,
        strict_hh_only=bool(args.strict_hh_only),
    )
    if not runs:
        raise RuntimeError("No HH runs detected in scanned artifact directories.")
    for run in runs:
        run["model_target_inference"] = _infer_model_target_for_run(run)

    configured_defaults = _extract_configured_defaults(include=bool(args.include_config_evidence))
    coverage = _build_coverage(runs)
    method_summary = _build_method_summary(runs)
    threshold_summary = _build_threshold_summary(runs)
    stratified_distributions = _build_stratified_distributions(runs)
    termwise_evidence = _termwise_componentwise_evidence(runs)
    paop_summary = _paop_usage_summary(runs)
    parameter_grid_rows = _build_parameter_grid_rows(runs)

    config = {
        "artifact_dirs": [str(path) for path in artifact_dirs],
        "output_pdf": str(Path(args.output_pdf).resolve()),
        "output_csv": str(Path(args.output_csv).resolve()),
        "output_json": str(Path(args.output_json).resolve()),
        "rows_per_page": int(args.rows_per_page),
        "max_cell_chars": int(args.max_cell_chars),
        "include_config_evidence": bool(args.include_config_evidence),
        "strict_hh_only": bool(args.strict_hh_only),
        "thresholds": [f"{value:.0e}" for value in THRESHOLDS],
    }

    summary_payload = {
        "config": config,
        "provenance": provenance,
        "configured_defaults": configured_defaults,
        "coverage": coverage,
        "method_summary": method_summary,
        "threshold_summary": threshold_summary,
        "stratified_distributions": stratified_distributions,
        "termwise_componentwise_evidence": termwise_evidence,
        "paop_usage": paop_summary,
        "parameter_grid": parameter_grid_rows,
        "runs": runs,
    }

    _write_csv_summary(Path(args.output_csv).resolve(), runs)
    _write_json_summary(Path(args.output_json).resolve(), summary_payload)
    _build_pdf_report(
        output_pdf=Path(args.output_pdf).resolve(),
        runs=runs,
        config=config,
        provenance=provenance,
        coverage=coverage,
        method_summary=method_summary,
        threshold_summary=threshold_summary,
        stratified_distributions=stratified_distributions,
        configured_defaults=configured_defaults,
        termwise_evidence=termwise_evidence,
        paop_summary=paop_summary,
        parameter_grid_rows=parameter_grid_rows,
    )

    print(f"Wrote {Path(args.output_pdf).resolve()}")
    print(f"Wrote {Path(args.output_csv).resolve()}")
    print(f"Wrote {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
