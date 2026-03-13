#!/usr/bin/env python3
"""Fixed-handoff HH replay optimizer probe.

Wrapper-only diagnostic: hold one replay scaffold fixed and compare replay
optimizers on the exact same handoff state. This is intentionally outside the
canonical staged runner and does not alter the production replay CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.hardcoded import hh_vqe_from_adapt_family as replay_mod
from src.quantum.vqe_latex_python_pairs import vqe_minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_JSON = REPO_ROOT / "artifacts/json/hh_fixed_handoff_replay_optimizer_probe.json"
_DEFAULT_OUTPUT_CSV = REPO_ROOT / "artifacts/json/hh_fixed_handoff_replay_optimizer_probe.csv"
_DEFAULT_TAG = "hh_fixed_handoff_replay_optimizer_probe"
_ALLOWED_DETERMINISTIC_METHODS = ("L-BFGS-B", "SLSQP", "Powell")


@dataclass(frozen=True)
class ReplayOptimizerProbeConfig:
    adapt_input_json: Path
    output_json: Path = _DEFAULT_OUTPUT_JSON
    output_csv: Path = _DEFAULT_OUTPUT_CSV
    tag: str = _DEFAULT_TAG
    generator_family: str = "match_adapt"
    fallback_family: str = "full_meta"
    legacy_paop_key: str = "paop_lf_std"
    replay_seed_policy: str = "auto"
    replay_continuation_mode: str = "legacy"
    reps: int | None = None
    seed: int = 7
    baseline_restarts: int = 6
    baseline_maxiter: int = 4000
    heavy_restarts: int = 10
    heavy_maxiter: int = 8000
    deterministic_method: str = "L-BFGS-B"
    deterministic_restarts: int = 6
    deterministic_maxiter: int = 4000
    progress_every_s: float = 60.0
    wallclock_cap_s: int = 43200
    energy_backend: str = "one_apply_compiled"
    spsa_a: float = 0.2
    spsa_c: float = 0.1
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    spsa_A: float = 10.0
    spsa_avg_last: int = 0
    spsa_eval_repeats: int = 1
    spsa_eval_agg: str = "mean"


@dataclass(frozen=True)
class ProbeVariant:
    label: str
    method: str
    restarts: int
    maxiter: int
    seed_offset: int = 0


_MATH_DELTA_ABS = "Δ_abs := |E(θ) - E_exact|"
_MATH_DELTA_IMPROVE = "Δ_improve(seed→variant) := Δ_abs(seed) - Δ_abs(variant)"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(v) for k, v in row.items()})


def _variant_specs(cfg: ReplayOptimizerProbeConfig) -> tuple[ProbeVariant, ...]:
    det_method = str(cfg.deterministic_method).strip()
    if det_method not in _ALLOWED_DETERMINISTIC_METHODS:
        raise ValueError(
            f"deterministic_method must be one of {_ALLOWED_DETERMINISTIC_METHODS}; got {det_method!r}."
        )
    return (
        ProbeVariant(
            label="baseline_spsa",
            method="SPSA",
            restarts=int(cfg.baseline_restarts),
            maxiter=int(cfg.baseline_maxiter),
            seed_offset=0,
        ),
        ProbeVariant(
            label="heavy_spsa",
            method="SPSA",
            restarts=int(cfg.heavy_restarts),
            maxiter=int(cfg.heavy_maxiter),
            seed_offset=0,
        ),
        ProbeVariant(
            label=f"deterministic_{det_method.lower().replace('-', '_')}",
            method=det_method,
            restarts=int(cfg.deterministic_restarts),
            maxiter=int(cfg.deterministic_maxiter),
            seed_offset=0,
        ),
    )


def _base_replay_args(cfg: ReplayOptimizerProbeConfig) -> list[str]:
    argv = [
        "--adapt-input-json",
        str(cfg.adapt_input_json),
        "--tag",
        str(cfg.tag),
        "--generator-family",
        str(cfg.generator_family),
        "--fallback-family",
        str(cfg.fallback_family),
        "--legacy-paop-key",
        str(cfg.legacy_paop_key),
        "--replay-seed-policy",
        str(cfg.replay_seed_policy),
        "--replay-continuation-mode",
        str(cfg.replay_continuation_mode),
        "--seed",
        str(int(cfg.seed)),
        "--restarts",
        str(int(cfg.baseline_restarts)),
        "--maxiter",
        str(int(cfg.baseline_maxiter)),
        "--method",
        "SPSA",
        "--energy-backend",
        str(cfg.energy_backend),
        "--progress-every-s",
        str(float(cfg.progress_every_s)),
        "--wallclock-cap-s",
        str(int(cfg.wallclock_cap_s)),
        "--spsa-a",
        str(float(cfg.spsa_a)),
        "--spsa-c",
        str(float(cfg.spsa_c)),
        "--spsa-alpha",
        str(float(cfg.spsa_alpha)),
        "--spsa-gamma",
        str(float(cfg.spsa_gamma)),
        "--spsa-A",
        str(float(cfg.spsa_A)),
        "--spsa-avg-last",
        str(int(cfg.spsa_avg_last)),
        "--spsa-eval-repeats",
        str(int(cfg.spsa_eval_repeats)),
        "--spsa-eval-agg",
        str(cfg.spsa_eval_agg),
    ]
    if cfg.reps is not None:
        argv.extend(["--reps", str(int(cfg.reps))])
    return argv


def build_base_replay_config(cfg: ReplayOptimizerProbeConfig) -> replay_mod.RunConfig:
    if str(cfg.replay_continuation_mode).strip().lower() != "legacy":
        raise ValueError(
            "Fixed-handoff replay optimizer probe is legacy-only. "
            "Use replay_continuation_mode='legacy' so the probe changes optimizer only, not replay controller logic."
        )
    payload = json.loads(Path(cfg.adapt_input_json).read_text(encoding="utf-8"))
    args = replay_mod.parse_args(_base_replay_args(cfg))
    return replay_mod._build_cfg(args, payload)


_MATH_EXACT_E = "E_exact := payload exact sector energy if available, else exact ED"
def _resolve_exact_energy(cfg: replay_mod.RunConfig, payload: Mapping[str, Any], h_poly: Any) -> tuple[float, str]:
    from_payload = replay_mod._resolve_exact_energy_from_payload(payload)
    if from_payload is not None:
        return float(from_payload), "payload"
    exact = float(
        replay_mod.exact_ground_energy_sector_hh(
            h_poly,
            num_sites=int(cfg.L),
            num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(cfg.boson_encoding),
            indexing=str(cfg.ordering),
        )
    )
    return exact, "ed"


def build_probe_context(cfg: ReplayOptimizerProbeConfig) -> dict[str, Any]:
    base_cfg = build_base_replay_config(cfg)
    psi_ref, payload = replay_mod._read_input_state_and_payload(base_cfg.adapt_input_json)
    family_info = replay_mod._resolve_family(base_cfg, payload)
    h_poly = replay_mod._build_hh_hamiltonian(base_cfg)
    exact_energy, exact_energy_source = _resolve_exact_energy(base_cfg, payload, h_poly)
    replay_ctx = replay_mod.build_replay_ansatz_context(
        base_cfg,
        payload_in=payload,
        psi_ref=psi_ref,
        h_poly=h_poly,
        family_info=family_info,
        e_exact=float(exact_energy),
    )
    return {
        "base_cfg": base_cfg,
        "payload": payload,
        "psi_ref": np.asarray(psi_ref, dtype=complex).reshape(-1).copy(),
        "h_poly": h_poly,
        "exact_energy": float(exact_energy),
        "exact_energy_source": str(exact_energy_source),
        "replay_ctx": replay_ctx,
    }


def _variant_log_path(cfg: ReplayOptimizerProbeConfig, variant: ProbeVariant) -> Path:
    return cfg.output_json.parent / f"{cfg.tag}_{variant.label}.log"


def _scipy_available() -> bool:
    try:
        from scipy.optimize import minimize as _  # type: ignore
    except Exception:
        return False
    return True


def _progress_logger_factory(logger: replay_mod.RunLogger, *, exact_energy: float) -> tuple[Any, list[dict[str, Any]]]:
    progress_tail: list[dict[str, Any]] = []

    def _progress_logger(ev: dict[str, Any]) -> None:
        row = dict(ev)
        e_cur = row.get("energy_current", None)
        e_best = row.get("energy_best_global", None)
        if isinstance(e_cur, (int, float)):
            row["delta_abs_current"] = float(abs(float(e_cur) - float(exact_energy)))
        if isinstance(e_best, (int, float)):
            row["delta_abs_best"] = float(abs(float(e_best) - float(exact_energy)))
        progress_tail.append(row)
        if len(progress_tail) > 200:
            del progress_tail[:-200]
        if str(row.get("event", "")) in {"heartbeat", "restart_end", "run_end", "early_stop_triggered"}:
            logger.log(
                f"VQE {row.get('event')} elapsed_s={float(row.get('elapsed_s', 0.0)):.1f} "
                f"nfev={int(row.get('nfev_so_far', 0))} delta_abs_best={row.get('delta_abs_best')}"
            )

    return _progress_logger, progress_tail


def run_probe_variant(
    workflow_cfg: ReplayOptimizerProbeConfig,
    context: Mapping[str, Any],
    variant: ProbeVariant,
) -> dict[str, Any]:
    base_cfg: replay_mod.RunConfig = context["base_cfg"]
    replay_ctx = context["replay_ctx"]
    exact_energy = float(context["exact_energy"])
    payload = context["payload"]
    psi_ref = np.asarray(context["psi_ref"], dtype=complex).reshape(-1)
    h_poly = context["h_poly"]

    variant_cfg = replace(
        base_cfg,
        method=str(variant.method),
        restarts=int(variant.restarts),
        maxiter=int(variant.maxiter),
        seed=int(base_cfg.seed) + int(variant.seed_offset),
    )
    logger = replay_mod.RunLogger(_variant_log_path(workflow_cfg, variant))
    logger.log(
        f"START variant={variant.label} method={variant.method} restarts={variant.restarts} "
        f"maxiter={variant.maxiter} seed={variant_cfg.seed}"
    )
    seed_energy = float(replay_ctx["seed_energy"])
    seed_delta_abs = float(replay_ctx["seed_delta_abs"])
    logger.log(
        f"Seed baseline: E={seed_energy:.12f} |DeltaE|={seed_delta_abs:.6e} "
        f"family={replay_ctx['family_resolved']} replay_terms={len(replay_ctx['replay_terms'])}"
    )
    progress_logger, progress_tail = _progress_logger_factory(logger, exact_energy=exact_energy)
    wall_hit = False
    t0 = time.perf_counter()

    def _early_stop_checker(ev: dict[str, Any]) -> bool:
        nonlocal wall_hit
        elapsed = float(ev.get("elapsed_s", 0.0))
        if elapsed >= float(variant_cfg.wallclock_cap_s):
            wall_hit = True
            return True
        return False

    common_opt_kwargs = {
        "spsa_a": float(variant_cfg.spsa_a),
        "spsa_c": float(variant_cfg.spsa_c),
        "spsa_alpha": float(variant_cfg.spsa_alpha),
        "spsa_gamma": float(variant_cfg.spsa_gamma),
        "spsa_A": float(variant_cfg.spsa_A),
        "spsa_avg_last": int(variant_cfg.spsa_avg_last),
        "spsa_eval_repeats": int(variant_cfg.spsa_eval_repeats),
        "spsa_eval_agg": str(variant_cfg.spsa_eval_agg),
        "energy_backend": str(variant_cfg.energy_backend),
    }
    result = vqe_minimize(
        h_poly,
        replay_ctx["ansatz"],
        psi_ref,
        restarts=int(variant_cfg.restarts),
        seed=int(variant_cfg.seed),
        initial_point=np.asarray(replay_ctx["seed_theta"], dtype=float),
        use_initial_point_first_restart=True,
        method=str(variant_cfg.method),
        maxiter=int(variant_cfg.maxiter),
        progress_logger=progress_logger,
        progress_every_s=float(variant_cfg.progress_every_s),
        progress_label=f"hh_fixed_handoff_probe:{variant.label}",
        track_history=False,
        emit_theta_in_progress=False,
        return_best_on_keyboard_interrupt=True,
        early_stop_checker=_early_stop_checker,
        **common_opt_kwargs,
    )
    runtime_s = float(time.perf_counter() - t0)
    energy = float(result.energy)
    delta_abs = float(abs(energy - exact_energy))
    row = {
        "tag": str(workflow_cfg.tag),
        "variant": str(variant.label),
        "method": str(variant.method),
        "restarts": int(variant.restarts),
        "maxiter": int(variant.maxiter),
        "seed": int(variant_cfg.seed),
        "energy": float(energy),
        "exact_energy": float(exact_energy),
        "delta_abs": float(delta_abs),
        "seed_energy": float(seed_energy),
        "seed_delta_abs": float(seed_delta_abs),
        "improvement_from_seed": float(seed_delta_abs - delta_abs),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "best_restart": int(result.best_restart),
        "runtime_s": float(runtime_s),
        "wallclock_cap_hit": bool(wall_hit),
        "family_requested": str(replay_ctx["family_info"]["requested"]),
        "family_resolved": str(replay_ctx["family_resolved"]),
        "replay_terms": int(len(replay_ctx["replay_terms"])),
        "adapt_depth": int(len(replay_ctx["adapt_labels"])),
        "scipy_available": bool(_scipy_available()),
        "log_path": str(_variant_log_path(workflow_cfg, variant)),
    }
    if isinstance(payload.get("continuation"), Mapping):
        row["continuation_source"] = str(payload.get("continuation", {}).get("generator_family_resolution", ""))
    logger.log(
        f"DONE variant={variant.label} method={variant.method} abs_delta_e={delta_abs:.6e} "
        f"improvement_from_seed={row['improvement_from_seed']:.6e} runtime_s={runtime_s:.1f}"
    )
    row["progress_tail"] = progress_tail[-20:]
    row["restart_summaries"] = result.restart_summaries
    row["optimizer_memory"] = result.optimizer_memory
    return row


def _material_gain(best_delta_abs: float, baseline_delta_abs: float, *, threshold: float = 5e-3) -> bool:
    return float(baseline_delta_abs - best_delta_abs) >= float(threshold)


def build_probe_payload(cfg: ReplayOptimizerProbeConfig) -> dict[str, Any]:
    context = build_probe_context(cfg)
    rows = [run_probe_variant(cfg, context, variant) for variant in _variant_specs(cfg)]
    baseline_row = next(row for row in rows if row["variant"] == "baseline_spsa")
    for row in rows:
        row["improvement_vs_baseline"] = float(baseline_row["delta_abs"] - row["delta_abs"])
    best_row = min(rows, key=lambda row: float(row["delta_abs"]))
    diagnostic_signal = (
        "optimizer_sensitive"
        if _material_gain(float(best_row["delta_abs"]), float(baseline_row["delta_abs"]))
        else "scaffold_limited_suspected"
    )
    replay_ctx = context["replay_ctx"]
    base_cfg: replay_mod.RunConfig = context["base_cfg"]
    payload = {
        "created_utc": _now_utc(),
        "probe_scope": {
            "local_only": True,
            "noise_enabled": False,
            "mitigation_enabled": False,
            "patch_selection_enabled": False,
            "fixed_handoff": True,
            "production_replay_cli_modified": False,
        },
        "config": asdict(cfg),
        "base_replay_config": asdict(base_cfg),
        "math_contract": {
            "delta_abs": _MATH_DELTA_ABS,
            "delta_improve": _MATH_DELTA_IMPROVE,
            "exact_energy": _MATH_EXACT_E,
        },
        "handoff": {
            "adapt_input_json": str(cfg.adapt_input_json),
            "exact_energy": float(context["exact_energy"]),
            "exact_energy_source": str(context["exact_energy_source"]),
            "seed_energy": float(replay_ctx["seed_energy"]),
            "seed_delta_abs": float(replay_ctx["seed_delta_abs"]),
            "family_requested": str(replay_ctx["family_info"]["requested"]),
            "family_resolved": str(replay_ctx["family_resolved"]),
            "replay_terms": int(len(replay_ctx["replay_terms"])),
            "adapt_depth": int(len(replay_ctx["adapt_labels"])),
            "resolved_seed_policy": str(replay_ctx["resolved_seed_policy"]),
            "handoff_state_kind": str(replay_ctx["handoff_state_kind"]),
        },
        "variant_rows": rows,
        "summary": {
            "best_variant": {k: v for k, v in best_row.items() if k not in {"progress_tail", "restart_summaries", "optimizer_memory"}},
            "baseline_variant": {k: v for k, v in baseline_row.items() if k not in {"progress_tail", "restart_summaries", "optimizer_memory"}},
            "diagnostic_signal": diagnostic_signal,
            "material_gain_threshold": 5e-3,
        },
    }
    return payload


def emit_probe_files(payload: Mapping[str, Any], cfg: ReplayOptimizerProbeConfig) -> None:
    _write_json(cfg.output_json, payload)
    rows = []
    for row in payload.get("variant_rows", []):
        rows.append({k: v for k, v in row.items() if k not in {"progress_tail", "restart_summaries", "optimizer_memory"}})
    _write_csv(cfg.output_csv, rows)


def run_fixed_handoff_replay_optimizer_probe(cfg: ReplayOptimizerProbeConfig) -> dict[str, Any]:
    payload = build_probe_payload(cfg)
    emit_probe_files(payload, cfg)
    return payload


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-handoff HH replay optimizer probe.")
    parser.add_argument("--adapt-input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=_DEFAULT_OUTPUT_CSV)
    parser.add_argument("--tag", type=str, default=_DEFAULT_TAG)
    parser.add_argument("--generator-family", type=str, default="match_adapt")
    parser.add_argument("--fallback-family", type=str, default="full_meta")
    parser.add_argument("--legacy-paop-key", type=str, default="paop_lf_std")
    parser.add_argument("--replay-seed-policy", type=str, default="auto")
    parser.add_argument(
        "--replay-continuation-mode",
        type=str,
        default="legacy",
        choices=["legacy"],
        help="Fixed-handoff optimizer probe is intentionally legacy-only so the replay scaffold stays fixed while only optimizer choice changes.",
    )
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--baseline-restarts", type=int, default=6)
    parser.add_argument("--baseline-maxiter", type=int, default=4000)
    parser.add_argument("--heavy-restarts", type=int, default=10)
    parser.add_argument("--heavy-maxiter", type=int, default=8000)
    parser.add_argument("--deterministic-method", type=str, default="L-BFGS-B", choices=list(_ALLOWED_DETERMINISTIC_METHODS))
    parser.add_argument("--deterministic-restarts", type=int, default=6)
    parser.add_argument("--deterministic-maxiter", type=int, default=4000)
    parser.add_argument("--progress-every-s", type=float, default=60.0)
    parser.add_argument("--wallclock-cap-s", type=int, default=43200)
    parser.add_argument("--energy-backend", type=str, default="one_apply_compiled", choices=["legacy", "one_apply_compiled"])
    parser.add_argument("--spsa-a", type=float, default=0.2)
    parser.add_argument("--spsa-c", type=float, default=0.1)
    parser.add_argument("--spsa-alpha", type=float, default=0.602)
    parser.add_argument("--spsa-gamma", type=float, default=0.101)
    parser.add_argument("--spsa-A", type=float, default=10.0)
    parser.add_argument("--spsa-avg-last", type=int, default=0)
    parser.add_argument("--spsa-eval-repeats", type=int, default=1)
    parser.add_argument("--spsa-eval-agg", type=str, default="mean", choices=["mean", "median"])
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> ReplayOptimizerProbeConfig:
    args = build_cli_parser().parse_args(argv)
    return ReplayOptimizerProbeConfig(
        adapt_input_json=Path(args.adapt_input_json),
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
        tag=str(args.tag),
        generator_family=str(args.generator_family),
        fallback_family=str(args.fallback_family),
        legacy_paop_key=str(args.legacy_paop_key),
        replay_seed_policy=str(args.replay_seed_policy),
        replay_continuation_mode=str(args.replay_continuation_mode),
        reps=(None if args.reps is None else int(args.reps)),
        seed=int(args.seed),
        baseline_restarts=int(args.baseline_restarts),
        baseline_maxiter=int(args.baseline_maxiter),
        heavy_restarts=int(args.heavy_restarts),
        heavy_maxiter=int(args.heavy_maxiter),
        deterministic_method=str(args.deterministic_method),
        deterministic_restarts=int(args.deterministic_restarts),
        deterministic_maxiter=int(args.deterministic_maxiter),
        progress_every_s=float(args.progress_every_s),
        wallclock_cap_s=int(args.wallclock_cap_s),
        energy_backend=str(args.energy_backend),
        spsa_a=float(args.spsa_a),
        spsa_c=float(args.spsa_c),
        spsa_alpha=float(args.spsa_alpha),
        spsa_gamma=float(args.spsa_gamma),
        spsa_A=float(args.spsa_A),
        spsa_avg_last=int(args.spsa_avg_last),
        spsa_eval_repeats=int(args.spsa_eval_repeats),
        spsa_eval_agg=str(args.spsa_eval_agg),
    )


def format_compact_summary(payload: Mapping[str, Any]) -> list[str]:
    summary = payload.get("summary", {})
    best_row = summary.get("best_variant", {})
    baseline_row = summary.get("baseline_variant", {})
    return [
        f"handoff: {payload.get('handoff', {}).get('adapt_input_json')}",
        f"baseline: {baseline_row.get('variant')} delta_abs={baseline_row.get('delta_abs')}",
        f"best: {best_row.get('variant')} delta_abs={best_row.get('delta_abs')}",
        f"signal: {summary.get('diagnostic_signal')}",
        f"json: {payload.get('config', {}).get('output_json', '')}",
        f"csv: {payload.get('config', {}).get('output_csv', '')}",
    ]


__all__ = [
    "ReplayOptimizerProbeConfig",
    "ProbeVariant",
    "build_base_replay_config",
    "build_probe_context",
    "build_probe_payload",
    "emit_probe_files",
    "run_fixed_handoff_replay_optimizer_probe",
    "build_cli_parser",
    "parse_cli_args",
    "format_compact_summary",
]
