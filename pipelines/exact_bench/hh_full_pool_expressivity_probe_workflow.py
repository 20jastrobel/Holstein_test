#!/usr/bin/env python3
"""HH full-pool conventional VQE expressivity probe.

Purpose:
- Hold physics fixed at HH L=2, n_ph_max=2.
- Remove ADAPT/replay scaffold effects entirely.
- Test whether the current full generator library can reach the exact energy with
  a direct conventional VQE built from the full deduplicated pool.

This is a wrapper-only diagnostic. It does not modify the canonical staged path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded import hh_vqe_from_adapt_family as replay_mod
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.vqe_latex_python_pairs import exact_ground_energy_sector_hh, vqe_minimize

_DEFAULT_OUTPUT_JSON = REPO_ROOT / "artifacts/json/hh_full_pool_expressivity_probe.json"
_DEFAULT_OUTPUT_CSV = REPO_ROOT / "artifacts/json/hh_full_pool_expressivity_probe.csv"
_DEFAULT_TAG = "hh_full_pool_expressivity_probe"
_ALLOWED_METHODS = ("SPSA", "Powell")


def _scipy_available() -> bool:
    try:
        from scipy.optimize import minimize as _  # type: ignore
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class FullPoolExpressivityProbeConfig:
    output_json: Path = _DEFAULT_OUTPUT_JSON
    output_csv: Path = _DEFAULT_OUTPUT_CSV
    tag: str = _DEFAULT_TAG
    only_variants: tuple[str, ...] = ()
    base_family: str = "full_meta"
    extra_families: tuple[str, ...] = ()
    t: float = 1.0
    u: float = 4.0
    dv: float = 0.0
    omega0: float = 1.0
    g_ep: float = 1.0
    n_ph_max: int = 2
    order_seed: int = 1000
    random_orderings_x1: int = 2
    include_canonical_x1_powell: bool = True
    include_canonical_x2_spsa: bool = True
    spsa_restarts: int = 6
    spsa_maxiter: int = 4000
    powell_restarts: int = 2
    powell_maxiter: int = 3000
    seed: int = 19
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
class ExpressivityVariant:
    label: str
    reps: int
    ordering_kind: str
    ordering_seed: int | None
    method: str
    restarts: int
    maxiter: int


_MATH_DELTA_ABS = "Δ_abs := |E(θ) - E_exact|"
_MATH_CAPACITY = "capacity probe := direct conventional VQE over full deduplicated generator pool"


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


def _half_filled_particles(L: int) -> tuple[int, int]:
    return ((int(L) + 1) // 2, int(L) // 2)


def _base_run_cfg(cfg: FullPoolExpressivityProbeConfig) -> replay_mod.RunConfig:
    n_up, n_dn = _half_filled_particles(2)
    n_ph_max = int(cfg.n_ph_max)
    return replay_mod.RunConfig(
        adapt_input_json=REPO_ROOT / "artifacts/json/unused_full_pool_probe_handoff.json",
        output_json=cfg.output_json,
        output_csv=cfg.output_csv,
        output_md=cfg.output_json.with_suffix(".md"),
        output_log=cfg.output_json.with_suffix(".log"),
        tag=str(cfg.tag),
        generator_family=str(cfg.base_family),
        fallback_family=str(cfg.base_family),
        legacy_paop_key="paop_lf_std",
        replay_seed_policy="auto",
        replay_continuation_mode="legacy",
        L=2,
        t=float(cfg.t),
        u=float(cfg.u),
        dv=float(cfg.dv),
        omega0=float(cfg.omega0),
        g_ep=float(cfg.g_ep),
        n_ph_max=int(n_ph_max),
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        sector_n_up=int(n_up),
        sector_n_dn=int(n_dn),
        reps=1,
        restarts=int(cfg.spsa_restarts),
        maxiter=int(cfg.spsa_maxiter),
        method="SPSA",
        seed=int(cfg.seed),
        energy_backend=str(cfg.energy_backend),
        progress_every_s=float(cfg.progress_every_s),
        wallclock_cap_s=int(cfg.wallclock_cap_s),
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        spsa_a=float(cfg.spsa_a),
        spsa_c=float(cfg.spsa_c),
        spsa_alpha=float(cfg.spsa_alpha),
        spsa_gamma=float(cfg.spsa_gamma),
        spsa_A=float(cfg.spsa_A),
        spsa_avg_last=int(cfg.spsa_avg_last),
        spsa_eval_repeats=int(cfg.spsa_eval_repeats),
        spsa_eval_agg=str(cfg.spsa_eval_agg),
        replay_freeze_fraction=0.2,
        replay_unfreeze_fraction=0.3,
        replay_full_fraction=0.5,
        replay_qn_spsa_refresh_every=5,
        replay_qn_spsa_refresh_mode="diag_rms_grad",
        phase3_symmetry_mitigation_mode="off",
    )


def _default_recipe(cfg: FullPoolExpressivityProbeConfig) -> bool:
    return str(cfg.base_family).strip().lower() == "full_meta" and len(tuple(cfg.extra_families)) == 0


def _validated_recipe(cfg: FullPoolExpressivityProbeConfig) -> tuple[str, tuple[str, ...]]:
    base = replay_mod._canonical_family(cfg.base_family)
    if base is None:
        raise ValueError(f"Unsupported base_family '{cfg.base_family}'.")
    extras: list[str] = []
    seen: set[str] = set()
    for raw in cfg.extra_families:
        extra = replay_mod._canonical_family(raw)
        if extra is None:
            raise ValueError(f"Unsupported extra_family '{raw}'.")
        if extra == base:
            raise ValueError(f"extra_family '{extra}' duplicates base_family '{base}'.")
        if extra in seen:
            raise ValueError(f"Duplicate extra_family '{extra}' is not allowed.")
        seen.add(extra)
        extras.append(extra)
    return str(base), tuple(extras)


def _pool_recipe_label(cfg: FullPoolExpressivityProbeConfig) -> str:
    base, extras = _validated_recipe(cfg)
    return base if not extras else "_plus_".join([base] + list(extras))


def _variant_label(cfg: FullPoolExpressivityProbeConfig, suffix: str) -> str:
    return suffix if _default_recipe(cfg) else f"{_pool_recipe_label(cfg)}_{suffix}"


def _order_hash(labels: Sequence[str]) -> str:
    digest = hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()
    return digest[:16]


def _pool_labels(terms: Sequence[Any]) -> list[str]:
    labels: list[str] = []
    for idx, term in enumerate(terms):
        label = getattr(term, "label", None)
        labels.append(str(label) if label not in (None, "") else f"term_{idx}")
    return labels


def build_probe_context(cfg: FullPoolExpressivityProbeConfig) -> dict[str, Any]:
    base_family, extra_families = _validated_recipe(cfg)
    base_cfg = _base_run_cfg(cfg)
    h_poly = replay_mod._build_hh_hamiltonian(base_cfg)
    pool, pool_recipe_meta = replay_mod._build_pool_recipe(
        base_cfg,
        base_family=str(base_family),
        extra_families=tuple(extra_families),
        h_poly=h_poly,
    )
    psi_ref = np.asarray(
        hubbard_holstein_reference_state(
            dims=2,
            num_particles=(int(base_cfg.sector_n_up), int(base_cfg.sector_n_dn)),
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(base_cfg.boson_encoding),
            indexing=str(base_cfg.ordering),
        ),
        dtype=complex,
    ).reshape(-1)
    exact_energy = float(
        exact_ground_energy_sector_hh(
            h_poly,
            num_sites=2,
            num_particles=(int(base_cfg.sector_n_up), int(base_cfg.sector_n_dn)),
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(base_cfg.boson_encoding),
            indexing=str(base_cfg.ordering),
        )
    )
    labels = _pool_labels(pool)
    return {
        "base_cfg": base_cfg,
        "h_poly": h_poly,
        "pool_terms": list(pool),
        "pool_recipe_meta": dict(pool_recipe_meta),
        "pool_labels": labels,
        "pool_size": int(len(pool)),
        "pool_order_hash": _order_hash(labels),
        "pool_recipe_label": _pool_recipe_label(cfg),
        "psi_ref": psi_ref / max(np.linalg.norm(psi_ref), 1e-16),
        "exact_energy": float(exact_energy),
    }


def build_variants(cfg: FullPoolExpressivityProbeConfig) -> tuple[ExpressivityVariant, ...]:
    _validated_recipe(cfg)
    variants: list[ExpressivityVariant] = [
        ExpressivityVariant(
            label=_variant_label(cfg, "fullmeta_x1_canonical_spsa" if _default_recipe(cfg) else "x1_canonical_spsa"),
            reps=1,
            ordering_kind="canonical",
            ordering_seed=None,
            method="SPSA",
            restarts=int(cfg.spsa_restarts),
            maxiter=int(cfg.spsa_maxiter),
        )
    ]
    for idx in range(int(cfg.random_orderings_x1)):
        variants.append(
            ExpressivityVariant(
                label=_variant_label(
                    cfg,
                    f"fullmeta_x1_random_{idx + 1}_spsa" if _default_recipe(cfg) else f"x1_random_{idx + 1}_spsa",
                ),
                reps=1,
                ordering_kind="random",
                ordering_seed=int(cfg.order_seed) + idx,
                method="SPSA",
                restarts=int(cfg.spsa_restarts),
                maxiter=int(cfg.spsa_maxiter),
            )
        )
    if bool(cfg.include_canonical_x1_powell):
        variants.append(
            ExpressivityVariant(
                label=_variant_label(cfg, "fullmeta_x1_canonical_powell" if _default_recipe(cfg) else "x1_canonical_powell"),
                reps=1,
                ordering_kind="canonical",
                ordering_seed=None,
                method="Powell",
                restarts=int(cfg.powell_restarts),
                maxiter=int(cfg.powell_maxiter),
            )
        )
    if bool(cfg.include_canonical_x2_spsa):
        variants.append(
            ExpressivityVariant(
                label=_variant_label(cfg, "fullmeta_x2_canonical_spsa" if _default_recipe(cfg) else "x2_canonical_spsa"),
                reps=2,
                ordering_kind="canonical",
                ordering_seed=None,
                method="SPSA",
                restarts=int(cfg.spsa_restarts),
                maxiter=int(cfg.spsa_maxiter),
            )
        )
    if not cfg.only_variants:
        return tuple(variants)
    wanted = set(str(label) for label in cfg.only_variants)
    selected = [variant for variant in variants if str(variant.label) in wanted]
    missing = sorted(wanted.difference({variant.label for variant in variants}))
    if missing:
        raise ValueError(f"Unknown only_variants labels: {missing}")
    return tuple(selected)


def _ordered_terms(context: Mapping[str, Any], variant: ExpressivityVariant) -> tuple[list[Any], list[str]]:
    terms = list(context["pool_terms"])
    labels = list(context["pool_labels"])
    if variant.ordering_kind == "canonical":
        return terms, labels
    rng = np.random.default_rng(int(variant.ordering_seed))
    perm = np.arange(len(terms), dtype=int)
    rng.shuffle(perm)
    return [terms[i] for i in perm], [labels[i] for i in perm]


def _variant_log_path(cfg: FullPoolExpressivityProbeConfig, variant: ExpressivityVariant) -> Path:
    return cfg.output_json.parent / f"{cfg.tag}_{variant.label}.log"


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


def run_variant(cfg: FullPoolExpressivityProbeConfig, context: Mapping[str, Any], variant: ExpressivityVariant) -> dict[str, Any]:
    if str(variant.method) not in _ALLOWED_METHODS:
        raise ValueError(f"Unsupported method {variant.method!r}")
    if str(variant.method) != "SPSA" and not _scipy_available():
        raise RuntimeError(f"SciPy unavailable for method={variant.method}")
    base_cfg: replay_mod.RunConfig = context["base_cfg"]
    exact_energy = float(context["exact_energy"])
    psi_ref = np.asarray(context["psi_ref"], dtype=complex).reshape(-1)
    h_poly = context["h_poly"]
    ordered_terms, ordered_labels = _ordered_terms(context, variant)
    nq = int(psi_ref.size.bit_length() - 1)
    ansatz = replay_mod.PoolTermwiseAnsatz(terms=ordered_terms, reps=int(variant.reps), nq=nq)
    logger = replay_mod.RunLogger(_variant_log_path(cfg, variant))
    logger.log(
        f"START variant={variant.label} method={variant.method} reps={variant.reps} "
        f"pool_size={len(ordered_terms)} npar={ansatz.num_parameters} order_hash={_order_hash(ordered_labels)}"
    )
    progress_logger, progress_tail = _progress_logger_factory(logger, exact_energy=exact_energy)
    wall_hit = False
    t0 = time.perf_counter()

    def _early_stop_checker(ev: dict[str, Any]) -> bool:
        nonlocal wall_hit
        elapsed = float(ev.get("elapsed_s", 0.0))
        if elapsed >= float(cfg.wallclock_cap_s):
            wall_hit = True
            return True
        return False

    result = vqe_minimize(
        h_poly,
        ansatz,
        psi_ref,
        restarts=int(variant.restarts),
        seed=int(cfg.seed) + (0 if variant.ordering_seed is None else int(variant.ordering_seed)),
        initial_point=np.zeros(int(ansatz.num_parameters), dtype=float),
        use_initial_point_first_restart=True,
        method=str(variant.method),
        maxiter=int(variant.maxiter),
        progress_logger=progress_logger,
        progress_every_s=float(cfg.progress_every_s),
        progress_label=f"hh_full_pool_expressivity:{variant.label}",
        track_history=False,
        emit_theta_in_progress=False,
        return_best_on_keyboard_interrupt=True,
        early_stop_checker=_early_stop_checker,
        spsa_a=float(base_cfg.spsa_a),
        spsa_c=float(base_cfg.spsa_c),
        spsa_alpha=float(base_cfg.spsa_alpha),
        spsa_gamma=float(base_cfg.spsa_gamma),
        spsa_A=float(base_cfg.spsa_A),
        spsa_avg_last=int(base_cfg.spsa_avg_last),
        spsa_eval_repeats=int(base_cfg.spsa_eval_repeats),
        spsa_eval_agg=str(base_cfg.spsa_eval_agg),
        energy_backend=str(base_cfg.energy_backend),
    )
    runtime_s = float(time.perf_counter() - t0)
    energy = float(result.energy)
    delta_abs = float(abs(energy - exact_energy))
    row = {
        "tag": str(cfg.tag),
        "variant": str(variant.label),
        "base_family": str(cfg.base_family).strip().lower(),
        "extra_families": [str(item).strip().lower() for item in cfg.extra_families],
        "pool_recipe_label": str(context["pool_recipe_label"]),
        "combined_pool_size": int(context["pool_size"]),
        "method": str(variant.method),
        "reps": int(variant.reps),
        "ordering_kind": str(variant.ordering_kind),
        "ordering_seed": (None if variant.ordering_seed is None else int(variant.ordering_seed)),
        "pool_size": int(len(ordered_terms)),
        "parameter_count": int(ansatz.num_parameters),
        "order_hash": _order_hash(ordered_labels),
        "energy": float(energy),
        "exact_energy": float(exact_energy),
        "delta_abs": float(delta_abs),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "best_restart": int(result.best_restart),
        "runtime_s": float(runtime_s),
        "wallclock_cap_hit": bool(wall_hit),
        "log_path": str(_variant_log_path(cfg, variant)),
        "ordered_label_preview": ordered_labels[:8],
    }
    logger.log(
        f"DONE variant={variant.label} method={variant.method} abs_delta_e={delta_abs:.6e} runtime_s={runtime_s:.1f}"
    )
    row["progress_tail"] = progress_tail[-20:]
    row["restart_summaries"] = result.restart_summaries
    row["optimizer_memory"] = result.optimizer_memory
    return row


def _failure_row(cfg: FullPoolExpressivityProbeConfig, context: Mapping[str, Any], variant: ExpressivityVariant, exc: Exception) -> dict[str, Any]:
    ordered_terms, ordered_labels = _ordered_terms(context, variant)
    return {
        "tag": str(cfg.tag),
        "variant": str(variant.label),
        "base_family": str(cfg.base_family).strip().lower(),
        "extra_families": [str(item).strip().lower() for item in cfg.extra_families],
        "pool_recipe_label": str(context["pool_recipe_label"]),
        "combined_pool_size": int(context["pool_size"]),
        "method": str(variant.method),
        "reps": int(variant.reps),
        "ordering_kind": str(variant.ordering_kind),
        "ordering_seed": (None if variant.ordering_seed is None else int(variant.ordering_seed)),
        "pool_size": int(len(ordered_terms)),
        "parameter_count": int(len(ordered_terms) * int(variant.reps)),
        "order_hash": _order_hash(ordered_labels),
        "energy": None,
        "exact_energy": float(context["exact_energy"]),
        "delta_abs": float("inf"),
        "success": False,
        "message": f"variant_failed: {exc}",
        "nfev": 0,
        "nit": 0,
        "best_restart": -1,
        "runtime_s": 0.0,
        "wallclock_cap_hit": False,
        "log_path": str(_variant_log_path(cfg, variant)),
        "ordered_label_preview": ordered_labels[:8],
        "error": str(exc),
        "progress_tail": [],
        "restart_summaries": [],
        "optimizer_memory": None,
    }


def build_probe_payload(cfg: FullPoolExpressivityProbeConfig) -> dict[str, Any]:
    base_family, extra_families = _validated_recipe(cfg)
    context = build_probe_context(cfg)
    rows: list[dict[str, Any]] = []
    for variant in build_variants(cfg):
        try:
            rows.append(run_variant(cfg, context, variant))
        except Exception as exc:
            rows.append(_failure_row(cfg, context, variant, exc))
    best_row = min(rows, key=lambda row: float(row["delta_abs"]))
    canonical_x1_label = _variant_label(cfg, "fullmeta_x1_canonical_spsa" if _default_recipe(cfg) else "x1_canonical_spsa")
    canonical_x1 = next((row for row in rows if row["variant"] == canonical_x1_label), None)
    payload = {
        "created_utc": _now_utc(),
        "probe_scope": {
            "local_only": True,
            "noise_enabled": False,
            "mitigation_enabled": False,
            "patch_selection_enabled": False,
            "scaffold_free": True,
            "full_generator_pool": True,
        },
        "config": asdict(cfg),
        "math_contract": {
            "delta_abs": _MATH_DELTA_ABS,
            "capacity_probe": _MATH_CAPACITY,
        },
        "physics": {
            "L": 2,
            "n_ph_max": int(cfg.n_ph_max),
            "t": float(cfg.t),
            "u": float(cfg.u),
            "dv": float(cfg.dv),
            "omega0": float(cfg.omega0),
            "g_ep": float(cfg.g_ep),
            "sector_n_up": int(context["base_cfg"].sector_n_up),
            "sector_n_dn": int(context["base_cfg"].sector_n_dn),
        },
        "pool": {
            "base_family": str(base_family),
            "extra_families": list(extra_families),
            "pool_recipe_label": str(context["pool_recipe_label"]),
            "pool_size": int(context["pool_size"]),
            "combined_dedup_total": int(context["pool_size"]),
            "pool_recipe_meta": dict(context["pool_recipe_meta"]),
            "canonical_order_hash": str(context["pool_order_hash"]),
        },
        "exact_energy": float(context["exact_energy"]),
        "variant_rows": rows,
        "summary": {
            "best_variant": {k: v for k, v in best_row.items() if k not in {"progress_tail", "restart_summaries", "optimizer_memory"}},
            "best_overall_expressive_enough_at_1e3": bool(float(best_row["delta_abs"]) <= 1e-3),
            "canonical_x1_variant": (
                None
                if canonical_x1 is None
                else {k: v for k, v in canonical_x1.items() if k not in {"progress_tail", "restart_summaries", "optimizer_memory"}}
            ),
            "canonical_x1_expressive_enough_at_1e3": bool(canonical_x1 is not None and float(canonical_x1["delta_abs"]) <= 1e-3),
            "expressive_gate_delta_abs": 1e-3,
        },
    }
    return payload


def emit_probe_files(payload: Mapping[str, Any], cfg: FullPoolExpressivityProbeConfig) -> None:
    _write_json(cfg.output_json, payload)
    rows = [{k: v for k, v in row.items() if k not in {"progress_tail", "restart_summaries", "optimizer_memory"}} for row in payload.get("variant_rows", [])]
    _write_csv(cfg.output_csv, rows)


def run_full_pool_expressivity_probe(cfg: FullPoolExpressivityProbeConfig) -> dict[str, Any]:
    payload = build_probe_payload(cfg)
    emit_probe_files(payload, cfg)
    return payload


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HH full-pool conventional VQE expressivity probe.")
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=_DEFAULT_OUTPUT_CSV)
    parser.add_argument("--tag", type=str, default=_DEFAULT_TAG)
    parser.add_argument(
        "--only-variant",
        dest="only_variants",
        action="append",
        default=None,
        help="Restrict the run to specific variant labels. May be repeated.",
    )
    parser.add_argument("--base-family", type=str, default="full_meta")
    parser.add_argument(
        "--extra-family",
        dest="extra_families",
        action="append",
        default=None,
        help="Append an extra family block after base_family. May be repeated.",
    )
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--dv", type=float, default=0.0)
    parser.add_argument("--omega0", type=float, default=1.0)
    parser.add_argument("--g-ep", type=float, default=1.0, dest="g_ep")
    parser.add_argument("--n-ph-max", type=int, default=2, dest="n_ph_max")
    parser.add_argument("--order-seed", type=int, default=1000)
    parser.add_argument("--random-orderings-x1", type=int, default=2)
    parser.add_argument("--include-canonical-x1-powell", action="store_true")
    parser.add_argument("--no-canonical-x1-powell", dest="include_canonical_x1_powell", action="store_false")
    parser.set_defaults(include_canonical_x1_powell=True)
    parser.add_argument("--include-canonical-x2-spsa", action="store_true")
    parser.add_argument("--no-canonical-x2-spsa", dest="include_canonical_x2_spsa", action="store_false")
    parser.set_defaults(include_canonical_x2_spsa=True)
    parser.add_argument("--spsa-restarts", type=int, default=6)
    parser.add_argument("--spsa-maxiter", type=int, default=4000)
    parser.add_argument("--powell-restarts", type=int, default=2)
    parser.add_argument("--powell-maxiter", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=19)
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


def parse_cli_args(argv: Sequence[str] | None = None) -> FullPoolExpressivityProbeConfig:
    args = build_cli_parser().parse_args(argv)
    return FullPoolExpressivityProbeConfig(
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
        tag=str(args.tag),
        only_variants=tuple(str(x) for x in (args.only_variants or [])),
        base_family=str(args.base_family),
        extra_families=tuple(str(x) for x in (args.extra_families or [])),
        t=float(args.t),
        u=float(args.u),
        dv=float(args.dv),
        omega0=float(args.omega0),
        g_ep=float(args.g_ep),
        n_ph_max=int(args.n_ph_max),
        order_seed=int(args.order_seed),
        random_orderings_x1=int(args.random_orderings_x1),
        include_canonical_x1_powell=bool(args.include_canonical_x1_powell),
        include_canonical_x2_spsa=bool(args.include_canonical_x2_spsa),
        spsa_restarts=int(args.spsa_restarts),
        spsa_maxiter=int(args.spsa_maxiter),
        powell_restarts=int(args.powell_restarts),
        powell_maxiter=int(args.powell_maxiter),
        seed=int(args.seed),
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
    return [
        f"pool_size: {payload.get('pool', {}).get('pool_size')}",
        f"recipe: {payload.get('pool', {}).get('pool_recipe_label')}",
        f"best: {best_row.get('variant')} delta_abs={best_row.get('delta_abs')}",
        f"best_overall_expressive_enough_at_1e3: {summary.get('best_overall_expressive_enough_at_1e3')}",
        f"canonical_x1_expressive_enough_at_1e3: {summary.get('canonical_x1_expressive_enough_at_1e3')}",
        f"json: {payload.get('config', {}).get('output_json', '')}",
        f"csv: {payload.get('config', {}).get('output_csv', '')}",
    ]


__all__ = [
    "FullPoolExpressivityProbeConfig",
    "ExpressivityVariant",
    "build_probe_context",
    "build_variants",
    "build_probe_payload",
    "emit_probe_files",
    "run_full_pool_expressivity_probe",
    "build_cli_parser",
    "parse_cli_args",
    "format_compact_summary",
]


def main(argv: Sequence[str] | None = None) -> int:
    cfg = parse_cli_args(argv)
    payload = run_full_pool_expressivity_probe(cfg)
    for line in format_compact_summary(payload):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
