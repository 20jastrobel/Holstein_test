#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinLayerwiseAnsatz,
    exact_ground_energy_sector_hh,
    half_filled_num_particles,
    vqe_minimize,
)


@dataclass(frozen=True)
class RunConfig:
    boundary: str
    ordering: str
    n_ph_max: int
    reps: int
    method: str
    restarts: int
    maxiter: int
    seed: int
    initial_point_stddev: float


# Built-in math symbolic expression:
# ΔE = |E_VQE(θ*) - E_exact,filtered| where E_exact,filtered = min eig(H|sector).
def evaluate_config(*, L: int, J: float, U: float, omega0: float, g_ep: float, cfg: RunConfig) -> dict[str, Any]:
    num_particles = half_filled_num_particles(L)
    pbc = cfg.boundary == "periodic"

    H = build_hubbard_holstein_hamiltonian(
        dims=L,
        J=J,
        U=U,
        omega0=omega0,
        g=g_ep,
        n_ph_max=cfg.n_ph_max,
        boson_encoding="binary",
        indexing=cfg.ordering,
        pbc=pbc,
    )
    psi_ref = hubbard_holstein_reference_state(
        dims=L,
        num_particles=num_particles,
        n_ph_max=cfg.n_ph_max,
        boson_encoding="binary",
        indexing=cfg.ordering,
    )
    ansatz = HubbardHolsteinLayerwiseAnsatz(
        dims=L,
        J=J,
        U=U,
        omega0=omega0,
        g=g_ep,
        n_ph_max=cfg.n_ph_max,
        boson_encoding="binary",
        reps=cfg.reps,
        indexing=cfg.ordering,
        pbc=pbc,
    )

    t0 = time.perf_counter()
    vqe = vqe_minimize(
        H,
        ansatz,
        psi_ref,
        restarts=cfg.restarts,
        seed=cfg.seed,
        initial_point_stddev=cfg.initial_point_stddev,
        method=cfg.method,
        maxiter=cfg.maxiter,
    )
    elapsed = time.perf_counter() - t0

    exact_filtered = exact_ground_energy_sector_hh(
        H,
        num_sites=L,
        num_particles=num_particles,
        n_ph_max=cfg.n_ph_max,
        boson_encoding="binary",
        indexing=cfg.ordering,
    )
    delta = abs(float(vqe.energy) - float(exact_filtered))

    return {
        "config": asdict(cfg),
        "exact_energy_filtered": float(exact_filtered),
        "vqe_energy": float(vqe.energy),
        "delta_e_abs": float(delta),
        "vqe_success": bool(vqe.success),
        "vqe_message": str(vqe.message),
        "elapsed_sec": float(elapsed),
    }


# Built-in math symbolic expression:
# c* = argmin_{c in C_tested} ΔE(c), evaluated under fixed (L,J,U,omega0,g).
def run_search(*, L: int, J: float, U: float, omega0: float, g_ep: float, budget_minutes: float) -> dict[str, Any]:
    # Prioritized configurations first (include previously-best known points), then broader sweep.
    configs: list[RunConfig] = [
        RunConfig("periodic", "blocked", 1, 4, "COBYLA", 6, 1200, 37, 1.0),
        RunConfig("periodic", "blocked", 1, 3, "COBYLA", 4, 500, 13, 0.3),
        RunConfig("periodic", "blocked", 1, 6, "COBYLA", 4, 800, 73, 1.0),
        RunConfig("open", "blocked", 1, 6, "COBYLA", 4, 800, 71, 1.0),
        RunConfig("periodic", "interleaved", 1, 6, "COBYLA", 4, 800, 79, 1.0),
        RunConfig("periodic", "blocked", 2, 6, "COBYLA", 4, 800, 83, 1.0),
        RunConfig("periodic", "blocked", 1, 6, "SLSQP", 4, 800, 89, 1.0),
    ]

    for boundary in ("open", "periodic"):
        for ordering in ("blocked", "interleaved"):
            for n_ph_max in (1, 2):
                for reps in (3, 4, 5, 6, 7, 8):
                    for method in ("COBYLA", "SLSQP"):
                        for restarts in (6, 10):
                            for maxiter in (1200, 2400):
                                for std in (0.3, 1.0, 2.0):
                                    seed = 5 + reps * 17 + (0 if ordering == "blocked" else 3) + (0 if boundary == "open" else 7)
                                    configs.append(
                                        RunConfig(
                                            boundary=boundary,
                                            ordering=ordering,
                                            n_ph_max=n_ph_max,
                                            reps=reps,
                                            method=method,
                                            restarts=restarts,
                                            maxiter=maxiter,
                                            seed=seed,
                                            initial_point_stddev=std,
                                        )
                                    )

    t_start = time.perf_counter()
    budget_sec = float(budget_minutes) * 60.0
    out: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for idx, cfg in enumerate(configs, start=1):
        elapsed = time.perf_counter() - t_start
        if elapsed >= budget_sec:
            print(f"[budget-stop] elapsed={elapsed:.1f}s / budget={budget_sec:.1f}s", flush=True)
            break

        print(f"[{idx}/{len(configs)}] {cfg}", flush=True)
        result = evaluate_config(L=L, J=J, U=U, omega0=omega0, g_ep=g_ep, cfg=cfg)
        out.append(result)

        if (best is None) or (float(result["delta_e_abs"]) < float(best["delta_e_abs"])):
            best = result
            print(
                f"  [new-best] delta_e={best['delta_e_abs']:.6e} "
                f"E_vqe={best['vqe_energy']:.12f} E_exact={best['exact_energy_filtered']:.12f}",
                flush=True,
            )

        if best is not None and float(best["delta_e_abs"]) < 1e-3:
            print("[target-hit] delta_e < 1e-3", flush=True)
            break

    ranked = sorted(out, key=lambda row: float(row["delta_e_abs"]))
    best_final = ranked[0] if ranked else None

    return {
        "fixed_physics": {
            "model": "Hubbard-Holstein",
            "L": int(L),
            "J": float(J),
            "U": float(U),
            "omega0": float(omega0),
            "g_ep": float(g_ep),
            "num_particles_half_filled": list(half_filled_num_particles(L)),
        },
        "search": {
            "budget_minutes": float(budget_minutes),
            "tested": len(out),
            "candidate_pool": len(configs),
            "target_delta_e": 1e-3,
        },
        "best": best_final,
        "all_results_ranked": ranked,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search L=2 HH settings to minimize |ΔE|.")
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--U", type=float, default=2.0)
    parser.add_argument("--omega0", type=float, default=1.0)
    parser.add_argument("--g-ep", type=float, default=1.0)
    parser.add_argument("--budget-minutes", type=float, default=8.0)
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/l2_hh_delta_e_search.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_search(
        L=args.L,
        J=args.J,
        U=args.U,
        omega0=args.omega0,
        g_ep=args.g_ep,
        budget_minutes=args.budget_minutes,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if payload["best"] is None:
        raise RuntimeError("No configurations were evaluated before budget stop.")

    print(json.dumps(payload["best"], indent=2), flush=True)


if __name__ == "__main__":
    main()
