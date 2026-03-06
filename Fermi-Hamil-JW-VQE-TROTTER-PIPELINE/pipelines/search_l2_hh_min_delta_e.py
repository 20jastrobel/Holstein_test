#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
    apply_pauli_rotation,
    exact_ground_energy_sector_hh,
    half_filled_num_particles,
    vqe_minimize,
)


@dataclass(frozen=True)
class RunConfig:
    ansatz_family: str
    ordering: str
    n_ph_max: int
    boson_encoding: str
    reps: int
    restarts: int
    method: str
    maxiter: int
    seed: int


class HHFullTermwiseAnsatz:
    """Term-wise HH ansatz with independent angle per Pauli term per repetition."""

    def __init__(self, hamiltonian, reps: int) -> None:
        terms: list[tuple[str, float]] = []
        for term in hamiltonian.return_polynomial():
            coeff = complex(term.p_coeff)
            pauli = term.pw2strng()
            if abs(coeff) < 1e-12:
                continue
            if pauli == ("e" * term.nqubit()):
                continue
            if abs(coeff.imag) > 1e-12:
                raise ValueError(f"non-real coefficient for term {pauli}: {coeff}")
            terms.append((pauli, float(coeff.real)))

        self.reps = int(reps)
        self.terms = sorted(terms, key=lambda item: item[0])
        self.num_parameters = int(self.reps * len(self.terms))

    # Built-in math symbolic expression:
    # |ψ(θ)⟩ = ∏_{r=1}^{R} ∏_{j=1}^{M} exp(-i θ_{r,j} h_j P_j) |ψ_ref⟩.
    def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        if int(theta.size) != int(self.num_parameters):
            raise ValueError("theta length mismatch")
        psi = np.array(psi_ref, copy=True)
        cursor = 0
        for _ in range(self.reps):
            for pauli, coeff in self.terms:
                angle = 2.0 * float(theta[cursor]) * float(coeff)
                psi = apply_pauli_rotation(psi, pauli, angle)
                cursor += 1
        return psi


# Built-in math symbolic expression:
# ΔE = |E_VQE(θ*) - E_exact,filtered|.
def run_one(cfg: RunConfig, *, L: int, J: float, U: float, omega0: float, g_ep: float) -> dict[str, Any]:
    num_particles = half_filled_num_particles(L)
    hamiltonian = build_hubbard_holstein_hamiltonian(
        dims=L,
        J=J,
        U=U,
        omega0=omega0,
        g=g_ep,
        n_ph_max=cfg.n_ph_max,
        boson_encoding=cfg.boson_encoding,
        indexing=cfg.ordering,
        pbc=True,
    )

    psi_ref = hubbard_holstein_reference_state(
        dims=L,
        num_particles=num_particles,
        n_ph_max=cfg.n_ph_max,
        boson_encoding=cfg.boson_encoding,
        indexing=cfg.ordering,
    )

    if cfg.ansatz_family == "hh_hva":
        ansatz = HubbardHolsteinLayerwiseAnsatz(
            dims=L,
            J=J,
            U=U,
            omega0=omega0,
            g=g_ep,
            n_ph_max=cfg.n_ph_max,
            boson_encoding=cfg.boson_encoding,
            reps=cfg.reps,
            indexing=cfg.ordering,
            pbc=True,
        )
    elif cfg.ansatz_family == "hh_full_termwise":
        ansatz = HHFullTermwiseAnsatz(hamiltonian, reps=cfg.reps)
    else:
        raise ValueError(f"unknown ansatz family: {cfg.ansatz_family}")

    start = time.perf_counter()
    vqe = vqe_minimize(
        hamiltonian,
        ansatz,
        psi_ref,
        restarts=cfg.restarts,
        seed=cfg.seed,
        method=cfg.method,
        maxiter=cfg.maxiter,
    )
    elapsed = float(time.perf_counter() - start)

    exact_filtered = exact_ground_energy_sector_hh(
        hamiltonian,
        num_sites=L,
        num_particles=num_particles,
        n_ph_max=cfg.n_ph_max,
        boson_encoding=cfg.boson_encoding,
        indexing=cfg.ordering,
    )

    delta_e_abs = abs(float(vqe.energy) - float(exact_filtered))
    return {
        "config": asdict(cfg),
        "num_parameters": int(ansatz.num_parameters),
        "exact_energy_filtered": float(exact_filtered),
        "vqe_energy": float(vqe.energy),
        "delta_e_abs": float(delta_e_abs),
        "vqe_success": bool(vqe.success),
        "vqe_message": str(vqe.message),
        "elapsed_sec": elapsed,
    }


def run_search(*, L: int, J: float, U: float, omega0: float, g_ep: float) -> dict[str, Any]:
    configs = [
        RunConfig("hh_hva", "blocked", 1, "binary", 2, 4, "SLSQP", 500, 11),
        RunConfig("hh_hva", "blocked", 1, "binary", 3, 4, "SLSQP", 700, 13),
        RunConfig("hh_hva", "blocked", 1, "binary", 4, 6, "SLSQP", 900, 17),
        RunConfig("hh_full_termwise", "blocked", 1, "binary", 1, 4, "SLSQP", 500, 7),
        RunConfig("hh_full_termwise", "blocked", 1, "binary", 2, 4, "SLSQP", 500, 7),
        RunConfig("hh_full_termwise", "blocked", 1, "binary", 2, 6, "SLSQP", 700, 19),
    ]

    results = []
    for i, cfg in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] {cfg}", flush=True)
        results.append(run_one(cfg, L=L, J=J, U=U, omega0=omega0, g_ep=g_ep))

    ranked = sorted(results, key=lambda row: row["delta_e_abs"])
    payload = {
        "fixed_physics": {
            "L": int(L),
            "J": float(J),
            "U": float(U),
            "omega0": float(omega0),
            "g_ep": float(g_ep),
            "num_particles_half_filled": list(half_filled_num_particles(L)),
        },
        "best": ranked[0],
        "all_results_ranked": ranked,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="L=2 HH ΔE search with sector-filtered exact reference.")
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--U", type=float, default=2.0)
    parser.add_argument("--omega0", type=float, default=1.0)
    parser.add_argument("--g-ep", type=float, default=1.0)
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/l2_hh_delta_e_search.json"))
    args = parser.parse_args()

    payload = run_search(L=args.L, J=args.J, U=args.U, omega0=args.omega0, g_ep=args.g_ep)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["best"], indent=2))


if __name__ == "__main__":
    main()
