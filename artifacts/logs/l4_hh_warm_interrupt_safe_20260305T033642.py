#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinPhysicalTermwiseAnsatz,
    half_filled_num_particles,
    hubbard_holstein_reference_state,
    vqe_minimize,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ai_log(event: str, **fields: Any) -> None:
    payload = {"event": str(event), "ts_utc": _utc_now(), **fields}
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("zero norm")
    return psi / nrm


def _state_to_amplitudes_qn_to_q0(psi: np.ndarray, cutoff: float = 1e-12) -> dict[str, dict[str, float]]:
    vec = np.asarray(psi, dtype=complex).reshape(-1)
    nq = int(round(math.log2(vec.size)))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(vec):
        if abs(amp) <= float(cutoff):
            continue
        out[format(idx, f"0{nq}b")] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


def _write_state_json(path: Path, *, settings: dict[str, Any], vqe: dict[str, Any], psi: np.ndarray, source: str) -> None:
    payload = {
        "generated_utc": _utc_now(),
        "pipeline": "hva_warm_interrupt_safe",
        "settings": settings,
        "vqe": vqe,
        "initial_state": {
            "source": str(source),
            "amplitudes_qn_to_q0": _state_to_amplitudes_qn_to_q0(psi),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L4 HH warm HVA SPSA runner with interrupt-safe state export.")
    p.add_argument("--checkpoint-json", type=Path, required=True)
    p.add_argument("--final-json", type=Path, required=True)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=4.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--g-ep", type=float, default=0.5, dest="g_ep")
    p.add_argument("--n-ph-max", type=int, default=2, dest="n_ph_max")
    p.add_argument("--boson-encoding", type=str, default="binary", choices=["binary"])
    p.add_argument("--ordering", type=str, default="blocked", choices=["blocked", "interleaved"])
    p.add_argument("--boundary", type=str, default="open", choices=["open", "periodic"])
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--restarts", type=int, default=7)
    p.add_argument("--maxiter", type=int, default=7111)
    p.add_argument("--method", type=str, default="SPSA", choices=["SPSA", "COBYLA", "SLSQP", "L-BFGS-B", "Powell", "Nelder-Mead"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--progress-every-s", type=float, default=60.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.L) != 4:
        raise ValueError("Runner is locked to L=4 workflow.")

    stop_requested = False

    def _sigint_handler(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        _ai_log("warm_interrupt_requested", signal=int(signum))

    signal.signal(signal.SIGINT, _sigint_handler)

    settings = {
        "problem": "hh",
        "L": int(args.L),
        "t": float(args.t),
        "u": float(args.u),
        "dv": float(args.dv),
        "omega0": float(args.omega0),
        "g_ep": float(args.g_ep),
        "n_ph_max": int(args.n_ph_max),
        "boson_encoding": str(args.boson_encoding),
        "ordering": str(args.ordering),
        "boundary": str(args.boundary),
        "vqe_ansatz": "hh_hva_ptw",
        "vqe_reps": int(args.reps),
        "vqe_restarts": int(args.restarts),
        "vqe_maxiter": int(args.maxiter),
        "vqe_method": str(args.method),
        "vqe_energy_backend": "one_apply_compiled",
    }

    _ai_log("warm_runner_start", settings=settings, checkpoint_json=str(args.checkpoint_json), final_json=str(args.final_json))

    h_poly = build_hubbard_holstein_hamiltonian(
        dims=int(args.L),
        J=float(args.t),
        U=float(args.u),
        omega0=float(args.omega0),
        g=float(args.g_ep),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        repr_mode="JW",
        indexing=str(args.ordering),
        pbc=(str(args.boundary).strip().lower() == "periodic"),
        include_zero_point=True,
    )

    num_particles = tuple(half_filled_num_particles(int(args.L)))
    psi_ref = np.asarray(
        hubbard_holstein_reference_state(
            dims=int(args.L),
            num_particles=num_particles,
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            indexing=str(args.ordering),
        ),
        dtype=complex,
    )

    ansatz = HubbardHolsteinPhysicalTermwiseAnsatz(
        dims=int(args.L),
        J=float(args.t),
        U=float(args.u),
        omega0=float(args.omega0),
        g=float(args.g_ep),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        reps=int(args.reps),
        repr_mode="JW",
        indexing=str(args.ordering),
        pbc=(str(args.boundary).strip().lower() == "periodic"),
    )

    best_energy_written = float("inf")
    best_theta_written: np.ndarray | None = None

    evt_map = {
        "run_start": "hardcoded_vqe_run_start",
        "restart_start": "hardcoded_vqe_restart_start",
        "heartbeat": "hardcoded_vqe_heartbeat",
        "restart_end": "hardcoded_vqe_restart_end",
        "run_end": "hardcoded_vqe_run_end",
        "early_stop_triggered": "hardcoded_vqe_early_stop_triggered",
    }

    def _progress_logger(payload: dict[str, Any]) -> None:
        nonlocal best_energy_written, best_theta_written
        raw = str(payload.get("event", ""))
        mapped = evt_map.get(raw)
        if mapped is not None:
            fields = {k: v for k, v in payload.items() if k != "event"}
            _ai_log(mapped, **fields)

        e_bg = payload.get("energy_best_global")
        th_best = payload.get("theta_restart_best")
        if e_bg is None or th_best is None:
            return
        try:
            e_val = float(e_bg)
            th = np.asarray(th_best, dtype=float)
        except Exception:
            return
        if not np.isfinite(e_val):
            return
        if e_val < best_energy_written - 1e-12:
            psi = _normalize_state(np.asarray(ansatz.prepare_state(th, psi_ref), dtype=complex).reshape(-1))
            vqe_meta = {
                "success": False,
                "stage": "warm_checkpoint",
                "energy": float(e_val),
                "ansatz": "hh_hva_ptw",
                "optimizer_method": str(args.method),
                "num_parameters": int(ansatz.num_parameters),
                "restarts": int(args.restarts),
                "maxiter": int(args.maxiter),
            }
            _write_state_json(
                Path(args.checkpoint_json),
                settings=settings,
                vqe=vqe_meta,
                psi=psi,
                source="warm_hva_checkpoint",
            )
            best_energy_written = float(e_val)
            best_theta_written = np.array(th, copy=True)
            _ai_log("warm_checkpoint_written", energy=float(e_val), checkpoint_json=str(args.checkpoint_json))

    def _early_stop_checker(_payload: dict[str, Any]) -> bool:
        return bool(stop_requested)

    t0 = time.perf_counter()
    result = vqe_minimize(
        h_poly,
        ansatz,
        psi_ref,
        restarts=int(args.restarts),
        seed=int(args.seed),
        maxiter=int(args.maxiter),
        method=str(args.method),
        energy_backend="one_apply_compiled",
        progress_logger=_progress_logger,
        progress_every_s=float(args.progress_every_s),
        progress_label="hardcoded_vqe",
        emit_theta_in_progress=True,
        return_best_on_keyboard_interrupt=True,
        early_stop_checker=_early_stop_checker,
    )

    theta_final = np.asarray(result.theta, dtype=float)
    psi_final = _normalize_state(np.asarray(ansatz.prepare_state(theta_final, psi_ref), dtype=complex).reshape(-1))

    vqe_final = {
        "success": True,
        "stage": "warm_final",
        "energy": float(result.energy),
        "ansatz": "hh_hva_ptw",
        "optimizer_method": str(args.method),
        "num_parameters": int(ansatz.num_parameters),
        "best_restart": int(getattr(result, "best_restart", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
        "nit": int(getattr(result, "nit", 0)),
        "message": str(getattr(result, "message", "")),
        "elapsed_s": float(time.perf_counter() - t0),
        "interrupted_requested": bool(stop_requested),
    }
    _write_state_json(Path(args.final_json), settings=settings, vqe=vqe_final, psi=psi_final, source="warm_hva_final")
    _ai_log("warm_final_written", energy=float(result.energy), final_json=str(args.final_json), interrupted_requested=bool(stop_requested))

    if best_theta_written is not None and float(result.energy) > float(best_energy_written) + 1e-12:
        _ai_log("warm_note_best_checkpoint_better_than_final", checkpoint_energy=float(best_energy_written), final_energy=float(result.energy))


if __name__ == "__main__":
    main()
