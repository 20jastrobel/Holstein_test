#!/usr/bin/env python3
"""Run a Hubbard-Holstein sweep over coupling strengths and report energy errors.

Usage:
  python Adapt-VQE-Pipeline/run_hh_crowd_state_sweep.py --method adapt --min-g 0.8 --max-g 3.0 --step 0.2
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ADAPT_PIPELINE = ROOT / "pipelines" / "hardcoded_adapt_pipeline.py"
VQE_PIPELINE = ROOT.parent / "pipelines" / "hardcoded_hubbard_pipeline.py"
ARTIFACT_DIR = ROOT / "artifacts" / "sweep_runs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str], out_json: Path) -> dict[str, Any]:
    cmd = [str(c) for c in cmd] + ["--skip-pdf", "--output-json", str(out_json)]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    del proc
    return json.loads(out_json.read_text(encoding="utf-8"))


def run_adapt_paop_full(g: float, reps: int = 220) -> tuple[float, int, str]:
    payload_file = ARTIFACT_DIR / f"adapt_paop_full_g{g:.2f}.json"
    payload = _run(
        [
            sys.executable,
            str(ADAPT_PIPELINE),
            "--L",
            "2",
            "--problem",
            "hh",
            "--t",
            "0.2",
            "--u",
            "0.2",
            "--omega0",
            "0.2",
            "--g-ep",
            str(g),
            "--n-ph-max",
            "1",
            "--boson-encoding",
            "binary",
            "--boundary",
            "open",
            "--ordering",
            "blocked",
            "--adapt-pool",
            "paop_full",
            "--adapt-max-depth",
            str(reps),
            "--adapt-maxiter",
            "5000",
            "--adapt-eps-grad",
            "1e-12",
            "--adapt-eps-energy",
            "1e-10",
            "--adapt-no-repeats",
            "--adapt-no-finite-angle-fallback",
            "--adapt-seed",
            "7",
            "--paop-r",
            "1",
            "--initial-state-source",
            "hf",
        ],
        payload_file,
    )
    ad = payload["adapt_vqe"]
    return float(abs(float(ad["abs_delta_e"]))), int(payload["adapt_vqe"]["ansatz_depth"]), "adapt_vqe"


def run_hh_hva_vqe(g: float, reps: int = 8, restarts: int = 3) -> tuple[float, None, str]:
    payload_file = ARTIFACT_DIR / f"vqe_hh_hva_g{g:.2f}.json"
    payload = _run(
        [
            sys.executable,
            str(VQE_PIPELINE),
            "--L",
            "2",
            "--t",
            "0.2",
            "--u",
            "0.2",
            "--omega0",
            "0.2",
            "--g-ep",
            str(g),
            "--n-ph-max",
            "1",
            "--boson-encoding",
            "binary",
            "--boundary",
            "open",
            "--ordering",
            "blocked",
            "--vqe-ansatz",
            "hh_hva",
            "--vqe-reps",
            str(reps),
            "--vqe-restarts",
            str(restarts),
            "--vqe-maxiter",
            "1200",
            "--vqe-seed",
            "7",
            "--skip-qpe",
            "--initial-state-source",
            "hf",
        ],
        payload_file,
    )
    v = payload["vqe"]
    return float(abs(float(v["energy"]) - float(v["exact_filtered_energy"]))), None, "vqe_hh_hva"


def run_hh_uccsd_vqe(g: float, reps: int = 6, restarts: int = 3) -> tuple[float, None, str]:
    payload_file = ARTIFACT_DIR / f"vqe_uccsd_g{g:.2f}.json"
    payload = _run(
        [
            sys.executable,
            str(VQE_PIPELINE),
            "--L",
            "2",
            "--t",
            "0.2",
            "--u",
            "0.2",
            "--omega0",
            "0.2",
            "--g-ep",
            str(g),
            "--n-ph-max",
            "1",
            "--boson-encoding",
            "binary",
            "--boundary",
            "open",
            "--ordering",
            "blocked",
            "--vqe-ansatz",
            "uccsd",
            "--vqe-reps",
            str(reps),
            "--vqe-restarts",
            str(restarts),
            "--vqe-maxiter",
            "1200",
            "--vqe-seed",
            "7",
            "--skip-qpe",
            "--initial-state-source",
            "hf",
        ],
        payload_file,
    )
    v = payload["vqe"]
    return float(abs(float(v["energy"]) - float(v["exact_filtered_energy"]))), None, "vqe_uccsd"


def make_grid(min_g: float, max_g: float, step: float) -> list[float]:
    vals = []
    g = float(min_g)
    eps = 1e-12
    while g <= max_g + eps:
        vals.append(round(g, 12))
        g = round(g + step, 12)
    return vals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["adapt", "vqe"], default="adapt")
    parser.add_argument(
        "--vqe-ansatz",
        choices=["hh_hva", "uccsd"],
        default="hh_hva",
        help="VQE ansatz for method='vqe'.",
    )
    parser.add_argument("--min-g", type=float, default=0.8)
    parser.add_argument("--max-g", type=float, default=3.0)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--vqe-reps", type=int, default=8)
    parser.add_argument("--vqe-restarts", type=int, default=3)
    args = parser.parse_args()

    grid = make_grid(args.min_g, args.max_g, args.step)
    print(f"method={args.method}  threshold={args.threshold}")
    print("g\tabs_delta_e\tstatus\tmeta")
    for g in grid:
        if args.method == "adapt":
            abs_delta, depth, label = run_adapt_paop_full(g)
            meta = f"depth={depth}"
        else:
            if str(args.vqe_ansatz).lower() == "uccsd":
                abs_delta, depth, label = run_hh_uccsd_vqe(g, reps=args.vqe_reps, restarts=args.vqe_restarts)
            else:
                abs_delta, depth, label = run_hh_hva_vqe(g, reps=args.vqe_reps, restarts=args.vqe_restarts)
            meta = "depth=n/a"
        status = "PASS" if abs_delta < args.threshold else "MISS"
        print(f"{g:.2f}\t{abs_delta:.12e}\t{status}\t{label}\t{meta}")


if __name__ == "__main__":
    main()
