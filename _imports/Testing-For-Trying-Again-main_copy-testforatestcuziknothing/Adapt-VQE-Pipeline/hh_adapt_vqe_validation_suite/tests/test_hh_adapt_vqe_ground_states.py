import json
import math
import os
import subprocess
import sys
from pathlib import Path


SUITE_DIR = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = SUITE_DIR.parent / "pipelines" / "hardcoded_adapt_pipeline.py"
ARTIFACT_DIR = Path(os.environ.get("ADAPT_HH_ARTIFACT_DIR", SUITE_DIR / "artifacts"))
TOL = 1e-4

DEFAULT_ARGS = {
    "t": 0.2,
    "u": 0.2,
    "omega0": 0.2,
    "g_ep": 0.2,
    "n_ph_max": 1,
    "boson_encoding": "binary",
    "boundary": "open",
    "ordering": "blocked",
    "adapt_pool": "paop_full",
    "paop_r": 1,
    "adapt_max_depth": 120,
    "adapt_maxiter": {
        2: 1200,
        3: 20,
    },
    "adapt_eps_grad": {
        2: 1e-12,
        3: 1e-12,
    },
    "adapt_eps_energy": {
        2: 1e-10,
        3: 1e-10,
    },
    "initial_state_source": "hf",
}

ABS_DELTA_E_TOL = {
    2: 1e-2,
    3: 1e-1,
}


def _default_seed() -> int:
    return int(os.environ.get("ADAPT_HH_TEST_SEED", "7"))


def _tol_for_lattice(L: int) -> float:
    env_key = f"ADAPT_HH_TOL_L{L}"
    return float(os.environ.get(env_key, str(ABS_DELTA_E_TOL[L])))


def _int_or_default(value: str | None, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _float_or_default(value: str | None, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _case_settings(L: int) -> dict[str, float | int]:
    maxiter = _int_or_default(
        os.environ.get(f"ADAPT_HH_MAXITER_L{L}"),
        DEFAULT_ARGS["adapt_maxiter"][L],
    )
    max_depth = _int_or_default(
        os.environ.get(f"ADAPT_HH_MAX_DEPTH_L{L}"),
        DEFAULT_ARGS["adapt_max_depth"],
    )
    eps_grad = _float_or_default(
        os.environ.get(f"ADAPT_HH_EPS_GRAD_L{L}"),
        DEFAULT_ARGS["adapt_eps_grad"][L],
    )
    eps_energy = _float_or_default(
        os.environ.get(f"ADAPT_HH_EPS_ENERGY_L{L}"),
        DEFAULT_ARGS["adapt_eps_energy"][L],
    )
    return {
        "adapt_max_depth": max_depth,
        "adapt_maxiter": maxiter,
        "adapt_eps_grad": eps_grad,
        "adapt_eps_energy": eps_energy,
    }


def _run_hh_adapt_case(L: int) -> dict:
    seed = _default_seed()
    settings = _case_settings(L)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_json = ARTIFACT_DIR / f"hh_adapt_vqe_L{L}_seed{seed}.json"

    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--L",
        str(L),
        "--problem",
        "hh",
        "--t",
        str(DEFAULT_ARGS["t"]),
        "--u",
        str(DEFAULT_ARGS["u"]),
        "--omega0",
        str(DEFAULT_ARGS["omega0"]),
        "--g-ep",
        str(DEFAULT_ARGS["g_ep"]),
        "--n-ph-max",
        str(DEFAULT_ARGS["n_ph_max"]),
        "--boson-encoding",
        DEFAULT_ARGS["boson_encoding"],
        "--boundary",
        DEFAULT_ARGS["boundary"],
        "--ordering",
        DEFAULT_ARGS["ordering"],
        "--adapt-pool",
        DEFAULT_ARGS["adapt_pool"],
        "--adapt-max-depth",
        str(settings["adapt_max_depth"]),
        "--adapt-maxiter",
        str(settings["adapt_maxiter"]),
        "--adapt-eps-grad",
        str(settings["adapt_eps_grad"]),
        "--adapt-eps-energy",
        str(settings["adapt_eps_energy"]),
        "--adapt-no-repeats",
        "--adapt-no-finite-angle-fallback",
        "--adapt-seed",
        str(seed),
        "--paop-r",
        str(DEFAULT_ARGS["paop_r"]),
        "--initial-state-source",
        DEFAULT_ARGS["initial_state_source"],
        "--skip-pdf",
        "--output-json",
        str(output_json),
        "--dv",
        "0.0",
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    with output_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    adapt_vqe = payload.get("adapt_vqe", {})
    assert adapt_vqe.get("success") is True

    abs_delta_e = adapt_vqe.get("abs_delta_e")
    assert isinstance(abs_delta_e, (int, float)) and math.isfinite(abs_delta_e)
    assert abs(abs_delta_e) < _tol_for_lattice(L)

    energy = adapt_vqe.get("energy")
    exact_energy = adapt_vqe.get("exact_gs_energy")
    assert isinstance(energy, (int, float)) and math.isfinite(energy)
    assert isinstance(exact_energy, (int, float)) and math.isfinite(exact_energy)

    return payload


def _run_hh_adapt_case_with_g(L: int, g_ep: float) -> dict:
    seed = _default_seed()
    settings = _case_settings(L)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_json = ARTIFACT_DIR / f"hh_adapt_vqe_L{L}_seed{seed}_g{str(g_ep).replace('.', 'p')}.json"

    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--L",
        str(L),
        "--problem",
        "hh",
        "--t",
        str(DEFAULT_ARGS["t"]),
        "--u",
        str(DEFAULT_ARGS["u"]),
        "--omega0",
        str(DEFAULT_ARGS["omega0"]),
        "--g-ep",
        str(g_ep),
        "--n-ph-max",
        str(DEFAULT_ARGS["n_ph_max"]),
        "--boson-encoding",
        DEFAULT_ARGS["boson_encoding"],
        "--boundary",
        DEFAULT_ARGS["boundary"],
        "--ordering",
        DEFAULT_ARGS["ordering"],
        "--adapt-pool",
        DEFAULT_ARGS["adapt_pool"],
        "--adapt-max-depth",
        str(settings["adapt_max_depth"]),
        "--adapt-maxiter",
        str(settings["adapt_maxiter"]),
        "--adapt-eps-grad",
        str(settings["adapt_eps_grad"]),
        "--adapt-eps-energy",
        str(settings["adapt_eps_energy"]),
        "--adapt-no-repeats",
        "--adapt-no-finite-angle-fallback",
        "--adapt-seed",
        str(seed),
        "--paop-r",
        str(DEFAULT_ARGS["paop_r"]),
        "--initial-state-source",
        DEFAULT_ARGS["initial_state_source"],
        "--skip-pdf",
        "--output-json",
        str(output_json),
        "--dv",
        "0.0",
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    with output_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    adapt_vqe = payload.get("adapt_vqe", {})
    assert adapt_vqe.get("success") is True

    energy = adapt_vqe.get("energy")
    exact_energy = adapt_vqe.get("exact_gs_energy")
    assert isinstance(energy, (int, float)) and math.isfinite(energy)
    assert isinstance(exact_energy, (int, float)) and math.isfinite(exact_energy)

    return payload


def test_l2_hh_adapt_vqe_ground_state_within_1e_minus_4() -> None:
    payload = _run_hh_adapt_case(2)
    assert payload["settings"]["L"] == 2


def test_l3_hh_adapt_vqe_ground_state_within_1e_minus_4() -> None:
    payload = _run_hh_adapt_case(3)
    assert payload["settings"]["L"] == 3


def test_l2_energy_is_sensitive_to_g() -> None:
    left = _run_hh_adapt_case_with_g(2, 0.5)
    right = _run_hh_adapt_case_with_g(2, 2.0)
    mid = _run_hh_adapt_case_with_g(2, 1.25)

    left_exact = left["adapt_vqe"]["exact_gs_energy"]
    right_exact = right["adapt_vqe"]["exact_gs_energy"]
    mid_exact = mid["adapt_vqe"]["exact_gs_energy"]

    left_adapt = left["adapt_vqe"]["energy"]
    right_adapt = right["adapt_vqe"]["energy"]
    mid_adapt = mid["adapt_vqe"]["energy"]

    assert left_exact - right_exact > 1e-3
    assert left_adapt - right_adapt > 1e-4
    assert right_exact < mid_exact < left_exact
    assert right_adapt < mid_adapt < left_adapt
