from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.noise_model_spec import normalize_to_resolved_noise_spec


def test_ideal_noise_spec_uses_not_applicable_backend_profile() -> None:
    spec = normalize_to_resolved_noise_spec({"noise_mode": "ideal"})
    assert spec.executor == "statevector"
    assert spec.noise_kind == "none"
    assert spec.backend_profile_kind == "not_applicable"


def test_ideal_noise_spec_rejects_allow_noisy_fallback() -> None:
    with pytest.raises(ValueError, match="allow_noisy_fallback"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "ideal",
                "allow_noisy_fallback": True,
            }
        )


def test_backend_basic_rejects_schedule_aware_tuple() -> None:
    with pytest.raises(ValueError, match="backend_basic must use schedule_policy='none'"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "backend_basic",
                "backend_profile": "generic_seeded",
                "schedule_policy": "asap",
                "shots": 128,
            }
        )


def test_fixed_couplers_require_fixed_patch() -> None:
    with pytest.raises(ValueError, match="fixed_couplers requires fixed_physical_patch"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "shots",
                "backend_profile": "generic_seeded",
                "shots": 128,
                "fixed_couplers": [[1, 2]],
            }
        )
