from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.noise_model_spec import (
    NoiseArtifact,
    noise_artifact_metadata,
    normalize_to_resolved_noise_spec,
)


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


def test_fixed_couplers_must_lie_within_fixed_patch() -> None:
    with pytest.raises(ValueError, match="fixed_couplers must lie within fixed_physical_patch"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "shots",
                "backend_profile": "generic_seeded",
                "shots": 128,
                "fixed_physical_patch": [0, 1],
                "fixed_couplers": [[0, 2]],
            }
        )


def test_patch_snapshot_with_fixed_couplers_is_rejected_by_current_normalized_contract() -> None:
    with pytest.raises(ValueError, match="patch_snapshot requires layout_policy='frozen_layout'"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "patch_snapshot",
                "backend_profile": "frozen_snapshot_json",
                "noise_snapshot_json": "artifacts/json/frozen_snapshot.json",
                "shots": 128,
                "fixed_physical_patch": [0, 1],
                "fixed_couplers": [[0, 1], [1, 0]],
            }
        )


def test_runtime_suppressed_bundle_normalizes_consistently_for_runtime_and_qpu_suppressed() -> None:
    runtime_spec = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "runtime",
            "mitigation": "readout",
        }
    )
    qpu_suppressed_spec = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "qpu_suppressed",
            "mitigation": "readout",
        }
    )

    assert runtime_spec.executor == "runtime_qpu"
    assert qpu_suppressed_spec.executor == "runtime_qpu"
    assert runtime_spec.noise_kind == "qpu_suppressed"
    assert qpu_suppressed_spec.noise_kind == "qpu_suppressed"
    assert runtime_spec.mitigation_bundle == "runtime_suppressed"
    assert qpu_suppressed_spec.mitigation_bundle == "runtime_suppressed"


def test_runtime_gate_twirling_normalizes_to_runtime_suppressed_bundle() -> None:
    spec = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "runtime",
            "mitigation": "none",
            "runtime_twirling": {
                "enable_gates": True,
                "enable_measure": False,
                "num_randomizations": 16,
                "strategy": "active",
            },
        }
    )

    assert spec.executor == "runtime_qpu"
    assert spec.noise_kind == "qpu_suppressed"
    assert spec.mitigation_bundle == "runtime_suppressed"
    assert spec.labels["runtime_twirling"]["enable_gates"] is True
    assert spec.labels["runtime_twirling"]["num_randomizations"] == 16
    assert spec.labels["runtime_twirling"]["strategy"] == "active"


def test_qpu_raw_rejects_suppressed_runtime_bundle() -> None:
    with pytest.raises(ValueError, match="qpu_raw requires mitigation='none'"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "qpu_raw",
                "mitigation": "readout",
            }
        )


def test_qpu_raw_rejects_runtime_twirling() -> None:
    with pytest.raises(ValueError, match="runtime twirling disabled"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "qpu_raw",
                "mitigation": "none",
                "runtime_twirling": {"enable_gates": True},
            }
        )


def test_qpu_suppressed_rejects_missing_runtime_suppressed_bundle() -> None:
    with pytest.raises(ValueError, match=r"qpu_suppressed requires mitigation \{readout,zne,dd\} or runtime twirling"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "qpu_suppressed",
                "mitigation": "none",
            }
        )


def test_qpu_layer_learned_normalizes_to_runtime_layer_learned_bundle() -> None:
    spec = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "qpu_layer_learned",
            "mitigation": {"mode": "zne", "zne_scales": [1.0, 2.0, 3.0]},
        }
    )

    assert spec.executor == "runtime_qpu"
    assert spec.noise_kind == "qpu_layer_learned"
    assert spec.mitigation_bundle == "runtime_layer_learned"


def test_qpu_layer_learned_accepts_file_backed_layer_noise_model_intent() -> None:
    spec = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "qpu_layer_learned",
            "mitigation": {
                "mode": "zne",
                "zne_scales": [1.0, 2.0, 3.0],
                "layer_noise_model_json": "artifacts/json/layer_noise_model.json",
            },
        }
    )

    assert spec.mitigation_bundle == "runtime_layer_learned"
    assert spec.labels["layer_noise_model_supplied"] is True


def test_qpu_layer_learned_requires_zne_bundle() -> None:
    with pytest.raises(ValueError, match="qpu_layer_learned requires mitigation='zne'"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "qpu_layer_learned",
                "mitigation": "readout",
            }
        )


def test_qpu_layer_learned_rejects_explicit_runtime_twirling() -> None:
    with pytest.raises(ValueError, match="does not accept explicit runtime twirling"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "qpu_layer_learned",
                "mitigation": {"mode": "zne", "zne_scales": [1.0, 2.0]},
                "runtime_twirling": {"enable_gates": True},
            }
        )


def test_external_layer_noise_model_requires_qpu_layer_learned_mode() -> None:
    with pytest.raises(ValueError, match="layer_noise_model / layer_noise_model_json are only supported on noise_mode='qpu_layer_learned'"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "runtime",
                "mitigation": {
                    "mode": "zne",
                    "zne_scales": [1.0, 2.0],
                    "layer_noise_model": object(),
                },
            }
        )


def test_file_backed_layer_noise_model_requires_qpu_layer_learned_mode() -> None:
    with pytest.raises(ValueError, match="layer_noise_model / layer_noise_model_json are only supported on noise_mode='qpu_layer_learned'"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "runtime",
                "mitigation": {
                    "mode": "zne",
                    "zne_scales": [1.0, 2.0],
                    "layer_noise_model_json": "artifacts/json/layer_noise_model.json",
                },
            }
        )


def test_runtime_twirling_parameters_require_enabled_toggle() -> None:
    with pytest.raises(ValueError, match="runtime twirling parameters require gate or measure twirling"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "runtime",
                "runtime_twirling": {
                    "enable_gates": False,
                    "enable_measure": False,
                    "num_randomizations": 8,
                },
            }
        )


def test_patch_snapshot_requires_frozen_layout_policy() -> None:
    with pytest.raises(ValueError, match="patch_snapshot requires layout_policy='frozen_layout'"):
        normalize_to_resolved_noise_spec(
            {
                "noise_mode": "patch_snapshot",
                "layout_policy": "auto_then_lock",
                "backend_profile": "frozen_snapshot_json",
                "noise_snapshot_json": "artifacts/json/frozen_snapshot.json",
                "shots": 128,
            }
        )



def test_noise_artifact_metadata_preserves_optional_patch_selection_summary() -> None:
    spec = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "backend_scheduled",
            "backend_profile": "generic_seeded",
            "schedule_policy": "asap",
            "shots": 128,
        }
    )
    artifact = NoiseArtifact(
        resolved_spec=spec,
        calibration_snapshot=None,
        transpile_snapshot=None,
        qiskit_backend=None,
        qiskit_noise_model=None,
        transpiled_circuit=None,
        scheduled_circuit_or_none=None,
        patch_selection_summary={
            "patch_selection_objective": ["min_extra_2q_count"],
            "selected_physical_qubits": [0, 1],
            "selected_edges": [[0, 1], [1, 0]],
            "patch_score_summary": {"extra_2q_count": 0},
        },
    )

    metadata = noise_artifact_metadata(artifact)

    assert metadata["patch_selection_summary"]["selected_physical_qubits"] == [0, 1]
    assert metadata["patch_selection_summary"]["patch_score_summary"]["extra_2q_count"] == 0



def test_noise_artifact_metadata_keeps_backwards_compatible_none_patch_selection_summary() -> None:
    spec = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "backend_basic",
            "backend_profile": "generic_seeded",
            "shots": 128,
        }
    )
    artifact = NoiseArtifact(
        resolved_spec=spec,
        calibration_snapshot=None,
        transpile_snapshot=None,
        qiskit_backend=None,
        qiskit_noise_model=None,
        transpiled_circuit=None,
        scheduled_circuit_or_none=None,
    )

    metadata = noise_artifact_metadata(artifact)

    assert "patch_selection_summary" in metadata
    assert metadata["patch_selection_summary"] is None
