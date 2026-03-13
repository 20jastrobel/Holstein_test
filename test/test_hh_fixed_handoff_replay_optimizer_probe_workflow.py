from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.exact_bench.hh_fixed_handoff_replay_optimizer_probe_workflow as wf
from pipelines.hardcoded.hh_vqe_from_adapt_family import RunConfig
from src.quantum.vqe_latex_python_pairs import VQEResult


class _DummyAnsatz:
    num_parameters = 2

    def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        return np.asarray(psi_ref, dtype=complex)


def _run_cfg(tmp_path: Path) -> RunConfig:
    return RunConfig(
        adapt_input_json=tmp_path / "handoff.json",
        output_json=tmp_path / "replay.json",
        output_csv=tmp_path / "replay.csv",
        output_md=tmp_path / "replay.md",
        output_log=tmp_path / "replay.log",
        tag="probe_test",
        generator_family="match_adapt",
        fallback_family="full_meta",
        legacy_paop_key="paop_lf_std",
        replay_seed_policy="auto",
        replay_continuation_mode="legacy",
        L=2,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=1.0,
        g_ep=1.0,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        sector_n_up=1,
        sector_n_dn=1,
        reps=3,
        restarts=6,
        maxiter=4000,
        method="SPSA",
        seed=19,
        energy_backend="one_apply_compiled",
        progress_every_s=60.0,
        wallclock_cap_s=3600,
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        spsa_a=0.2,
        spsa_c=0.1,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        spsa_A=10.0,
        spsa_avg_last=0,
        spsa_eval_repeats=1,
        spsa_eval_agg="mean",
        replay_freeze_fraction=0.2,
        replay_unfreeze_fraction=0.3,
        replay_full_fraction=0.5,
        replay_qn_spsa_refresh_every=5,
        replay_qn_spsa_refresh_mode="diag_rms_grad",
        phase3_symmetry_mitigation_mode="off",
    )


def test_variant_specs_include_baseline_heavy_and_deterministic(tmp_path: Path) -> None:
    cfg = wf.ReplayOptimizerProbeConfig(adapt_input_json=tmp_path / "handoff.json")
    variants = wf._variant_specs(cfg)
    assert [variant.label for variant in variants] == [
        "baseline_spsa",
        "heavy_spsa",
        "deterministic_l_bfgs_b",
    ]
    assert [variant.method for variant in variants] == ["SPSA", "SPSA", "L-BFGS-B"]


def test_variant_specs_reject_unsupported_deterministic_method(tmp_path: Path) -> None:
    cfg = wf.ReplayOptimizerProbeConfig(
        adapt_input_json=tmp_path / "handoff.json",
        deterministic_method="CG",
    )
    with pytest.raises(ValueError):
        wf._variant_specs(cfg)


def test_build_base_replay_config_rejects_nonlegacy_continuation(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps({"settings": {"L": 2, "t": 1.0, "u": 4.0, "dv": 0.0, "omega0": 1.0, "g_ep": 1.0, "n_ph_max": 2}}), encoding="utf-8")
    cfg = wf.ReplayOptimizerProbeConfig(
        adapt_input_json=handoff,
        replay_continuation_mode="phase3_v1",
    )
    with pytest.raises(ValueError, match="legacy-only"):
        wf.build_base_replay_config(cfg)


def test_run_probe_variant_allows_non_spsa_method(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base_cfg = _run_cfg(tmp_path)
    context = {
        "base_cfg": base_cfg,
        "payload": {},
        "psi_ref": np.array([1.0 + 0.0j, 0.0 + 0.0j]),
        "h_poly": object(),
        "exact_energy": 1.0,
        "replay_ctx": {
            "ansatz": _DummyAnsatz(),
            "seed_theta": np.array([0.1, -0.2]),
            "seed_energy": 1.25,
            "seed_delta_abs": 0.25,
            "family_info": {"requested": "match_adapt"},
            "family_resolved": "full_meta",
            "replay_terms": ["a", "b"],
            "adapt_labels": ["x", "y"],
        },
    }
    seen: dict[str, object] = {}

    def _fake_vqe_minimize(*args, **kwargs):
        seen["method"] = kwargs["method"]
        seen["initial_point"] = np.asarray(kwargs["initial_point"], dtype=float).copy()
        return VQEResult(
            energy=0.91,
            theta=np.array([0.3, 0.4]),
            success=True,
            message="ok",
            nfev=12,
            nit=7,
            best_restart=0,
            restart_summaries=[{"restart_index": 1, "energy": 0.91}],
            optimizer_memory={"available": False},
        )

    monkeypatch.setattr(wf, "vqe_minimize", _fake_vqe_minimize)
    monkeypatch.setattr(wf, "_scipy_available", lambda: True)
    variant = wf.ProbeVariant(
        label="deterministic_l_bfgs_b",
        method="L-BFGS-B",
        restarts=3,
        maxiter=200,
    )
    probe_cfg = wf.ReplayOptimizerProbeConfig(
        adapt_input_json=tmp_path / "handoff.json",
        output_json=tmp_path / "probe.json",
        output_csv=tmp_path / "probe.csv",
        tag="probe_test",
    )

    row = wf.run_probe_variant(probe_cfg, context, variant)

    assert seen["method"] == "L-BFGS-B"
    assert np.allclose(seen["initial_point"], np.array([0.1, -0.2]))
    assert row["delta_abs"] == pytest.approx(0.09)
    assert row["improvement_from_seed"] == pytest.approx(0.16)
    assert row["family_resolved"] == "full_meta"


def test_build_payload_and_emit_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = wf.ReplayOptimizerProbeConfig(
        adapt_input_json=tmp_path / "handoff.json",
        output_json=tmp_path / "probe.json",
        output_csv=tmp_path / "probe.csv",
        tag="probe_test",
    )
    base_cfg = _run_cfg(tmp_path)
    monkeypatch.setattr(
        wf,
        "build_probe_context",
        lambda _cfg: {
            "base_cfg": base_cfg,
            "payload": {},
            "psi_ref": np.array([1.0 + 0.0j, 0.0 + 0.0j]),
            "h_poly": object(),
            "exact_energy": 1.0,
            "exact_energy_source": "payload",
            "replay_ctx": {
                "seed_energy": 1.2,
                "seed_delta_abs": 0.2,
                "family_info": {"requested": "match_adapt"},
                "family_resolved": "full_meta",
                "replay_terms": ["a", "b"],
                "adapt_labels": ["x", "y"],
                "resolved_seed_policy": "residual_only",
                "handoff_state_kind": "prepared_state",
            },
        },
    )
    rows_iter = iter(
        [
            {
                "variant": "baseline_spsa",
                "method": "SPSA",
                "delta_abs": 0.03,
                "seed_delta_abs": 0.2,
                "improvement_from_seed": 0.17,
                "progress_tail": [],
                "restart_summaries": [],
                "optimizer_memory": None,
            },
            {
                "variant": "heavy_spsa",
                "method": "SPSA",
                "delta_abs": 0.028,
                "seed_delta_abs": 0.2,
                "improvement_from_seed": 0.172,
                "progress_tail": [],
                "restart_summaries": [],
                "optimizer_memory": None,
            },
            {
                "variant": "deterministic_l_bfgs_b",
                "method": "L-BFGS-B",
                "delta_abs": 0.004,
                "seed_delta_abs": 0.2,
                "improvement_from_seed": 0.196,
                "progress_tail": [],
                "restart_summaries": [],
                "optimizer_memory": None,
            },
        ]
    )
    monkeypatch.setattr(wf, "run_probe_variant", lambda *_args, **_kwargs: dict(next(rows_iter)))

    payload = wf.build_probe_payload(cfg)
    wf.emit_probe_files(payload, cfg)

    assert payload["summary"]["diagnostic_signal"] == "optimizer_sensitive"
    assert payload["summary"]["best_variant"]["variant"] == "deterministic_l_bfgs_b"

    json_payload = json.loads(cfg.output_json.read_text(encoding="utf-8"))
    assert json_payload["summary"]["diagnostic_signal"] == "optimizer_sensitive"

    with cfg.output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["variant"] for row in rows] == [
        "baseline_spsa",
        "heavy_spsa",
        "deterministic_l_bfgs_b",
    ]
    assert float(rows[-1]["improvement_vs_baseline"]) == pytest.approx(0.026)
