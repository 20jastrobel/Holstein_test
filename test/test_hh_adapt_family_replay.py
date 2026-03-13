from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.hardcoded.hh_vqe_from_adapt_family as replay_mod
from pipelines.hardcoded.hh_vqe_from_adapt_family import (
    RunConfig,
    _build_cfg,
    _build_pool_for_family,
    _resolve_family,
    _resolve_family_from_metadata,
    build_replay_ansatz_context,
    parse_args,
)


def _mk_cfg(tmp_path: Path, *, generator_family: str = "match_adapt", fallback_family: str = "full_meta") -> RunConfig:
    return RunConfig(
        adapt_input_json=tmp_path / "in.json",
        output_json=tmp_path / "out.json",
        output_csv=tmp_path / "out.csv",
        output_md=tmp_path / "out.md",
        output_log=tmp_path / "out.log",
        tag="test",
        generator_family=generator_family,
        fallback_family=fallback_family,
        legacy_paop_key="paop_lf_std",
        replay_seed_policy="auto",
        replay_continuation_mode="legacy",
        L=2,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        sector_n_up=1,
        sector_n_dn=1,
        reps=2,
        restarts=2,
        maxiter=20,
        method="SPSA",
        seed=7,
        energy_backend="one_apply_compiled",
        progress_every_s=60.0,
        wallclock_cap_s=600,
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


def test_parse_defaults_match_adapt_and_spsa() -> None:
    args = parse_args(["--adapt-input-json", "dummy.json"])
    assert str(args.generator_family) == "match_adapt"
    assert str(args.fallback_family) == "full_meta"
    assert str(args.method) == "SPSA"
    assert args.replay_continuation_mode is None



def test_parse_rejects_non_spsa_method() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--adapt-input-json", "dummy.json", "--method", "COBYLA"])


def test_build_cfg_keeps_replay_continuation_mode_none(tmp_path: Path) -> None:
    payload = {
        "settings": {
            "L": 2,
            "t": 1.0,
            "U": 4.0,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.5,
            "n_ph_max": 1,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "sector_n_up": 1,
            "sector_n_dn": 1,
        }
    }
    in_json = tmp_path / "adapt_contract_replay_mode.json"
    in_json.write_text(json.dumps(payload), encoding="utf-8")
    payload = json.loads(in_json.read_text(encoding="utf-8"))
    args = parse_args(["--adapt-input-json", str(in_json)])
    cfg = _build_cfg(args, payload)
    assert cfg.replay_continuation_mode is None


def test_parse_rejects_auto_replay_continuation_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--adapt-input-json", "dummy.json", "--replay-continuation-mode", "auto"])


def test_parse_accepts_phase2_replay_continuation_mode() -> None:
    args = parse_args(["--adapt-input-json", "dummy.json", "--replay-continuation-mode", "phase2_v1"])
    assert str(args.replay_continuation_mode) == "phase2_v1"


def test_parse_accepts_phase3_replay_continuation_mode() -> None:
    args = parse_args(["--adapt-input-json", "dummy.json", "--replay-continuation-mode", "phase3_v1"])
    assert str(args.replay_continuation_mode) == "phase3_v1"


def test_resolve_family_prefers_adapt_vqe_pool_type() -> None:
    fam, src = _resolve_family_from_metadata({"adapt_vqe": {"pool_type": "full_meta"}})
    assert fam == "full_meta"
    assert src == "adapt_vqe.pool_type"


def test_resolve_family_uses_settings_adapt_pool() -> None:
    fam, src = _resolve_family_from_metadata({"settings": {"adapt_pool": "uccsd_paop_lf_full"}})
    assert fam == "uccsd_paop_lf_full"
    assert src == "settings.adapt_pool"


def test_resolve_family_accepts_new_experimental_paop_tokens() -> None:
    for token in (
        "paop_lf3_std",
        "paop_lf4_std",
        "paop_sq_std",
        "paop_sq_full",
        "paop_bond_disp_std",
        "paop_hop_sq_std",
        "paop_pair_sq_std",
    ):
        fam, src = _resolve_family_from_metadata({"adapt_vqe": {"pool_type": token}})
        assert fam == token
        assert src == "adapt_vqe.pool_type"


def test_resolve_family_accepts_new_vlf_sq_tokens() -> None:
    for token in ("vlf_only", "sq_only", "vlf_sq", "sq_dens_only", "vlf_sq_dens"):
        fam, src = _resolve_family_from_metadata({"adapt_vqe": {"pool_type": token}})
        assert fam == token
        assert src == "adapt_vqe.pool_type"


def test_build_pool_for_family_materializes_new_paop_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _mk_cfg(tmp_path)
    monkeypatch.setattr(
        replay_mod,
        "_build_paop_pool",
        lambda *args, **kwargs: [type("_T", (), {"label": f"{kwargs.get('pool_key', args[5])}:term"})()],
    )
    for token in (
        "paop_lf3_std",
        "paop_lf4_std",
        "paop_sq_std",
        "paop_sq_full",
        "paop_bond_disp_std",
        "paop_hop_sq_std",
        "paop_pair_sq_std",
    ):
        pool, meta = _build_pool_for_family(cfg, family=token, h_poly=object())
        assert len(pool) == 1
        assert meta["family"] == token


def test_build_pool_for_family_materializes_new_vlf_sq_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _mk_cfg(tmp_path)
    monkeypatch.setattr(
        replay_mod,
        "_build_vlf_sq_pool",
        lambda *args, **kwargs: (
            [type("_T", (), {"label": f"{kwargs.get('pool_key', args[5])}:macro"})()],
            {"family": kwargs.get('pool_key', args[5]), "parameter_count": 1},
        ),
    )
    for token in ("vlf_only", "sq_only", "vlf_sq", "sq_dens_only", "vlf_sq_dens"):
        pool, meta = _build_pool_for_family(cfg, family=token, h_poly=object())
        assert len(pool) == 1
        assert meta["family"] == token
        assert meta["family_kind"] == "macro_probe"


def test_build_pool_for_family_rejects_split_paulis_for_vlf_sq(
    tmp_path: Path,
) -> None:
    cfg = _mk_cfg(tmp_path)
    cfg = cfg.__class__(**{**cfg.__dict__, "paop_split_paulis": True})
    with pytest.raises(ValueError, match="do not support --paop-split-paulis"):
        _build_pool_for_family(cfg, family="vlf_sq", h_poly=object())


def test_resolve_family_uses_nested_selected_generator_metadata() -> None:
    fam, src = _resolve_family_from_metadata(
        {
            "adapt_vqe": {
                "pool_type": "phase3_v1",
                "continuation": {
                    "selected_generator_metadata": [
                        {"family_id": "paop_lf_std"},
                        {"family_id": "paop_lf_std"},
                    ]
                },
            }
        }
    )
    assert fam == "paop_lf_std"
    assert src == "adapt_vqe.continuation.selected_generator_metadata.family_id"


def test_resolve_family_maps_legacy_pool_variant() -> None:
    fam, src = _resolve_family_from_metadata({"meta": {"pool_variant": "B"}})
    assert fam == "pool_b"
    assert src == "meta.pool_variant"


def test_resolve_family_match_adapt_prefers_contract_and_falls_back_when_missing(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, generator_family="match_adapt", fallback_family="full_meta")
    info = _resolve_family(
        cfg,
        {
            "continuation": {
                "replay_contract": {
                    "contract_version": 2,
                    "generator_family": {
                        "requested": "match_adapt",
                        "resolved": "paop_lf_std",
                        "resolution_source": "selected_generator_metadata.family_id",
                        "fallback_family": "full_meta",
                        "fallback_used": False,
                    },
                    "seed_policy_requested": "auto",
                    "seed_policy_resolved": "residual_only",
                    "handoff_state_kind": "prepared_state",
                    "continuation_mode": "phase1_v1",
                }
            }
        }
    )
    assert info["requested"] == "match_adapt"
    assert info["resolved"] == "paop_lf_std"
    assert info["resolution_source"] == "selected_generator_metadata.family_id"
    assert info["fallback_used"] is False


def test_resolve_family_match_adapt_falls_back_when_missing(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, generator_family="match_adapt", fallback_family="full_meta")
    info = _resolve_family(cfg, {})
    assert info["requested"] == "match_adapt"
    assert info["resolved"] == "full_meta"
    assert bool(info["fallback_used"]) is True
    assert info["resolution_source"] == "fallback_family"


def test_resolve_family_rejects_malformed_contract(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, generator_family="match_adapt", fallback_family="full_meta")
    with pytest.raises(ValueError, match="continuation.replay_contract"):
        _resolve_family(
            cfg,
            {
                "continuation": {
                    "replay_contract": {
                        "contract_version": 2,
                    "generator_family": {
                        "requested": "match_adapt",
                        "resolved": "not_a_family",
                    },
                    "seed_policy_requested": "auto",
                    "seed_policy_resolved": "residual_only",
                    "handoff_state_kind": "prepared_state",
                    "continuation_mode": "legacy",
                }
            }
        }
    )


def test_build_replay_ansatz_context_retries_with_full_meta_on_missing_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _mk_cfg(tmp_path, generator_family="match_adapt", fallback_family="full_meta")
    payload = {
        "adapt_vqe": {"operators": ["g0"], "optimal_point": [0.1]},
        "initial_state": {"handoff_state_kind": "prepared_state"},
    }
    family_info = {
        "requested": "match_adapt",
        "resolved": "paop_lf_std",
        "resolution_source": "selected_generator_metadata.family_id",
        "fallback_family": "full_meta",
        "fallback_used": False,
        "warning": None,
    }
    calls: list[str] = []

    class _FakeAnsatz:
        def __init__(self, *, terms: list[object], reps: int, nq: int) -> None:
            self.terms = list(terms)
            self.num_parameters = int(len(terms) * reps)

        def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
            return np.asarray(psi_ref, dtype=complex)

    def _fake_build(
        _cfg: RunConfig,
        *,
        family: str,
        h_poly: object,
        adapt_labels: list[str],
        payload: dict[str, object] | None = None,
    ) -> tuple[list[object], dict[str, object], int]:
        del _cfg, h_poly, adapt_labels, payload
        calls.append(str(family))
        if family == "paop_lf_std":
            raise ValueError(
                "ADAPT operators are not present in the resolved replay family pool. "
                "Missing examples: ['g0']"
            )
        assert family == "full_meta"
        return ["term0"], {"family": "full_meta", "raw_total": 7}, 7

    monkeypatch.setattr(replay_mod, "_build_replay_terms_for_family", _fake_build)
    monkeypatch.setattr(replay_mod, "PoolTermwiseAnsatz", _FakeAnsatz)
    monkeypatch.setattr(
        replay_mod,
        "_build_replay_seed_theta_policy",
        lambda adapt_theta, reps, policy, handoff_state_kind: (np.zeros(int(reps), dtype=float), "residual_only"),
    )
    monkeypatch.setattr(replay_mod, "expval_pauli_polynomial", lambda psi, h_poly: 0.25)

    replay_ctx = build_replay_ansatz_context(
        cfg,
        payload_in=payload,
        psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        h_poly=object(),
        family_info=family_info,
        e_exact=0.2,
    )

    assert calls == ["paop_lf_std", "full_meta"]
    assert replay_ctx["family_resolved"] == "full_meta"
    assert replay_ctx["family_terms_count"] == 7
    assert replay_ctx["pool_meta"]["family"] == "full_meta"
    assert replay_ctx["family_info"]["resolved"] == "full_meta"
    assert replay_ctx["family_info"]["resolution_source"] == "fallback_family_missing_labels"
    assert replay_ctx["family_info"]["fallback_used"] is True
    assert "retrying replay with fallback family 'full_meta'" in str(replay_ctx["family_info"]["warning"])


def test_run_records_effective_family_after_replay_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _mk_cfg(tmp_path, generator_family="match_adapt", fallback_family="full_meta")
    psi_ref = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)

    class _FakeAnsatz:
        def __init__(self, *, terms: list[object], reps: int, nq: int) -> None:
            del terms, reps, nq
            self.num_parameters = 2

        def prepare_state(self, theta: np.ndarray, psi_ref_in: np.ndarray) -> np.ndarray:
            del theta
            return np.asarray(psi_ref_in, dtype=complex)

    monkeypatch.setattr(replay_mod, "_read_input_state_and_payload", lambda path: (psi_ref, {}))
    monkeypatch.setattr(
        replay_mod,
        "_resolve_family",
        lambda cfg_in, payload_in: {
            "requested": "match_adapt",
            "resolved": "paop_lf_std",
            "resolution_source": "selected_generator_metadata.family_id",
            "fallback_family": "full_meta",
            "fallback_used": False,
            "warning": None,
        },
    )
    monkeypatch.setattr(replay_mod, "_build_hh_hamiltonian", lambda cfg_in: object())
    monkeypatch.setattr(replay_mod, "_resolve_exact_energy_from_payload", lambda payload_in: 0.0)
    monkeypatch.setattr(
        replay_mod,
        "build_replay_ansatz_context",
        lambda cfg_in, **kwargs: {
            "adapt_labels": ["g0"],
            "adapt_theta": np.array([0.1], dtype=float),
            "handoff_state_kind": "prepared_state",
            "provenance_source": "explicit",
            "seed_theta": np.zeros(2, dtype=float),
            "resolved_seed_policy": "residual_only",
            "family_info": {
                "requested": "match_adapt",
                "resolved": "full_meta",
                "resolution_source": "fallback_family_missing_labels",
                "fallback_family": "full_meta",
                "fallback_used": True,
                "warning": "Resolved replay family could not represent the ADAPT-selected labels; retrying replay with fallback family 'full_meta'.",
            },
            "family_resolved": "full_meta",
            "family_terms_count": 7,
            "pool_meta": {"family": "full_meta", "raw_total": 7},
            "replay_terms": ["term0"],
            "ansatz": _FakeAnsatz(terms=["term0"], reps=2, nq=1),
            "nq": 1,
            "psi_seed": psi_ref.copy(),
            "seed_energy": 0.25,
            "seed_delta_abs": 0.25,
            "seed_relative_abs": 0.25,
        },
    )
    monkeypatch.setattr(
        replay_mod,
        "vqe_minimize",
        lambda *args, **kwargs: type(
            "_Res",
            (),
            {
                "theta": np.zeros(2, dtype=float),
                "energy": 0.1,
                "success": True,
                "message": "ok",
                "nfev": 1,
                "nit": 1,
                "best_restart": 0,
            },
        )(),
    )

    result = replay_mod.run(cfg)
    written = json.loads(cfg.output_json.read_text(encoding="utf-8"))

    assert result["generator_family"]["resolved"] == "full_meta"
    assert result["generator_family"]["fallback_used"] is True
    assert result["replay_contract"]["generator_family"]["resolved"] == "full_meta"
    assert result["replay_contract"]["generator_family"]["resolution_source"] == "fallback_family_missing_labels"
    assert written["generator_family"]["resolved"] == "full_meta"
    assert written["replay_contract"]["generator_family"]["resolved"] == "full_meta"
