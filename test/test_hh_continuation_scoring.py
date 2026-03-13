from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_generators import build_generator_metadata
from pipelines.hardcoded.hh_continuation_scoring import (
    FullScoreConfig,
    MeasurementCacheAudit,
    Phase2CurvatureOracle,
    Phase2NoveltyOracle,
    Phase1CompileCostOracle,
    SimpleScoreConfig,
    build_candidate_features,
    build_full_candidate_features,
    family_repeat_cost_from_history,
    full_v2_score,
    lifetime_weight_components,
    remaining_evaluations_proxy,
    shortlist_records,
    trust_region_drop,
)
from pipelines.hardcoded.hh_continuation_symmetry import build_symmetry_spec
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm


def _term(label: str) -> object:
    return type(
        "_DummyAnsatzTerm",
        (),
        {"label": str(label), "polynomial": PauliPolynomial("JW", [PauliTerm(1, ps=str(label), pc=1.0)])},
    )()


def _feat(
    *,
    gradient_signed: float = 0.4,
    metric_proxy: float = 0.5,
    sigma_hat: float = 0.0,
    refit_window_indices: list[int] | None = None,
    family_repeat_cost: float = 0.0,
    stage_gate_open: bool = True,
    cfg: SimpleScoreConfig | None = None,
) -> object:
    cfg_use = cfg or SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.0, z_alpha=0.0)
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    return build_candidate_features(
        stage_name="core",
        candidate_label="cand",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=float(gradient_signed),
        metric_proxy=float(metric_proxy),
        sigma_hat=float(sigma_hat),
        refit_window_indices=list(refit_window_indices or []),
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=len(refit_window_indices or [])),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=bool(stage_gate_open),
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg_use,
        family_repeat_cost=float(family_repeat_cost),
    )


def test_simple_v1_prefers_higher_gradient_with_equal_costs() -> None:
    cfg = SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.0, z_alpha=0.0)
    feat_a = _feat(gradient_signed=0.4, metric_proxy=0.5, cfg=cfg)
    feat_b = _feat(gradient_signed=0.2, metric_proxy=0.5, cfg=cfg)
    assert float(feat_a.simple_score or 0.0) > float(feat_b.simple_score or 0.0)


def test_stage_gate_blocks_score() -> None:
    feat = _feat(stage_gate_open=False)
    assert feat.simple_score == float("-inf")


def test_simple_v1_uses_g_lcb_not_g_abs() -> None:
    cfg = SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.0, z_alpha=10.0)
    feat = _feat(gradient_signed=0.4, metric_proxy=0.5, sigma_hat=0.03, cfg=cfg)
    assert float(feat.g_lcb) == pytest.approx(0.1)
    assert float(feat.simple_score or 0.0) == pytest.approx(0.5 * 0.1 * 0.1 / 0.5)


def test_family_repeat_cost_lowers_screen_score() -> None:
    cfg = SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.2, z_alpha=0.0)
    feat_a = _feat(gradient_signed=0.4, metric_proxy=0.5, family_repeat_cost=0.0, cfg=cfg)
    feat_b = _feat(gradient_signed=0.4, metric_proxy=0.5, family_repeat_cost=2.0, cfg=cfg)
    assert float(feat_a.simple_score or 0.0) > float(feat_b.simple_score or 0.0)


def test_measurement_cache_reuse_accounting() -> None:
    cache = MeasurementCacheAudit(nominal_shots_per_group=10)
    first = cache.estimate(["a", "b"])
    assert first.groups_new == 2
    cache.commit(["a", "b"])
    second = cache.estimate(["a", "b", "c"])
    assert second.groups_reused == 2
    assert second.groups_new == 1
    summary = cache.summary()
    assert str(summary["plan_version"]) == "phase1_grouped_label_reuse"


def test_trust_region_drop_matches_newton_branch() -> None:
    got = trust_region_drop(0.4, 2.0, 1.0, 1.0)
    assert got == pytest.approx(0.04)


def test_full_v2_uses_reduced_fields() -> None:
    cfg = FullScoreConfig(z_alpha=0.0, rho=1.0, gamma_N=1.0, wD=0.0, wG=0.0, wC=0.0, wc=0.0)
    feat = _feat(gradient_signed=0.5, metric_proxy=0.5, refit_window_indices=[0])
    feat = type(feat)(
        **{
            **feat.__dict__,
            "novelty": 0.5,
            "h_eff": 2.0,
            "F_red": 1.0,
            "h_raw": 2.0,
            "ridge_used": cfg.lambda_H,
        }
    )
    score, fallback = full_v2_score(feat, cfg)
    assert score == pytest.approx(0.03125)
    assert fallback == "append_exact_reduced_path"


def test_full_v2_zeroes_metric_collapse() -> None:
    cfg = FullScoreConfig(z_alpha=0.0, rho=1.0, gamma_N=1.0, wD=0.0, wG=0.0, wC=0.0, wc=0.0)
    feat = _feat(gradient_signed=0.5, metric_proxy=0.5, refit_window_indices=[0])
    feat = type(feat)(
        **{
            **feat.__dict__,
            "novelty": 0.0,
            "h_eff": 0.0,
            "F_red": cfg.metric_floor,
            "h_raw": 0.0,
            "ridge_used": cfg.lambda_H,
            "curvature_mode": "append_exact_metric_collapse_v1",
        }
    )
    score, fallback = full_v2_score(feat, cfg)
    assert score == 0.0
    assert fallback == "reduced_metric_collapse"


def test_full_v2_ignores_motif_bonus_in_active_score() -> None:
    cfg = FullScoreConfig(z_alpha=0.0, rho=1.0, gamma_N=1.0, wD=0.0, wG=0.0, wC=0.0, wc=0.0)
    feat = _feat(gradient_signed=0.5, metric_proxy=0.5, refit_window_indices=[0])
    feat_base = type(feat)(
        **{
            **feat.__dict__,
            "novelty": 1.0,
            "h_eff": 1.0,
            "F_red": 1.0,
            "h_raw": 1.0,
            "ridge_used": cfg.lambda_H,
            "motif_bonus": 0.0,
        }
    )
    feat_bonus = type(feat_base)(**{**feat_base.__dict__, "motif_bonus": 10.0})
    score_a, _ = full_v2_score(feat_base, cfg)
    score_b, _ = full_v2_score(feat_bonus, cfg)
    assert score_a == pytest.approx(score_b)


def test_build_full_candidate_features_emits_reduced_path_fields() -> None:
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    h_poly = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])
    h_compiled = compile_polynomial_action(h_poly)
    selected_ops = [_term("x")]
    theta = np.asarray([0.2], dtype=float)
    executor = CompiledAnsatzExecutor(selected_ops, pauli_action_cache={})
    psi_state = executor.prepare_state(theta, psi_ref)
    hpsi_state = apply_compiled_polynomial(psi_state, h_compiled)

    base = _feat(
        gradient_signed=0.3,
        metric_proxy=0.3,
        refit_window_indices=[0],
        cfg=SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.0),
    )
    novelty_oracle = Phase2NoveltyOracle()
    scaffold_context = novelty_oracle.prepare_scaffold_context(
        selected_ops=selected_ops,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        hpsi_state=hpsi_state,
        refit_window_indices=[0],
        pauli_action_cache={},
    )
    feat = build_full_candidate_features(
        base_feature=base,
        candidate_term=_term("y"),
        cfg=FullScoreConfig(shortlist_size=2),
        novelty_oracle=novelty_oracle,
        curvature_oracle=Phase2CurvatureOracle(),
        scaffold_context=scaffold_context,
        h_compiled=h_compiled,
        compiled_cache={},
        pauli_action_cache={},
        optimizer_memory=None,
    )
    assert 0.0 <= float(feat.novelty or 0.0) <= 1.0
    assert feat.refit_window_indices == [0]
    assert feat.F_raw is not None and feat.F_raw >= 0.0
    assert feat.F_red is not None and feat.F_red > 0.0
    assert feat.Q_window is not None
    assert feat.H_window_hessian is not None
    assert feat.h_eff is not None
    assert feat.full_v2_score is not None


def test_shortlist_only_expensive_scoring_calls_oracles_for_shortlist() -> None:
    class _CountingNovelty(Phase2NoveltyOracle):
        def __init__(self) -> None:
            self.calls = 0

        def estimate(self, *args, **kwargs):
            self.calls += 1
            return super().estimate(*args, **kwargs)

    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    h_poly = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])
    h_compiled = compile_polynomial_action(h_poly)
    hpsi_state = apply_compiled_polynomial(psi_ref, h_compiled)
    novelty = _CountingNovelty()
    scaffold_context = novelty.prepare_scaffold_context(
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi_ref,
        psi_state=psi_ref,
        h_compiled=h_compiled,
        hpsi_state=hpsi_state,
        refit_window_indices=[],
        pauli_action_cache={},
    )

    cheap_records = []
    for idx, grad in enumerate([0.9, 0.8, 0.3, 0.2]):
        feat = _feat(
            gradient_signed=float(grad),
            metric_proxy=1.0,
            refit_window_indices=[],
            cfg=SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0),
        )
        cheap_records.append(
            {
                "feature": feat,
                "simple_score": float(feat.simple_score or 0.0),
                "candidate_pool_index": idx,
                "position_id": 0,
                "candidate_term": _term("x"),
            }
        )
    shortlisted = shortlist_records(cheap_records, cfg=FullScoreConfig(shortlist_fraction=0.5, shortlist_size=2))
    for rec in shortlisted:
        build_full_candidate_features(
            base_feature=rec["feature"],
            candidate_term=rec["candidate_term"],
            cfg=FullScoreConfig(shortlist_fraction=0.5, shortlist_size=2),
            novelty_oracle=novelty,
            curvature_oracle=Phase2CurvatureOracle(),
            scaffold_context=scaffold_context,
            h_compiled=h_compiled,
            compiled_cache={},
            pauli_action_cache={},
            optimizer_memory=None,
        )
    assert len(shortlisted) == 2
    assert novelty.calls == 2


def test_remaining_evaluations_proxy_uses_remaining_depth_mode() -> None:
    got = remaining_evaluations_proxy(current_depth=2, max_depth=6, mode="remaining_depth")
    assert got == pytest.approx(5.0)


def test_lifetime_weight_components_are_zero_when_mode_off() -> None:
    cfg = FullScoreConfig(lifetime_cost_mode="off")
    feat = _feat(
        gradient_signed=0.5,
        metric_proxy=0.5,
        refit_window_indices=[0],
        cfg=SimpleScoreConfig(),
    )
    feat = type(feat)(
        **{
            **feat.__dict__,
            "remaining_evaluations_proxy": 5.0,
            "lifetime_cost_mode": "off",
        }
    )
    comps = lifetime_weight_components(feat, cfg)
    assert comps["remaining_evaluations_proxy"] == pytest.approx(5.0)
    assert comps["total"] == pytest.approx(0.0)


def test_family_repeat_cost_from_history_uses_consecutive_streak() -> None:
    history = [
        {"candidate_family": "a"},
        {"candidate_family": "b"},
        {"candidate_family": "b"},
    ]
    assert family_repeat_cost_from_history(history_rows=history, candidate_family="a") == pytest.approx(0.0)
    assert family_repeat_cost_from_history(history_rows=history, candidate_family="b") == pytest.approx(2.0)


def test_build_candidate_features_carries_generator_and_symmetry_metadata() -> None:
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(6, ps="eyeexy", pc=1.0),
            PauliTerm(6, ps="eyeeyx", pc=-1.0),
        ],
    )
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    meta = build_generator_metadata(
        label="macro_candidate",
        polynomial=poly,
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="macro_candidate",
        candidate_family="paop_lf_std",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.4,
        metric_proxy=0.4,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=2, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["macro"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
        generator_metadata=meta.__dict__,
        symmetry_spec=sym.__dict__,
        symmetry_mode="phase3_shared_spec",
        symmetry_mitigation_mode="verify_only",
        current_depth=0,
        max_depth=3,
        lifetime_cost_mode="phase3_v1",
        remaining_evaluations_proxy_mode="remaining_depth",
    )
    assert feat.generator_id == meta.generator_id
    assert feat.template_id == meta.template_id
    assert feat.is_macro_generator is True
    assert feat.symmetry_mode == "phase3_shared_spec"
    assert feat.symmetry_mitigation_mode == "verify_only"
    assert feat.remaining_evaluations_proxy == pytest.approx(4.0)
