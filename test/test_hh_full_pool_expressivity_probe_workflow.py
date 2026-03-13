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

import pipelines.exact_bench.hh_full_pool_expressivity_probe_workflow as wf


class _DummyTerm:
    def __init__(self, label: str) -> None:
        self.label = label


class _DummyAnsatz:
    num_parameters = 3

    def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        return np.asarray(psi_ref, dtype=complex)


def test_base_run_cfg_propagates_n_ph_max() -> None:
    cfg = wf.FullPoolExpressivityProbeConfig(n_ph_max=3, g_ep=0.5)
    run_cfg = wf._base_run_cfg(cfg)
    assert run_cfg.n_ph_max == 3
    assert run_cfg.g_ep == pytest.approx(0.5)


def test_build_variants_default_matrix() -> None:
    cfg = wf.FullPoolExpressivityProbeConfig(random_orderings_x1=2)
    variants = wf.build_variants(cfg)
    assert [variant.label for variant in variants] == [
        "fullmeta_x1_canonical_spsa",
        "fullmeta_x1_random_1_spsa",
        "fullmeta_x1_random_2_spsa",
        "fullmeta_x1_canonical_powell",
        "fullmeta_x2_canonical_spsa",
    ]


def test_default_recipe_keeps_legacy_variant_labels() -> None:
    cfg = wf.FullPoolExpressivityProbeConfig()
    assert [variant.label for variant in wf.build_variants(cfg)[:2]] == [
        "fullmeta_x1_canonical_spsa",
        "fullmeta_x1_random_1_spsa",
    ]


def test_nondefault_recipe_uses_recipe_specific_variant_prefix() -> None:
    cfg = wf.FullPoolExpressivityProbeConfig(base_family="full_meta", extra_families=("paop_lf4_std",))
    labels = [variant.label for variant in wf.build_variants(cfg)]
    assert labels[0] == "full_meta_plus_paop_lf4_std_x1_canonical_spsa"
    assert labels[-1] == "full_meta_plus_paop_lf4_std_x2_canonical_spsa"


def test_build_variants_can_filter_to_named_subset() -> None:
    cfg = wf.FullPoolExpressivityProbeConfig(only_variants=("fullmeta_x1_random_2_spsa", "fullmeta_x2_canonical_spsa"))
    variants = wf.build_variants(cfg)
    assert [variant.label for variant in variants] == [
        "fullmeta_x1_random_2_spsa",
        "fullmeta_x2_canonical_spsa",
    ]


def test_ordered_terms_random_is_deterministic() -> None:
    context = {
        "pool_terms": [_DummyTerm(f"t{i}") for i in range(6)],
        "pool_labels": [f"t{i}" for i in range(6)],
    }
    variant = wf.ExpressivityVariant(
        label="v",
        reps=1,
        ordering_kind="random",
        ordering_seed=123,
        method="SPSA",
        restarts=2,
        maxiter=10,
    )
    terms_a, labels_a = wf._ordered_terms(context, variant)
    terms_b, labels_b = wf._ordered_terms(context, variant)
    assert labels_a == labels_b
    assert [term.label for term in terms_a] == [term.label for term in terms_b]
    assert labels_a != context["pool_labels"]


def test_build_probe_context_combines_recipe_and_records_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = wf.FullPoolExpressivityProbeConfig(base_family="full_meta", extra_families=("paop_lf4_std",))
    monkeypatch.setattr(wf.replay_mod, "_build_hh_hamiltonian", lambda _cfg: object())
    monkeypatch.setattr(
        wf.replay_mod,
        "_build_pool_recipe",
        lambda _cfg, **kwargs: (
            [_DummyTerm("base:a"), _DummyTerm("extra:b")],
            {
                "base_family": "full_meta",
                "extra_families": ["paop_lf4_std"],
                "raw_counts_by_family": {"full_meta": 1, "paop_lf4_std": 1},
                "combined_dedup_total": 2,
            },
        ),
    )
    monkeypatch.setattr(wf, "hubbard_holstein_reference_state", lambda **kwargs: np.array([1.0, 0.0], dtype=complex))
    monkeypatch.setattr(wf, "exact_ground_energy_sector_hh", lambda *args, **kwargs: 0.0)

    context = wf.build_probe_context(cfg)
    assert context["pool_recipe_label"] == "full_meta_plus_paop_lf4_std"
    assert context["pool_recipe_meta"]["raw_counts_by_family"]["paop_lf4_std"] == 1
    assert context["pool_size"] == 2


def test_duplicate_extra_families_rejected_cleanly() -> None:
    cfg = wf.FullPoolExpressivityProbeConfig(extra_families=("paop_lf4_std", "paop_lf4_std"))
    with pytest.raises(ValueError, match="Duplicate extra_family"):
        wf.build_variants(cfg)


def test_vlf_sq_recipe_is_accepted_and_labeled_deterministically() -> None:
    cfg = wf.FullPoolExpressivityProbeConfig(base_family="full_meta", extra_families=("vlf_sq",))
    assert wf._validated_recipe(cfg) == ("full_meta", ("vlf_sq",))
    labels = [variant.label for variant in wf.build_variants(cfg)]
    assert labels[0] == "full_meta_plus_vlf_sq_x1_canonical_spsa"
    assert labels[-1] == "full_meta_plus_vlf_sq_x2_canonical_spsa"


def test_build_payload_and_emit_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = wf.FullPoolExpressivityProbeConfig(
        output_json=tmp_path / "probe.json",
        output_csv=tmp_path / "probe.csv",
        tag="fullmeta_probe_test",
    )
    monkeypatch.setattr(
        wf,
        "build_probe_context",
        lambda _cfg: {
            "base_cfg": type("Cfg", (), {"sector_n_up": 1, "sector_n_dn": 1})(),
            "pool_size": 78,
            "pool_recipe_meta": {"combined_dedup_total": 78},
            "pool_order_hash": "abc123",
            "pool_recipe_label": "full_meta",
            "exact_energy": 0.11,
        },
    )
    rows_iter = iter(
        [
            {
                "variant": "fullmeta_x1_canonical_spsa",
                "delta_abs": 0.02,
                "progress_tail": [],
                "restart_summaries": [],
                "optimizer_memory": None,
            },
            {
                "variant": "fullmeta_x1_canonical_powell",
                "delta_abs": 0.0005,
                "progress_tail": [],
                "restart_summaries": [],
                "optimizer_memory": None,
            },
        ]
    )
    monkeypatch.setattr(
        wf,
        "build_variants",
        lambda _cfg: (
            wf.ExpressivityVariant("fullmeta_x1_canonical_spsa", 1, "canonical", None, "SPSA", 6, 4000),
            wf.ExpressivityVariant("fullmeta_x1_canonical_powell", 1, "canonical", None, "Powell", 2, 3000),
        ),
    )
    monkeypatch.setattr(wf, "run_variant", lambda *_args, **_kwargs: dict(next(rows_iter)))

    payload = wf.build_probe_payload(cfg)
    wf.emit_probe_files(payload, cfg)

    assert payload["summary"]["best_variant"]["variant"] == "fullmeta_x1_canonical_powell"
    assert payload["summary"]["best_overall_expressive_enough_at_1e3"] is True
    assert payload["summary"]["canonical_x1_expressive_enough_at_1e3"] is False

    json_payload = json.loads(cfg.output_json.read_text(encoding="utf-8"))
    assert json_payload["summary"]["best_overall_expressive_enough_at_1e3"] is True
    assert json_payload["summary"]["canonical_x1_expressive_enough_at_1e3"] is False

    with cfg.output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["variant"] for row in rows] == [
        "fullmeta_x1_canonical_spsa",
        "fullmeta_x1_canonical_powell",
    ]


def test_build_payload_records_variant_failure_without_aborting(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = wf.FullPoolExpressivityProbeConfig(tag="fail_probe")
    monkeypatch.setattr(
        wf,
        "build_probe_context",
        lambda _cfg: {
            "base_cfg": type("Cfg", (), {"sector_n_up": 1, "sector_n_dn": 1})(),
            "pool_size": 4,
            "pool_recipe_meta": {},
            "pool_order_hash": "abcd",
            "pool_recipe_label": "full_meta",
            "exact_energy": 0.11,
            "pool_terms": [_DummyTerm("a"), _DummyTerm("b")],
            "pool_labels": ["a", "b"],
        },
    )
    monkeypatch.setattr(
        wf,
        "build_variants",
        lambda _cfg: (
            wf.ExpressivityVariant("fullmeta_x1_canonical_spsa", 1, "canonical", None, "SPSA", 6, 4000),
            wf.ExpressivityVariant("fullmeta_x1_canonical_powell", 1, "canonical", None, "Powell", 2, 3000),
        ),
    )

    def _fake_run_variant(_cfg, _context, variant):
        if variant.method == "Powell":
            raise RuntimeError("powell unavailable")
        return {
            "variant": variant.label,
            "delta_abs": 0.02,
            "progress_tail": [],
            "restart_summaries": [],
            "optimizer_memory": None,
            "success": True,
        }

    monkeypatch.setattr(wf, "run_variant", _fake_run_variant)
    payload = wf.build_probe_payload(cfg)
    failure_row = next(row for row in payload["variant_rows"] if row["variant"] == "fullmeta_x1_canonical_powell")
    assert failure_row["success"] is False
    assert failure_row["delta_abs"] == float("inf")
    assert "powell unavailable" in failure_row["message"]
