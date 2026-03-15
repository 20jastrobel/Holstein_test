from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.operator_pools.polaron_paop import _to_signature, make_phonon_motifs, make_pool


def _build_pool(name: str, *, n_ph_max: int) -> list[tuple[str, object]]:
    return make_pool(
        name,
        num_sites=2,
        num_particles=(1, 1),
        n_ph_max=int(n_ph_max),
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
    )


def _signature_set(pool: list[tuple[str, object]]) -> set[tuple[tuple[str, float], ...]]:
    return {_to_signature(poly) for _label, poly in pool}


def test_paop_lf4_std_strictly_extends_lf2_and_lf3_at_l2_nph2() -> None:
    lf2 = _build_pool("paop_lf2_std", n_ph_max=2)
    lf3 = _build_pool("paop_lf3_std", n_ph_max=2)
    lf4 = _build_pool("paop_lf4_std", n_ph_max=2)

    sig2 = _signature_set(lf2)
    sig3 = _signature_set(lf3)
    sig4 = _signature_set(lf4)

    assert sig2 < sig3
    assert sig3 < sig4
    assert any("paop_curdrag3(" in label for label, _poly in lf3)
    assert any("paop_hop4(" in label for label, _poly in lf4)


def test_paop_sq_std_produces_squeeze_terms_at_nph2() -> None:
    sq_std = _build_pool("paop_sq_std", n_ph_max=2)
    labels = [label for label, _poly in sq_std]
    assert any(":paop_sq(site=" in label for label in labels)
    assert any(":paop_dens_sq(site=" in label for label in labels)


def test_paop_sq_std_reduces_to_non_squeeze_baseline_at_nph1() -> None:
    sq_std = _build_pool("paop_sq_std", n_ph_max=1)
    lf_std = _build_pool("paop_lf_std", n_ph_max=1)
    labels = [label for label, _poly in sq_std]
    assert not any("paop_sq(site=" in label for label in labels)
    assert not any("paop_dens_sq(site=" in label for label in labels)
    assert _signature_set(sq_std) == _signature_set(lf_std)


def test_paop_lf_full_retains_doublon_translation_x_and_p_labels() -> None:
    lf_full = _build_pool("paop_lf_full", n_ph_max=2)
    labels = [label for label, _poly in lf_full]
    assert any(":paop_dbl_p(site=" in label for label in labels)
    assert any(":paop_dbl_x(site=" in label for label in labels)


def test_new_structural_paop_families_emit_expected_labels() -> None:
    bond_disp = _build_pool("paop_bond_disp_std", n_ph_max=2)
    hop_sq = _build_pool("paop_hop_sq_std", n_ph_max=2)
    pair_sq = _build_pool("paop_pair_sq_std", n_ph_max=2)

    assert any(":paop_bond_disp(" in label for label, _poly in bond_disp)
    assert any(":paop_hop_sq(" in label for label, _poly in hop_sq)
    assert any(":paop_pair_sq(" in label for label, _poly in pair_sq)


def test_new_paop_family_coefficients_remain_effectively_real() -> None:
    for family in (
        "paop_lf4_std",
        "paop_sq_full",
        "paop_bond_disp_std",
        "paop_hop_sq_std",
        "paop_pair_sq_std",
    ):
        for _label, poly in _build_pool(family, n_ph_max=2):
            for term in poly.return_polynomial():
                coeff = complex(term.p_coeff)
                assert abs(coeff.imag) <= 1e-10


def test_make_phonon_motifs_lf_std_has_expected_order_metadata_and_boson_only_support() -> None:
    motifs = make_phonon_motifs(
        "paop_lf_std",
        num_sites=2,
        n_ph_max=1,
        boson_encoding="binary",
        boundary="open",
        prune_eps=0.0,
        normalization="none",
    )
    assert [motif.label for motif in motifs] == ["p(site=0)", "p(site=1)", "delta_p(0,1)"]
    assert motifs[0].sites == (0,)
    assert motifs[0].bonds == ()
    assert motifs[0].uses_sq is False
    assert motifs[-1].sites == (0, 1)
    assert motifs[-1].bonds == ((0, 1),)

    fermion_identity = "e" * 4
    for motif in motifs:
        for term in motif.poly.return_polynomial():
            assert str(term.pw2strng())[-4:] == fermion_identity


def test_make_phonon_motifs_lf2_std_adds_delta_p2_without_squeeze_flag() -> None:
    motifs = make_phonon_motifs(
        "paop_lf2_std",
        num_sites=2,
        n_ph_max=2,
        boson_encoding="binary",
        boundary="open",
        prune_eps=0.0,
        normalization="none",
    )
    assert any(motif.label.startswith("delta_p2(") for motif in motifs)
    assert all(motif.uses_sq is False for motif in motifs)


def test_make_phonon_motifs_bond_disp_adds_bond_sum_terms_with_real_coefficients() -> None:
    motifs = make_phonon_motifs(
        "paop_bond_disp_std",
        num_sites=2,
        n_ph_max=2,
        boson_encoding="binary",
        boundary="open",
        prune_eps=0.0,
        normalization="none",
    )
    assert any(motif.label.startswith("bond_p_sum(") for motif in motifs)
    for motif in motifs:
        for term in motif.poly.return_polynomial():
            assert abs(complex(term.p_coeff).imag) <= 1e-10
