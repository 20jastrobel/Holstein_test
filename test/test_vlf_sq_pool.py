from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_vqe_from_adapt_family import PoolTermwiseAnsatz
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.operator_pools.polaron_paop import _to_signature, make_pool
from src.quantum.operator_pools.vlf_sq import build_vlf_sq_pool
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, hamiltonian_matrix


def _build(name: str, *, n_ph_max: int) -> tuple[list[tuple[str, object]], dict[str, object]]:
    return build_vlf_sq_pool(
        name,
        num_sites=2,
        num_particles=(1, 1),
        n_ph_max=int(n_ph_max),
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        shell_radius=1,
        prune_eps=0.0,
        normalization="none",
    )


def test_vlf_only_shell_tying_count_and_labels() -> None:
    pool, meta = _build("vlf_only", n_ph_max=2)
    labels = [label for label, _poly in pool]
    assert meta["shells"] == [0, 1]
    assert meta["parameter_count"] == 2
    assert labels == [
        "vlf_only:vlf_shell(r=0)",
        "vlf_only:vlf_shell(r=1)",
    ]


def test_sq_only_and_vlf_sq_variant_selection_counts() -> None:
    sq_only, sq_meta = _build("sq_only", n_ph_max=2)
    vlf_sq, vlf_sq_meta = _build("vlf_sq", n_ph_max=2)
    dens_only, dens_meta = _build("sq_dens_only", n_ph_max=2)
    assert [label for label, _poly in sq_only] == ["sq_only:sq_global"]
    assert [label for label, _poly in dens_only] == ["sq_dens_only:dens_sq_global"]
    assert [label for label, _poly in vlf_sq] == [
        "vlf_sq:vlf_shell(r=0)",
        "vlf_sq:vlf_shell(r=1)",
        "vlf_sq:sq_global",
    ]
    assert sq_meta["parameter_count"] == 1
    assert dens_meta["density_conditioned_sq"] is True
    assert vlf_sq_meta["parameter_count"] == 3


def test_default_vlf_shell_enumeration_uses_all_shells() -> None:
    pool, meta = build_vlf_sq_pool(
        "vlf_only",
        num_sites=3,
        num_particles=(2, 1),
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        prune_eps=0.0,
        normalization="none",
    )
    assert meta["shells"] == [0, 1, 2]
    assert [label for label, _poly in pool] == [
        "vlf_only:vlf_shell(r=0)",
        "vlf_only:vlf_shell(r=1)",
        "vlf_only:vlf_shell(r=2)",
    ]


def test_sq_only_matches_sum_of_repo_paop_sq_generators() -> None:
    sq_only, _meta = _build("sq_only", n_ph_max=2)
    sq_only_sig = _to_signature(sq_only[0][1])
    paop_sq_std = make_pool(
        "paop_sq_std",
        num_sites=2,
        num_particles=(1, 1),
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
    )
    sq_sum = None
    for label, poly in paop_sq_std:
        if ":paop_sq(site=" not in label:
            continue
        sq_sum = poly if sq_sum is None else (sq_sum + poly)
    assert sq_sum is not None
    assert sq_only_sig == _to_signature(sq_sum)


def test_vlf_sq_generators_are_hermitian_and_effectively_real() -> None:
    for family in ("vlf_only", "sq_only", "vlf_sq", "sq_dens_only", "vlf_sq_dens"):
        pool, _meta = _build(family, n_ph_max=2)
        for _label, poly in pool:
            for term in poly.return_polynomial():
                coeff = complex(term.p_coeff)
                assert abs(coeff.imag) <= 1e-10
            mat = hamiltonian_matrix(poly)
            assert np.max(np.abs(mat - mat.conj().T)) <= 1e-10


def test_zero_parameter_state_prep_is_identity_for_vlf_sq() -> None:
    pool, _meta = _build("vlf_sq", n_ph_max=2)
    ansatz = PoolTermwiseAnsatz(
        terms=[AnsatzTerm(label=label, polynomial=poly) for label, poly in pool],
        reps=1,
        nq=8,
    )
    psi_ref = np.asarray(
        hubbard_holstein_reference_state(
            dims=2,
            num_particles=(1, 1),
            n_ph_max=2,
            boson_encoding="binary",
            indexing="blocked",
        ),
        dtype=complex,
    ).reshape(-1)
    psi = ansatz.prepare_state(np.zeros(ansatz.num_parameters, dtype=float), psi_ref)
    assert np.allclose(psi, psi_ref)


def test_sq_only_rejects_too_small_cutoff_and_vlf_sq_reduces_to_vlf_only() -> None:
    with pytest.raises(ValueError, match="sq_only produced no surviving squeeze generators"):
        _build("sq_only", n_ph_max=1)
    vlf_only, _ = _build("vlf_only", n_ph_max=1)
    vlf_sq, _ = _build("vlf_sq", n_ph_max=1)
    assert {_to_signature(poly) for _label, poly in vlf_only} == {_to_signature(poly) for _label, poly in vlf_sq}
