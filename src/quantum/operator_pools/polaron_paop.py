"""Polaron-adapted operator pool for Hubbard-Holstein (HH) ADAPT-VQE.

This module exposes a lightweight, composable PAOP pool builder using the
existing Pauli-layer operators from the repository math stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from src.quantum.hubbard_latex_python_pairs import (
    bravais_nearest_neighbor_edges,
    boson_operator,
    boson_displacement_operator,
    boson_qubits_per_site,
    jw_number_operator,
    mode_index,
    phonon_qubit_indices_for_site,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial, fermion_minus_operator, fermion_plus_operator
from src.quantum.qubitization_module import PauliTerm


@dataclass(frozen=True)
class PhononMotifSpec:
    label: str
    family: str
    poly: PauliPolynomial
    sites: tuple[int, ...]
    bonds: tuple[tuple[int, int], ...]
    uses_sq: bool


def _to_signature(poly: PauliPolynomial, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    items: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"PAOP generator has non-negligible imaginary term: {coeff}")
        items.append((str(term.pw2strng()), float(round(coeff.real, 12))))
    items.sort()
    return tuple(items)


def _clean_poly(poly: PauliPolynomial, prune_eps: float) -> PauliPolynomial:
    """Drop tiny coefficients and enforce purely-real Pauli coefficients."""
    return _prune_poly(poly, prune_eps, enforce_real=True)


def _prune_poly(poly: PauliPolynomial, prune_eps: float, *, enforce_real: bool) -> PauliPolynomial:
    """Drop tiny coefficients; optionally enforce purely-real Pauli coefficients."""
    terms = poly.return_polynomial()
    if not terms:
        return PauliPolynomial("JW")
    nq = int(terms[0].nqubit())
    cleaned = PauliPolynomial("JW")
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(prune_eps):
            continue
        if enforce_real and abs(coeff.imag) > 1e-10:
            raise ValueError(f"PAOP generator has non-negligible imaginary coefficient: {coeff}")
        coeff_out: complex | float
        if enforce_real:
            coeff_out = float(coeff.real)
        else:
            coeff_out = complex(coeff)
        cleaned.add_term(PauliTerm(nq, ps=str(term.pw2strng()), pc=coeff_out))
    cleaned._reduce()
    return cleaned


def _normalize_poly(poly: PauliPolynomial, mode: str) -> PauliPolynomial:
    mode_key = str(mode).strip().lower()
    if mode_key == "none":
        return poly
    terms = poly.return_polynomial()
    if not terms:
        return poly

    if mode_key == "maxcoeff":
        max_coeff = max(abs(complex(term.p_coeff)) for term in terms)
        if max_coeff <= 0.0:
            return poly
        return (1.0 / max_coeff) * poly

    if mode_key == "fro":
        norm = math.sqrt(sum(abs(complex(term.p_coeff)) ** 2 for term in terms))
        if norm <= 0.0:
            return poly
        return (1.0 / norm) * poly

    raise ValueError(f"Unknown PAOP normalization '{mode_key}'. Use none|fro|maxcoeff.")


def _append_operator(
    pool: list[tuple[str, PauliPolynomial]],
    label: str,
    poly: PauliPolynomial,
    split_paulis: bool,
    prune_eps: float,
) -> None:
    poly = _clean_poly(poly, prune_eps)
    if not poly.return_polynomial():
        return
    if not split_paulis:
        pool.append((label, poly))
        return

    for term_idx, term in enumerate(poly.return_polynomial()):
        coeff = complex(term.p_coeff)
        if abs(coeff) <= prune_eps:
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"PAOP generator has non-negligible imaginary coefficient: {coeff}")
        sub_label = f"{label}[{term_idx}]_{term.pw2strng()}"
        single = PauliPolynomial("JW", [PauliTerm(int(term.nqubit()), ps=str(term.pw2strng()), pc=float(coeff.real))])
        pool.append((sub_label, single))


def _mul_clean(
    left: PauliPolynomial,
    right: PauliPolynomial,
    prune_eps: float,
    *,
    enforce_real: bool = True,
) -> PauliPolynomial:
    """(AB)_clean := clean(A * B) after each nontrivial multiplication."""
    return _prune_poly(left * right, float(prune_eps), enforce_real=bool(enforce_real))


def _distance_1d(i: int, j: int, n_sites: int, periodic: bool) -> int:
    dist = abs(int(i) - int(j))
    if periodic and n_sites > 0:
        period = int(n_sites)
        dist = min(dist, period - dist)
    return int(dist)


def _word_from_qubit_letters(nq: int, letters: dict[int, str]) -> str:
    word = ["e"] * int(nq)
    for qubit, letter in letters.items():
        q = int(qubit)
        if q < 0 or q >= int(nq):
            raise ValueError(f"Qubit index {q} out of range for nq={nq}")
        idx = int(nq) - 1 - q
        word[idx] = str(letter)
    return "".join(word)


def jw_current_hop(nq: int, p: int, q: int) -> PauliPolynomial:
    r"""Build Hermitian odd hopping channel in JW form.

    J_{pq} = i (c^†_p c_q - c^†_q c_p)
          = 1/2 * (X_hi Z_{lo+1..hi-1} Y_lo - Y_hi Z_{lo+1..hi-1} X_lo)
    """
    p_i = int(p)
    q_i = int(q)
    nq_i = int(nq)
    if p_i == q_i:
        return PauliPolynomial("JW")
    if p_i < 0 or p_i >= nq_i or q_i < 0 or q_i >= nq_i:
        raise ValueError(f"jw_current_hop indices out of range: p={p_i}, q={q_i}, nq={nq_i}")

    lo = min(p_i, q_i)
    hi = max(p_i, q_i)
    z_letters = {k: "z" for k in range(lo + 1, hi)}

    xy = dict(z_letters)
    xy[hi] = "x"
    xy[lo] = "y"

    yx = dict(z_letters)
    yx[hi] = "y"
    yx[lo] = "x"

    out = PauliPolynomial("JW")
    out.add_term(PauliTerm(nq_i, ps=_word_from_qubit_letters(nq_i, xy), pc=0.5))
    out.add_term(PauliTerm(nq_i, ps=_word_from_qubit_letters(nq_i, yx), pc=-0.5))
    out._reduce()
    if p_i > q_i:
        return (-1.0) * out
    return out


def _drop_terms_with_identity_on_qubits(poly: PauliPolynomial, qubits: tuple[int, ...]) -> PauliPolynomial:
    terms = poly.return_polynomial()
    if not terms:
        return poly
    nq = int(terms[0].nqubit())
    keep = PauliPolynomial("JW")
    qidx = [int(q) for q in qubits]
    for term in terms:
        word = str(term.pw2strng())
        if all(word[nq - 1 - q] == "e" for q in qidx):
            continue
        keep.add_term(PauliTerm(nq, ps=word, pc=complex(term.p_coeff)))
    keep._reduce()
    return keep


def make_phonon_motifs(
    family: str,
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    boundary: str,
    prune_eps: float = 0.0,
    normalization: str = "none",
) -> list[PhononMotifSpec]:
    family_key = str(family).strip().lower()
    if family_key not in {"paop_lf_std", "paop_lf2_std", "paop_bond_disp_std"}:
        raise ValueError(
            "Phonon motif family must be one of paop_lf_std, paop_lf2_std, paop_bond_disp_std."
        )

    n_sites = int(num_sites)
    if n_sites <= 0:
        return []

    n_ph_max_i = int(n_ph_max)
    boson_encoding_i = str(boson_encoding)
    periodic = str(boundary).strip().lower() == "periodic"
    qpb = boson_qubits_per_site(n_ph_max_i, boson_encoding_i)
    nq = 2 * n_sites + n_sites * qpb
    phonon_qubits = tuple(range(2 * n_sites, nq))
    repr_mode = "JW"

    phonon_qubit_cache: dict[int, tuple[int, ...]] = {}
    b_cache: dict[int, PauliPolynomial] = {}
    bdag_cache: dict[int, PauliPolynomial] = {}
    p_cache: dict[int, PauliPolynomial] = {}
    delta_p_cache: dict[tuple[int, int], PauliPolynomial] = {}
    delta_p_power_cache: dict[tuple[int, int, int], PauliPolynomial] = {}
    bond_p_sum_cache: dict[tuple[int, int], PauliPolynomial] = {}

    def local_qubits(site: int) -> tuple[int, ...]:
        key = int(site)
        if key not in phonon_qubit_cache:
            phonon_qubit_cache[key] = tuple(
                phonon_qubit_indices_for_site(
                    key,
                    n_sites=n_sites,
                    qpb=qpb,
                    fermion_qubits=2 * n_sites,
                )
            )
        return phonon_qubit_cache[key]

    def b_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in b_cache:
            b_cache[key] = boson_operator(
                repr_mode,
                nq,
                local_qubits(key),
                which="b",
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
        return b_cache[key]

    def bdag_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in bdag_cache:
            bdag_cache[key] = boson_operator(
                repr_mode,
                nq,
                local_qubits(key),
                which="bdag",
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
        return bdag_cache[key]

    def p_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in p_cache:
            p_cache[key] = _clean_poly((1j * bdag_i(key)) + ((-1j) * b_i(key)), prune_eps)
        return p_cache[key]

    def delta_p_ij(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in delta_p_cache:
            delta_p_cache[key] = _clean_poly(p_i(key[0]) + ((-1.0) * p_i(key[1])), prune_eps)
        return delta_p_cache[key]

    def delta_p_power(i_site: int, j_site: int, exponent: int) -> PauliPolynomial:
        power = int(exponent)
        if power < 1:
            raise ValueError("delta_p_power exponent must be >= 1")
        key = (int(i_site), int(j_site), power)
        if key in delta_p_power_cache:
            return delta_p_power_cache[key]
        base = delta_p_ij(i_site, j_site)
        if power == 1:
            delta_p_power_cache[key] = base
            return base
        acc = base
        for _ in range(1, power):
            acc = _mul_clean(acc, base, prune_eps)
        delta_p_power_cache[key] = acc
        return acc

    def bond_p_sum(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in bond_p_sum_cache:
            bond_p_sum_cache[key] = _clean_poly(p_i(key[0]) + p_i(key[1]), prune_eps)
        return bond_p_sum_cache[key]

    motifs: list[PhononMotifSpec] = []

    def _append_motif(
        *,
        label: str,
        poly: PauliPolynomial,
        sites: tuple[int, ...],
        bonds: tuple[tuple[int, int], ...] = (),
        uses_sq: bool = False,
        drop_phonon_identity: bool = False,
    ) -> None:
        poly_out = poly
        if drop_phonon_identity:
            poly_out = _drop_terms_with_identity_on_qubits(poly_out, phonon_qubits)
        poly_out = _clean_poly(poly_out, prune_eps)
        poly_out = _normalize_poly(poly_out, normalization)
        poly_out = _clean_poly(poly_out, prune_eps)
        if not poly_out.return_polynomial():
            return
        bonds_canon = tuple(sorted({tuple(sorted((int(i), int(j)))) for i, j in bonds}))
        motifs.append(
            PhononMotifSpec(
                label=str(label),
                family=family_key,
                poly=poly_out,
                sites=tuple(sorted({int(site) for site in sites})),
                bonds=bonds_canon,
                uses_sq=bool(uses_sq),
            )
        )

    for site in range(n_sites):
        _append_motif(
            label=f"p(site={site})",
            poly=p_i(site),
            sites=(site,),
        )

    edges = bravais_nearest_neighbor_edges(n_sites, pbc=periodic)
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        _append_motif(
            label=f"delta_p({i},{j})",
            poly=delta_p_ij(i, j),
            sites=(i, j),
            bonds=((i, j),),
        )

    if family_key == "paop_lf2_std":
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_motif(
                label=f"delta_p2({i},{j})",
                poly=delta_p_power(i, j, 2),
                sites=(i, j),
                bonds=((i, j),),
                drop_phonon_identity=True,
            )

    if family_key == "paop_bond_disp_std":
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_motif(
                label=f"bond_p_sum({i},{j})",
                poly=bond_p_sum(i, j),
                sites=(i, j),
                bonds=((i, j),),
            )

    return motifs


def _make_paop_core(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    num_particles: tuple[int, int],
    include_disp: bool,
    include_doublon: bool,
    include_hopdrag: bool,
    include_curdrag: bool,
    include_hop2: bool,
    include_curdrag3: bool,
    include_hop4: bool,
    include_bond_disp: bool,
    include_hop_sq: bool,
    include_pair_sq: bool,
    drop_hop2_phonon_identity: bool,
    include_extended_cloud: bool,
    cloud_radius: int,
    include_cloud_x: bool,
    include_doublon_translation_p: bool,
    include_doublon_translation_x: bool,
    include_sq: bool,
    include_dens_sq: bool,
    include_cloud_sq: bool,
    include_doublon_sq: bool,
    split_paulis: bool,
    prune_eps: float,
    normalization: str,
    pool_name: str,
) -> list[tuple[str, PauliPolynomial]]:
    n_sites = int(num_sites)
    if n_sites <= 0:
        return []

    n_ph_max_i = int(n_ph_max)
    boson_encoding_i = str(boson_encoding)
    ordering_i = str(ordering)
    boundary_i = str(boundary).strip().lower()
    periodic = boundary_i == "periodic"
    total_electrons = int(num_particles[0]) + int(num_particles[1]) if num_particles else 0
    nbar = (float(total_electrons) / float(n_sites)) if total_electrons > 0 else 1.0
    if nbar <= 0.0:
        nbar = 1.0

    nq = 2 * n_sites + n_sites * boson_qubits_per_site(n_ph_max_i, boson_encoding_i)
    repr_mode = "JW"
    id_label = "e" * nq
    id_poly = PauliPolynomial(repr_mode, [PauliTerm(nq, ps=id_label, pc=1.0)])

    number_cache: dict[int, PauliPolynomial] = {}
    doublon_cache: dict[int, PauliPolynomial] = {}
    phonon_qubit_cache: dict[int, tuple[int, ...]] = {}
    b_cache: dict[int, PauliPolynomial] = {}
    bdag_cache: dict[int, PauliPolynomial] = {}
    p_cache: dict[int, PauliPolynomial] = {}
    x_cache: dict[int, PauliPolynomial] = {}
    sq_cache: dict[int, PauliPolynomial] = {}
    hopping_cache: dict[tuple[int, int], PauliPolynomial] = {}
    current_cache: dict[tuple[int, int], PauliPolynomial] = {}
    delta_p_cache: dict[tuple[int, int], PauliPolynomial] = {}
    delta_p_power_cache: dict[tuple[int, int, int], PauliPolynomial] = {}
    bond_p_sum_cache: dict[tuple[int, int], PauliPolynomial] = {}
    bond_sq_sum_cache: dict[tuple[int, int], PauliPolynomial] = {}
    pair_sq_cache: dict[tuple[int, int], PauliPolynomial] = {}
    pool: list[tuple[str, PauliPolynomial]] = []
    phonon_qubits = tuple(range(2 * n_sites, nq))

    def local_qubits(site: int) -> tuple[int, ...]:
        key = int(site)
        if key not in phonon_qubit_cache:
            phonon_qubit_cache[key] = tuple(
                phonon_qubit_indices_for_site(
                    key,
                    n_sites=n_sites,
                    qpb=boson_qubits_per_site(n_ph_max_i, boson_encoding_i),
                    fermion_qubits=2 * n_sites,
                )
            )
        return phonon_qubit_cache[key]

    def n_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in number_cache:
            up = mode_index(int(key), 0, indexing=ordering_i, n_sites=n_sites)
            down = mode_index(int(key), 1, indexing=ordering_i, n_sites=n_sites)
            n_up = jw_number_operator(repr_mode, nq, up)
            n_dn = jw_number_operator(repr_mode, nq, down)
            cached = n_up + n_dn
            number_cache[key] = cached
        return number_cache[key]

    def b_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in b_cache:
            b_cache[key] = boson_operator(
                repr_mode,
                nq,
                local_qubits(key),
                which="b",
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
        return b_cache[key]

    def bdag_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in bdag_cache:
            bdag_cache[key] = boson_operator(
                repr_mode,
                nq,
                local_qubits(key),
                which="bdag",
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
        return bdag_cache[key]

    def p_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in p_cache:
            # P = i (b^† - b)
            p_cache[key] = _clean_poly((1j * bdag_i(key)) + (-1j * b_i(key)), prune_eps)
        return p_cache[key]

    def x_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in x_cache:
            x_cache[key] = boson_displacement_operator(
                repr_mode,
                nq,
                local_qubits(key),
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
        return x_cache[key]

    def sq_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in sq_cache:
            b2 = _mul_clean(b_i(key), b_i(key), prune_eps, enforce_real=False)
            bdag2 = _mul_clean(bdag_i(key), bdag_i(key), prune_eps, enforce_real=False)
            sq_cache[key] = _clean_poly((1j * bdag2) + ((-1j) * b2), prune_eps)
        return sq_cache[key]

    def shifted_density(site: int) -> PauliPolynomial:
        n_site = n_i(site)
        if abs(nbar) < 1e-15:
            return n_site
        return n_site + ((-nbar) * id_poly)

    def doublon_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in doublon_cache:
            up = mode_index(key, 0, indexing=ordering_i, n_sites=n_sites)
            down = mode_index(key, 1, indexing=ordering_i, n_sites=n_sites)
            n_up = jw_number_operator(repr_mode, nq, up)
            n_dn = jw_number_operator(repr_mode, nq, down)
            doublon_cache[key] = _mul_clean(n_up, n_dn, prune_eps)
        return doublon_cache[key]

    def k_ij(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in hopping_cache:
            hopping = PauliPolynomial(repr_mode)
            for spin in (0, 1):
                i_spin = mode_index(key[0], spin, indexing=ordering_i, n_sites=n_sites)
                j_spin = mode_index(key[1], spin, indexing=ordering_i, n_sites=n_sites)
                term_ij = fermion_plus_operator(repr_mode, nq, i_spin) * fermion_minus_operator(repr_mode, nq, j_spin)
                term_ji = fermion_plus_operator(repr_mode, nq, j_spin) * fermion_minus_operator(repr_mode, nq, i_spin)
                hopping += term_ij
                hopping += term_ji
            hopping_cache[key] = _clean_poly(hopping, prune_eps)
        return hopping_cache[key]

    def j_ij(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in current_cache:
            current = PauliPolynomial(repr_mode)
            for spin in (0, 1):
                i_spin = mode_index(key[0], spin, indexing=ordering_i, n_sites=n_sites)
                j_spin = mode_index(key[1], spin, indexing=ordering_i, n_sites=n_sites)
                current += jw_current_hop(nq, i_spin, j_spin)
            current_cache[key] = _clean_poly(current, prune_eps)
        return current_cache[key]

    def delta_p_ij(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in delta_p_cache:
            delta_p_cache[key] = _clean_poly(p_i(key[0]) + ((-1.0) * p_i(key[1])), prune_eps)
        return delta_p_cache[key]

    def delta_p_power(i_site: int, j_site: int, exponent: int) -> PauliPolynomial:
        power = int(exponent)
        if power < 1:
            raise ValueError("delta_p_power exponent must be >= 1")
        key = (int(i_site), int(j_site), power)
        if key in delta_p_power_cache:
            return delta_p_power_cache[key]
        base = delta_p_ij(i_site, j_site)
        if power == 1:
            delta_p_power_cache[key] = base
            return base
        acc = base
        for _ in range(1, power):
            acc = _mul_clean(acc, base, prune_eps)
        delta_p_power_cache[key] = acc
        return acc

    def bond_p_sum(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in bond_p_sum_cache:
            bond_p_sum_cache[key] = _clean_poly(p_i(key[0]) + p_i(key[1]), prune_eps)
        return bond_p_sum_cache[key]

    def bond_sq_sum(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in bond_sq_sum_cache:
            bond_sq_sum_cache[key] = _clean_poly(sq_i(key[0]) + sq_i(key[1]), prune_eps)
        return bond_sq_sum_cache[key]

    def pair_sq_ij(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in pair_sq_cache:
            pair_create = _mul_clean(bdag_i(key[0]), bdag_i(key[1]), prune_eps, enforce_real=False)
            pair_annih = _mul_clean(b_i(key[0]), b_i(key[1]), prune_eps, enforce_real=False)
            pair_sq_cache[key] = _clean_poly((1j * pair_create) + ((-1j) * pair_annih), prune_eps)
        return pair_sq_cache[key]

    # (A) local conditional displacement dressing
    if include_disp:
        for site in range(n_sites):
            _append_operator(
                pool,
                f"paop_disp(site={site})",
                _normalize_poly(shifted_density(site) * p_i(site), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (B) legacy local doublon dressing
    if include_doublon:
        for site in range(n_sites):
            _append_operator(
                pool,
                f"paop_dbl(site={site})",
                _normalize_poly(shifted_density(site) * doublon_i(site), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    edges = bravais_nearest_neighbor_edges(n_sites, pbc=periodic) if (include_hopdrag or include_curdrag or include_hop2 or include_curdrag3 or include_hop4 or include_bond_disp or include_hop_sq or include_pair_sq) else []

    # (C) dressed hopping K_{ij}(P_i - P_j)
    if include_hopdrag:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_operator(
                pool,
                f"paop_hopdrag({i},{j})",
                _normalize_poly(_mul_clean(k_ij(i, j), delta_p_ij(i, j), prune_eps), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (D) LF-leading odd channel J_{ij}(P_i - P_j)
    if include_curdrag:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_operator(
                pool,
                f"paop_curdrag({i},{j})",
                _normalize_poly(_mul_clean(j_ij(i, j), delta_p_ij(i, j), prune_eps), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (E) LF second-order even channel K_{ij}(P_i - P_j)^2
    if include_hop2:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            hop2_poly = _mul_clean(k_ij(i, j), delta_p_power(i, j, 2), prune_eps)
            if drop_hop2_phonon_identity:
                hop2_poly = _drop_terms_with_identity_on_qubits(hop2_poly, phonon_qubits)
            _append_operator(
                pool,
                f"paop_hop2({i},{j})",
                _normalize_poly(hop2_poly, normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (E3) LF third-order odd current channel J_{ij}(P_i - P_j)^3
    if include_curdrag3:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_operator(
                pool,
                f"paop_curdrag3({i},{j})",
                _normalize_poly(_mul_clean(j_ij(i, j), delta_p_power(i, j, 3), prune_eps), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (E4) LF fourth-order even hopping channel K_{ij}(P_i - P_j)^4
    if include_hop4:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            hop4_poly = _mul_clean(k_ij(i, j), delta_p_power(i, j, 4), prune_eps)
            hop4_poly = _drop_terms_with_identity_on_qubits(hop4_poly, phonon_qubits)
            _append_operator(
                pool,
                f"paop_hop4({i},{j})",
                _normalize_poly(hop4_poly, normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (Bdisp) bond-conditioned symmetric displacement K_{ij}(P_i + P_j)
    if include_bond_disp:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_operator(
                pool,
                f"paop_bond_disp({i},{j})",
                _normalize_poly(_mul_clean(k_ij(i, j), bond_p_sum(i, j), prune_eps), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (HSQ) hopping-conditioned local squeeze K_{ij}(S_i + S_j)
    if include_hop_sq:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_operator(
                pool,
                f"paop_hop_sq({i},{j})",
                _normalize_poly(_mul_clean(k_ij(i, j), bond_sq_sum(i, j), prune_eps), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (PSQ) two-mode phonon pair squeeze i(b_i^† b_j^† - b_i b_j)
    if include_pair_sq:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_operator(
                pool,
                f"paop_pair_sq({i},{j})",
                _normalize_poly(pair_sq_ij(i, j), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (S) local phonon squeeze generator S_i and density-conditioned variants
    if include_sq:
        for site in range(n_sites):
            _append_operator(
                pool,
                f"paop_sq(site={site})",
                _normalize_poly(sq_i(site), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    if include_dens_sq:
        for site in range(n_sites):
            _append_operator(
                pool,
                f"paop_dens_sq(site={site})",
                _normalize_poly(_mul_clean(shifted_density(site), sq_i(site), prune_eps), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # Optional radius-R extension for cloud dressing
    if include_extended_cloud and cloud_radius >= 0:
        radius = int(cloud_radius)
        for i_site in range(n_sites):
            for j_site in range(n_sites):
                if i_site == j_site:
                    continue
                if _distance_1d(i_site, j_site, n_sites, periodic) > radius:
                    continue
                _append_operator(
                    pool,
                    f"paop_cloud_p(site={i_site}->phonon={j_site})",
                    _normalize_poly(shifted_density(i_site) * p_i(j_site), normalization),
                    split_paulis=split_paulis,
                    prune_eps=prune_eps,
                )
                if include_cloud_x:
                    _append_operator(
                        pool,
                        f"paop_cloud_x(site={i_site}->phonon={j_site})",
                        _normalize_poly(shifted_density(i_site) * x_i(j_site), normalization),
                        split_paulis=split_paulis,
                        prune_eps=prune_eps,
                    )

    if include_cloud_sq and cloud_radius >= 0:
        radius = int(cloud_radius)
        for i_site in range(n_sites):
            for j_site in range(n_sites):
                if i_site == j_site:
                    continue
                if _distance_1d(i_site, j_site, n_sites, periodic) > radius:
                    continue
                _append_operator(
                    pool,
                    f"paop_cloud_sq(site={i_site}->phonon={j_site})",
                    _normalize_poly(_mul_clean(shifted_density(i_site), sq_i(j_site), prune_eps), normalization),
                    split_paulis=split_paulis,
                    prune_eps=prune_eps,
                )

    # (F) LF doublon-conditioned phonon translation D_i p_j / D_i x_j
    if (include_doublon_translation_p or include_doublon_translation_x) and cloud_radius >= 0:
        radius = int(cloud_radius)
        for i_site in range(n_sites):
            for j_site in range(n_sites):
                if _distance_1d(i_site, j_site, n_sites, periodic) > radius:
                    continue
                if include_doublon_translation_p:
                    _append_operator(
                        pool,
                        f"paop_dbl_p(site={i_site}->phonon={j_site})",
                        _normalize_poly(doublon_i(i_site) * p_i(j_site), normalization),
                        split_paulis=split_paulis,
                        prune_eps=prune_eps,
                    )

    if (include_doublon_sq or include_doublon_translation_x) and cloud_radius >= 0:
        radius = int(cloud_radius)
        for i_site in range(n_sites):
            for j_site in range(n_sites):
                if _distance_1d(i_site, j_site, n_sites, periodic) > radius:
                    continue
                if include_doublon_sq:
                    _append_operator(
                        pool,
                        f"paop_dbl_sq(site={i_site}->phonon={j_site})",
                        _normalize_poly(_mul_clean(doublon_i(i_site), sq_i(j_site), prune_eps), normalization),
                        split_paulis=split_paulis,
                        prune_eps=prune_eps,
                    )
                if include_doublon_translation_x:
                    _append_operator(
                        pool,
                        f"paop_dbl_x(site={i_site}->phonon={j_site})",
                        _normalize_poly(doublon_i(i_site) * x_i(j_site), normalization),
                        split_paulis=split_paulis,
                        prune_eps=prune_eps,
                    )

    # Keep deterministic ordering and drop exact duplicates
    dedup: list[tuple[str, PauliPolynomial]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for label, poly in pool:
        sig = _to_signature(poly)
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append((f"{pool_name}:{label}", poly))
    return dedup


def make_pool(
    name: str,
    *,
    num_sites: int,
    num_particles: tuple[int, int] | None,
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
    ordering: str = "blocked",
    boundary: str = "open",
    paop_r: int = 0,
    paop_split_paulis: bool = False,
    paop_prune_eps: float = 0.0,
    paop_normalization: str = "none",
) -> list[tuple[str, PauliPolynomial]]:
    """Build PAOP pools for HH.

    Names accepted:
      - paop (alias to paop_std)
      - paop_min
      - paop_std
      - paop_full
      - paop_lf (alias to paop_lf_std)
      - paop_lf_std
      - paop_lf2_std
      - paop_lf3_std
      - paop_lf4_std
      - paop_lf_full
      - paop_sq_std
      - paop_sq_full
      - paop_bond_disp_std
      - paop_hop_sq_std
      - paop_pair_sq_std
    """
    mode = str(name).strip().lower()
    if mode == "paop":
        mode = "paop_std"
    if mode == "paop_lf":
        mode = "paop_lf_std"

    if mode not in {
        "paop_min",
        "paop_std",
        "paop_full",
        "paop_lf_std",
        "paop_lf2_std",
        "paop_lf3_std",
        "paop_lf4_std",
        "paop_lf_full",
        "paop_sq_std",
        "paop_sq_full",
        "paop_bond_disp_std",
        "paop_hop_sq_std",
        "paop_pair_sq_std",
    }:
        raise ValueError(
            "PAOP pool name must be one of paop, paop_min, paop_std, paop_full, "
            "paop_lf, paop_lf_std, paop_lf2_std, paop_lf3_std, paop_lf4_std, "
            "paop_lf_full, paop_sq_std, paop_sq_full, paop_bond_disp_std, paop_hop_sq_std, paop_pair_sq_std."
        )

    include_disp = True
    include_doublon = mode == "paop_full"
    include_hopdrag = mode in {
        "paop_std",
        "paop_full",
        "paop_lf_std",
        "paop_lf2_std",
        "paop_lf3_std",
        "paop_lf4_std",
        "paop_lf_full",
        "paop_sq_std",
        "paop_sq_full",
        "paop_bond_disp_std",
        "paop_hop_sq_std",
        "paop_pair_sq_std",
    }
    include_curdrag = mode in {
        "paop_lf_std",
        "paop_lf2_std",
        "paop_lf3_std",
        "paop_lf4_std",
        "paop_lf_full",
        "paop_sq_std",
        "paop_sq_full",
        "paop_bond_disp_std",
        "paop_hop_sq_std",
        "paop_pair_sq_std",
    }
    include_hop2 = mode in {"paop_lf2_std", "paop_lf3_std", "paop_lf4_std", "paop_lf_full"}
    include_curdrag3 = mode in {"paop_lf3_std", "paop_lf4_std"}
    include_hop4 = mode in {"paop_lf4_std"}
    include_bond_disp = mode in {"paop_bond_disp_std"}
    include_hop_sq = mode in {"paop_hop_sq_std"}
    include_pair_sq = mode in {"paop_pair_sq_std"}
    drop_hop2_phonon_identity = include_hop2
    include_extended = mode in {"paop_full", "paop_lf_full"}
    include_cloud_x = mode in {"paop_full", "paop_lf_full"}
    include_dbl_p = mode == "paop_lf_full"
    include_dbl_x = mode == "paop_lf_full"
    include_sq = mode in {"paop_sq_std", "paop_sq_full"}
    include_dens_sq = mode in {"paop_sq_std", "paop_sq_full"}
    include_cloud_sq = mode == "paop_sq_full"
    include_dbl_sq = mode == "paop_sq_full"
    radius = max(0, int(paop_r))
    if (include_extended or include_cloud_sq or include_dbl_sq) and radius == 0:
        radius = 1

    return _make_paop_core(
        num_sites=int(num_sites),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        num_particles=tuple(num_particles) if num_particles is not None else (),
        include_disp=include_disp,
        include_doublon=include_doublon,
        include_hopdrag=include_hopdrag,
        include_curdrag=include_curdrag,
        include_hop2=include_hop2,
        include_curdrag3=include_curdrag3,
        include_hop4=include_hop4,
        include_bond_disp=include_bond_disp,
        include_hop_sq=include_hop_sq,
        include_pair_sq=include_pair_sq,
        drop_hop2_phonon_identity=drop_hop2_phonon_identity,
        include_extended_cloud=include_extended,
        cloud_radius=radius,
        include_cloud_x=include_cloud_x,
        include_doublon_translation_p=include_dbl_p,
        include_doublon_translation_x=include_dbl_x,
        include_sq=include_sq,
        include_dens_sq=include_dens_sq,
        include_cloud_sq=include_cloud_sq,
        include_doublon_sq=include_dbl_sq,
        split_paulis=bool(paop_split_paulis),
        prune_eps=float(paop_prune_eps),
        normalization=str(paop_normalization),
        pool_name=mode,
    )
