"""Polaron-adapted operator pool for Hubbard-Holstein (HH) ADAPT-VQE.

This module exposes a lightweight, composable PAOP pool builder using the
existing Pauli-layer operators from the repository math stack.
"""

from __future__ import annotations

import math

try:
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
    from src.quantum.pauli_words import PauliTerm
except Exception:
    from pydephasing.quantum.hubbard_latex_python_pairs import (
        bravais_nearest_neighbor_edges,
        boson_operator,
        boson_displacement_operator,
        boson_qubits_per_site,
        jw_number_operator,
        mode_index,
        phonon_qubit_indices_for_site,
    )
    from pydephasing.quantum.pauli_polynomial_class import PauliPolynomial, fermion_minus_operator, fermion_plus_operator
    from pydephasing.quantum.pauli_words import PauliTerm


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
    terms = poly.return_polynomial()
    if not terms:
        return PauliPolynomial("JW")
    nq = int(terms[0].nqubit())
    cleaned = PauliPolynomial("JW")
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(prune_eps):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"PAOP generator has non-negligible imaginary coefficient: {coeff}")
        cleaned.add_term(PauliTerm(nq, ps=str(term.pw2strng()), pc=float(coeff.real)))
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


def _distance_1d(i: int, j: int, n_sites: int, periodic: bool) -> int:
    dist = abs(int(i) - int(j))
    if periodic and n_sites > 0:
        period = int(n_sites)
        dist = min(dist, period - dist)
    return int(dist)


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
    include_extended_cloud: bool,
    cloud_radius: int,
    include_cloud_x: bool,
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
    p_cache: dict[int, PauliPolynomial] = {}
    x_cache: dict[int, PauliPolynomial] = {}
    pool: list[tuple[str, PauliPolynomial]] = []

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

    def p_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in p_cache:
            qubits = phonon_qubit_indices_for_site(
                int(key),
                n_sites=n_sites,
                qpb=boson_qubits_per_site(n_ph_max_i, boson_encoding_i),
                fermion_qubits=2 * n_sites,
            )
            b_op = boson_operator(
                repr_mode,
                nq,
                qubits,
                which="b",
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
            bdag_op = boson_operator(
                repr_mode,
                nq,
                qubits,
                which="bdag",
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
            # P = i (b^† - b)
            p_cache[key] = (1j * bdag_op) + (-1j * b_op)
        return p_cache[key]

    def x_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in x_cache:
            qubits = phonon_qubit_indices_for_site(
                int(key),
                n_sites=n_sites,
                qpb=boson_qubits_per_site(n_ph_max_i, boson_encoding_i),
                fermion_qubits=2 * n_sites,
            )
            x_cache[key] = boson_displacement_operator(
                repr_mode,
                nq,
                qubits,
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
        return x_cache[key]

    def shifted_density(site: int) -> PauliPolynomial:
        n_site = n_i(site)
        if abs(nbar) < 1e-15:
            return n_site
        return n_site + ((-nbar) * id_poly)

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

    # (B) local doublon dressing
    if include_doublon:
        for site in range(n_sites):
            up = mode_index(int(site), 0, indexing=ordering_i, n_sites=n_sites)
            down = mode_index(int(site), 1, indexing=ordering_i, n_sites=n_sites)
            n_up = jw_number_operator(repr_mode, nq, up)
            n_dn = jw_number_operator(repr_mode, nq, down)
            doublon = n_up * n_dn
            _append_operator(
                pool,
                f"paop_dbl(site={site})",
                _normalize_poly(shifted_density(site) * doublon, normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (C) dressed hopping K_{ij}(P_i - P_j)
    if include_hopdrag:
        edges = bravais_nearest_neighbor_edges(n_sites, pbc=periodic)
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            hopping = PauliPolynomial(repr_mode)
            for spin in (0, 1):
                i_spin = mode_index(i, spin, indexing=ordering_i, n_sites=n_sites)
                j_spin = mode_index(j, spin, indexing=ordering_i, n_sites=n_sites)
                term_ij = fermion_plus_operator(repr_mode, nq, i_spin) * fermion_minus_operator(repr_mode, nq, j_spin)
                term_ji = fermion_plus_operator(repr_mode, nq, j_spin) * fermion_minus_operator(repr_mode, nq, i_spin)
                hopping += term_ij
                hopping += term_ji
            _append_operator(
                pool,
                f"paop_hopdrag({i},{j})",
                _normalize_poly(hopping * (p_i(i) + ((-1.0) * p_i(j))), normalization),
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
    """Build PAOP pools for HH: paop_min, paop_std, paop_full.

    Names accepted:
      - paop (alias to paop_std)
      - paop_min
      - paop_std
      - paop_full
    """
    mode = str(name).strip().lower()
    if mode == "paop":
        mode = "paop_std"

    if mode not in {"paop_min", "paop_std", "paop_full"}:
        raise ValueError("PAOP pool name must be one of paop, paop_min, paop_std, paop_full.")

    include_disp = True
    include_doublon = mode == "paop_full"
    include_hopdrag = mode in {"paop_std", "paop_full"}
    include_extended = mode == "paop_full"
    include_cloud_x = mode == "paop_full"
    radius = max(0, int(paop_r))
    if include_extended and radius == 0:
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
        include_extended_cloud=include_extended,
        cloud_radius=radius,
        include_cloud_x=include_cloud_x,
        split_paulis=bool(paop_split_paulis),
        prune_eps=float(paop_prune_eps),
        normalization=str(paop_normalization),
        pool_name=mode,
    )
