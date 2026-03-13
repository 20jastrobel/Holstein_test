from __future__ import annotations

from typing import Any

from src.quantum.hubbard_latex_python_pairs import (
    boson_displacement_operator,
    boson_operator,
    boson_qubits_per_site,
    jw_number_operator,
    mode_index,
    phonon_qubit_indices_for_site,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.operator_pools.polaron_paop import (
    _clean_poly,
    _distance_1d,
    _mul_clean,
    _normalize_poly,
    _to_signature,
)

__all__ = ["build_vlf_sq_pool", "make_vlf_sq_pool"]

_MATH_SHIFTED_DENSITY = "δn_i := n_i - nbar I"
_MATH_VLF_SHELL = "G_r^VLF := Σ_{dist(i,j)=r} δn_i P_j"
_MATH_SQ = "G^SQ := Σ_i 1/2 (X_i P_i + P_i X_i) = Σ_i i[(b_i^†)^2 - b_i^2]"
_MATH_DENS_SQ = "G^(n)_SQ := Σ_i δn_i · 1/2 (X_i P_i + P_i X_i)"


def _family_flags(name: str) -> tuple[str, bool, bool, bool]:
    mode = str(name).strip().lower()
    if mode not in {"vlf_only", "sq_only", "vlf_sq", "sq_dens_only", "vlf_sq_dens"}:
        raise ValueError(
            "VLF/SQ family name must be one of vlf_only, sq_only, vlf_sq, sq_dens_only, vlf_sq_dens."
        )
    include_vlf = mode in {"vlf_only", "vlf_sq", "vlf_sq_dens"}
    include_sq = mode in {"sq_only", "vlf_sq", "vlf_sq_dens"}
    include_dens_sq = mode in {"sq_dens_only", "vlf_sq_dens"}
    return mode, include_vlf, include_sq, include_dens_sq


# Math: shells(periodic/open) := { r | 0 <= r <= r_max_effective and ∃(i,j) with dist(i,j)=r }
def _shells_for_radius(*, num_sites: int, periodic: bool, shell_radius: int | None) -> list[int]:
    if int(num_sites) <= 0:
        return []
    if bool(periodic):
        max_possible = int(num_sites) // 2
    else:
        max_possible = max(0, int(num_sites) - 1)
    if shell_radius is None:
        return list(range(max_possible + 1))
    cap = min(max(0, int(shell_radius)), max_possible)
    return list(range(cap + 1))


# Math: I := e^{\otimes nq}
def _identity_poly(nq: int) -> PauliPolynomial:
    return PauliPolynomial("JW", [PauliTerm(int(nq), ps="e" * int(nq), pc=1.0)])


# Math: build_vlf_sq_pool(name) -> {prefixed macro generators, metadata}
def build_vlf_sq_pool(
    name: str,
    *,
    num_sites: int,
    num_particles: tuple[int, int] | None,
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
    ordering: str = "blocked",
    boundary: str = "open",
    shell_radius: int | None = None,
    prune_eps: float = 0.0,
    normalization: str = "none",
) -> tuple[list[tuple[str, PauliPolynomial]], dict[str, Any]]:
    mode, include_vlf, include_sq, include_dens_sq = _family_flags(name)
    n_sites = int(num_sites)
    if n_sites <= 0:
        return [], {
            "family": mode,
            "nbar": 0.0,
            "shell_radius": None if shell_radius is None else int(shell_radius),
            "shells": [],
            "parameter_count": 0,
            "sq_parameterization": "global_shared",
            "density_conditioned_sq": bool(include_dens_sq),
            "math_contract": {
                "shifted_density": _MATH_SHIFTED_DENSITY,
                "vlf_shell": _MATH_VLF_SHELL,
                "sq": _MATH_SQ,
                "dens_sq": _MATH_DENS_SQ,
            },
        }

    n_ph_max_i = int(n_ph_max)
    boson_encoding_i = str(boson_encoding)
    ordering_i = str(ordering)
    periodic = str(boundary).strip().lower() == "periodic"
    total_electrons = int(num_particles[0]) + int(num_particles[1]) if num_particles else 0
    nbar = (float(total_electrons) / float(n_sites)) if total_electrons > 0 else 1.0
    if nbar <= 0.0:
        nbar = 1.0

    nq = 2 * n_sites + n_sites * boson_qubits_per_site(n_ph_max_i, boson_encoding_i)
    id_poly = _identity_poly(nq)
    phonon_qubit_cache: dict[int, tuple[int, ...]] = {}
    number_cache: dict[int, PauliPolynomial] = {}
    p_cache: dict[int, PauliPolynomial] = {}
    x_cache: dict[int, PauliPolynomial] = {}
    sq_cache: dict[int, PauliPolynomial] = {}

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
            up = mode_index(key, 0, indexing=ordering_i, n_sites=n_sites)
            dn = mode_index(key, 1, indexing=ordering_i, n_sites=n_sites)
            number_cache[key] = jw_number_operator("JW", nq, up) + jw_number_operator("JW", nq, dn)
        return number_cache[key]

    def shifted_density(site: int) -> PauliPolynomial:
        return n_i(int(site)) + ((-float(nbar)) * id_poly)

    def b_i(site: int) -> PauliPolynomial:
        return boson_operator(
            "JW",
            nq,
            local_qubits(int(site)),
            which="b",
            n_ph_max=n_ph_max_i,
            encoding=boson_encoding_i,
        )

    def bdag_i(site: int) -> PauliPolynomial:
        return boson_operator(
            "JW",
            nq,
            local_qubits(int(site)),
            which="bdag",
            n_ph_max=n_ph_max_i,
            encoding=boson_encoding_i,
        )

    def p_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in p_cache:
            p_cache[key] = _clean_poly((1j * bdag_i(key)) + ((-1j) * b_i(key)), float(prune_eps))
        return p_cache[key]

    def x_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in x_cache:
            x_cache[key] = boson_displacement_operator(
                "JW",
                nq,
                local_qubits(int(key)),
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
        return x_cache[key]

    def sq_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in sq_cache:
            xp = _mul_clean(x_i(key), p_i(key), float(prune_eps), enforce_real=False)
            px = _mul_clean(p_i(key), x_i(key), float(prune_eps), enforce_real=False)
            sq_cache[key] = _clean_poly(0.5 * (xp + px), float(prune_eps))
        return sq_cache[key]

    shells = _shells_for_radius(num_sites=n_sites, periodic=periodic, shell_radius=shell_radius)
    raw_pool: list[tuple[str, PauliPolynomial]] = []

    if include_vlf:
        for shell in shells:
            shell_poly = PauliPolynomial("JW")
            pair_count = 0
            for i_site in range(n_sites):
                for j_site in range(n_sites):
                    if _distance_1d(i_site, j_site, n_sites, periodic) != int(shell):
                        continue
                    shell_poly += _mul_clean(shifted_density(i_site), p_i(j_site), float(prune_eps))
                    pair_count += 1
            shell_poly = _clean_poly(shell_poly, float(prune_eps))
            if shell_poly.return_polynomial():
                raw_pool.append((f"vlf_shell(r={shell})", _normalize_poly(shell_poly, str(normalization))))

    if include_sq:
        sq_poly = PauliPolynomial("JW")
        for site in range(n_sites):
            sq_poly += sq_i(site)
        sq_poly = _clean_poly(sq_poly, float(prune_eps))
        if sq_poly.return_polynomial():
            raw_pool.append(("sq_global", _normalize_poly(sq_poly, str(normalization))))

    if include_dens_sq:
        dens_sq_poly = PauliPolynomial("JW")
        for site in range(n_sites):
            dens_sq_poly += _mul_clean(shifted_density(site), sq_i(site), float(prune_eps))
        dens_sq_poly = _clean_poly(dens_sq_poly, float(prune_eps))
        if dens_sq_poly.return_polynomial():
            raw_pool.append(("dens_sq_global", _normalize_poly(dens_sq_poly, str(normalization))))

    if mode == "sq_only" and not raw_pool:
        raise ValueError("sq_only produced no surviving squeeze generators; n_ph_max may be too small.")
    if mode == "sq_dens_only" and not raw_pool:
        raise ValueError("sq_dens_only produced no surviving density-conditioned squeeze generators; n_ph_max may be too small.")

    dedup: list[tuple[str, PauliPolynomial]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for label, poly in raw_pool:
        sig = _to_signature(poly)
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append((f"{mode}:{label}", poly))

    meta = {
        "family": mode,
        "nbar": float(nbar),
        "shell_radius": None if shell_radius is None else int(shell_radius),
        "shells": list(shells if include_vlf else []),
        "parameter_count": int(len(dedup)),
        "sq_parameterization": "global_shared" if include_sq or include_dens_sq else "off",
        "density_conditioned_sq": bool(include_dens_sq),
        "math_contract": {
            "shifted_density": _MATH_SHIFTED_DENSITY,
            "vlf_shell": _MATH_VLF_SHELL,
            "sq": _MATH_SQ,
            "dens_sq": _MATH_DENS_SQ,
        },
    }
    return dedup, meta


def make_vlf_sq_pool(
    name: str,
    *,
    num_sites: int,
    num_particles: tuple[int, int] | None,
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
    ordering: str = "blocked",
    boundary: str = "open",
    shell_radius: int | None = None,
    prune_eps: float = 0.0,
    normalization: str = "none",
) -> list[tuple[str, PauliPolynomial]]:
    pool, _meta = build_vlf_sq_pool(
        name,
        num_sites=int(num_sites),
        num_particles=tuple(num_particles) if num_particles is not None else (),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        shell_radius=None if shell_radius is None else int(shell_radius),
        prune_eps=float(prune_eps),
        normalization=str(normalization),
    )
    return pool
