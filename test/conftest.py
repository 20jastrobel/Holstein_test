"""
Shared Hubbard-Holstein test constants & fixtures.

Canonical parameter values for L=2 HH integration tests.
Individual test files may re-declare local aliases for convenience,
but should keep values consistent with the definitions here.

Pytest fixtures (lowercase names) are available for fixture-style injection.
"""
from __future__ import annotations

import pytest

# ── Default L=2 Hubbard-Holstein parameter set ───────────────────────────
HH_L: int = 2
HH_T: float = 1.0
HH_U: float = 4.0
HH_DV: float = 0.0
HH_OMEGA0: float = 1.0
HH_G_EP: float = 0.5
HH_N_PH_MAX: int = 1
HH_BOSON_ENCODING: str = "binary"
HH_BOUNDARY: str = "periodic"
HH_ORDERING: str = "blocked"
HH_HALF_FILL: tuple[int, int] = (1, 1)   # n_alpha = n_beta = L//2
HH_ENCODINGS: tuple[str, ...] = ("binary", "unary")


# ── Pytest fixtures ──────────────────────────────────────────────────────
@pytest.fixture()
def hh_params() -> dict:
    """Return the default L=2 HH parameter dict for injection."""
    return dict(
        L=HH_L, t=HH_T, U=HH_U, dv=HH_DV,
        omega0=HH_OMEGA0, g_ep=HH_G_EP,
        n_ph_max=HH_N_PH_MAX, boson_encoding=HH_BOSON_ENCODING,
        boundary=HH_BOUNDARY, ordering=HH_ORDERING,
        half_fill=HH_HALF_FILL,
    )
