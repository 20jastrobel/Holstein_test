from __future__ import annotations

from test_hh_adapt_vqe_ground_states import _case_settings, _tol_for_lattice


def test_tolerance_can_be_overridden(monkeypatch: object) -> None:
    monkeypatch.setenv("ADAPT_HH_TOL_L2", "0.007")
    monkeypatch.setenv("ADAPT_HH_TOL_L3", "0.9")

    assert _tol_for_lattice(2) == 0.007
    assert _tol_for_lattice(3) == 0.9


def test_case_settings_respect_env_overrides(monkeypatch: object) -> None:
    monkeypatch.setenv("ADAPT_HH_MAXITER_L2", "11")
    monkeypatch.setenv("ADAPT_HH_MAX_DEPTH_L2", "21")
    monkeypatch.setenv("ADAPT_HH_EPS_GRAD_L2", "1e-3")
    monkeypatch.setenv("ADAPT_HH_EPS_ENERGY_L2", "1e-6")

    settings = _case_settings(2)

    assert settings["adapt_maxiter"] == 11
    assert settings["adapt_max_depth"] == 21
    assert settings["adapt_eps_grad"] == 1e-3
    assert settings["adapt_eps_energy"] == 1e-6
