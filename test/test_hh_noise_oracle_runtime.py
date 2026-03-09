from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.reports.qiskit_circuit_report import (
    adapt_ops_to_circuit as shared_adapt_ops_to_circuit,
    ansatz_to_circuit as shared_ansatz_to_circuit,
    build_cfqm_time_dependent_circuit as shared_build_cfqm_time_dependent_circuit,
    build_suzuki2_time_dependent_circuit as shared_build_suzuki2_time_dependent_circuit,
)
import pipelines.exact_bench.noise_oracle_runtime as nor
import pipelines.exact_bench.noise_snapshot as nsnap
from pipelines.exact_bench.noise_aer_builders import build_shots_only_artifact
from pipelines.exact_bench.noise_model_spec import normalize_to_resolved_noise_spec
from pipelines.exact_bench.noise_oracle_runtime import (
    ExpectationOracle,
    OracleConfig,
    _ansatz_to_circuit,
    _pauli_poly_to_sparse_pauli_op,
    normalize_mitigation_config,
    normalize_ideal_reference_symmetry_mitigation,
    normalize_symmetry_mitigation_config,
)
from pipelines.exact_bench.noise_snapshot import freeze_backend_snapshot
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.qubitization_module import PauliTerm
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, HubbardHolsteinTermwiseAnsatz, apply_exp_pauli_polynomial


def test_pauli_poly_to_sparse_pauli_op_preserves_exyz_to_ixyz_mapping() -> None:
    poly = PauliPolynomial(3)
    poly.add_term(PauliTerm(3, ps="xez", pc=0.5))
    poly.add_term(PauliTerm(3, ps="eey", pc=-0.25))
    poly.add_term(PauliTerm(3, ps="xez", pc=0.5))

    qop = _pauli_poly_to_sparse_pauli_op(poly)
    coeffs = {lbl: complex(c) for lbl, c in qop.to_list()}

    assert "XIZ" in coeffs
    assert "IIY" in coeffs
    assert coeffs["XIZ"] == pytest.approx(1.0 + 0.0j)
    assert coeffs["IIY"] == pytest.approx(-0.25 + 0.0j)


def test_ansatz_to_circuit_matches_prepare_state_for_hh_termwise_small_case() -> None:
    num_sites = 2
    num_particles = (1, 1)
    ansatz = HubbardHolsteinTermwiseAnsatz(
        dims=num_sites,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        reps=1,
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
    )
    psi_ref = hubbard_holstein_reference_state(
        dims=num_sites,
        num_particles=num_particles,
        n_ph_max=1,
        boson_encoding="binary",
        indexing="blocked",
    )
    rng = np.random.default_rng(123)
    theta = 0.05 * rng.normal(size=int(ansatz.num_parameters))

    qc = _ansatz_to_circuit(
        ansatz,
        theta,
        num_qubits=int(ansatz.nq),
        reference_state=np.asarray(psi_ref, dtype=complex),
    )
    qc_shared = shared_ansatz_to_circuit(
        ansatz,
        theta,
        num_qubits=int(ansatz.nq),
        reference_state=np.asarray(psi_ref, dtype=complex),
    )
    psi_circuit = np.asarray(Statevector.from_instruction(qc).data, dtype=complex).reshape(-1)
    psi_shared = np.asarray(Statevector.from_instruction(qc_shared).data, dtype=complex).reshape(-1)
    psi_expected = np.asarray(ansatz.prepare_state(theta, psi_ref), dtype=complex).reshape(-1)

    fidelity = float(abs(np.vdot(psi_expected, psi_circuit)) ** 2)
    assert fidelity > 1.0 - 1e-10
    assert float(abs(np.vdot(psi_circuit, psi_shared)) ** 2) > 1.0 - 1e-10


def test_adapt_ops_to_circuit_matches_selected_operator_state_preparation() -> None:
    poly_0 = PauliPolynomial(2)
    poly_0.add_term(PauliTerm(2, ps="xe", pc=0.5))
    poly_1 = PauliPolynomial(2)
    poly_1.add_term(PauliTerm(2, ps="ez", pc=-0.25))
    ops = [
        AnsatzTerm(label="op_0", polynomial=poly_0),
        AnsatzTerm(label="op_1", polynomial=poly_1),
    ]
    theta = np.array([0.2, -0.15], dtype=float)
    psi_ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

    qc = shared_adapt_ops_to_circuit(
        ops,
        theta,
        num_qubits=2,
        reference_state=psi_ref,
    )
    psi_circuit = np.asarray(Statevector.from_instruction(qc).data, dtype=complex).reshape(-1)

    psi_expected = np.array(psi_ref, copy=True)
    for op, ang in zip(ops, theta):
        psi_expected = apply_exp_pauli_polynomial(psi_expected, op.polynomial, float(ang))
    fidelity = float(abs(np.vdot(psi_expected, psi_circuit)) ** 2)
    assert fidelity > 1.0 - 1e-10


def test_time_dependent_macro_step_builders_return_valid_circuits_and_zero_time_identity() -> None:
    initial = QuantumCircuit(2)
    initial.x(0)
    ordered_labels_exyz = ["ze", "xz"]
    coeff_map_exyz = {"ze": 1.0 + 0.0j, "xz": 0.25 + 0.0j}

    suzuki = shared_build_suzuki2_time_dependent_circuit(
        initial_circuit=initial,
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=coeff_map_exyz,
        drive_provider_exyz=None,
        time_value=0.2,
        trotter_steps=1,
        drive_t0=0.0,
        drive_time_sampling="midpoint",
    )
    cfqm = shared_build_cfqm_time_dependent_circuit(
        method="cfqm4",
        initial_circuit=initial,
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=coeff_map_exyz,
        drive_provider_exyz=None,
        time_value=0.2,
        trotter_steps=1,
        drive_t0=0.0,
        coeff_drop_abs_tol=0.0,
    )
    suzuki_zero = shared_build_suzuki2_time_dependent_circuit(
        initial_circuit=initial,
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=coeff_map_exyz,
        drive_provider_exyz=None,
        time_value=0.0,
        trotter_steps=1,
        drive_t0=0.0,
        drive_time_sampling="midpoint",
    )
    cfqm_zero = shared_build_cfqm_time_dependent_circuit(
        method="cfqm4",
        initial_circuit=initial,
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=coeff_map_exyz,
        drive_provider_exyz=None,
        time_value=0.0,
        trotter_steps=1,
        drive_t0=0.0,
        coeff_drop_abs_tol=0.0,
    )

    assert suzuki.num_qubits == 2
    assert cfqm.num_qubits == 2
    assert suzuki.depth() >= initial.depth()
    assert cfqm.depth() >= initial.depth()
    assert Statevector.from_instruction(suzuki_zero).equiv(Statevector.from_instruction(initial))
    assert Statevector.from_instruction(cfqm_zero).equiv(Statevector.from_instruction(initial))


def test_ideal_oracle_matches_statevector_expectation() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])

    with ExpectationOracle(
        OracleConfig(noise_mode="ideal", oracle_repeats=3, oracle_aggregate="mean")
    ) as oracle:
        est = oracle.evaluate(qc, obs)

    exact = float(np.real(Statevector.from_instruction(qc).expectation_value(obs)))
    assert est.mean == pytest.approx(exact, abs=1e-10)
    assert est.std == pytest.approx(0.0, abs=1e-12)
    assert est.stdev == pytest.approx(0.0, abs=1e-12)
    assert est.stderr == pytest.approx(0.0, abs=1e-12)
    assert str(est.aggregate) == "mean"


def test_oracle_mitigation_config_is_normalized_and_recorded() -> None:
    qc = QuantumCircuit(1)
    obs = SparsePauliOp.from_list([("I", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="ideal",
            mitigation={"mode": "zne", "zne_scales": "1.0,2.0,3.0"},
        )
    ) as oracle:
        _ = oracle.evaluate(qc, obs)
        mit = oracle.config.mitigation
        assert isinstance(mit, dict)
        assert mit["mode"] == "zne"
        assert mit["zne_scales"] == [1.0, 2.0, 3.0]
        assert mit["dd_sequence"] is None
        assert dict(oracle.backend_info.details.get("mitigation", {})) == mit


def test_symmetry_mitigation_config_is_normalized() -> None:
    cfg = normalize_symmetry_mitigation_config(
        {
            "mode": "postselect_diag_v1",
            "num_sites": 2,
            "ordering": "blocked",
            "sector_n_up": 1,
            "sector_n_dn": 1,
        }
    )
    assert cfg == {
        "mode": "postselect_diag_v1",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }


def test_ideal_reference_symmetry_mitigation_downgrades_runtime_to_verify_only() -> None:
    cfg = normalize_ideal_reference_symmetry_mitigation(
        {
            "mode": "projector_renorm_v1",
            "num_sites": 2,
            "ordering": "blocked",
            "sector_n_up": 1,
            "sector_n_dn": 1,
        },
        noise_mode="runtime",
    )
    assert cfg["mode"] == "verify_only"
    assert cfg["num_sites"] == 2
    assert cfg["sector_n_up"] == 1
    assert cfg["sector_n_dn"] == 1


def test_postselect_diag_v1_filters_to_target_sector_for_diagonal_observable() -> None:
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="ideal",
            oracle_repeats=2,
            oracle_aggregate="mean",
            symmetry_mitigation={
                "mode": "postselect_diag_v1",
                "num_sites": 1,
                "ordering": "blocked",
                "sector_n_up": 1,
                "sector_n_dn": 0,
            },
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        sym = dict(oracle.backend_info.details.get("symmetry_mitigation", {}))

    assert est.mean == pytest.approx(1.0, abs=1e-10)
    assert sym.get("applied_mode") == "postselect_diag_v1"
    assert sym.get("retained_fraction_mean") == pytest.approx(0.5, abs=1e-10)
    assert sym.get("sector_probability_mean") == pytest.approx(0.5, abs=1e-10)


def test_projector_renorm_v1_matches_postselect_diag_for_diagonal_case() -> None:
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])
    sym_cfg = {
        "mode": "postselect_diag_v1",
        "num_sites": 1,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 0,
    }

    with ExpectationOracle(OracleConfig(noise_mode="ideal", symmetry_mitigation=sym_cfg)) as post_oracle:
        post_est = post_oracle.evaluate(qc, obs)

    with ExpectationOracle(
        OracleConfig(
            noise_mode="ideal",
            symmetry_mitigation={**sym_cfg, "mode": "projector_renorm_v1"},
        )
    ) as proj_oracle:
        proj_est = proj_oracle.evaluate(qc, obs)
        sym = dict(proj_oracle.backend_info.details.get("symmetry_mitigation", {}))

    assert proj_est.mean == pytest.approx(post_est.mean, abs=1e-10)
    assert proj_est.mean == pytest.approx(1.0, abs=1e-10)
    assert sym.get("applied_mode") == "projector_renorm_v1"
    assert sym.get("estimator_form") == "projector_ratio_diag_v1"


def test_symmetry_mitigation_non_diagonal_falls_back_to_verify_only() -> None:
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    obs = SparsePauliOp.from_list([("XI", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="ideal",
            symmetry_mitigation={
                "mode": "postselect_diag_v1",
                "num_sites": 1,
                "ordering": "blocked",
                "sector_n_up": 1,
                "sector_n_dn": 0,
            },
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        sym = dict(oracle.backend_info.details.get("symmetry_mitigation", {}))

    assert np.isfinite(est.mean)
    assert sym.get("applied_mode") == "verify_only"
    assert sym.get("fallback_reason") == "observable_not_diagonal"


def test_symmetry_mitigation_zero_retained_probability_falls_back_explicitly() -> None:
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="ideal",
            symmetry_mitigation={
                "mode": "postselect_diag_v1",
                "num_sites": 1,
                "ordering": "blocked",
                "sector_n_up": 0,
                "sector_n_dn": 0,
            },
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        sym = dict(oracle.backend_info.details.get("symmetry_mitigation", {}))

    assert np.isfinite(est.mean)
    assert sym.get("applied_mode") == "verify_only"
    assert "zero probability mass" in str(sym.get("fallback_reason", ""))


def test_local_symmetry_non_diagonal_raises_without_allow_noisy_fallback() -> None:
    pytest.importorskip("qiskit_aer")

    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    obs = SparsePauliOp.from_list([("XI", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="shots",
            shots=64,
            seed=21,
            layout_lock_key="symmetry_non_diag_raise_2q",
            symmetry_mitigation={
                "mode": "postselect_diag_v1",
                "num_sites": 1,
                "ordering": "blocked",
                "sector_n_up": 1,
                "sector_n_dn": 0,
            },
        )
    ) as oracle:
        with pytest.raises(RuntimeError, match="allow-noisy-fallback"):
            _ = oracle.evaluate(qc, obs)


def test_local_symmetry_non_diagonal_downgrades_with_allow_noisy_fallback() -> None:
    pytest.importorskip("qiskit_aer")

    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    obs = SparsePauliOp.from_list([("XI", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="shots",
            shots=64,
            seed=21,
            allow_noisy_fallback=True,
            layout_lock_key="symmetry_non_diag_fallback_2q",
            symmetry_mitigation={
                "mode": "postselect_diag_v1",
                "num_sites": 1,
                "ordering": "blocked",
                "sector_n_up": 1,
                "sector_n_dn": 0,
            },
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        sym = dict(oracle.backend_info.details.get("symmetry_mitigation", {}))
        warnings = list(oracle.backend_info.details.get("warnings", []))

    assert np.isfinite(est.mean)
    assert sym.get("applied_mode") == "verify_only"
    assert sym.get("fallback_reason") == "observable_not_diagonal"
    assert any("symmetry_mitigation_downgraded:observable_not_diagonal" in str(w) for w in warnings)


def test_shots_oracle_standard_error_improves_with_more_repeats() -> None:
    pytest.importorskip("qiskit_aer")
    qc = QuantumCircuit(1)
    qc.h(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])

    errs_r2: list[float] = []
    errs_r8: list[float] = []
    for seed in range(30, 40):
        with ExpectationOracle(
            OracleConfig(
                noise_mode="shots",
                shots=128,
                seed=seed,
                oracle_repeats=2,
                oracle_aggregate="mean",
            )
        ) as o2:
            e2 = o2.evaluate(qc, obs)
        with ExpectationOracle(
            OracleConfig(
                noise_mode="shots",
                shots=128,
                seed=seed,
                oracle_repeats=8,
                oracle_aggregate="mean",
            )
        ) as o8:
            e8 = o8.evaluate(qc, obs)

        errs_r2.append(abs(float(e2.mean)))
        errs_r8.append(abs(float(e8.mean)))
        assert float(e8.stderr) <= float(e2.stderr) + 1e-9

    assert float(np.mean(errs_r8)) <= float(np.mean(errs_r2)) + 1e-9


def test_aer_noise_mode_deterministic_with_fixed_seed_and_fake_backend() -> None:
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_ibm_runtime")

    qc = QuantumCircuit(1)
    qc.x(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    cfg = OracleConfig(
        noise_mode="aer_noise",
        shots=256,
        seed=111,
        oracle_repeats=4,
        oracle_aggregate="mean",
        backend_name="FakeManilaV2",
        use_fake_backend=True,
    )

    with ExpectationOracle(cfg) as oracle_a:
        est_a = oracle_a.evaluate(qc, obs)
        meta_a = dict(oracle_a.backend_info.details)
    with ExpectationOracle(cfg) as oracle_b:
        est_b = oracle_b.evaluate(qc, obs)
        meta_b = dict(oracle_b.backend_info.details)

    assert est_a.raw_values == pytest.approx(est_b.raw_values)
    assert est_a.mean == pytest.approx(est_b.mean)
    assert est_a.std == pytest.approx(est_b.std)
    assert est_a.stdev == pytest.approx(est_b.stdev)
    assert est_a.stderr == pytest.approx(est_b.stderr)
    assert meta_a.get("resolved_noise_spec", {}).get("noise_kind") == "backend_scheduled"
    assert meta_a.get("snapshot_hash") == meta_b.get("snapshot_hash")
    assert meta_a.get("layout_hash") == meta_b.get("layout_hash")
    assert meta_a.get("noise_artifact_hash") == meta_b.get("noise_artifact_hash")


def test_forced_aer_failure_raises_without_explicit_noisy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_cfg, **_kwargs):
        raise RuntimeError("OMP: Error #178: Function Can't open SHM2 failed")

    monkeypatch.setattr(nor, "_build_estimator", _boom)
    with pytest.raises(RuntimeError):
        _ = ExpectationOracle(
            OracleConfig(
                noise_mode="shots",
                shots=128,
                seed=9,
                oracle_repeats=3,
            )
        )


def test_forced_aer_failure_triggers_sampler_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_cfg):
        raise RuntimeError("OMP: Error #178: Function Can't open SHM2 failed")

    monkeypatch.setattr(nor, "_build_estimator", _boom)
    qc = QuantumCircuit(1)
    qc.h(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="shots",
            shots=128,
            seed=9,
            oracle_repeats=3,
            allow_noisy_fallback=True,
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)

    assert np.isfinite(est.mean)
    assert bool(oracle.backend_info.details.get("fallback_used")) is True
    assert bool(oracle.backend_info.details.get("aer_failed")) is True
    assert oracle.backend_info.details.get("resolved_noise_spec", {}).get("noise_kind") == "shots_only"
    assert oracle.backend_info.details.get("source_kind") == "generic_seeded"
    assert oracle.backend_info.details.get("resolved_noise_spec_hash")
    assert str(oracle.backend_info.estimator_kind).startswith("qiskit.primitives.StatevectorSampler")


def test_sampler_fallback_deterministic_with_fixed_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_cfg):
        raise RuntimeError("OMP: Error #178: Function Can't open SHM2 failed")

    monkeypatch.setattr(nor, "_build_estimator", _boom)
    qc = QuantumCircuit(1)
    qc.h(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])

    cfg = OracleConfig(
        noise_mode="shots",
        shots=128,
        seed=77,
        oracle_repeats=4,
        allow_noisy_fallback=True,
    )
    with ExpectationOracle(cfg) as oa:
        ea = oa.evaluate(qc, obs)
    with ExpectationOracle(cfg) as ob:
        eb = ob.evaluate(qc, obs)

    assert ea.raw_values == pytest.approx(eb.raw_values)
    assert ea.mean == pytest.approx(eb.mean)


def test_local_symmetry_execution_error_is_not_silently_downgraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("qiskit_aer")

    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="shots",
            shots=64,
            seed=21,
            symmetry_mitigation={
                "mode": "postselect_diag_v1",
                "num_sites": 1,
                "ordering": "blocked",
                "sector_n_up": 1,
                "sector_n_dn": 0,
            },
        )
    ) as oracle:
        def _boom(*_args, **_kwargs):
            raise RuntimeError("synthetic local counts failure")

        monkeypatch.setattr(oracle, "_run_local_measurement_counts", _boom)
        with pytest.raises(RuntimeError, match="synthetic local counts failure"):
            _ = oracle.evaluate(qc, obs)


def test_snapshot_hash_ignores_retrieved_at_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
    except Exception as exc:  # pragma: no cover - qiskit version guard
        pytest.skip(f"GenericBackendV2 unavailable: {exc}")

    backend = GenericBackendV2(
        2,
        basis_gates=["id", "rz", "sx", "x", "cx", "measure", "delay", "reset"],
        coupling_map=[[0, 1], [1, 0]],
        dt=2.2222222222222221e-10,
        seed=7,
        noise_info=True,
    )
    monkeypatch.setattr(nsnap, "_now_utc", lambda: "2026-03-09T00:00:00Z")
    snap_a = nsnap.freeze_backend_snapshot(backend)
    monkeypatch.setattr(nsnap, "_now_utc", lambda: "2026-03-09T00:30:00Z")
    snap_b = nsnap.freeze_backend_snapshot(backend)

    assert snap_a.retrieved_at_utc != snap_b.retrieved_at_utc
    assert snap_a.snapshot_hash == snap_b.snapshot_hash


def test_mitigation_config_rejects_aux_fields_for_wrong_mode() -> None:
    with pytest.raises(ValueError, match="zne_scales require mitigation mode 'zne'"):
        normalize_mitigation_config({"mode": "none", "zne_scales": [1.0, 2.0]})
    with pytest.raises(ValueError, match="dd_sequence requires mitigation mode 'dd'"):
        normalize_mitigation_config({"mode": "readout", "dd_sequence": "XY4"})


def test_shots_only_artifact_reports_backend_physical_edges_for_fixed_patch() -> None:
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
    except Exception as exc:  # pragma: no cover - qiskit version guard
        pytest.skip(f"GenericBackendV2 unavailable: {exc}")

    backend = GenericBackendV2(
        5,
        basis_gates=["id", "rz", "sx", "x", "cx", "measure", "delay", "reset"],
        coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]],
        dt=2.2222222222222221e-10,
        seed=7,
        noise_info=True,
    )
    qc = QuantumCircuit(3)
    qc.cx(0, 2)

    resolved_spec = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "shots",
            "backend_profile": "generic_seeded",
            "shots": 128,
            "schedule_policy": "asap",
            "layout_policy": "fixed_patch",
            "fixed_physical_patch": [2, 3, 4],
            "seed_transpiler": 7,
            "seed_simulator": 7,
        }
    )
    snapshot = freeze_backend_snapshot(backend)
    artifact = build_shots_only_artifact(
        circuit=qc,
        observable=None,
        resolved_spec=resolved_spec,
        resolved_backend=backend,
        calibration_snapshot=snapshot,
    )
    tx = artifact.transpile_snapshot
    assert tx is not None
    assert set(tx.used_physical_qubits).issubset({2, 3, 4})
    assert tx.used_physical_edges
    assert all(edge[0] in {2, 3, 4} and edge[1] in {2, 3, 4} for edge in tx.used_physical_edges)
