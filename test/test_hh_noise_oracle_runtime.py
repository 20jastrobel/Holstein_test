from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

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
import pipelines.exact_bench.noise_aer_builders as nab
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
from src.quantum.time_propagation.cfqm_propagator import cfqm_step
from src.quantum.time_propagation.cfqm_schemes import get_cfqm_scheme
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, HubbardHolsteinTermwiseAnsatz, apply_exp_pauli_polynomial


def _generic_backend_2q():
    from qiskit.providers.fake_provider import GenericBackendV2

    return GenericBackendV2(
        2,
        basis_gates=["id", "rz", "sx", "x", "cx", "measure", "delay", "reset"],
        coupling_map=[[0, 1], [1, 0]],
        dt=2.2222222222222221e-10,
        seed=7,
        noise_info=True,
    )


def _patch_runtime_stack(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    import qiskit_ibm_runtime as qir

    captured: dict[str, object] = {}
    backend = _generic_backend_2q()
    captured["backend"] = backend

    class _FakeRuntimeService:
        def backend(self, name: str) -> object:
            captured["backend_name"] = str(name)
            return backend

    class _FakeSession:
        def __init__(self, *, service: object, backend: object):
            self.service = service
            self.backend = backend
            self.closed = False
            captured["session"] = self

        def close(self) -> None:
            self.closed = True

    class _FakeJob:
        def __init__(self, pubs):
            self._pubs = pubs

        def result(self):
            class _Result:
                values = np.asarray([0.125], dtype=float)

            return _Result()

    class _FakeEstimator:
        def __init__(self, mode=None, options=None):
            self.mode = mode
            self.options = options
            captured["estimator"] = self
            captured["options"] = options

        def run(self, pubs):
            captured.setdefault("run_calls", []).append({"estimator": self, "pubs": pubs})
            return _FakeJob(pubs)

    monkeypatch.setattr(qir, "QiskitRuntimeService", _FakeRuntimeService)
    monkeypatch.setattr(qir, "Session", _FakeSession)
    monkeypatch.setattr(qir, "EstimatorV2", _FakeEstimator)
    return captured


def _write_runtime_layer_noise_model_json(tmp_path: Path, name: str, obj: object) -> tuple[Path, str]:
    from qiskit_ibm_runtime.utils.json import RuntimeEncoder

    path = tmp_path / name
    payload = json.dumps(obj, cls=RuntimeEncoder)
    path.write_text(payload, encoding="utf-8")
    return path, payload


def _circuit_signature(qc: QuantumCircuit) -> list[tuple[str, tuple[int, ...], tuple[int, ...], tuple[str, ...]]]:
    rows: list[tuple[str, tuple[int, ...], tuple[int, ...], tuple[str, ...]]] = []
    for inst in qc.data:
        rows.append(
            (
                str(inst.operation.name),
                tuple(int(qc.find_bit(q).index) for q in inst.qubits),
                tuple(int(qc.find_bit(c).index) for c in inst.clbits),
                tuple(str(param) for param in list(getattr(inst.operation, "params", []))),
            )
        )
    return rows


def _observable_signature(obs: SparsePauliOp) -> list[tuple[str, complex]]:
    return [(str(label), complex(coeff)) for label, coeff in obs.to_list()]


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
        cfqm_stage_exp="pauli_suzuki2",
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
        cfqm_stage_exp="pauli_suzuki2",
    )

    assert suzuki.num_qubits == 2
    assert cfqm.num_qubits == 2
    assert suzuki.depth() >= initial.depth()
    assert cfqm.depth() >= initial.depth()
    assert Statevector.from_instruction(suzuki_zero).equiv(Statevector.from_instruction(initial))
    assert Statevector.from_instruction(cfqm_zero).equiv(Statevector.from_instruction(initial))


def test_cfqm_circuit_builder_matches_numerical_pauli_suzuki2_stage_order() -> None:
    ordered_labels_exyz = ["x", "z"]
    static_coeff_map_exyz = {"z": 0.4 + 0.0j}

    def _drive_provider(time_value: float) -> dict[str, complex]:
        return {"x": complex(0.7 + (0.3 * float(time_value)))}

    qc = shared_build_cfqm_time_dependent_circuit(
        method="cfqm4",
        initial_circuit=QuantumCircuit(1),
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        drive_provider_exyz=_drive_provider,
        time_value=1.0,
        trotter_steps=1,
        drive_t0=0.0,
        coeff_drop_abs_tol=0.0,
        cfqm_stage_exp="pauli_suzuki2",
    )
    psi_circuit = np.asarray(Statevector.from_instruction(qc).data, dtype=complex).reshape(-1)
    psi_numeric = cfqm_step(
        psi=np.asarray([1.0, 0.0], dtype=complex),
        t_abs=0.0,
        dt=1.0,
        static_coeff_map=dict(static_coeff_map_exyz),
        drive_coeff_provider=_drive_provider,
        ordered_labels=list(ordered_labels_exyz),
        scheme=get_cfqm_scheme("cfqm4"),
        config=SimpleNamespace(
            backend="pauli_suzuki2",
            scheme_name="cfqm4",
            emit_inner_order_warning=False,
        ),
    )
    assert abs(np.vdot(psi_numeric, psi_circuit)) > 0.9998


def test_cfqm_circuit_builder_rejects_numerical_only_stage_exp_for_circuitization() -> None:
    initial = QuantumCircuit(1)
    with pytest.raises(ValueError, match="pauli_suzuki2"):
        shared_build_cfqm_time_dependent_circuit(
            method="cfqm4",
            initial_circuit=initial,
            ordered_labels_exyz=["z"],
            static_coeff_map_exyz={"z": 0.5 + 0.0j},
            drive_provider_exyz=None,
            time_value=0.2,
            trotter_steps=1,
            drive_t0=0.0,
            coeff_drop_abs_tol=0.0,
            cfqm_stage_exp="dense_expm",
        )


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


def test_evaluate_result_exposes_canonical_execution_snapshot() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])

    with ExpectationOracle(OracleConfig(noise_mode="ideal")) as oracle:
        result = oracle.evaluate_result(qc, obs)
        result.execution.provenance_summary["classification"] = "mutated"
        result.estimate.raw_values.append(123.0)
        current_execution = oracle.current_execution
        last_result = oracle.last_result

    assert result.estimate.mean == pytest.approx(
        float(np.real(Statevector.from_instruction(qc).expectation_value(obs))),
        abs=1e-10,
    )
    assert result.execution.noise_mode == "ideal"
    assert result.execution.estimator_kind == "qiskit.primitives.StatevectorEstimator"
    assert result.execution.resolved_noise_spec["executor"] == "statevector"
    assert (
        result.execution.provenance_summary["classification"]
        == "mutated"
    )
    assert current_execution.provenance_summary["classification"] == "ideal_statevector_control"
    assert last_result is not None
    assert last_result.execution.provenance_summary["classification"] == "ideal_statevector_control"
    assert last_result.estimate.raw_values == [0.0]
    assert oracle.current_execution == current_execution
    assert oracle.last_result == last_result
    assert oracle.backend_info.details.get("resolved_noise_spec") == current_execution.resolved_noise_spec
    assert (
        oracle.backend_info.details.get("symmetry_mitigation_config")
        == current_execution.symmetry_mitigation_config
    )


def test_mitigation_config_is_normalized() -> None:
    mit = normalize_mitigation_config({"mode": "zne", "zne_scales": "1.0,2.0,3.0"})
    assert mit["mode"] == "zne"
    assert mit["zne_scales"] == [1.0, 2.0, 3.0]
    assert mit["dd_sequence"] is None


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


@pytest.mark.parametrize("noise_mode", ["runtime", "qpu_raw"])
def test_ideal_reference_symmetry_mitigation_downgrades_runtime_like_modes_to_verify_only(
    noise_mode: str,
) -> None:
    cfg = normalize_ideal_reference_symmetry_mitigation(
        {
            "mode": "projector_renorm_v1",
            "num_sites": 2,
            "ordering": "blocked",
            "sector_n_up": 1,
            "sector_n_dn": 1,
        },
        noise_mode=str(noise_mode),
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


def test_freeze_backend_snapshot_falls_back_to_target_metadata() -> None:
    backend = _generic_backend_2q()
    snapshot = freeze_backend_snapshot(backend)

    assert snapshot.per_qubit
    assert snapshot.per_gate
    assert any(rec.gate_name == "cx" and rec.qubits == [0, 1] and rec.duration_s is not None for rec in snapshot.per_gate)
    assert snapshot.per_qubit[0].T1_s is not None
    assert snapshot.per_qubit[0].measure_duration_s is not None
    assert snapshot.median_2q_error is not None
    assert snapshot.median_readout_error is not None


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


def test_frozen_snapshot_backend_basic_replay_is_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("qiskit_aer")

    backend = _generic_backend_2q()
    snapshot = freeze_backend_snapshot(backend)
    snapshot_path = tmp_path / "backend_basic_snapshot.json"
    nsnap.write_calibration_snapshot(snapshot_path, snapshot)

    qc = QuantumCircuit(1)
    qc.x(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    cfg = OracleConfig(
        noise_mode="backend_basic",
        backend_profile="frozen_snapshot_json",
        noise_snapshot_json=str(snapshot_path),
        shots=256,
        seed=111,
        oracle_repeats=4,
        oracle_aggregate="mean",
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
    assert est_a.stderr == pytest.approx(est_b.stderr)
    assert meta_a.get("resolved_noise_spec", {}).get("backend_profile_kind") == "frozen_snapshot_json"
    assert meta_a.get("resolved_noise_spec", {}).get("noise_kind") == "backend_basic"
    assert meta_a.get("snapshot_hash") == snapshot.snapshot_hash
    assert meta_a.get("snapshot_hash") == meta_b.get("snapshot_hash")
    assert meta_a.get("source_kind") == snapshot.source_kind
    assert meta_a.get("layout_hash") == meta_b.get("layout_hash")
    assert meta_a.get("noise_artifact_hash") == meta_b.get("noise_artifact_hash")


def test_frozen_snapshot_backend_scheduled_replay_records_schedule_metadata(tmp_path: Path) -> None:
    pytest.importorskip("qiskit_aer")

    backend = _generic_backend_2q()
    snapshot = freeze_backend_snapshot(backend)
    snapshot_path = tmp_path / "backend_scheduled_snapshot.json"
    nsnap.write_calibration_snapshot(snapshot_path, snapshot)

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            backend_profile="frozen_snapshot_json",
            noise_snapshot_json=str(snapshot_path),
            shots=128,
            seed=19,
            oracle_repeats=2,
            oracle_aggregate="mean",
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        meta = dict(oracle.backend_info.details)

    assert np.isfinite(est.mean)
    assert meta.get("resolved_noise_spec", {}).get("noise_kind") == "backend_scheduled"
    assert meta.get("provenance_summary", {}).get("classification") == "local_snapshot_replay"
    assert meta.get("snapshot_hash") == snapshot.snapshot_hash
    assert meta.get("scheduled_duration_total") is not None
    assert meta.get("transpile_hash")
    assert meta.get("layout_hash")


def test_backend_scheduled_evaluation_does_not_mutate_time_dependent_input_circuit(tmp_path: Path) -> None:
    pytest.importorskip("qiskit_aer")

    backend = _generic_backend_2q()
    snapshot = freeze_backend_snapshot(backend)
    snapshot_path = tmp_path / "backend_scheduled_no_mutation_snapshot.json"
    nsnap.write_calibration_snapshot(snapshot_path, snapshot)

    initial = QuantumCircuit(2)
    initial.x(0)
    qc = shared_build_suzuki2_time_dependent_circuit(
        initial_circuit=initial,
        ordered_labels_exyz=["zz"],
        static_coeff_map_exyz={"zz": 1.0 + 0.0j},
        drive_provider_exyz=None,
        time_value=0.3,
        trotter_steps=2,
        drive_t0=0.0,
        drive_time_sampling="midpoint",
    )
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    noisy_cfg = OracleConfig(
        noise_mode="backend_scheduled",
        backend_profile="frozen_snapshot_json",
        noise_snapshot_json=str(snapshot_path),
        schedule_policy="asap",
        shots=128,
        seed=23,
        oracle_repeats=1,
        oracle_aggregate="mean",
        layout_lock_key="backend_scheduled_no_mutation",
    )
    ideal_cfg = OracleConfig(
        noise_mode="ideal",
        shots=128,
        seed=23,
        oracle_repeats=1,
        oracle_aggregate="mean",
    )

    with ExpectationOracle(noisy_cfg) as noisy_oracle, ExpectationOracle(ideal_cfg) as ideal_oracle:
        noisy_oracle.prime_layout(initial)
        noisy_est_1 = noisy_oracle.evaluate(qc, obs)
        ideal_est_1 = ideal_oracle.evaluate(qc, obs)
        noisy_est_2 = noisy_oracle.evaluate(qc, obs)
        ideal_est_2 = ideal_oracle.evaluate(qc, obs)

    assert np.isfinite(float(noisy_est_1.mean))
    assert np.isfinite(float(ideal_est_1.mean))
    assert np.isfinite(float(noisy_est_2.mean))
    assert np.isfinite(float(ideal_est_2.mean))
    assert len(qc.parameters) == 0
    for inst in qc.data:
        for param in list(getattr(inst.operation, "params", [])):
            assert "__noise_tpl_" not in str(param)


def test_patch_snapshot_requires_calibration_snapshot_path() -> None:
    with pytest.raises(ValueError, match="backend_profile_kind='frozen_snapshot_json' requires noise_snapshot_json/snapshot_path"):
        ExpectationOracle(
            OracleConfig(
                noise_mode="patch_snapshot",
                backend_profile="frozen_snapshot_json",
                shots=128,
                seed=13,
            )
        )


def test_patch_snapshot_requires_persisted_layout_replay_data(tmp_path: Path) -> None:
    pytest.importorskip("qiskit_aer")

    backend = _generic_backend_2q()
    snapshot = freeze_backend_snapshot(backend)
    snapshot_path = tmp_path / "patch_snapshot.json"
    nsnap.write_calibration_snapshot(snapshot_path, snapshot)

    qc = QuantumCircuit(1)
    qc.x(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="patch_snapshot",
            backend_profile="frozen_snapshot_json",
            noise_snapshot_json=str(snapshot_path),
            schedule_policy="asap",
            shots=128,
            seed=13,
            layout_lock_key="patch_snapshot_missing_lock",
        )
    ) as oracle:
        with pytest.raises(RuntimeError, match="patch_snapshot requires persisted patch/layout replay data"):
            _ = oracle.evaluate(qc, obs)


def test_frozen_layout_without_persisted_lock_fails_explicitly() -> None:
    pytest.importorskip("qiskit_aer")

    qc = QuantumCircuit(1)
    qc.x(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="shots",
            backend_profile="generic_seeded",
            layout_policy="frozen_layout",
            layout_lock_key="first_run_frozen_layout_missing_lock",
            shots=128,
            seed=17,
        )
    ) as oracle:
        with pytest.raises(RuntimeError, match="layout_policy='frozen_layout' requires an existing persisted layout lock"):
            _ = oracle.evaluate(qc, obs)


def test_patch_snapshot_replay_does_not_overwrite_captured_layout_lock(tmp_path: Path) -> None:
    pytest.importorskip("qiskit_aer")

    backend = _generic_backend_2q()
    snapshot = freeze_backend_snapshot(backend)
    snapshot_path = tmp_path / "patch_snapshot_read_only.json"
    nsnap.write_calibration_snapshot(snapshot_path, snapshot)

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])
    capture_cfg = OracleConfig(
        noise_mode="backend_scheduled",
        backend_profile="frozen_snapshot_json",
        noise_snapshot_json=str(snapshot_path),
        schedule_policy="asap",
        layout_lock_key="patch_snapshot_read_only_lock",
        shots=128,
        seed=23,
    )
    replay_cfg = OracleConfig(
        noise_mode="patch_snapshot",
        backend_profile="frozen_snapshot_json",
        noise_snapshot_json=str(snapshot_path),
        schedule_policy="asap",
        layout_lock_key="patch_snapshot_read_only_lock",
        shots=128,
        seed=23,
    )

    with ExpectationOracle(capture_cfg) as oracle:
        _ = oracle.evaluate(qc, obs)

    capture_spec = normalize_to_resolved_noise_spec(capture_cfg)
    lock_key = nab._layout_lock_registry_key(capture_spec, snapshot)
    before = nab._load_persisted_layout_lock(lock_key)
    assert before is not None

    with ExpectationOracle(replay_cfg) as oracle:
        _ = oracle.evaluate(qc, obs)

    after = nab._load_persisted_layout_lock(lock_key)
    assert after == before


@pytest.mark.parametrize("noise_mode", ["runtime", "qpu_raw"])
def test_runtime_symmetry_mitigation_downgrades_to_verify_only_when_counts_path_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    noise_mode: str,
) -> None:
    _patch_runtime_stack(monkeypatch)
    qc = QuantumCircuit(2)
    qc.x(0)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode=str(noise_mode),
            backend_name="ibm_fake_runtime",
            mitigation="none",
            symmetry_mitigation={
                "mode": "postselect_diag_v1",
                "num_sites": 1,
                "ordering": "blocked",
                "sector_n_up": 1,
                "sector_n_dn": 0,
            },
        )
    ) as oracle:
        out = oracle._maybe_evaluate_symmetry_mitigated(qc, obs)
        sym = dict(oracle.backend_info.details.get("symmetry_mitigation", {}))
        warnings = list(oracle.backend_info.details.get("warnings", []))

    assert out is None
    assert sym.get("applied_mode") == "verify_only"
    assert sym.get("fallback_reason") == "runtime_counts_path_unavailable"
    assert any("symmetry_mitigation_downgraded:runtime_counts_path_unavailable" in str(w) for w in warnings)


def test_runtime_mitigation_none_explicitly_disables_runtime_suppression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    estimator, session, info, local_ctx = nor._build_estimator(
        OracleConfig(
            noise_mode="runtime",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation="none",
        )
    )

    opts = captured["options"]
    assert estimator is captured["estimator"]
    assert session is captured["session"]
    assert local_ctx is not None
    assert local_ctx.get("resolved_backend") is captured["backend"]
    assert local_ctx.get("calibration_snapshot") is not None
    assert info.details.get("snapshot_hash") == local_ctx["calibration_snapshot"].snapshot_hash
    assert opts.resilience_level == 0
    assert opts.resilience.measure_mitigation is False
    assert opts.resilience.zne_mitigation is False
    assert opts.dynamical_decoupling.enable is False
    assert opts.twirling.enable_gates is False
    assert opts.twirling.enable_measure is False
    assert info.details.get("runtime_mitigation_options") == {
        "mode": "none",
        "resilience_level": 0,
        "measure_mitigation": False,
        "zne_mitigation": False,
        "zne_noise_factors": [],
        "zne_amplifier": None,
        "dynamical_decoupling_enable": False,
        "dynamical_decoupling_sequence_type": None,
        "layer_noise_learning_requested": False,
        "layer_noise_model_supplied": False,
        "layer_noise_model_source": None,
        "layer_noise_model_kind": None,
        "layer_noise_model_entry_count": None,
        "layer_noise_model_fingerprint": None,
        "implicit_gate_twirling_via_pea": False,
        "twirling_enable_gates": False,
        "twirling_enable_measure": False,
        "twirling_num_randomizations": None,
        "twirling_strategy": None,
        "trex_like_measure_suppression": False,
    }
    assert info.details.get("runtime_execution_bundle") == {
        "requested_noise_mode": "runtime",
        "resolved_noise_kind": "qpu_raw",
        "mitigation_bundle": "none",
        "mitigation_mode": "none",
        "twirling_enable_gates": False,
        "twirling_enable_measure": False,
        "twirling_num_randomizations": None,
        "twirling_strategy": None,
        "trex_like_measure_suppression": False,
        "zne_amplifier": None,
        "layer_noise_learning_requested": False,
        "layer_noise_model_supplied": False,
        "layer_noise_model_source": None,
        "layer_noise_model_kind": None,
        "layer_noise_model_entry_count": None,
        "layer_noise_model_fingerprint": None,
        "implicit_gate_twirling_via_pea": False,
        "suppression_components": [],
    }


def test_qpu_raw_maps_to_explicit_runtime_raw_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    _, _, info, _ = nor._build_estimator(
        OracleConfig(
            noise_mode="qpu_raw",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation="none",
        )
    )

    opts = captured["options"]
    assert opts.resilience_level == 0
    assert opts.resilience.measure_mitigation is False
    assert opts.resilience.zne_mitigation is False
    assert opts.dynamical_decoupling.enable is False
    assert info.details.get("runtime_execution_bundle") == {
        "requested_noise_mode": "qpu_raw",
        "resolved_noise_kind": "qpu_raw",
        "mitigation_bundle": "none",
        "mitigation_mode": "none",
        "twirling_enable_gates": False,
        "twirling_enable_measure": False,
        "twirling_num_randomizations": None,
        "twirling_strategy": None,
        "trex_like_measure_suppression": False,
        "zne_amplifier": None,
        "layer_noise_learning_requested": False,
        "layer_noise_model_supplied": False,
        "layer_noise_model_source": None,
        "layer_noise_model_kind": None,
        "layer_noise_model_entry_count": None,
        "layer_noise_model_fingerprint": None,
        "implicit_gate_twirling_via_pea": False,
        "suppression_components": [],
    }


def test_runtime_mitigation_readout_maps_to_measure_mitigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    _, _, info, _ = nor._build_estimator(
        OracleConfig(
            noise_mode="runtime",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation="readout",
        )
    )

    opts = captured["options"]
    assert opts.resilience.measure_mitigation is True
    assert opts.resilience.zne_mitigation is False
    assert opts.dynamical_decoupling.enable is False
    assert opts.twirling.enable_gates is False
    assert opts.twirling.enable_measure is False
    assert info.details.get("runtime_mitigation_options", {}).get("measure_mitigation") is True
    assert info.details.get("runtime_execution_bundle") == {
        "requested_noise_mode": "runtime",
        "resolved_noise_kind": "qpu_suppressed",
        "mitigation_bundle": "runtime_suppressed",
        "mitigation_mode": "readout",
        "twirling_enable_gates": False,
        "twirling_enable_measure": False,
        "twirling_num_randomizations": None,
        "twirling_strategy": None,
        "trex_like_measure_suppression": False,
        "zne_amplifier": None,
        "layer_noise_learning_requested": False,
        "layer_noise_model_supplied": False,
        "layer_noise_model_source": None,
        "layer_noise_model_kind": None,
        "layer_noise_model_entry_count": None,
        "layer_noise_model_fingerprint": None,
        "implicit_gate_twirling_via_pea": False,
        "suppression_components": ["measure_mitigation"],
    }


def test_runtime_gate_twirling_maps_to_runtime_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    _, _, info, _ = nor._build_estimator(
        OracleConfig(
            noise_mode="runtime",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation="none",
            runtime_twirling={
                "enable_gates": True,
                "num_randomizations": 16,
                "strategy": "active",
            },
        )
    )

    opts = captured["options"]
    assert opts.twirling.enable_gates is True
    assert opts.twirling.enable_measure is False
    assert int(opts.twirling.num_randomizations) == 16
    assert str(opts.twirling.strategy) == "active"
    assert info.details.get("runtime_execution_bundle", {}).get("resolved_noise_kind") == "qpu_suppressed"
    assert info.details.get("runtime_execution_bundle", {}).get("suppression_components") == ["gate_twirling"]
    assert info.details.get("runtime_mitigation_options", {}).get("twirling_enable_gates") is True
    assert info.details.get("provenance_summary", {}).get("runtime_gate_twirling") is True


def test_runtime_measure_twirling_with_readout_maps_to_trex_like_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    _, _, info, _ = nor._build_estimator(
        OracleConfig(
            noise_mode="runtime",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation="readout",
            runtime_twirling={
                "enable_measure": True,
                "num_randomizations": 8,
                "strategy": "active",
            },
        )
    )

    opts = captured["options"]
    assert opts.twirling.enable_gates is False
    assert opts.twirling.enable_measure is True
    assert int(opts.twirling.num_randomizations) == 8
    assert str(opts.twirling.strategy) == "active"
    assert opts.resilience.measure_mitigation is True
    assert info.details.get("runtime_mitigation_options", {}).get("trex_like_measure_suppression") is True
    assert info.details.get("runtime_execution_bundle", {}).get("trex_like_measure_suppression") is True
    assert info.details.get("runtime_execution_bundle", {}).get("suppression_components") == [
        "measure_mitigation",
        "measure_twirling",
        "trex_like_readout",
    ]
    assert info.details.get("provenance_summary", {}).get("runtime_measure_twirling") is True
    assert info.details.get("provenance_summary", {}).get("runtime_trex_like_measure_suppression") is True


def test_qpu_suppressed_maps_to_supported_runtime_option_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    _, _, info, _ = nor._build_estimator(
        OracleConfig(
            noise_mode="qpu_suppressed",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation="readout",
        )
    )

    opts = captured["options"]
    assert opts.resilience.measure_mitigation is True
    assert opts.resilience.zne_mitigation is False
    assert opts.dynamical_decoupling.enable is False
    assert info.details.get("runtime_execution_bundle") == {
        "requested_noise_mode": "qpu_suppressed",
        "resolved_noise_kind": "qpu_suppressed",
        "mitigation_bundle": "runtime_suppressed",
        "mitigation_mode": "readout",
        "twirling_enable_gates": False,
        "twirling_enable_measure": False,
        "twirling_num_randomizations": None,
        "twirling_strategy": None,
        "trex_like_measure_suppression": False,
        "zne_amplifier": None,
        "layer_noise_learning_requested": False,
        "layer_noise_model_supplied": False,
        "layer_noise_model_source": None,
        "layer_noise_model_kind": None,
        "layer_noise_model_entry_count": None,
        "layer_noise_model_fingerprint": None,
        "implicit_gate_twirling_via_pea": False,
        "suppression_components": ["measure_mitigation"],
    }


def test_runtime_submission_uses_transpiled_isa_circuit_and_mapped_observable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="qpu_raw",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation="none",
            schedule_policy="asap",
            layout_policy="fixed_patch",
            fixed_physical_patch=[1, 0],
            seed_transpiler=7,
            layout_lock_key="runtime_fixed_patch_submission",
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        details = dict(oracle.backend_info.details)

    submitted_circuit, submitted_observable = captured["run_calls"][-1]["pubs"][0]
    assert est.mean == pytest.approx(0.125)
    assert submitted_circuit is not qc
    assert submitted_observable is not obs
    assert getattr(submitted_circuit, "layout", None) is not None
    assert details.get("snapshot_hash")
    assert details.get("transpile_hash")
    assert details.get("layout_hash")
    assert details.get("provenance_summary", {}).get("classification") == "runtime_submitted_raw"
    assert details.get("layout_anchor_source") == "fixed_patch"
    assert details.get("used_physical_qubits")
    assert details.get("used_physical_edges")
    assert details.get("transpile_snapshot", {}).get("initial_layout") == [1, 0]
    assert details.get("runtime_execution_bundle", {}).get("resolved_noise_kind") == "qpu_raw"


def test_runtime_no_snapshot_layout_lock_key_still_distinguishes_backend_name() -> None:
    spec_a = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "qpu_raw",
            "backend_name": "ibm_backend_a",
            "mitigation": "none",
            "layout_lock_key": "runtime_missing_snapshot_lock_domain",
        }
    )
    spec_b = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "qpu_raw",
            "backend_name": "ibm_backend_b",
            "mitigation": "none",
            "layout_lock_key": "runtime_missing_snapshot_lock_domain",
        }
    )

    assert nab._layout_lock_registry_key(spec_a, None) != nab._layout_lock_registry_key(spec_b, None)



def test_qpu_raw_and_qpu_suppressed_reuse_same_runtime_submission_under_shared_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])
    shared_lock_key = "runtime_shared_submission_lock"
    shared_kwargs = {
        "backend_name": "ibm_fake_runtime",
        "shots": 128,
        "schedule_policy": "asap",
        "layout_policy": "auto_then_lock",
        "layout_lock_key": shared_lock_key,
        "seed_transpiler": 11,
    }

    with ExpectationOracle(
        OracleConfig(
            noise_mode="runtime",
            mitigation="none",
            **shared_kwargs,
        )
    ) as oracle:
        oracle.prime_layout(qc)

    captured["run_calls"] = []

    with ExpectationOracle(
        OracleConfig(
            noise_mode="qpu_raw",
            mitigation="none",
            **shared_kwargs,
        )
    ) as raw_oracle:
        _ = raw_oracle.evaluate(qc, obs)
        raw_details = dict(raw_oracle.backend_info.details)

    with ExpectationOracle(
        OracleConfig(
            noise_mode="qpu_suppressed",
            mitigation="readout",
            **shared_kwargs,
        )
    ) as suppressed_oracle:
        _ = suppressed_oracle.evaluate(qc, obs)
        suppressed_details = dict(suppressed_oracle.backend_info.details)

    raw_circuit, raw_observable = captured["run_calls"][-2]["pubs"][0]
    suppressed_circuit, suppressed_observable = captured["run_calls"][-1]["pubs"][0]

    assert raw_details.get("layout_anchor_source") == "persisted_lock"
    assert suppressed_details.get("layout_anchor_source") == "persisted_lock"
    assert raw_details.get("snapshot_hash") == suppressed_details.get("snapshot_hash")
    assert raw_details.get("layout_hash") == suppressed_details.get("layout_hash")
    assert raw_details.get("transpile_hash") == suppressed_details.get("transpile_hash")
    assert raw_details.get("used_physical_qubits") == suppressed_details.get("used_physical_qubits")
    assert raw_details.get("used_physical_edges") == suppressed_details.get("used_physical_edges")
    assert _circuit_signature(raw_circuit) == _circuit_signature(suppressed_circuit)
    assert _observable_signature(raw_observable) == _observable_signature(suppressed_observable)
    assert raw_details.get("runtime_execution_bundle", {}).get("mitigation_mode") == "none"
    assert suppressed_details.get("runtime_execution_bundle", {}).get("mitigation_mode") == "readout"


def test_runtime_reuses_layout_from_local_backend_scheduled_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    backend = _generic_backend_2q()
    snapshot = freeze_backend_snapshot(backend)
    snapshot_path = tmp_path / "runtime_local_bridge_snapshot.json"
    nsnap.write_calibration_snapshot(snapshot_path, snapshot)

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])
    shared_lock_key = "runtime_local_bridge_lock"

    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            backend_profile="frozen_snapshot_json",
            noise_snapshot_json=str(snapshot_path),
            schedule_policy="asap",
            layout_policy="auto_then_lock",
            layout_lock_key=shared_lock_key,
            shots=128,
            seed=29,
        )
    ) as local_oracle:
        _ = local_oracle.evaluate(qc, obs)
        local_details = dict(local_oracle.backend_info.details)

    captured["run_calls"] = []

    with ExpectationOracle(
        OracleConfig(
            noise_mode="qpu_raw",
            backend_name="ibm_fake_runtime",
            mitigation="none",
            schedule_policy="asap",
            layout_policy="auto_then_lock",
            layout_lock_key=shared_lock_key,
            shots=128,
            seed=29,
            seed_transpiler=29,
        )
    ) as runtime_oracle:
        _ = runtime_oracle.evaluate(qc, obs)
        runtime_details = dict(runtime_oracle.backend_info.details)

    submitted_circuit, submitted_observable = captured["run_calls"][-1]["pubs"][0]
    assert getattr(submitted_circuit, "layout", None) is not None
    assert submitted_observable is not obs
    assert runtime_details.get("layout_anchor_source") == "persisted_lock"
    assert runtime_details.get("snapshot_hash") == local_details.get("snapshot_hash")
    assert runtime_details.get("layout_hash") == local_details.get("layout_hash")
    assert runtime_details.get("used_physical_qubits") == local_details.get("used_physical_qubits")
    assert runtime_details.get("used_physical_edges") == local_details.get("used_physical_edges")



def test_runtime_frozen_layout_without_persisted_lock_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    qc = QuantumCircuit(1)
    qc.x(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="qpu_raw",
            backend_name="ibm_fake_runtime",
            mitigation="none",
            schedule_policy="asap",
            layout_policy="frozen_layout",
            layout_lock_key="runtime_missing_frozen_layout_lock",
            shots=128,
        )
    ) as oracle:
        with pytest.raises(RuntimeError, match="layout_policy='frozen_layout' requires an existing persisted layout lock"):
            _ = oracle.evaluate(qc, obs)

    assert not captured.get("run_calls")


def test_runtime_mitigation_zne_maps_to_runtime_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    _, _, info, _ = nor._build_estimator(
        OracleConfig(
            noise_mode="runtime",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation={"mode": "zne", "zne_scales": [1.0, 2.0, 3.0]},
        )
    )

    opts = captured["options"]
    assert opts.resilience.measure_mitigation is False
    assert opts.resilience.zne_mitigation is True
    assert list(opts.resilience.zne.noise_factors) == [1.0, 2.0, 3.0]
    assert opts.dynamical_decoupling.enable is False
    assert info.details.get("runtime_mitigation_options", {}).get("zne_noise_factors") == [1.0, 2.0, 3.0]


def test_runtime_mitigation_dd_maps_to_runtime_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    _, _, info, _ = nor._build_estimator(
        OracleConfig(
            noise_mode="runtime",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation={"mode": "dd", "dd_sequence": "XY4"},
        )
    )

    opts = captured["options"]
    assert opts.resilience.measure_mitigation is False
    assert opts.resilience.zne_mitigation is False
    assert opts.dynamical_decoupling.enable is True
    assert str(opts.dynamical_decoupling.sequence_type) == "XY4"
    assert opts.twirling.enable_gates is False
    assert opts.twirling.enable_measure is False
    assert info.details.get("runtime_mitigation_options", {}).get("dynamical_decoupling_sequence_type") == "XY4"


@pytest.mark.parametrize(
    "noise_mode, mitigation",
    [
        ("ideal", "readout"),
        ("ideal", {"mode": "zne", "zne_scales": [1.0, 2.0]}),
        ("ideal", {"mode": "dd", "dd_sequence": "XY4"}),
        ("shots", "readout"),
        ("shots", {"mode": "zne", "zne_scales": [1.0, 2.0]}),
        ("shots", {"mode": "dd", "dd_sequence": "XY4"}),
    ],
)
def test_non_runtime_mitigation_modes_fail_explicitly(
    noise_mode: str,
    mitigation: object,
) -> None:
    kwargs = {"noise_mode": str(noise_mode), "mitigation": mitigation}
    if str(noise_mode) == "shots":
        kwargs["shots"] = 64
    with pytest.raises(RuntimeError, match="only executable on runtime_qpu"):
        nor._build_estimator(OracleConfig(**kwargs))


@pytest.mark.parametrize("noise_mode", ["ideal", "shots"])
def test_non_runtime_twirling_requests_fail_explicitly(noise_mode: str) -> None:
    kwargs = {
        "noise_mode": str(noise_mode),
        "runtime_twirling": {"enable_gates": True},
    }
    if str(noise_mode) == "shots":
        kwargs["shots"] = 64
    with pytest.raises(RuntimeError, match="Runtime twirling is only executable on runtime_qpu"):
        nor._build_estimator(OracleConfig(**kwargs))


def test_measure_twirling_requires_readout_mitigation() -> None:
    with pytest.raises(RuntimeError, match="Measurement twirling is only wired with mitigation='readout'"):
        nor._build_estimator(
            OracleConfig(
                noise_mode="runtime",
                backend_name="ibm_fake_runtime",
                shots=128,
                mitigation="none",
                runtime_twirling={"enable_measure": True},
            )
        )


def test_qpu_suppressed_requires_supported_runtime_bundle() -> None:
    with pytest.raises(ValueError, match=r"qpu_suppressed requires mitigation \{readout,zne,dd\}"):
        nor._build_estimator(
            OracleConfig(
                noise_mode="qpu_suppressed",
                backend_name="ibm_fake_runtime",
                shots=128,
                mitigation="none",
            )
        )


def test_qpu_layer_learned_maps_to_pea_backed_runtime_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    _, _, info, _ = nor._build_estimator(
        OracleConfig(
            noise_mode="qpu_layer_learned",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation={"mode": "zne", "zne_scales": [1.0, 2.0, 3.0]},
        )
    )

    opts = captured["options"]
    assert opts.resilience.measure_mitigation is False
    assert opts.resilience.zne_mitigation is True
    assert list(opts.resilience.zne.noise_factors) == [1.0, 2.0, 3.0]
    assert str(opts.resilience.zne.amplifier) == "pea"
    mitigation_details = dict(info.details.get("runtime_mitigation_options", {}))
    runtime_bundle = dict(info.details.get("runtime_execution_bundle", {}))
    provenance = dict(info.details.get("provenance_summary", {}))
    assert mitigation_details.get("zne_amplifier") == "pea"
    assert mitigation_details.get("layer_noise_learning_requested") is True
    assert mitigation_details.get("layer_noise_model_supplied") is False
    assert mitigation_details.get("layer_noise_model_source") == "runtime_service_learning"
    assert mitigation_details.get("layer_noise_model_kind") is None
    assert mitigation_details.get("layer_noise_model_entry_count") is None
    assert mitigation_details.get("layer_noise_model_fingerprint") is None
    assert mitigation_details.get("implicit_gate_twirling_via_pea") is True
    assert runtime_bundle.get("resolved_noise_kind") == "qpu_layer_learned"
    assert runtime_bundle.get("mitigation_bundle") == "runtime_layer_learned"
    assert runtime_bundle.get("zne_amplifier") == "pea"
    assert runtime_bundle.get("layer_noise_learning_requested") is True
    assert runtime_bundle.get("layer_noise_model_supplied") is False
    assert runtime_bundle.get("layer_noise_model_source") == "runtime_service_learning"
    assert runtime_bundle.get("layer_noise_model_kind") is None
    assert runtime_bundle.get("layer_noise_model_entry_count") is None
    assert runtime_bundle.get("layer_noise_model_fingerprint") is None
    assert runtime_bundle.get("implicit_gate_twirling_via_pea") is True
    assert runtime_bundle.get("twirling_enable_gates") is False
    assert runtime_bundle.get("suppression_components") == [
        "zne",
        "pea_amplifier",
        "layer_noise_learning",
        "implicit_gate_twirling_via_pea",
    ]
    assert provenance.get("classification") == "runtime_submitted_layer_learned"
    assert provenance.get("runtime_layer_noise_learning") is True
    assert provenance.get("runtime_layer_noise_model_supplied") is False
    assert provenance.get("runtime_layer_noise_model_source") == "runtime_service_learning"
    assert provenance.get("runtime_layer_noise_model_kind") is None
    assert provenance.get("runtime_layer_noise_model_fingerprint") is None
    assert provenance.get("runtime_zne_amplifier") == "pea"


def test_qpu_layer_learned_requires_zne_runtime_bundle() -> None:
    with pytest.raises(ValueError, match="qpu_layer_learned requires mitigation='zne'"):
        nor._build_estimator(
            OracleConfig(
                noise_mode="qpu_layer_learned",
                backend_name="ibm_fake_runtime",
                shots=128,
                mitigation="readout",
            )
        )


def test_qpu_layer_learned_accepts_external_layer_error_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qiskit_ibm_runtime.utils.noise_learner_result import LayerError

    captured = _patch_runtime_stack(monkeypatch)
    layer_noise_model = [LayerError(QuantumCircuit(2), [0, 1], None)]

    _, _, info, _ = nor._build_estimator(
        OracleConfig(
            noise_mode="qpu_layer_learned",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation={
                "mode": "zne",
                "zne_scales": [1.0, 2.0, 3.0],
                "layer_noise_model": layer_noise_model,
            },
        )
    )

    opts = captured["options"]
    mitigation_details = dict(info.details.get("runtime_mitigation_options", {}))
    runtime_bundle = dict(info.details.get("runtime_execution_bundle", {}))
    provenance = dict(info.details.get("provenance_summary", {}))
    assert opts.resilience.zne.amplifier == "pea"
    assert opts.resilience.layer_noise_model == layer_noise_model
    assert mitigation_details.get("layer_noise_learning_requested") is False
    assert mitigation_details.get("layer_noise_model_supplied") is True
    assert mitigation_details.get("layer_noise_model_source") == "programmatic_object"
    assert mitigation_details.get("layer_noise_model_kind") == "Sequence[LayerError]"
    assert mitigation_details.get("layer_noise_model_entry_count") == 1
    assert mitigation_details.get("layer_noise_model_fingerprint") is None
    assert runtime_bundle.get("layer_noise_learning_requested") is False
    assert runtime_bundle.get("layer_noise_model_supplied") is True
    assert runtime_bundle.get("layer_noise_model_source") == "programmatic_object"
    assert runtime_bundle.get("layer_noise_model_kind") == "Sequence[LayerError]"
    assert runtime_bundle.get("layer_noise_model_entry_count") == 1
    assert runtime_bundle.get("layer_noise_model_fingerprint") is None
    assert runtime_bundle.get("suppression_components") == [
        "zne",
        "pea_amplifier",
        "external_layer_noise_model",
        "implicit_gate_twirling_via_pea",
    ]
    assert provenance.get("classification") == "runtime_submitted_layer_learned"
    assert provenance.get("runtime_layer_noise_learning") is False
    assert provenance.get("runtime_layer_noise_model_supplied") is True
    assert provenance.get("runtime_layer_noise_model_source") == "programmatic_object"
    assert provenance.get("runtime_layer_noise_model_kind") == "Sequence[LayerError]"
    assert provenance.get("runtime_layer_noise_model_fingerprint") is None


def test_qpu_layer_learned_accepts_external_noise_learner_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qiskit_ibm_runtime.utils.noise_learner_result import LayerError, NoiseLearnerResult

    captured = _patch_runtime_stack(monkeypatch)
    layer_errors = [LayerError(QuantumCircuit(2), [0, 1], None)]
    layer_noise_model = NoiseLearnerResult(layer_errors, metadata={"source": "test"})

    _, _, info, _ = nor._build_estimator(
        OracleConfig(
            noise_mode="qpu_layer_learned",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation={
                "mode": "zne",
                "zne_scales": [1.0, 2.0, 3.0],
                "layer_noise_model": layer_noise_model,
            },
        )
    )

    opts = captured["options"]
    runtime_bundle = dict(info.details.get("runtime_execution_bundle", {}))
    assert opts.resilience.layer_noise_model is layer_noise_model
    assert runtime_bundle.get("layer_noise_model_supplied") is True
    assert runtime_bundle.get("layer_noise_model_source") == "programmatic_object"
    assert runtime_bundle.get("layer_noise_model_kind") == "NoiseLearnerResult"
    assert runtime_bundle.get("layer_noise_model_entry_count") == 1
    assert runtime_bundle.get("layer_noise_model_fingerprint") is None


def test_qpu_layer_learned_accepts_file_backed_layer_error_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from qiskit_ibm_runtime.utils.noise_learner_result import LayerError

    captured = _patch_runtime_stack(monkeypatch)
    layer_noise_model = [LayerError(QuantumCircuit(2), [0, 1], None)]
    model_path, payload = _write_runtime_layer_noise_model_json(
        tmp_path,
        "layer_errors.json",
        layer_noise_model,
    )
    expected_fingerprint = f"sha256:{nor.hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    _, _, info, _ = nor._build_estimator(
        OracleConfig(
            noise_mode="qpu_layer_learned",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation={
                "mode": "zne",
                "zne_scales": [1.0, 2.0, 3.0],
                "layer_noise_model_json": str(model_path),
            },
        )
    )

    opts = captured["options"]
    runtime_bundle = dict(info.details.get("runtime_execution_bundle", {}))
    provenance = dict(info.details.get("provenance_summary", {}))
    assert isinstance(opts.resilience.layer_noise_model, list)
    assert len(opts.resilience.layer_noise_model) == 1
    assert runtime_bundle.get("layer_noise_model_source") == "file_backed_json"
    assert runtime_bundle.get("layer_noise_model_kind") == "Sequence[LayerError]"
    assert runtime_bundle.get("layer_noise_model_entry_count") == 1
    assert runtime_bundle.get("layer_noise_model_fingerprint") == expected_fingerprint
    assert provenance.get("runtime_layer_noise_model_source") == "file_backed_json"
    assert provenance.get("runtime_layer_noise_model_kind") == "Sequence[LayerError]"
    assert provenance.get("runtime_layer_noise_model_fingerprint") == expected_fingerprint


def test_qpu_layer_learned_accepts_file_backed_noise_learner_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from qiskit_ibm_runtime.utils.noise_learner_result import LayerError, NoiseLearnerResult

    captured = _patch_runtime_stack(monkeypatch)
    layer_noise_model = NoiseLearnerResult(
        [LayerError(QuantumCircuit(2), [0, 1], None)],
        metadata={"source": "test"},
    )
    model_path, payload = _write_runtime_layer_noise_model_json(
        tmp_path,
        "noise_learner_result.json",
        layer_noise_model,
    )
    expected_fingerprint = f"sha256:{nor.hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    _, _, info, _ = nor._build_estimator(
        OracleConfig(
            noise_mode="qpu_layer_learned",
            backend_name="ibm_fake_runtime",
            shots=128,
            mitigation={
                "mode": "zne",
                "zne_scales": [1.0, 2.0, 3.0],
                "layer_noise_model_json": str(model_path),
            },
        )
    )

    opts = captured["options"]
    runtime_bundle = dict(info.details.get("runtime_execution_bundle", {}))
    assert opts.resilience.layer_noise_model.metadata == {"source": "test"}
    assert runtime_bundle.get("layer_noise_model_source") == "file_backed_json"
    assert runtime_bundle.get("layer_noise_model_kind") == "NoiseLearnerResult"
    assert runtime_bundle.get("layer_noise_model_entry_count") == 1
    assert runtime_bundle.get("layer_noise_model_fingerprint") == expected_fingerprint


def test_qpu_layer_learned_rejects_invalid_external_layer_noise_model_type() -> None:
    with pytest.raises(ValueError, match="layer_noise_model must be a NoiseLearnerResult or Sequence\\[LayerError\\]"):
        nor._build_estimator(
            OracleConfig(
                noise_mode="qpu_layer_learned",
                backend_name="ibm_fake_runtime",
                shots=128,
                mitigation={
                    "mode": "zne",
                    "zne_scales": [1.0, 2.0],
                    "layer_noise_model": {"not": "valid"},
                },
            )
        )


def test_qpu_layer_learned_rejects_invalid_layer_noise_model_json(
    tmp_path: Path,
) -> None:
    bad_path = tmp_path / "bad_layer_noise_model.json"
    bad_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to decode mitigation layer_noise_model_json"):
        nor._build_estimator(
            OracleConfig(
                noise_mode="qpu_layer_learned",
                backend_name="ibm_fake_runtime",
                shots=128,
                mitigation={
                    "mode": "zne",
                    "zne_scales": [1.0, 2.0],
                    "layer_noise_model_json": str(bad_path),
                },
            )
        )


def test_external_layer_noise_model_requires_qpu_layer_learned_mode() -> None:
    from qiskit_ibm_runtime.utils.noise_learner_result import LayerError

    with pytest.raises(ValueError, match="layer_noise_model / layer_noise_model_json are only supported on noise_mode='qpu_layer_learned'"):
        nor._build_estimator(
            OracleConfig(
                noise_mode="runtime",
                backend_name="ibm_fake_runtime",
                shots=128,
                mitigation={
                    "mode": "zne",
                    "zne_scales": [1.0, 2.0],
                    "layer_noise_model": [LayerError(QuantumCircuit(2), [0, 1], None)],
                },
            )
        )


def test_file_backed_external_layer_noise_model_requires_qpu_layer_learned_mode(
    tmp_path: Path,
) -> None:
    bad_path = tmp_path / "layer_noise_model.json"
    bad_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="layer_noise_model / layer_noise_model_json are only supported on noise_mode='qpu_layer_learned'"):
        nor._build_estimator(
            OracleConfig(
                noise_mode="runtime",
                backend_name="ibm_fake_runtime",
                shots=128,
                mitigation={
                    "mode": "zne",
                    "zne_scales": [1.0, 2.0],
                    "layer_noise_model_json": str(bad_path),
                },
            )
        )


def test_qpu_layer_learned_rejects_explicit_runtime_twirling() -> None:
    with pytest.raises(ValueError, match="does not accept explicit runtime twirling"):
        nor._build_estimator(
            OracleConfig(
                noise_mode="qpu_layer_learned",
                backend_name="ibm_fake_runtime",
                shots=128,
                mitigation={"mode": "zne", "zne_scales": [1.0, 2.0]},
                runtime_twirling={"enable_gates": True},
            )
        )


def test_runtime_zne_requires_non_empty_scales() -> None:
    with pytest.raises(ValueError, match="requires non-empty zne_scales"):
        nor._build_estimator(
            OracleConfig(
                noise_mode="runtime",
                backend_name="ibm_fake_runtime",
                shots=128,
                mitigation={"mode": "zne"},
            )
        )


def test_runtime_dd_requires_sequence() -> None:
    with pytest.raises(ValueError, match="requires dd_sequence"):
        nor._build_estimator(
            OracleConfig(
                noise_mode="runtime",
                backend_name="ibm_fake_runtime",
                shots=128,
                mitigation={"mode": "dd"},
            )
        )


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


def test_backend_scheduled_fixed_couplers_are_enforced_and_reported() -> None:
    pytest.importorskip("qiskit_aer")

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            backend_profile="generic_seeded",
            shots=128,
            seed=37,
            seed_transpiler=37,
            schedule_policy="asap",
            layout_policy="fixed_patch",
            fixed_physical_patch=[0, 1],
            fixed_couplers=[[0, 1], [1, 0]],
            layout_lock_key="fixed_couplers_local_backend_scheduled",
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        details = dict(oracle.backend_info.details)

    status = dict(details.get("fixed_couplers_status", {}))
    assert np.isfinite(est.mean)
    assert details.get("used_physical_edges")
    assert all(edge in [[0, 1], [1, 0]] for edge in details.get("used_physical_edges", []))
    assert details.get("provenance_summary", {}).get("classification") == "local_generic_aer_execution"
    assert status.get("requested") == [[0, 1], [1, 0]]
    assert status.get("enforced") is True
    assert status.get("enforcement_scope") == "transpile_target_subgraph"
    assert status.get("compatibility_mode") == "subset"
    assert status.get("verified_used_edges_subset") is True
    assert status.get("verified_used_qubits_subset") is True



def test_runtime_raw_and_suppressed_share_same_fixed_coupler_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_stack(monkeypatch)

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])
    shared_kwargs = {
        "backend_name": "ibm_fake_runtime",
        "shots": 128,
        "schedule_policy": "asap",
        "layout_policy": "fixed_patch",
        "fixed_physical_patch": [0, 1],
        "fixed_couplers": [[0, 1], [1, 0]],
        "seed_transpiler": 41,
        "layout_lock_key": "runtime_fixed_couplers_shared_submission",
    }

    with ExpectationOracle(
        OracleConfig(
            noise_mode="qpu_raw",
            mitigation="none",
            **shared_kwargs,
        )
    ) as raw_oracle:
        _ = raw_oracle.evaluate(qc, obs)
        raw_details = dict(raw_oracle.backend_info.details)

    with ExpectationOracle(
        OracleConfig(
            noise_mode="qpu_suppressed",
            mitigation="readout",
            **shared_kwargs,
        )
    ) as suppressed_oracle:
        _ = suppressed_oracle.evaluate(qc, obs)
        suppressed_details = dict(suppressed_oracle.backend_info.details)

    raw_circuit, raw_observable = captured["run_calls"][-2]["pubs"][0]
    suppressed_circuit, suppressed_observable = captured["run_calls"][-1]["pubs"][0]
    raw_status = dict(raw_details.get("fixed_couplers_status", {}))
    suppressed_status = dict(suppressed_details.get("fixed_couplers_status", {}))

    assert raw_details.get("layout_anchor_source") == "fixed_patch"
    assert suppressed_details.get("layout_anchor_source") == "fixed_patch"
    assert raw_details.get("provenance_summary", {}).get("classification") == "runtime_submitted_raw"
    assert suppressed_details.get("provenance_summary", {}).get("classification") == "runtime_submitted_suppressed"
    assert raw_details.get("layout_hash") == suppressed_details.get("layout_hash")
    assert raw_details.get("transpile_hash") == suppressed_details.get("transpile_hash")
    assert raw_details.get("used_physical_qubits") == suppressed_details.get("used_physical_qubits")
    assert raw_details.get("used_physical_edges") == suppressed_details.get("used_physical_edges")
    assert raw_status == suppressed_status
    assert raw_status.get("requested") == [[0, 1], [1, 0]]
    assert raw_status.get("verified_used_edges_subset") is True
    assert raw_status.get("verified_used_qubits_subset") is True
    assert _circuit_signature(raw_circuit) == _circuit_signature(suppressed_circuit)
    assert _observable_signature(raw_observable) == _observable_signature(suppressed_observable)
    assert raw_details.get("runtime_execution_bundle", {}).get("mitigation_mode") == "none"
    assert suppressed_details.get("runtime_execution_bundle", {}).get("mitigation_mode") == "readout"



def test_fixed_couplers_outside_concrete_initial_layout_fail_explicitly() -> None:
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
    except Exception as exc:  # pragma: no cover - qiskit version guard
        pytest.skip(f"GenericBackendV2 unavailable: {exc}")

    backend = GenericBackendV2(
        3,
        basis_gates=["id", "rz", "sx", "x", "cx", "measure", "delay", "reset"],
        coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1]],
        dt=2.2222222222222221e-10,
        seed=7,
        noise_info=True,
    )
    snapshot = freeze_backend_snapshot(backend)
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    resolved_spec = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "shots",
            "backend_profile": "generic_seeded",
            "shots": 128,
            "schedule_policy": "asap",
            "layout_policy": "fixed_patch",
            "fixed_physical_patch": [0, 1, 2],
            "fixed_couplers": [[1, 2], [2, 1]],
            "seed_transpiler": 7,
            "seed_simulator": 7,
        }
    )

    with pytest.raises(RuntimeError, match="does not include a separate coupler/layout planner"):
        _ = build_shots_only_artifact(
            circuit=qc,
            observable=None,
            resolved_spec=resolved_spec,
            resolved_backend=backend,
            calibration_snapshot=snapshot,
        )



def test_fixed_couplers_unsupported_by_backend_target_fail_explicitly() -> None:
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
    except Exception as exc:  # pragma: no cover - qiskit version guard
        pytest.skip(f"GenericBackendV2 unavailable: {exc}")

    backend = GenericBackendV2(
        3,
        basis_gates=["id", "rz", "sx", "x", "cx", "measure", "delay", "reset"],
        coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1]],
        dt=2.2222222222222221e-10,
        seed=7,
        noise_info=True,
    )
    snapshot = freeze_backend_snapshot(backend)
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    resolved_spec = normalize_to_resolved_noise_spec(
        {
            "noise_mode": "shots",
            "backend_profile": "generic_seeded",
            "shots": 128,
            "schedule_policy": "asap",
            "layout_policy": "fixed_patch",
            "fixed_physical_patch": [0, 2],
            "fixed_couplers": [[0, 2], [2, 0]],
            "seed_transpiler": 7,
            "seed_simulator": 7,
        }
    )
    with pytest.raises(RuntimeError, match="unsupported by the resolved backend target"):
        _ = build_shots_only_artifact(
            circuit=qc,
            observable=None,
            resolved_spec=resolved_spec,
            resolved_backend=backend,
            calibration_snapshot=snapshot,
        )
