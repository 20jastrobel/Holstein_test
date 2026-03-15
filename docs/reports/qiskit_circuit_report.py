from __future__ import annotations

import importlib
import json
import warnings
from typing import Any, Mapping, Sequence

import numpy as np

from docs.reports.pdf_utils import get_plt, render_text_page, require_matplotlib
from pipelines.exact_bench.hh_seq_transition_utils import build_time_dependent_sparse_qop
from src.quantum.time_propagation.cfqm_schemes import get_cfqm_scheme

_QISKIT_IMPORT_ERROR: str | None = None


def _require_qiskit() -> tuple[Any, Any, Any, Any]:
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.synthesis import SuzukiTrotter

        return QuantumCircuit, PauliEvolutionGate, SparsePauliOp, SuzukiTrotter
    except Exception as exc:  # pragma: no cover
        global _QISKIT_IMPORT_ERROR
        _QISKIT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            "Qiskit circuit rendering helpers require qiskit with mpl drawer support. "
            f"Original error: {_QISKIT_IMPORT_ERROR}"
        ) from exc


def _to_ixyz(label_exyz: str) -> str:
    return (
        str(label_exyz)
        .replace("e", "I")
        .replace("x", "X")
        .replace("y", "Y")
        .replace("z", "Z")
    )


CFQM_FIXED_NODE_WARNING = "CFQM ignores midpoint/left/right sampling; uses fixed scheme nodes c_j."
CFQM_INNER_ORDER_WARNING = (
    "Inner Suzuki-2 makes overall method 2nd order; use expm_multiply_sparse/dense_expm for true CFQM order."
)
CFQM_CIRCUITIZATION_REASON = (
    "Circuitized CFQM requires cfqm_stage_exp='pauli_suzuki2'; dense_expm/expm_multiply_sparse are numerical-only."
)


def is_cfqm_dynamics_method(method: str) -> bool:
    return str(method).strip().lower() in {"cfqm4", "cfqm6"}


def time_dynamics_circuitization_reason(
    *,
    method: str,
    cfqm_stage_exp: str,
) -> str | None:
    if not is_cfqm_dynamics_method(str(method)):
        return None
    if str(cfqm_stage_exp).strip().lower() == "pauli_suzuki2":
        return None
    return str(CFQM_CIRCUITIZATION_REASON)


def warn_time_dynamics_circuit_semantics(
    *,
    method: str,
    cfqm_stage_exp: str,
    drive_time_sampling: str,
) -> None:
    if not is_cfqm_dynamics_method(str(method)):
        return
    if str(drive_time_sampling).strip().lower() != "midpoint":
        warnings.warn(
            CFQM_FIXED_NODE_WARNING,
            RuntimeWarning,
            stacklevel=2,
        )
    if str(cfqm_stage_exp).strip().lower() == "pauli_suzuki2":
        warnings.warn(
            CFQM_INNER_ORDER_WARNING,
            RuntimeWarning,
            stacklevel=2,
        )


def pauli_poly_to_sparse_pauli_op(poly: Any, tol: float = 1e-12) -> Any:
    """H = sum_j c_j P_j -> SparsePauliOp[(P_j, c_j)]"""
    _, _, SparsePauliOp, _ = _require_qiskit()
    terms = list(poly.return_polynomial())
    if not terms:
        return SparsePauliOp.from_list([("I", 0.0)])

    nq = int(terms[0].nqubit())
    coeff_map: dict[str, complex] = {}
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        lbl = _to_ixyz(str(term.pw2strng()))
        coeff_map[lbl] = coeff_map.get(lbl, 0.0 + 0.0j) + coeff

    cleaned = [(lbl, coeff) for lbl, coeff in coeff_map.items() if abs(coeff) > float(tol)]
    if not cleaned:
        cleaned = [("I" * nq, 0.0 + 0.0j)]
    return SparsePauliOp.from_list(cleaned).simplify(atol=float(tol))


def append_reference_state(circuit: Any, reference_state: np.ndarray) -> None:
    ref = np.asarray(reference_state, dtype=complex).reshape(-1)
    dim = int(1 << int(circuit.num_qubits))
    if ref.size != dim:
        raise ValueError(
            f"reference_state dimension {ref.size} does not match num_qubits={circuit.num_qubits}"
        )
    nrm = float(np.linalg.norm(ref))
    if nrm <= 0.0:
        raise ValueError("reference_state has zero norm")
    ref = ref / nrm

    nz = np.where(np.abs(ref) > 1e-12)[0]
    if nz.size == 1:
        idx = int(nz[0])
        phase = complex(ref[idx])
        if abs(abs(phase) - 1.0) <= 1e-10:
            bit = format(idx, f"0{circuit.num_qubits}b")
            for q in range(circuit.num_qubits):
                if bit[circuit.num_qubits - 1 - q] == "1":
                    circuit.x(q)
            return

    circuit.initialize(ref, list(range(circuit.num_qubits)))


def _ansatz_terms_with_parameters(ansatz: Any, theta: np.ndarray) -> list[tuple[Any, float]]:
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    num_parameters = int(getattr(ansatz, "num_parameters", -1))
    if num_parameters < 0:
        raise ValueError("ansatz is missing num_parameters")
    if int(theta_vec.size) != num_parameters:
        raise ValueError(
            f"theta length {int(theta_vec.size)} does not match ansatz.num_parameters={num_parameters}"
        )

    reps = int(getattr(ansatz, "reps", 1))
    out: list[tuple[Any, float]] = []
    k = 0
    layer_term_groups = getattr(ansatz, "layer_term_groups", None)
    if isinstance(layer_term_groups, list) and layer_term_groups:
        for _ in range(reps):
            for _name, terms in layer_term_groups:
                shared_theta = float(theta_vec[k])
                for term in terms:
                    out.append((term.polynomial, shared_theta))
                k += 1
    else:
        base_terms = list(getattr(ansatz, "base_terms", []))
        if not base_terms:
            raise ValueError("ansatz has no base_terms/layer_term_groups")
        for _ in range(reps):
            for term in base_terms:
                out.append((term.polynomial, float(theta_vec[k])))
                k += 1

    if k != int(theta_vec.size):
        raise RuntimeError(
            f"ansatz parameter traversal consumed {k}, expected {int(theta_vec.size)}"
        )
    return out


def ansatz_to_circuit(
    ansatz: Any,
    theta: np.ndarray,
    *,
    num_qubits: int,
    reference_state: np.ndarray | None = None,
    coefficient_tolerance: float = 1e-12,
) -> Any:
    QuantumCircuit, _, _, SuzukiTrotter = _require_qiskit()
    qc = QuantumCircuit(int(num_qubits))
    if reference_state is not None:
        append_reference_state(qc, np.asarray(reference_state, dtype=complex))

    terms = _ansatz_terms_with_parameters(ansatz, np.asarray(theta, dtype=float))
    synthesis = SuzukiTrotter(order=2, reps=1, preserve_order=True)
    for poly, angle in terms:
        qop = pauli_poly_to_sparse_pauli_op(poly, tol=float(coefficient_tolerance))
        coeffs = np.asarray(qop.coeffs, dtype=complex).reshape(-1)
        if coeffs.size == 0 or np.max(np.abs(coeffs)) <= float(coefficient_tolerance):
            continue
        gate = _require_qiskit()[1](qop, time=float(angle), synthesis=synthesis)
        qc.append(gate, list(range(int(num_qubits))))
    return qc


def adapt_ops_to_circuit(
    ops: list[Any],
    theta: np.ndarray,
    *,
    num_qubits: int,
    reference_state: np.ndarray,
    coefficient_tolerance: float = 1e-12,
) -> Any:
    QuantumCircuit, PauliEvolutionGate, _, SuzukiTrotter = _require_qiskit()
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    if int(theta_vec.size) != int(len(ops)):
        raise ValueError(
            f"theta length {int(theta_vec.size)} does not match selected ops {int(len(ops))}"
        )
    qc = QuantumCircuit(int(num_qubits))
    append_reference_state(qc, np.asarray(reference_state, dtype=complex))
    synthesis = SuzukiTrotter(order=2, reps=1, preserve_order=True)
    for op, ang in zip(ops, theta_vec):
        qop = pauli_poly_to_sparse_pauli_op(op.polynomial, tol=float(coefficient_tolerance))
        coeffs = np.asarray(qop.coeffs, dtype=complex).reshape(-1)
        if coeffs.size == 0 or np.max(np.abs(coeffs)) <= float(coefficient_tolerance):
            continue
        qc.append(PauliEvolutionGate(qop, time=float(ang), synthesis=synthesis), list(range(int(num_qubits))))
    return qc


def ops_to_circuit(
    ops: Sequence[Any],
    theta: np.ndarray,
    *,
    num_qubits: int,
    coefficient_tolerance: float = 1e-12,
) -> Any:
    QuantumCircuit, PauliEvolutionGate, _, SuzukiTrotter = _require_qiskit()
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    if int(theta_vec.size) != int(len(ops)):
        raise ValueError(
            f"theta length {int(theta_vec.size)} does not match op count {int(len(ops))}"
        )
    qc = QuantumCircuit(int(num_qubits))
    synthesis = SuzukiTrotter(order=2, reps=1, preserve_order=True)
    for op, ang in zip(ops, theta_vec):
        qop = pauli_poly_to_sparse_pauli_op(op.polynomial, tol=float(coefficient_tolerance))
        coeffs = np.asarray(qop.coeffs, dtype=complex).reshape(-1)
        if coeffs.size == 0 or np.max(np.abs(coeffs)) <= float(coefficient_tolerance):
            continue
        qc.append(PauliEvolutionGate(qop, time=float(ang), synthesis=synthesis), list(range(int(num_qubits))))
    return qc


def _time_sample(step_idx: int, dt: float, sampling: str) -> float:
    mode = str(sampling).strip().lower()
    if mode == "midpoint":
        return float((float(step_idx) + 0.5) * float(dt))
    if mode == "left":
        return float(float(step_idx) * float(dt))
    if mode == "right":
        return float((float(step_idx) + 1.0) * float(dt))
    raise ValueError(f"Unsupported time sampling {sampling!r}")


def build_suzuki2_time_dependent_circuit(
    *,
    initial_circuit: Any,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_provider_exyz: Any | None,
    time_value: float,
    trotter_steps: int,
    drive_t0: float,
    drive_time_sampling: str,
) -> Any:
    _, PauliEvolutionGate, _, SuzukiTrotter = _require_qiskit()
    qc = initial_circuit.copy()
    if abs(float(time_value)) <= 1e-15:
        return qc

    dt = float(time_value) / float(trotter_steps)
    synthesis = SuzukiTrotter(order=2, reps=1, preserve_order=True)
    qubits = list(range(int(initial_circuit.num_qubits)))

    for step_idx in range(int(trotter_steps)):
        t_sample = _time_sample(step_idx, dt, drive_time_sampling)
        drive_map = {}
        if drive_provider_exyz is not None:
            drive_map = dict(drive_provider_exyz(float(drive_t0) + float(t_sample)))
        qop = build_time_dependent_sparse_qop(
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
            drive_coeff_map_exyz=drive_map,
        )
        qc.append(PauliEvolutionGate(qop, time=float(dt), synthesis=synthesis), qubits)
    return qc


def _build_cfqm_stage_map_exyz(
    *,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_maps_exyz: list[dict[str, complex]],
    a_row: list[float],
    s_static: float,
    coeff_drop_abs_tol: float,
) -> dict[str, complex]:
    ordered_set = set(ordered_labels_exyz)
    stage_map: dict[str, complex] = {}

    for lbl in ordered_labels_exyz:
        coeff0 = static_coeff_map_exyz.get(lbl, 0.0 + 0.0j)
        scaled = complex(float(s_static)) * complex(coeff0)
        if scaled != 0.0:
            stage_map[lbl] = scaled

    for j, drive_map in enumerate(drive_maps_exyz):
        weight = float(a_row[j])
        if weight == 0.0:
            continue
        for lbl, coeff_drive in drive_map.items():
            if lbl not in ordered_set:
                continue
            inc = complex(weight) * complex(coeff_drive)
            if inc == 0.0 and lbl not in stage_map:
                continue
            stage_map[lbl] = stage_map.get(lbl, 0.0 + 0.0j) + inc

    drop = float(max(0.0, coeff_drop_abs_tol))
    if drop > 0.0:
        for lbl in list(stage_map):
            if abs(stage_map[lbl]) < drop:
                del stage_map[lbl]
    return stage_map


def build_cfqm_time_dependent_circuit(
    *,
    method: str,
    initial_circuit: Any,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_provider_exyz: Any | None,
    time_value: float,
    trotter_steps: int,
    drive_t0: float,
    coeff_drop_abs_tol: float,
    cfqm_stage_exp: str,
) -> Any:
    reason = time_dynamics_circuitization_reason(
        method=str(method),
        cfqm_stage_exp=str(cfqm_stage_exp),
    )
    if reason is not None:
        raise ValueError(str(reason))
    _, PauliEvolutionGate, _, SuzukiTrotter = _require_qiskit()
    qc = initial_circuit.copy()
    if abs(float(time_value)) <= 1e-15:
        return qc

    scheme = get_cfqm_scheme(str(method))
    c_nodes = [float(x) for x in scheme["c"]]
    a_rows = [[float(v) for v in row] for row in scheme["a"]]
    s_static = [float(v) for v in scheme["s_static"]]
    dt = float(time_value) / float(trotter_steps)
    synthesis = SuzukiTrotter(order=2, reps=1, preserve_order=True)
    qubits = list(range(int(initial_circuit.num_qubits)))

    for step_idx in range(int(trotter_steps)):
        t_abs = float(drive_t0) + float(step_idx) * float(dt)
        drive_maps_exyz: list[dict[str, complex]] = []
        for c_j in c_nodes:
            t_node = float(t_abs) + float(c_j) * float(dt)
            raw = {} if drive_provider_exyz is None else dict(drive_provider_exyz(float(t_node)))
            drive_maps_exyz.append({str(k): complex(v) for k, v in raw.items()})
        for k in range(len(a_rows) - 1, -1, -1):
            a_row = a_rows[k]
            stage_map = _build_cfqm_stage_map_exyz(
                ordered_labels_exyz=list(ordered_labels_exyz),
                static_coeff_map_exyz=dict(static_coeff_map_exyz),
                drive_maps_exyz=drive_maps_exyz,
                a_row=[float(v) for v in a_row],
                s_static=float(s_static[k]),
                coeff_drop_abs_tol=float(coeff_drop_abs_tol),
            )
            qop = build_time_dependent_sparse_qop(
                ordered_labels_exyz=ordered_labels_exyz,
                static_coeff_map_exyz={},
                drive_coeff_map_exyz=stage_map,
            )
            qc.append(PauliEvolutionGate(qop, time=float(dt), synthesis=synthesis), qubits)
    return qc


def build_time_dynamics_circuit(
    *,
    method: str,
    initial_circuit: Any,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_provider_exyz: Any | None,
    time_value: float,
    trotter_steps: int,
    drive_t0: float,
    drive_time_sampling: str,
    cfqm_stage_exp: str,
    cfqm_coeff_drop_abs_tol: float,
) -> Any:
    method_norm = str(method).strip().lower()
    reason = time_dynamics_circuitization_reason(
        method=str(method_norm),
        cfqm_stage_exp=str(cfqm_stage_exp),
    )
    if reason is not None:
        raise ValueError(str(reason))
    if method_norm == "suzuki2":
        return build_suzuki2_time_dependent_circuit(
            initial_circuit=initial_circuit,
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
            drive_provider_exyz=drive_provider_exyz,
            time_value=time_value,
            trotter_steps=trotter_steps,
            drive_t0=drive_t0,
            drive_time_sampling=drive_time_sampling,
        )
    if is_cfqm_dynamics_method(method_norm):
        return build_cfqm_time_dependent_circuit(
            method=str(method_norm),
            initial_circuit=initial_circuit,
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
            drive_provider_exyz=drive_provider_exyz,
            time_value=time_value,
            trotter_steps=trotter_steps,
            drive_t0=drive_t0,
            coeff_drop_abs_tol=cfqm_coeff_drop_abs_tol,
            cfqm_stage_exp=str(cfqm_stage_exp),
        )
    raise ValueError(f"Unsupported dynamics method {method!r}.")


def _pauli_weight(label_exyz: str) -> int:
    return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y", "z"}))


def _pauli_xy_count(label_exyz: str) -> int:
    return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y"}))


def _cx_proxy_term(label_exyz: str) -> int:
    return int(2 * max(_pauli_weight(label_exyz) - 1, 0))


def _sq_proxy_term(label_exyz: str) -> int:
    return int(2 * _pauli_xy_count(label_exyz) + 1)


def _active_labels_exyz(
    coeff_map_exyz: Mapping[str, complex],
    ordered_labels_exyz: Sequence[str],
    tol: float,
) -> list[str]:
    threshold = float(max(0.0, tol))
    return [
        str(lbl)
        for lbl in ordered_labels_exyz
        if abs(complex(coeff_map_exyz.get(str(lbl), 0.0 + 0.0j))) > threshold
    ]


def compute_sweep_proxy_cost(active_labels_exyz: list[str]) -> dict[str, int]:
    term_exp_count = int(2 * len(active_labels_exyz))
    cx_proxy = int(2 * sum(_cx_proxy_term(lbl) for lbl in active_labels_exyz))
    sq_proxy = int(2 * sum(_sq_proxy_term(lbl) for lbl in active_labels_exyz))
    return {
        "term_exp_count": int(term_exp_count),
        "cx_proxy": int(cx_proxy),
        "sq_proxy": int(sq_proxy),
    }


def compute_time_dynamics_proxy_cost(
    *,
    method: str,
    t_final: float,
    trotter_steps: int,
    drive_t0: float,
    drive_time_sampling: str,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_provider_exyz: Any | None,
    active_coeff_tol: float = 1e-12,
    coeff_drop_abs_tol: float = 0.0,
    cfqm_stage_exp: str,
) -> dict[str, int]:
    method_norm = str(method).strip().lower()
    if int(trotter_steps) < 1:
        raise ValueError("trotter_steps must be >= 1")
    if float(t_final) < 0.0:
        raise ValueError("t_final must be >= 0")

    total_term = 0
    total_cx = 0
    total_sq = 0
    dt = float(t_final) / float(trotter_steps)

    if method_norm == "suzuki2":
        for step_idx in range(int(trotter_steps)):
            t_sample = _time_sample(step_idx, dt, str(drive_time_sampling))
            raw = {} if drive_provider_exyz is None else dict(drive_provider_exyz(float(drive_t0) + float(t_sample)))
            merged = {
                str(lbl): complex(static_coeff_map_exyz.get(str(lbl), 0.0 + 0.0j)) + complex(raw.get(str(lbl), 0.0))
                for lbl in ordered_labels_exyz
            }
            active = _active_labels_exyz(merged, ordered_labels_exyz, float(active_coeff_tol))
            sweep = compute_sweep_proxy_cost(active)
            total_term += int(sweep["term_exp_count"])
            total_cx += int(sweep["cx_proxy"])
            total_sq += int(sweep["sq_proxy"])
    else:
        reason = time_dynamics_circuitization_reason(
            method=str(method_norm),
            cfqm_stage_exp=str(cfqm_stage_exp),
        )
        if reason is not None:
            raise ValueError(str(reason))
        scheme = get_cfqm_scheme(str(method_norm))
        c_nodes = [float(x) for x in scheme["c"]]
        a_rows = [[float(v) for v in row] for row in scheme["a"]]
        s_static = [float(v) for v in scheme["s_static"]]
        for step_idx in range(int(trotter_steps)):
            t_abs = float(drive_t0) + float(step_idx) * float(dt)
            drive_maps_exyz: list[dict[str, complex]] = []
            for c_j in c_nodes:
                t_node = float(t_abs) + float(c_j) * float(dt)
                raw = {} if drive_provider_exyz is None else dict(drive_provider_exyz(float(t_node)))
                drive_maps_exyz.append({str(k): complex(v) for k, v in raw.items()})
            for k, a_row in enumerate(a_rows):
                stage_map = _build_cfqm_stage_map_exyz(
                    ordered_labels_exyz=list(ordered_labels_exyz),
                    static_coeff_map_exyz=dict(static_coeff_map_exyz),
                    drive_maps_exyz=drive_maps_exyz,
                    a_row=[float(v) for v in a_row],
                    s_static=float(s_static[k]),
                    coeff_drop_abs_tol=float(coeff_drop_abs_tol),
                )
                active = _active_labels_exyz(stage_map, ordered_labels_exyz, float(active_coeff_tol))
                sweep = compute_sweep_proxy_cost(active)
                total_term += int(sweep["term_exp_count"])
                total_cx += int(sweep["cx_proxy"])
                total_sq += int(sweep["sq_proxy"])

    return {
        "term_exp_count_total": int(total_term),
        "pauli_rot_count_total": int(total_term),
        "cx_proxy_total": int(total_cx),
        "sq_proxy_total": int(total_sq),
        "depth_proxy_total": int(total_term),
    }


def _op_count_bundle(circuit: Any) -> dict[str, Any]:
    count_ops = {str(key): int(val) for key, val in dict(circuit.count_ops()).items()}
    count_1q = 0
    count_2q = 0
    count_measure = 0
    for inst in circuit.data:
        name = str(getattr(inst.operation, "name", ""))
        if name == "delay":
            continue
        nq = int(len(inst.qubits))
        if name == "measure":
            count_measure += 1
            continue
        if nq == 1:
            count_1q += 1
        elif nq == 2:
            count_2q += 1
    return {
        "num_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "num_parameters": int(len(getattr(circuit, "parameters", []))),
        "count_ops": count_ops,
        "count_1q": int(count_1q),
        "count_2q": int(count_2q),
        "count_measure": int(count_measure),
        "cx_count": int(count_ops.get("cx", 0)),
    }


def _resolve_fake_backend_instance(backend_name: str) -> tuple[Any, str]:
    errors: list[str] = []
    candidates = (
        "qiskit_ibm_runtime.fake_provider",
        "qiskit.providers.fake_provider",
    )
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        backend_cls = getattr(module, str(backend_name), None)
        if backend_cls is None:
            continue
        try:
            return backend_cls(), str(module_name)
        except Exception as exc:  # pragma: no cover
            errors.append(f"{module_name}.{backend_name}: {type(exc).__name__}: {exc}")
    if str(backend_name) == "GenericBackendV2":
        try:
            module = importlib.import_module("qiskit.providers.fake_provider")
            return getattr(module, "GenericBackendV2"), "qiskit.providers.fake_provider"
        except Exception as exc:  # pragma: no cover
            errors.append(f"GenericBackendV2: {type(exc).__name__}: {exc}")
    detail = "; ".join(errors) if errors else "backend class not found"
    raise ValueError(f"Unable to resolve fake backend {backend_name!r}. {detail}")


def transpile_circuit_metrics(
    circuit: Any,
    *,
    backend_name: str,
    use_fake_backend: bool,
    optimization_level: int = 3,
    seed_transpiler: int = 7,
    basis_gates: Sequence[str] = ("rz", "sx", "x", "cx"),
) -> dict[str, Any]:
    try:
        from qiskit import transpile
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Qiskit transpile metrics require qiskit transpiler support. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    backend_name_str = str(backend_name).strip()
    if backend_name_str == "":
        raise ValueError("backend_name must be non-empty for transpile metrics.")

    backend = None
    backend_source = None
    coupling_map = None
    backend_qubits = None
    if bool(use_fake_backend):
        backend, backend_source = _resolve_fake_backend_instance(backend_name_str)
        if callable(backend) and backend_name_str == "GenericBackendV2":
            backend = backend(
                num_qubits=int(circuit.num_qubits),
                basis_gates=[str(x) for x in basis_gates],
                seed=int(seed_transpiler),
            )
        backend_qubits = getattr(backend, "num_qubits", None)
        if backend_qubits is not None and int(circuit.num_qubits) > int(backend_qubits):
            raise ValueError(
                f"Backend {backend_name_str} has {int(backend_qubits)} qubits, "
                f"but the circuit needs {int(circuit.num_qubits)}."
            )
        coupling_map = getattr(backend, "coupling_map", None)
    else:
        raise ValueError(
            "Only fake-backend local transpile metrics are supported; "
            "set use_fake_backend=True."
        )

    expanded = expand_pauli_evolution_once(circuit)
    transpiled = transpile(
        expanded,
        basis_gates=[str(x) for x in basis_gates],
        coupling_map=coupling_map,
        optimization_level=int(optimization_level),
        seed_transpiler=int(seed_transpiler),
    )
    target_info = {
        "backend_name": str(backend_name_str),
        "backend_source": str(backend_source),
        "backend_num_qubits": (None if backend_qubits is None else int(backend_qubits)),
        "use_fake_backend": bool(use_fake_backend),
        "basis_gates": [str(x) for x in basis_gates],
        "optimization_level": int(optimization_level),
        "seed_transpiler": int(seed_transpiler),
    }
    return {
        "target": target_info,
        "raw": _op_count_bundle(circuit),
        "expanded_once": _op_count_bundle(expanded),
        "transpiled": _op_count_bundle(transpiled),
    }


def _instruction_is_pauli_evolution(op: Any, pauli_evolution_gate_cls: Any) -> bool:
    return isinstance(op, pauli_evolution_gate_cls) or str(getattr(op, "name", "")) == "PauliEvolution"


def expand_pauli_evolution_once(circuit: Any) -> Any:
    QuantumCircuit, PauliEvolutionGate, _, _ = _require_qiskit()
    expanded = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=str(getattr(circuit, "name", "")))
    for inst in circuit.data:
        op = inst.operation
        if _instruction_is_pauli_evolution(op, PauliEvolutionGate):
            definition = getattr(op, "definition", None)
            if definition is None:
                expanded.append(op, inst.qubits, inst.clbits)
                continue
            qubit_indices = [int(circuit.find_bit(qubit).index) for qubit in inst.qubits]
            clbit_indices = [int(circuit.find_bit(clbit).index) for clbit in inst.clbits]
            expanded.compose(definition, qubits=qubit_indices, clbits=clbit_indices, inplace=True)
        else:
            expanded.append(op, inst.qubits, inst.clbits)
    expanded.global_phase = getattr(circuit, "global_phase", 0.0)
    return expanded


def _stringify(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_stringify(item) for item in value) or "none"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def render_circuit_summary_page(
    pdf: Any,
    *,
    title: str,
    circuit: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    notes: Sequence[str] | None = None,
) -> None:
    lines = [title, ""]
    if circuit is not None:
        count_ops = {
            str(key): int(val)
            for key, val in dict(circuit.count_ops()).items()
        }
        lines.extend(
            [
                "Circuit summary",
                f"  - num_qubits: {int(circuit.num_qubits)}",
                f"  - depth: {int(circuit.depth())}",
                f"  - size: {int(circuit.size())}",
                f"  - num_parameters: {int(len(getattr(circuit, 'parameters', [])))}",
                f"  - count_ops: {_stringify(count_ops)}",
            ]
        )
    if metadata:
        lines.extend(["", "Metadata"])
        for key, value in metadata.items():
            lines.append(f"  - {key}: {_stringify(value)}")
    note_lines = [str(line) for line in (notes or []) if str(line).strip()]
    if note_lines:
        lines.extend(["", "Notes"])
        lines.extend(f"  - {line}" for line in note_lines)
    render_text_page(pdf, lines, fontsize=10, line_spacing=0.03, max_line_width=110)


def render_circuit_page(
    pdf: Any,
    *,
    circuit: Any,
    title: str,
    subtitle: str | None = None,
    notes: Sequence[str] | None = None,
    fold: int = 30,
    scale: float = 0.65,
    idle_wires: bool = False,
    expand_evolution: bool = False,
) -> None:
    require_matplotlib()
    plt = get_plt()
    rendered_circuit = expand_pauli_evolution_once(circuit) if bool(expand_evolution) else circuit
    fig = rendered_circuit.draw(
        output="mpl",
        fold=int(fold),
        idle_wires=bool(idle_wires),
        scale=float(scale),
    )
    fig.set_size_inches(11.0, 8.5, forward=True)
    fig.suptitle(str(title), fontsize=14, y=0.99)
    if subtitle:
        fig.text(0.02, 0.965, str(subtitle), ha="left", va="top", fontsize=9)
    note_lines = [str(line).strip() for line in (notes or []) if str(line).strip()]
    if note_lines:
        fig.text(0.02, 0.02, "\n".join(note_lines), ha="left", va="bottom", fontsize=8, family="monospace")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


__all__ = [
    "adapt_ops_to_circuit",
    "ansatz_to_circuit",
    "append_reference_state",
    "build_cfqm_time_dependent_circuit",
    "build_time_dynamics_circuit",
    "build_suzuki2_time_dependent_circuit",
    "compute_sweep_proxy_cost",
    "compute_time_dynamics_proxy_cost",
    "expand_pauli_evolution_once",
    "is_cfqm_dynamics_method",
    "ops_to_circuit",
    "pauli_poly_to_sparse_pauli_op",
    "render_circuit_page",
    "render_circuit_summary_page",
    "time_dynamics_circuitization_reason",
    "transpile_circuit_metrics",
    "warn_time_dynamics_circuit_semantics",
]
