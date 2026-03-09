from __future__ import annotations

import json
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
) -> Any:
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
        for k, a_row in enumerate(a_rows):
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
        "cx_proxy_total": int(total_cx),
        "sq_proxy_total": int(total_sq),
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
    "build_suzuki2_time_dependent_circuit",
    "compute_sweep_proxy_cost",
    "compute_time_dynamics_proxy_cost",
    "expand_pauli_evolution_once",
    "pauli_poly_to_sparse_pauli_op",
    "render_circuit_page",
    "render_circuit_summary_page",
]
