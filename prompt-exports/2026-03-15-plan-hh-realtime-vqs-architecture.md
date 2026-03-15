<file_map>
/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2
├── MATH
│   └── IMPLEMENT_SOON.md *
├── pipelines
│   ├── exact_bench
│   │   └── hh_fixed_seed_budgeted_projected_dynamics.py * +
│   ├── hardcoded
│   │   ├── adapt_pipeline.py * +
│   │   ├── handoff_state_bundle.py * +
│   │   ├── hh_continuation_generators.py * +
│   │   ├── hh_continuation_pruning.py * +
│   │   ├── hh_continuation_replay.py * +
│   │   ├── hh_continuation_rescue.py * +
│   │   ├── hh_continuation_scoring.py * +
│   │   ├── hh_continuation_stage_control.py * +
│   │   ├── hh_continuation_symmetry.py * +
│   │   ├── hh_continuation_types.py * +
│   │   ├── hh_staged_workflow.py * +
│   │   └── hh_vqe_from_adapt_family.py * +
│   └── run_guide.md *
├── src
│   └── quantum
│       ├── operator_pools
│       │   ├── polaron_paop.py * +
│       │   └── vlf_sq.py * +
│       ├── time_propagation
│       │   ├── __init__.py * +
│       │   ├── cfqm_propagator.py * +
│       │   ├── cfqm_schemes.py * +
│       │   ├── local_checkpoint_fit.py * +
│       │   └── projected_real_time.py * +
│       ├── compiled_ansatz.py * +
│       └── compiled_polynomial.py * +
├── test
│   ├── test_hh_adapt_beam_search.py * +
│   ├── test_hh_continuation_generators.py * +
│   ├── test_hh_continuation_replay.py * +
│   ├── test_hh_continuation_scoring.py * +
│   ├── test_local_checkpoint_fit.py * +
│   ├── test_projected_real_time.py * +
│   └── test_staged_export_replay_roundtrip.py * +
└── README.md *


(* denotes selected files)
(+ denotes code-map available)

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/time_propagation/__init__.py
Imports:
  - from src.quantum.time_propagation.cfqm_propagator import CFQMConfig, cfqm_step
  - from src.quantum.time_propagation.cfqm_schemes import get_cfqm_scheme
  - from src.quantum.time_propagation.local_checkpoint_fit import (
    CheckpointFitConfig,
    CheckpointFitStepResult,
    CheckpointFitTrajectoryResult,
    LocalPauliAnsatzSpec,
    build_local_pauli_ansatz_terms,
    default_chain_edges,
    fit_checkpoint_target_state,
    fit_checkpoint_trajectory,
)
  - from src.quantum.time_propagation.projected_real_time import (
    ExactDrivenReferenceResult,
    ProjectedRealTimeConfig,
    ProjectedRealTimeResult,
    build_tangent_vectors,
    expectation_total_hamiltonian,
    run_exact_driven_reference,
    run_projected_real_time_trajectory,
    solve_mclachlan_step,
    state_fidelity,
)
---

Global vars:
  - __all__
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/time_propagation/projected_real_time.py
Imports:
  - import math
  - from dataclasses import dataclass
  - from typing import Any, Mapping, Sequence
  - import numpy as np
  - from src.quantum.vqe_latex_python_pairs import apply_exp_pauli_polynomial
  - from scipy.sparse import csc_matrix, diags
  - from scipy.sparse.linalg import expm_multiply as scipy_expm_multiply
  - from scipy.linalg import expm as scipy_dense_expm
---
Classes:
  - ProjectedRealTimeConfig
    Properties:
      - t_final
      - num_times
      - ode_substeps
      - tangent_eps
      - lambda_reg
      - svd_rcond
      - coefficient_tolerance
      - sort_terms
  - ProjectedRealTimeResult
    Properties:
      - times
      - theta_history
      - states
      - trajectory_rows
  - ExactDrivenReferenceResult
    Properties:
      - times
      - states
      - energies_total
      - trajectory_rows

Functions:
  - L48: def _normalize_state(psi: np.ndarray) -> np.ndarray:
  - L59: def _pauli_matrix_exyz(label: str) -> np.ndarray:
  - L77: def _is_all_z_type(label: str) -> bool:
  - L84: def _build_drive_diagonal(
    drive_map: Mapping[str, complex],
    *,
    dim: int,
    nq: int,
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
  - L112: def _term_polynomial(term: Any) -> Any:
  - L121: def _apply_ordered_terms(
    reference_state: np.ndarray,
    terms: Sequence[Any],
    theta: np.ndarray,
    *,
    coefficient_tolerance: float,
    sort_terms: bool,
) -> np.ndarray:
  - L149: def build_tangent_vectors(
    reference_state: np.ndarray,
    terms: Sequence[Any],
    theta: np.ndarray,
    *,
    tangent_eps: float = 1e-6,
    coefficient_tolerance: float = 1e-12,
    sort_terms: bool = True,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
  - L215: def solve_mclachlan_step(
    tangents: Sequence[np.ndarray],
    hpsi: np.ndarray,
    *,
    lambda_reg: float = 1e-8,
    svd_rcond: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
  - L265: def _apply_total_hamiltonian(
    psi: np.ndarray,
    hmat_static: np.ndarray,
    *,
    drive_coeff_provider_exyz: Any | None,
    t_physical: float,
    drive_diag_cache: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
  - L303: def _rhs_theta_dot(
    t_rel: float,
    theta: np.ndarray,
    *,
    reference_state: np.ndarray,
    terms: Sequence[Any],
    hmat_static: np.ndarray,
    drive_coeff_provider_exyz: Any | None,
    drive_t0: float,
    tangent_eps: float,
    lambda_reg: float,
    svd_rcond: float,
    coefficient_tolerance: float,
    sort_terms: bool,
    drive_diag_cache: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
  - L351: def run_projected_real_time_trajectory(
    reference_state: np.ndarray,
    terms: Sequence[Any],
    hmat_static: np.ndarray,
    *,
    config: ProjectedRealTimeConfig,
    drive_coeff_provider_exyz: Any | None = None,
    drive_t0: float = 0.0,
    theta_init: np.ndarray | None = None,
) -> ProjectedRealTimeResult:
  - L498: def run_exact_driven_reference(
    initial_state: np.ndarray,
    hmat_static: np.ndarray,
    *,
    t_final: float,
    num_times: int,
    reference_steps: int,
    drive_coeff_provider_exyz: Any | None = None,
    drive_t0: float = 0.0,
    time_sampling: str = "midpoint",
) -> ExactDrivenReferenceResult:
  - L639: def expectation_total_hamiltonian(
    psi: np.ndarray,
    hmat_static: np.ndarray,
    *,
    drive_coeff_provider_exyz: Any | None = None,
    t_physical: float = 0.0,
) -> float:
  - L659: def state_fidelity(psi_left: np.ndarray, psi_right: np.ndarray) -> float:

Global vars:
  - _EXPM_SPARSE_MIN_DIM
  - _MATH_NORMALIZE_STATE
  - _MATH_PAULI_MATRIX_EXYZ
  - _MATH_DRIVE_IS_Z_TYPE
  - _MATH_DRIVE_DIAGONAL
  - _MATH_TERM_POLY
  - _MATH_APPLY_ORDERED_TERMS
  - _MATH_TANGENT_VECTORS
  - _MATH_SOLVE_MCLACHLAN
  - _MATH_TOTAL_H_ACTION
  - _MATH_RHS
  - _MATH_RUN_PROJECTED_RT
  - _MATH_EXACT_REFERENCE
  - _MATH_TOTAL_ENERGY
  - _MATH_STATE_FIDELITY
  - __all__
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/time_propagation/local_checkpoint_fit.py
Imports:
  - import math
  - from dataclasses import dataclass
  - from typing import Any, Sequence
  - import numpy as np
  - from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
  - from src.quantum.pauli_polynomial_class import PauliPolynomial
  - from src.quantum.qubitization_module import PauliTerm
  - from src.quantum.vqe_latex_python_pairs import AnsatzTerm
  - from scipy.optimize import minimize  # type: ignore
---
Classes:
  - LocalPauliAnsatzSpec
    Properties:
      - num_qubits
      - reps
      - single_axes
      - entangler_axes
      - entangler_edges
      - repr_mode
  - CheckpointFitConfig
    Properties:
      - optimizer_method
      - maxiter
      - gtol
      - ftol
      - angle_bound
      - param_shift
      - coefficient_tolerance
  - CheckpointFitStepResult
    Properties:
      - theta
      - state
      - fidelity
      - objective
      - solver_row
  - CheckpointFitTrajectoryResult
    Properties:
      - times
      - theta_history
      - states
      - solver_rows

Functions:
  - L58: def _normalize_state(psi: np.ndarray) -> np.ndarray:
  - L69: def default_chain_edges(num_qubits: int) -> tuple[tuple[int, int], ...]:
  - L79: def _pauli_word(num_qubits: int, ops: dict[int, str]) -> str:
  - L96: def _monomial_term(
    *,
    num_qubits: int,
    label: str,
    ops: dict[int, str],
    repr_mode: str,
) -> AnsatzTerm:
  - L114: def build_local_pauli_ansatz_terms(spec: LocalPauliAnsatzSpec) -> tuple[AnsatzTerm, ...]:
  - L157: def state_fidelity(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
  - L169: def _evaluate_fidelity_objective(
    theta: np.ndarray,
    *,
    executor: CompiledAnsatzExecutor,
    reference_state: np.ndarray,
    target_state: np.ndarray,
    param_shift: float,
) -> tuple[float, np.ndarray, np.ndarray, float]:
  - L204: def _try_import_scipy_minimize() -> Any:
  - L218: def fit_checkpoint_target_state(
    reference_state: np.ndarray,
    target_state: np.ndarray,
    terms: Sequence[AnsatzTerm],
    *,
    config: CheckpointFitConfig,
    theta_init: np.ndarray | None = None,
    executor: CompiledAnsatzExecutor | None = None,
) -> CheckpointFitStepResult:
  - L246: def _evaluate(theta_raw: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
  - L325: def fit_checkpoint_trajectory(
    reference_state: np.ndarray,
    target_states: Sequence[np.ndarray],
    times: Sequence[float],
    terms: Sequence[AnsatzTerm],
    *,
    config: CheckpointFitConfig,
    theta_init: np.ndarray | None = None,
) -> CheckpointFitTrajectoryResult:

Global vars:
  - _MATH_NORMALIZE_STATE
  - _MATH_CHAIN_EDGES
  - _MATH_PAULI_WORD
  - _MATH_MONOMIAL_TERM
  - _MATH_BUILD_LOCAL_ANSA
  - _MATH_STATE_FIDELITY
  - _MATH_FIDELITY_OBJECTIVE
  - _MATH_FIT_CHECKPOINT
  - _MATH_FIT_TRAJECTORY
  - __all__
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/time_propagation/cfqm_propagator.py
Imports:
  - import math
  - import warnings
  - from dataclasses import dataclass
  - from typing import Any, Callable, Mapping
  - import numpy as np
  - from src.quantum.pauli_polynomial_class import PauliPolynomial
  - from src.quantum.qubitization_module import PauliTerm
  - from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix
  - from scipy.sparse import csc_matrix, coo_matrix
  - from src.quantum.pauli_actions import compile_pauli_action_exyz
  - from src.quantum.pauli_actions import apply_exp_term, compile_pauli_action_exyz
  - from scipy.linalg import expm
  - from scipy.sparse.linalg import expm_multiply
---
Classes:
  - CFQMConfig
    Properties:
      - backend
      - coeff_drop_abs_tol
      - normalize
      - sparse_min_dim
      - norm_drift_logger
      - emit_inner_order_warning

Functions:
  - L33: def _cfg_get(config: object, key: str, default: Any) -> Any:
  - L41: def _build_dense_stage_matrix_via_repo_utility(
    stage_coeff_map: Mapping[str, complex | float],
    ordered_labels: list[str],
) -> np.ndarray:
  - L79: def _build_sparse_stage_matrix_via_compiled_actions(
    stage_coeff_map: Mapping[str, complex | float],
    ordered_labels: list[str],
):
  - L132: def _apply_stage_pauli_suzuki2(
    psi: np.ndarray,
    stage_coeff_map: Mapping[str, complex | float],
    dt: float,
    ordered_labels: list[str],
) -> np.ndarray:
  - L159: def apply_stage_exponential(
    psi: np.ndarray,
    stage_coeff_map: dict[str, complex | float],
    dt: float,
    ordered_labels: list[str],
    backend_config: object,
) -> np.ndarray:
  - L221: def cfqm_step(
    psi: np.ndarray,
    t_abs: float,
    dt: float,
    static_coeff_map: dict[str, complex | float],
    drive_coeff_provider: Callable[[float], Mapping[str, complex | float]] | None,
    ordered_labels: list[str],
    scheme: dict,
    config: object,
) -> np.ndarray:
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/time_propagation/cfqm_schemes.py
Imports:
  - from typing import Any
---

Functions:
  - L80: def _norm_key(scheme_id: str) -> str:
  - L84: def _close(lhs: float, rhs: float, tol: float) -> bool:
  - L88: def _row_sums(a: list[list[float]]) -> list[float]:
  - L92: def _col_sums(a: list[list[float]]) -> list[float]:
  - L103: def validate_scheme(scheme: dict[str, Any], *, tol: float = _DEFAULT_TOL) -> None:
  - L170: def get_cfqm_scheme(scheme_id: str) -> dict[str, Any]:

Global vars:
  - _DEFAULT_TOL
  - _SCHEME_DATA
  - _SCHEME_ALIASES
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/compiled_ansatz.py
Imports:
  - from dataclasses import dataclass
  - from typing import TYPE_CHECKING, Any, Sequence
  - import numpy as np
  - from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_exp_term,
    compile_pauli_action_exyz,
)
  - from src.quantum.vqe_latex_python_pairs import AnsatzTerm
---
Classes:
  - CompiledRotationStep
    Properties:
      - coeff_real
      - action
  - CompiledPolynomialRotationPlan
    Properties:
      - nq
      - steps
  - CompiledAnsatzExecutor
    Methods:
      - L44: def __init__(
        self,
        terms: Sequence["AnsatzTerm"],
        *,
        coefficient_tolerance: float = 1e-12,
        ignore_identity: bool = True,
        sort_terms: bool = True,
        pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
    ):
      - L80: def _compile_polynomial_plan(self, poly: Any) -> CompiledPolynomialRotationPlan:
      - L122: def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
    Properties:
      - _MATH_INIT
      - _MATH_COMPILE_POLYNOMIAL_PLAN
      - _MATH_PREPARE_STATE

Global vars:
  - __all__
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/compiled_polynomial.py
Imports:
  - from dataclasses import dataclass
  - import numpy as np
  - from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_compiled_pauli,
    compile_pauli_action_exyz,
)
  - from src.quantum.pauli_polynomial_class import PauliPolynomial
---
Classes:
  - CompiledPolynomialTerm
    Properties:
      - coeff
      - action
  - CompiledPolynomialAction
    Properties:
      - nq
      - terms

Functions:
  - L40: def compile_polynomial_action(
    poly: PauliPolynomial,
    *,
    tol: float = 1e-15,
    pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
) -> CompiledPolynomialAction:
  - L85: def apply_compiled_polynomial(psi: np.ndarray, compiled: CompiledPolynomialAction) -> np.ndarray:
  - L106: def energy_via_one_apply(
    psi: np.ndarray, compiled_h: CompiledPolynomialAction,
) -> tuple[float, np.ndarray]:
  - L119: def adapt_commutator_grad_from_hpsi(Hpsi: np.ndarray, Apsi: np.ndarray) -> float:

Global vars:
  - _MATH_COMPILE_POLYNOMIAL_ACTION
  - _MATH_APPLY_COMPILED_POLYNOMIAL
  - _MATH_ENERGY_VIA_ONE_APPLY
  - _MATH_ADAPT_COMMUTATOR_GRAD
  - __all__
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/operator_pools/polaron_paop.py
Imports:
  - from dataclasses import dataclass
  - import math
  - from src.quantum.hubbard_latex_python_pairs import (
    bravais_nearest_neighbor_edges,
    boson_operator,
    boson_displacement_operator,
    boson_qubits_per_site,
    jw_number_operator,
    mode_index,
    phonon_qubit_indices_for_site,
)
  - from src.quantum.pauli_polynomial_class import PauliPolynomial, fermion_minus_operator, fermion_plus_operator
  - from src.quantum.qubitization_module import PauliTerm
---
Classes:
  - PhononMotifSpec
    Properties:
      - label
      - family
      - poly
      - sites
      - bonds
      - uses_sq

Functions:
  - L35: def _to_signature(poly: PauliPolynomial, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
  - L48: def _clean_poly(poly: PauliPolynomial, prune_eps: float) -> PauliPolynomial:
  - L53: def _prune_poly(poly: PauliPolynomial, prune_eps: float, *, enforce_real: bool) -> PauliPolynomial:
  - L76: def _normalize_poly(poly: PauliPolynomial, mode: str) -> PauliPolynomial:
  - L99: def _append_operator(
    pool: list[tuple[str, PauliPolynomial]],
    label: str,
    poly: PauliPolynomial,
    split_paulis: bool,
    prune_eps: float,
) -> None:
  - L124: def _mul_clean(
    left: PauliPolynomial,
    right: PauliPolynomial,
    prune_eps: float,
    *,
    enforce_real: bool = True,
) -> PauliPolynomial:
  - L135: def _distance_1d(i: int, j: int, n_sites: int, periodic: bool) -> int:
  - L143: def _word_from_qubit_letters(nq: int, letters: dict[int, str]) -> str:
  - L154: def jw_current_hop(nq: int, p: int, q: int) -> PauliPolynomial:
  - L189: def _drop_terms_with_identity_on_qubits(poly: PauliPolynomial, qubits: tuple[int, ...]) -> PauliPolynomial:
  - L205: def make_phonon_motifs(
    family: str,
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    boundary: str,
    prune_eps: float = 0.0,
    normalization: str = "none",
) -> list[PhononMotifSpec]:
  - L241: def local_qubits(site: int) -> tuple[int, ...]:
  - L254: def b_i(site: int) -> PauliPolynomial:
  - L267: def bdag_i(site: int) -> PauliPolynomial:
  - L280: def p_i(site: int) -> PauliPolynomial:
  - L286: def delta_p_ij(i_site: int, j_site: int) -> PauliPolynomial:
  - L292: def delta_p_power(i_site: int, j_site: int, exponent: int) -> PauliPolynomial:
  - L309: def bond_p_sum(i_site: int, j_site: int) -> PauliPolynomial:
  - L317: def _append_motif(
        *,
        label: str,
        poly: PauliPolynomial,
        sites: tuple[int, ...],
        bonds: tuple[tuple[int, int], ...] = (),
        uses_sq: bool = False,
        drop_phonon_identity: bool = False,
    ) -> None:
  - L387: def _make_paop_core(
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
  - L469: def n_i(site: int) -> PauliPolynomial:
  - L513: def x_i(site: int) -> PauliPolynomial:
  - L525: def sq_i(site: int) -> PauliPolynomial:
  - L533: def shifted_density(site: int) -> PauliPolynomial:
  - L539: def doublon_i(site: int) -> PauliPolynomial:
  - L549: def k_ij(i_site: int, j_site: int) -> PauliPolynomial:
  - L563: def j_ij(i_site: int, j_site: int) -> PauliPolynomial:
  - L603: def bond_sq_sum(i_site: int, j_site: int) -> PauliPolynomial:
  - L609: def pair_sq_ij(i_site: int, j_site: int) -> PauliPolynomial:
  - L855: def make_pool(
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
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/operator_pools/vlf_sq.py
Imports:
  - from typing import Any
  - from src.quantum.hubbard_latex_python_pairs import (
    boson_displacement_operator,
    boson_operator,
    boson_qubits_per_site,
    jw_number_operator,
    mode_index,
    phonon_qubit_indices_for_site,
)
  - from src.quantum.pauli_polynomial_class import PauliPolynomial
  - from src.quantum.qubitization_module import PauliTerm
  - from src.quantum.operator_pools.polaron_paop import (
    _clean_poly,
    _distance_1d,
    _mul_clean,
    _normalize_poly,
    _to_signature,
)
---

Functions:
  - L31: def _family_flags(name: str) -> tuple[str, bool, bool, bool]:
  - L44: def _shells_for_radius(*, num_sites: int, periodic: bool, shell_radius: int | None) -> list[int]:
  - L58: def _identity_poly(nq: int) -> PauliPolynomial:
  - L63: def build_vlf_sq_pool(
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
  - L112: def local_qubits(site: int) -> tuple[int, ...]:
  - L125: def n_i(site: int) -> PauliPolynomial:
  - L133: def shifted_density(site: int) -> PauliPolynomial:
  - L136: def b_i(site: int) -> PauliPolynomial:
  - L146: def bdag_i(site: int) -> PauliPolynomial:
  - L156: def p_i(site: int) -> PauliPolynomial:
  - L162: def x_i(site: int) -> PauliPolynomial:
  - L174: def sq_i(site: int) -> PauliPolynomial:
  - L247: def make_vlf_sq_pool(
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

Global vars:
  - __all__
  - _MATH_SHIFTED_DENSITY
  - _MATH_VLF_SHELL
  - _MATH_SQ
  - _MATH_DENS_SQ
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/exact_bench/hh_fixed_seed_budgeted_projected_dynamics.py
Imports:
  - import argparse
  - import csv
  - import json
  - import sys
  - from dataclasses import asdict, dataclass
  - from datetime import datetime, timezone
  - from pathlib import Path
  - from typing import Any, Mapping, Sequence
  - import numpy as np
  - from docs.reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    get_plt,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)
  - from docs.reports.qiskit_circuit_report import ops_to_circuit, transpile_circuit_metrics
  - from pipelines.hardcoded.hh_vqe_from_adapt_family import build_replay_sequence_from_input_json
  - from src.quantum.drives_time_potential import build_gaussian_sinusoid_density_drive
  - from src.quantum.time_propagation import (
    ProjectedRealTimeConfig,
    expectation_total_hamiltonian,
    run_exact_driven_reference,
    run_projected_real_time_trajectory,
    state_fidelity,
)
---
Classes:
  - SweepConfig
    Properties:
      - fixed_seed_json
      - output_json
      - output_csv
      - output_pdf
      - run_root
      - tag
      - backend_name
      - use_fake_backend
      - circuit_optimization_level
      - circuit_seed_transpiler
      - max_cx_budget
      - t_final
      - num_times
      - reference_steps
      - ode_substeps
      - tangent_eps
      - lambda_reg
      - svd_rcond
      - drive_A
      - drive_omega
      - drive_tbar
      - drive_phi
      - drive_pattern
      - drive_t0
      - drive_time_sampling
      - prefix_limit
      - representative_prefixes
      - skip_pdf

Functions:
  - L84: def _now_utc() -> str:
  - L88: def _jsonable(value: Any) -> Any:
  - L102: def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
  - L107: def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
  - L125: def _parse_int_tuple(raw: str) -> tuple[int, ...]:
  - L145: def _collect_hardcoded_terms_exyz(h_poly: Any) -> tuple[list[str], dict[str, complex]]:
  - L159: def _pauli_matrix_exyz(label: str) -> np.ndarray:
  - L172: def _build_hamiltonian_matrix(coeff_map_exyz: Mapping[str, complex]) -> np.ndarray:
  - L183: def _build_drive_provider(
    *,
    num_sites: int,
    nq_total: int,
    ordering: str,
    cfg: SweepConfig,
) -> tuple[Any, dict[str, Any]]:
  - L213: def _transpile_theta_history(
    *,
    terms: Sequence[Any],
    theta_history: np.ndarray,
    num_qubits: int,
    cfg: SweepConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
  - L258: def _candidate_row(
    *,
    prefix_k: int,
    times: np.ndarray,
    exact_states: Sequence[np.ndarray],
    exact_energies: np.ndarray,
    projected_result: Any,
    replay_terms: Sequence[Any],
    hmat_static: np.ndarray,
    drive_provider: Any,
    cfg: SweepConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
  - L326: def _best_row(rows: Sequence[Mapping[str, Any]], *, key: str) -> dict[str, Any] | None:
  - L353: def write_summary_pdf(
    *,
    cfg: SweepConfig,
    payload: Mapping[str, Any],
    candidate_details: Mapping[int, Mapping[str, Any]],
    seed_payload: Mapping[str, Any],
) -> None:
  - L457: def run_sweep(cfg: SweepConfig) -> dict[str, Any]:
  - L582: def build_cli_parser() -> argparse.ArgumentParser:
  - L624: def parse_args(argv: Sequence[str] | None = None) -> SweepConfig:
  - L671: def main(argv: Sequence[str] | None = None) -> int:

Global vars:
  - REPO_ROOT
  - _DEFAULT_FIXED_SEED_JSON
  - _DEFAULT_OUTPUT_JSON
  - _DEFAULT_OUTPUT_CSV
  - _DEFAULT_OUTPUT_PDF
  - _DEFAULT_RUN_ROOT
  - _DEFAULT_TAG
  - _DEFAULT_BACKEND_NAME
  - _DEFAULT_REPRESENTATIVE_PREFIXES
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_scoring.py
Imports:
  - from dataclasses import dataclass
  - import math
  - from typing import Any, Iterable, Mapping, Sequence
  - import numpy as np
  - from pipelines.hardcoded.hh_continuation_types import (
    CandidateFeatures,
    CompileCostEstimate,
    CurvatureOracle,
    MeasurementCacheStats,
    MeasurementPlan,
    NoveltyOracle,
)
  - from pipelines.hardcoded.hh_continuation_motifs import motif_bonus_for_generator
  - from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
  - from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    apply_compiled_polynomial,
    compile_polynomial_action,
)
  - from src.quantum.pauli_actions import apply_compiled_pauli
---
Classes:
  - SimpleScoreConfig
    Properties:
      - lambda_F
      - lambda_compile
      - lambda_measure
      - lambda_leak
      - z_alpha
      - wD
      - wG
      - wC
      - wc
      - depth_ref
      - group_ref
      - shot_ref
      - family_ref
      - lifetime_cost_mode
      - score_version
  - FullScoreConfig
    Properties:
      - z_alpha
      - lambda_F
      - lambda_H
      - rho
      - eta_L
      - gamma_N
      - wD
      - wG
      - wC
      - wP
      - wc
      - depth_ref
      - group_ref
      - shot_ref
      - optdim_ref
      - reuse_ref
      - family_ref
      - novelty_eps
      - shortlist_fraction
      - shortlist_size
      - batch_target_size
      - batch_size_cap
      - batch_near_degenerate_ratio
      - compat_overlap_weight
      - compat_comm_weight
      - compat_curv_weight
      - compat_sched_weight
      - leakage_cap
      - lifetime_cost_mode
      - remaining_evaluations_proxy_mode
      - lifetime_weight
      - motif_bonus_weight
      - metric_floor
      - reduced_metric_collapse_rel_tol
      - ridge_growth_factor
      - ridge_max_steps
      - score_version
  - _ScaffoldDerivativeContext
    Properties:
      - psi_state
      - hpsi_state
      - refit_window_indices
      - dpsi_window
      - tangents_window
      - Q_window
      - H_window_hessian
  - Phase1CompileCostOracle
    Methods:
      - L106: def estimate(
        self,
        *,
        candidate_term_count: int,
        position_id: int,
        append_position: int,
        refit_active_count: int,
    ) -> CompileCostEstimate:
  - MeasurementCacheAudit
    Methods:
      - L131: def __init__(
        self,
        nominal_shots_per_group: int = 1,
        *,
        plan_version: str = "phase1_grouped_label_reuse",
        grouping_mode: str = "grouped_label_reuse",
    ) -> None:
      - L143: def clone(self) -> "MeasurementCacheAudit":
      - L152: def snapshot(self) -> dict[str, Any]:
      - L161: def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "MeasurementCacheAudit":
      - L174: def plan_for(self, group_keys: Iterable[str]) -> MeasurementPlan:
      - L187: def estimate(self, group_keys: Iterable[str]) -> MeasurementCacheStats:
      - L209: def commit(self, group_keys: Iterable[str]) -> None:
      - L215: def summary(self) -> dict[str, float]:
  - Phase2NoveltyOracle
    Methods:
      - L643: def prepare_scaffold_context(
        self,
        *,
        selected_ops: Sequence[Any],
        theta: np.ndarray,
        psi_ref: np.ndarray,
        psi_state: np.ndarray,
        h_compiled: CompiledPolynomialAction,
        hpsi_state: np.ndarray,
        refit_window_indices: Sequence[int],
        pauli_action_cache: dict[str, Any] | None = None,
    ) -> _ScaffoldDerivativeContext:
      - L707: def estimate(
        self,
        *,
        scaffold_context: _ScaffoldDerivativeContext,
        candidate_label: str,
        candidate_term: Any,
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
        novelty_eps: float = 1e-6,
    ) -> Mapping[str, Any]:
  - Phase2CurvatureOracle
    Methods:
      - L748: def estimate(
        self,
        *,
        base_feature: CandidateFeatures,
        novelty_info: Mapping[str, Any],
        scaffold_context: _ScaffoldDerivativeContext,
        h_compiled: CompiledPolynomialAction,
        cfg: FullScoreConfig,
        optimizer_memory: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
  - CompatibilityPenaltyOracle
    Methods:
      - L1183: def __init__(
        self,
        *,
        cfg: FullScoreConfig,
        psi_state: np.ndarray | None = None,
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
    ) -> None:
      - L1196: def penalty(self, record_a: Mapping[str, Any], record_b: Mapping[str, Any]) -> dict[str, float]:

Functions:
  - L228: def _replace_feature(feat: CandidateFeatures, **updates: Any) -> CandidateFeatures:
  - L232: def normalize(value: float, ref: float) -> float:
  - L239: def trust_region_drop(g_lcb: float, h_eff: float, F: float, rho: float) -> float:
  - L252: def remaining_evaluations_proxy(
    *,
    current_depth: int | None,
    max_depth: int | None,
    mode: str,
) -> float:
  - L268: def family_repeat_cost_from_history(
    *,
    history_rows: Sequence[Mapping[str, Any]],
    candidate_family: str,
) -> float:
  - L289: def lifetime_weight_components(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> dict[str, float]:
  - L308: def _depth_life_cost(feat: CandidateFeatures, cfg: FullScoreConfig) -> float:
  - L316: def _score_denominator(feat: CandidateFeatures, cfg: FullScoreConfig) -> float:
  - L330: def _screen_denominator(feat: CandidateFeatures, cfg: SimpleScoreConfig) -> float:
  - L347: def simple_v1_score(
    feat: CandidateFeatures,
    cfg: SimpleScoreConfig,
) -> float:
  - L362: def full_v2_score(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> tuple[float, str]:
  - L395: def shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    score_key: str = "simple_score",
) -> list[dict[str, Any]]:
  - L432: def _executor_for_terms(
    terms: Sequence[Any],
    *,
    pauli_action_cache: dict[str, Any] | None,
) -> CompiledAnsatzExecutor:
  - L446: def _rotation_triplet(vec: np.ndarray, step: Any, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  - L459: def _horizontal_tangent(psi_state: np.ndarray, dpsi: np.ndarray) -> np.ndarray:
  - L466: def _tangent_overlap_matrix(tangents: Sequence[np.ndarray]) -> np.ndarray:
  - L477: def _energy_hessian_entry(
    *,
    dpsi_left: np.ndarray,
    dpsi_right: np.ndarray,
    d2psi: np.ndarray,
    hpsi_state: np.ndarray,
    hdpsi_right: np.ndarray,
) -> float:
  - L494: def _propagate_executor_derivatives(
    *,
    executor: CompiledAnsatzExecutor,
    theta: np.ndarray,
    psi_ref: np.ndarray,
    active_indices: Sequence[int],
) -> tuple[np.ndarray, list[np.ndarray], list[list[np.ndarray]]]:
  - L558: def _propagate_append_candidate(
    *,
    candidate_term: Any,
    psi_state: np.ndarray,
    window_dpsi: Sequence[np.ndarray],
    pauli_action_cache: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
  - L605: def _regularized_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    base_ridge: float,
    growth_factor: float,
    max_steps: int,
    require_pd: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
  - L858: def build_full_candidate_features(
    *,
    base_feature: CandidateFeatures,
    candidate_term: Any,
    cfg: FullScoreConfig,
    novelty_oracle: NoveltyOracle,
    curvature_oracle: CurvatureOracle,
    scaffold_context: _ScaffoldDerivativeContext,
    h_compiled: CompiledPolynomialAction,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    optimizer_memory: Mapping[str, Any] | None = None,
    motif_library: Mapping[str, Any] | None = None,
    target_num_sites: int | None = None,
) -> CandidateFeatures:
  - L947: def build_candidate_features(
    *,
    stage_name: str,
    candidate_label: str,
    candidate_family: str,
    candidate_pool_index: int,
    position_id: int,
    append_position: int,
    positions_considered: list[int],
    gradient_signed: float,
    metric_proxy: float,
    sigma_hat: float,
    refit_window_indices: list[int],
    compile_cost: CompileCostEstimate,
    measurement_stats: MeasurementCacheStats,
    leakage_penalty: float,
    stage_gate_open: bool,
    leakage_gate_open: bool,
    trough_probe_triggered: bool,
    trough_detected: bool,
    cfg: SimpleScoreConfig,
    generator_metadata: Mapping[str, Any] | None = None,
    symmetry_spec: Mapping[str, Any] | None = None,
    symmetry_mode: str = "none",
    symmetry_mitigation_mode: str = "off",
    motif_metadata: Mapping[str, Any] | None = None,
    motif_bonus: float = 0.0,
    motif_source: str = "none",
    current_depth: int | None = None,
    max_depth: int | None = None,
    lifetime_cost_mode: str = "off",
    remaining_evaluations_proxy_mode: str = "none",
    family_repeat_cost: float = 0.0,
) -> CandidateFeatures:
  - L1085: def _pauli_labels_from_term(term: Any) -> list[str]:
  - L1094: def _support_set(term: Any) -> set[int]:
  - L1104: def _pauli_strings_commute(lhs: str, rhs: str) -> bool:
  - L1113: def _polynomials_commute(term_a: Any, term_b: Any) -> bool:
  - L1125: def compatibility_penalty(
    *,
    record_a: Mapping[str, Any],
    record_b: Mapping[str, Any],
    cfg: FullScoreConfig,
    psi_state: np.ndarray | None = None,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> dict[str, float]:
  - L1207: def greedy_batch_select(
    ranked_records: Sequence[Mapping[str, Any]],
    compat_oracle: CompatibilityPenaltyOracle,
    cfg: FullScoreConfig,
) -> tuple[list[dict[str, Any]], float]:
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_stage_control.py
Imports:
  - from dataclasses import dataclass
  - from typing import Iterable
---
Classes:
  - StageControllerConfig
    Properties:
      - plateau_patience
      - weak_drop_threshold
      - probe_margin_ratio
      - max_probe_positions
      - append_admit_threshold
      - family_repeat_patience
  - PositionProbeDecision
    Properties:
      - should_probe
      - reason
      - positions
  - StageController
    Methods:
      - L93: def __init__(self, cfg: StageControllerConfig) -> None:
      - L97: def clone(self) -> "StageController":
      - L102: def snapshot(self) -> dict[str, str | StageControllerConfig]:
      - L109: def from_snapshot(cls, snapshot: dict[str, object]) -> "StageController":
      - L118: def stage_name(self) -> str:
      - L121: def start_with_seed(self) -> None:
      - L124: def begin_core(self) -> None:
      - L127: def resolve_stage_transition(
        self,
        *,
        drop_plateau_hits: int,
        trough_detected: bool,
        residual_opened: bool,
    ) -> tuple[str, str]:

Functions:
  - L27: def allowed_positions(
    *,
    n_params: int,
    append_position: int,
    active_window_indices: Iterable[int],
    max_positions: int,
) -> list[int]:
  - L52: def detect_trough(
    *,
    append_score: float,
    best_non_append_score: float,
    best_non_append_g_lcb: float,
    margin_ratio: float,
    append_admit_threshold: float,
) -> bool:
  - L70: def should_probe_positions(
    *,
    stage_name: str,
    drop_plateau_hits: int,
    max_grad: float,
    eps_grad: float,
    append_score: float,
    finite_angle_flat: bool,
    repeated_family_flat: bool,
    cfg: StageControllerConfig,
) -> tuple[bool, str]:
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_generators.py
Imports:
  - from dataclasses import asdict
  - import hashlib
  - from typing import Any, Mapping, Sequence
  - from pipelines.hardcoded.hh_continuation_types import GeneratorMetadata, GeneratorSplitEvent
  - from src.quantum.pauli_polynomial_class import PauliPolynomial
  - from src.quantum.qubitization_module import PauliTerm
---

Functions:
  - L15: def _polynomial_signature(poly: Any, *, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
  - L28: def _support_qubits(poly: Any) -> list[int]:
  - L40: def _qubit_to_site(
    qubit: int,
    *,
    num_sites: int,
    ordering: str,
    qpb: int,
) -> int:
  - L58: def _support_sites(
    support_qubits: Sequence[int],
    *,
    num_sites: int,
    ordering: str,
    qpb: int,
) -> list[int]:
  - L72: def _relative_site_offsets(sites: Sequence[int]) -> list[int]:
  - L79: def _template_id(
    *,
    family_id: str,
    support_site_offsets: Sequence[int],
    n_poly_terms: int,
    has_boson_support: bool,
    has_fermion_support: bool,
    is_macro_generator: bool,
) -> str:
  - L99: def _serialize_polynomial_terms(poly: Any, *, tol: float = 1e-12) -> list[dict[str, Any]]:
  - L116: def rebuild_polynomial_from_serialized_terms(
    serialized_terms: Sequence[Mapping[str, Any]],
) -> PauliPolynomial:
  - L139: def build_generator_metadata(
    *,
    label: str,
    polynomial: Any,
    family_id: str,
    num_sites: int,
    ordering: str,
    qpb: int,
    split_policy: str = "preserve",
    parent_generator_id: str | None = None,
    symmetry_spec: Mapping[str, Any] | None = None,
) -> GeneratorMetadata:
  - L200: def build_pool_generator_registry(
    *,
    terms: Sequence[Any],
    family_ids: Sequence[str],
    num_sites: int,
    ordering: str,
    qpb: int,
    symmetry_specs: Sequence[Mapping[str, Any] | None] | None = None,
    split_policy: str = "preserve",
) -> dict[str, dict[str, Any]]:
  - L229: def build_runtime_split_children(
    *,
    parent_label: str,
    polynomial: Any,
    family_id: str,
    num_sites: int,
    ordering: str,
    qpb: int,
    split_mode: str,
    parent_generator_metadata: Mapping[str, Any] | None = None,
    symmetry_spec: Mapping[str, Any] | None = None,
    max_children: int | None = None,
    tol: float = 1e-12,
) -> list[dict[str, Any]]:
  - L292: def selected_generator_metadata_for_labels(
    labels: Sequence[str],
    registry: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
  - L304: def build_split_event(
    *,
    parent_generator_id: str,
    child_generator_ids: Sequence[str],
    reason: str,
    split_mode: str,
) -> dict[str, Any]:
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_replay.py
Imports:
  - from dataclasses import dataclass
  - from typing import Any, Callable, Mapping, Sequence
  - import numpy as np
  - from pipelines.hardcoded.hh_continuation_types import (
    Phase2OptimizerMemoryAdapter,
    QNSPSARefreshPlan,
    ReplayPhaseTelemetry,
    ReplayPlan,
)
---
Classes:
  - ReplayControllerConfig
    Properties:
      - freeze_fraction
      - unfreeze_fraction
      - full_fraction
      - trust_radius_initial
      - trust_radius_growth
      - trust_radius_max
      - qn_spsa_refresh_every
      - qn_spsa_refresh_mode
      - symmetry_mitigation_mode
  - RestrictedAnsatzView
    Methods:
      - L35: def __init__(
        self,
        *,
        base_ansatz: Any,
        base_point: np.ndarray,
        active_indices: Sequence[int],
    ) -> None:
      - L47: def _merge(self, x: np.ndarray) -> np.ndarray:
      - L54: def prepare_state(self, x: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
      - L57: def parameter_labels(self) -> list[str]:
  - _ReplayNoopResult
    Properties:
      - energy
      - nfev
      - nit
      - success
      - message

Functions:
  - L65: def build_replay_plan(
    *,
    continuation_mode: str,
    seed_policy_resolved: str,
    handoff_state_kind: str,
    scaffold_block_indices: Sequence[int],
    residual_block_indices: Sequence[int],
    maxiter: int,
    cfg: ReplayControllerConfig,
    symmetry_mitigation_mode: str = "off",
    generator_ids: Sequence[str] | None = None,
    motif_reference_ids: Sequence[str] | None = None,
) -> ReplayPlan:
  - L117: def _run_phase(
    *,
    phase_name: str,
    vqe_minimize_fn: Callable[..., Any],
    h_poly: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    active_indices: list[int],
    full_theta_seed: np.ndarray,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    progress_every_s: float,
    exact_energy: float | None,
    kwargs: dict[str, Any],
    optimizer_memory: Mapping[str, Any] | None,
    refresh_plan: QNSPSARefreshPlan,
) -> tuple[np.ndarray, ReplayPhaseTelemetry, Any]:
  - L197: def _refresh_plan_for_phase(
    *,
    phase_name: str,
    method: str,
    cfg: ReplayControllerConfig,
) -> QNSPSARefreshPlan:
  - L217: def _seed_full_optimizer_memory(
    *,
    adapter: Phase2OptimizerMemoryAdapter,
    incoming_memory: Mapping[str, Any] | None,
    total_parameters: int,
    scaffold_block_size: int,
) -> tuple[dict[str, Any], str, bool]:
  - L268: def _run_phase_replay_controller(
    *,
    continuation_mode: str,
    vqe_minimize_fn: Callable[..., Any],
    h_poly: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    seed_theta: np.ndarray,
    scaffold_block_size: int,
    seed_policy_resolved: str,
    handoff_state_kind: str,
    cfg: ReplayControllerConfig,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    progress_every_s: float,
    exact_energy: float | None = None,
    kwargs: dict[str, Any] | None = None,
    incoming_optimizer_memory: Mapping[str, Any] | None = None,
    symmetry_mitigation_mode: str = "off",
    generator_ids: Sequence[str] | None = None,
    motif_reference_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
  - L470: def run_phase1_replay(
    *,
    vqe_minimize_fn: Callable[..., Any],
    h_poly: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    seed_theta: np.ndarray,
    scaffold_block_size: int,
    seed_policy_resolved: str,
    handoff_state_kind: str,
    cfg: ReplayControllerConfig,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    progress_every_s: float,
    exact_energy: float | None = None,
    kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
  - L527: def run_phase2_replay(
    *,
    vqe_minimize_fn: Callable[..., Any],
    h_poly: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    seed_theta: np.ndarray,
    scaffold_block_size: int,
    seed_policy_resolved: str,
    handoff_state_kind: str,
    cfg: ReplayControllerConfig,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    progress_every_s: float,
    exact_energy: float | None = None,
    kwargs: dict[str, Any] | None = None,
    incoming_optimizer_memory: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
  - L572: def run_phase3_replay(
    *,
    vqe_minimize_fn: Callable[..., Any],
    h_poly: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    seed_theta: np.ndarray,
    scaffold_block_size: int,
    seed_policy_resolved: str,
    handoff_state_kind: str,
    cfg: ReplayControllerConfig,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    progress_every_s: float,
    exact_energy: float | None = None,
    kwargs: dict[str, Any] | None = None,
    incoming_optimizer_memory: Mapping[str, Any] | None = None,
    generator_ids: Sequence[str] | None = None,
    motif_reference_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_types.py
Imports:
  - from dataclasses import dataclass, field
  - from typing import Any, Mapping, Protocol, Sequence
---
Classes:
  - CandidateFeatures
    Properties:
      - stage_name
      - candidate_label
      - candidate_family
      - candidate_pool_index
      - position_id
      - append_position
      - positions_considered
      - g_signed
      - g_abs
      - g_lcb
      - sigma_hat
      - F_metric
      - metric_proxy
      - novelty
      - curvature_mode
      - novelty_mode
      - refit_window_indices
      - compiled_position_cost_proxy
      - measurement_cache_stats
      - leakage_penalty
      - stage_gate_open
      - leakage_gate_open
      - trough_probe_triggered
      - trough_detected
      - simple_score
      - score_version
      - F_raw
      - Q_window
      - q_window
      - h_raw
      - b_mixed
      - H_window_hessian
      - M_window
      - h_eff
      - F_red
      - q_reduced
      - ridge_used
      - h_hat
      - b_hat
      - H_window
      - depth_cost
      - new_group_cost
      - new_shot_cost
      - opt_dim_cost
      - reuse_count_cost
      - family_repeat_cost
      - full_v2_score
      - shortlist_rank
      - shortlist_size
      - actual_fallback_mode
      - compatibility_penalty_total
      - generator_id
      - template_id
      - is_macro_generator
      - parent_generator_id
      - runtime_split_mode
      - runtime_split_parent_label
      - runtime_split_child_index
      - runtime_split_child_count
      - generator_metadata
      - symmetry_spec
      - symmetry_mode
      - symmetry_mitigation_mode
      - motif_metadata
      - motif_bonus
      - motif_source
      - remaining_evaluations_proxy
      - remaining_evaluations_proxy_mode
      - lifetime_cost_mode
      - lifetime_weight_components
      - placeholder_hooks
  - MeasurementPlan
    Properties:
      - plan_version
      - group_keys
      - nominal_shots_per_group
      - grouping_mode
  - MeasurementCacheStats
    Properties:
      - groups_total
      - groups_reused
      - groups_new
      - shots_reused
      - shots_new
      - reuse_count_cost
  - CompileCostEstimate
    Properties:
      - new_pauli_actions
      - new_rotation_steps
      - position_shift_span
      - refit_active_count
      - proxy_total
  - ScaffoldFingerprintLite
    Properties:
      - selected_operator_labels
      - selected_generator_ids
      - num_parameters
      - generator_family
      - continuation_mode
      - compiled_pauli_cache_size
      - measurement_plan_version
      - post_prune
      - split_event_count
      - motif_record_ids
  - PruneDecision
    Properties:
      - index
      - label
      - accepted
      - energy_before
      - energy_after
      - regression
      - reason
  - ReplayPlan
    Properties:
      - continuation_mode
      - seed_policy_resolved
      - handoff_state_kind
      - freeze_scaffold_steps
      - unfreeze_steps
      - full_replay_steps
      - trust_radius_initial
      - trust_radius_growth
      - trust_radius_max
      - scaffold_block_indices
      - residual_block_indices
      - qn_spsa_refresh_every
      - trust_radius_schedule
      - optimizer_memory_source
      - optimizer_memory_reused
      - refresh_mode
      - symmetry_mitigation_mode
      - generator_ids
      - motif_reference_ids
  - ReplayPhaseTelemetry
    Properties:
      - phase_name
      - nfev
      - nit
      - success
      - energy_before
      - energy_after
      - delta_abs_before
      - delta_abs_after
      - active_count
      - frozen_count
      - optimizer_memory_reused
      - optimizer_memory_source
      - qn_spsa_refresh_points
      - residual_zero_initialized
  - NoveltyOracle
    Methods:
      - L179: def estimate(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
  - CurvatureOracle
    Methods:
      - L184: def estimate(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
  - OptimizerMemoryAdapter
    Methods:
      - L189: def unavailable(self, *, method: str, parameter_count: int, reason: str) -> dict[str, Any]:
      - L192: def from_result(
        self,
        result: Any,
        *,
        method: str,
        parameter_count: int,
        source: str,
    ) -> dict[str, Any]:
      - L202: def remap_insert(
        self,
        state: Mapping[str, Any] | None,
        *,
        position_id: int,
        count: int = 1,
    ) -> dict[str, Any]:
      - L211: def remap_remove(
        self,
        state: Mapping[str, Any] | None,
        *,
        indices: Sequence[int],
    ) -> dict[str, Any]:
      - L219: def select_active(
        self,
        state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        source: str,
    ) -> dict[str, Any]:
      - L228: def merge_active(
        self,
        base_state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        active_state: Mapping[str, Any] | None,
        source: str,
    ) -> dict[str, Any]:
  - QNSPSARefreshPlan
    Properties:
      - enabled
      - refresh_every
      - mode
      - skip_reason
      - refresh_points
  - MotifMetadata
    Properties:
      - enabled
      - motif_tags
      - motif_ids
      - motif_source
      - tiled_from_num_sites
      - target_num_sites
      - boundary_behavior
      - transfer_mode
  - SymmetrySpec
    Properties:
      - spec_version
      - particle_number_mode
      - spin_sector_mode
      - phonon_number_mode
      - leakage_risk
      - mitigation_eligible
      - grouping_eligible
      - hard_guard
      - tags
  - GeneratorMetadata
    Properties:
      - generator_id
      - family_id
      - template_id
      - candidate_label
      - support_qubits
      - support_sites
      - support_site_offsets
      - is_macro_generator
      - split_policy
      - parent_generator_id
      - symmetry_spec
      - compile_metadata
  - GeneratorSplitEvent
    Properties:
      - parent_generator_id
      - child_generator_ids
      - reason
      - split_mode
  - MotifRecord
    Properties:
      - motif_id
      - family_id
      - template_id
      - source_num_sites
      - relative_order
      - support_site_offsets
      - mean_theta
      - mean_abs_theta
      - sign_hint
      - generator_ids
      - symmetry_spec
      - boundary_behavior
      - source_tags
  - MotifLibrary
    Properties:
      - library_version
      - source_tag
      - source_num_sites
      - ordering
      - boson_encoding
      - source_tags
      - records
  - RescueDiagnostic
    Properties:
      - enabled
      - triggered
      - reason
      - shortlisted_labels
      - selected_label
      - selected_position
      - overlap_gain
  - Phase2OptimizerMemoryAdapter
    Methods:
      - L344: def unavailable(self, *, method: str, parameter_count: int, reason: str) -> dict[str, Any]:
      - L360: def from_result(
        self,
        result: Any,
        *,
        method: str,
        parameter_count: int,
        source: str,
    ) -> dict[str, Any]:
      - L379: def remap_insert(
        self,
        state: Mapping[str, Any] | None,
        *,
        position_id: int,
        count: int = 1,
    ) -> dict[str, Any]:
      - L397: def remap_remove(
        self,
        state: Mapping[str, Any] | None,
        *,
        indices: Sequence[int],
    ) -> dict[str, Any]:
      - L414: def select_active(
        self,
        state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        source: str,
    ) -> dict[str, Any]:
      - L436: def merge_active(
        self,
        base_state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        active_state: Mapping[str, Any] | None,
        source: str,
    ) -> dict[str, Any]:
      - L464: def _parameter_count(self, state: Mapping[str, Any] | None) -> int:
      - L474: def _normalize(self, state: Mapping[str, Any] | None, *, parameter_count: int) -> dict[str, Any]:
      - L498: def _append_remap_event(self, state: dict[str, Any], event: Mapping[str, Any]) -> None:
    Properties:
      - _VECTOR_KEYS
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_symmetry.py
Imports:
  - from dataclasses import asdict
  - from typing import Any, Mapping, Sequence
  - from pipelines.hardcoded.hh_continuation_types import SymmetrySpec
---

Functions:
  - L30: def normalize_phase3_symmetry_mitigation_mode(mode: str | None) -> str:
  - L42: def build_symmetry_spec(
    *,
    family_id: str,
    mitigation_mode: str = "off",
) -> SymmetrySpec:
  - L70: def symmetry_spec_to_dict(spec: SymmetrySpec | Mapping[str, Any] | None) -> dict[str, Any] | None:
  - L78: def leakage_penalty_from_spec(spec: SymmetrySpec | Mapping[str, Any] | None) -> float:
  - L86: def verify_symmetry_sequence(
    *,
    generator_metadata: Sequence[Mapping[str, Any]],
    mitigation_mode: str,
) -> dict[str, Any]:

Global vars:
  - _LOW_RISK_FAMILIES
  - _ALLOWED_SYMMETRY_MODES
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_pruning.py
Imports:
  - from dataclasses import dataclass
  - from typing import Callable
  - import numpy as np
  - from pipelines.hardcoded.hh_continuation_types import PruneDecision
---
Classes:
  - PruneConfig
    Properties:
      - max_candidates
      - min_candidates
      - fraction_candidates
      - max_regression

Functions:
  - L22: def rank_prune_candidates(
    *,
    theta: np.ndarray,
    labels: list[str],
    marginal_proxy_benefit: list[float] | None,
    max_candidates: int,
    min_candidates: int,
    fraction_candidates: float,
) -> list[int]:
  - L39: def _benefit_key(i: int) -> float:
  - L58: def apply_pruning(
    *,
    theta: np.ndarray,
    labels: list[str],
    candidate_indices: list[int],
    eval_with_removal: Callable[..., tuple[float, np.ndarray]],
    energy_before: float,
    max_regression: float,
) -> tuple[np.ndarray, list[str], list[PruneDecision], float]:
  - L105: def post_prune_refit(
    *,
    theta: np.ndarray,
    refit_fn: Callable[[np.ndarray], tuple[np.ndarray, float]],
) -> tuple[np.ndarray, float]:
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_rescue.py
Imports:
  - from dataclasses import dataclass
  - from typing import Any, Callable, Mapping, Sequence
---
Classes:
  - RescueConfig
    Properties:
      - enabled
      - simulator_only
      - recent_drop_patience
      - weak_drop_threshold
      - shortlist_flat_ratio
      - max_candidates
      - min_overlap_gain

Functions:
  - L21: def should_trigger_rescue(
    *,
    enabled: bool,
    exact_state_available: bool,
    residual_opened: bool,
    trough_detected: bool,
    history: Sequence[Mapping[str, Any]],
    shortlist_records: Sequence[Mapping[str, Any]],
    cfg: RescueConfig,
) -> tuple[bool, str]:
  - L54: def rank_rescue_candidates(
    *,
    records: Sequence[Mapping[str, Any]],
    overlap_gain_fn: Callable[[Mapping[str, Any]], float],
    cfg: RescueConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/handoff_state_bundle.py
Imports:
  - import json
  - import math
  - from dataclasses import dataclass
  - from datetime import datetime, timezone
  - from pathlib import Path
  - from typing import Any
  - import numpy as np
---
Classes:
  - HandoffStateBundleConfig
    Properties:
      - L
      - t
      - U
      - dv
      - omega0
      - g_ep
      - n_ph_max
      - boson_encoding
      - ordering
      - boundary
      - sector_n_up
      - sector_n_dn

Functions:
  - L32: def build_handoff_settings_manifest(
    cfg: HandoffStateBundleConfig,
    *,
    adapt_pool: str | None = None,
) -> dict[str, Any]:
  - L57: def _statevector_to_amplitudes_qn_to_q0(
    psi_state: np.ndarray,
    *,
    cutoff: float,
) -> dict[str, dict[str, float]]:
  - L75: def write_handoff_state_bundle(
    *,
    path: Path,
    psi_state: np.ndarray,
    cfg: HandoffStateBundleConfig,
    source: str,
    exact_energy: float,
    energy: float,
    delta_E_abs: float,
    relative_error_abs: float,
    meta: dict[str, Any] | None = None,
    adapt_operators: list[str] | None = None,
    adapt_optimal_point: list[float] | None = None,
    adapt_pool_type: str | None = None,
    settings_adapt_pool: str | None = None,
    handoff_state_kind: str | None = None,
    continuation_mode: str | None = None,
    continuation_scaffold: dict[str, Any] | None = None,
    optimizer_memory: dict[str, Any] | None = None,
    selected_generator_metadata: list[dict[str, Any]] | None = None,
    generator_split_events: list[dict[str, Any]] | None = None,
    motif_library: dict[str, Any] | None = None,
    motif_usage: dict[str, Any] | None = None,
    symmetry_mitigation: dict[str, Any] | None = None,
    rescue_history: list[dict[str, Any]] | None = None,
    prune_summary: dict[str, Any] | None = None,
    pre_prune_scaffold: dict[str, Any] | None = None,
    replay_contract_hint: dict[str, Any] | None = None,
    replay_contract: dict[str, Any] | None = None,
    vqe_payload: dict[str, Any] | None = None,
    seed_provenance: dict[str, Any] | None = None,
    amplitude_cutoff: float = 1e-14,
) -> None:
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_hh_adapt_beam_search.py
Imports:
  - import sys
  - from pathlib import Path
  - from types import SimpleNamespace
  - import numpy as np
  - from pipelines.hardcoded.adapt_pipeline import _BeamBranchState, _run_hardcoded_adapt_vqe
  - from pipelines.hardcoded.hh_continuation_scoring import MeasurementCacheAudit
  - from pipelines.hardcoded.hh_continuation_stage_control import StageController, StageControllerConfig
  - from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
---

Functions:
  - L19: def _make_branch() -> _BeamBranchState:
  - L64: def _hh_h() -> object:
  - L82: def test_beam_branch_clone_for_child_isolates_branch_owned_state() -> None:
  - L123: def test_beam_branch_sibling_clones_diverge_independently() -> None:
  - L155: def test_true_beam_defaults_are_exposed_and_winner_history_stays_singleton() -> None:

Global vars:
  - REPO_ROOT
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_hh_continuation_generators.py
Imports:
  - import sys
  - from pathlib import Path
  - from pipelines.hardcoded.hh_continuation_generators import (
    build_generator_metadata,
    build_pool_generator_registry,
    build_runtime_split_children,
    build_split_event,
)
  - from pipelines.hardcoded.hh_continuation_symmetry import build_symmetry_spec
  - from src.quantum.pauli_polynomial_class import PauliPolynomial
  - from src.quantum.pauli_words import PauliTerm
---

Functions:
  - L21: def _term(label: str, poly: PauliPolynomial):
  - L25: def _macro_poly() -> PauliPolynomial:
  - L35: def test_build_generator_metadata_is_stable_for_same_structure() -> None:
  - L61: def test_pool_registry_carries_symmetry_metadata() -> None:
  - L77: def test_deliberate_split_marks_child_metadata() -> None:
  - L93: def test_build_split_event_keeps_parent_child_provenance() -> None:
  - L105: def test_build_runtime_split_children_emits_serialized_single_term_children() -> None:

Global vars:
  - REPO_ROOT
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_hh_continuation_replay.py
Imports:
  - from dataclasses import dataclass
  - import sys
  - from pathlib import Path
  - import numpy as np
  - from pipelines.hardcoded.hh_continuation_replay import (
    ReplayControllerConfig,
    build_replay_plan,
    run_phase1_replay,
    run_phase2_replay,
    run_phase3_replay,
)
---
Classes:
  - _DummyAnsatz
    Methods:
      - L23: def __init__(self, npar: int) -> None:
      - L26: def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
  - _DummyRes
    Properties:
      - x
      - energy
      - nfev
      - nit
      - success
      - message
      - best_restart
      - restart_summaries
      - optimizer_memory

Functions:
  - L46: def _fake_vqe_minimize(_h, ansatz, _psi_ref, **kwargs):
  - L75: def test_build_replay_plan_splits_steps() -> None:
  - L95: def test_run_phase1_replay_emits_phase_history() -> None:
  - L129: def test_run_phase2_replay_reuses_memory_and_logs_refresh() -> None:
  - L174: def test_run_phase2_replay_missing_optimizer_memory_degrades_gracefully() -> None:
  - L206: def test_run_phase3_replay_emits_generator_motif_and_symmetry_fields() -> None:
  - L256: def test_run_phase3_replay_skips_empty_seed_burn_in_phase() -> None:

Global vars:
  - REPO_ROOT
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_hh_continuation_scoring.py
Imports:
  - import sys
  - from pathlib import Path
  - import numpy as np
  - import pytest
  - from pipelines.hardcoded.hh_continuation_generators import build_generator_metadata
  - from pipelines.hardcoded.hh_continuation_scoring import (
    FullScoreConfig,
    MeasurementCacheAudit,
    Phase2CurvatureOracle,
    Phase2NoveltyOracle,
    Phase1CompileCostOracle,
    SimpleScoreConfig,
    build_candidate_features,
    build_full_candidate_features,
    family_repeat_cost_from_history,
    full_v2_score,
    lifetime_weight_components,
    remaining_evaluations_proxy,
    shortlist_records,
    trust_region_drop,
)
  - from pipelines.hardcoded.hh_continuation_symmetry import build_symmetry_spec
  - from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
  - from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action
  - from src.quantum.pauli_polynomial_class import PauliPolynomial
  - from src.quantum.pauli_words import PauliTerm
---
Classes:
  - _CountingNovelty
    Methods:
      - L262: def __init__(self) -> None:
      - L265: def estimate(self, *args, **kwargs):

Functions:
  - L37: def _term(label: str) -> object:
  - L45: def _feat(
    *,
    gradient_signed: float = 0.4,
    metric_proxy: float = 0.5,
    sigma_hat: float = 0.0,
    refit_window_indices: list[int] | None = None,
    family_repeat_cost: float = 0.0,
    stage_gate_open: bool = True,
    cfg: SimpleScoreConfig | None = None,
) -> object:
  - L82: def test_simple_v1_prefers_higher_gradient_with_equal_costs() -> None:
  - L89: def test_stage_gate_blocks_score() -> None:
  - L94: def test_simple_v1_uses_g_lcb_not_g_abs() -> None:
  - L101: def test_family_repeat_cost_lowers_screen_score() -> None:
  - L108: def test_measurement_cache_reuse_accounting() -> None:
  - L120: def test_measurement_cache_clone_isolated_from_parent_and_sibling() -> None:
  - L138: def test_measurement_cache_snapshot_roundtrip() -> None:
  - L148: def test_trust_region_drop_matches_newton_branch() -> None:
  - L153: def test_full_v2_uses_reduced_fields() -> None:
  - L171: def test_full_v2_zeroes_metric_collapse() -> None:
  - L190: def test_full_v2_ignores_motif_bonus_in_active_score() -> None:
  - L210: def test_build_full_candidate_features_emits_reduced_path_fields() -> None:
  - L260: def test_shortlist_only_expensive_scoring_calls_oracles_for_shortlist() -> None:
  - L321: def test_remaining_evaluations_proxy_uses_remaining_depth_mode() -> None:
  - L326: def test_lifetime_weight_components_are_zero_when_mode_off() -> None:
  - L346: def test_family_repeat_cost_from_history_uses_consecutive_streak() -> None:
  - L356: def test_build_candidate_features_carries_generator_and_symmetry_metadata() -> None:

Global vars:
  - REPO_ROOT
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_projected_real_time.py
Imports:
  - from pathlib import Path
  - import sys
  - import numpy as np
  - from src.quantum.pauli_polynomial_class import PauliPolynomial
  - from src.quantum.qubitization_module import PauliTerm
  - from src.quantum.time_propagation.projected_real_time import (
    ProjectedRealTimeConfig,
    build_tangent_vectors,
    run_exact_driven_reference,
    run_projected_real_time_trajectory,
    solve_mclachlan_step,
    state_fidelity,
)
  - from src.quantum.vqe_latex_python_pairs import AnsatzTerm
---

Functions:
  - L25: def _basis(dim: int, idx: int) -> np.ndarray:
  - L31: def _term(label: str, *, nq: int = 1, coeff: complex = 1.0) -> AnsatzTerm:
  - L38: def test_build_tangent_vectors_matches_single_pauli_derivative() -> None:
  - L52: def test_solve_mclachlan_step_solves_regularized_system() -> None:
  - L63: def test_run_projected_real_time_trajectory_zero_hamiltonian_keeps_state() -> None:
  - L79: def test_run_exact_driven_reference_static_matches_analytic_state() -> None:

Global vars:
  - REPO_ROOT
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_local_checkpoint_fit.py
Imports:
  - from pathlib import Path
  - import sys
  - import numpy as np
  - from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
  - from src.quantum.time_propagation.local_checkpoint_fit import (
    CheckpointFitConfig,
    LocalPauliAnsatzSpec,
    build_local_pauli_ansatz_terms,
    fit_checkpoint_target_state,
    fit_checkpoint_trajectory,
)
---

Functions:
  - L22: def test_build_local_pauli_ansatz_terms_counts_chain_terms() -> None:
  - L36: def test_fit_checkpoint_target_state_recovers_one_qubit_rotation() -> None:
  - L62: def test_fit_checkpoint_trajectory_warm_starts_across_times() -> None:

Global vars:
  - REPO_ROOT
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_staged_export_replay_roundtrip.py
Imports:
  - import json
  - import math
  - from pathlib import Path
  - import sys
  - import numpy as np
  - import pytest
  - from pipelines.hardcoded.hh_vqe_from_adapt_family import (
    _extract_adapt_operator_theta_sequence,
    _extract_replay_contract,
    _infer_handoff_state_kind,
    _resolve_family_from_metadata,
)
  - from pipelines.hardcoded.handoff_state_bundle import (
    HandoffStateBundleConfig,
    write_handoff_state_bundle,
)
---
Classes:
  - TestCanonicalReplayFieldsPresent
    Methods:
      - L116: def test_extract_succeeds_for_staged_export(self) -> None:
      - L122: def test_operators_and_optimal_point_length_match(self) -> None:
      - L127: def test_pool_type_resolves_pool_a(self) -> None:
      - L133: def test_pool_type_resolves_pool_b(self) -> None:
      - L139: def test_seed_provenance_is_ignored_for_replay_family_resolution(self) -> None:
      - L152: def test_ansatz_depth_matches_operators(self) -> None:
      - L158: def test_settings_has_required_hh_keys(self) -> None:
      - L168: def test_initial_state_has_amplitudes(self) -> None:
  - TestStagedExportRejectsIncomplete
    Methods:
      - L177: def test_missing_operators_rejected(self) -> None:
      - L183: def test_missing_optimal_point_rejected(self) -> None:
      - L189: def test_length_mismatch_rejected(self) -> None:
  - TestArbitraryLRoundTrip
    Methods:
      - L199: def test_roundtrip_for_L(self, L: int) -> None:
      - L213: def test_settings_sector_half_filling(self, L: int) -> None:
  - TestWriteStateBundleRoundTrip
    Methods:
      - L223: def test_write_and_read_back(self, tmp_path: Path) -> None:
      - L336: def test_write_and_read_back_with_seed_provenance(self, tmp_path: Path) -> None:
      - L376: def test_write_with_contract_is_parseable(self, tmp_path: Path) -> None:
      - L433: def test_write_sparse_bundle_without_continuation_preserves_legacy_load(self, tmp_path: Path) -> None:
  - TestStagedExportProvenance
    Methods:
      - L495: def test_staged_export_has_prepared_state_kind(self) -> None:
      - L499: def test_provenance_is_explicit(self) -> None:
      - L506: def test_provenance_present_for_all_L(self, L: int) -> None:
      - L513: def test_legacy_payload_without_provenance_infers_from_source(self) -> None:
      - L522: def test_write_state_bundle_stamps_provenance(self, tmp_path: Path) -> None:

Functions:
  - L38: def _make_staged_export_payload(
    *,
    L: int = 2,
    operators: list[str] | None = None,
    optimal_point: list[float] | None = None,
    pool_type: str | None = None,
) -> dict:

Global vars:
  - REPO_ROOT
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_vqe_from_adapt_family.py
Imports:
  - import argparse
  - import csv
  - import gc
  - import json
  - import math
  - import sys
  - import time
  - from dataclasses import dataclass
  - from datetime import datetime, timezone
  - from pathlib import Path
  - from typing import Any, Mapping, Optional, Sequence
  - import numpy as np
  - from pipelines.hardcoded.adapt_pipeline import (
    _HH_UCCSD_PAOP_PRODUCT_SPECS,
    _build_hh_termwise_augmented_pool,
    _build_hh_full_meta_pool,
    _build_hh_all_meta_v1_pool,
    _build_hh_uccsd_fermion_lifted_pool,
    _build_hh_uccsd_paop_product_pool,
    _build_hva_pool,
    _build_paop_pool,
    _build_vlf_sq_pool,
    _deduplicate_pool_terms,
    _deduplicate_pool_terms_lightweight,
)
  - from pipelines.hardcoded.hh_continuation_generators import rebuild_polynomial_from_serialized_terms
  - from src.quantum.operator_pools.polaron_paop import _make_paop_core
  - from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_holstein_hamiltonian,
)
  - from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    apply_exp_pauli_polynomial,
    expval_pauli_polynomial,
    exact_ground_energy_sector_hh,
    vqe_minimize,
)
  - from pipelines.hardcoded.hh_continuation_replay import (
    ReplayControllerConfig,
    run_phase1_replay,
    run_phase2_replay,
    run_phase3_replay,
)
---
Classes:
  - RunConfig
    Properties:
      - adapt_input_json
      - output_json
      - output_csv
      - output_md
      - output_log
      - tag
      - generator_family
      - fallback_family
      - legacy_paop_key
      - replay_seed_policy
      - replay_continuation_mode
      - L
      - t
      - u
      - dv
      - omega0
      - g_ep
      - n_ph_max
      - boson_encoding
      - ordering
      - boundary
      - sector_n_up
      - sector_n_dn
      - reps
      - restarts
      - maxiter
      - method
      - seed
      - energy_backend
      - progress_every_s
      - wallclock_cap_s
      - paop_r
      - paop_split_paulis
      - paop_prune_eps
      - paop_normalization
      - spsa_a
      - spsa_c
      - spsa_alpha
      - spsa_gamma
      - spsa_A
      - spsa_avg_last
      - spsa_eval_repeats
      - spsa_eval_agg
      - replay_freeze_fraction
      - replay_unfreeze_fraction
      - replay_full_fraction
      - replay_qn_spsa_refresh_every
      - replay_qn_spsa_refresh_mode
      - phase3_symmetry_mitigation_mode
  - RunLogger
    Methods:
      - L335: def __init__(self, path: Path) -> None:
      - L340: def log(self, msg: str) -> None:
  - PoolTermwiseAnsatz
    Methods:
      - L350: def __init__(self, *, terms: list[AnsatzTerm], reps: int, nq: int) -> None:
      - L358: def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: Optional[float] = None,
        sort_terms: Optional[bool] = None,
    ) -> np.ndarray:

Functions:
  - L103: def _now_utc() -> str:
  - L107: def _canonical_family(raw: Any) -> str | None:
  - L114: def _extract_nested(payload: Mapping[str, Any], *keys: str) -> Any:
  - L123: def _resolve_exact_energy_from_payload(payload: Mapping[str, Any]) -> float | None:
  - L142: def _resolve_family_from_metadata_sequence(
    records: Any,
    *,
    family_key: str,
) -> tuple[str | None, str | None]:
  - L164: def _resolve_family_from_metadata(payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
  - L220: def _parse_int_setting(settings: Mapping[str, Any], key: str, fallback_key: str | None = None) -> int | None:
  - L229: def _parse_float_setting(settings: Mapping[str, Any], key: str, fallback_key: str | None = None) -> float | None:
  - L238: def _half_filled_particles(num_sites: int) -> tuple[int, int]:
  - L242: def _require(name: str, val: Any) -> Any:
  - L248: def _amplitudes_qn_to_q0_to_statevector(payload: Mapping[str, Any], *, nq: int) -> np.ndarray:
  - L267: def _statevector_to_amplitudes_qn_to_q0(
    vec: np.ndarray, *, cutoff: float = 1e-14
) -> dict[str, dict[str, float]]:
  - L394: def _build_hh_hamiltonian(cfg: RunConfig) -> Any:
  - L411: def _dedup_terms(terms: list[AnsatzTerm], *, n_ph_max: int) -> list[AnsatzTerm]:
  - L417: def _build_pool_for_family(cfg: RunConfig, *, family: str, h_poly: Any) -> tuple[list[AnsatzTerm], dict[str, Any]]:
  - L669: def _build_pool_recipe(
    cfg: RunConfig,
    *,
    base_family: str,
    extra_families: Sequence[str],
    h_poly: Any,
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
  - L722: def _read_input_state_and_payload(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
  - L739: def _extract_adapt_operator_theta_sequence(payload: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
  - L784: def _inject_replay_terms_from_payload(
    label_to_term: dict[str, AnsatzTerm],
    payload: Mapping[str, Any] | None,
) -> None:
  - L817: def _build_replay_terms_from_adapt_labels(
    family_pool: Sequence[AnsatzTerm],
    adapt_labels: Sequence[str],
    payload: Mapping[str, Any] | None = None,
) -> list[AnsatzTerm]:
  - L856: def _build_replay_seed_theta(adapt_theta: np.ndarray, *, reps: int) -> np.ndarray:
  - L883: def _coerce_bool(value: Any, path: str) -> bool:
  - L897: def _extract_replay_contract(payload: Mapping[str, Any]) -> dict[str, Any] | None:
  - L1063: def _canonical_seed_policy(raw: Any) -> str | None:
  - L1070: def _infer_handoff_state_kind(
    payload: Mapping[str, Any],
) -> tuple[str, str]:
  - L1112: def _build_replay_seed_theta_policy(
    adapt_theta: np.ndarray,
    *,
    reps: int,
    policy: str,
    handoff_state_kind: str,
) -> tuple[np.ndarray, str]:
  - L1161: def _extract_canonical_label_family(label: str) -> str | None:
  - L1179: def _expand_product_labels_and_theta_for_family(
    labels: Sequence[str],
    theta: np.ndarray,
    *,
    family: str,
) -> tuple[list[str], np.ndarray, str | None]:
  - L1215: def _build_exact_label_subset_for_family(
    cfg: RunConfig,
    *,
    family: str,
    h_poly: Any,
    needed_labels: Sequence[str],
) -> tuple[list[AnsatzTerm], int]:
  - L1236: def _build_full_meta_replay_terms_sparse(
    cfg: RunConfig,
    *,
    h_poly: Any,
    adapt_labels: Sequence[str],
    payload: Mapping[str, Any] | None = None,
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
  - L1257: def _consume(component_key: str, terms: Sequence[AnsatzTerm], *, raw_count: int | None = None) -> None:
  - L1264: def _build_paop_subset(pool_name: str) -> tuple[list[AnsatzTerm], int]:
  - L1436: def _build_replay_terms_for_family(
    cfg: RunConfig,
    *,
    family: str,
    h_poly: Any,
    adapt_labels: Sequence[str],
    payload: Mapping[str, Any] | None = None,
) -> tuple[list[AnsatzTerm], dict[str, Any], int]:
  - L1460: def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
  - L1465: def _write_csv(path: Path, row: Mapping[str, Any]) -> None:
  - L1474: def _write_md(path: Path, lines: list[str]) -> None:
  - L1479: def _build_cfg(args: argparse.Namespace, payload: Mapping[str, Any]) -> RunConfig:
  - L1568: def _resolve_family(cfg: RunConfig, payload: Mapping[str, Any]) -> dict[str, Any]:
  - L1645: def _resolve_replay_continuation_mode(raw: str | None) -> str:
  - L1654: def _build_sequence_cfg_from_payload(
    adapt_input_json: Path,
    payload: Mapping[str, Any],
    *,
    generator_family: str,
    fallback_family: str,
    legacy_paop_key: str,
    replay_continuation_mode: str | None,
) -> RunConfig:
  - L1724: def build_replay_sequence_from_input_json(
    adapt_input_json: Path,
    *,
    generator_family: str = "match_adapt",
    fallback_family: str = "full_meta",
    legacy_paop_key: str = "paop_lf_std",
    replay_continuation_mode: str | None = "phase3_v1",
) -> dict[str, Any]:
  - L1785: def build_family_ansatz_context(
    cfg: RunConfig,
    *,
    psi_ref: np.ndarray,
    h_poly: Any,
    family: str,
    e_exact: float,
    seed_theta: Sequence[float] | np.ndarray | None = None,
    terms_override: Sequence[AnsatzTerm] | None = None,
    pool_meta_override: Mapping[str, Any] | None = None,
    family_terms_count_override: int | None = None,
    family_info_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  - L1857: def build_replay_ansatz_context(
    cfg: RunConfig,
    *,
    payload_in: Mapping[str, Any],
    psi_ref: np.ndarray,
    h_poly: Any,
    family_info: Mapping[str, Any],
    e_exact: float,
) -> dict[str, Any]:
  - L1976: def run(cfg: RunConfig, diagnostics_out: dict[str, Any] | None = None) -> dict[str, Any]:
  - L2077: def _progress_logger(ev: dict[str, Any]) -> None:
  - L2096: def _early_stop_checker(ev: dict[str, Any]) -> bool:
  - L2475: def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  - L2565: def main(argv: list[str] | None = None) -> int:

Global vars:
  - REPO_ROOT
  - EXPLICIT_FAMILIES
  - REPLAY_SEED_POLICIES
  - REPLAY_CONTRACT_VERSION
  - _PREPARED_STATE
  - _REFERENCE_STATE
  - _LEGACY_REFERENCE_SOURCES
  - _LEGACY_PREPARED_FINAL_SUFFIXES
  - _LEGACY_PREPARED_SOURCES
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/adapt_pipeline.py
Imports:
  - import argparse
  - import copy
  - import hashlib
  - import json
  - import math
  - import os
  - import re
  - import sys
  - import time
  - from dataclasses import dataclass
  - from datetime import datetime, timezone
  - from pathlib import Path
  - from types import SimpleNamespace
  - from typing import Any, Mapping, Sequence
  - import numpy as np
  - from docs.reports.pdf_utils import (
    HAS_MATPLOTLIB,
    require_matplotlib,
    get_plt,
    get_PdfPages,
    render_command_page,
    render_text_page,
    current_command_string,
)
  - from docs.reports.report_pages import (
    render_executive_summary_page,
    render_manifest_overview_page,
    render_section_divider_page,
)
  - from src.quantum.hubbard_latex_python_pairs import (
    bravais_nearest_neighbor_edges,
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
    boson_qubits_per_site,
)
  - from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
  - from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
  - from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial as _apply_compiled_polynomial_shared,
    compile_polynomial_action as _compile_polynomial_action_shared,
    energy_via_one_apply,
)
  - from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_compiled_pauli as _apply_compiled_pauli_shared,
    apply_exp_term as _apply_exp_term_shared,
    compile_pauli_action_exyz as _compile_pauli_action_exyz_shared,
)
  - from src.quantum.pauli_polynomial_class import PauliPolynomial
  - from src.quantum.pauli_words import PauliTerm
  - from src.quantum.spsa_optimizer import spsa_minimize
  - from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HardcodedUCCSDAnsatz,
    HubbardHolsteinLayerwiseAnsatz,
    HubbardTermwiseAnsatz,
    apply_exp_pauli_polynomial,
    apply_pauli_string,
    basis_state,
    exact_ground_energy_sector,
    exact_ground_energy_sector_hh,
    expval_pauli_polynomial,
    half_filled_num_particles,
    hamiltonian_matrix,
    hartree_fock_bitstring,
    hubbard_holstein_reference_state,
)
  - from pipelines.hardcoded.hh_continuation_types import (
    CandidateFeatures,
    Phase2OptimizerMemoryAdapter,
    ScaffoldFingerprintLite,
)
  - from pipelines.hardcoded.hh_continuation_generators import (
    build_pool_generator_registry,
    build_runtime_split_children,
    build_split_event,
    selected_generator_metadata_for_labels,
)
  - from pipelines.hardcoded.hh_continuation_motifs import (
    extract_motif_library,
    load_motif_library_from_json,
    select_tiled_generators_from_library,
)
  - from pipelines.hardcoded.hh_continuation_symmetry import (
    build_symmetry_spec,
    leakage_penalty_from_spec,
    verify_symmetry_sequence,
)
  - from pipelines.hardcoded.hh_continuation_rescue import (
    RescueConfig,
    rank_rescue_candidates,
    should_trigger_rescue,
)
  - from pipelines.hardcoded.hh_continuation_stage_control import (
    StageController,
    StageControllerConfig,
    allowed_positions,
    detect_trough,
    should_probe_positions,
)
  - from pipelines.hardcoded.hh_continuation_scoring import (
    CompatibilityPenaltyOracle,
    FullScoreConfig,
    MeasurementCacheAudit,
    Phase2CurvatureOracle,
    Phase1CompileCostOracle,
    Phase2NoveltyOracle,
    SimpleScoreConfig,
    build_candidate_features,
    build_full_candidate_features,
    family_repeat_cost_from_history,
    greedy_batch_select,
    shortlist_records,
)
  - from pipelines.hardcoded.hh_continuation_pruning import (
    PruneConfig,
    apply_pruning,
    post_prune_refit,
    rank_prune_candidates,
)
  - from src.quantum.operator_pools import make_pool as make_paop_pool
  - from src.quantum.operator_pools.polaron_paop import make_phonon_motifs
  - from src.quantum.operator_pools.vlf_sq import build_vlf_sq_pool as build_vlf_sq_family
  - from src.quantum.ed_hubbard_holstein import build_hh_sector_hamiltonian_ed
  - from scipy.sparse import spmatrix as _spmatrix
  - from scipy.sparse.linalg import eigsh as _eigsh
  - from scipy.optimize import minimize as scipy_minimize
---
Classes:
  - AdaptVQEResult
    Properties:
      - energy
      - theta
      - selected_ops
      - history
      - stop_reason
      - nfev_total
  - _ADAPTLogicalCandidate
    Properties:
      - logical_label
      - pool_indices
      - parameterization
      - family_id
  - _BeamBranchState
    Methods:
      - L592: def clone_for_child(self, *, branch_id: int) -> "_BeamBranchState":
    Properties:
      - branch_id
      - parent_branch_id
      - depth_local
      - terminated
      - stop_reason
      - selected_ops
      - theta
      - energy_current
      - available_indices
      - selection_counts
      - history
      - phase1_stage
      - phase1_residual_opened
      - phase1_last_probe_reason
      - phase1_last_positions_considered
      - phase1_last_trough_detected
      - phase1_last_trough_probe_triggered
      - phase1_last_selected_score
      - phase1_features_history
      - phase1_stage_events
      - phase1_measure_cache
      - phase2_optimizer_memory
      - phase2_last_shortlist_records
      - phase2_last_batch_selected
      - phase2_last_batch_penalty_total
      - phase2_last_optimizer_memory_reused
      - phase2_last_optimizer_memory_source
      - phase2_last_shortlist_eval_records
      - drop_prev_delta_abs
      - drop_plateau_hits
      - eps_energy_low_streak
      - phase3_split_events
      - phase3_runtime_split_summary
      - phase3_motif_usage
      - phase3_rescue_history
      - nfev_total_local
  - _BranchExpansionPlan
    Properties:
      - candidate_pool_index
      - position_id
      - selection_mode
      - candidate_label
      - candidate_term
      - feature_row
      - init_theta
  - _BranchStepScratch
    Properties:
      - energy_current
      - psi_current
      - hpsi_current
      - gradients
      - grad_magnitudes
      - max_grad
      - gradient_eval_elapsed_s
      - append_position
      - best_idx
      - selected_position
      - selection_mode
      - stage_name
      - phase1_feature_selected
      - phase1_stage_transition_reason
      - phase1_stage_now
      - phase1_stage_after_transition
      - phase1_last_probe_reason
      - phase1_last_positions_considered
      - phase1_last_trough_detected
      - phase1_last_trough_probe_triggered
      - phase1_last_selected_score
      - phase2_last_shortlist_records
      - phase2_last_batch_selected
      - phase2_last_batch_penalty_total
      - phase2_last_optimizer_memory_reused
      - phase2_last_optimizer_memory_source
      - phase2_last_shortlist_eval_records
      - phase1_residual_opened
      - available_indices_after_transition
      - phase1_stage_events_after_transition
      - phase3_runtime_split_summary_after_eval
      - proposals
      - stop_reason
      - fallback_scan_size
      - fallback_best_probe_delta_e
      - fallback_best_probe_theta
  - ResolvedAdaptStopPolicy
    Properties:
      - adapt_drop_floor
      - adapt_drop_patience
      - adapt_drop_min_depth
      - adapt_grad_floor
      - adapt_drop_floor_source
      - adapt_drop_patience_source
      - adapt_drop_min_depth_source
      - adapt_grad_floor_source
      - drop_policy_enabled
      - drop_policy_source
      - eps_energy_termination_enabled
      - eps_grad_termination_enabled
  - ResolvedBeamCapacityPolicy
    Properties:
      - live_branches_requested
      - children_per_parent_requested
      - terminated_keep_requested
      - live_branches_effective
      - children_per_parent_effective
      - terminated_keep_effective
      - beam_enabled
      - source_children_per_parent
      - source_terminated_keep

Functions:
  - L183: def _ai_log(event: str, **fields: Any) -> None:
  - L196: def _to_ixyz(label_exyz: str) -> str:
  - L200: def _normalize_state(psi: np.ndarray) -> np.ndarray:
  - L207: def _collect_hardcoded_terms_exyz(poly: Any, tol: float = 1e-12) -> tuple[list[str], dict[str, complex]]:
  - L224: def _pauli_matrix_exyz(label: str) -> np.ndarray:
  - L232: def _build_hamiltonian_matrix(coeff_map_exyz: dict[str, complex]) -> np.ndarray:
  - L247: def _compile_pauli_action(label_exyz: str, nq: int) -> CompiledPauliAction:
  - L251: def _apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
  - L255: def _compile_polynomial_action(
    poly: Any,
    tol: float = 1e-15,
    *,
    pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
) -> CompiledPolynomialAction:
  - L272: def _apply_compiled_polynomial(state: np.ndarray, compiled_poly: CompiledPolynomialAction) -> np.ndarray:
  - L279: def _apply_exp_term(
    psi: np.ndarray, action: CompiledPauliAction, coeff: complex, alpha: float, tol: float = 1e-12,
) -> np.ndarray:
  - L291: def _evolve_trotter_suzuki2_absolute(
    psi0, ordered_labels, coeff_map, compiled_actions, time_value, trotter_steps,
) -> np.ndarray:
  - L307: def _expectation_hamiltonian(psi: np.ndarray, hmat: np.ndarray) -> float:
  - L315: def _occupation_site0(psi: np.ndarray, num_sites: int) -> tuple[float, float]:
  - L325: def _doublon_total(psi: np.ndarray, num_sites: int) -> float:
  - L338: def _state_to_amplitudes_qn_to_q0(psi: np.ndarray, cutoff: float = 1e-12) -> dict[str, dict[str, float]]:
  - L349: def _state_from_amplitudes_qn_to_q0(
    amplitudes_qn_to_q0: dict[str, Any],
    nq_total: int,
) -> np.ndarray:
  - L369: def _load_adapt_initial_state(
    adapt_json_path: Path,
    nq_total: int,
) -> tuple[np.ndarray, dict[str, Any]]:
  - L430: def _extract_nested(payload: Mapping[str, Any], *keys: str) -> Any:
  - L439: def _resolve_exact_energy_from_payload(payload: Mapping[str, Any]) -> float | None:
  - L458: def _validate_adapt_ref_metadata_for_exact_reuse(
    *,
    adapt_settings: Mapping[str, Any],
    args: argparse.Namespace,
    is_hh: bool,
    float_tol: float = 1e-10,
) -> list[str]:
  - L470: def _cmp_scalar(field: str, expected: Any, actual: Any) -> None:
  - L474: def _cmp_float(field: str, expected: float, actual_raw: Any) -> None:
  - L500: def _resolve_exact_energy_override_from_adapt_ref(
    *,
    adapt_ref_meta: Mapping[str, Any] | None,
    args: argparse.Namespace,
    problem: str,
    continuation_mode: str | None,
) -> tuple[float | None, str, list[str]]:
  - L686: def _parse_seq2p_step_label(label: str) -> tuple[str, str] | None:
  - L695: def _build_seq2p_logical_candidates(
    pool: Sequence[AnsatzTerm],
    *,
    family_id: str,
) -> list[_ADAPTLogicalCandidate]:
  - L725: def _logical_candidate_gradient_summary(
    candidate: _ADAPTLogicalCandidate,
    gradients: np.ndarray,
) -> tuple[float, list[float], list[float]]:
  - L735: def _build_uccsd_pool(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> list[AnsatzTerm]:
  - L757: def _build_cse_pool(
    num_sites: int,
    ordering: str,
    t: float,
    u: float,
    dv: float,
    boundary: str,
) -> list[AnsatzTerm]:
  - L780: def _build_full_hamiltonian_pool(
    h_poly: Any,
    tol: float = 1e-12,
    normalize_coeff: bool = False,
) -> list[AnsatzTerm]:
  - L812: def _polynomial_signature(poly: Any, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
  - L827: def _build_hh_termwise_augmented_pool(h_poly: Any, tol: float = 1e-12) -> list[AnsatzTerm]:
  - L866: def _build_hva_pool(
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
) -> list[AnsatzTerm]:
  - L942: def _build_hh_uccsd_fermion_lifted_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    num_particles: tuple[int, int] | None = None,
) -> list[AnsatzTerm]:
  - L999: def _build_paop_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    pool_key: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> list[AnsatzTerm]:
  - L1031: def _build_vlf_sq_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    pool_key: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
  - L1063: def _clean_real_pool_polynomial(poly: Any, prune_eps: float = 0.0) -> PauliPolynomial:
  - L1080: def _fermion_mode_to_site(mode: int, *, num_sites: int, ordering: str) -> int:
  - L1095: def _parse_lifted_uccsd_support(
    label: str,
    *,
    num_sites: int,
    ordering: str,
) -> tuple[str, tuple[int, ...]]:
  - L1134: def _motif_matches_excitation_support(
    *,
    motif: Any,
    motif_family: str,
    support_sites: tuple[int, ...],
    nearest_neighbor_bonds: set[tuple[int, int]],
) -> bool:
  - L1160: def _build_hh_uccsd_paop_product_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    family_key: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
  - L1283: def _deduplicate_pool_terms(pool: list[AnsatzTerm]) -> list[AnsatzTerm]:
  - L1296: def _polynomial_signature_digest(poly: Any, tol: float = 1e-12) -> str:
  - L1314: def _deduplicate_pool_terms_lightweight(pool: list[AnsatzTerm]) -> list[AnsatzTerm]:
  - L1327: def _build_hh_full_meta_pool(
    *,
    h_poly: Any,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, int]]:
  - L1423: def _build_hh_all_meta_v1_pool(
    *,
    h_poly: Any,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, int]]:
  - L1528: def _apply_pauli_polynomial_uncached(state: np.ndarray, poly: Any) -> np.ndarray:
  - L1551: def _apply_pauli_polynomial(
    state: np.ndarray,
    poly: Any,
    *,
    compiled: CompiledPolynomialAction | None = None,
) -> np.ndarray:
  - L1562: def _commutator_gradient(
    h_poly: Any,
    pool_op: AnsatzTerm,
    psi_current: np.ndarray,
    *,
    h_compiled: CompiledPolynomialAction | None = None,
    pool_compiled: CompiledPolynomialAction | None = None,
    hpsi_precomputed: np.ndarray | None = None,
) -> float:
  - L1592: def _prepare_adapt_state(
    psi_ref: np.ndarray,
    selected_ops: list[AnsatzTerm],
    theta: np.ndarray,
) -> np.ndarray:
  - L1604: def _adapt_energy_fn(
    h_poly: Any,
    psi_ref: np.ndarray,
    selected_ops: list[AnsatzTerm],
    theta: np.ndarray,
    *,
    h_compiled: CompiledPolynomialAction | None = None,
) -> float:
  - L1620: def _exact_gs_energy_for_problem(
    h_poly: Any,
    *,
    problem: str,
    num_sites: int,
    num_particles: tuple[int, int],
    indexing: str,
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
    t: float | None = None,
    u: float | None = None,
    dv: float | None = None,
    omega0: float | None = None,
    g_ep: float | None = None,
    boundary: str = "open",
) -> float:
  - L1714: def _exact_reference_state_for_hh(
    *,
    num_sites: int,
    num_particles: tuple[int, int],
    indexing: str,
    n_ph_max: int,
    boson_encoding: str,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    boundary: str,
) -> np.ndarray | None:
  - L1783: def _scipy_adapt_heartbeat_event(method_key: str) -> str:
  - L1790: def _scipy_adapt_optimizer_options(*, method_key: str, maxiter: int) -> dict[str, Any]:
  - L1798: def _run_scipy_adapt_optimizer(
    *,
    method_key: str,
    objective: Any,
    x0: np.ndarray,
    maxiter: int,
    context_label: str,
    scipy_minimize_fn: Any,
) -> Any:
  - L1817: def _resolve_reopt_active_indices(
    *,
    policy: str,
    n: int,
    theta: np.ndarray,
    window_size: int = 3,
    window_topk: int = 0,
    periodic_full_refit_triggered: bool = False,
) -> tuple[list[int], str]:
  - L1877: def _make_reduced_objective(
    full_theta: np.ndarray,
    active_indices: list[int],
    obj_fn: Any,
) -> tuple[Any, np.ndarray]:
  - L1898: def _reduced(x_active: np.ndarray) -> float:
  - L1908: def _resolve_adapt_continuation_mode(*, problem: str, requested_mode: str | None) -> str:
  - L1946: def _resolve_beam_capacity_policy(
    *,
    adapt_beam_live_branches: int,
    adapt_beam_children_per_parent: int | None,
    adapt_beam_terminated_keep: int | None,
) -> ResolvedBeamCapacityPolicy:
  - L2003: def _resolve_adapt_stop_policy(
    *,
    problem: str,
    continuation_mode: str,
    adapt_drop_floor: float | None,
    adapt_drop_patience: int | None,
    adapt_drop_min_depth: int | None,
    adapt_grad_floor: float | None,
) -> ResolvedAdaptStopPolicy:
  - L2017: def _resolve_float(raw: float | None, *, staged_value: float, default_value: float) -> tuple[float, str]:
  - L2024: def _resolve_int(raw: int | None, *, staged_value: int, default_value: int) -> tuple[int, str]:
  - L2085: def _phase1_repeated_family_flat(
    *,
    history: list[dict[str, Any]],
    candidate_family: str,
    patience: int,
    weak_drop_threshold: float,
) -> bool:
  - L2110: def _splice_candidate_at_position(
    *,
    ops: list[AnsatzTerm],
    theta: np.ndarray,
    op: AnsatzTerm,
    position_id: int,
    init_theta: float = 0.0,
) -> tuple[list[AnsatzTerm], np.ndarray]:
  - L2127: def _splice_logical_candidate_at_position(
    *,
    ops: list[AnsatzTerm],
    theta: np.ndarray,
    candidate: _ADAPTLogicalCandidate,
    pool: Sequence[AnsatzTerm],
    position_id: int,
    init_theta_values: Sequence[float] | None = None,
) -> tuple[list[AnsatzTerm], np.ndarray]:
  - L2156: def _predict_reopt_window_for_position(
    *,
    theta: np.ndarray,
    position_id: int,
    policy: str,
    window_size: int,
    window_topk: int,
    periodic_full_refit_triggered: bool,
) -> list[int]:
  - L2180: def _window_terms_for_position(
    *,
    selected_ops: list[AnsatzTerm],
    refit_window_indices: list[int],
    position_id: int,
) -> tuple[list[AnsatzTerm], list[str]]:
  - L2201: def _phase2_record_sort_key(record: Mapping[str, Any]) -> tuple[float, float, int, int]:
  - L2210: def _run_hardcoded_adapt_vqe(
    *,
    h_poly: Any,
    num_sites: int,
    ordering: str,
    problem: str,
    adapt_pool: str | None,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    max_depth: int,
    eps_grad: float,
    eps_energy: float,
    maxiter: int,
    seed: int,
    adapt_inner_optimizer: str = "SPSA",
    adapt_spsa_a: float = 0.2,
    adapt_spsa_c: float = 0.1,
    adapt_spsa_alpha: float = 0.602,
    adapt_spsa_gamma: float = 0.101,
    adapt_spsa_A: float = 10.0,
    adapt_spsa_avg_last: int = 0,
    adapt_spsa_eval_repeats: int = 1,
    adapt_spsa_eval_agg: str = "mean",
    adapt_spsa_callback_every: int = 1,
    adapt_spsa_progress_every_s: float = 60.0,
    allow_repeats: bool,
    finite_angle_fallback: bool,
    finite_angle: float,
    finite_angle_min_improvement: float,
    adapt_drop_floor: float | None = None,
    adapt_drop_patience: int | None = None,
    adapt_drop_min_depth: int | None = None,
    adapt_grad_floor: float | None = None,
    adapt_eps_energy_min_extra_depth: int = -1,
    adapt_eps_energy_patience: int = -1,
    adapt_ref_base_depth: int = 0,
    paop_r: int = 0,
    paop_split_paulis: bool = False,
    paop_prune_eps: float = 0.0,
    paop_normalization: str = "none",
    disable_hh_seed: bool = False,
    psi_ref_override: np.ndarray | None = None,
    adapt_ref_json: Path | None = None,
    adapt_gradient_parity_check: bool = False,
    adapt_state_backend: str = "compiled",
    adapt_reopt_policy: str = "append_only",
    adapt_window_size: int = 3,
    adapt_window_topk: int = 0,
    adapt_full_refit_every: int = 0,
    adapt_final_full_refit: bool = True,
    exact_gs_override: float | None = None,
    adapt_continuation_mode: str | None = "phase3_v1",
    phase1_lambda_F: float = 1.0,
    phase1_lambda_compile: float = 0.05,
    phase1_lambda_measure: float = 0.02,
    phase1_lambda_leak: float = 0.0,
    phase1_score_z_alpha: float = 0.0,
    phase1_probe_max_positions: int = 6,
    phase1_plateau_patience: int = 2,
    phase1_trough_margin_ratio: float = 1.0,
    phase1_prune_enabled: bool = True,
    phase1_prune_fraction: float = 0.25,
    phase1_prune_max_candidates: int = 6,
    phase1_prune_max_regression: float = 1e-8,
    phase2_shortlist_fraction: float = 0.2,
    phase2_shortlist_size: int = 12,
    phase2_lambda_H: float = 1e-6,
    phase2_rho: float = 0.25,
    phase2_gamma_N: float = 1.0,
    phase2_enable_batching: bool = True,
    phase2_batch_target_size: int = 2,
    phase2_batch_size_cap: int = 3,
    phase2_batch_near_degenerate_ratio: float = 0.9,
    phase3_motif_source_json: Path | None = None,
    phase3_symmetry_mitigation_mode: str = "off",
    phase3_enable_rescue: bool = False,
    phase3_lifetime_cost_mode: str = "phase3_v1",
    phase3_runtime_split_mode: str = "off",
    adapt_beam_live_branches: int = 1,
    adapt_beam_children_per_parent: int | None = None,
    adapt_beam_terminated_keep: int | None = None,
    diagnostics_out: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
  - L2594: def _build_hh_pool_by_key(pool_key_hh: str) -> tuple[list[AnsatzTerm], str]:
  - L3021: def _build_compiled_executor(ops: list[AnsatzTerm]) -> CompiledAnsatzExecutor:
  - L3203: def _seed_obj(x: np.ndarray) -> float:
  - L3244: def _seed_spsa_callback(ev: dict[str, Any]) -> None:
  - L3436: def _phase3_try_rescue(
        *,
        psi_current_state: np.ndarray,
        shortlist_eval_records: list[dict[str, Any]],
        selected_position_append: int,
        history_rows: list[dict[str, Any]],
        trough_detected_now: bool,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
  - L3470: def _overlap_gain(rec: Mapping[str, Any]) -> float:
  - L3532: def _beam_clone_branch(
        branch: _BeamBranchState,
        *,
        branch_id: int,
        parent_branch_id: int | None,
    ) -> _BeamBranchState:
  - L3544: def _beam_executor_key(ops: Sequence[AnsatzTerm]) -> tuple[str, ...]:
  - L3547: def _get_beam_executor(ops: Sequence[AnsatzTerm]) -> CompiledAnsatzExecutor | None:
  - L3557: def _branch_state_fingerprint(branch: _BeamBranchState) -> str:
  - L3576: def _proposal_fingerprint(
        *,
        parent: _BeamBranchState,
        plan: _BranchExpansionPlan,
    ) -> str:
  - L3593: def _branch_optimizer_seed(
        *,
        base_seed: int,
        stage_tag: str,
        depth_local: int,
        parent_state_fingerprint: str,
        proposal_fingerprint: str | None,
    ) -> int:
  - L3613: def _beam_prune_key(branch: _BeamBranchState) -> tuple[Any, ...]:
  - L3625: def _beam_dedup(branches: Sequence[_BeamBranchState]) -> list[_BeamBranchState]:
  - L3634: def _beam_prune(
        branches: Sequence[_BeamBranchState],
        *,
        cap: int,
    ) -> list[_BeamBranchState]:
  - L3642: def _evaluate_beam_branch(
        branch: _BeamBranchState,
        *,
        depth: int,
        children_cap: int,
    ) -> _BranchStepScratch:
  - L3769: def _evaluate_phase1_positions_local(
            positions_considered_local: list[int],
            *,
            trough_probe_triggered_local: bool,
        ) -> dict[str, Any]:
  - L3977: def _full_record_for_candidate_local(
                    *,
                    candidate_term: AnsatzTerm,
                    candidate_label: str,
                    generator_metadata: Mapping[str, Any] | None,
                    symmetry_spec_candidate: Mapping[str, Any] | None,
                    runtime_split_mode_value: str = "off",
                    runtime_split_parent_label_value: str | None = None,
                    runtime_split_child_index_value: int | None = None,
                    runtime_split_child_count_value: int | None = None,
                ) -> dict[str, Any]:
  - L4502: def _materialize_beam_child(
        base_branch: _BeamBranchState,
        scratch: _BranchStepScratch,
        plan: _BranchExpansionPlan,
        *,
        depth: int,
        branch_id: int,
    ) -> _BeamBranchState:
  - L4569: def _obj_child(x: np.ndarray) -> float:
  - L5448: def _evaluate_phase1_positions(
                positions_considered_local: list[int],
                *,
                trough_probe_triggered_local: bool,
            ) -> dict[str, Any]:
  - L5664: def _full_record_for_candidate(
                        *,
                        candidate_term: AnsatzTerm,
                        candidate_label: str,
                        generator_metadata: Mapping[str, Any] | None,
                        symmetry_spec_candidate: Mapping[str, Any] | None,
                        runtime_split_mode_value: str = "off",
                        runtime_split_parent_label_value: str | None = None,
                        runtime_split_child_index_value: int | None = None,
                        runtime_split_child_count_value: int | None = None,
                    ) -> dict[str, Any]:
  - L6384: def _obj(x: np.ndarray) -> float:
  - L6474: def _depth_spsa_callback(ev: dict[str, Any]) -> None:
  - L7175: def _obj_final(x: np.ndarray) -> float:
  - L7311: def _reconstruct_phase1_proxy_benefits() -> list[float]:
  - L7355: def _refit_given_ops(ops_refit: list[AnsatzTerm], theta0: np.ndarray) -> tuple[np.ndarray, float]:
  - L7361: def _obj_prune(x: np.ndarray) -> float:
  - L7419: def _ops_from_labels(labels_cur: list[str]) -> list[AnsatzTerm]:
  - L7432: def _eval_with_removal(
            idx_remove: int,
            theta_cur: np.ndarray,
            labels_cur: list[str],
        ) -> tuple[float, np.ndarray]:
  - L7873: def _simulate_trajectory(
    *,
    num_sites: int,
    psi0: np.ndarray,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    trotter_steps: int,
    t_final: float,
    num_times: int,
    suzuki_order: int,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
  - L7953: def _write_pipeline_pdf(pdf_path: Path, payload: dict[str, Any], run_command: str) -> None:
  - L8162: def parse_args() -> argparse.Namespace:
  - L8501: def main() -> None:

Global vars:
  - REPO_ROOT
  - plt
  - PdfPages
  - EXACT_LABEL
  - EXACT_METHOD
  - _ADAPT_GRADIENT_PARITY_RTOL
  - PAULI_MATS
  - _HH_STAGED_CONTINUATION_MODES
  - _HH_UCCSD_PAOP_PRODUCT_SPECS
  - _UCCSD_SINGLE_LABEL_RE
  - _UCCSD_DOUBLE_LABEL_RE
  - _VALID_REOPT_POLICIES
  - _VALID_ADAPT_INNER_OPTIMIZERS
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_staged_workflow.py
Imports:
  - import hashlib
  - import json
  - import math
  - import sys
  - import time
  - from dataclasses import asdict, dataclass, field
  - from datetime import datetime, timezone
  - from pathlib import Path
  - from types import SimpleNamespace
  - from typing import Any, Mapping, Sequence
  - import numpy as np
  - from docs.reports.pdf_utils import (
    HAS_MATPLOTLIB,
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)
  - from docs.reports.qiskit_circuit_report import (
    adapt_ops_to_circuit,
    ansatz_to_circuit,
    build_time_dynamics_circuit,
    compute_time_dynamics_proxy_cost,
    is_cfqm_dynamics_method,
    render_circuit_page,
    render_circuit_summary_page,
    time_dynamics_circuitization_reason,
    transpile_circuit_metrics,
    warn_time_dynamics_circuit_semantics,
)
  - from pipelines.hardcoded import adapt_pipeline as adapt_mod
  - from pipelines.hardcoded import hh_vqe_from_adapt_family as replay_mod
  - from pipelines.hardcoded import hubbard_pipeline as hc_pipeline
  - from pipelines.hardcoded.handoff_state_bundle import (
    HandoffStateBundleConfig,
    write_handoff_state_bundle,
)
  - from src.quantum.drives_time_potential import (
    build_gaussian_sinusoid_density_drive,
    reference_method_name,
)
  - from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
  - from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_holstein_hamiltonian,
)
  - from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinLayerwiseAnsatz,
    HubbardHolsteinPhysicalTermwiseAnsatz,
    HubbardHolsteinTermwiseAnsatz,
    exact_ground_energy_sector_hh,
)
  - from qiskit import QuantumCircuit
---
Classes:
  - PhysicsConfig
    Properties:
      - L
      - t
      - u
      - dv
      - omega0
      - g_ep
      - n_ph_max
      - boson_encoding
      - ordering
      - boundary
      - sector_n_up
      - sector_n_dn
  - WarmStartConfig
    Properties:
      - ansatz_name
      - reps
      - restarts
      - maxiter
      - method
      - seed
      - progress_every_s
      - energy_backend
      - spsa_a
      - spsa_c
      - spsa_alpha
      - spsa_gamma
      - spsa_A
      - spsa_avg_last
      - spsa_eval_repeats
      - spsa_eval_agg
  - SeedRefineConfig
    Properties:
      - family
      - reps
      - maxiter
      - optimizer
  - AdaptConfig
    Properties:
      - pool
      - continuation_mode
      - max_depth
      - maxiter
      - eps_grad
      - eps_energy
      - drop_floor
      - drop_patience
      - drop_min_depth
      - grad_floor
      - seed
      - inner_optimizer
      - allow_repeats
      - finite_angle_fallback
      - finite_angle
      - finite_angle_min_improvement
      - disable_hh_seed
      - reopt_policy
      - window_size
      - window_topk
      - full_refit_every
      - final_full_refit
      - beam_live_branches
      - beam_children_per_parent
      - beam_terminated_keep
      - paop_r
      - paop_split_paulis
      - paop_prune_eps
      - paop_normalization
      - spsa_a
      - spsa_c
      - spsa_alpha
      - spsa_gamma
      - spsa_A
      - spsa_avg_last
      - spsa_eval_repeats
      - spsa_eval_agg
      - spsa_callback_every
      - spsa_progress_every_s
      - phase1_lambda_F
      - phase1_lambda_compile
      - phase1_lambda_measure
      - phase1_lambda_leak
      - phase1_score_z_alpha
      - phase1_probe_max_positions
      - phase1_plateau_patience
      - phase1_trough_margin_ratio
      - phase1_prune_enabled
      - phase1_prune_fraction
      - phase1_prune_max_candidates
      - phase1_prune_max_regression
      - phase3_motif_source_json
      - phase3_symmetry_mitigation_mode
      - phase3_enable_rescue
      - phase3_lifetime_cost_mode
      - phase3_runtime_split_mode
  - ReplayConfig
    Properties:
      - enabled
      - generator_family
      - fallback_family
      - legacy_paop_key
      - replay_seed_policy
      - continuation_mode
      - reps
      - restarts
      - maxiter
      - method
      - seed
      - energy_backend
      - progress_every_s
      - wallclock_cap_s
      - paop_r
      - paop_split_paulis
      - paop_prune_eps
      - paop_normalization
      - spsa_a
      - spsa_c
      - spsa_alpha
      - spsa_gamma
      - spsa_A
      - spsa_avg_last
      - spsa_eval_repeats
      - spsa_eval_agg
      - replay_freeze_fraction
      - replay_unfreeze_fraction
      - replay_full_fraction
      - replay_qn_spsa_refresh_every
      - replay_qn_spsa_refresh_mode
      - phase3_symmetry_mitigation_mode
  - DynamicsConfig
    Properties:
      - enabled
      - methods
      - t_final
      - num_times
      - trotter_steps
      - exact_steps_multiplier
      - fidelity_subspace_energy_tol
      - cfqm_stage_exp
      - cfqm_coeff_drop_abs_tol
      - cfqm_normalize
      - enable_drive
      - drive_A
      - drive_omega
      - drive_tbar
      - drive_phi
      - drive_pattern
      - drive_custom_s
      - drive_include_identity
      - drive_time_sampling
      - drive_t0
  - FixedFinalStateConfig
    Properties:
      - json_path
      - strict_match
  - CircuitMetricConfig
    Properties:
      - backend_name
      - use_fake_backend
      - optimization_level
      - seed_transpiler
  - WarmCheckpointConfig
    Properties:
      - stop_energy
      - stop_delta_abs
      - state_export_dir
      - state_export_prefix
      - resume_from_warm_checkpoint
      - handoff_from_warm_checkpoint
  - ArtifactConfig
    Properties:
      - tag
      - output_json
      - output_pdf
      - handoff_json
      - warm_checkpoint_json
      - warm_cutover_json
      - replay_output_json
      - replay_output_csv
      - replay_output_md
      - replay_output_log
      - workflow_log
      - skip_pdf
  - GateConfig
    Properties:
      - ecut_1
      - ecut_2
  - StagedHHConfig
    Properties:
      - physics
      - warm_start
      - seed_refine
      - adapt
      - replay
      - dynamics
      - fixed_final_state
      - circuit_metrics
      - warm_checkpoint
      - artifacts
      - gates
      - smoke_test_intentionally_weak
      - default_provenance
      - external_noise_handle
  - StageExecutionResult
    Properties:
      - h_poly
      - hmat
      - ordered_labels_exyz
      - coeff_map_exyz
      - nq_total
      - psi_hf
      - psi_warm
      - psi_adapt
      - psi_final
      - warm_payload
      - adapt_payload
      - replay_payload
      - psi_seed_refine
      - seed_refine_payload
      - fixed_final_state_import
      - warm_circuit_context
      - adapt_circuit_context
      - replay_circuit_context

Functions:
  - L334: def _relative_error_abs(value: float, reference: float) -> float:
  - L338: def _now_utc() -> str:
  - L342: def _jsonable(value: Any) -> Any:
  - L352: def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
  - L358: def _append_workflow_log(cfg: StagedHHConfig, event: str, **fields: Any) -> None:
  - L370: def _handoff_bundle_cfg(cfg: StagedHHConfig) -> HandoffStateBundleConfig:
  - L387: def _seed_refine_state_json_path(cfg: StagedHHConfig) -> Path:
  - L393: def _build_seed_refine_run_cfg(cfg: StagedHHConfig) -> replay_mod.RunConfig:
  - L447: def _build_seed_provenance(
    cfg: StagedHHConfig,
    seed_refine_payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
  - L477: def _warm_stop_required(cfg: StagedHHConfig) -> bool:
  - L484: def _warm_stop_status(
    cfg: StagedHHConfig,
    *,
    energy: float,
    exact_filtered_energy: float,
) -> dict[str, Any]:
  - L521: def _write_warm_checkpoint_bundle(
    cfg: StagedHHConfig,
    *,
    path: Path,
    psi_state: np.ndarray,
    energy: float,
    exact_filtered_energy: float,
    theta: Sequence[float] | None,
    role: str,
    cutoff_status: Mapping[str, Any],
    event_meta: Mapping[str, Any] | None = None,
    source_json: Path | None = None,
) -> None:
  - L569: def _expected_adapt_ref_args(cfg: StagedHHConfig) -> SimpleNamespace:
  - L585: def _resolve_checkpoint_energy(payload: Mapping[str, Any]) -> float | None:
  - L602: def _build_fixed_final_state_import(
    cfg: StagedHHConfig,
    *,
    source_json: Path,
    raw_payload: Mapping[str, Any],
    nq_total: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
  - L701: def _write_fixed_final_state_sidecars(
    cfg: StagedHHConfig,
    *,
    psi_final: np.ndarray,
    fixed_import: Mapping[str, Any],
    replay_payload: Mapping[str, Any],
) -> None:
  - L742: def _run_warm_start_stage(
    cfg: StagedHHConfig,
    *,
    h_poly: Any,
    psi_hf: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, Path]:
  - L766: def _load_checkpoint_for_handoff(
        checkpoint_json: Path,
    ) -> tuple[dict[str, Any], np.ndarray, Mapping[str, Any], list[float] | None, float, float, dict[str, Any]]:
  - L1008: def _emit_checkpoint(theta_values: Sequence[float], energy_value: float, event_meta: Mapping[str, Any] | None) -> None:
  - L1046: def _progress_observer(event: Mapping[str, Any]) -> None:
  - L1062: def _early_stop_checker(event: Mapping[str, Any]) -> bool:
  - L1186: def _bool_flag(raw: Any) -> bool:
  - L1197: def _parse_noiseless_methods(raw: str | Sequence[str]) -> tuple[str, ...]:
  - L1218: def _parse_drive_custom_weights(raw: str | None) -> list[float] | None:
  - L1231: def _half_filled_particles(L: int) -> tuple[int, int]:
  - L1236: def _hh_nq_total(L: int, n_ph_max: int, boson_encoding: str) -> int:
  - L1241: def _default_output_tag(
    *,
    L: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    ordering: str,
    boundary: str,
    sector_n_up: int,
    sector_n_dn: int,
    drive_enabled: bool,
    drive_pattern: str,
    drive_A: float,
    drive_omega: float,
    drive_tbar: float,
    drive_phi: float,
    drive_time_sampling: str,
    noiseless_methods: str,
    adapt_continuation_mode: str,
    warm_ansatz: str,
    seed_refine_family: str | None,
    fixed_final_state_json: str | None,
    circuit_backend_name: str | None,
    circuit_use_fake_backend: bool,
) -> str:
  - L1320: def _scaled_defaults(L: int) -> dict[str, Any]:
  - L1340: def _resolve_with_default(
    *,
    name: str,
    raw: Any,
    default: Any,
    provenance: dict[str, str],
    default_source: str,
) -> Any:
  - L1355: def _enforce_not_weaker(
    *,
    cfg_values: Mapping[str, Any],
    baseline: Mapping[str, Any],
    smoke_test_intentionally_weak: bool,
) -> None:
  - L1390: def resolve_staged_hh_config(args: Any) -> StagedHHConfig:
  - L1934: def _build_hh_context(cfg: StagedHHConfig) -> tuple[Any, np.ndarray, list[str], dict[str, complex], np.ndarray]:
  - L1969: def _handoff_continuation_meta(adapt_payload: Mapping[str, Any]) -> dict[str, Any]:
  - L2028: def _infer_replay_family_from_operator_labels(labels: Any) -> tuple[str | None, str | None]:
  - L2048: def _infer_handoff_adapt_pool(cfg: StagedHHConfig, adapt_payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
  - L2104: def _canonical_replay_family(raw: Any) -> str | None:
  - L2111: def _seed_policy_for_handoff_state(raw_state: str, raw_policy: Any) -> tuple[str, str]:
  - L2127: def _build_replay_contract(
    cfg: StagedHHConfig,
    handoff_adapt_pool: str | None,
    handoff_adapt_pool_source: str | None = None,
) -> dict[str, Any]:
  - L2175: def _run_seed_refine_stage(
    cfg: StagedHHConfig,
    *,
    h_poly: Any,
    psi_ref: np.ndarray,
    exact_filtered_energy: float,
) -> tuple[dict[str, Any], np.ndarray, Path]:
  - L2349: def _write_adapt_handoff(
    cfg: StagedHHConfig,
    adapt_payload: Mapping[str, Any],
    psi_adapt: np.ndarray,
    *,
    seed_provenance: Mapping[str, Any] | None = None,
) -> None:
  - L2409: def _staged_ansatz_manifest(cfg: StagedHHConfig) -> str:
  - L2422: def _build_hh_warm_ansatz(cfg: StagedHHConfig) -> Any:
  - L2446: def _assemble_stage_circuit_contexts(
    *,
    cfg: StagedHHConfig,
    psi_hf: np.ndarray,
    warm_payload: Mapping[str, Any],
    adapt_diagnostics: Mapping[str, Any] | None,
    replay_diagnostics: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any] | None]:
  - L2500: def _workflow_stage_chain(cfg: StagedHHConfig, *, fixed_mode: bool) -> list[str]:
  - L2515: def _terminal_reference_energy(stage_result: StageExecutionResult, cfg: StagedHHConfig) -> tuple[float, str]:
  - L2524: def run_stage_pipeline(cfg: StagedHHConfig) -> StageExecutionResult:
  - L2818: def _build_drive_provider(
    *,
    cfg: StagedHHConfig,
    nq_total: int,
    ordered_labels_exyz: Sequence[str],
) -> tuple[Any | None, dict[str, Any] | None, list[str], dict[str, Any] | None]:
  - L2867: def _run_noiseless_profile(
    *,
    cfg: StagedHHConfig,
    psi_seed: np.ndarray,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    drive_enabled: bool,
    ground_state_reference_energy: float,
    ground_state_reference_source: str,
) -> dict[str, Any]:
  - L2982: def run_noiseless_profiles(stage_result: StageExecutionResult, cfg: StagedHHConfig) -> dict[str, Any]:
  - L3010: def _empty_qiskit_circuit(num_qubits: int) -> Any:
  - L3016: def _transpile_target_metadata(cfg: StagedHHConfig) -> dict[str, Any] | None:
  - L3028: def _transpile_metrics_or_error(
    cfg: StagedHHConfig,
    *,
    circuit: Any | None,
    enabled: bool = True,
    reason: str | None = None,
) -> dict[str, Any] | None:
  - L3060: def _strip_circuit_objects(value: Any) -> Any:
  - L3072: def build_stage_circuit_report_artifacts(
    stage_result: StageExecutionResult,
    cfg: StagedHHConfig,
) -> dict[str, Any]:
  - L3329: def write_hh_staged_circuit_report_section(
    pdf: Any,
    *,
    cfg: StagedHHConfig,
    stage_result: StageExecutionResult,
    run_command: str | None = None,
) -> None:
  - L3486: def _stage_delta(payload: Mapping[str, Any], *, energy_key: str, exact_key: str) -> float:
  - L3490: def _stage_summary(stage_result: StageExecutionResult, cfg: StagedHHConfig) -> dict[str, Any]:
  - L3590: def _compute_comparisons(payload: Mapping[str, Any]) -> dict[str, Any]:
  - L3640: def _payload_artifacts(cfg: StagedHHConfig) -> dict[str, Any]:
  - L3662: def assemble_payload(
    *,
    cfg: StagedHHConfig,
    stage_result: StageExecutionResult,
    dynamics_noiseless: Mapping[str, Any],
    circuit_report: Mapping[str, Any] | None = None,
    run_command: str,
) -> dict[str, Any]:
  - L3709: def _profile_plot_page(pdf: Any, profile_name: str, profile_payload: Mapping[str, Any]) -> None:
  - L3739: def write_staged_hh_pdf(payload: Mapping[str, Any], cfg: StagedHHConfig, run_command: str) -> None:
  - L3902: def run_staged_hh_noiseless(cfg: StagedHHConfig, *, run_command: str | None = None) -> dict[str, Any]:

Global vars:
  - REPO_ROOT
  - _PREPARED_STATE
  - _REFERENCE_STATE
  - _ALLOWED_NOISELESS_METHODS
---

</file_map>
<file_contents>
File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/run_guide.md
(lines 132-176: Run guide canonical HH staged contract: warm start, optional seed-refine, phase3_v1 ADAPT, matched-family replay, and opt-in runtime split or symmetry hooks.)
```md
1. Warm-start stage: conventional VQE with intermediate HH ansatz `hh_hva_ptw`.
   - `hh_hva_ptw` is the canonical staged default.
   - `hh_hva` remains an explicit override only.
2. Optional seed-refine stage: when `--seed-refine-family` is set, run one
   explicit-family conventional VQE refine stage between warm-start and ADAPT.
   - Supported v1 families: `uccsd_otimes_paop_lf_std`,
     `uccsd_otimes_paop_lf2_std`, `uccsd_otimes_paop_bond_disp_std`.
   - This stage materializes the requested explicit family directly; it does
     **not** use `match_adapt` and does **not** auto-fallback to `full_meta`.
   - If refine fails, abort before ADAPT.
3. ADAPT stage: follow the target pool curriculum from `MATH/IMPLEMENT_SOON.md`:
   start from a narrow HH physics-aligned pool and do **not** open `full_meta`
   at depth 0; treat `full_meta` only as controlled residual enrichment after
   plateau diagnosis.
   - Canonical stage selection mode for new HH agent-directed runs is `phase3_v1` (phase1_v1 and phase2_v1 remain opt-in).
4. ADAPT -> final VQE switch: apply an energy-drop switching criterion (see "ADAPT continuation stop policy (energy-first, mandatory for agent runs)").
5. Optional final VQE replay: when `--run-replay` is enabled, initialize from ADAPT state and replay with the same variational generator family ADAPT used (`--generator-family match_adapt`, fallback `full_meta`), using `vqe_reps=L` by default.

Pool curriculum transition note:
- `AGENTS target`: for this HH pool-curriculum transition, treat
  `MATH/IMPLEMENT_SOON.md` as the target spec for new agent-directed pool
  decisions. Depth-0 `full_meta` is not the intended default.
- `Current code behavior`: current CLI and older workflows still support
  `--adapt-pool full_meta`, and historical reference runs below may use it.
- `Required action: ask user before proceeding` if you plan to start a new
  agent-directed HH ADAPT run at depth 0 with `--adapt-pool full_meta`.

CLI note:
- `--adapt-pool full_meta` remains a supported HH pool token in current code; do
  not treat it as the canonical depth-0 target for new agent work.
- Canonical continuation default for new HH staged runs is `--adapt-continuation-mode phase3_v1`.
  (`phase1_v1` and `phase2_v1` are opt-in legacy/experimental follow-ons.)
- Legacy nearest subset remains `--adapt-pool uccsd_paop_lf_full` (`uccsd_lifted + paop_lf_full`).
- Explicit product families for refine/replay or direct ADAPT materialization are
  `uccsd_otimes_paop_lf_std`, `uccsd_otimes_paop_lf2_std`,
  `uccsd_otimes_paop_bond_disp_std`.
- Logical two-parameter variants
  (`uccsd_otimes_paop_lf_std_seq2p`, `...lf2_std_seq2p`,
  `...bond_disp_std_seq2p`) are additive opt-in surfaces and do not change the
  staged `phase3_v1` default contract.

Opt-in phase-3 follow-ons (keep defaults off unless explicitly requested):
- `--phase3-runtime-split-mode shortlist_pauli_children_v1` is an optional continuation aid for HH staged ADAPT/hardcoded paths: shortlisted macro generators may be probed as single-term children, with parent/child provenance exported in continuation metadata.
- `--phase3-symmetry-mitigation-mode {off,verify_only,postselect_diag_v1,projector_renorm_v1}` is an optional phase-3 continuation hook. On raw ADAPT / hardcoded / replay paths it is a metadata-and-telemetry surface; active counts-based symmetry mitigation is enforced only in the oracle-backed noise runners.
- These follow-ons do **not** change the canonical HH contract above: narrow-core first, no depth-0 `full_meta` for new agent-directed runs, and opt-in matched-family replay via `--generator-family match_adapt` with `full_meta` fallback.

```

(lines 380-385: Run guide stop-policy semantics showing eps-energy and eps-grad become telemetry-only in HH phase1_v1/phase2_v1/phase3_v1.)
```md
  - `adapt_drop_min_depth = 12`
  - `adapt_grad_floor = 2e-2`
- Explicit CLI values override these resolved defaults; passing negative/off values disables the corresponding staged guard explicitly.
- In HH `phase1_v1` / `phase2_v1` / `phase3_v1`, `eps_energy` telemetry remains active but does **not** terminate ADAPT.
- In HH `phase1_v1` / `phase2_v1` / `phase3_v1`, legacy `eps_grad` is no longer a terminating stop path; low-gradient diagnostics feed the drop-first plateau policy instead.
- In Hubbard and HH `legacy`, the `eps_energy` guard remains depth-gated and patience-gated by defaults:

```

(lines 665-677: Run guide historical/current HH pipeline summary tying warm start, ADAPT pool curriculum, switch criterion, replay, and noisy dynamics benchmark together.)
```md
1. HVA warm-start with `hh_hva_ptw` (intermediate HH variant)
2. ADAPT stage:
   - target policy for new agent work: narrow HH physics-aligned pool first;
     `full_meta` only as controlled residual enrichment after plateau diagnosis
     (see `MATH/IMPLEMENT_SOON.md`)
   - strict legacy mode: Pool B union (`UCCSD_lifted + HVA + PAOP_full`)
   - current broad-pool executable mode: `--adapt-pool full_meta`
     (`UCCSD_lifted + HVA + PAOP_full + PAOP_lf_full`)
   - `Required action: ask user before proceeding` before using depth-0
     `full_meta` for a new agent-directed HH staged run
3. switch from ADAPT to final VQE using the energy-drop criterion (see ADAPT continuation stop policy)
4. conventional VQE replay seeded from ADAPT operator/theta sequence with `vqe_reps = L`
5. noisy dynamics benchmark for selected methods (default `cfqm4,suzuki2`)

```

(lines 900-904: Run guide replay provenance note about handoff_state_kind and serialized runtime-split children in continuation metadata.)
```md
Replay provenance notes:
- Exported staged HH payloads stamp `initial_state.handoff_state_kind` when available.
- `prepared_state` means replay `--replay-seed-policy auto` resolves to `residual_only`; `reference_state` means `auto` resolves to `scaffold_plus_zero`.
- If an opt-in runtime split selected child labels that are not present in the resolved replay family pool, keep `continuation.selected_generator_metadata[*].compile_metadata.serialized_terms_exyz`; replay uses that serialized metadata to rebuild those operators.


```

(lines 1006-1033: Run guide CLI contract for ADAPT pool, continuation mode, drop policy, and reference import semantics that constrain phase3_v1-style workflow design.)
```md
| `--adapt-pool` | choice | `uccsd` | Pool type: `uccsd`, `cse`, `full_hamiltonian`, `hva` (HH only), `full_meta` (HH only), `paop`, `paop_min`, `paop_std`, `paop_full`, `paop_lf`, `paop_lf_std`, `paop_lf2_std`, `paop_lf_full` (HH only) |
| `--adapt-max-depth` | int | `20` | Maximum ADAPT iterations (operators appended) |
| `--adapt-eps-grad` | float | `1e-4` | Gradient convergence threshold |
| `--adapt-eps-energy` | float | `1e-8` | Energy convergence threshold. Hard-stop guard for Hubbard / HH `legacy`; telemetry-only in HH `phase1_v1|phase2_v1|phase3_v1` |
| `--adapt-eps-energy-min-extra-depth` | int | `-1` | Minimum extra depth before eps-energy guard can trigger; `-1 => L`. Telemetry-only in HH `phase1_v1|phase2_v1|phase3_v1` |
| `--adapt-eps-energy-patience` | int | `-1` | Consecutive low-improvement depths required for eps-energy guard; `-1 => L`. Telemetry-only in HH `phase1_v1|phase2_v1|phase3_v1` |
| `--adapt-inner-optimizer` | choice | `SPSA` | Inner optimizer per ADAPT re-optimization step: `COBYLA` or `SPSA`. |
| `--adapt-state-backend` | choice | `compiled` | ADAPT state action backend: `compiled` (production cached path) or `legacy` (lower-memory fallback) |
| `--adapt-reopt-policy` | choice | `append_only` | Per-depth ADAPT re-optimization policy: `append_only` (default; newest theta only), `full` (legacy all-parameter re-opt), or `windowed` (sliding window + top-k carry). |
| `--adapt-window-size` | int | `3` | Window width W for `windowed` policy (newest W parameters always active). |
| `--adapt-window-topk` | int | `0` | Top-K older parameters (by `|θ|`) carried into the active set; `0` = window only. |
| `--adapt-full-refit-every` | int | `0` | Periodic full-prefix refit cadence (every N cumulative depths); `0` = disabled. |
| `--adapt-final-full-refit` | str | `true` | Run a post-loop full-prefix refit before export (windowed only); `true`/`false`. |
| `--adapt-maxiter` | int | `300` | Inner optimizer maxiter per re-optimization |
| `--adapt-seed` | int | `7` | Random seed |
| `--phase3-symmetry-mitigation-mode` | choice | `off` | Phase-3 continuation symmetry hook: `off`, `verify_only`, `postselect_diag_v1`, `projector_renorm_v1`. On ADAPT/hardcoded paths active estimator behavior is enforced only in oracle-backed noise runners. |
| `--phase3-runtime-split-mode` | choice | `off` | HH continuation add-on: `off` or `shortlist_pauli_children_v1`. Shortlist-only macro splitting for staged continuation/replay metadata; not a default pool-expansion policy. |
| `--adapt-allow-repeats` / `--adapt-no-repeats` | flag | `allow` | Allow selecting the same pool operator more than once |
| `--adapt-finite-angle-fallback` / `--adapt-no-finite-angle-fallback` | flag | `enabled` | Scan ±theta probes when gradients are below threshold |
| `--adapt-finite-angle` | float | `0.1` | Probe angle for finite-angle fallback |
| `--adapt-finite-angle-min-improvement` | float | `1e-12` | Minimum energy drop from probe to accept fallback |
| `--adapt-disable-hh-seed` | flag | `false` | Disable HH quadrature seed pre-optimization |
| `--adapt-drop-floor` | float | `auto` | Energy-drop plateau floor (`drop = ΔE_abs(d-1)-ΔE_abs(d)`). Omitted => HH `phase1_v1|phase2_v1|phase3_v1` resolves to `5e-4`; Hubbard / HH `legacy` stay off; pass negative to disable explicitly |
| `--adapt-drop-patience` | int | `auto` | Consecutive low-drop depths needed for plateau stop. Omitted => HH `phase1_v1|phase2_v1|phase3_v1` resolves to `3`; Hubbard / HH `legacy` stay off |
| `--adapt-drop-min-depth` | int | `auto` | Minimum depth before applying drop plateau stop. Omitted => HH `phase1_v1|phase2_v1|phase3_v1` resolves to `12`; Hubbard / HH `legacy` stay off |
| `--adapt-grad-floor` | float | `auto` | Optional secondary gradient floor guard for plateau stop. Omitted => HH `phase1_v1|phase2_v1|phase3_v1` resolves to `2e-2`; Hubbard / HH `legacy` disable it; pass negative to disable explicitly |
| `--adapt-ref-json` | path | `None` | Import ADAPT reference state from JSON `initial_state.amplitudes_qn_to_q0`; in HH `phase1_v1`/`phase2_v1`/`phase3_v1`, metadata-compatible warm/ADAPT JSON also reuses `ground_state.exact_energy_filtered` when present |
| `--dense-eigh-max-dim` | int | `8192` | Skip full dense diagonalization when Hilbert dim exceeds threshold (sector exact remains; trajectory skipped) |

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/README.md
(lines 24-75: README staged HH continuation contract, warm-start to ADAPT to matched-family replay, and report/dynamics artifact expectations relevant to a realtime-VQS extension.)
```md
### Warm-start chain

Default staged HH runs follow the active three-stage continuation contract:

1. Run HH-HVA VQE warm start with `hh_hva_ptw`.
   - `hh_hva_ptw` remains the canonical staged warm-start default.
   - `hh_hva` remains an explicit override only.
2. Use that warm-start state as the ADAPT reference state.
3. Run ADAPT from that prepared state in staged HH continuation mode (`phase3_v1`, canonical default).
   - For new HH agent work, depth-0 ADAPT starts from the narrow physics-aligned core; current runtime resolves this to `paop_lf_std`.
   - `full_meta` remains a supported broad-pool preset, but only as controlled residual enrichment after plateau diagnosis, not the default depth-0 path.
   - Optional phase-3 follow-ons stay opt-in: `--phase3-runtime-split-mode shortlist_pauli_children_v1` is a shortlist-only continuation aid, and widened `--phase3-symmetry-mitigation-mode` choices remain phase-3 metadata/telemetry hooks on raw staged/hardcoded/replay paths.
4. Replay conventional VQE from ADAPT with ADAPT-family matching (`--generator-family match_adapt`, fallback `full_meta`) via `pipelines/hardcoded/hh_vqe_from_adapt_family.py`.

Optional staged seed-refine insertion:

- `pipelines/hardcoded/hh_staged_noiseless.py` can insert one explicit-family conventional VQE refine stage between warm start and ADAPT via `--seed-refine-family`.
- Supported explicit seed-refine families are:
  - `uccsd_otimes_paop_lf_std`
  - `uccsd_otimes_paop_lf2_std`
  - `uccsd_otimes_paop_bond_disp_std`
- The refine stage materializes the requested family directly; it does **not** use `match_adapt` and does **not** auto-fallback to `full_meta`.
- If the refine stage succeeds, the handoff bundle carries additive `seed_provenance`.
- If the refine stage fails, the staged workflow aborts before ADAPT rather than silently skipping forward.

One-shot noiseless wrapper for the default or refined contract:

```bash
python pipelines/hardcoded/hh_staged_noiseless.py --L 2

# Optional refine insertion:
python pipelines/hardcoded/hh_staged_noiseless.py --L 2 \
  --seed-refine-family uccsd_otimes_paop_lf_std
```

This wrapper keeps drive opt-in, runs final matched-family replay (not fixed `hh_hva_*` replay), and reports Suzuki/CFQM dynamics from the replay seed with GS-baseline energy error plus seeded exact-reference fidelity.

Combined staged circuit PDF for `L=2,3`:

```bash
python pipelines/hardcoded/hh_staged_circuit_report.py
```

Default artifact:
- `artifacts/pdf/hh_staged_circuit_report_L2_L3.pdf`

Report contract:
- one combined PDF with separate `L=2` and `L=3` sections,
- per-`L` pages for manifest, stage summary, warm HH-HVA, ADAPT, matched-family replay, Suzuki2 macro-step, and a CFQM4 dynamics section,
- each circuit stage/method gets both a representative view (high-level `PauliEvolutionGate` blocks) and an expanded one-level decomposition view when circuitization is supported,
- dynamics pages show one representative macro-step only; the PDF states the repeat count and proxy totals for the full `trotter_steps` trajectory.
- Numerical-only CFQM stage backends (`expm_multiply_sparse`, `dense_expm`) stay in the report as dynamics metadata but are marked unsupported to avoid misleading compiled-circuit artifacts; representative/expanded circuit pages and transpile/proxy summaries are skipped for those modes.

```

(lines 190-252: README ADAPT pool summary, phase3_v1 defaults, runtime split notes, compiled-action acceleration, and matched-family replay contract surfaces.)
```md
### ADAPT Pool Summary (plaintext fallback)

- `hubbard` pools: `uccsd`, `cse`, `full_hamiltonian`.
- `hh` pools: `hva`, `full_hamiltonian`, `paop_min`, `paop_std`, `paop_full`, `paop_lf` (`paop_lf_std` alias), `paop_lf2_std`, `paop_lf_full`.
- Experimental offline/local exact-noiseless probe families: `paop_lf3_std`, `paop_lf4_std`, `paop_sq_std`, `paop_sq_full`.
- HH staged continuation default for new agent work: `phase3_v1` start from the narrow HH core and runtime-resolve depth-0 HH ADAPT to `paop_lf_std`.
- HH built-in combined preset: `uccsd_paop_lf_full` = `uccsd_lifted + paop_lf_full` (deduplicated) via one CLI value.
- HH explicit product families: `uccsd_otimes_paop_lf_std`, `uccsd_otimes_paop_lf2_std`, `uccsd_otimes_paop_bond_disp_std`.
  - These are the canonical lifted-UCCSD ⊗ boson-only-phonon constructions in this repo: one lifted fermionic UCCSD factor times one boson-only phonon motif, locality-filtered, canonicalized, and deduplicated.
  - They are available as explicit families for seed-refine, replay, and direct ADAPT pool materialization without mutating the older additive unions.
- HH logical two-parameter product variants: `uccsd_otimes_paop_lf_std_seq2p`, `uccsd_otimes_paop_lf2_std_seq2p`, `uccsd_otimes_paop_bond_disp_std_seq2p`.
  - These treat one logical `(F_a, M_μ)` pair as separate fermion/motif parameters during execution and replay.
  - They are additive opt-in surfaces and do not change the staged `phase3_v1` default path.
- HH full-meta preset: `full_meta` = `uccsd_lifted + hva + paop_full + paop_lf_full` (deduplicated) via one CLI value; keep it as a compatibility/broad-pool preset and replay fallback, not the default depth-0 staged HH pool.
- Opt-in runtime split (`--phase3-runtime-split-mode shortlist_pauli_children_v1`) probes shortlisted macro generators as serialized child terms for continuation/replay provenance; it does **not** change the default HH pool curriculum or create a new replay mode.
- `paop_min`: displacement-focused PAOP operators.
- `paop_std`: displacement plus dressed-hopping (`hopdrag`) operators.
- `paop_full`: `paop_std` plus doublon dressing and extended cloud operators.
- `paop_lf_std`: `paop_std` plus LF-leading odd channel (`curdrag`).
- These experimental families are opt-in only; they are not part of the canonical staged default and are not folded into default `full_meta`.
- HH merge behavior (when `g_ep != 0`): merge `hva` + `hh_termwise_augmented` + selected `paop_*` pool, then deduplicate by polynomial signature.

### Compiled speedup stack note (2026-03-04)

The hardcoded VQE/ADAPT path now includes a shared compiled-action acceleration stack, with additive (backward-compatible) interfaces and parity tests.

- Shared compiled polynomial utility:
  - `src/quantum/compiled_polynomial.py`
  - Provides `compile_polynomial_action`, `apply_compiled_polynomial`, `energy_via_one_apply`, and `adapt_commutator_grad_from_hpsi`.
- Compiled ansatz executor:
  - `src/quantum/compiled_ansatz.py`
  - Applies Pauli rotations through compiled permutation+phase actions (no per-amplitude string loops).
- VQE one-apply energy backend:
  - `src/quantum/vqe_latex_python_pairs.py` adds `expval_pauli_polynomial_one_apply(...)`.
- `vqe_minimize(...)` supports `energy_backend="legacy"|"one_apply_compiled"` (default is `one_apply_compiled`).
  - `pipelines/hardcoded/hubbard_pipeline.py` exposes `--vqe-energy-backend {legacy,one_apply_compiled}` and defaults to `one_apply_compiled`.
  - Hardcoded VQE can emit live progress heartbeats via `--vqe-progress-every-s` (default `60` seconds), including restart lifecycle and periodic energy/nfev telemetry.
- ADAPT runtime acceleration:
  - `pipelines/hardcoded/adapt_pipeline.py` compiles Hamiltonian/pool once, computes `H|psi>` once per depth, evaluates pool gradients via `2*Im(<Hpsi|Apsi>)`, and uses compiled ansatz execution in COBYLA objective/state updates.
- Regression coverage added:
  - `test/test_compiled_polynomial.py`
  - `test/test_compiled_ansatz.py`
  - `test/test_vqe_energy_backend.py`
  - existing ADAPT integration suite remains passing.
- Additive ADAPT telemetry fields:
  - `adapt_vqe.compiled_pauli_cache`
  - `adapt_vqe.history[*].gradient_eval_elapsed_s`
  - `adapt_vqe.history[*].optimizer_elapsed_s`
Fast VQE-from-ADAPT replay (HH, ADAPT-family matched):

```bash
python pipelines/hardcoded/hh_vqe_from_adapt_family.py \
  --adapt-input-json <adapt_hh_json_path> \
  --generator-family match_adapt --fallback-family full_meta \
  --L 4 --boundary open --ordering blocked \
  --boson-encoding binary --n-ph-max 1 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 \
  --reps 4 --restarts 16 --maxiter 12000 --method SPSA --seed 7 \
  --energy-backend one_apply_compiled --progress-every-s 60 \
  --output-json artifacts/json/hc_hh_L4_from_adaptB_family_matched_fastcomp.json
```

This path matches the ADAPT generator-family contract. Replay remains canonical via `--generator-family match_adapt` with `--fallback-family full_meta`, and replay continuation modes stay `legacy | phase1_v1 | phase2_v1 | phase3_v1`.
If an opt-in runtime split admitted child labels outside the resolved family pool, replay can still rebuild them from serialized continuation metadata when that metadata is present.

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/MATH/IMPLEMENT_SOON.md
(lines 430-549: Mathematical notes on tangent-space novelty score construction and candidate-selection targets relevant to replacing static phase3 scoring with realtime-VQS criteria.)
```md

This is the right numerator for an admission score because it answers:

> How much useful energy drop should this candidate produce on the current scaffold, under the refit policy I will actually execute, without taking a state-space step that is too large?

---

## Tangent-Space Novelty

A candidate should not be rewarded if its tangent direction is already spanned by the unlocked scaffold window.

Define the centered tangent direction
\[
t_{m,p}
=
-i\big(\widetilde A_{m,p}-\langle\widetilde A_{m,p}\rangle_\psi\big)|\psi\rangle.
\]

For tangent vectors \(\{t_j\}_{j\in W(p)}\) of the unlocked window, define
\[
[s_{m,p}]_j = \mathrm{Re}\,\langle t_j|t_{m,p}\rangle,
\qquad
[S_{W(p)}]_{jk} = \mathrm{Re}\,\langle t_j|t_k\rangle.
\]

Then define novelty
\[
N_{m,p}
=
1
-
\frac{s_{m,p}^{\top}(S_{W(p)}+\epsilon_N I)^{-1}s_{m,p}}{F_{m,p}}.
\]

Numerically clip \(N_{m,p}\) to \([0,1]\).

Interpretation:
- \(N_{m,p}\approx 1\): mostly outside the current window tangent span.
- \(N_{m,p}\approx 0\): mostly redundant with the current window.

Novelty is geometric; it is more principled than a crude family-repeat penalty.

---

## Lifetime Hardware Burden

A candidate should be ranked by predicted benefit per effective marginal burden.

Define
\[
K_{m,p}
=
1
+
w_D \bar D_{m,p}
+
w_G \bar G^{\mathrm{new}}_{m,p}
+
w_C \bar C^{\mathrm{new}}_{m,p}
+
w_P \bar P_{m,p}
+
w_c \bar c_m.
\]

Where:
- \(D_{m,p}\): compiled depth / entangler / 2-qubit burden increment;
- \(G^{\mathrm{new}}_{m,p}\): number of new commuting measurement groups introduced;
- \(C^{\mathrm{new}}_{m,p}\): extra shots required after reuse;
- \(P_{m,p}\): optimizer-dimension burden;
- \(c_m\): family reuse count or repeat count.

All costs should be normalized:
\[
\bar x = \frac{x}{x_{\mathrm{ref}}+\epsilon},
\]
where \(x_{\mathrm{ref}}\) is either a hard budget or a current-pool robust statistic such as a median.

### Additive, not multiplicative
The denominator should be additive because wall-clock cost, new measurement groups, shots, and optimizer dimension are first-order additive resources.
Do **not** multiply unrelated penalty factors.

---

## Primary Target Score

For candidate \(m\), evaluate a small set of allowed positions \(\mathcal P_m\).

Define the candidate score as

\[
S_m
=
\max_{p\in\mathcal P_m}
\left[
\mathbf 1(m\in\Omega_{\mathrm{stage}})
\mathbf 1(L_{m,p}\le L^{(\mathrm{stage})}_{\max})
e^{-\eta_L L_{m,p}}
N_{m,p}^{\gamma_N}
\frac{\Delta \widehat E^{\mathrm{TR}}_{m,p}}{K_{m,p}}
\right].
\]

Where:
- \(\Omega_{\mathrm{stage}}\): active pool for the current curriculum stage;
- \(L_{m,p}\): symmetry/leakage penalty;
- \(L_{\max}^{(\mathrm{stage})}\): hard leakage cap for the stage;
- \(\eta_L\): soft leakage penalty scale;
- \(\gamma_N\): novelty exponent.

This is the **full target score**, not the initial implementation requirement.

---

## Simplified Production-First Score

The first implementation may use a reduced score, provided the interface is built for a later upgrade.

### Simplified score
\[

```

(lines 1830-1919: Approximation ladder for novelty and curvature metrics, useful for deciding which local-geometry calculations are already represented in code versus missing for realtime VQS.)
```md
    # phase 3: full replay
    params = optimize(params, active_mask=active_mask, method=cfg.base_optimizer, backend_ctx=backend_ctx)
    return params
```

---

## Novelty Estimation Notes

A first implementation may not be able to compute exact tangent-span novelty on hardware.

### Acceptable approximation ladder
1. **Exact simulator tangent novelty**
   - Use explicit statevector tangent overlaps.

2. **Approximate novelty from compiled-action tangents**
   - If tangent vectors can be produced cheaply.

3. **Cheap proxy**
   - Family-level novelty proxy,
   - support-overlap proxy,
   - or novelty omitted in `simple_v1`.

### Rule
If novelty is omitted in `simple_v1`, do not fake it with an arbitrary family penalty.
Instead, leave the interface open and rely on stage gating + repeat cost until a better novelty estimate is available.

---

## Curvature Estimation Notes

A first implementation may also lack robust \(h,b,H\) estimates.

### Acceptable approximation ladder
1. **Simulator finite differences**
   - use exact energy evaluations;
2. **Quasi-Newton recycle**
   - inherit approximate inverse-Hessian data from previous optimizer steps;
3. **QN-SPSA / metric proxy**
   - use a metric-aware surrogate in replay;
4. **LM surrogate**
   - use \(h_{\mathrm{eff}} \approx \lambda_F F\) in `simple_v1`.

### Rule
`simple_v1` should not block `full_v2`.
The code must already have an abstraction boundary for curvature estimation.

---

## Cost Normalization and Weights

A clean generic normalization is
\[
\bar x = \frac{x}{x_{\mathrm{ref}}+\epsilon}.
\]

### Reference choice
Use one of:
- fixed budget;
- rolling median over active pool;
- rolling median over accepted candidates.

### Recommended initial weights
\[
w_D = w_G = w_C = 1,
\qquad
w_P = 0 \text{ if every candidate adds one parameter},
\qquad
w_c \in [0.1, 0.5].
\]

### Confidence multiplier
- early screening:
  \[
  z_\alpha \approx 1;
  \]
- late residual enrichment:
  \[
  z_\alpha \in [1.5, 2].
  \]

### Novelty exponent
- early targeted ADAPT:
  \[
  \gamma_N = 1;
  \]
- late residual enrichment:
  \[
  \gamma_N \approx 1/2.
  \]

```

(lines 2150-2259: Likely repo touchpoints and anti-pattern guidance for integrating new HH selection logic into existing continuation and compiled-action surfaces.)
```md
### Deliverable
A hardware-oriented, scalable continuation framework.

---

## Likely Repo Touchpoints (To Confirm by Audit)

These are likely integration surfaces inferred from current repo documentation; a repo-aware agent should confirm them before editing.

### Likely orchestration touchpoints
- `pipelines/hardcoded/adapt_pipeline.py` for the ADAPT control loop, scoring, pruning hooks, and telemetry.
- `pipelines/hardcoded/hh_vqe_from_adapt_family.py` for scaffolded replay, provenance-aware replay seeds, and replay optimizer phases.
- `src/quantum/operator_pools/` or equivalent HH pool builders for stage-aware pool curriculum and macro-generator metadata.
- `src/quantum/compiled_polynomial.py` and `src/quantum/compiled_ansatz.py` or equivalent compiled-action utilities for executable-scaffold reuse, tangent/metric helper hooks, and cost proxies.
- measurement grouping / reporting / JSON artifact writers for cache accounting and telemetry emission.
- test suites covering ADAPT, replay, windowed refits, and compiled parity.


### Likely “do not rewrite” surfaces
- operator algebra core;
- canonical PauliTerm source;
- JW ladder/operator source of truth.

### Likely new service modules
A clean implementation would likely introduce new service modules or shims for:
- stage control;
- score engine;
- insertion policy;
- compile-cost oracle;
- measurement cache;
- curvature/novelty oracles;
- pruning engine;
- replay controller.

The exact module names should be chosen by the repo-aware agent after audit.

---

## Required Anti-Patterns to Avoid

1. **Do not reopen `full_meta` from depth 0.**
2. **Do not score by raw \(|g|\) alone.**
3. **Do not assume append-only gradients diagnose true convergence.**
4. **Do not multiply unrelated penalty factors in the denominator.**
5. **Do not tile inherited angles across replay layers by default.**
6. **Do not prune during noisy early growth.**
7. **Do not atomize every macro-generator into Pauli factors.**
8. **Do not discard optimizer memory when the ansatz grows.**
9. **Do not rebuild the executable scaffold from abstract operator labels if compiled artifacts can be preserved.**
10. **Do not introduce architectural dependence on unavailable curvature estimates.**
    Build the interfaces first; fill them progressively.

---

## Optional Simulator-Only Rescue Mode

A simulator-only rescue mode may be kept for diagnostics.

### Purpose
When ordinary adaptive growth is clearly trapped in a bad local-minimum structure,
allow an overlap-guided or alternate-objective pass to produce a compact rescue scaffold.

### Rules
- simulator-only by default;
- off the main QPU path;
- used only when ordinary continuation diagnostics indicate failure;
- result should feed back into the normal scaffold representation.

This is a diagnostic valve, not the default control logic.

---

## Motif Tiling for Larger Systems

When scaling in system size \(L\), the implementation should be able to learn operator motifs on smaller systems and lift/tile them.

### Motif extraction
After solving a smaller instance, extract:
- accepted local operator families;
- relative ordering motifs;
- retained macro-generator blocks;
- common parameter sign/magnitude patterns;
- local symmetry metadata.

### Motif library
Store motifs in a transferable representation:
- local support pattern;
- family/template id;
- translation rules;
- admissible boundary behavior.

### Larger-\(L\) use
Use the motif library to:
- seed the pool,
- seed the scaffold,
- initialize replay families.

This is not Phase 1 work, but the operator metadata should be designed so motifs can later be extracted.

---

## Suggested JSON/Artifact Payload Additions

Actual field names are for the repo-aware agent to decide, but the payload should eventually capture:

### Per depth
- `stage_name`
- `positions_considered`
- `selected_position`
- `cheap_score`

```

(lines 2350-2409: Symbol glossary mapping the math notes onto the repo's tangent-vector and reduced-geometry terminology.)
```md

\(p\): insertion position in the current scaffold.

\(\mathcal P_m\): allowed insertion positions for candidate \(m\).

\(\widetilde A_{m,p}\): candidate dressed by the scaffold suffix after position \(p\).

\(\hat g_{m,p}\): estimated zero-initialized gradient.

\(\hat \sigma_{m,p}\): standard deviation of the gradient estimator.

\(g^\downarrow_{m,p}\): lower-confidence gradient magnitude.

\(F_{m,p}\): Fubini–Study metric element / generator variance.

\(W(p)\): actual inherited-parameter window to be unlocked after admission at \(p\).

\(H_{W(p)}\): Hessian or Hessian proxy over the unlocked window.

\(h_{m,p}\): candidate self-curvature.

\(b_{m,p}\): candidate–window cross-curvature vector.

\(\widetilde h_{m,p}\): effective curvature after window relaxation.

\(t_{m,p}\): candidate tangent vector in Hilbert space.

\(N_{m,p}\): novelty, i.e. tangent-norm fraction outside the active window span.

\(D_{m,p}\): compiled depth / entangler burden.

\(G^{\mathrm{new}}_{m,p}\): number of new measurement groups.

\(C^{\mathrm{new}}_{m,p}\): additional shots required after reuse.

\(P_{m,p}\): optimizer dimension burden.

\(c_m\): family repeat/reuse count.

\(L_{m,p}\): symmetry/leakage penalty.

\(\Omega_{\mathrm{stage}}\): currently active pool stage.

\(\rho\): state-space trust radius.

\(z_\alpha\): confidence multiplier.

\(\lambda_H\): Hessian regularization ridge.

\(\lambda_F\): metric-curvature surrogate scale for simplified scoring.

\(\gamma_N\): novelty exponent.

\(\eta_L\): softness of the leakage penalty.

\(N_{\mathrm{rem}}\): estimated remaining objective evaluations after candidate admission.

---

## Appendix B — Immediate Post-Audit Questions for a Coding Agent

```
</file_contents>
<meta prompt 1 = "System: MCP Pair Program">
**CRITICAL: THIS IS YOUR OPERATING SYSTEM - FOLLOW THESE INSTRUCTIONS EXACTLY**

You are a **pair‑programming conductor**. Drive the **entire implementation** inside **one long chat thread** using `chat_send`. Act as **context curator** and **project manager**: plan the work, curate files between steps, trigger edits, verify diffs, mend gaps, and push to completion.

**Provenance & State**
This prompt comes directly from RepoPrompt. The MCP server’s workspace state already matches what you see here — a partial, selected-mode file tree (trimmed for size), codemaps for some selected files, and the current `user_instructions` prompt are embedded. No pre-flight verification is required for the first turn. If you need to drill deeper into specific directories, use `get_file_tree`.

**MANDATORY WORKFLOW - DO NOT DEVIATE:**
1. ALL code changes happen through `chat_send` with `mode:"edit"`
2. You NEVER use built-in edit tools - they are FORBIDDEN
3. You NEVER use `apply_edits` except for mechanical bulk changes
4. The chat is your implementation engine - USE IT FOR EVERYTHING

You do NOT write code yourself - you orchestrate the chat to write ALL code.

**North Star**
- **Start fresh**: Use `new_chat:true` for your FIRST message (unless user explicitly says "continue/resume")
- **Stay in session**: After that first message, ALWAYS use `new_chat:false` for the entire implementation
- Keep this **single chat session** for the whole task (set a descriptive `chat_name`)
- **ALL implementation MUST happen through `chat_send` with `mode:"edit"`**. The chat writes ALL the code.
- **NEVER use your own built-in edit capabilities** - always use RepoPrompt tools:
	- Primary: `chat_send` with `mode:"edit"` for ALL implementation
	- Fallback: `apply_edits` ONLY if chat_send truly cannot handle it (extremely rare)
	- FORBIDDEN: Any built-in file editing, writing, or modification tools you may have
- When tempted to edit files directly, use `chat_send` `mode:"edit"` instead
- If you absolutely must use `apply_edits` (mechanical 20+ file renames), inform the chat afterward.
- After **every step**, **check your work** (diffs + targeted reads + searches) and **mend** incomplete edits.
- **Context is a tool**: curate the selection between steps. Target **50–80k tokens** but go higher when needed for completeness.

---

## Prime the session (one time)
- Check available models and their capabilities:
	```json
	{"tool":"list_models","args":{}}
	```
- Inspect current state (selection, tokens):
	```json
	{"tool":"workspace_context","args":{"include":["selection","tokens"],"path_display":"relative"}}
	```
- Overview of modules (folder map first, fast): prefer `get_file_tree` **auto** mode
  ```json
  {"tool":"get_file_tree","args":{"type":"files","mode":"auto"}}
  ```
  Then narrow scope as needed:
  ```json
  {"tool":"get_file_tree","args":{"type":"files","mode":"auto","path":"RootName/feature","max_depth":2}}
  ```
- Curate an initial focused selection (directories + key files). Prefer root‑prefixed paths:
  ```json
	{"tool":"manage_selection","args":{"op":"set","paths":["Root/src/feature","Root/src/shared/Types.swift"],"view":"files","strict":true}}
	```
---

## The core loop: **Plan → Edit → Verify & Mend**

Repeat this loop until done. Stay in the **same chat**.

### 1) Plan (clarify scope, break into steps)
**ALWAYS start with `new_chat:true` unless user explicitly says to continue/resume:**
```json
{"tool":"chat_send","args":{
	"message":"Plan: Implement user preferences system. Context: Currently settings are scattered across UserDefaults keys (found 'app.theme', 'notifications.enabled' via search). Existing UserStore at Root/Stores/UserStore.swift handles user data. No centralized preferences model exists. Goal: Create unified preferences system that consolidates all user settings, provides type-safe access, and persists reliably. Selected Root/Models, Root/ViewModels, Root/Views/Settings, and Root/Stores for context. Please outline architecture and implementation approach.",
	"new_chat":true,
	"mode":"plan",
	"model":"model-id-from-list_models",
	"chat_name":"User Preferences Implementation"
}}
```
After this first message, use `new_chat:false` for the entire rest of the implementation.

If already mid‑execution, **reorient** instead:
```json
{"tool":"chat_send","args":{
	"message":"Reorient: We are at step N. Current state: <high‑level summary>. Next up: <focused objective>. Confirm or refine steps.",
	"new_chat":false,
	"mode":"plan"
}}
```

### 2) Curate selection for the current step (tight, relevant)
```json
{"tool":"manage_selection","args":{
	"op":"set",
	"paths":["Root/src/feature","Root/src/feature/View.swift","Root/src/shared/Store.swift"],
	"view":"files",
	"strict":true
}}
```

**CRITICAL:** If the chat asks for files you haven't selected, immediately add them using `manage_selection` before sending your next message. The chat cannot see files that aren't selected.

If token pressure is high, pivot to slices:
- Read sections with `read_file` before slicing
- Use `ranges` with descriptions (`description`/`desc`/`label`)
- Preview with `op:"preview"`, apply with `op:"set"` `mode:"slices"`
- Prefer 80–150+ line self-contained slices
- If you omit critical context, the task will fail

**Token policy:** Target 50–80k; exceed when needed for completeness. After each step, prune only what's definitively no longer relevant:
```json
{"tool":"manage_selection","args":{"op":"get","view":"files"}}
```

### 3) Edit (the chat performs the change)
**THE CHAT IS YOUR IMPLEMENTATION ENGINE** - let it write all the code:
```json
{"tool":"chat_send","args":{
	"message":"Edit: Based on the plan, implement the preferences model. Context: Found existing UserDefaults keys 'app.theme' and 'notifications.enabled' that need to be migrated. UserStore.swift has currentUser property we'll need to associate preferences with. PreferencesView.swift currently has placeholder Text views at lines 45-60. Goal: Create the model with proper persistence, ensuring existing settings are preserved when users update. You determine the best approach for structure and implementation.",
	"new_chat":false,
	"mode":"edit"
}}
```

**Edit ambition & scope**
Be strategically ambitious per `chat_send` turn: a single `mode:"edit"` request may span multiple sections across multiple files when they collectively serve one clear objective.

**Remember: `chat_send` mode="edit" is for ALL real implementation**
- Describe the PROBLEM clearly (what's broken, what's missing)
- Provide CONTEXT (what you found, where you looked, existing patterns)
- State the GOAL (desired behavior, not how to code it)
- Reference specific discoveries (revelant symbols, existing implementations)
- Let the chat decide HOW to implement based on its plan
- DO NOT use `apply_edits` for implementation—that's the chat's job

> Optional: Per‑message overrides via `selected_paths` if you don't want to change the global selection.

### 4) Verify & Mend (after **every** edit)
- Inspect diffs of the last messages:
	```json
	{"tool":"chats","args":{"action":"log","limit":1,"include_diffs":true}}
	```
- Targeted reads around modified areas:
	```json
	{"tool":"read_file","args":{"path":"Root/src/feature/View.swift","start_line":40,"limit":40}}
	```
- Sanity/search checks (rename completeness, TODOs, old symbols, etc.):
	```json
	{"tool":"file_search","args":{"pattern":"OldSymbol","regex":false,"mode":"both","max_results":250}}
	```

**If the chat's edit is incomplete or inconsistent:**
- **ALWAYS prefer asking the chat to fix it** - describe what's wrong, not how to fix it:
	```json
	{"tool":"chat_send","args":{
	"message":"Mend: Issue found - preferences aren't persisting after app restart. When I set notifications to OFF and restart, they're back ON. Also, the app crashes when UserDefaults contains malformed data (tested by manually corrupting the plist). The save appears to execute but changes don't persist. Need reliable persistence and graceful handling of corrupted data.",
	"new_chat":false,
	"mode":"edit"
	}}
	```
- Use `apply_edits` ONLY for truly trivial fixes (missing semicolon, typo in string):
	```json
	{"tool":"apply_edits","args":{
	"path":"Root/src/feature/View.swift",
	"search":"old","replace":"new","all":false,
	"verbose":true
	}}
	```
	Then immediately summarize to the chat:
	```json
	{"tool":"chat_send","args":{
	"message":"Delta: Applied a small fix to View.swift (replaced 'old'→'new'). Proceeding to the next sub‑step.",
	"new_chat":false,
	"mode":"chat"
	}}
	```

---

## Repo‑wide mechanical changes (e.g., "replace all references of X")
Use **`file_search` + `apply_edits`** (mechanical), then return control to chat:
```json
{"tool":"file_search","args":{"pattern":"\bOldName\b","regex":true,"mode":"both","max_results":250}}
```
(Iterate per matched file)
```json
{"tool":"apply_edits","args":{
	"path":"Root/path/File.swift",
	"search":"OldName","replace":"NewName","all":true,
	"verbose":true
}}
```
Verify zero remaining:
```json
{"tool":"file_search","args":{"pattern":"\bOldName\b","regex":true,"mode":"both","max_results":250}}
```
And **brief the chat** with a high‑level delta before continuing:
```json
{"tool":"chat_send","args":{
	"message":"Delta: Completed repo‑wide OldName→NewName refactor across N files. No matches remain. Next: <next step>.",
	"new_chat":false,
	"mode":"chat"
}}
```

---

## Mid‑execution recovery / staying in one thread
- Find available sessions (returns ID, name, selected files, last activity):
	```json
	{"tool":"chats","args":{"action":"list","limit":10}}
	```
- View detailed history of a specific chat:
	```json
	{"tool":"chats","args":{"action":"log","chat_id":"abc-123","limit":5,"include_diffs":true}}
	```
- Continue the most recent or specify `chat_id`. Re‑state current state and next objective:
	```json
	{"tool":"chat_send","args":{
	"message":"Re‑sync: Since the last turn we accomplished <X>. Next objective: <Y>. Confirm plan.",
	"new_chat":false,
	"chat_id":"abc-123",
	"mode":"plan"
	}}
	```

---

## Selection curation doctrine
- Before each sub‑task: **include everything relevant** for that step; add needed neighbors (interfaces, shared types, dependencies).
- After the step: **keep context that might be needed** for upcoming steps; only prune what's definitively complete.
- **Bias toward inclusion**: Target 50–80k tokens, but go higher when needed for completeness. Better to have context available than to miss critical dependencies.

---

## Limits & reminders
- No terminal/commands/tests—verification relies on diffs, searches, and targeted reads.
- Chat sees **selected files + history**, not build output.
- Diffs are authoritative when produced by tools (`apply_edits` with `verbose:true`, or `chats log` with `include_diffs:true`). Use them to drive the mend loop.

**Your job:** keep a single, coherent thread; orchestrate plan→edit→verify→mend cycles; curate context relentlessly; and deliver a finished implementation.

**FINAL REMINDER: You are the conductor, not the implementer. EVERY line of code must be written by the chat through `chat_send`. If you find yourself writing code, you're doing it wrong. Use `chat_send` with `mode:"edit"` for ALL implementation.**
</meta prompt 1>
<user_instructions>
<mcp_metadata>
window_id: 1
workspace_instance: 1
workspace_name: holstein_clone_2

tab_id: 3D27C63F-ED15-4F88-AB6A-A6AC2F352E3F
tab_name: Plan export: realtime VQS pipeline

context_binding_guide:
  Before making tool calls to RepoPrompt, bind to this context:
  1. Select the window:
     tool: select_window
     args: {"window_id": 1}
  2. Bind to this tab (recommended for stable context):
     tool: manage_workspaces
     args: {"action": "select_tab", "tab": "3D27C63F-ED15-4F88-AB6A-A6AC2F352E3F"}
</mcp_metadata>

<taskname="RT VQS handoff"/>
<task>
Produce a single strong ChatGPT-ready architecture-plan prompt export for a new Hubbard-Holstein real-time dynamics workflow based on McLachlan variational quantum simulation. Do not implement code. Use the selected repo context to determine which existing `phase3_v1` / beam / replay / export scaffolding should be reused, which static ADAPT-VQE scoring and stop logic should be replaced for time-dependent evolution, what mathematically appropriate realtime branch-selection and local-geometry criteria should be proposed, and where a repo-native new subtree should live. Keep the plan additive and repo-native rather than a greenfield rewrite. The prompt export should target `prompt-exports/2026-03-15-plan-hh-realtime-vqs-architecture.md`.
</task>

<architecture>
- `src/quantum/time_propagation/*` is the existing generic dynamics/VQS layer. `projected_real_time.py` already implements McLachlan-style tangent construction and RK4 trajectory integration; `local_checkpoint_fit.py` is the adjacent local-ansatz fitting surface; `cfqm_propagator.py` / `cfqm_schemes.py` cover exact or CFQM propagation utilities.
- `pipelines/hardcoded/adapt_pipeline.py` owns the HH staged ADAPT continuation entrypoint and branch-state containers; `hh_continuation_scoring.py`, `hh_continuation_types.py`, `hh_continuation_stage_control.py`, `hh_continuation_generators.py`, `hh_continuation_replay.py`, `hh_continuation_symmetry.py`, `hh_continuation_pruning.py`, and `hh_continuation_rescue.py` provide the reusable phase scaffolding around selection, beam control, runtime split, replay, and recovery.
- `pipelines/hardcoded/hh_vqe_from_adapt_family.py` is the current matched-family replay and seed-contract layer. `pipelines/hardcoded/hh_staged_workflow.py` is the production HH orchestration boundary that writes handoff bundles, runs replay, and then runs noiseless dynamics/reporting. `pipelines/hardcoded/handoff_state_bundle.py` is the artifact/export schema.
- `src/quantum/operator_pools/polaron_paop.py` and `src/quantum/operator_pools/vlf_sq.py` are the main HH/operator-pool family builders.
- `pipelines/exact_bench/hh_fixed_seed_budgeted_projected_dynamics.py` is the clearest existing prototype that connects replay sequences to McLachlan projected real-time dynamics.
- Discovery inference: a new HH-specific realtime-VQS subtree most naturally belongs under `pipelines/hardcoded/` for orchestration, branch state, replay/export, and workflow glue, while any generic tangent/McLachlan math should remain in or extend `src/quantum/time_propagation/`.
</architecture>

<selected_context>
README.md: staged HH contract, `phase3_v1` default, runtime split note, compiled ADAPT speedup note, matched-family replay contract.
pipelines/run_guide.md: canonical HH staged workflow, `phase3_v1` stop-policy semantics, replay provenance fields, and relevant CLI knobs.
pipelines/hardcoded/adapt_pipeline.py: `_BeamBranchState`, `_ADAPTLogicalCandidate`, `_run_hardcoded_adapt_vqe()`; reusable branch/continuation shell and phase3 controls.
pipelines/hardcoded/hh_continuation_scoring.py: `Phase2NoveltyOracle`, `Phase2CurvatureOracle`, `build_full_candidate_features()`, tangent-overlap and derivative propagation helpers; main static geometry/scoring surface to compare against realtime needs.
pipelines/hardcoded/hh_continuation_types.py: `CandidateFeatures`, `ReplayPlan`, `ReplayPhaseTelemetry`, `GeneratorMetadata`, `GeneratorSplitEvent`, `Phase2OptimizerMemoryAdapter`; state payloads reused across selection/replay/export.
pipelines/hardcoded/hh_continuation_stage_control.py: `StageController` and trough/probe gating heuristics.
pipelines/hardcoded/hh_continuation_generators.py: generator metadata registry, runtime split children, split-event serialization.
pipelines/hardcoded/hh_continuation_replay.py: `build_replay_plan()`, `run_phase1_replay()`, `run_phase2_replay()`, `run_phase3_replay()`; existing phase-aware replay semantics.
pipelines/hardcoded/hh_continuation_symmetry.py: symmetry mitigation modes and verification hooks.
pipelines/hardcoded/hh_continuation_pruning.py: prune/refit helpers.
pipelines/hardcoded/hh_continuation_rescue.py: rescue trigger/ranking helpers.
pipelines/hardcoded/hh_continuation_motifs.py: motif-library extraction, transfer, and tiled-generator seeding surfaces.
pipelines/hardcoded/hh_vqe_from_adapt_family.py: replay contract parsing, `handoff_state_kind` inference, replay seed policies, `build_replay_sequence_from_input_json()`, `build_family_ansatz_context()`, `build_replay_ansatz_context()`, `run()`.
pipelines/hardcoded/hh_staged_workflow.py: `AdaptConfig`, `ReplayConfig`, `DynamicsConfig`, `StageExecutionResult`, `_write_adapt_handoff()`, `run_stage_pipeline()`, `_run_noiseless_profile()`, `run_noiseless_profiles()`, `build_stage_circuit_report_artifacts()`.
pipelines/hardcoded/handoff_state_bundle.py: `write_handoff_state_bundle()` and the current continuation/export schema.
pipelines/hardcoded/hubbard_pipeline.py: legacy `_simulate_trajectory()` surface still used by staged noiseless profiles.
pipelines/exact_bench/hh_fixed_seed_budgeted_projected_dynamics.py: `run_sweep()`; current experiment that turns replay sequences into projected-real-time prefix sweeps under a budget.
pipelines/exact_bench/hh_fixed_seed_local_checkpoint_fit.py: local-checkpoint-fit sweep surface using replay sequences and exact references.
src/quantum/time_propagation/projected_real_time.py: `ProjectedRealTimeConfig`, `build_tangent_vectors()`, `solve_mclachlan_step()`, `run_projected_real_time_trajectory()`, `run_exact_driven_reference()`.
src/quantum/time_propagation/local_checkpoint_fit.py: `LocalPauliAnsatzSpec`, `CheckpointFitConfig`, `fit_checkpoint_target_state()`, `fit_checkpoint_trajectory()`.
src/quantum/time_propagation/cfqm_propagator.py: `CFQMConfig`, `cfqm_step()`.
src/quantum/compiled_ansatz.py: `CompiledAnsatzExecutor`; reused by continuation scoring derivative propagation.
src/quantum/compiled_polynomial.py: compiled Hamiltonian application and commutator-gradient utilities.
src/quantum/operator_pools/polaron_paop.py: PAOP family builders and HH pool surfaces.
src/quantum/operator_pools/vlf_sq.py: VLF/SQ operator-pool builders.
src/quantum/drives_time_potential.py: drive-waveform provider surfaces used by staged and exact-bench dynamics.
MATH/IMPLEMENT_SOON.md: tangent-space novelty math, approximation ladder, repo touchpoints, and symbol glossary.
test/test_hh_adapt_beam_search.py: branch-clone and beam invariants around `_BeamBranchState`.
test/test_hh_continuation_scoring.py: contracts for novelty/curvature/full_v2 scoring and reduced-path fields.
test/test_hh_continuation_replay.py: replay phase telemetry and optimizer-memory reuse behavior.
test/test_hh_staged_noiseless_workflow.py: staged workflow defaults, seed-refine insertion, matched-family replay, and handoff-pool inference contracts.
test/test_hh_vqe_from_adapt_family_seed.py: replay contract and seed-policy behavior.
test/test_staged_export_replay_roundtrip.py: staged export/replay roundtrip and provenance expectations.
test/test_projected_real_time.py: McLachlan tangent/trajectory tests.
test/test_local_checkpoint_fit.py: checkpoint-fit behavior.
</selected_context>

<relationships>
- `hh_staged_workflow.run_stage_pipeline()` -> `adapt_pipeline._run_hardcoded_adapt_vqe()` -> `_write_adapt_handoff()` -> `handoff_state_bundle.write_handoff_state_bundle()` -> `hh_vqe_from_adapt_family.run()` -> `run_noiseless_profiles()` -> `hubbard_pipeline._simulate_trajectory()`.
- `adapt_pipeline._BeamBranchState` is the branch container that can carry optimizer memory, scaffold history, selected generator metadata, split events, motif usage, symmetry mitigation, prune summaries, and rescue history across continuation steps.
- `hh_continuation_scoring.build_full_candidate_features()` uses compiled-action and tangent machinery (`CompiledAnsatzExecutor`, compiled polynomial application, derivative propagation) to estimate novelty/curvature-style fields for static ADAPT phase scoring.
- `hh_continuation_generators.build_runtime_split_children()` and `build_split_event()` define how macro generators become serialized child terms that replay/export can reconstruct later.
- `hh_vqe_from_adapt_family.build_replay_sequence_from_input_json()` is the bridge from staged ADAPT exports to replay or exact-bench realtime sweeps.
- `hh_fixed_seed_budgeted_projected_dynamics.run_sweep()` already combines replay-sequence extraction with `run_exact_driven_reference()` and `run_projected_real_time_trajectory()`, making it the closest existing realtime-VQS workflow prototype.
- `projected_real_time.solve_mclachlan_step()` and `build_tangent_vectors()` are the generic math anchors; `local_checkpoint_fit.fit_checkpoint_trajectory()` is adjacent state-fitting machinery that may inform branch refresh or checkpointing strategies.
- Best repo-native placement appears to be a new HH-specific subtree such as `pipelines/hardcoded/hh_realtime_vqs/` for workflow, branch, and export logic, while keeping generic reusable solvers in `src/quantum/time_propagation/`.
</relationships>

<ambiguities>
- `phase3_v1` currently means a static ADAPT-VQE continuation/scoring regime; the reusable pieces are the branch/replay/export shells, not the static objective itself.
- The repo already has generic McLachlan and checkpoint-fit code, but no dedicated production HH realtime-VQS orchestration package yet.
- Staged post-replay dynamics currently run via exact, CFQM, or Suzuki propagation in `hubbard_pipeline.py`; this is adjacent to, but not the same as, branchwise variational realtime evolution.
</ambiguities>
</user_instructions>
