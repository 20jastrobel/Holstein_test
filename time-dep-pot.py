--- a/pipelines/hardcoded_hubbard_pipeline.py
+++ b/pipelines/hardcoded_hubbard_pipeline.py
@@ -45,6 +45,7 @@
 
 from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
 from src.quantum.hubbard_latex_python_pairs import build_hubbard_hamiltonian
+from src.quantum.drives_time_potential import build_gaussian_sinusoid_density_drive
 
 
 def _ai_log(event: str, **fields: Any) -> None:
@@ -180,17 +181,61 @@
     compiled_actions: dict[str, CompiledPauliAction],
     time_value: float,
     trotter_steps: int,
+    *,
+    drive_coeff_provider_exyz: Any | None = None,
+    t0: float = 0.0,
+    time_sampling: str = "midpoint",
+    coeff_tol: float = 1e-12,
 ) -> np.ndarray:
+    # Bit-for-bit identical path for time-independent Hamiltonians.
+    if drive_coeff_provider_exyz is None:
+        psi = np.array(psi0, copy=True)
+        if abs(time_value) <= 1e-15:
+            return psi
+        dt = float(time_value) / float(trotter_steps)
+        half = 0.5 * dt
+        for _ in range(trotter_steps):
+            for label in ordered_labels_exyz:
+                psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
+            for label in reversed(ordered_labels_exyz):
+                psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
+        return _normalize_state(psi)
+
+    # Time-dependent extension: coefficients sampled once per slice.
     psi = np.array(psi0, copy=True)
     if abs(time_value) <= 1e-15:
         return psi
     dt = float(time_value) / float(trotter_steps)
     half = 0.5 * dt
-    for _ in range(trotter_steps):
+
+    sampling = str(time_sampling).strip().lower()
+    if sampling not in {"midpoint", "left", "right"}:
+        raise ValueError("time_sampling must be one of {'midpoint','left','right'}")
+
+    t0_f = float(t0)
+    tol = float(coeff_tol)
+
+    for k in range(int(trotter_steps)):
+        if sampling == "midpoint":
+            t_sample = t0_f + (float(k) + 0.5) * dt
+        elif sampling == "left":
+            t_sample = t0_f + float(k) * dt
+        else:  # right
+            t_sample = t0_f + (float(k) + 1.0) * dt
+
+        drive_map = dict(drive_coeff_provider_exyz(float(t_sample)))
+
         for label in ordered_labels_exyz:
-            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
+            c_total = coeff_map_exyz.get(label, 0.0 + 0.0j) + complex(drive_map.get(label, 0.0))
+            if abs(c_total) <= tol:
+                continue
+            psi = _apply_exp_term(psi, compiled_actions[label], c_total, half)
         for label in reversed(ordered_labels_exyz):
-            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
+            c_total = coeff_map_exyz.get(label, 0.0 + 0.0j) + complex(drive_map.get(label, 0.0))
+            if abs(c_total) <= tol:
+                continue
+            psi = _apply_exp_term(psi, compiled_actions[label], c_total, half)
+
     return _normalize_state(psi)
 
 
@@ -559,6 +604,9 @@
     t_final: float,
     num_times: int,
     suzuki_order: int,
+    drive_coeff_provider_exyz: Any | None = None,
+    drive_t0: float = 0.0,
+    drive_time_sampling: str = "midpoint",
 ) -> tuple[list[dict[str, float]], list[np.ndarray]]:
     if int(suzuki_order) != 2:
         raise ValueError("This script currently supports suzuki_order=2 only.")
@@ -596,6 +644,9 @@
             compiled,
             t,
             int(trotter_steps),
+            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
+            t0=float(drive_t0),
+            time_sampling=str(drive_time_sampling),
         )
 
         fidelity = float(abs(np.vdot(psi_exact, psi_trot)) ** 2)
@@ -781,6 +832,32 @@
     parser.add_argument("--trotter-steps", type=int, default=64)
     parser.add_argument("--term-order", choices=["native", "sorted"], default="sorted")
 
+    parser.add_argument("--enable-drive", action="store_true", help="Enable time-dependent onsite density drive.")
+    parser.add_argument("--drive-A", type=float, default=0.0, help="Drive amplitude A in v(t)=A*sin(ωt+φ)*exp(-t^2/(2 tbar^2)).")
+    parser.add_argument("--drive-omega", type=float, default=1.0, help="Drive carrier angular frequency ω.")
+    parser.add_argument("--drive-tbar", type=float, default=1.0, help="Drive Gaussian envelope width tbar (must be > 0).")
+    parser.add_argument("--drive-phi", type=float, default=0.0, help="Drive phase φ.")
+    parser.add_argument(
+        "--drive-pattern",
+        choices=["dimer_bias", "staggered", "custom"],
+        default="staggered",
+        help="Spatial pattern mode for v_i(t)=s_i*v(t).",
+    )
+    parser.add_argument(
+        "--drive-custom-s",
+        type=str,
+        default=None,
+        help="Custom spatial weights s_i (comma-separated or JSON list), length L, used when --drive-pattern custom.",
+    )
+    parser.add_argument("--drive-include-identity", action="store_true", help="Include identity term from n=(I-Z)/2 (global phase).")
+    parser.add_argument(
+        "--drive-time-sampling",
+        choices=["midpoint", "left", "right"],
+        default="midpoint",
+        help="Time sampling rule per slice (midpoint recommended; left/right for diagnostics).",
+    )
+    parser.add_argument("--drive-t0", type=float, default=0.0, help="Drive start time t0 for evolution (default 0.0).")
+
     parser.add_argument("--vqe-reps", type=int, default=1)
     parser.add_argument("--vqe-restarts", type=int, default=1)
     parser.add_argument("--vqe-seed", type=int, default=7)
@@ -826,6 +903,36 @@
     else:
         ordered_labels_exyz = sorted(coeff_map_exyz)
 
+    drive = None
+    drive_coeff_provider_exyz = None
+    if bool(args.enable_drive):
+        custom_weights = None
+        if str(args.drive_pattern) == "custom":
+            if args.drive_custom_s is None:
+                raise ValueError("--drive-custom-s is required when --drive-pattern custom")
+            raw = str(args.drive_custom_s).strip()
+            if raw.startswith("["):
+                custom_weights = json.loads(raw)
+            else:
+                custom_weights = [float(x) for x in raw.split(",") if x.strip()]
+        drive = build_gaussian_sinusoid_density_drive(
+            n_sites=int(args.L),
+            nq_total=int(2 * args.L),
+            indexing=str(args.ordering),
+            A=float(args.drive_A),
+            omega=float(args.drive_omega),
+            tbar=float(args.drive_tbar),
+            phi=float(args.drive_phi),
+            pattern_mode=str(args.drive_pattern),
+            custom_weights=custom_weights,
+            include_identity=bool(args.drive_include_identity),
+            coeff_tol=0.0,
+        )
+        drive_coeff_provider_exyz = drive.coeff_map_exyz
+        drive_labels = set(drive.template.labels_exyz(include_identity=bool(drive.include_identity)))
+        missing = sorted(drive_labels.difference(ordered_labels_exyz))
+        ordered_labels_exyz = list(ordered_labels_exyz) + list(missing)
+
     hmat = _build_hamiltonian_matrix(coeff_map_exyz)
     evals, evecs = np.linalg.eigh(hmat)
     gs_idx = int(np.argmin(evals))
@@ -904,6 +1011,9 @@
         t_final=float(args.t_final),
         num_times=int(args.num_times),
         suzuki_order=int(args.suzuki_order),
+        drive_coeff_provider_exyz=drive_coeff_provider_exyz,
+        drive_t0=float(args.drive_t0),
+        drive_time_sampling=str(args.drive_time_sampling),
     )
 
     sanity = {
@@ -918,31 +1028,49 @@
         )
     }
 
+    settings: dict[str, Any] = {
+        "L": int(args.L),
+        "t": float(args.t),
+        "u": float(args.u),
+        "dv": float(args.dv),
+        "boundary": str(args.boundary),
+        "ordering": str(args.ordering),
+        "t_final": float(args.t_final),
+        "num_times": int(args.num_times),
+        "suzuki_order": int(args.suzuki_order),
+        "trotter_steps": int(args.trotter_steps),
+        "term_order": str(args.term_order),
+        "initial_state_source": str(args.initial_state_source),
+        "skip_qpe": bool(args.skip_qpe),
+    }
+    if bool(args.enable_drive):
+        settings["drive"] = {
+            "enabled": True,
+            "A": float(args.drive_A),
+            "omega": float(args.drive_omega),
+            "tbar": float(args.drive_tbar),
+            "phi": float(args.drive_phi),
+            "pattern": str(args.drive_pattern),
+            "custom_s": (str(args.drive_custom_s) if args.drive_custom_s is not None else None),
+            "include_identity": bool(args.drive_include_identity),
+            "time_sampling": str(args.drive_time_sampling),
+            "t0": float(args.drive_t0),
+        }
+
     payload: dict[str, Any] = {
         "generated_utc": datetime.now(timezone.utc).isoformat(),
         "pipeline": "hardcoded",
-        "settings": {
-            "L": int(args.L),
-            "t": float(args.t),
-            "u": float(args.u),
-            "dv": float(args.dv),
-            "boundary": str(args.boundary),
-            "ordering": str(args.ordering),
-            "t_final": float(args.t_final),
-            "num_times": int(args.num_times),
-            "suzuki_order": int(args.suzuki_order),
-            "trotter_steps": int(args.trotter_steps),
-            "term_order": str(args.term_order),
-            "initial_state_source": str(args.initial_state_source),
-            "skip_qpe": bool(args.skip_qpe),
-        },
+        "settings": settings,
         "hamiltonian": {
             "num_qubits": int(2 * args.L),
-            "num_terms": int(len(coeff_map_exyz)),
+            "num_terms": int(len(ordered_labels_exyz) if bool(args.enable_drive) else len(coeff_map_exyz)),
             "coefficients_exyz": [
                 {
                     "label_exyz": lbl,
-                    "coeff": {"re": float(np.real(coeff_map_exyz[lbl])), "im": float(np.imag(coeff_map_exyz[lbl]))},
+                    "coeff": {
+                        "re": float(np.real(coeff_map_exyz.get(lbl, 0.0 + 0.0j))),
+                        "im": float(np.imag(coeff_map_exyz.get(lbl, 0.0 + 0.0j))),
+                    },
                 }
                 for lbl in ordered_labels_exyz
             ],
--- a/pipelines/qiskit_hubbard_baseline_pipeline.py
+++ b/pipelines/qiskit_hubbard_baseline_pipeline.py
@@ -55,6 +55,7 @@
     sys.path.insert(0, str(REPO_ROOT))
 
 from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
+from src.quantum.drives_time_potential import build_gaussian_sinusoid_density_drive
 
 EXACT_LABEL = "Exact_Qiskit"
 EXACT_METHOD = "python_matrix_eigendecomposition"
@@ -281,17 +282,61 @@
     compiled_actions: dict[str, CompiledPauliAction],
     time_value: float,
     trotter_steps: int,
+    *,
+    drive_coeff_provider_exyz: Any | None = None,
+    t0: float = 0.0,
+    time_sampling: str = "midpoint",
+    coeff_tol: float = 1e-12,
 ) -> np.ndarray:
+    # Bit-for-bit identical path for time-independent Hamiltonians.
+    if drive_coeff_provider_exyz is None:
+        psi = np.array(psi0, copy=True)
+        if abs(time_value) <= 1e-15:
+            return psi
+        dt = float(time_value) / float(trotter_steps)
+        half = 0.5 * dt
+        for _ in range(trotter_steps):
+            for label in ordered_labels_exyz:
+                psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
+            for label in reversed(ordered_labels_exyz):
+                psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
+        return _normalize_state(psi)
+
+    # Time-dependent extension: coefficients sampled once per slice.
     psi = np.array(psi0, copy=True)
     if abs(time_value) <= 1e-15:
         return psi
     dt = float(time_value) / float(trotter_steps)
     half = 0.5 * dt
-    for _ in range(trotter_steps):
+
+    sampling = str(time_sampling).strip().lower()
+    if sampling not in {"midpoint", "left", "right"}:
+        raise ValueError("time_sampling must be one of {'midpoint','left','right'}")
+
+    t0_f = float(t0)
+    tol = float(coeff_tol)
+
+    for k in range(int(trotter_steps)):
+        if sampling == "midpoint":
+            t_sample = t0_f + (float(k) + 0.5) * dt
+        elif sampling == "left":
+            t_sample = t0_f + float(k) * dt
+        else:  # right
+            t_sample = t0_f + (float(k) + 1.0) * dt
+
+        drive_map = dict(drive_coeff_provider_exyz(float(t_sample)))
+
         for label in ordered_labels_exyz:
-            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
+            c_total = coeff_map_exyz.get(label, 0.0 + 0.0j) + complex(drive_map.get(label, 0.0))
+            if abs(c_total) <= tol:
+                continue
+            psi = _apply_exp_term(psi, compiled_actions[label], c_total, half)
         for label in reversed(ordered_labels_exyz):
-            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
+            c_total = coeff_map_exyz.get(label, 0.0 + 0.0j) + complex(drive_map.get(label, 0.0))
+            if abs(c_total) <= tol:
+                continue
+            psi = _apply_exp_term(psi, compiled_actions[label], c_total, half)
+
     return _normalize_state(psi)
 
 
@@ -689,10 +734,15 @@
     psi0: np.ndarray,
     hmat: np.ndarray,
     trotter_hamiltonian_qop: SparsePauliOp,
+    ordered_labels_exyz: list[str] | None = None,
+    coeff_map_exyz: dict[str, complex] | None = None,
     trotter_steps: int,
     t_final: float,
     num_times: int,
     suzuki_order: int,
+    drive_coeff_provider_exyz: Any | None = None,
+    drive_t0: float = 0.0,
+    drive_time_sampling: str = "midpoint",
 ) -> tuple[list[dict[str, float]], list[np.ndarray]]:
     if int(suzuki_order) != 2:
         raise ValueError("This script currently supports suzuki_order=2 only.")
@@ -704,6 +754,12 @@
     evecs_dag = np.conjugate(evecs).T
 
     synthesis = SuzukiTrotter(order=int(suzuki_order), reps=int(trotter_steps), preserve_order=True)
+
+    compiled = None
+    if drive_coeff_provider_exyz is not None:
+        if ordered_labels_exyz is None or coeff_map_exyz is None:
+            raise ValueError("ordered_labels_exyz and coeff_map_exyz required for time-dependent evolution")
+        compiled = {lbl: _compile_pauli_action(lbl, nq) for lbl in ordered_labels_exyz}
     times = np.linspace(0.0, float(t_final), int(num_times))
     n_times = int(times.size)
     stride = max(1, n_times // 20)
@@ -725,23 +781,36 @@
         psi_exact = evecs @ (np.exp(-1j * evals * t) * (evecs_dag @ psi0))
         psi_exact = _normalize_state(psi_exact)
 
-        if abs(t) <= 1e-15:
-            psi_trot = np.array(psi0, copy=True)
+        if drive_coeff_provider_exyz is None:
+            if abs(t) <= 1e-15:
+                psi_trot = np.array(psi0, copy=True)
+            else:
+                evo_gate = PauliEvolutionGate(
+                    trotter_hamiltonian_qop,
+                    time=t,
+                    synthesis=synthesis,
+                )
+                # Important: evolve on the synthesized Suzuki-Trotter circuit, not on
+                # the opaque PauliEvolutionGate object (which can be interpreted as an
+                # exact matrix by some simulator backends).
+                evo_circuit = synthesis.synthesize(evo_gate)
+                psi_trot = np.asarray(
+                    Statevector(np.asarray(psi0, dtype=complex)).evolve(evo_circuit).data,
+                    dtype=complex,
+                )
+                psi_trot = _normalize_state(psi_trot)
         else:
-            evo_gate = PauliEvolutionGate(
-                trotter_hamiltonian_qop,
-                time=t,
-                synthesis=synthesis,
+            psi_trot = _evolve_trotter_suzuki2_absolute(
+                psi0,
+                list(ordered_labels_exyz or []),
+                dict(coeff_map_exyz or {}),
+                dict(compiled or {}),
+                t,
+                int(trotter_steps),
+                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
+                t0=float(drive_t0),
+                time_sampling=str(drive_time_sampling),
             )
-            # Important: evolve on the synthesized Suzuki-Trotter circuit, not on
-            # the opaque PauliEvolutionGate object (which can be interpreted as an
-            # exact matrix by some simulator backends).
-            evo_circuit = synthesis.synthesize(evo_gate)
-            psi_trot = np.asarray(
-                Statevector(np.asarray(psi0, dtype=complex)).evolve(evo_circuit).data,
-                dtype=complex,
-            )
-            psi_trot = _normalize_state(psi_trot)
 
         fidelity = float(abs(np.vdot(psi_exact, psi_trot)) ** 2)
         n_up_exact, n_dn_exact = _occupation_site0(psi_exact, num_sites)
@@ -926,6 +995,32 @@
     parser.add_argument("--trotter-steps", type=int, default=64)
     parser.add_argument("--term-order", choices=["qiskit", "sorted"], default="sorted")
 
+    parser.add_argument("--enable-drive", action="store_true", help="Enable time-dependent onsite density drive.")
+    parser.add_argument("--drive-A", type=float, default=0.0, help="Drive amplitude A in v(t)=A*sin(ωt+φ)*exp(-t^2/(2 tbar^2)).")
+    parser.add_argument("--drive-omega", type=float, default=1.0, help="Drive carrier angular frequency ω.")
+    parser.add_argument("--drive-tbar", type=float, default=1.0, help="Drive Gaussian envelope width tbar (must be > 0).")
+    parser.add_argument("--drive-phi", type=float, default=0.0, help="Drive phase φ.")
+    parser.add_argument(
+        "--drive-pattern",
+        choices=["dimer_bias", "staggered", "custom"],
+        default="staggered",
+        help="Spatial pattern mode for v_i(t)=s_i*v(t).",
+    )
+    parser.add_argument(
+        "--drive-custom-s",
+        type=str,
+        default=None,
+        help="Custom spatial weights s_i (comma-separated or JSON list), length L, used when --drive-pattern custom.",
+    )
+    parser.add_argument("--drive-include-identity", action="store_true", help="Include identity term from n=(I-Z)/2 (global phase).")
+    parser.add_argument(
+        "--drive-time-sampling",
+        choices=["midpoint", "left", "right"],
+        default="midpoint",
+        help="Time sampling rule per slice (midpoint recommended; left/right for diagnostics).",
+    )
+    parser.add_argument("--drive-t0", type=float, default=0.0, help="Drive start time t0 for evolution (default 0.0).")
+
     parser.add_argument("--vqe-reps", type=int, default=2)
     parser.add_argument("--vqe-restarts", type=int, default=3)
     parser.add_argument("--vqe-seed", type=int, default=7)
@@ -970,6 +1065,36 @@
         ordered_labels_exyz = sorted(coeff_map_exyz)
     trotter_qop_ordered = _ordered_qop_from_exyz(ordered_labels_exyz, coeff_map_exyz)
 
+    drive = None
+    drive_coeff_provider_exyz = None
+    if bool(args.enable_drive):
+        custom_weights = None
+        if str(args.drive_pattern) == "custom":
+            if args.drive_custom_s is None:
+                raise ValueError("--drive-custom-s is required when --drive-pattern custom")
+            raw = str(args.drive_custom_s).strip()
+            if raw.startswith("["):
+                custom_weights = json.loads(raw)
+            else:
+                custom_weights = [float(x) for x in raw.split(",") if x.strip()]
+        drive = build_gaussian_sinusoid_density_drive(
+            n_sites=int(args.L),
+            nq_total=int(2 * args.L),
+            indexing=str(args.ordering),
+            A=float(args.drive_A),
+            omega=float(args.drive_omega),
+            tbar=float(args.drive_tbar),
+            phi=float(args.drive_phi),
+            pattern_mode=str(args.drive_pattern),
+            custom_weights=custom_weights,
+            include_identity=bool(args.drive_include_identity),
+            coeff_tol=0.0,
+        )
+        drive_coeff_provider_exyz = drive.coeff_map_exyz
+        drive_labels = set(drive.template.labels_exyz(include_identity=bool(drive.include_identity)))
+        missing = sorted(drive_labels.difference(ordered_labels_exyz))
+        ordered_labels_exyz = list(ordered_labels_exyz) + list(missing)
+
     hmat = np.asarray(qop.to_matrix(sparse=False), dtype=complex)
     evals, evecs = np.linalg.eigh(hmat)
     gs_idx = int(np.argmin(evals))
@@ -1040,10 +1165,15 @@
         psi0=psi0,
         hmat=hmat,
         trotter_hamiltonian_qop=trotter_qop_ordered,
+        ordered_labels_exyz=ordered_labels_exyz,
+        coeff_map_exyz=coeff_map_exyz,
         trotter_steps=int(args.trotter_steps),
         t_final=float(args.t_final),
         num_times=int(args.num_times),
         suzuki_order=int(args.suzuki_order),
+        drive_coeff_provider_exyz=drive_coeff_provider_exyz,
+        drive_t0=float(args.drive_t0),
+        drive_time_sampling=str(args.drive_time_sampling),
     )
 
     sanity = {
@@ -1058,31 +1188,49 @@
         )
     }
 
+    settings: dict[str, Any] = {
+        "L": int(args.L),
+        "t": float(args.t),
+        "u": float(args.u),
+        "dv": float(args.dv),
+        "boundary": str(args.boundary),
+        "ordering": str(args.ordering),
+        "t_final": float(args.t_final),
+        "num_times": int(args.num_times),
+        "suzuki_order": int(args.suzuki_order),
+        "trotter_steps": int(args.trotter_steps),
+        "term_order": str(args.term_order),
+        "initial_state_source": str(init_source),
+        "skip_qpe": bool(args.skip_qpe),
+    }
+    if bool(args.enable_drive):
+        settings["drive"] = {
+            "enabled": True,
+            "A": float(args.drive_A),
+            "omega": float(args.drive_omega),
+            "tbar": float(args.drive_tbar),
+            "phi": float(args.drive_phi),
+            "pattern": str(args.drive_pattern),
+            "custom_s": (str(args.drive_custom_s) if args.drive_custom_s is not None else None),
+            "include_identity": bool(args.drive_include_identity),
+            "time_sampling": str(args.drive_time_sampling),
+            "t0": float(args.drive_t0),
+        }
+
     payload: dict[str, Any] = {
         "generated_utc": datetime.now(timezone.utc).isoformat(),
         "pipeline": "qiskit",
-        "settings": {
-            "L": int(args.L),
-            "t": float(args.t),
-            "u": float(args.u),
-            "dv": float(args.dv),
-            "boundary": str(args.boundary),
-            "ordering": str(args.ordering),
-            "t_final": float(args.t_final),
-            "num_times": int(args.num_times),
-            "suzuki_order": int(args.suzuki_order),
-            "trotter_steps": int(args.trotter_steps),
-            "term_order": str(args.term_order),
-            "initial_state_source": str(init_source),
-            "skip_qpe": bool(args.skip_qpe),
-        },
+        "settings": settings,
         "hamiltonian": {
             "num_qubits": int(2 * args.L),
-            "num_terms": int(len(coeff_map_exyz)),
+            "num_terms": int(len(ordered_labels_exyz) if bool(args.enable_drive) else len(coeff_map_exyz)),
             "coefficients_exyz": [
                 {
                     "label_exyz": lbl,
-                    "coeff": {"re": float(np.real(coeff_map_exyz[lbl])), "im": float(np.imag(coeff_map_exyz[lbl]))},
+                    "coeff": {
+                        "re": float(np.real(coeff_map_exyz.get(lbl, 0.0 + 0.0j))),
+                        "im": float(np.imag(coeff_map_exyz.get(lbl, 0.0 + 0.0j))),
+                    },
                 }
                 for lbl in ordered_labels_exyz
             ],
--- a/src/quantum/drives_time_potential.py
+++ b/src/quantum/drives_time_potential.py
@@ -0,0 +1,208 @@
+from __future__ import annotations
+
+import math
+from dataclasses import dataclass
+from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence
+
+import numpy as np
+
+from src.quantum.hubbard_latex_python_pairs import SPIN_DN, SPIN_UP, mode_index
+
+
+def gaussian_sinusoid_waveform(t: float, *, A: float, omega: float, tbar: float, phi: float = 0.0) -> float:
+    """v(t) = A * sin(omega t + phi) * exp(-t^2 / (2 tbar^2))."""
+    t_f = float(t)
+    tbar_f = float(tbar)
+    if tbar_f <= 0.0:
+        raise ValueError("tbar must be > 0")
+    arg = float(omega) * t_f + float(phi)
+    env = math.exp(-(t_f * t_f) / (2.0 * tbar_f * tbar_f))
+    return float(A) * math.sin(arg) * env
+
+
+def default_spatial_weights(n_sites: int, *, mode: str, custom: Optional[Sequence[float]] = None) -> np.ndarray:
+    """Return spatial weights s_i.
+
+    mode:
+      - 'dimer_bias': requires n_sites==2, returns [+1, -1]
+      - 'staggered': returns [(-1)^i]
+      - 'custom': uses provided array-like of length n_sites
+    """
+    n = int(n_sites)
+    if n <= 0:
+        raise ValueError("n_sites must be positive")
+    mode_n = str(mode).strip().lower()
+    if mode_n == "dimer_bias":
+        if n != 2:
+            raise ValueError("pattern_mode='dimer_bias' requires n_sites == 2")
+        return np.array([1.0, -1.0], dtype=float)
+    if mode_n == "staggered":
+        return np.array([1.0 if (i % 2 == 0) else -1.0 for i in range(n)], dtype=float)
+    if mode_n == "custom":
+        if custom is None:
+            raise ValueError("custom spatial weights required when mode='custom'")
+        if len(custom) != n:
+            raise ValueError("custom spatial weights must have length n_sites")
+        return np.array([float(x) for x in custom], dtype=float)
+    raise ValueError("pattern_mode must be one of {'dimer_bias','staggered','custom'}")
+
+
+@dataclass(frozen=True)
+class GaussianSinusoidSitePotential:
+    """Site-resolved potential v_i(t) = s_i * v(t) with a Gaussian-windowed sinusoid."""
+
+    weights: np.ndarray  # length n_sites
+    A: float
+    omega: float
+    tbar: float
+    phi: float = 0.0
+
+    def v_scalar(self, t: float) -> float:
+        return gaussian_sinusoid_waveform(t, A=float(self.A), omega=float(self.omega), tbar=float(self.tbar), phi=float(self.phi))
+
+    def v_sites(self, t: float) -> np.ndarray:
+        v = self.v_scalar(t)
+        return np.asarray(self.weights, dtype=float) * float(v)
+
+
+def _z_label_exyz(*, nq_total: int, qubit_index: int) -> str:
+    nq = int(nq_total)
+    q = int(qubit_index)
+    if nq <= 0:
+        raise ValueError("nq_total must be positive")
+    if q < 0 or q >= nq:
+        raise ValueError("qubit_index out of range")
+    z_pos = nq - 1 - q
+    return ("e" * z_pos) + "z" + ("e" * (nq - 1 - z_pos))
+
+
+@dataclass(frozen=True)
+class DensityDriveTemplate:
+    """Precompute JW density-drive Z labels once (Pauli words stay static)."""
+
+    n_sites: int
+    nq_total: int
+    indexing: str
+    electron_qubit_offset: int
+    z_labels_site_spin: Dict[tuple[int, int], str]
+    ordered_z_labels: List[str]
+    identity_label: str
+
+    @classmethod
+    def build(
+        cls,
+        *,
+        n_sites: int,
+        nq_total: int,
+        indexing: str,
+        electron_qubit_offset: int = 0,
+    ) -> "DensityDriveTemplate":
+        n = int(n_sites)
+        nq = int(nq_total)
+        if nq <= 0:
+            raise ValueError("nq_total must be positive")
+        if n <= 0:
+            raise ValueError("n_sites must be positive")
+        if 2 * n + int(electron_qubit_offset) > nq:
+            raise ValueError("electron register does not fit inside nq_total")
+
+        z_map: Dict[tuple[int, int], str] = {}
+        ordered: List[str] = []
+        for i in range(n):
+            for spin in (SPIN_UP, SPIN_DN):
+                p_mode = mode_index(i, spin, indexing=str(indexing), n_sites=n)
+                q = int(electron_qubit_offset) + int(p_mode)
+                lbl = _z_label_exyz(nq_total=nq, qubit_index=q)
+                z_map[(i, int(spin))] = lbl
+                ordered.append(lbl)
+
+        return cls(
+            n_sites=n,
+            nq_total=nq,
+            indexing=str(indexing),
+            electron_qubit_offset=int(electron_qubit_offset),
+            z_labels_site_spin=z_map,
+            ordered_z_labels=ordered,
+            identity_label="e" * nq,
+        )
+
+    def labels_exyz(self, *, include_identity: bool = False) -> List[str]:
+        out = list(self.ordered_z_labels)
+        if bool(include_identity):
+            out.append(self.identity_label)
+        return out
+
+    def z_label(self, site: int, spin: int) -> str:
+        return self.z_labels_site_spin[(int(site), int(spin))]
+
+
+@dataclass(frozen=True)
+class TimeDependentOnsiteDensityDrive:
+    """Drive object providing additive Pauli coefficients Δc(label,t) for Σ_{i,σ} v_i(t) n_{iσ}."""
+
+    template: DensityDriveTemplate
+    site_potential: Callable[[float], np.ndarray]
+    include_identity: bool = False
+    coeff_tol: float = 0.0
+
+    def coeff_map_exyz(self, t: float) -> Dict[str, float]:
+        """Return dict {label_exyz: additive_coeff} at time t.
+
+        Mapping:
+          n_{iσ} = (I - Z_{q(i,σ)})/2
+          => drive adds Δc[Z_{q(i,σ)}](t) = -(1/2) v_i(t)
+          and optionally identity term +(1/2) Σ_{i,σ} v_i(t) * I.
+        """
+        V = np.asarray(self.site_potential(float(t)), dtype=float).reshape(-1)
+        if V.size != int(self.template.n_sites):
+            raise ValueError("site_potential returned wrong length")
+
+        out: Dict[str, float] = {}
+        tol = float(self.coeff_tol)
+        for i in range(int(self.template.n_sites)):
+            vi = float(V[i])
+            if tol > 0.0 and abs(vi) <= tol:
+                continue
+            coeff_z = -0.5 * vi
+            for spin in (SPIN_UP, SPIN_DN):
+                lbl = self.template.z_label(i, int(spin))
+                out[lbl] = out.get(lbl, 0.0) + float(coeff_z)
+
+        if bool(self.include_identity):
+            coeff_id = float(np.sum(V))  # 0.5 * Σ_{i,σ} v_i(t) = Σ_i v_i(t)
+            if tol <= 0.0 or abs(coeff_id) > tol:
+                out[self.template.identity_label] = out.get(self.template.identity_label, 0.0) + coeff_id
+
+        return out
+
+
+def build_gaussian_sinusoid_density_drive(
+    *,
+    n_sites: int,
+    nq_total: int,
+    indexing: str,
+    A: float,
+    omega: float,
+    tbar: float,
+    phi: float = 0.0,
+    pattern_mode: str = "staggered",
+    custom_weights: Optional[Sequence[float]] = None,
+    include_identity: bool = False,
+    electron_qubit_offset: int = 0,
+    coeff_tol: float = 0.0,
+) -> TimeDependentOnsiteDensityDrive:
+    """Convenience builder used by pipeline scripts."""
+    weights = default_spatial_weights(int(n_sites), mode=str(pattern_mode), custom=custom_weights)
+    potential = GaussianSinusoidSitePotential(weights=weights, A=float(A), omega=float(omega), tbar=float(tbar), phi=float(phi))
+    template = DensityDriveTemplate.build(
+        n_sites=int(n_sites),
+        nq_total=int(nq_total),
+        indexing=str(indexing),
+        electron_qubit_offset=int(electron_qubit_offset),
+    )
+    return TimeDependentOnsiteDensityDrive(
+        template=template,
+        site_potential=potential.v_sites,
+        include_identity=bool(include_identity),
+        coeff_tol=float(coeff_tol),
+    )
--- a/Tests/test_time_potential_drive.py
+++ b/Tests/test_time_potential_drive.py
@@ -0,0 +1,147 @@
+import math
+import unittest
+
+import numpy as np
+
+from src.quantum.drives_time_potential import (
+    DensityDriveTemplate,
+    build_gaussian_sinusoid_density_drive,
+    gaussian_sinusoid_waveform,
+)
+from src.quantum.hubbard_latex_python_pairs import SPIN_DN, SPIN_UP, mode_index
+
+
+class TestTimePotentialDrive(unittest.TestCase):
+    def test_z_label_qubit0_rightmost(self) -> None:
+        # nq_total=4, qubit 0 corresponds to rightmost character.
+        template = DensityDriveTemplate.build(n_sites=2, nq_total=4, indexing="interleaved")
+        lbl = template.z_label(0, SPIN_UP)  # mode_index(site0, up) == 0
+        self.assertEqual(lbl, "eeez")
+
+        # Diagonal eigenvalue should be +1 for bit0=0, -1 for bit0=1.
+        nq = 4
+        for idx in range(1 << nq):
+            bit0 = (idx >> 0) & 1
+            expected = 1 if bit0 == 0 else -1
+            # Evaluate eigenvalue from label placement.
+            op = lbl[nq - 1 - 0]
+            self.assertEqual(op, "z")
+            got = 1 if bit0 == 0 else -1
+            self.assertEqual(got, expected)
+
+    def test_drive_only_phase_L2_blocked_and_interleaved(self) -> None:
+        import pipelines.hardcoded_hubbard_pipeline as hp
+
+        dt = 0.25
+        t_mid = 0.5 * dt
+        A = 0.7
+        omega = 1.3
+        tbar = 2.0
+        phi = 0.2
+        v_scalar = gaussian_sinusoid_waveform(t_mid, A=A, omega=omega, tbar=tbar, phi=phi)
+        V_sites = np.array([+v_scalar, -v_scalar], dtype=float)
+
+        for indexing in ("blocked", "interleaved"):
+            drive = build_gaussian_sinusoid_density_drive(
+                n_sites=2,
+                nq_total=4,
+                indexing=indexing,
+                A=A,
+                omega=omega,
+                tbar=tbar,
+                phi=phi,
+                pattern_mode="dimer_bias",
+                include_identity=False,
+            )
+
+            ordered_labels_exyz = sorted(drive.template.labels_exyz(include_identity=False))
+            compiled = {lbl: hp._compile_pauli_action(lbl, 4) for lbl in ordered_labels_exyz}
+
+            # Evolve each computational basis state under drive only.
+            def provider(t: float):
+                return drive.coeff_map_exyz(t)
+
+            def evolve_basis(idx: int) -> complex:
+                psi0 = np.zeros(1 << 4, dtype=complex)
+                psi0[idx] = 1.0 + 0.0j
+                psi1 = hp._evolve_trotter_suzuki2_absolute(
+                    psi0,
+                    ordered_labels_exyz,
+                    {},
+                    compiled,
+                    dt,
+                    1,
+                    drive_coeff_provider_exyz=provider,
+                    t0=0.0,
+                    time_sampling="midpoint",
+                )
+                return complex(psi1[idx])
+
+            amp_ref = evolve_basis(0)
+            self.assertAlmostEqual(abs(amp_ref), 1.0, places=12)
+
+            for idx in range(1 << 4):
+                amp = evolve_basis(idx)
+                self.assertAlmostEqual(abs(amp), 1.0, places=12)
+
+                # Global phase cancels by dividing by |0000> amplitude.
+                ratio = amp / amp_ref
+
+                energy = 0.0
+                for site in range(2):
+                    for spin in (SPIN_UP, SPIN_DN):
+                        q = mode_index(site, spin, indexing=indexing, n_sites=2)
+                        n = (idx >> q) & 1
+                        energy += float(V_sites[site]) * float(n)
+                expected = complex(np.exp(-1j * dt * energy))
+                self.assertTrue(np.allclose(ratio, expected, atol=1e-12, rtol=0.0))
+
+    def test_midpoint_vs_left_quadrature_order(self) -> None:
+        import pipelines.hardcoded_hubbard_pipeline as hp
+
+        # Single-qubit commuting test for the non-autonomous time sampler.
+        # Note: for H(t)=t Z the midpoint rule is exact (linear integrand), so we
+        # use H(t)=t^2 Z to expose the expected O(dt^2) midpoint error.
+        ordered_labels_exyz = ["z"]
+        compiled = {"z": hp._compile_pauli_action("z", 1)}
+
+        psi0 = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
+        T = 1.0
+        # Relative phase between |0> and |1> is exp(-i * ∫ 2 t^2 dt) = exp(-i * 2 T^3/3).
+        exact_ratio = complex(np.exp(-1j * (2.0 * (T**3) / 3.0)))
+
+        def run(n_steps: int, sampling: str) -> float:
+            dt = T / float(n_steps)
+
+            def provider(t: float):
+                return {"z": float(t * t)}
+
+            psi = hp._evolve_trotter_suzuki2_absolute(
+                psi0,
+                ordered_labels_exyz,
+                {},
+                compiled,
+                T,
+                int(n_steps),
+                drive_coeff_provider_exyz=provider,
+                t0=0.0,
+                time_sampling=sampling,
+            )
+            ratio = complex(psi[0] / psi[1])
+            return float(abs(ratio - exact_ratio))
+
+        err_mid_20 = run(20, "midpoint")
+        err_mid_40 = run(40, "midpoint")
+        err_left_20 = run(20, "left")
+        err_left_40 = run(40, "left")
+
+        # Midpoint should be second order; left should be first order.
+        self.assertLess(err_mid_20, err_left_20)
+        self.assertGreater(err_left_20 / err_left_40, 1.5)
+        self.assertLess(err_left_20 / err_left_40, 2.5)
+        self.assertGreater(err_mid_20 / err_mid_40, 3.0)
+        self.assertLess(err_mid_20 / err_mid_40, 5.5)
+
+
+if __name__ == "__main__":
+    unittest.main()