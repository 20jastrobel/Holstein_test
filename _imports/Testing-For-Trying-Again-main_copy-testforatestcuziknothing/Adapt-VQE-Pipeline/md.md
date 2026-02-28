# ADAPT-VQE: Polaron-Adapted Operator Pool (PAOP) — Implementation + Test Plan

## Goal

Fix the observed lack of ground-state energy convergence when the electron–phonon coupling `g != 0` by adding a **polaron-native operator pool** to the existing ADAPT-VQE pipeline (and optionally a single Lang–Firsov “dressing” layer to the fixed HVA ansatz).

The repo already contains:
- Hubbard–Holstein Hamiltonian in the **qubit basis**
- Jordan–Wigner mapping for fermions
- Binary boson encoding
- Periodic and open boundary conditions
- VQE, HVA, and ADAPT-VQE drivers

This plan **only** adds/changes code inside the sandbox folder: **`adapt VQE pipeline/`**.


## Hard constraints (do not violate)

- Work only inside the folder: **`adapt VQE pipeline/`**
- You may **copy** code from elsewhere in the repo into the sandbox, then modify the copies.
- Do not modify the original (non-sandbox) code.
- Keep all new tests + scripts in the sandbox.


## Deliverables checklist

Inside `adapt VQE pipeline/`:

1) `operator_pools/polaron_paop.py`
- Generates the PAOP operator pool in the qubit basis using existing operator/mapping utilities.

2) `operator_pools/__init__.py`
- Exposes `make_pool(name="paop", ...)`.

3) `tests/test_paop_operators.py`
- Unit tests: Hermiticity, commutation sanity, term counts, numerical spot-checks.

4) `benchmarks/paop_convergence_sweep.py`
- Overnight benchmark script: compares baseline pools vs PAOP on a small ED-verifiable suite.

5) Minimal integration patch
- A small change in the copied ADAPT driver code to allow `pool_name="paop"`.


---

# 0) Sandbox setup

## 0.1 Create sandbox folder

Create (or recreate) the sandbox folder:

```bash
mkdir -p "adapt VQE pipeline"
```

## 0.2 Copy the minimal working ADAPT-VQE stack into the sandbox

Inside the repo, identify the modules used for:
- building the HH qubit Hamiltonian
- building ansätze (HVA / ADAPT)
- representing operators (Pauli sums / QubitOperator / SparsePauliOp / etc.)
- expectation measurement and/or statevector simulation

Copy those modules into `adapt VQE pipeline/` preserving their relative import structure, then adjust imports **inside the sandbox copies only** so the sandbox can run standalone.

Practical search commands (run at repo root):

```bash
rg -n "ADAPT" .
rg -n "operator pool|operator_pool|pool" .
rg -n "Hubbard.*Holstein|Holstein" .
rg -n "binary.*boson|boson.*encoding" .
rg -n "SparsePauliOp|QubitOperator|PauliSum" .
```


---

# 1) Identify and reuse existing primitives

Before writing new pool code, locate the existing helper(s) that already build the HH Hamiltonian terms in the qubit basis:

- **Fermion number operators** `n_{i,σ}` and `n_i`
- **Fermion hopping operators** on bonds `(i,j)` for each spin
- **Boson ladder operators** `a_i`, `a_i^\dagger` in binary encoding
- The existing Holstein coupling term uses `(a_i + a_i^\dagger)`; reuse that machinery.

Record the following from the existing code (in comments at the top of `polaron_paop.py`):
- the operator class used (`QubitOperator`, `SparsePauliOp`, etc.)
- how operator multiplication is done
- how operators are “embedded” into the full register (if needed)
- whether coefficients can be complex (they probably can)


---

# 2) PAOP design (what to implement)

## 2.1 Use Hermitian generators (recommended)

Most VQE/ADAPT pipelines implement unitaries as:
\[
U(\theta) = \exp(-i \theta \, G)
\]
where `G` is **Hermitian**.

Implement each pool element as a **Hermitian** qubit operator `G` (sum of Pauli strings), then use the existing exp(-i θ G) machinery.

### Boson quadratures (per phonon site/mode `j`)

Use the ladder operators already present:

- `X_j = a_j + a_j^\dagger`  (Hermitian)
- `P_j = i (a_j^\dagger - a_j)` (Hermitian)

Use exactly these (or the repo’s existing sign convention, but be consistent across pool + Hamiltonian).

**Do not** implement the pool in terms of non-Hermitian `a` and `a^\dagger` directly. Always use `X` and `P`.

## 2.2 Minimal PAOP pool (implement this first)

This is the smallest set that should directly address g≠0 nonconvergence via Lang–Firsov dressing.

Let:
- `n_i = n_{i↑} + n_{i↓}`
- `nbar` = average density used in your HH model (often `1` at half-filling; use whatever the Hamiltonian uses)
- `D_i = n_{i↑} n_{i↓}` (doublon projector)

### (A) Local conditional displacement (“dressing”) operators

For each lattice site `i`:
- `G_disp_i = (n_i - nbar) * P_i`

Add all `i`.

### (B) Optional (but low-cost) doublon dressing

For each site `i`:
- `G_dbl_i = D_i * P_i`

Add all `i`.

### (C) Dressed hopping (nearest-neighbor cloud drag)

For each bond `<i,j>` in the chosen boundary condition:
- Let `K_ij = Σσ (c†_{iσ} c_{jσ} + c†_{jσ} c_{iσ})` mapped to qubits (Hermitian)
- `G_hopdrag_ij = K_ij * (P_i - P_j)`

Add all bonds.

This term is the lowest-order proxy for phonon-dressed hopping.

## 2.3 Extended PAOP (second pass)

Add a tunable “cloud radius” `R` (default 0 or 1).

For each electron site `i`, for each phonon site `j` with `dist(i,j) <= R`:
- `G_disp_{i->j} = (n_i - nbar) * P_j`

This produces an **extended cloud**. Keep `R=1` for most tests.

Also add the `X`-quadrature analog only if needed:
- `G_dispX_{i->j} = (n_i - nbar) * X_j`

## 2.4 Pool variants to expose to the driver

Implement the pool generator with named presets:

- `paop_min`: (A) only
- `paop_std`: (A) + (C)
- `paop_full`: (A) + (B) + (C) + extended radius `R=1`

The driver should accept:

- `pool_name`: `"baseline"` (existing) vs `"paop_min"|"paop_std"|"paop_full"`
- `paop_R`: integer (0/1/2)
- `paop_split_paulis`: bool (see below)
- `paop_prune_eps`: float (default 0; if >0, prune tiny Pauli coefficients after products)


## 2.5 Optional: “Pauli-split” mode (only if baseline ADAPT expects Pauli strings)

If your current ADAPT implementation expects each pool element to be a **single Pauli string**, add a mode:

- If `paop_split_paulis=True`, then for each composite generator `G` (a Pauli sum),
  split it into separate pool elements `{P_k}` (each a single Pauli string term of `G`).

This can explode pool size; keep it OFF by default.

If you have both options, benchmark both:
- composite operators (physics-motivated, small pool)
- Pauli-split (hardware-efficient, large pool)


---

# 3) Implementation details (how to build PAOP from existing code)

## 3.1 Build and cache building blocks

Implement a cache keyed by `(site/mode index)` and `(bond index)`:

- `N_i_up`, `N_i_dn`, `N_i = N_i_up + N_i_dn`
- `D_i = N_i_up * N_i_dn`
- `X_i`, `P_i`
- `K_ij` (spin-summed hopping operator)

Caching is mandatory to avoid re-decomposing the same boson operators many times.

## 3.2 Operator products

Since fermionic and bosonic operators act on disjoint qubits, their product is just the algebraic product in the operator class.

However, you must handle:
- Pauli multiplication phases (±1, ±i)
- coefficient growth / floating noise

Add a helper:
- `multiply_ops(A, B) -> AB` with optional `prune_eps`

And a helper:
- `is_hermitian(G, atol=1e-10)` using either:
  - built-in `.adjoint()` / `.dagger()` if available, or
  - dense-matrix conversion for tiny test systems only

Every pool operator must pass `is_hermitian`.

## 3.3 Normalization / scaling (avoid optimizer pathologies)

Add optional normalization modes for each pool element `G`:
- `"none"`: no rescaling
- `"fro"`: scale by `1/||G||_F` computed from Pauli coefficients
- `"maxcoeff"`: scale by `1/max_k |c_k|`

Expose as a flag, default `"none"`.

Then benchmark whether normalization improves convergence at g≠0.


---

# 4) Tests

## 4.1 Unit tests: operator sanity

Create `tests/test_paop_operators.py` with tests that run fast.

### Test A: Hermiticity

For a tiny instance (e.g., 2 sites, 1 boson qubit per site), build each pool variant and assert every operator is Hermitian.

### Test B: commutation sanity

Pick one fermion-only operator `F` and one boson-only operator `B` (on disjoint qubits) and confirm:
- `[F, B] = 0` (numerically as a matrix for the tiny instance)

This ensures your embedding/multiplication is correct.

### Test C: term count guardrails

Assert that term counts stay below a fixed threshold for the tiny instance:
- if `paop_split_paulis=False`, composite elements should be manageable
- if it explodes, add pruning or reduce boson qubits in tests

## 4.2 Integration tests: “g turns on, baseline fails, PAOP succeeds”

Implement a benchmark harness (can be a test if runtime is small, otherwise a benchmark script).

For each test instance, compute a reference ground energy `E_ref` via:
- existing exact diagonalization module in the repo, OR
- dense diagonalization of the qubit Hamiltonian for very small instances

Then run:
1) baseline HVA (existing)
2) baseline ADAPT (existing pool)
3) PAOP-ADAPT (`paop_min`, `paop_std`, `paop_full`)

and compare final energy error `|E - E_ref|`.

### Suggested minimal instance suite (ED-friendly)

Use the smallest sizes that still show the failure:

- Lattice: 2-site chain
- Fermions: spinful (same as your HH code)
- Bosons: 1 mode per site
- Boson cutoff: start with 2 qubits per mode (4 levels), increase if needed

Parameter sets:

- Control (should already work):
  - `t=1, U=4, ω=1, g=0`

- Problem case:
  - `t=1, U=4, ω=1, g=0.5`
  - `t=1, U=4, ω=1, g=1.0`

Optional stress:
  - smaller `ω` increases phonon dressing difficulty:
  - `t=1, U=4, ω=0.5, g=1.0`

For each case run both OBC and PBC if possible.

### Metrics to log per run

Write a JSON line per run (append-only):

- instance definition (L, BC, cutoff, t,U,ω,g, filling)
- ansatz type (HVA, ADAPT baseline, ADAPT paop_*)
- optimizer name + settings
- final energy, best energy, iterations
- number of parameters
- for ADAPT: the selected operator IDs in order and their gradient magnitudes


---

# 5) “HVA repair” experiment (optional but quick)

If HVA fails only when g≠0, add one experiment that does not require ADAPT:

## Add a single Lang–Firsov dressing layer before HVA

Define an ansatz:
\[
U(\vec\theta) = \Big(\prod_i e^{-i \phi_i (n_i - nbar) P_i}\Big)\, U_{\mathrm{HVA}}(\vec\theta_{\mathrm{HVA}})
\]

Initialize:
- `phi_i = g/ω` (or the repo’s sign convention)

Then optimize all parameters (or freeze `phi_i` for the first optimizer stage).

This is a fast “does the pool idea fix the pathology?” test:
- if this alone restores convergence, PAOP should also fix ADAPT behavior.


---

# 6) Overnight benchmark script (the one to run while you sleep)

Create: `benchmarks/paop_convergence_sweep.py`

It should:
1) Loop over the instance suite above
2) For each instance run:
   - HVA baseline
   - ADAPT baseline
   - ADAPT paop_min / paop_std / paop_full
   - (optional) HVA + LF dressing layer
3) Save:
   - `results/paop_sweep_<timestamp>.jsonl`
   - `results/paop_sweep_<timestamp>_summary.csv`

Keep runtime bounded by:
- hard cap on optimizer iterations
- ADAPT max operators (e.g., 30–80 depending on size)
- early stopping on energy error threshold

---

# 7) Expected outcomes / acceptance criteria

A run is a success if, for the small ED-verifiable instances:

1) For `g=0`:
- PAOP does not break anything (matches baseline).

2) For `g>0`:
- Baseline HVA/ADAPT reproduces the currently observed convergence failure (or poor accuracy).
- PAOP variants significantly improve convergence:
  - `paop_min` should already help materially
  - `paop_std` should further help
  - `paop_full` should be best (but may cost more)

3) Operator selection sanity (for ADAPT+PAOP):
- The earliest selected operators should often include `(n_i - nbar) P_i` terms.

If these hold, the operator pool hypothesis is supported.


---

# Notes (if the above still fails)

If PAOP still fails at larger `g`:

- Check boson cutoff adequacy:
  - compute ⟨n_ph⟩ per mode and the probability of the highest encoded level
  - if the top level is significantly occupied, increase boson qubits/cutoff

- Try normalization modes for pool elements.

- Try seeding the initial state with the LF dressing layer as in Section 5, then run ADAPT from that dressed reference.

All of the above must remain inside the sandbox.
