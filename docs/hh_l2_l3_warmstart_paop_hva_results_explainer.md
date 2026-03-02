# L2 and L3 HH Runs: Intuition First, Then Exact Math

## Read This First (No Ambiguity)

If you only read one section, read this.

### What pipeline was used for the main L=3 result?

For the **warm-start B/C workflow**, the sequence was:

1. Run HH-HVA VQE warm-start (`hh_hva_ptw`, `reps=3`).
2. Use that warm-start state as ADAPT reference.
3. Run ADAPT with **PAOP-LF pool only** (`paop_lf_std`), not HVA pool.

So this was **HVA warm-start + ADAPT(PAOP-LF)**.

### Did ADAPT help beyond warm-start alone?

Yes, strongly.

- Warm-start only error:
  - $\Delta E_{\text{warm}} = 1.9686554792588795\times10^{-2}$
  - $\epsilon_{\text{rel,warm}} = 8.037273830893803\times10^{-2}$

- Warm-start B full (after ADAPT with `paop_lf_std`):
  - $\Delta E_{B} = 6.507533402310917\times10^{-3}$
  - $\epsilon_{\text{rel},B} = 2.6567791301781902\times10^{-2}$

- Warm-start C (heavier rung):
  - $\Delta E_{C} = 4.393299375013565\times10^{-3}$
  - $\epsilon_{\text{rel},C} = 1.793617546091431\times10^{-2}$

Improvement from ADAPT over warm-start:

$$
\frac{\Delta E_{\text{warm}}}{\Delta E_{B}} \approx 3.025
\quad\Rightarrow\quad
\text{about }66.94\%\text{ error reduction},
$$

$$
\frac{\Delta E_{\text{warm}}}{\Delta E_{C}} \approx 4.481
\quad\Rightarrow\quad
\text{about }77.68\%\text{ error reduction}.
$$

### What about the separate meta-pool experiment?

That is a different experiment family:

- Pool A: `UCCSD + PAOP`
- Pool B: `UCCSD + PAOP + HVA`

Do not mix that with warm-start B/C when interpreting results.

---

## 0) What we actually did (plain English)
You were right to ask for this clearly.

There are **two different L=3 experiment families** in this repo, and mixing them causes confusion:

1. **Warm-start B/C chain (main workflow used for the good L3 ADAPT seed):**
   - First run HH-HVA VQE to create a warm-start state.
   - Then run ADAPT from that warm-start state.
   - ADAPT pool in this workflow is **PAOP LF only** (`paop_lf_std`), not HVA.

2. **Separate meta-pool trend experiment:**
   - ADAPT with pool A = `UCCSD + PAOP`
   - ADAPT with pool B = `UCCSD + PAOP + HVA`
   - This is a different experiment, not the same as warm-start B/C.

For L=2, the reference artifact here is a strong **plain HH-HVA VQE** run.

---

## 1) Mathematical setup used in these runs

### 1.1 Hamiltonian form
The HH model used is of the standard form

$$
H = H_t + H_U + H_{ph} + H_{e-ph},
$$

with representative terms

$$
H_t = -t \sum_{\langle i,j\rangle,\sigma}
\left(c_{i\sigma}^\dagger c_{j\sigma} + c_{j\sigma}^\dagger c_{i\sigma}\right),
$$

$$
H_U = U \sum_i n_{i\uparrow}n_{i\downarrow},
\qquad
H_{ph} = \omega_0 \sum_i b_i^\dagger b_i,
$$

$$
H_{e-ph} = g \sum_i \tilde n_i (b_i + b_i^\dagger),
\qquad
\tilde n_i := n_i - \bar n.
$$

For driven runs, the propagation Hamiltonian is time-dependent:

$$
H(t) = H + H_{drive}(t).
$$

### 1.2 What energy we compare to
All reported errors are against **sector-filtered exact energy**:

$$
E_{\mathrm{exact,sector}} =
\min_{\psi \in \mathcal H_{(N_\uparrow,N_\downarrow)}}
\langle \psi|H|\psi\rangle.
$$

Error metrics:

$$
\Delta E := |E_{\mathrm{best}} - E_{\mathrm{exact,sector}}|,
\qquad
\epsilon_{\mathrm{rel}} := \frac{\Delta E}{|E_{\mathrm{exact,sector}}|}.
$$

### 1.3 ADAPT selection rule and PAOP-LF generators
In ADAPT, at step $k$, candidates are ranked by commutator-gradient signal:

$$
g_m = i\langle \psi_k |[H,G_m]|\psi_k\rangle.
$$

For the LF-extended PAOP pool, the key operators are (schematically):

$$
G^{\mathrm{disp}}_i \sim \tilde n_i p_i,
\qquad
G^{\mathrm{hopdrag}}_{ij} \sim K_{ij}(p_i-p_j),
\qquad
G^{\mathrm{curdrag}}_{ij} \sim J_{ij}(p_i-p_j),
$$

with

$$
K_{ij}=\sum_\sigma (c^\dagger_{i\sigma}c_{j\sigma}+c^\dagger_{j\sigma}c_{i\sigma}),
$$

$$
J_{ij}=\sum_\sigma i(c^\dagger_{i\sigma}c_{j\sigma}-c^\dagger_{j\sigma}c_{i\sigma}).
$$

This is the mathematical reason the warm-start + PAOP-LF ADAPT chain can correct HH-specific residual structure better than plain warm-start alone.

---

## 2) L=2 run (what happened and why it worked)

### 2.1 Intuition
At L=2 the Hilbert space is small enough that a sufficiently heavy HH-HVA VQE can already get very close to the sector exact energy. In practice, optimizer effort matters most here.

### 2.2 Exact run details
Artifact:

- `artifacts/useful/L2/H_L2_hh_termwise_regular_lbfgs_t1.0_U2.0_g1_nph1.json`

Settings/results:

- Ansatz: `hh_hva_tw` (termwise HH-HVA)
- Optimizer: `L-BFGS-B`
- HVA reps: `6`
- Parameters: `108`
- Work: `nfev=15042`, `nit=128`

Energy numbers:

- $E_{\mathrm{exact,sector}} = -0.38955310329705545$
- $E_{\mathrm{best}} = -0.38955250044542605$
- $\Delta E = 6.028516293943298\times 10^{-7}$
- $\epsilon_{\mathrm{rel}} = 1.5475467254450868\times 10^{-6}$

Interpretation: for this L=2 setup, convergence is essentially exact for benchmarking purposes.

---

## 3) L=3 runs (split by experiment family)

## 3.1 L=3 warm-start B/C chain (main workflow)

### 3.1.1 Intuition
L=3 is harder because the search space is larger and local minima/plateaus become much more likely. Warm-start gives ADAPT a much better initial state than a bare reference, and PAOP-LF operators then refine along physically targeted electron-phonon directions.

### 3.1.2 Exact sequence executed
Canonical artifact for rebuilt B state:

- `artifacts/useful/L3/warmstart_states/fix1_warm_start_B_full_state.json`

Pipeline sequence:

1. **Warm-start VQE (HH-HVA physical-termwise)**
2. **ADAPT refinement from that state using `paop_lf_std` pool**

Warm-start stage metadata:

- Ansatz style: `hh_hva_ptw`
- Reps: `3`
- Restarts: `5`
- Maxiter: `4000`
- Warm work: `warm_nfev=4000`, `warm_nit=0`
- Warm energy: $E_{\mathrm{warm}}=0.26462725492050154$

Warm seed error:

- $E_{\mathrm{exact,sector}}=0.2449407001279198$
- $\Delta E_{\mathrm{warm}}=1.9686554792588795\times10^{-2}$
- $\epsilon_{\mathrm{rel,warm}}=8.037273830893803\times10^{-2}$

ADAPT stage metadata (B rebuild):

- Pool type: `paop_lf_std`
- Pool size: `7`
- ADAPT depth: `42`
- Parameters: `42`
- ADAPT runtime: `1201.4293 s`
- Stop reason: `wallclock_cap`

Family usage in selected generators:

- `curdrag=28`
- `hopdrag=11`
- `disp=3`

Final B-state error:

- $E_{\mathrm{best}}=0.2514482335302307$
- $\Delta E=6.507533402310917\times10^{-3}$
- $\epsilon_{\mathrm{rel}}=2.6567791301781902\times10^{-2}$

### 3.1.3 How much ADAPT improved the warm seed (quantitatively)
For B rebuild:

$$
\text{improvement factor} = \frac{\Delta E_{\mathrm{warm}}}{\Delta E_{\mathrm{B}}}
= \frac{1.968655\times10^{-2}}{6.507533\times10^{-3}} \approx 3.025,
$$

$$
\text{error reduction} = 1 - \frac{\Delta E_{\mathrm{B}}}{\Delta E_{\mathrm{warm}}}
\approx 66.94\%.
$$

So ADAPT+PAOP-LF reduced warm-seed error by about two-thirds in this run.

### 3.1.4 C rung (same family, heavier budget)
Artifact:

- `artifacts/useful/L3/l3_hh_accessibility_fixes_under8pct.json`

C result summary:

- Warm start: `reps=3`, `restarts=6`, `maxiter=6000`
- ADAPT depth: `38`
- Params: `38`
- Work: `nfev_total=6227`, `nit_total=0`
- Runtime: `3984.1409 s`
- Stop: `wallclock_cap`
- $\Delta E = 4.393299375013565\times10^{-3}$
- $\epsilon_{\mathrm{rel}} = 1.793617546091431\times10^{-2}$

Relative to warm seed:

$$
\frac{\Delta E_{\mathrm{warm}}}{\Delta E_{\mathrm{C}}} \approx 4.481,
\qquad
\text{error reduction} \approx 77.68\%.
$$

So C improves further over B in accuracy, at higher wall time.

## 3.2 L=3 separate meta-pool experiment (different run family)

Artifact:

- `artifacts/useful/L3/l3_uccsd_paop_hva_trend_full_20260302T000521.json`

This is **not** the same as warm-start B/C above.

Pool definitions here:

- Pool A: `UCCSD + PAOP` (dedup size `15`)
- Pool B: `UCCSD + PAOP + HVA` (dedup size `28`)
- Raw component counts: `uccsd=8`, `paop=7`, `hva=13`

Best run in this file:

- Run: `A_medium` (`max_depth=20`, `maxiter=1200`)
- Depth: `20`
- Params: `20`
- Work: `nfev_total=12640`, `nit_total=0`
- Runtime: `623.9736 s`
- Stop: `max_depth`
- $\Delta E = 2.622402274776725\times10^{-4}$
- $\epsilon_{\mathrm{rel}} = 1.0706274103924667\times10^{-3}$

Interpretation: this particular meta-pool run reached the smallest L=3 static error among the artifacts referenced here.

---

## 4) Why the driven PDF can look contradictory

In the L3 driven run file:

- `artifacts/useful/L3/drive_from_fix1_warm_start_B_full.json`

there are multiple branches in one output:

- `exact_gs_filtered`
- `exact_paop`, `trotter_paop`
- `exact_hva`, `trotter_hva`

Branch definition in settings confirms this split.

### 4.1 The specific confusion
Inside the same driven JSON:

- Internal hardcoded VQE error is large:
  - $E_{\mathrm{vqe,internal}}=0.4226190970102364$
  - $E_{\mathrm{exact,sector}}=0.24494070012791275$
  - $\Delta E_{\mathrm{internal}}=1.7767839688232367\times10^{-1}$
  - $\epsilon_{\mathrm{rel,internal}}\approx 0.7254$

- But the propagated PAOP-seeded trajectory starts from the warm-start-B ADAPT state energy around $0.25144823353023055$, which matches the lower-error ADAPT seed.

This is not a bug; it is two different initial-state branches in one report.

---

## 5) Is generator/pool usage already in the PDF?
Short answer: **partly**.

- PDF contains summary-level provenance and trajectory visuals.
- Full generator-by-generator operator history is in JSON (`selected_trace`).

For exact operator audit, use:

- `artifacts/useful/L3/warmstart_states/fix1_warm_start_B_full_state.json`

That file has:

- pool type and size,
- family counts,
- full ordered selected generator list.

---

## 6) Final clarification in one sentence
For the warm-start B/C result: we did **HVA warm-start first**, then **ADAPT with PAOP-LF pool**.  
For the separate trend experiment: ADAPT pools were **`UCCSD+PAOP`** and **`UCCSD+PAOP+HVA`**.
