#!/usr/bin/env bash
set -euo pipefail
cd "/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test"

echo "[CHAIN] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[CHAIN] tag=l4_hh_nph2_hvaSPSA_to_adapt_to_vqe_chain_20260305T022312"
echo "[CHAIN] warm_json=/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/json/l4_hh_nph2_hvaSPSA_to_adapt_to_vqe_chain_20260305T022312_warm_hva_seed.json"
echo "[CHAIN] adapt_json=/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/json/l4_hh_nph2_hvaSPSA_to_adapt_to_vqe_chain_20260305T022312_adapt_from_hva.json"
echo "[CHAIN] vqe_json=/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/json/l4_hh_nph2_hvaSPSA_to_adapt_to_vqe_chain_20260305T022312_vqe_from_adapt.json"

# Step 1: build HVA warm-start seed JSON (adapt_json-compatible initial_state)
python - <<'PYIN'
from datetime import datetime, timezone
from pathlib import Path
import json
from pipelines.hardcoded import hubbard_pipeline as hp
from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian

warm_json = Path(r"/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/json/l4_hh_nph2_hvaSPSA_to_adapt_to_vqe_chain_20260305T022312_warm_hva_seed.json")

h_poly = build_hubbard_holstein_hamiltonian(
    dims=4,
    J=1.0,
    U=4.0,
    omega0=1.0,
    g=0.5,
    n_ph_max=2,
    boson_encoding='binary',
    repr_mode='JW',
    indexing='blocked',
    pbc=False,
    include_zero_point=True,
)

vqe_payload, psi_vqe = hp._run_hardcoded_vqe(
    num_sites=4,
    ordering='blocked',
    boundary='open',
    hopping_t=1.0,
    onsite_u=4.0,
    potential_dv=0.0,
    h_poly=h_poly,
    reps=4,
    restarts=7,
    seed=7,
    maxiter=7111,
    method='SPSA',
    energy_backend='one_apply_compiled',
    vqe_progress_every_s=60.0,
    ansatz_name='hh_hva_ptw',
    problem='hh',
    omega0=1.0,
    g_ep=0.5,
    n_ph_max=2,
    boson_encoding='binary',
)

payload = {
    'generated_utc': datetime.now(timezone.utc).isoformat(),
    'pipeline': 'hva_warm_seed_builder',
    'settings': {
        'problem': 'hh', 'L': 4,
        't': 1.0, 'u': 4.0, 'dv': 0.0,
        'omega0': 1.0, 'g_ep': 0.5,
        'n_ph_max': 2, 'boson_encoding': 'binary',
        'boundary': 'open', 'ordering': 'blocked',
        'vqe_ansatz': 'hh_hva_ptw',
        'vqe_reps': 4, 'vqe_restarts': 7,
        'vqe_maxiter': 7111, 'vqe_method': 'SPSA',
        'vqe_energy_backend': 'one_apply_compiled',
    },
    'vqe': vqe_payload,
    'initial_state': {
        'source': 'vqe',
        'amplitudes_qn_to_q0': hp._state_to_amplitudes_qn_to_q0(psi_vqe),
    },
}
warm_json.parent.mkdir(parents=True, exist_ok=True)
warm_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(f"[CHAIN] hva_seed_written={warm_json}")
print(f"[CHAIN] hva_seed_energy={vqe_payload.get('energy')}")
print(f"[CHAIN] hva_seed_abs_delta_e={vqe_payload.get('abs_delta_e')}")
PYIN

# Step 2: ADAPT from HVA seed (default inner optimizer COBYLA)
python pipelines/hardcoded/adapt_pipeline.py   --problem hh   --L 4 --boundary open --ordering blocked   --boson-encoding binary --n-ph-max 2   --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5   --adapt-pool full_meta   --adapt-state-backend legacy   --adapt-ref-json "/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/json/l4_hh_nph2_hvaSPSA_to_adapt_to_vqe_chain_20260305T022312_warm_hva_seed.json"   --adapt-max-depth 160 --adapt-maxiter 5000   --adapt-eps-grad 5e-7 --adapt-eps-energy 1e-9   --adapt-drop-floor 5e-4 --adapt-drop-patience 3 --adapt-drop-min-depth 12 --adapt-grad-floor 2e-2   --initial-state-source adapt_vqe   --t-final 0.0 --num-times 1 --trotter-steps 1   --skip-pdf   --output-json "/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/json/l4_hh_nph2_hvaSPSA_to_adapt_to_vqe_chain_20260305T022312_adapt_from_hva.json"

# Step 3: VQE run auto-fed from ADAPT state JSON
python pipelines/hardcoded/hubbard_pipeline.py   --problem hh   --L 4 --boundary open --ordering blocked   --boson-encoding binary --n-ph-max 2   --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5   --vqe-ansatz hh_hva_ptw --vqe-reps 4   --vqe-restarts 16 --vqe-maxiter 12000 --vqe-method SPSA --vqe-seed 7   --vqe-energy-backend one_apply_compiled --vqe-progress-every-s 60   --initial-state-source adapt_json   --adapt-input-json "/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/json/l4_hh_nph2_hvaSPSA_to_adapt_to_vqe_chain_20260305T022312_adapt_from_hva.json"   --skip-qpe --num-times 1 --t-final 0.0 --skip-pdf   --output-json "/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/json/l4_hh_nph2_hvaSPSA_to_adapt_to_vqe_chain_20260305T022312_vqe_from_adapt.json"

echo "[CHAIN] done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
