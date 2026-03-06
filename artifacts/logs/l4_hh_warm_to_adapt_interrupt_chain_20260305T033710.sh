#!/usr/bin/env bash
set -euo pipefail
cd "/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test"

echo "[CHAIN] start_utc=\[0m\2026-03-05T03:37:10Z"
echo "[CHAIN] warm_script=artifacts/logs/l4_hh_warm_interrupt_safe_20260305T033642.py"
echo "[CHAIN] warm_checkpoint_json=artifacts/json/l4_hh_warm_to_adapt_interrupt_chain_20260305T033710_warm_checkpoint_state.json"
echo "[CHAIN] warm_final_json=artifacts/json/l4_hh_warm_to_adapt_interrupt_chain_20260305T033710_warm_final_state.json"
echo "[CHAIN] adapt_json=artifacts/json/l4_hh_warm_to_adapt_interrupt_chain_20260305T033710_adapt_from_warm.json"

python "artifacts/logs/l4_hh_warm_interrupt_safe_20260305T033642.py"   --checkpoint-json "artifacts/json/l4_hh_warm_to_adapt_interrupt_chain_20260305T033710_warm_checkpoint_state.json"   --final-json "artifacts/json/l4_hh_warm_to_adapt_interrupt_chain_20260305T033710_warm_final_state.json"   --L 4 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5   --n-ph-max 2 --boson-encoding binary --ordering blocked --boundary open   --reps 4 --restarts 7 --maxiter 7111 --method SPSA --seed 7 --progress-every-s 60   &
WARM_PID=$!
echo "[CHAIN] warm_pid=$WARM_PID"
echo "[CHAIN] to_stop_early: kill -INT $WARM_PID"

set +e
wait "$WARM_PID"
WARM_RC=$?
set -e

echo "[CHAIN] warm_exit_code=$WARM_RC"

a=0
REF_JSON=""
if [[ -f "artifacts/json/l4_hh_warm_to_adapt_interrupt_chain_20260305T033710_warm_final_state.json" ]]; then
  REF_JSON="artifacts/json/l4_hh_warm_to_adapt_interrupt_chain_20260305T033710_warm_final_state.json"
elif [[ -f "artifacts/json/l4_hh_warm_to_adapt_interrupt_chain_20260305T033710_warm_checkpoint_state.json" ]]; then
  REF_JSON="artifacts/json/l4_hh_warm_to_adapt_interrupt_chain_20260305T033710_warm_checkpoint_state.json"
else
  echo "[CHAIN][ERROR] no warm state JSON found; cannot transition to ADAPT"
  exit 1
fi

echo "[CHAIN] adapt_ref_json=$REF_JSON"
python pipelines/hardcoded/adapt_pipeline.py   --problem hh   --L 4 --boundary open --ordering blocked   --boson-encoding binary --n-ph-max 2   --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5   --adapt-pool full_meta   --adapt-state-backend legacy   --adapt-ref-json "$REF_JSON"   --adapt-max-depth 160 --adapt-maxiter 5000   --adapt-eps-grad 5e-7 --adapt-eps-energy 1e-9   --adapt-drop-floor 5e-4 --adapt-drop-patience 3 --adapt-drop-min-depth 12 --adapt-grad-floor 2e-2   --initial-state-source adapt_vqe   --t-final 0.0 --num-times 1 --trotter-steps 1   --skip-pdf   --output-json "artifacts/json/l4_hh_warm_to_adapt_interrupt_chain_20260305T033710_adapt_from_warm.json"

echo "[CHAIN] done_utc=\[0m\2026-03-05T03:37:10Z"
