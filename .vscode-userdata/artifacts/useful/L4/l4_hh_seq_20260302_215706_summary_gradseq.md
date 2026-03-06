# L4 HH Checkpoint Continuation Summary (l4_hh_seq_20260302_215706)

## Policy

- Warm/ADAPT progress preserved from prior run artifacts; no warm recomputation for continuation.
- External probe wallclock cap removed for continuation step.
- Stop condition: gradient-sequence cutoff (`grad_floor=0.2`, `patience=3`, `abs_depth_min=12`).

## Progression

- Prior partial best (before policy switch): `|DeltaE|=7.365631e-01`
- Continuation start checkpoint depth: `9`
- Continuation added depth: `6` (final absolute depth `15`)
- Continuation best: `|DeltaE|=7.115062e-01`, `rel=4.271282e+01`
- Continuation runtime: `999.46s`
- Stop reason: `grad_floor_patience(abs_depth>=12,floor=0.2,patience=3)`

## Gate Status

- Probe gate (`<=5e-2`): FAIL
- Production gate (`<=1e-2`): not run in this policy-switched continuation step

## Artifacts

- Continuation JSON: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_probe_cont_grad.json`
- Continuation log: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_probe_cont_grad.log`
- Continuation state export: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_probe_cont_grad_A_probe_state.json`
- Attempts CSV: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/artifacts/useful/L4/l4_hh_seq_20260302_215706_attempts_gradseq.csv`
