<file_map>
/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2
├── MATH
│   ├── IMPLEMENT_SOON.md *
│   ├── IMPLEMENT_NEXT.md
│   ├── Math.md
│   └── Math.pdf
├── pipelines
│   ├── exact_bench
│   │   ├── hh_fixed_seed_budgeted_projected_dynamics.py * +
│   │   ├── README.md
│   │   ├── benchmark_metrics_proxy.py +
│   │   ├── cross_check_suite.py +
│   │   ├── hh_fixed_handoff_replay_optimizer_probe.py +
│   │   ├── hh_fixed_handoff_replay_optimizer_probe_workflow.py +
│   │   ├── hh_fixed_seed_local_checkpoint_fit.py +
│   │   ├── hh_fixed_seed_qpu_prep_sweep.py +
│   │   ├── hh_full_pool_expressivity_probe.py +
│   │   ├── hh_full_pool_expressivity_probe_workflow.py +
│   │   ├── hh_l2_heavy_prune.py +
│   │   ├── hh_l2_heavy_prune_workflow.py +
│   │   ├── hh_l2_logical_screen.py +
│   │   ├── hh_l2_logical_screen_workflow.py +
│   │   ├── hh_l2_stage_unit_audit.py +
│   │   ├── hh_l2_stage_unit_audit_workflow.py +
│   │   ├── hh_noise_hardware_validation.py +
│   │   ├── hh_noise_robustness_seq_report.py +
│   │   ├── hh_seq_transition_utils.py +
│   │   ├── noise_aer_builders.py +
│   │   ├── noise_model_spec.py +
│   │   ├── noise_oracle_runtime.py +
│   │   ├── noise_patch_selection.py +
│   │   ├── noise_snapshot.py +
│   │   └── statevector_kernels.py +
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
│   │   ├── hh_vqe_from_adapt_family.py * +
│   │   ├── hh_continuation_motifs.py +
│   │   ├── hh_staged_circuit_report.py +
│   │   ├── hh_staged_cli_args.py +
│   │   ├── hh_staged_noise.py +
│   │   ├── hh_staged_noise_workflow.py +
│   │   ├── hh_staged_noiseless.py +
│   │   ├── hubbard_pipeline.py +
│   │   └── qpe_qiskit_shim.py +
│   ├── shell
│   │   ├── build_hh_noise_robustness_report.sh
│   │   └── run_drive_accurate.sh
│   └── run_guide.md *
├── src
│   ├── quantum
│   │   ├── operator_pools
│   │   │   ├── polaron_paop.py * +
│   │   │   ├── vlf_sq.py * +
│   │   │   └── __init__.py +
│   │   ├── time_propagation
│   │   │   ├── __init__.py * +
│   │   │   ├── cfqm_propagator.py * +
│   │   │   ├── cfqm_schemes.py * +
│   │   │   ├── local_checkpoint_fit.py * +
│   │   │   └── projected_real_time.py * +
│   │   ├── compiled_ansatz.py * +
│   │   ├── compiled_polynomial.py * +
│   │   ├── __init__.py
│   │   ├── drives_time_potential.py +
│   │   ├── ed_hubbard_holstein.py +
│   │   ├── hartree_fock_reference_state.py +
│   │   ├── hubbard_latex_python_pairs.py +
│   │   ├── pauli_actions.py +
│   │   ├── pauli_letters_module.py +
│   │   ├── pauli_polynomial_class.py +
│   │   ├── pauli_words.py +
│   │   ├── qubitization_module.py +
│   │   ├── spsa_optimizer.py +
│   │   └── vqe_latex_python_pairs.py +
│   └── __init__.py +
├── test
│   ├── test_hh_adapt_beam_search.py * +
│   ├── test_hh_continuation_generators.py * +
│   ├── test_hh_continuation_replay.py * +
│   ├── test_hh_continuation_scoring.py * +
│   ├── test_local_checkpoint_fit.py * +
│   ├── test_projected_real_time.py * +
│   ├── test_staged_export_replay_roundtrip.py * +
│   ├── conftest.py +
│   ├── test_adapt_vqe_integration.py +
│   ├── test_benchmark_metrics_proxy.py +
│   ├── test_cfqm_acceptance.py +
│   ├── test_cfqm_propagator.py +
│   ├── test_cfqm_schemes.py +
│   ├── test_compiled_ansatz.py +
│   ├── test_compiled_polynomial.py +
│   ├── test_cross_check_suite_cli.py +
│   ├── test_ed_crosscheck.py +
│   ├── test_exact_steps_multiplier.py +
│   ├── test_hardcoded_qpe_isolation.py +
│   ├── test_hh_adapt_family_replay.py +
│   ├── test_hh_continuation_motifs.py +
│   ├── test_hh_continuation_pruning.py +
│   ├── test_hh_continuation_rescue.py +
│   ├── test_hh_continuation_stage_control.py +
│   ├── test_hh_continuation_symmetry.py +
│   ├── test_hh_fixed_handoff_replay_optimizer_probe_workflow.py +
│   ├── test_hh_fixed_seed_budgeted_projected_dynamics.py +
│   ├── test_hh_fixed_seed_local_checkpoint_fit.py +
│   ├── test_hh_fixed_seed_qpu_prep_sweep.py +
│   ├── test_hh_full_pool_expressivity_probe_workflow.py +
│   ├── test_hh_l2_heavy_prune_workflow.py +
│   ├── test_hh_l2_logical_screen_workflow.py +
│   ├── test_hh_l2_stage_unit_audit_workflow.py +
│   ├── test_hh_noise_hardware_validation.py +
│   ├── test_hh_noise_model_spec.py +
│   ├── test_hh_noise_oracle_runtime.py +
│   ├── test_hh_noise_patch_selection.py +
│   ├── test_hh_noise_robustness_benchmarks.py +
│   ├── test_hh_noise_statevector_kernels.py +
│   ├── test_hh_noise_validation_cli.py +
│   ├── test_hh_staged_circuit_report.py +
│   ├── test_hh_staged_noise_workflow.py +
│   ├── test_hh_staged_noiseless_workflow.py +
│   ├── test_hh_vqe_from_adapt_family_seed.py +
│   ├── test_hubbard_adapt_ref_source.py +
│   ├── test_polaron_paop.py +
│   ├── test_report_layers.py +
│   ├── test_spsa_optimizer.py +
│   ├── test_time_potential_drive.py +
│   ├── test_trotter_hh_integration.py +
│   ├── test_vlf_sq_pool.py +
│   └── test_vqe_energy_backend.py +
├── .obsidian
│   ├── app.json
│   ├── appearance.json
│   ├── core-plugins.json
│   └── workspace.json
├── HH
│   ├── .obsidian
│   │   ├── app.json
│   │   ├── appearance.json
│   │   ├── core-plugins.json
│   │   └── workspace.json
│   ├── Untitled.md
│   └── artifacts 1.md
├── artifacts
│   └── agent_runs
│       └── overnight_shallow_g0p5_depth400
│           └── summarize_when_done.py +
├── artifacts 1
│   ├── json
│   │   ├── noise_l2_pdf
│   │   │   ├── basic.json
│   │   │   ├── ideal_control.json
│   │   │   ├── ideal_control_compiled.json
│   │   │   ├── ideal_control_compiled_spsa.json
│   │   │   ├── ideal_control_ptw_spsa.json
│   │   │   ├── scheduled.json
│   │   │   └── shots.json
│   │   ├── noise_l2_test
│   │   │   ├── basic.json
│   │   │   ├── scheduled.json
│   │   │   └── shots.json
│   │   ├── 20260309_knee_layerwise_warm_hh_hva.json
│   │   ├── 20260309_knee_layerwise_warm_hh_hva_adapt_handoff.json
│   │   ├── 20260309_knee_layerwise_warm_hh_hva_replay.csv
│   │   ├── 20260309_knee_layerwise_warm_hh_hva_replay.json
│   │   ├── 20260309_knee_ptw_warm_hh_hva_ptw.json
│   │   ├── 20260309_knee_ptw_warm_hh_hva_ptw_adapt_handoff.json
│   │   ├── 20260309_knee_ptw_warm_hh_hva_ptw_replay.csv
│   │   ├── 20260309_knee_ptw_warm_hh_hva_ptw_replay.json
│   │   ├── 20260309_plateau_v3_knee_layerwise.json
│   │   ├── 20260309_plateau_v3_knee_layerwise_adapt_handoff.json
│   │   ├── 20260309_plateau_v3_knee_layerwise_replay.csv
│   │   ├── 20260309_plateau_v3_knee_layerwise_replay.json
│   │   ├── 20260309_plateau_v3_knee_ptw.json
│   │   ├── 20260309_plateau_v3_knee_ptw_adapt_handoff.json
│   │   ├── 20260309_plateau_v3_knee_ptw_replay.csv
│   │   ├── 20260309_plateau_v3_knee_ptw_replay.json
│   │   ├── 20260309_plateau_v3_strong_layerwise.json
│   │   ├── 20260309_plateau_v3_strong_layerwise_adapt_handoff.json
│   │   ├── 20260309_plateau_v3_strong_layerwise_replay.csv
│   │   ├── 20260309_plateau_v3_strong_layerwise_replay.json
│   │   ├── 20260309_plateau_v3_strong_ptw.json
│   │   ├── 20260309_plateau_v3_strong_ptw_adapt_handoff.json
│   │   ├── 20260309_plateau_v3_strong_ptw_replay.csv
│   │   ├── 20260309_plateau_v3_strong_ptw_replay.json
│   │   ├── 20260309_plateau_v3_weak_layerwise.json
│   │   ├── 20260309_plateau_v3_weak_layerwise_adapt_handoff.json
│   │   ├── 20260309_plateau_v3_weak_layerwise_replay.csv
│   │   ├── 20260309_plateau_v3_weak_layerwise_replay.json
│   │   ├── 20260309_plateau_v3_weak_ptw.json
│   │   ├── 20260309_plateau_v3_weak_ptw_adapt_handoff.json
│   │   ├── 20260309_plateau_v3_weak_ptw_replay.csv
│   │   ├── 20260309_plateau_v3_weak_ptw_replay.json
│   │   ├── 20260309_strong_layerwise_warm_hh_hva.json
│   │   ├── 20260309_strong_layerwise_warm_hh_hva_adapt_handoff.json
│   │   ├── 20260309_strong_layerwise_warm_hh_hva_replay.csv
│   │   ├── 20260309_strong_layerwise_warm_hh_hva_replay.json
│   │   ├── 20260309_strong_ptw_warm_hh_hva_ptw.json
│   │   ├── 20260309_strong_ptw_warm_hh_hva_ptw_adapt_handoff.json
│   │   ├── 20260309_strong_ptw_warm_hh_hva_ptw_replay.csv
│   │   ├── 20260309_strong_ptw_warm_hh_hva_ptw_replay.json
│   │   ├── 20260309_weak_layerwise_warm_hh_hva.json
│   │   ├── 20260309_weak_layerwise_warm_hh_hva_adapt_handoff.json
│   │   ├── 20260309_weak_layerwise_warm_hh_hva_replay.csv
│   │   ├── 20260309_weak_layerwise_warm_hh_hva_replay.json
│   │   ├── 20260309_weak_ptw_warm_hh_hva_ptw.json
│   │   ├── 20260309_weak_ptw_warm_hh_hva_ptw_adapt_handoff.json
│   │   ├── 20260309_weak_ptw_warm_hh_hva_ptw_replay.csv
│   │   ├── 20260309_weak_ptw_warm_hh_hva_ptw_replay.json
│   │   ├── L3_hh_drive_nph2_heavy_cutover_03268371471231371_adapt_handoff.json
│   │   ├── L3_hh_drive_nph2_heavy_cutover_03268371471231371_warm_checkpoint_state.json
│   │   ├── L3_hh_drive_nph2_heavy_cutover_03268371471231371_warm_cutover_state.json
│   │   ├── cfqm4_hh_L2_driveA1.0_U4_nph1_t10.json
│   │   ├── cfqm4_vs_suzuki2_hh_L2_hwmatch.json
│   │   ├── hh_noise_validation_L2_hh_hva_ptw_noiseless_ideal.json
│   │   ├── hh_staged_L2_drive_ptw_spsa_heavy_d120.json
│   │   ├── hh_staged_L2_drive_ptw_spsa_heavy_d120_adapt_handoff.json
│   │   ├── hh_staged_L2_drive_ptw_spsa_heavy_d120_replay.csv
│   │   ├── hh_staged_L2_drive_ptw_spsa_heavy_d120_replay.json
│   │   ├── hh_staged_L2_drive_t1_U2_dv0_w1_g1_nph1_warmhh_hva_ptw_b3d876db0d_adapt_handoff.json
│   │   ├── hh_staged_L2_drive_t1_U2_dv0_w1_g1_nph1_warmhh_hva_ptw_b3d876db0d_warm_checkpoint_state.json
│   │   ├── hh_staged_L2_drive_t1_U2_dv0_w1_g1_nph1_warmhh_hva_ptw_b3d876db0d_warm_cutover_state.json
│   │   ├── hh_staged_L2_static_t1_U2_dv0_w1_g1_nph1_warmhh_hva_ptw_a56ab6fb09_adapt_handoff.json
│   │   ├── hh_staged_L2_static_t1_U2_dv0_w1_g1_nph1_warmhh_hva_ptw_a56ab6fb09_replay.csv
│   │   ├── hh_staged_L2_static_t1_U2_dv0_w1_g1_nph1_warmhh_hva_ptw_a56ab6fb09_replay.json
│   │   ├── hh_staged_L3_static_t1_U2_dv0_w1_g1_nph1_warmhh_hva_ptw_9afb5d05d5_adapt_handoff.json
│   │   ├── hh_staged_L3_static_t1_U2_dv0_w1_g1_nph1_warmhh_hva_ptw_9afb5d05d5_replay.csv
│   │   ├── hh_staged_L3_static_t1_U2_dv0_w1_g1_nph1_warmhh_hva_ptw_9afb5d05d5_replay.json
│   │   ├── hh_staged_circuit_L2_plateau_v3_weak_ptw_L2_adapt_handoff.json
│   │   ├── hh_staged_circuit_L2_plateau_v3_weak_ptw_L2_replay.csv
│   │   ├── hh_staged_circuit_L2_plateau_v3_weak_ptw_L2_replay.json
│   │   ├── hh_staged_circuit_L2_reasonable_L2_adapt_handoff.json
│   │   ├── hh_staged_circuit_L2_reasonable_L2_replay.csv
│   │   ├── hh_staged_circuit_L2_reasonable_L2_replay.json
│   │   ├── hh_staged_circuit_smoke_legacy_L2_adapt_handoff.json
│   │   ├── hh_staged_circuit_smoke_legacy_nohhseed_L2_adapt_handoff.json
│   │   ├── hh_staged_circuit_smoke_legacy_nohhseed_L2_replay.csv
│   │   ├── hh_staged_circuit_smoke_legacy_nohhseed_L2_replay.json
│   │   ├── hh_staged_circuit_smoke_legacy_nohhseed_L3_adapt_handoff.json
│   │   ├── hh_staged_circuit_smoke_legacy_nohhseed_L3_replay.csv
│   │   ├── hh_staged_circuit_smoke_legacy_nohhseed_L3_replay.json
│   │   ├── hh_staged_circuit_smoke_noprune_L2_adapt_handoff.json
│   │   ├── l2_first_noise_anchor_adapt_handoff.json
│   │   ├── l2_first_noise_anchor_replay.csv
│   │   ├── l2_first_noise_anchor_replay.json
│   │   ├── l2_first_noise_anchor_snapshot.json
│   │   ├── l2_first_noise_anchor_warm_checkpoint_state.json
│   │   ├── l2_first_noise_anchor_warm_cutover_state.json
│   │   ├── l2_first_noise_backend_scheduled.json
│   │   ├── l2_first_noise_generic6_adapt_handoff.json
│   │   ├── l2_first_noise_generic6_backend_scheduled.json
│   │   ├── l2_first_noise_generic6_replay.csv
│   │   ├── l2_first_noise_generic6_replay.json
│   │   ├── l2_first_noise_generic6_snapshot.json
│   │   ├── l2_first_noise_generic6_warm_checkpoint_state.json
│   │   ├── l2_first_noise_generic6_warm_cutover_state.json
│   │   ├── l2_first_noise_jakarta_adapt_handoff.json
│   │   ├── l2_first_noise_jakarta_backend_scheduled.json
│   │   └── l2_first_noise_jakarta_replay.csv
│   ├── logs
│   │   ├── L3_hh_drive_nph2_heavy_cutover_03268371471231371.log.pre_resume_20260310T173403Z
│   │   └── L3_hh_drive_nph2_heavy_cutover_03268371471231371.stdout.log.pre_resume_20260310T173403Z
│   ├── pdf
│   │   ├── noise_l2_pdf
│   │   │   ├── basic.pdf
│   │   │   ├── scheduled.pdf
│   │   │   └── shots.pdf
│   │   ├── cfqm4_hh_L2_driveA1.0_U4_nph1_t10.pdf
│   │   ├── cfqm4_vs_suzuki2_hh_L2_hwmatch.pdf
│   │   ├── hh_staged_L2_drive_ptw_spsa_heavy_d120.pdf
│   │   ├── hh_staged_circuit_report_L2_L3.pdf
│   │   ├── hh_staged_circuit_report_L2_L3_smoke.pdf
│   │   ├── hh_staged_circuit_report_L2_plateau_v3_weak_ptw.pdf
│   │   ├── hh_staged_circuit_report_L2_reasonable.pdf
│   │   ├── suzuki2_hh_L2_driveA1.0_U4_nph1_t10_trotter128.pdf
│   │   └── suzuki2_hh_L2_driveA1.0_U4_nph1_t10_trotter64.pdf
│   ├── useful
│   │   ├── L2
│   │   │   ├── 20260309_knee_layerwise_warm_hh_hva_replay.md
│   │   │   ├── 20260309_knee_ptw_warm_hh_hva_ptw_replay.md
│   │   │   ├── 20260309_plateau_v3_knee_layerwise_replay.md
│   │   │   ├── 20260309_plateau_v3_knee_ptw_replay.md
│   │   │   ├── 20260309_plateau_v3_strong_layerwise_replay.md
│   │   │   ├── 20260309_plateau_v3_strong_ptw_replay.md
│   │   │   ├── 20260309_plateau_v3_summary.md
│   │   │   ├── 20260309_plateau_v3_weak_layerwise_replay.md
│   │   │   ├── 20260309_plateau_v3_weak_ptw_replay.md
│   │   │   ├── 20260309_strong_layerwise_warm_hh_hva_replay.md
│   │   │   ├── 20260309_strong_ptw_warm_hh_hva_ptw_replay.md
│   │   │   ├── 20260309_weak_layerwise_warm_hh_hva_replay.md
│   │   │   ├── 20260309_weak_ptw_warm_hh_hva_ptw_replay.md
│   │   │   ├── hh_staged_L2_drive_ptw_spsa_heavy_d120_replay.md
│   │   │   ├── hh_staged_L2_static_t1_U2_dv0_w1_g1_nph1_warmhh_hva_ptw_a56ab6fb09_replay.md
│   │   │   ├── hh_staged_circuit_L2_plateau_v3_weak_ptw_L2_replay.md
│   │   │   ├── hh_staged_circuit_L2_reasonable_L2_replay.md
│   │   │   ├── hh_staged_circuit_smoke_legacy_nohhseed_L2_replay.md
│   │   │   ├── l2_first_noise_anchor_replay.md
│   │   │   ├── l2_first_noise_generic6_replay.md
│   │   │   ├── l2_first_noise_jakarta_replay.md
│   │   │   ├── l2_first_noise_jakarta_s2_fast_replay.md
│   │   │   ├── l2_first_noise_jakarta_s2_ok_replay.md
│   │   │   └── l2_first_noise_jakarta_s2_replay.md
│   │   └── L3
│   │       ├── hh_staged_L3_static_t1_U2_dv0_w1_g1_nph1_warmhh_hva_ptw_9afb5d05d5_replay.md
│   │       └── hh_staged_circuit_smoke_legacy_nohhseed_L3_replay.md
│   ├── user_runs
│   │   └── 20260309_hh_l2_noiseless
│   │       └── json
│   │           ├── raw_ptw_g0.5_nph1.json
│   │           ├── raw_ptw_g1.0_nph1.json
│   │           ├── raw_ptw_g1.0_nph2.json
│   │           ├── raw_ptw_g1.25_nph1.json
│   │           ├── raw_ptw_g1.25_nph2.json
│   │           ├── raw_ptw_g1.5_nph1.json
│   │           └── raw_ptw_g1.5_nph2.json
│   └── hh_noise_validation_L2_hh_hva_ptw_run_summary.md
├── docs
│   └── reports
│       ├── __init__.py
│       ├── pdf_utils.py +
│       ├── qiskit_circuit_report.py +
│       ├── report_labels.py +
│       └── report_pages.py +
├── prompt-exports
│   ├── 2026-03-12-2315-plan-hh-failure-test-ladder.md
│   ├── 2026-03-12-2359-plan-hh-operator-pool-expansion.md
│   ├── 2026-03-13-0010-plan-hh-sweep-experiments.md
│   ├── 2026-03-14-0030-question-hh-pareto-problem-definition.md
│   ├── 2026-03-14-140128-plan-hh-uccsd-paop-seed-adapt-workflow.md
│   ├── 2026-03-14-1502-plan-hh-generator-parallel-experiments.md
│   ├── 2026-03-14-162545-plan-hh-adapt-beam-search-architecture.md
│   ├── 2026-03-14-175710-plan-hh-adapt-beam-refactor-contract.md
│   └── 2026-03-15-plan-hh-realtime-vqs-architecture.md
├── README.md *
├── .gitignore
├── AGENTS.md
├── L2_hh_smart_adapt.json
├── L2_hh_smart_replay.csv
├── L2_hh_smart_replay.json
├── L2_hh_smart_replay.md
├── L2_hh_smart_replay_bundle_diagnostic.json
├── L2_hh_smart_results_diagnostic.json
├── L2_hh_smart_results_diagnostic.pdf
├── L2_hh_smart_warm.json
├── activate_ibm_runtime.py +
├── investigation_cfqm_qpu_mapping.md
├── investigation_hh_noise_boundaries.md
├── pareto_flow_chart.md
└── pareto_flow_chart.pdf


(* denotes selected files)
(+ denotes code-map available)

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/pauli_polynomial_class.py
Imports:
  - import numpy as np
  - from src.quantum.qubitization_module import PauliTerm
---
Classes:
  - PauliPolynomial
    Methods:
      - L9: def __init__(self, repr_mode, pol=None):
      - L15: def get_nq(self):
      - L21: def return_polynomial(self):
      - L23: def count_number_terms(self):
      - L25: def add_term(self, pt):
      - L28: def _clone_term(pt):
      - L31: def _clone_terms(cls, terms):
      - L33: def __add__(self, pp):
      - L44: def __iadd__(self, pp):
      - L56: def __sub__(self, pp):
      - L68: def __isub__(self, pp):
      - L81: def __mul__(self, pp):
      - L101: def __rmul__(self, other):
      - L111: def __pow__(self, exponent):
      - L124: def _reduce(self):
      - L140: def visualize_polynomial(self):
  - fermion_plus_operator
    Methods:
      - L157: def __init__(self, repr_mode, nq, j):
      - L163: def __set_JW_operator(self, nq, j):
  - fermion_minus_operator
    Methods:
      - L185: def __init__(self, repr_mode, nq, j):
      - L191: def __set_JW_operator(self, nq, j):
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/pauli_actions.py
Imports:
  - import math
  - from dataclasses import dataclass
  - import numpy as np
---
Classes:
  - CompiledPauliAction
    Properties:
      - label_exyz
      - perm
      - phase

Functions:
  - L27: def compile_pauli_action_exyz(label_exyz: str, nq: int) -> CompiledPauliAction:
  - L59: def apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
  - L66: def apply_exp_term(
    psi: np.ndarray,
    action: CompiledPauliAction,
    coeff: complex,
    dt: float,
    tol: float = 1e-12,
) -> np.ndarray:

Global vars:
  - __all__
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/exact_bench/hh_fixed_seed_qpu_prep_sweep.py
Imports:
  - import argparse
  - import csv
  - import json
  - import sys
  - from dataclasses import asdict, dataclass
  - from datetime import datetime, timezone
  - from pathlib import Path
  - from typing import Any, Mapping, Sequence
  - from docs.reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    get_plt,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)
  - from docs.reports.qiskit_circuit_report import build_time_dynamics_circuit, transpile_circuit_metrics
  - from pipelines.hardcoded.hh_staged_noiseless import parse_args as parse_staged_args
  - from pipelines.hardcoded.hh_staged_workflow import resolve_staged_hh_config, run_staged_hh_noiseless
  - from src.quantum.drives_time_potential import build_gaussian_sinusoid_density_drive
  - from pipelines.hardcoded.hh_vqe_from_adapt_family import build_replay_sequence_from_input_json
  - from qiskit import QuantumCircuit
---
Classes:
  - SweepConfig
    Properties:
      - fixed_final_state_json
      - output_json
      - output_csv
      - output_pdf
      - run_root
      - tag
      - backend_name
      - use_fake_backend
      - circuit_optimization_level
      - circuit_seed_transpiler
      - suzuki_steps
      - cfqm_steps
      - t_final
      - num_times
      - exact_steps_multiplier
      - drive_A
      - drive_omega
      - drive_tbar
      - drive_phi
      - drive_pattern
      - drive_t0
      - drive_time_sampling
      - budget_mode
      - cfqm_stage_exp
      - cfqm_coeff_drop_abs_tol
      - cfqm_normalize

Functions:
  - L74: def _now_utc() -> str:
  - L78: def _jsonable(value: Any) -> Any:
  - L88: def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
  - L93: def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
  - L111: def _parse_steps(raw: str) -> tuple[int, ...]:
  - L131: def _load_seed_settings(path: Path) -> dict[str, Any]:
  - L143: def _collect_hardcoded_terms_exyz(h_poly: Any) -> tuple[list[str], dict[str, complex]]:
  - L157: def _build_drive_provider(
    *,
    num_sites: int,
    nq_total: int,
    ordering: str,
    cfg: SweepConfig,
) -> Any:
  - L181: def _build_snapshot_budget_context(cfg: SweepConfig, *, seed_settings: Mapping[str, Any]) -> dict[str, Any]:
  - L200: def _build_candidate_args(
    *,
    cfg: SweepConfig,
    seed_settings: Mapping[str, Any],
    method: str,
    trotter_steps: int,
    run_dir: Path,
) -> list[str]:
  - L284: def _extract_drive_profile(payload: Mapping[str, Any]) -> Mapping[str, Any]:
  - L294: def _snapshot_budget_details(
    *,
    cfg: SweepConfig,
    snapshot_ctx: Mapping[str, Any],
    method: str,
    trotter_steps: int,
    times: Sequence[float],
) -> dict[str, Any]:
  - L356: def _candidate_row(
    payload: Mapping[str, Any],
    *,
    method: str,
    trotter_steps: int,
    run_dir: Path,
    cfg: SweepConfig,
    snapshot_budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  - L455: def _is_dominated(row_i: Mapping[str, Any], row_j: Mapping[str, Any]) -> bool:
  - L465: def _pareto_shortlist(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
  - L481: def _plot_energy_page(pdf: Any, *, method: str, candidates: Sequence[Mapping[str, Any]], title_suffix: str) -> None:
  - L506: def _plot_fidelity_page(pdf: Any, *, method: str, candidates: Sequence[Mapping[str, Any]], title_suffix: str) -> None:
  - L527: def _write_summary_pdf(
    cfg: SweepConfig,
    *,
    seed_settings: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    pareto_shortlist: Sequence[Mapping[str, Any]],
    run_command: str,
) -> None:
  - L644: def build_parser() -> argparse.ArgumentParser:
  - L686: def parse_args(argv: list[str] | None = None) -> SweepConfig:
  - L718: def run_sweep(cfg: SweepConfig, *, run_command: str | None = None) -> dict[str, Any]:
  - L817: def main(argv: list[str] | None = None) -> None:

Global vars:
  - REPO_ROOT
  - _DEFAULT_FIXED_FINAL_STATE_JSON
  - _DEFAULT_OUTPUT_JSON
  - _DEFAULT_OUTPUT_CSV
  - _DEFAULT_OUTPUT_PDF
  - _DEFAULT_RUN_ROOT
  - _DEFAULT_TAG
  - _DEFAULT_SUZUKI_STEPS
  - _DEFAULT_CFQM_STEPS
  - _DEFAULT_BACKEND_NAME
  - _DEFAULT_BUDGET_MODE
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/vqe_latex_python_pairs.py
Imports:
  - import sys
  - from pathlib import Path
  - import inspect
  - import math
  - import time
  - from dataclasses import dataclass
  - from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
  - import numpy as np
  - from src.quantum.spsa_optimizer import SPSAResult, spsa_minimize
  - from IPython.display import Markdown, Math as IPyMath, display
  - from src.quantum.pauli_polynomial_class import (
        PauliPolynomial,
        fermion_minus_operator,
        fermion_plus_operator,
    )
  - from src.quantum.pauli_words import PauliTerm
  - from src.quantum.hubbard_latex_python_pairs import (
        Dims,
        SPIN_DN,
        SPIN_UP,
        Spin,
        boson_displacement_operator,
        boson_number_operator,
        boson_qubits_per_site,
        build_holstein_coupling,
        build_holstein_phonon_energy,
        build_hubbard_holstein_drive,
        build_hubbard_kinetic,
        build_hubbard_onsite,
        build_hubbard_potential,
        bravais_nearest_neighbor_edges,
        mode_index,
        n_sites_from_dims,
        phonon_qubit_indices_for_site,
    )
  - from src.quantum.compiled_polynomial import compile_polynomial_action, energy_via_one_apply
  - from hartree_fock_reference_state import hartree_fock_bitstring
  - from src.quantum.hartree_fock_reference_state import hartree_fock_bitstring
  - from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
  - from scipy.optimize import minimize  # type: ignore
  - from src.quantum.compiled_polynomial import compile_polynomial_action
---
Classes:
  - _FallbackLog
    Methods:
      - L36: def error(msg: str):
      - L40: def info(msg: str):
  - AnsatzTerm
    Properties:
      - label
      - polynomial
  - HubbardTermwiseAnsatz
    Methods:
      - L543: def __init__(
        self,
        dims: Dims,
        t: float,
        U: float,
        *,
        v: Optional[Union[float, Sequence[float], Dict[int, float]]] = None,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        edges: Optional[Sequence[Tuple[int, int]]] = None,
        pbc: Union[bool, Sequence[bool]] = True,
        include_potential_terms: bool = True,
    ):
      - L579: def _build_base_terms(self) -> None:
      - L606: def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ) -> np.ndarray:
  - HubbardLayerwiseAnsatz
    Methods:
      - L643: def _build_base_terms(self) -> None:
      - L688: def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ) -> np.ndarray:
  - HardcodedUCCSDAnsatz
    Methods:
      - L728: def __init__(
        self,
        dims: Dims,
        num_particles: Tuple[int, int],
        *,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        include_singles: bool = True,
        include_doubles: bool = True,
    ):
      - L764: def _single_generator(self, p_occ: int, q_virt: int) -> PauliPolynomial:
      - L774: def _double_generator(self, i_occ: int, j_occ: int, a_virt: int, b_virt: int) -> PauliPolynomial:
      - L789: def _build_base_terms(self) -> None:
      - L871: def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ) -> np.ndarray:
  - HardcodedUCCSDLayerwiseAnsatz
    Methods:
      - L908: def _build_base_terms(self) -> None:
      - L1001: def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ) -> np.ndarray:
  - HubbardHolsteinTermwiseAnsatz
    Methods:
      - L1054: def __init__(
        self,
        dims: Dims,
        J: float,
        U: float,
        omega0: float,
        g: float,
        n_ph_max: int,
        *,
        boson_encoding: str = "binary",
        v: SitePotential = None,
        v_t: TimePotential = None,
        v0: SitePotential = None,
        t_eval: Optional[float] = None,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        edges: Optional[Sequence[Tuple[int, int]]] = None,
        pbc: Union[bool, Sequence[bool]] = True,
        include_zero_point: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ):
      - L1110: def _build_base_terms(self) -> None:
      - L1115: def _add_terms_from_poly(label_prefix: str, poly: PauliPolynomial) -> None:
      - L1178: def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: Optional[float] = None,
        sort_terms: Optional[bool] = None,
    ) -> np.ndarray:
  - HubbardHolsteinPhysicalTermwiseAnsatz
    Methods:
      - L1231: def __init__(
        self,
        dims: Dims,
        J: float,
        U: float,
        omega0: float,
        g: float,
        n_ph_max: int,
        *,
        boson_encoding: str = "binary",
        v: SitePotential = None,
        v_t: TimePotential = None,
        v0: SitePotential = None,
        t_eval: Optional[float] = None,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        edges: Optional[Sequence[Tuple[int, int]]] = None,
        pbc: Union[bool, Sequence[bool]] = True,
        include_zero_point: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ):
      - L1287: def _build_base_terms(self) -> None:
      - L1348: def n_op(p_mode: int) -> PauliPolynomial:
      - L1395: def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: Optional[float] = None,
        sort_terms: Optional[bool] = None,
    ) -> np.ndarray:
  - HubbardHolsteinLayerwiseAnsatz
    Methods:
      - L1450: def __init__(
        self,
        dims: Dims,
        J: float,
        U: float,
        omega0: float,
        g: float,
        n_ph_max: int,
        *,
        boson_encoding: str = "binary",
        v: SitePotential = None,
        v_t: TimePotential = None,
        v0: SitePotential = None,
        t_eval: Optional[float] = None,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        edges: Optional[Sequence[Tuple[int, int]]] = None,
        pbc: Union[bool, Sequence[bool]] = True,
        include_zero_point: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ):
      - L1509: def _poly_group(
        self,
        label: str,
        poly: PauliPolynomial,
    ) -> None:
      - L1533: def _build_base_terms(self) -> None:
      - L1608: def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: Optional[float] = None,
        sort_terms: Optional[bool] = None,
    ) -> np.ndarray:
  - VQEResult
    Properties:
      - energy
      - theta
      - success
      - message
      - nfev
      - nit
      - best_restart
      - progress_history
      - restart_summaries
      - optimizer_memory

Functions:
  - L56: def _missing_dep(*_args, **_kwargs):
  - L92: def n_sites_from_dims(dims: Dims) -> int:
  - L100: def bravais_nearest_neighbor_edges(dims: Dims, pbc: Union[bool, Sequence[bool]] = True):
  - L103: def mode_index(site: int, spin: Spin, indexing: str = "interleaved", n_sites: Optional[int] = None) -> int:
  - L106: def _missing_hh(*_args, **_kwargs):
  - L207: def _normalize_pauli_string(pauli: str) -> str:
  - L222: def basis_state(nq: int, bitstring: Optional[str] = None) -> np.ndarray:
  - L238: def apply_pauli_string(state: np.ndarray, pauli: str) -> np.ndarray:
  - L275: def expval_pauli_string(state: np.ndarray, pauli: str) -> complex:
  - L280: def expval_pauli_polynomial(state: np.ndarray, H: PauliPolynomial, tol: float = 1e-12) -> float:
  - L305: def expval_pauli_polynomial_one_apply(
    state: np.ndarray,
    H: PauliPolynomial,
    *,
    tol: float = 1e-12,
    cache: Optional[Dict[str, Any]] = None,
) -> float:
  - L341: def apply_pauli_rotation(state: np.ndarray, pauli: str, angle: float) -> np.ndarray:
  - L351: def apply_exp_pauli_polynomial(
    state: np.ndarray,
    H: PauliPolynomial,
    theta: float,
    *,
    ignore_identity: bool = True,
    coefficient_tolerance: float = 1e-12,
    sort_terms: bool = True,
) -> np.ndarray:
  - L389: def half_filled_num_particles(num_sites: int) -> Tuple[int, int]:
  - L402: def jw_number_operator(repr_mode: str, nq: int, p_mode: int) -> PauliPolynomial:
  - L423: def hubbard_hop_term(
    nq: int,
    p_mode: int,
    q_mode: int,
    t: float,
    *,
    repr_mode: str = "JW",
) -> PauliPolynomial:
  - L438: def hubbard_onsite_term(
    nq: int,
    p_up: int,
    p_dn: int,
    U: float,
    *,
    repr_mode: str = "JW",
) -> PauliPolynomial:
  - L451: def hubbard_potential_term(
    nq: int,
    p_mode: int,
    v_i: float,
    *,
    repr_mode: str = "JW",
) -> PauliPolynomial:
  - L461: def _parse_site_potential(
    v: Optional[Union[float, Sequence[float], Dict[int, float]]],
    n_sites: int,
) -> List[float]:
  - L493: def _single_term_polynomials_sorted(
    poly: PauliPolynomial,
    *,
    repr_mode: str,
    coefficient_tolerance: float = 1e-12,
) -> List[PauliPolynomial]:
  - L519: def hubbard_holstein_reference_state(**kwargs) -> np.ndarray:
  - L1659: def _try_import_scipy_minimize():
  - L1667: def vqe_minimize(
    H: PauliPolynomial,
    ansatz: Any,
    psi_ref: np.ndarray,
    *,
    restarts: int = 3,
    seed: int = 7,
    initial_point_stddev: float = 0.3,
    initial_point: Optional[np.ndarray] = None,
    use_initial_point_first_restart: bool = True,
    method: str = "SLSQP",
    maxiter: int = 1800,
    bounds: Optional[Tuple[float, float]] = (-math.pi, math.pi),
    progress_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_every_s: float = 60.0,
    progress_label: str = "vqe_minimize",
    track_history: bool = False,
    emit_theta_in_progress: bool = False,
    return_best_on_keyboard_interrupt: bool = False,
    early_stop_checker: Optional[Callable[[Dict[str, Any]], bool]] = None,
    spsa_a: float = 0.2,
    spsa_c: float = 0.1,
    spsa_alpha: float = 0.602,
    spsa_gamma: float = 0.101,
    spsa_A: float = 10.0,
    spsa_avg_last: int = 0,
    spsa_eval_repeats: int = 1,
    spsa_eval_agg: str = "mean",
    energy_backend: str = "legacy",
    optimizer_memory: Optional[Dict[str, Any]] = None,
    spsa_refresh_every: int = 0,
    spsa_precondition_mode: str = "none",
) -> VQEResult:
  - L1735: def energy_fn(x: np.ndarray) -> float:
  - L1758: def _emit_progress(event: str, **payload: Any) -> None:
  - L1815: def _objective_with_progress(x: np.ndarray) -> float:
  - L1933: def _spsa_heartbeat(_payload: Dict[str, Any]) -> None:
  - L2237: def pauli_matrix(pauli: str) -> np.ndarray:
  - L2245: def hamiltonian_matrix(H: PauliPolynomial, tol: float = 1e-12) -> np.ndarray:
  - L2260: def _spin_orbital_index_sets(num_sites: int, ordering: str) -> Tuple[List[int], List[int]]:
  - L2270: def _sector_basis_indices(
    nq: int,
    alpha_indices: Sequence[int],
    beta_indices: Sequence[int],
    n_alpha: int,
    n_beta: int,
) -> List[int]:
  - L2286: def exact_ground_energy_sector(
    H: PauliPolynomial,
    *,
    num_sites: int,
    num_particles: Tuple[int, int],
    indexing: str = "blocked",
    tol: float = 1e-12,
) -> float:
  - L2306: def _sector_basis_indices_fermion_only(
    nq_total: int,
    alpha_indices: Sequence[int],
    beta_indices: Sequence[int],
    n_alpha: int,
    n_beta: int,
) -> List[int]:
  - L2327: def exact_ground_energy_sector_hh(
    H: PauliPolynomial,
    *,
    num_sites: int,
    num_particles: Tuple[int, int],
    n_ph_max: int,
    boson_encoding: str = "binary",
    indexing: str = "blocked",
    tol: float = 1e-12,
) -> float:
  - L2365: def show_latex_and_code(title: str, latex_expr: str, fn) -> None:
  - L2377: def show_vqe_latex_python_pairs() -> None:

Global vars:
  - _cwd
  - log
  - __all__
  - LATEX_TERMS
  - SitePotential
  - TimePotential
  - _PAULI_MATS
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_motifs.py
Imports:
  - from dataclasses import asdict
  - import hashlib
  - from pathlib import Path
  - import json
  - from typing import Any, Mapping, Sequence
  - from pipelines.hardcoded.hh_continuation_types import MotifLibrary, MotifMetadata, MotifRecord
---

Functions:
  - L15: def _boundary_behavior_from_sites(
    support_sites: Sequence[int],
    *,
    num_sites: int,
) -> str:
  - L34: def _candidate_boundary_behavior(
    generator_metadata: Mapping[str, Any] | None,
    *,
    target_num_sites: int,
) -> str:
  - L50: def _boundary_behavior_matches(
    source_behavior: str,
    target_behavior: str,
    *,
    transfer_mode: str,
) -> bool:
  - L72: def merge_motif_libraries(
    libraries: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any] | None:
  - L194: def extract_motif_library(
    *,
    generator_metadata: Sequence[Mapping[str, Any]],
    theta: Sequence[float],
    source_num_sites: int,
    source_tag: str,
    ordering: str,
    boson_encoding: str,
) -> dict[str, Any]:
  - L257: def load_motif_library_from_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
  - L280: def load_motif_library_from_json(path: str | Path) -> dict[str, Any] | None:
  - L285: def motif_bonus_for_generator(
    *,
    generator_metadata: Mapping[str, Any] | None,
    motif_library: Mapping[str, Any] | None,
    target_num_sites: int,
    transfer_mode: str = "exact_match_v1",
) -> tuple[float, dict[str, Any] | None]:
  - L347: def select_tiled_generators_from_library(
    *,
    motif_library: Mapping[str, Any] | None,
    registry_by_label: Mapping[str, Mapping[str, Any]],
    target_num_sites: int,
    excluded_labels: Sequence[str],
    max_seed: int,
    transfer_mode: str = "exact_match_v1",
) -> list[dict[str, Any]]:
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hubbard_pipeline.py
Imports:
  - import argparse
  - import json
  - import math
  - import os
  - import sys
  - import time
  - import warnings
  - from datetime import datetime, timezone
  - from pathlib import Path
  - from typing import Any
  - import numpy as np
  - from docs.reports.pdf_utils import (
    HAS_MATPLOTLIB,
    require_matplotlib,
    get_plt,
    get_PdfPages,
    render_text_page,
    render_command_page,
    current_command_string,
)
  - from docs.reports.report_labels import report_branch_label, report_method_label
  - from docs.reports.report_pages import (
    render_executive_summary_page,
    render_manifest_overview_page,
    render_section_divider_page,
)
  - from src.quantum.hartree_fock_reference_state import (
    hartree_fock_statevector,
    hubbard_holstein_reference_state,
)
  - from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
  - from src.quantum.drives_time_potential import (
    build_gaussian_sinusoid_density_drive,
    evaluate_drive_waveform,
    reference_method_name,
)
  - from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_compiled_pauli as _apply_compiled_pauli_shared,
    apply_exp_term as _apply_exp_term_shared,
    compile_pauli_action_exyz as _compile_pauli_action_exyz_shared,
)
  - from src.quantum.time_propagation import (
    cfqm_step,
    get_cfqm_scheme,
)
  - from src.quantum.time_propagation.cfqm_schemes import validate_scheme
  - from src.quantum import vqe_latex_python_pairs as vqe_mod
  - from pipelines.hardcoded import adapt_pipeline as adapt_mod
  - from pipelines.hardcoded.qpe_qiskit_shim import run_qpe_adapter_qiskit
  - from scipy.sparse import csc_matrix as _csc_matrix, diags as _diags
  - from scipy.sparse.linalg import expm_multiply as _expm_multiply
  - from scipy.linalg import expm as _expm_dense
---

Functions:
  - L90: def _ai_log(event: str, **fields: Any) -> None:
  - L110: def _to_ixyz(label_exyz: str) -> str:
  - L114: def _normalize_state(psi: np.ndarray) -> np.ndarray:
  - L121: def _half_filled_particles(num_sites: int) -> tuple[int, int]:
  - L125: def _sector_basis_indices(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> np.ndarray:
  - L159: def _ground_manifold_basis_sector_filtered(
    hmat: np.ndarray,
    *,
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
    energy_tol: float,
) -> tuple[float, np.ndarray]:
  - L194: def _sector_basis_indices_hh(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
    nq_total: int,
) -> np.ndarray:
  - L231: def _ground_manifold_basis_sector_filtered_hh(
    hmat: np.ndarray,
    *,
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
    nq_total: int,
    energy_tol: float,
) -> tuple[float, np.ndarray]:
  - L263: def _orthonormalize_basis_columns(
    basis: np.ndarray,
    *,
    rank_tol: float = 1e-12,
) -> np.ndarray:
  - L279: def _projector_fidelity_from_basis(
    basis_orthonormal: np.ndarray,
    psi: np.ndarray,
) -> float:
  - L294: def _exact_ground_state_sector_filtered(
    hmat: np.ndarray,
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> tuple[float, np.ndarray]:
  - L312: def _exact_energy_sector_filtered(
    hmat: np.ndarray,
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> float:
  - L328: def _pauli_matrix_exyz(label: str) -> np.ndarray:
  - L336: def _collect_hardcoded_terms_exyz(
    h_poly,
) -> tuple[list[str], dict[str, complex]]:
  - L359: def _build_hamiltonian_matrix(coeff_map_exyz: dict[str, complex]) -> np.ndarray:
  - L370: def _compile_pauli_action(label_exyz: str, nq: int) -> CompiledPauliAction:
  - L374: def _apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
  - L378: def _apply_exp_term(
    psi: np.ndarray,
    action: CompiledPauliAction,
    coeff: complex,
    alpha: float,
    tol: float = 1e-12,
) -> np.ndarray:
  - L394: def _evolve_trotter_suzuki2_absolute(
    psi0: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    compiled_actions: dict[str, CompiledPauliAction],
    time_value: float,
    trotter_steps: int,
    *,
    drive_coeff_provider_exyz: Any | None = None,
    t0: float = 0.0,
    time_sampling: str = "midpoint",
    coeff_tol: float = 1e-12,
) -> np.ndarray:
  - L467: def _expectation_hamiltonian(psi: np.ndarray, hmat: np.ndarray) -> float:
  - L471: def _build_drive_matrix_at_time(
    drive_coeff_provider_exyz: Any,
    t_physical: float,
    nq: int,
) -> np.ndarray:
  - L503: def _spin_orbital_bit_index(site: int, spin: int, num_sites: int, ordering: str) -> int:
  - L512: def _site_resolved_number_observables(
    psi: np.ndarray,
    num_sites: int,
    ordering: str,
) -> tuple[np.ndarray, np.ndarray, float]:
  - L537: def _staggered_order(n_total_site: np.ndarray) -> float:
  - L544: def _state_to_amplitudes_qn_to_q0(psi: np.ndarray, cutoff: float = 1e-12) -> dict[str, dict[str, float]]:
  - L555: def _state_from_amplitudes_qn_to_q0(
    amplitudes_qn_to_q0: dict[str, Any],
    nq_total: int,
) -> np.ndarray:
  - L575: def _load_adapt_initial_state(
    adapt_json_path: Path,
    nq_total: int,
) -> tuple[np.ndarray, dict[str, Any]]:
  - L597: def _validate_adapt_metadata(
    *,
    adapt_settings: dict[str, Any],
    args: argparse.Namespace,
    is_hh: bool,
    float_tol: float = 1e-10,
) -> list[str]:
  - L609: def _cmp_scalar(field: str, expected: Any, actual: Any) -> None:
  - L613: def _cmp_float(field: str, expected: float, actual_raw: Any) -> None:
  - L639: def _load_hardcoded_vqe_namespace() -> dict[str, Any]:
  - L665: def _run_hardcoded_vqe(
    *,
    num_sites: int,
    ordering: str,
    boundary: str,
    hopping_t: float,
    onsite_u: float,
    potential_dv: float,
    h_poly: Any,
    reps: int,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    energy_backend: str,
    vqe_progress_every_s: float = 60.0,
    progress_observer: Any | None = None,
    emit_theta_in_progress: bool = False,
    return_best_on_keyboard_interrupt: bool = False,
    early_stop_checker: Any | None = None,
    initial_point: Sequence[float] | np.ndarray | None = None,
    ansatz_name: str,
    spsa_a: float = 0.2,
    spsa_c: float = 0.1,
    spsa_alpha: float = 0.602,
    spsa_gamma: float = 0.101,
    spsa_A: float = 10.0,
    spsa_avg_last: int = 0,
    spsa_eval_repeats: int = 1,
    spsa_eval_agg: str = "mean",
    # --- HH-specific (ignored when problem=hubbard) ---
    problem: str = "hubbard",
    omega0: float = 0.0,
    g_ep: float = 0.0,
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
) -> tuple[dict[str, Any], np.ndarray]:
  - L833: def _vqe_progress_logger(payload: dict[str, Any]) -> None:
  - L963: def _run_internal_adapt_paop(
    *,
    h_poly: Any,
    num_sites: int,
    ordering: str,
    problem: str,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    adapt_pool: str | None,
    adapt_max_depth: int,
    adapt_eps_grad: float,
    adapt_eps_energy: float,
    adapt_maxiter: int,
    adapt_seed: int,
    adapt_allow_repeats: bool,
    adapt_finite_angle_fallback: bool,
    adapt_finite_angle: float,
    adapt_finite_angle_min_improvement: float,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    adapt_disable_hh_seed: bool,
    psi_ref_override: np.ndarray | None = None,
    adapt_reopt_policy: str = "append_only",
    adapt_window_size: int = 3,
    adapt_window_topk: int = 0,
    adapt_full_refit_every: int = 0,
    adapt_final_full_refit: bool = True,
    adapt_beam_live_branches: int = 1,
    adapt_beam_children_per_parent: int | None = None,
    adapt_beam_terminated_keep: int | None = None,
    adapt_continuation_mode: str = "phase3_v1",
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
    phase3_motif_source_json: Path | None = None,
    phase3_symmetry_mitigation_mode: str = "off",
    phase3_enable_rescue: bool = False,
    phase3_lifetime_cost_mode: str = "phase3_v1",
    phase3_runtime_split_mode: str = "off",
) -> tuple[dict[str, Any], np.ndarray]:
  - L1088: def _run_qpe_adapter_qiskit(
    *,
    coeff_map_exyz: dict[str, complex],
    psi_init: np.ndarray,
    eval_qubits: int,
    shots: int,
    seed: int,
) -> dict[str, Any]:
  - L1107: def _reference_terms_for_case(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    ordering: str,
) -> dict[str, float] | None:
  - L1142: def _reference_sanity(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    ordering: str,
    coeff_map_exyz: dict[str, complex],
) -> dict[str, Any]:
  - L1201: def _is_all_z_type(label: str) -> bool:
  - L1216: def _build_drive_diagonal(
    drive_map: dict[str, complex],
    dim: int,
    nq: int,
) -> np.ndarray:
  - L1263: def _evolve_piecewise_exact(
    *,
    psi0: np.ndarray,
    hmat_static: np.ndarray,
    drive_coeff_provider_exyz: Any,
    time_value: float,
    trotter_steps: int,
    t0: float = 0.0,
    time_sampling: str = "midpoint",
) -> np.ndarray:
  - L1424: def _simulate_trajectory(
    *,
    num_sites: int,
    ordering: str,
    psi0_legacy_trot: np.ndarray | None = None,
    psi0_paop_trot: np.ndarray | None = None,
    psi0_hva_trot: np.ndarray | None = None,
    legacy_branch_label: str = "vqe",
    psi0_exact_ref: np.ndarray,
    fidelity_subspace_basis_v0: np.ndarray,
    fidelity_subspace_energy_tol: float,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    trotter_steps: int,
    t_final: float,
    num_times: int,
    suzuki_order: int,
    drive_coeff_provider_exyz: Any | None = None,
    drive_t0: float = 0.0,
    drive_time_sampling: str = "midpoint",
    exact_steps_multiplier: int = 1,
    propagator: str = "cfqm4",
    cfqm_stage_exp: str = "expm_multiply_sparse",
    cfqm_coeff_drop_abs_tol: float = 0.0,
    cfqm_normalize: bool = False,
    psi0_ansatz_trot: np.ndarray | None = None,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
  - L1546: def _exact_from_initial(psi0_branch: np.ndarray) -> np.ndarray:
  - L1560: def _trotter_from_initial(psi0_branch: np.ndarray) -> np.ndarray:
  - L1691: def _branch_observables(psi_branch: np.ndarray) -> dict[str, Any]:
  - L1821: def _write_pipeline_pdf(pdf_path: Path, payload: dict[str, Any], run_command: str) -> None:
  - L1829: def arr(key: str) -> np.ndarray:
  - L1832: def arr_optional(key: str, fallback: np.ndarray | None = None) -> np.ndarray:
  - L1843: def mat(key: str) -> np.ndarray:
  - L1854: def mat_optional(key: str, fallback: np.ndarray) -> np.ndarray:
  - L1868: def _plot_density_surface(
        ax: Any,
        data: np.ndarray,
        *,
        title: str,
        zlim: tuple[float, float],
        cmap: str,
    ) -> None:
  - L1894: def _plot_lane_3d(
        ax: Any,
        *,
        series: list[np.ndarray],
        labels: list[str],
        colors: list[str],
        title: str,
        zlabel: str,
    ) -> None:
  - L2621: def parse_args() -> argparse.Namespace:
  - L2967: def main() -> None:

Global vars:
  - REPO_ROOT
  - plt
  - PdfPages
  - PAULI_MATS
  - EXACT_LABEL
  - EXACT_METHOD
  - _EXPM_SPARSE_MIN_DIM
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/exact_bench/hh_fixed_seed_local_checkpoint_fit.py
Imports:
  - import argparse
  - import csv
  - import json
  - import math
  - from dataclasses import asdict, dataclass
  - from datetime import datetime, timezone
  - from pathlib import Path
  - import sys
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
  - from src.quantum.time_propagation import (
    CheckpointFitConfig,
    LocalPauliAnsatzSpec,
    build_local_pauli_ansatz_terms,
    default_chain_edges,
    expectation_total_hamiltonian,
    fit_checkpoint_trajectory,
    run_exact_driven_reference,
)
  - from src.quantum.drives_time_potential import build_gaussian_sinusoid_density_drive
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
      - single_axes
      - entangler_axes
      - reps_list
      - optimizer_method
      - optimizer_maxiter
      - optimizer_gtol
      - optimizer_ftol
      - angle_bound
      - param_shift
      - drive_A
      - drive_omega
      - drive_tbar
      - drive_phi
      - drive_pattern
      - drive_t0
      - drive_time_sampling
      - skip_pdf

Functions:
  - L88: def _now_utc() -> str:
  - L92: def _jsonable(value: Any) -> Any:
  - L108: def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
  - L113: def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
  - L131: def _parse_int_tuple(raw: str) -> tuple[int, ...]:
  - L151: def _parse_axis_tuple(raw: str, *, allowed: set[str]) -> tuple[str, ...]:
  - L170: def _collect_hardcoded_terms_exyz(h_poly: Any) -> tuple[list[str], dict[str, complex]]:
  - L184: def _pauli_matrix_exyz(label: str) -> np.ndarray:
  - L197: def _build_hamiltonian_matrix(coeff_map_exyz: Mapping[str, complex]) -> np.ndarray:
  - L208: def _build_drive_provider(
    *,
    num_sites: int,
    nq_total: int,
    ordering: str,
    cfg: SweepConfig,
) -> tuple[Any, dict[str, Any]]:
  - L238: def _transpile_theta_history(
    *,
    terms: Sequence[Any],
    theta_history: np.ndarray,
    num_qubits: int,
    cfg: SweepConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
  - L283: def _best_row(rows: Sequence[Mapping[str, Any]], *, key: str) -> dict[str, Any] | None:
  - L310: def _candidate_row(
    *,
    reps: int,
    label: str,
    times: np.ndarray,
    exact_states: Sequence[np.ndarray],
    exact_energies: np.ndarray,
    fit_result: Any,
    terms: Sequence[Any],
    hmat_static: np.ndarray,
    drive_provider: Any,
    cfg: SweepConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
  - L380: def write_summary_pdf(
    *,
    cfg: SweepConfig,
    payload: Mapping[str, Any],
    candidate_details: Mapping[str, Mapping[str, Any]],
    seed_payload: Mapping[str, Any],
) -> None:
  - L492: def run_sweep(cfg: SweepConfig) -> dict[str, Any]:
  - L621: def build_cli_parser() -> argparse.ArgumentParser:
  - L661: def parse_args(argv: Sequence[str] | None = None) -> SweepConfig:
  - L715: def main(argv: Sequence[str] | None = None) -> int:

Global vars:
  - REPO_ROOT
  - _DEFAULT_FIXED_SEED_JSON
  - _DEFAULT_OUTPUT_JSON
  - _DEFAULT_OUTPUT_CSV
  - _DEFAULT_OUTPUT_PDF
  - _DEFAULT_RUN_ROOT
  - _DEFAULT_BACKEND_NAME
  - _DEFAULT_REPS
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/drives_time_potential.py
Imports:
  - import math
  - from dataclasses import dataclass
  - from typing import Callable, Dict, List, Optional, Sequence
  - import numpy as np
  - from src.quantum.hubbard_latex_python_pairs import SPIN_DN, SPIN_UP, mode_index
---
Classes:
  - GaussianSinusoidSitePotential
    Methods:
      - L235: def v_scalar(self, t: float) -> float:
      - L245: def v_sites(self, t: float) -> np.ndarray:
    Properties:
      - weights
      - A
      - omega
      - tbar
      - phi
  - DensityDriveTemplate
    Methods:
      - L280: def build(
        cls,
        *,
        n_sites: int,
        nq_total: int,
        indexing: str,
        electron_qubit_offset: int = 0,
    ) -> "DensityDriveTemplate":
      - L317: def labels_exyz(self, *, include_identity: bool = False) -> List[str]:
      - L324: def z_label(self, site: int, spin: int) -> str:
    Properties:
      - n_sites
      - nq_total
      - indexing
      - electron_qubit_offset
      - z_labels_site_spin
      - ordered_z_labels
      - identity_label
  - TimeDependentOnsiteDensityDrive
    Methods:
      - L351: def coeff_map_exyz(self, t: float) -> Dict[str, float]:
    Properties:
      - template
      - site_potential
      - include_identity
      - coeff_tol

Functions:
  - L82: def reference_method_name(time_sampling: str) -> str:
  - L119: def gaussian_sinusoid_waveform(
    t: float,
    *,
    A: float,
    omega: float,
    tbar: float,
    phi: float = 0.0,
) -> float:
  - L137: def evaluate_drive_waveform(
    times: Sequence[float] | np.ndarray,
    drive_config: Dict[str, float | int | str | None],
    amplitude: float,
) -> np.ndarray:
  - L177: def default_spatial_weights(
    n_sites: int,
    *,
    mode: str,
    custom: Optional[Sequence[float]] = None,
) -> np.ndarray:
  - L255: def _z_label_exyz(*, nq_total: int, qubit_index: int) -> str:
  - L385: def build_gaussian_sinusoid_density_drive(
    *,
    n_sites: int,
    nq_total: int,
    indexing: str,
    A: float,
    omega: float,
    tbar: float,
    phi: float = 0.0,
    pattern_mode: str = "staggered",
    custom_weights: Optional[Sequence[float]] = None,
    include_identity: bool = False,
    electron_qubit_offset: int = 0,
    coeff_tol: float = 0.0,
) -> TimeDependentOnsiteDensityDrive:

Global vars:
  - REFERENCE_METHOD_NAMES
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_hh_staged_noiseless_workflow.py
Imports:
  - import json
  - from dataclasses import replace
  - from pathlib import Path
  - import sys
  - import numpy as np
  - import pytest
  - import pipelines.hardcoded.hh_staged_workflow as wf
  - from pipelines.hardcoded.hh_staged_noiseless import parse_args
  - from pipelines.hardcoded.hh_staged_workflow import resolve_staged_hh_config
---
Classes:
  - _FakeAnsatz
    Methods:
      - L334: def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
  - _FakeAnsatz
    Methods:
      - L430: def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
  - _FakeAnsatz
    Methods:
      - L607: def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
  - _FakeDriveTemplate
    Methods:
      - L1049: def labels_exyz(self, include_identity: bool = False):
  - _FakeDrive
    Methods:
      - L1053: def __init__(self):

Functions:
  - L20: def _basis(dim: int, idx: int) -> np.ndarray:
  - L26: def _amplitudes_qn_to_q0(psi: np.ndarray) -> dict[str, dict[str, float]]:
  - L36: def test_resolve_staged_defaults_from_run_guide_formulae() -> None:
  - L74: def test_replay_and_dynamics_flags_roundtrip() -> None:
  - L83: def test_warm_ansatz_override_is_resolved_and_retagged() -> None:
  - L93: def test_seed_refine_cli_fields_roundtrip() -> None:
  - L119: def test_adapt_beam_capacity_cli_fields_roundtrip() -> None:
  - L141: def test_adapt_drop_policy_overrides_roundtrip() -> None:
  - L166: def test_warm_checkpoint_cli_fields_roundtrip(tmp_path: Path) -> None:
  - L222: def test_fixed_final_state_cli_fields_roundtrip(tmp_path: Path) -> None:
  - L252: def test_nondefault_sector_override_rejected_cleanly() -> None:
  - L259: def test_staged_hh_parse_rejects_unsupported_warm_and_final_methods() -> None:
  - L278: def test_staged_hh_parse_accepts_powell_methods() -> None:
  - L302: def test_underparameterized_override_rejected_without_smoke_flag() -> None:
  - L325: def test_warm_stage_checkpoint_cutover_and_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  - L337: def _fake_run_hardcoded_vqe(**kwargs):
  - L419: def test_warm_stage_resume_from_precutoff_checkpoint_continues_optimization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  - L523: def test_warm_stage_handoff_from_checkpoint_skips_warm_optimization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  - L532: def _unexpected_warm_run(**kwargs):
  - L598: def test_warm_stage_below_cutoff_still_writes_cutover_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  - L656: def test_run_stage_pipeline_uses_fixed_final_state_and_skips_prep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  - L752: def test_run_stage_pipeline_inserts_seed_refine_before_adapt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  - L797: def _fake_seed_refine(_cfg, *, h_poly, psi_ref, exact_filtered_energy):
  - L814: def _fake_run_adapt(**kwargs):
  - L832: def _fake_replay_run(cfg, diagnostics_out=None):
  - L901: def test_seed_refine_failure_aborts_before_adapt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  - L965: def test_workflow_runs_matched_family_replay_and_static_plus_drive_profiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  - L1011: def _fake_replay_run(cfg):
  - L1026: def _fake_simulate_trajectory(**kwargs):
  - L1130: def test_workflow_skips_replay_and_dynamics_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  - L1215: def test_infer_handoff_adapt_pool_prefers_selected_and_rejects_mixed() -> None:
  - L1237: def test_infer_handoff_adapt_pool_rejects_mixed_operator_labels_without_metadata() -> None:
  - L1255: def test_infer_handoff_adapt_pool_rejects_mixed_split_termwise_operator_labels() -> None:
  - L1273: def test_infer_handoff_adapt_pool_prefers_operator_labels_over_motif_family() -> None:
  - L1296: def test_infer_handoff_adapt_pool_uses_motif_family_when_selected_missing() -> None:

Global vars:
  - REPO_ROOT
---


File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_hh_vqe_from_adapt_family_seed.py
Imports:
  - from dataclasses import dataclass
  - from types import SimpleNamespace
  - import numpy as np
  - import pytest
  - from pipelines.hardcoded.hh_vqe_from_adapt_family import (
    PoolTermwiseAnsatz,
    REPLAY_CONTRACT_VERSION,
    REPLAY_SEED_POLICIES,
    build_family_ansatz_context,
    _build_full_meta_replay_terms_sparse,
    _build_replay_seed_theta,
    _build_replay_seed_theta_policy,
    _build_replay_terms_from_adapt_labels,
    _expand_product_labels_and_theta_for_family,
    _extract_canonical_label_family,
    _extract_adapt_operator_theta_sequence,
    _extract_replay_contract,
    _infer_handoff_state_kind,
)
  - from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
---
Classes:
  - _DummyTerm
    Properties:
      - label
  - TestInferHandoffStateKind
    Methods:
      - L399: def test_explicit_prepared_state(self) -> None:
      - L405: def test_explicit_reference_state(self) -> None:
      - L411: def test_infer_hf_as_reference(self) -> None:
      - L417: def test_infer_exact_as_reference(self) -> None:
      - L423: def test_infer_adapt_vqe_as_prepared(self) -> None:
      - L429: def test_infer_a_probe_final_as_prepared(self) -> None:
      - L435: def test_infer_b_medium_final_as_prepared(self) -> None:
      - L441: def test_infer_warm_start_hva_as_prepared(self) -> None:
      - L447: def test_ambiguous_when_no_initial_state(self) -> None:
      - L452: def test_ambiguous_for_unknown_source(self) -> None:
      - L458: def test_explicit_overrides_source(self) -> None:
  - TestBuildReplaySeedThetaPolicy
    Methods:
      - L476: def test_tile_adapt(self) -> None:
      - L484: def test_scaffold_plus_zero(self) -> None:
      - L494: def test_residual_only(self) -> None:
      - L502: def test_auto_prepared_gives_residual_only(self) -> None:
      - L510: def test_auto_reference_gives_scaffold_plus_zero(self) -> None:
      - L520: def test_auto_seed_semantics_are_mode_independent(self) -> None:
      - L533: def test_auto_ambiguous_raises(self) -> None:
      - L540: def test_seed_length_matches_adapt_depth_times_reps(self) -> None:
      - L549: def test_scaffold_plus_zero_reps_1_equals_adapt_theta(self) -> None:
      - L557: def test_tile_adapt_reps_1_equals_adapt_theta(self) -> None:
      - L565: def test_residual_only_reps_1_is_all_zeros(self) -> None:

Functions:
  - L32: def test_build_family_ansatz_context_materializes_explicit_product_family() -> None:
  - L86: def test_extract_adapt_operator_theta_sequence_valid() -> None:
  - L98: def test_extract_adapt_operator_theta_sequence_rejects_missing_block() -> None:
  - L103: def test_extract_adapt_operator_theta_sequence_rejects_missing_operators() -> None:
  - L109: def test_extract_adapt_operator_theta_sequence_rejects_missing_optimal_point() -> None:
  - L115: def test_extract_adapt_operator_theta_sequence_rejects_length_mismatch() -> None:
  - L126: def test_extract_adapt_operator_theta_sequence_rejects_nonfinite_theta() -> None:
  - L137: def test_build_replay_terms_preserves_operator_order_and_duplicates() -> None:
  - L145: def test_build_replay_terms_rejects_unknown_label() -> None:
  - L151: def test_build_replay_terms_reconstructs_runtime_split_children_from_payload() -> None:
  - L184: def test_sparse_full_meta_replay_terms_reconstruct_runtime_split_children_from_payload() -> None:
  - L251: def test_build_replay_seed_theta_tiled_and_npar_matches_adapt_depth_times_reps() -> None:
  - L262: def test_extract_canonical_label_family_maps_uccsd_otimes_product_labels() -> None:
  - L270: def test_extract_canonical_label_family_maps_uccsd_otimes_seq2p_labels() -> None:
  - L278: def test_expand_product_labels_and_theta_for_seq2p_replay() -> None:
  - L296: def test_replay_contract_parses_match_adapt_mapping() -> None:
  - L326: def test_replay_contract_parses_map_with_resolved_only() -> None:
  - L347: def test_replay_contract_rejects_fallback_used_without_fallback_family() -> None:
  - L369: def test_replay_contract_rejects_malformed() -> None:
  - L386: def test_replay_contract_version_is_2() -> None:
  - L390: def test_replay_seed_policies_set() -> None:
---

</file_map>
<file_contents>
File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_staged_workflow.py
(lines 1-380: Staged HH workflow imports and configuration dataclasses, including AdaptConfig, ReplayConfig, DynamicsConfig, and StageExecutionResult, which define repo-native orchestration boundaries.)
```py
#!/usr/bin/env python3
"""Shared staged Hubbard-Holstein noiseless workflow orchestration.

This module keeps the stage-chain logic out of the existing monolithic
entrypoints. It reuses the production hardcoded primitives instead of
re-implementing warm-start VQE, ADAPT, replay, or time dynamics.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.reports.pdf_utils import (
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
from docs.reports.qiskit_circuit_report import (
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
from pipelines.hardcoded import adapt_pipeline as adapt_mod
from pipelines.hardcoded import hh_vqe_from_adapt_family as replay_mod
from pipelines.hardcoded import hubbard_pipeline as hc_pipeline

_PREPARED_STATE = replay_mod._PREPARED_STATE
_REFERENCE_STATE = replay_mod._REFERENCE_STATE
from pipelines.hardcoded.handoff_state_bundle import (
    HandoffStateBundleConfig,
    write_handoff_state_bundle,
)
from src.quantum.drives_time_potential import (
    build_gaussian_sinusoid_density_drive,
    reference_method_name,
)
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinLayerwiseAnsatz,
    HubbardHolsteinPhysicalTermwiseAnsatz,
    HubbardHolsteinTermwiseAnsatz,
    exact_ground_energy_sector_hh,
)


_ALLOWED_NOISELESS_METHODS = ("suzuki2", "cfqm4", "cfqm6", "piecewise_exact")


@dataclass(frozen=True)
class PhysicsConfig:
    L: int
    t: float
    u: float
    dv: float
    omega0: float
    g_ep: float
    n_ph_max: int
    boson_encoding: str
    ordering: str
    boundary: str
    sector_n_up: int
    sector_n_dn: int


@dataclass(frozen=True)
class WarmStartConfig:
    ansatz_name: str
    reps: int
    restarts: int
    maxiter: int
    method: str
    seed: int
    progress_every_s: float
    energy_backend: str
    spsa_a: float
    spsa_c: float
    spsa_alpha: float
    spsa_gamma: float
    spsa_A: float
    spsa_avg_last: int
    spsa_eval_repeats: int
    spsa_eval_agg: str


@dataclass(frozen=True)
class SeedRefineConfig:
    family: str | None
    reps: int
    maxiter: int
    optimizer: str


@dataclass(frozen=True)
class AdaptConfig:
    pool: str | None
    continuation_mode: str
    max_depth: int
    maxiter: int
    eps_grad: float
    eps_energy: float
    drop_floor: float | None
    drop_patience: int | None
    drop_min_depth: int | None
    grad_floor: float | None
    seed: int
    inner_optimizer: str
    allow_repeats: bool
    finite_angle_fallback: bool
    finite_angle: float
    finite_angle_min_improvement: float
    disable_hh_seed: bool
    reopt_policy: str
    window_size: int
    window_topk: int
    full_refit_every: int
    final_full_refit: bool
    beam_live_branches: int
    beam_children_per_parent: int | None
    beam_terminated_keep: int | None
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str
    spsa_a: float
    spsa_c: float
    spsa_alpha: float
    spsa_gamma: float
    spsa_A: float
    spsa_avg_last: int
    spsa_eval_repeats: int
    spsa_eval_agg: str
    spsa_callback_every: int
    spsa_progress_every_s: float
    phase1_lambda_F: float
    phase1_lambda_compile: float
    phase1_lambda_measure: float
    phase1_lambda_leak: float
    phase1_score_z_alpha: float
    phase1_probe_max_positions: int
    phase1_plateau_patience: int
    phase1_trough_margin_ratio: float
    phase1_prune_enabled: bool
    phase1_prune_fraction: float
    phase1_prune_max_candidates: int
    phase1_prune_max_regression: float
    phase3_motif_source_json: Path | None
    phase3_symmetry_mitigation_mode: str
    phase3_enable_rescue: bool
    phase3_lifetime_cost_mode: str
    phase3_runtime_split_mode: str


@dataclass(frozen=True)
class ReplayConfig:
    enabled: bool
    generator_family: str
    fallback_family: str
    legacy_paop_key: str
    replay_seed_policy: str
    continuation_mode: str
    reps: int
    restarts: int
    maxiter: int
    method: str
    seed: int
    energy_backend: str
    progress_every_s: float
    wallclock_cap_s: int
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str
    spsa_a: float
    spsa_c: float
    spsa_alpha: float
    spsa_gamma: float
    spsa_A: float
    spsa_avg_last: int
    spsa_eval_repeats: int
    spsa_eval_agg: str
    replay_freeze_fraction: float
    replay_unfreeze_fraction: float
    replay_full_fraction: float
    replay_qn_spsa_refresh_every: int
    replay_qn_spsa_refresh_mode: str
    phase3_symmetry_mitigation_mode: str


@dataclass(frozen=True)
class DynamicsConfig:
    enabled: bool
    methods: tuple[str, ...]
    t_final: float
    num_times: int
    trotter_steps: int
    exact_steps_multiplier: int
    fidelity_subspace_energy_tol: float
    cfqm_stage_exp: str
    cfqm_coeff_drop_abs_tol: float
    cfqm_normalize: bool
    enable_drive: bool
    drive_A: float
    drive_omega: float
    drive_tbar: float
    drive_phi: float
    drive_pattern: str
    drive_custom_s: str | None
    drive_include_identity: bool
    drive_time_sampling: str
    drive_t0: float


@dataclass(frozen=True)
class FixedFinalStateConfig:
    json_path: Path
    strict_match: bool


@dataclass(frozen=True)
class CircuitMetricConfig:
    backend_name: str | None
    use_fake_backend: bool
    optimization_level: int
    seed_transpiler: int


@dataclass(frozen=True)
class WarmCheckpointConfig:
    stop_energy: float | None
    stop_delta_abs: float | None
    state_export_dir: Path
    state_export_prefix: str
    resume_from_warm_checkpoint: Path | None
    handoff_from_warm_checkpoint: Path | None


@dataclass(frozen=True)
class ArtifactConfig:
    tag: str
    output_json: Path
    output_pdf: Path
    handoff_json: Path
    warm_checkpoint_json: Path
    warm_cutover_json: Path
    replay_output_json: Path
    replay_output_csv: Path
    replay_output_md: Path
    replay_output_log: Path
    workflow_log: Path
    skip_pdf: bool


@dataclass(frozen=True)
class GateConfig:
    ecut_1: float
    ecut_2: float


@dataclass(frozen=True)
class StagedHHConfig:
    physics: PhysicsConfig
    warm_start: WarmStartConfig
    seed_refine: SeedRefineConfig
    adapt: AdaptConfig
    replay: ReplayConfig
    dynamics: DynamicsConfig
    fixed_final_state: FixedFinalStateConfig | None
    circuit_metrics: CircuitMetricConfig
    warm_checkpoint: WarmCheckpointConfig
    artifacts: ArtifactConfig
    gates: GateConfig
    smoke_test_intentionally_weak: bool = False
    default_provenance: dict[str, str] = field(default_factory=dict)
    external_noise_handle: dict[str, Any] | None = None


@dataclass
class StageExecutionResult:
    h_poly: Any
    hmat: np.ndarray
    ordered_labels_exyz: list[str]
    coeff_map_exyz: dict[str, complex]
    nq_total: int
    psi_hf: np.ndarray
    psi_warm: np.ndarray
    psi_adapt: np.ndarray
    psi_final: np.ndarray
    warm_payload: dict[str, Any]
    adapt_payload: dict[str, Any]
    replay_payload: dict[str, Any]
    psi_seed_refine: np.ndarray | None = None
    seed_refine_payload: dict[str, Any] | None = None
    fixed_final_state_import: dict[str, Any] | None = None
    warm_circuit_context: dict[str, Any] | None = None
    adapt_circuit_context: dict[str, Any] | None = None
    replay_circuit_context: dict[str, Any] | None = None


"""
Δ_rel(E, E_ref) = |E - E_ref| / max(|E_ref|, 1e-14)
"""
def _relative_error_abs(value: float, reference: float) -> float:
    return float(abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-14))


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _append_workflow_log(cfg: StagedHHConfig, event: str, **fields: Any) -> None:
    payload = {
        "ts_utc": _now_utc(),
        "event": str(event),
        **_jsonable(fields),
    }
    log_path = Path(cfg.artifacts.workflow_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _handoff_bundle_cfg(cfg: StagedHHConfig) -> HandoffStateBundleConfig:
    return HandoffStateBundleConfig(
        L=int(cfg.physics.L),
        t=float(cfg.physics.t),
        U=float(cfg.physics.u),
        dv=float(cfg.physics.dv),
        omega0=float(cfg.physics.omega0),
        g_ep=float(cfg.physics.g_ep),
        n_ph_max=int(cfg.physics.n_ph_max),
        boson_encoding=str(cfg.physics.boson_encoding),
        ordering=str(cfg.physics.ordering),

```

(lines 2349-3139: Adapt handoff writing, run_stage_pipeline orchestration, post-replay noiseless profile execution, and stage circuit/report artifact assembly showing how staged HH workflows persist and consume continuation metadata.)
```py
def _write_adapt_handoff(
    cfg: StagedHHConfig,
    adapt_payload: Mapping[str, Any],
    psi_adapt: np.ndarray,
    *,
    seed_provenance: Mapping[str, Any] | None = None,
) -> None:
    exact_energy = float(adapt_payload.get("exact_gs_energy", float("nan")))
    energy = float(adapt_payload.get("energy", float("nan")))
    continuation_meta = _handoff_continuation_meta(adapt_payload)
    handoff_adapt_pool, handoff_adapt_pool_source = _infer_handoff_adapt_pool(cfg, adapt_payload)
    stage_chain = ["hf_reference", "warm_start_hva"]
    if cfg.seed_refine.family is not None:
        stage_chain.append("seed_refine_vqe")
    stage_chain.extend(["adapt_vqe", "matched_family_replay"])
    write_handoff_state_bundle(
        path=cfg.artifacts.handoff_json,
        psi_state=np.asarray(psi_adapt, dtype=complex).reshape(-1),
        cfg=_handoff_bundle_cfg(cfg),
        source="adapt_vqe",
        exact_energy=float(exact_energy),
        energy=float(energy),
        delta_E_abs=float(adapt_payload.get("abs_delta_e", abs(energy - exact_energy))),
        relative_error_abs=float(_relative_error_abs(energy, exact_energy)),
        meta={
            "pipeline": "hh_staged_noiseless",
            "workflow_tag": str(cfg.artifacts.tag),
            "stage_chain": stage_chain,
        },
        adapt_operators=[str(x) for x in adapt_payload.get("operators", [])],
        adapt_optimal_point=[float(x) for x in adapt_payload.get("optimal_point", [])],
        adapt_pool_type=handoff_adapt_pool,
        settings_adapt_pool=handoff_adapt_pool,
        handoff_state_kind="prepared_state",
        continuation_mode=str(continuation_meta.get("continuation_mode", cfg.adapt.continuation_mode)),
        continuation_scaffold=continuation_meta.get("continuation_scaffold"),
        replay_contract=_build_replay_contract(
            cfg,
            handoff_adapt_pool=handoff_adapt_pool,
            handoff_adapt_pool_source=handoff_adapt_pool_source,
        ),
        optimizer_memory=continuation_meta.get("optimizer_memory"),
        selected_generator_metadata=continuation_meta.get("selected_generator_metadata"),
        generator_split_events=continuation_meta.get("generator_split_events"),
        motif_library=continuation_meta.get("motif_library"),
        motif_usage=continuation_meta.get("motif_usage"),
        symmetry_mitigation=continuation_meta.get("symmetry_mitigation"),
        rescue_history=continuation_meta.get("rescue_history"),
        prune_summary=continuation_meta.get("prune_summary"),
        pre_prune_scaffold=continuation_meta.get("pre_prune_scaffold"),
        replay_contract_hint={
            "generator_family": str(cfg.replay.generator_family),
            "fallback_family": str(cfg.replay.fallback_family),
            "replay_seed_policy": str(cfg.replay.replay_seed_policy),
            "replay_continuation_mode": str(cfg.replay.continuation_mode),
        },
        seed_provenance=(dict(seed_provenance) if isinstance(seed_provenance, Mapping) else None),
    )


def _staged_ansatz_manifest(cfg: StagedHHConfig) -> str:
    parts = [f"warm: {cfg.warm_start.ansatz_name}"]
    if cfg.seed_refine.family is not None:
        parts.append(f"seed refine: {cfg.seed_refine.family}")
    parts.append(f"ADAPT: {cfg.adapt.continuation_mode}")
    parts.append(
        "final: matched-family replay"
        if bool(cfg.replay.enabled)
        else "final: replay disabled"
    )
    return "; ".join(parts)


def _build_hh_warm_ansatz(cfg: StagedHHConfig) -> Any:
    common_kwargs = {
        "dims": int(cfg.physics.L),
        "J": float(cfg.physics.t),
        "U": float(cfg.physics.u),
        "omega0": float(cfg.physics.omega0),
        "g": float(cfg.physics.g_ep),
        "n_ph_max": int(cfg.physics.n_ph_max),
        "boson_encoding": str(cfg.physics.boson_encoding),
        "reps": int(cfg.warm_start.reps),
        "repr_mode": "JW",
        "indexing": str(cfg.physics.ordering),
        "pbc": (str(cfg.physics.boundary).strip().lower() == "periodic"),
    }
    ansatz_name = str(cfg.warm_start.ansatz_name).strip().lower()
    if ansatz_name == "hh_hva":
        return HubbardHolsteinLayerwiseAnsatz(**common_kwargs)
    if ansatz_name == "hh_hva_tw":
        return HubbardHolsteinTermwiseAnsatz(**common_kwargs)
    if ansatz_name == "hh_hva_ptw":
        return HubbardHolsteinPhysicalTermwiseAnsatz(**common_kwargs)
    raise ValueError(f"Unsupported staged HH warm ansatz {cfg.warm_start.ansatz_name!r}.")


def _assemble_stage_circuit_contexts(
    *,
    cfg: StagedHHConfig,
    psi_hf: np.ndarray,
    warm_payload: Mapping[str, Any],
    adapt_diagnostics: Mapping[str, Any] | None,
    replay_diagnostics: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any] | None]:
    warm_ctx: dict[str, Any] | None = None
    adapt_ctx: dict[str, Any] | None = None
    replay_ctx: dict[str, Any] | None = None

    warm_theta_raw = warm_payload.get("optimal_point", None)
    if isinstance(warm_theta_raw, Sequence) and not isinstance(warm_theta_raw, (str, bytes)):
        warm_theta = np.asarray([float(x) for x in warm_theta_raw], dtype=float)
        if int(warm_theta.size) > 0:
            warm_ctx = {
                "ansatz": _build_hh_warm_ansatz(cfg),
                "theta": np.asarray(warm_theta, dtype=float).copy(),
                "reference_state": np.asarray(psi_hf, dtype=complex).reshape(-1).copy(),
                "num_qubits": int(round(math.log2(int(np.asarray(psi_hf).size)))),
                "ansatz_name": str(cfg.warm_start.ansatz_name),
            }

    if isinstance(adapt_diagnostics, Mapping) and adapt_diagnostics:
        adapt_ctx = {
            "selected_ops": list(adapt_diagnostics.get("selected_ops", [])),
            "theta": np.asarray(adapt_diagnostics.get("theta", []), dtype=float).copy(),
            "reference_state": np.asarray(adapt_diagnostics.get("reference_state"), dtype=complex).reshape(-1).copy(),
            "num_qubits": int(adapt_diagnostics.get("num_qubits", 0)),
            "pool_type": str(adapt_diagnostics.get("pool_type", cfg.adapt.pool or cfg.adapt.continuation_mode)),
            "continuation_mode": str(adapt_diagnostics.get("continuation_mode", cfg.adapt.continuation_mode)),
        }

    if isinstance(replay_diagnostics, Mapping) and replay_diagnostics:
        replay_ctx = {
            "ansatz": replay_diagnostics.get("ansatz"),
            "theta": np.asarray(replay_diagnostics.get("best_theta", []), dtype=float).copy(),
            "seed_theta": np.asarray(replay_diagnostics.get("seed_theta", []), dtype=float).copy(),
            "reference_state": np.asarray(replay_diagnostics.get("reference_state"), dtype=complex).reshape(-1).copy(),
            "num_qubits": int(replay_diagnostics.get("num_qubits", 0)),
            "family_info": dict(replay_diagnostics.get("family_info", {})),
            "handoff_state_kind": str(replay_diagnostics.get("handoff_state_kind", "prepared_state")),
            "provenance_source": str(replay_diagnostics.get("provenance_source", "explicit")),
            "resolved_seed_policy": str(replay_diagnostics.get("resolved_seed_policy", cfg.replay.replay_seed_policy)),
        }

    return {
        "warm_circuit_context": warm_ctx,
        "adapt_circuit_context": adapt_ctx,
        "replay_circuit_context": replay_ctx,
    }


def _workflow_stage_chain(cfg: StagedHHConfig, *, fixed_mode: bool) -> list[str]:
    if fixed_mode:
        chain = ["hf_reference", "fixed_final_state_import"]
    else:
        chain = ["hf_reference", "warm_start_hva"]
        if cfg.seed_refine.family is not None:
            chain.append("seed_refine_vqe")
        chain.append("adapt_vqe")
        if bool(cfg.replay.enabled):
            chain.append("matched_family_replay")
    if bool(cfg.dynamics.enabled):
        chain.append("final_only_noiseless_dynamics")
    return chain


def _terminal_reference_energy(stage_result: StageExecutionResult, cfg: StagedHHConfig) -> tuple[float, str]:
    if bool(cfg.replay.enabled) and not bool(stage_result.replay_payload.get("skipped", False)):
        replay_exact = float(stage_result.replay_payload.get("exact", {}).get("E_exact_sector", float("nan")))
        if math.isfinite(replay_exact):
            return replay_exact, "stage_pipeline.conventional_replay.exact_energy"
    adapt_exact = float(stage_result.adapt_payload.get("exact_gs_energy", float("nan")))
    return adapt_exact, "stage_pipeline.adapt_vqe.exact_energy"


def run_stage_pipeline(cfg: StagedHHConfig) -> StageExecutionResult:
    h_poly, hmat, ordered_labels_exyz, coeff_map_exyz, psi_hf = _build_hh_context(cfg)
    adapt_diagnostics: dict[str, Any] = {}
    replay_diagnostics: dict[str, Any] = {}
    _append_workflow_log(
        cfg,
        "stage_pipeline_start",
        tag=str(cfg.artifacts.tag),
        output_json=str(cfg.artifacts.output_json),
        warm_checkpoint_json=str(cfg.artifacts.warm_checkpoint_json),
        warm_cutover_json=str(cfg.artifacts.warm_cutover_json),
    )
    if cfg.fixed_final_state is not None:
        source_json = Path(cfg.fixed_final_state.json_path)
        raw_payload = json.loads(source_json.read_text(encoding="utf-8"))
        psi_final, fixed_import, warm_payload, adapt_payload, replay_payload = _build_fixed_final_state_import(
            cfg,
            source_json=source_json,
            raw_payload=raw_payload,
            nq_total=_hh_nq_total(cfg.physics.L, cfg.physics.n_ph_max, cfg.physics.boson_encoding),
        )
        _write_fixed_final_state_sidecars(
            cfg,
            psi_final=np.asarray(psi_final, dtype=complex).reshape(-1),
            fixed_import=fixed_import,
            replay_payload=replay_payload,
        )
        _append_workflow_log(
            cfg,
            "fixed_final_state_import",
            source_json=str(source_json),
            strict_match=bool(cfg.fixed_final_state.strict_match),
            mismatch_count=int(len(fixed_import.get("mismatches", []))),
        )
        return StageExecutionResult(
            h_poly=h_poly,
            hmat=np.asarray(hmat, dtype=complex),
            ordered_labels_exyz=list(ordered_labels_exyz),
            coeff_map_exyz=dict(coeff_map_exyz),
            nq_total=int(_hh_nq_total(cfg.physics.L, cfg.physics.n_ph_max, cfg.physics.boson_encoding)),
            psi_hf=np.asarray(psi_hf, dtype=complex).reshape(-1),
            psi_warm=np.asarray(psi_final, dtype=complex).reshape(-1),
            psi_adapt=np.asarray(psi_final, dtype=complex).reshape(-1),
            psi_final=np.asarray(psi_final, dtype=complex).reshape(-1),
            warm_payload=dict(warm_payload),
            adapt_payload=dict(adapt_payload),
            replay_payload=dict(replay_payload),
            fixed_final_state_import=dict(fixed_import),
            warm_circuit_context=None,
            adapt_circuit_context=None,
            replay_circuit_context=None,
        )

    warm_payload, psi_warm, warm_seed_json = _run_warm_start_stage(
        cfg,
        h_poly=h_poly,
        psi_hf=np.asarray(psi_hf, dtype=complex).reshape(-1),
    )
    adapt_seed_json = Path(warm_seed_json)
    psi_seed_refine: np.ndarray | None = None
    seed_refine_payload: dict[str, Any] | None = None
    if cfg.seed_refine.family is not None:
        seed_refine_payload, psi_seed_refine, adapt_seed_json = _run_seed_refine_stage(
            cfg,
            h_poly=h_poly,
            psi_ref=np.asarray(psi_warm, dtype=complex).reshape(-1),
            exact_filtered_energy=float(warm_payload.get("exact_filtered_energy", float("nan"))),
        )
    _append_workflow_log(
        cfg,
        "adapt_seed_checkpoint_selected",
        checkpoint_json=str(adapt_seed_json),
        source_stage=("seed_refine_vqe" if seed_refine_payload is not None else "warm_start_hva"),
        energy=float(
            seed_refine_payload.get("vqe", {}).get("energy", warm_payload.get("energy", float("nan")))
            if isinstance(seed_refine_payload, Mapping)
            else warm_payload.get("energy", float("nan"))
        ),
        exact_filtered_energy=float(warm_payload.get("exact_filtered_energy", float("nan"))),
        cutoff_triggered=bool(warm_payload.get("cutoff_triggered", False)),
        cutoff_reason=warm_payload.get("cutoff_reason"),
    )

    adapt_payload, psi_adapt = adapt_mod._run_hardcoded_adapt_vqe(
        h_poly=h_poly,
        num_sites=int(cfg.physics.L),
        ordering=str(cfg.physics.ordering),
        problem="hh",
        adapt_pool=cfg.adapt.pool,
        t=float(cfg.physics.t),
        u=float(cfg.physics.u),
        dv=float(cfg.physics.dv),
        boundary=str(cfg.physics.boundary),
        omega0=float(cfg.physics.omega0),
        g_ep=float(cfg.physics.g_ep),
        n_ph_max=int(cfg.physics.n_ph_max),
        boson_encoding=str(cfg.physics.boson_encoding),
        max_depth=int(cfg.adapt.max_depth),
        eps_grad=float(cfg.adapt.eps_grad),
        eps_energy=float(cfg.adapt.eps_energy),
        adapt_drop_floor=cfg.adapt.drop_floor,
        adapt_drop_patience=cfg.adapt.drop_patience,
        adapt_drop_min_depth=cfg.adapt.drop_min_depth,
        adapt_grad_floor=cfg.adapt.grad_floor,
        maxiter=int(cfg.adapt.maxiter),
        seed=int(cfg.adapt.seed),
        adapt_inner_optimizer=str(cfg.adapt.inner_optimizer),
        adapt_spsa_a=float(cfg.adapt.spsa_a),
        adapt_spsa_c=float(cfg.adapt.spsa_c),
        adapt_spsa_alpha=float(cfg.adapt.spsa_alpha),
        adapt_spsa_gamma=float(cfg.adapt.spsa_gamma),
        adapt_spsa_A=float(cfg.adapt.spsa_A),
        adapt_spsa_avg_last=int(cfg.adapt.spsa_avg_last),
        adapt_spsa_eval_repeats=int(cfg.adapt.spsa_eval_repeats),
        adapt_spsa_eval_agg=str(cfg.adapt.spsa_eval_agg),
        adapt_spsa_callback_every=int(cfg.adapt.spsa_callback_every),
        adapt_spsa_progress_every_s=float(cfg.adapt.spsa_progress_every_s),
        allow_repeats=bool(cfg.adapt.allow_repeats),
        finite_angle_fallback=bool(cfg.adapt.finite_angle_fallback),
        finite_angle=float(cfg.adapt.finite_angle),
        finite_angle_min_improvement=float(cfg.adapt.finite_angle_min_improvement),
        paop_r=int(cfg.adapt.paop_r),
        paop_split_paulis=bool(cfg.adapt.paop_split_paulis),
        paop_prune_eps=float(cfg.adapt.paop_prune_eps),
        paop_normalization=str(cfg.adapt.paop_normalization),
        disable_hh_seed=bool(cfg.adapt.disable_hh_seed),
        adapt_ref_json=Path(adapt_seed_json),
        adapt_reopt_policy=str(cfg.adapt.reopt_policy),
        adapt_window_size=int(cfg.adapt.window_size),
        adapt_window_topk=int(cfg.adapt.window_topk),
        adapt_full_refit_every=int(cfg.adapt.full_refit_every),
        adapt_final_full_refit=bool(cfg.adapt.final_full_refit),
        adapt_beam_live_branches=int(cfg.adapt.beam_live_branches),
        adapt_beam_children_per_parent=cfg.adapt.beam_children_per_parent,
        adapt_beam_terminated_keep=cfg.adapt.beam_terminated_keep,
        adapt_continuation_mode=str(cfg.adapt.continuation_mode),
        phase1_lambda_F=float(cfg.adapt.phase1_lambda_F),
        phase1_lambda_compile=float(cfg.adapt.phase1_lambda_compile),
        phase1_lambda_measure=float(cfg.adapt.phase1_lambda_measure),
        phase1_lambda_leak=float(cfg.adapt.phase1_lambda_leak),
        phase1_score_z_alpha=float(cfg.adapt.phase1_score_z_alpha),
        phase1_probe_max_positions=int(cfg.adapt.phase1_probe_max_positions),
        phase1_plateau_patience=int(cfg.adapt.phase1_plateau_patience),
        phase1_trough_margin_ratio=float(cfg.adapt.phase1_trough_margin_ratio),
        phase1_prune_enabled=bool(cfg.adapt.phase1_prune_enabled),
        phase1_prune_fraction=float(cfg.adapt.phase1_prune_fraction),
        phase1_prune_max_candidates=int(cfg.adapt.phase1_prune_max_candidates),
        phase1_prune_max_regression=float(cfg.adapt.phase1_prune_max_regression),
        phase3_motif_source_json=cfg.adapt.phase3_motif_source_json,
        phase3_symmetry_mitigation_mode=str(cfg.adapt.phase3_symmetry_mitigation_mode),
        phase3_enable_rescue=bool(cfg.adapt.phase3_enable_rescue),
        phase3_lifetime_cost_mode=str(cfg.adapt.phase3_lifetime_cost_mode),
        phase3_runtime_split_mode=str(cfg.adapt.phase3_runtime_split_mode),
        diagnostics_out=adapt_diagnostics,
    )
    adapt_payload["adapt_ref_json"] = str(adapt_seed_json)
    adapt_payload["initial_state_source"] = "adapt_ref_json"
    _append_workflow_log(
        cfg,
        "adapt_seed_checkpoint_used",
        checkpoint_json=str(adapt_seed_json),
        adapt_ref_base_depth=adapt_payload.get("adapt_ref_base_depth"),
        exact_gs_energy=adapt_payload.get("exact_gs_energy"),
    )

    _write_adapt_handoff(
        cfg,
        adapt_payload,
        np.asarray(psi_adapt, dtype=complex).reshape(-1),
        seed_provenance=_build_seed_provenance(cfg, seed_refine_payload),
    )
    nq_total = _hh_nq_total(cfg.physics.L, cfg.physics.n_ph_max, cfg.physics.boson_encoding)
    if bool(cfg.replay.enabled):
        replay_cfg = replay_mod.RunConfig(
            adapt_input_json=Path(cfg.artifacts.handoff_json),
            output_json=Path(cfg.artifacts.replay_output_json),
            output_csv=Path(cfg.artifacts.replay_output_csv),
            output_md=Path(cfg.artifacts.replay_output_md),
            output_log=Path(cfg.artifacts.replay_output_log),
            tag=f"{cfg.artifacts.tag}_replay",
            generator_family=str(cfg.replay.generator_family),
            fallback_family=str(cfg.replay.fallback_family),
            legacy_paop_key=str(cfg.replay.legacy_paop_key),
            replay_seed_policy=str(cfg.replay.replay_seed_policy),
            replay_continuation_mode=str(cfg.replay.continuation_mode),
            L=int(cfg.physics.L),
            t=float(cfg.physics.t),
            u=float(cfg.physics.u),
            dv=float(cfg.physics.dv),
            omega0=float(cfg.physics.omega0),
            g_ep=float(cfg.physics.g_ep),
            n_ph_max=int(cfg.physics.n_ph_max),
            boson_encoding=str(cfg.physics.boson_encoding),
            ordering=str(cfg.physics.ordering),
            boundary=str(cfg.physics.boundary),
            sector_n_up=int(cfg.physics.sector_n_up),
            sector_n_dn=int(cfg.physics.sector_n_dn),
            reps=int(cfg.replay.reps),
            restarts=int(cfg.replay.restarts),
            maxiter=int(cfg.replay.maxiter),
            method=str(cfg.replay.method),
            seed=int(cfg.replay.seed),
            energy_backend=str(cfg.replay.energy_backend),
            progress_every_s=float(cfg.replay.progress_every_s),
            wallclock_cap_s=int(cfg.replay.wallclock_cap_s),
            paop_r=int(cfg.replay.paop_r),
            paop_split_paulis=bool(cfg.replay.paop_split_paulis),
            paop_prune_eps=float(cfg.replay.paop_prune_eps),
            paop_normalization=str(cfg.replay.paop_normalization),
            spsa_a=float(cfg.replay.spsa_a),
            spsa_c=float(cfg.replay.spsa_c),
            spsa_alpha=float(cfg.replay.spsa_alpha),
            spsa_gamma=float(cfg.replay.spsa_gamma),
            spsa_A=float(cfg.replay.spsa_A),
            spsa_avg_last=int(cfg.replay.spsa_avg_last),
            spsa_eval_repeats=int(cfg.replay.spsa_eval_repeats),
            spsa_eval_agg=str(cfg.replay.spsa_eval_agg),
            replay_freeze_fraction=float(cfg.replay.replay_freeze_fraction),
            replay_unfreeze_fraction=float(cfg.replay.replay_unfreeze_fraction),
            replay_full_fraction=float(cfg.replay.replay_full_fraction),
            replay_qn_spsa_refresh_every=int(cfg.replay.replay_qn_spsa_refresh_every),
            replay_qn_spsa_refresh_mode=str(cfg.replay.replay_qn_spsa_refresh_mode),
            phase3_symmetry_mitigation_mode=str(cfg.replay.phase3_symmetry_mitigation_mode),
        )
        try:
            replay_payload = replay_mod.run(replay_cfg, diagnostics_out=replay_diagnostics)
        except TypeError as exc:
            if "diagnostics_out" not in str(exc):
                raise
            replay_payload = replay_mod.run(replay_cfg)
            replay_diagnostics = {}
        best_state = replay_payload.get("best_state", {})
        if not isinstance(best_state, Mapping):
            raise ValueError("Replay payload missing best_state block.")
        amplitudes = best_state.get("amplitudes_qn_to_q0", None)
        if not isinstance(amplitudes, Mapping):
            raise ValueError("Replay payload missing best_state.amplitudes_qn_to_q0.")
        psi_final = hc_pipeline._state_from_amplitudes_qn_to_q0(amplitudes, int(nq_total))
        psi_final = hc_pipeline._normalize_state(np.asarray(psi_final, dtype=complex).reshape(-1))
    else:
        replay_payload = {
            "generator_family": {
                "requested": str(cfg.replay.generator_family),
                "resolved": None,
                "fallback": str(cfg.replay.fallback_family),
            },
            "seed_baseline": {},
            "replay_contract": {
                "continuation_mode": str(cfg.replay.continuation_mode),
                "seed_policy_requested": str(cfg.replay.replay_seed_policy),
            },
            "vqe": {},
            "exact": {},
            "skipped": True,
            "skip_reason": "run_replay_false",
        }
        psi_final = hc_pipeline._normalize_state(np.asarray(psi_adapt, dtype=complex).reshape(-1))
        _append_workflow_log(
            cfg,
            "replay_stage_skipped",
            reason="run_replay_false",
            adapt_handoff_json=str(cfg.artifacts.handoff_json),
        )
    circuit_contexts = _assemble_stage_circuit_contexts(
        cfg=cfg,
        psi_hf=np.asarray(psi_hf, dtype=complex).reshape(-1),
        warm_payload=warm_payload,
        adapt_diagnostics=adapt_diagnostics,
        replay_diagnostics=replay_diagnostics,
    )

    return StageExecutionResult(
        h_poly=h_poly,
        hmat=np.asarray(hmat, dtype=complex),
        ordered_labels_exyz=list(ordered_labels_exyz),
        coeff_map_exyz=dict(coeff_map_exyz),
        nq_total=int(nq_total),
        psi_hf=np.asarray(psi_hf, dtype=complex).reshape(-1),
        psi_warm=np.asarray(psi_warm, dtype=complex).reshape(-1),
        psi_adapt=np.asarray(psi_adapt, dtype=complex).reshape(-1),
        psi_final=np.asarray(psi_final, dtype=complex).reshape(-1),
        warm_payload=dict(warm_payload),
        adapt_payload=dict(adapt_payload),
        replay_payload=dict(replay_payload),
        psi_seed_refine=(
            None if psi_seed_refine is None else np.asarray(psi_seed_refine, dtype=complex).reshape(-1)
        ),
        seed_refine_payload=(None if seed_refine_payload is None else dict(seed_refine_payload)),
        warm_circuit_context=circuit_contexts["warm_circuit_context"],
        adapt_circuit_context=circuit_contexts["adapt_circuit_context"],
        replay_circuit_context=circuit_contexts["replay_circuit_context"],
    )


def _build_drive_provider(
    *,
    cfg: StagedHHConfig,
    nq_total: int,
    ordered_labels_exyz: Sequence[str],
) -> tuple[Any | None, dict[str, Any] | None, list[str], dict[str, Any] | None]:
    if not bool(cfg.dynamics.enable_drive):
        return None, None, list(ordered_labels_exyz), None
    custom_weights = None
    if str(cfg.dynamics.drive_pattern) == "custom":
        custom_weights = _parse_drive_custom_weights(cfg.dynamics.drive_custom_s)
        if custom_weights is None:
            raise ValueError("--drive-custom-s is required when --drive-pattern custom.")
    drive = build_gaussian_sinusoid_density_drive(
        n_sites=int(cfg.physics.L),
        nq_total=int(nq_total),
        indexing=str(cfg.physics.ordering),
        A=float(cfg.dynamics.drive_A),
        omega=float(cfg.dynamics.drive_omega),
        tbar=float(cfg.dynamics.drive_tbar),
        phi=float(cfg.dynamics.drive_phi),
        pattern_mode=str(cfg.dynamics.drive_pattern),
        custom_weights=custom_weights,
        include_identity=bool(cfg.dynamics.drive_include_identity),
        coeff_tol=0.0,
    )
    drive_labels = set(drive.template.labels_exyz(include_identity=bool(drive.include_identity)))
    ordered = list(ordered_labels_exyz)
    missing = sorted(drive_labels.difference(ordered))
    ordered.extend(missing)
    profile = {
        "A": float(cfg.dynamics.drive_A),
        "omega": float(cfg.dynamics.drive_omega),
        "tbar": float(cfg.dynamics.drive_tbar),
        "phi": float(cfg.dynamics.drive_phi),
        "pattern": str(cfg.dynamics.drive_pattern),
        "custom_weights": custom_weights,
        "include_identity": bool(cfg.dynamics.drive_include_identity),
        "time_sampling": str(cfg.dynamics.drive_time_sampling),
        "t0": float(cfg.dynamics.drive_t0),
    }
    meta = {
        "reference_method": str(reference_method_name(str(cfg.dynamics.drive_time_sampling))),
        "missing_drive_labels_added": int(len(missing)),
        "drive_label_count": int(len(drive_labels)),
    }
    return drive.coeff_map_exyz, meta, ordered, profile


def _run_noiseless_profile(
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
    drive_provider = None
    drive_meta = None
    drive_profile = None
    ordered_for_run = list(ordered_labels_exyz)
    if drive_enabled:
        drive_provider, drive_meta, ordered_for_run, drive_profile = _build_drive_provider(
            cfg=cfg,
            nq_total=int(round(math.log2(int(np.asarray(psi_seed).size)))),
            ordered_labels_exyz=ordered_labels_exyz,
        )

    method_payloads: dict[str, Any] = {}
    reference_rows: list[dict[str, Any]] | None = None
    psi_seed_arr = np.asarray(psi_seed, dtype=complex).reshape(-1)
    ground_state_energy = float(ground_state_reference_energy)

    for method in cfg.dynamics.methods:
        rows, _ = hc_pipeline._simulate_trajectory(
            num_sites=int(cfg.physics.L),
            ordering=str(cfg.physics.ordering),
            psi0_legacy_trot=np.asarray(psi_seed_arr, dtype=complex),
            psi0_paop_trot=np.asarray(psi_seed_arr, dtype=complex),
            psi0_hva_trot=np.asarray(psi_seed_arr, dtype=complex),
            legacy_branch_label="replay",
            psi0_exact_ref=np.asarray(psi_seed_arr, dtype=complex),
            fidelity_subspace_basis_v0=np.asarray(psi_seed_arr, dtype=complex).reshape(-1, 1),
            fidelity_subspace_energy_tol=float(cfg.dynamics.fidelity_subspace_energy_tol),
            hmat=np.asarray(hmat, dtype=complex),
            ordered_labels_exyz=list(ordered_for_run),
            coeff_map_exyz=dict(coeff_map_exyz),
            trotter_steps=int(cfg.dynamics.trotter_steps),
            t_final=float(cfg.dynamics.t_final),
            num_times=int(cfg.dynamics.num_times),
            suzuki_order=2,
            drive_coeff_provider_exyz=drive_provider,
            drive_t0=float(cfg.dynamics.drive_t0 if drive_enabled else 0.0),
            drive_time_sampling=str(cfg.dynamics.drive_time_sampling),
            exact_steps_multiplier=(int(cfg.dynamics.exact_steps_multiplier) if drive_enabled else 1),
            propagator=str(method),
            cfqm_stage_exp=str(cfg.dynamics.cfqm_stage_exp),
            cfqm_coeff_drop_abs_tol=float(cfg.dynamics.cfqm_coeff_drop_abs_tol),
            cfqm_normalize=bool(cfg.dynamics.cfqm_normalize),
        )
        rows_with_metrics: list[dict[str, Any]] = []
        for row in rows:
            row_out = dict(row)
            row_out["abs_energy_error_vs_ground_state"] = float(
                abs(float(row_out["energy_total_trotter"]) - ground_state_energy)
            )
            rows_with_metrics.append(row_out)
        reference_rows = rows_with_metrics if reference_rows is None else reference_rows
        final_row = rows_with_metrics[-1]
        final_reference_error = float(
            abs(float(final_row["energy_total_trotter"]) - float(final_row["energy_total_exact"]))
        )
        method_payloads[str(method)] = {
            "propagator": str(method),
            "trajectory": rows_with_metrics,
            "final": {
                "energy_total_trotter": float(final_row["energy_total_trotter"]),
                "energy_total_exact": float(final_row["energy_total_exact"]),
                "abs_energy_total_error": float(final_reference_error),
                "abs_energy_total_error_vs_reference": float(final_reference_error),
                "abs_energy_error_vs_ground_state": float(final_row["abs_energy_error_vs_ground_state"]),
                "fidelity": float(final_row["fidelity"]),
                "doublon_trotter": float(final_row["doublon_trotter"]),
                "doublon_exact": float(final_row["doublon_exact"]),
            },
            "settings": {
                "trotter_steps": int(cfg.dynamics.trotter_steps),
                "num_times": int(cfg.dynamics.num_times),
                "t_final": float(cfg.dynamics.t_final),
                "cfqm_stage_exp": str(cfg.dynamics.cfqm_stage_exp),
                "cfqm_coeff_drop_abs_tol": float(cfg.dynamics.cfqm_coeff_drop_abs_tol),
                "cfqm_normalize": bool(cfg.dynamics.cfqm_normalize),
            },
        }

    assert reference_rows is not None
    return {
        "drive_enabled": bool(drive_enabled),
        "drive_profile": drive_profile,
        "drive_meta": drive_meta,
        "times": [float(row["time"]) for row in reference_rows],
        "ground_state_reference": {
            "energy": float(ground_state_energy),
            "kind": "filtered_sector_ground_state_static",
            "source": str(ground_state_reference_source),
        },
        "reference": {
            "kind": "seeded_exact_reference",
            "initial_state": "psi_final",
            "method": (
                "eigendecomposition"
                if not drive_enabled
                else str(reference_method_name(str(cfg.dynamics.drive_time_sampling)))
            ),
            "energy_total_exact": [float(row["energy_total_exact"]) for row in reference_rows],
            "doublon_exact": [float(row["doublon_exact"]) for row in reference_rows],
        },
        "methods": method_payloads,
    }


def run_noiseless_profiles(stage_result: StageExecutionResult, cfg: StagedHHConfig) -> dict[str, Any]:
    terminal_exact, terminal_source = _terminal_reference_energy(stage_result, cfg)
    profiles = {
        "static": _run_noiseless_profile(
            cfg=cfg,
            psi_seed=stage_result.psi_final,
            hmat=stage_result.hmat,
            ordered_labels_exyz=stage_result.ordered_labels_exyz,
            coeff_map_exyz=stage_result.coeff_map_exyz,
            drive_enabled=False,
            ground_state_reference_energy=terminal_exact,
            ground_state_reference_source=terminal_source,
        )
    }
    if bool(cfg.dynamics.enable_drive):
        profiles["drive"] = _run_noiseless_profile(
            cfg=cfg,
            psi_seed=stage_result.psi_final,
            hmat=stage_result.hmat,
            ordered_labels_exyz=stage_result.ordered_labels_exyz,
            coeff_map_exyz=stage_result.coeff_map_exyz,
            drive_enabled=True,
            ground_state_reference_energy=terminal_exact,
            ground_state_reference_source=terminal_source,
        )
    return {"profiles": profiles}


def _empty_qiskit_circuit(num_qubits: int) -> Any:
    from qiskit import QuantumCircuit

    return QuantumCircuit(int(num_qubits))


def _transpile_target_metadata(cfg: StagedHHConfig) -> dict[str, Any] | None:
    if cfg.circuit_metrics.backend_name is None:
        return None
    return {
        "backend_name": str(cfg.circuit_metrics.backend_name),
        "use_fake_backend": bool(cfg.circuit_metrics.use_fake_backend),
        "optimization_level": int(cfg.circuit_metrics.optimization_level),
        "seed_transpiler": int(cfg.circuit_metrics.seed_transpiler),
        "basis_gates": ["rz", "sx", "x", "cx"],
    }


def _transpile_metrics_or_error(
    cfg: StagedHHConfig,
    *,
    circuit: Any | None,
    enabled: bool = True,
    reason: str | None = None,
) -> dict[str, Any] | None:
    if circuit is None:
        return None
    if cfg.circuit_metrics.backend_name is None:
        return None
    if not bool(enabled):
        return {
            "target": _transpile_target_metadata(cfg),
            "skipped": True,
            "reason": str(reason or "transpile_metrics_disabled"),
        }
    try:
        return transpile_circuit_metrics(
            circuit,
            backend_name=str(cfg.circuit_metrics.backend_name),
            use_fake_backend=bool(cfg.circuit_metrics.use_fake_backend),
            optimization_level=int(cfg.circuit_metrics.optimization_level),
            seed_transpiler=int(cfg.circuit_metrics.seed_transpiler),
        )
    except Exception as exc:
        return {
            "target": _transpile_target_metadata(cfg),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _strip_circuit_objects(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_circuit_objects(val)
            for key, val in value.items()
            if str(key) != "circuit"
        }
    if isinstance(value, list):
        return [_strip_circuit_objects(item) for item in value]
    return value


def build_stage_circuit_report_artifacts(
    stage_result: StageExecutionResult,
    cfg: StagedHHConfig,
) -> dict[str, Any]:
    stage_bundles: dict[str, dict[str, Any] | None] = {
        "warm_start": None,
        "adapt_vqe": None,
        "conventional_replay": None,
    }

    warm_ctx = stage_result.warm_circuit_context
    if isinstance(warm_ctx, Mapping) and warm_ctx.get("ansatz") is not None:
        warm_circuit = ansatz_to_circuit(
            warm_ctx["ansatz"],
            np.asarray(warm_ctx.get("theta", []), dtype=float),
            num_qubits=int(warm_ctx.get("num_qubits", stage_result.nq_total)),
            reference_state=np.asarray(warm_ctx.get("reference_state"), dtype=complex),
        )
        stage_bundles["warm_start"] = {
            "title": f"L={int(cfg.physics.L)} warm HH-HVA",
            "circuit": warm_circuit,
            "metadata": {
                "ansatz": str(cfg.warm_start.ansatz_name),
                "reps": int(cfg.warm_start.reps),
                "energy": float(stage_result.warm_payload.get("energy", float("nan"))),
                "exact_energy": float(stage_result.warm_payload.get("exact_filtered_energy", float("nan"))),
                "delta_abs": float(
                    abs(
                        float(stage_result.warm_payload.get("energy", float("nan")))
                        - float(stage_result.warm_payload.get("exact_filtered_energy", float("nan")))
                    )
                ),
                "transpile_metrics": _transpile_metrics_or_error(cfg, circuit=warm_circuit),
            },
            "notes": [
                "Representative view keeps PauliEvolutionGate blocks intact.",
                "Expanded view applies one circuit-definition decomposition pass.",
            ],
        }

    adapt_ctx = stage_result.adapt_circuit_context
    adapt_circuit = None
    if isinstance(adapt_ctx, Mapping) and adapt_ctx.get("reference_state") is not None:
        adapt_circuit = adapt_ops_to_circuit(
            list(adapt_ctx.get("selected_ops", [])),
            np.asarray(adapt_ctx.get("theta", []), dtype=float),
            num_qubits=int(adapt_ctx.get("num_qubits", stage_result.nq_total)),
            reference_state=np.asarray(adapt_ctx.get("reference_state"), dtype=complex),
        )
        stage_bundles["adapt_vqe"] = {
            "title": f"L={int(cfg.physics.L)} ADAPT-VQE",
            "circuit": adapt_circuit,
            "metadata": {
                "depth": int(stage_result.adapt_payload.get("ansatz_depth", 0)),
                "pool_type": str(adapt_ctx.get("pool_type", stage_result.adapt_payload.get("pool_type", ""))),
                "continuation_mode": str(
                    adapt_ctx.get("continuation_mode", stage_result.adapt_payload.get("continuation_mode", ""))
                ),
                "energy": float(stage_result.adapt_payload.get("energy", float("nan"))),
                "exact_energy": float(stage_result.adapt_payload.get("exact_gs_energy", float("nan"))),
                "stop_reason": str(stage_result.adapt_payload.get("stop_reason", "")),
                "transpile_metrics": _transpile_metrics_or_error(cfg, circuit=adapt_circuit),
            },
            "notes": [
                "Circuit uses the actual selected ADAPT generators and optimized theta values.",
                "Reference state is the warm-stage output handed to ADAPT.",
            ],
        }

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/adapt_pipeline.py
(lines 540-759: ADAPT branch-management dataclasses and logical candidate materialization, including _BeamBranchState fields that carry scaffold history, optimizer memory, split events, motif usage, symmetry, and rescue state.)
```py
    stop_reason: str
    nfev_total: int


@dataclass(frozen=True)
class _ADAPTLogicalCandidate:
    """Private ADAPT selection unit for multi-parameter logical pool elements."""
    logical_label: str
    pool_indices: tuple[int, ...]
    parameterization: str
    family_id: str


@dataclass
class _BeamBranchState:
    branch_id: int
    parent_branch_id: int | None
    depth_local: int
    terminated: bool
    stop_reason: str | None
    selected_ops: list[AnsatzTerm]
    theta: np.ndarray
    energy_current: float
    available_indices: set[int]
    selection_counts: np.ndarray
    history: list[dict[str, Any]]
    phase1_stage: StageController
    phase1_residual_opened: bool
    phase1_last_probe_reason: str
    phase1_last_positions_considered: list[int]
    phase1_last_trough_detected: bool
    phase1_last_trough_probe_triggered: bool
    phase1_last_selected_score: float | None
    phase1_features_history: list[dict[str, Any]]
    phase1_stage_events: list[dict[str, Any]]
    phase1_measure_cache: MeasurementCacheAudit
    phase2_optimizer_memory: dict[str, Any]
    phase2_last_shortlist_records: list[dict[str, Any]]
    phase2_last_batch_selected: bool
    phase2_last_batch_penalty_total: float
    phase2_last_optimizer_memory_reused: bool
    phase2_last_optimizer_memory_source: str
    phase2_last_shortlist_eval_records: list[dict[str, Any]]
    drop_prev_delta_abs: float
    drop_plateau_hits: int
    eps_energy_low_streak: int
    phase3_split_events: list[dict[str, Any]]
    phase3_runtime_split_summary: dict[str, Any]
    phase3_motif_usage: dict[str, Any]
    phase3_rescue_history: list[dict[str, Any]]
    nfev_total_local: int

    def clone_for_child(self, *, branch_id: int) -> "_BeamBranchState":
        return _BeamBranchState(
            branch_id=int(branch_id),
            parent_branch_id=int(self.branch_id),
            depth_local=int(self.depth_local),
            terminated=bool(self.terminated),
            stop_reason=(None if self.stop_reason is None else str(self.stop_reason)),
            selected_ops=list(self.selected_ops),
            theta=np.asarray(self.theta, dtype=float).copy(),
            energy_current=float(self.energy_current),
            available_indices=set(int(x) for x in self.available_indices),
            selection_counts=np.asarray(self.selection_counts, dtype=np.int64).copy(),
            history=copy.deepcopy(self.history),
            phase1_stage=self.phase1_stage.clone(),
            phase1_residual_opened=bool(self.phase1_residual_opened),
            phase1_last_probe_reason=str(self.phase1_last_probe_reason),
            phase1_last_positions_considered=[int(x) for x in self.phase1_last_positions_considered],
            phase1_last_trough_detected=bool(self.phase1_last_trough_detected),
            phase1_last_trough_probe_triggered=bool(self.phase1_last_trough_probe_triggered),
            phase1_last_selected_score=(
                None if self.phase1_last_selected_score is None else float(self.phase1_last_selected_score)
            ),
            phase1_features_history=copy.deepcopy(self.phase1_features_history),
            phase1_stage_events=copy.deepcopy(self.phase1_stage_events),
            phase1_measure_cache=self.phase1_measure_cache.clone(),
            phase2_optimizer_memory=copy.deepcopy(self.phase2_optimizer_memory),
            phase2_last_shortlist_records=copy.deepcopy(self.phase2_last_shortlist_records),
            phase2_last_batch_selected=bool(self.phase2_last_batch_selected),
            phase2_last_batch_penalty_total=float(self.phase2_last_batch_penalty_total),
            phase2_last_optimizer_memory_reused=bool(self.phase2_last_optimizer_memory_reused),
            phase2_last_optimizer_memory_source=str(self.phase2_last_optimizer_memory_source),
            phase2_last_shortlist_eval_records=copy.deepcopy(self.phase2_last_shortlist_eval_records),
            drop_prev_delta_abs=float(self.drop_prev_delta_abs),
            drop_plateau_hits=int(self.drop_plateau_hits),
            eps_energy_low_streak=int(self.eps_energy_low_streak),
            phase3_split_events=copy.deepcopy(self.phase3_split_events),
            phase3_runtime_split_summary=copy.deepcopy(self.phase3_runtime_split_summary),
            phase3_motif_usage=copy.deepcopy(self.phase3_motif_usage),
            phase3_rescue_history=copy.deepcopy(self.phase3_rescue_history),
            nfev_total_local=int(self.nfev_total_local),
        )


@dataclass(frozen=True)
class _BranchExpansionPlan:
    candidate_pool_index: int
    position_id: int
    selection_mode: str
    candidate_label: str
    candidate_term: AnsatzTerm
    feature_row: dict[str, Any] | None
    init_theta: float = 0.0


@dataclass
class _BranchStepScratch:
    energy_current: float
    psi_current: np.ndarray
    hpsi_current: np.ndarray
    gradients: np.ndarray
    grad_magnitudes: np.ndarray
    max_grad: float
    gradient_eval_elapsed_s: float
    append_position: int
    best_idx: int
    selected_position: int
    selection_mode: str
    stage_name: str
    phase1_feature_selected: dict[str, Any] | None
    phase1_stage_transition_reason: str
    phase1_stage_now: str
    phase1_stage_after_transition: StageController
    phase1_last_probe_reason: str
    phase1_last_positions_considered: list[int]
    phase1_last_trough_detected: bool
    phase1_last_trough_probe_triggered: bool
    phase1_last_selected_score: float | None
    phase2_last_shortlist_records: list[dict[str, Any]]
    phase2_last_batch_selected: bool
    phase2_last_batch_penalty_total: float
    phase2_last_optimizer_memory_reused: bool
    phase2_last_optimizer_memory_source: str
    phase2_last_shortlist_eval_records: list[dict[str, Any]]
    phase1_residual_opened: bool
    available_indices_after_transition: set[int]
    phase1_stage_events_after_transition: list[dict[str, Any]]
    phase3_runtime_split_summary_after_eval: dict[str, Any]
    proposals: list[_BranchExpansionPlan]
    stop_reason: str | None
    fallback_scan_size: int
    fallback_best_probe_delta_e: float | None
    fallback_best_probe_theta: float | None


def _parse_seq2p_step_label(label: str) -> tuple[str, str] | None:
    raw = str(label).strip()
    for step_name in ("ferm", "motif"):
        suffix = f"::step={step_name}"
        if raw.endswith(suffix):
            return raw[: -len(suffix)], step_name
    return None


def _build_seq2p_logical_candidates(
    pool: Sequence[AnsatzTerm],
    *,
    family_id: str,
) -> list[_ADAPTLogicalCandidate]:
    candidates: list[_ADAPTLogicalCandidate] = []
    idx = 0
    while idx < len(pool):
        if idx + 1 >= len(pool):
            raise ValueError("Malformed seq2p pool: trailing unpaired flat term.")
        lhs = _parse_seq2p_step_label(str(pool[idx].label))
        rhs = _parse_seq2p_step_label(str(pool[idx + 1].label))
        if lhs is None or rhs is None:
            raise ValueError("Malformed seq2p pool: expected paired ::step=ferm/::step=motif labels.")
        lhs_base, lhs_step = lhs
        rhs_base, rhs_step = rhs
        if lhs_step != "ferm" or rhs_step != "motif" or lhs_base != rhs_base:
            raise ValueError("Malformed seq2p pool: expected adjacent ferm/motif terms for each logical pair.")
        candidates.append(
            _ADAPTLogicalCandidate(
                logical_label=str(lhs_base),
                pool_indices=(int(idx), int(idx + 1)),
                parameterization="double_sequential",
                family_id=str(family_id),
            )
        )
        idx += 2
    return candidates


def _logical_candidate_gradient_summary(
    candidate: _ADAPTLogicalCandidate,
    gradients: np.ndarray,
) -> tuple[float, list[float], list[float]]:
    signed_components = [float(gradients[int(idx)]) for idx in candidate.pool_indices]
    abs_components = [abs(float(val)) for val in signed_components]
    score = math.sqrt(sum(float(val) * float(val) for val in abs_components))
    return float(score), signed_components, abs_components


def _build_uccsd_pool(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> list[AnsatzTerm]:
    """Build the UCCSD operator pool using HardcodedUCCSDAnsatz.base_terms.

    This reuses the exact same excitation generators the VQE pipeline uses,
    ensuring apples-to-apples comparison with the Qiskit UCCSD pool.
    """
    dummy_ansatz = HardcodedUCCSDAnsatz(
        dims=int(num_sites),
        num_particles=num_particles,
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        include_singles=True,
        include_doubles=True,
    )
    return list(dummy_ansatz.base_terms)


def _build_cse_pool(
    num_sites: int,
    ordering: str,

```

(lines 2210-2449: _run_hardcoded_adapt_vqe entrypoint and resolved continuation/beam/phase3 controls that define the reusable HH phase3_v1 scaffolding for a realtime-VQS branch manager.)
```py
def _run_hardcoded_adapt_vqe(
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
    """Run standard ADAPT-VQE and return (payload, psi_ground)."""
    if float(finite_angle) <= 0.0:
        raise ValueError("finite_angle must be > 0.")
    if float(finite_angle_min_improvement) < 0.0:
        raise ValueError("finite_angle_min_improvement must be >= 0.")
    adapt_state_backend_key = str(adapt_state_backend).strip().lower()
    if adapt_state_backend_key not in {"legacy", "compiled"}:
        raise ValueError("adapt_state_backend must be one of {'legacy','compiled'}.")
    adapt_reopt_policy_key = str(adapt_reopt_policy).strip().lower()
    if adapt_reopt_policy_key not in _VALID_REOPT_POLICIES:
        raise ValueError(f"adapt_reopt_policy must be one of {_VALID_REOPT_POLICIES}.")
    adapt_window_size_val = int(adapt_window_size)
    adapt_window_topk_val = int(adapt_window_topk)
    adapt_full_refit_every_val = int(adapt_full_refit_every)
    adapt_final_full_refit_val = bool(adapt_final_full_refit)
    if adapt_window_size_val < 1:
        raise ValueError("adapt_window_size must be >= 1.")
    if adapt_window_topk_val < 0:
        raise ValueError("adapt_window_topk must be >= 0.")
    if adapt_full_refit_every_val < 0:
        raise ValueError("adapt_full_refit_every must be >= 0.")
    adapt_inner_optimizer_key = str(adapt_inner_optimizer).strip().upper()
    if adapt_inner_optimizer_key not in _VALID_ADAPT_INNER_OPTIMIZERS:
        raise ValueError(
            f"adapt_inner_optimizer must be one of {sorted(_VALID_ADAPT_INNER_OPTIMIZERS)}."
        )
    adapt_spsa_eval_agg_key = str(adapt_spsa_eval_agg).strip().lower()
    if adapt_spsa_eval_agg_key not in {"mean", "median"}:
        raise ValueError("adapt_spsa_eval_agg must be one of {'mean','median'}.")
    if int(adapt_spsa_callback_every) < 1:
        raise ValueError("adapt_spsa_callback_every must be >= 1.")
    if float(adapt_spsa_progress_every_s) < 0.0:
        raise ValueError("adapt_spsa_progress_every_s must be >= 0.")
    if int(adapt_eps_energy_min_extra_depth) < -1:
        raise ValueError("adapt_eps_energy_min_extra_depth must be >= 0 or -1 (auto=L).")
    if int(adapt_eps_energy_patience) < -1 or int(adapt_eps_energy_patience) == 0:
        raise ValueError("adapt_eps_energy_patience must be >= 1 or -1 (auto=L).")
    if int(adapt_ref_base_depth) < 0:
        raise ValueError("adapt_ref_base_depth must be >= 0.")
    problem_key = str(problem).strip().lower()
    continuation_mode = _resolve_adapt_continuation_mode(
        problem=str(problem_key),
        requested_mode=adapt_continuation_mode,
    )
    stop_policy = _resolve_adapt_stop_policy(
        problem=str(problem_key),
        continuation_mode=str(continuation_mode),
        adapt_drop_floor=adapt_drop_floor,
        adapt_drop_patience=adapt_drop_patience,
        adapt_drop_min_depth=adapt_drop_min_depth,
        adapt_grad_floor=adapt_grad_floor,
    )
    beam_policy = _resolve_beam_capacity_policy(
        adapt_beam_live_branches=int(adapt_beam_live_branches),
        adapt_beam_children_per_parent=adapt_beam_children_per_parent,
        adapt_beam_terminated_keep=adapt_beam_terminated_keep,
    )
    if bool(beam_policy.beam_enabled) and not (
        str(problem_key) == "hh"
        and str(continuation_mode).strip().lower() in _HH_STAGED_CONTINUATION_MODES
    ):
        raise ValueError(
            "True ADAPT beam mode is currently supported only for HH staged continuation modes."
        )
    adapt_drop_floor = float(stop_policy.adapt_drop_floor)
    adapt_drop_patience = int(stop_policy.adapt_drop_patience)
    adapt_drop_min_depth = int(stop_policy.adapt_drop_min_depth)
    adapt_grad_floor = float(stop_policy.adapt_grad_floor)
    drop_policy_enabled = bool(stop_policy.drop_policy_enabled)
    eps_energy_termination_enabled = bool(stop_policy.eps_energy_termination_enabled)
    eps_grad_termination_enabled = bool(stop_policy.eps_grad_termination_enabled)
    if float(adapt_drop_floor) >= 0.0 and int(adapt_drop_patience) < 1:
        raise ValueError("adapt_drop_patience must be >= 1 when adapt_drop_floor is enabled.")
    if float(adapt_drop_floor) >= 0.0 and int(adapt_drop_min_depth) < 1:
        raise ValueError("adapt_drop_min_depth must be >= 1 when adapt_drop_floor is enabled.")
    if int(adapt_drop_patience) < 0:
        raise ValueError("adapt_drop_patience must be >= 0.")
    if int(adapt_drop_min_depth) < 0:
        raise ValueError("adapt_drop_min_depth must be >= 0.")
    eps_energy_min_extra_depth_effective = (
        int(num_sites)
        if int(adapt_eps_energy_min_extra_depth) == -1
        else int(adapt_eps_energy_min_extra_depth)
    )
    eps_energy_patience_effective = (
        int(num_sites)
        if int(adapt_eps_energy_patience) == -1
        else int(adapt_eps_energy_patience)
    )
    if int(eps_energy_patience_effective) < 1:
        raise ValueError("resolved eps-energy patience must be >= 1.")
    if int(eps_energy_min_extra_depth_effective) < 0:
        raise ValueError("resolved eps-energy min extra depth must be >= 0.")
    phase3_symmetry_mitigation_mode_key = str(phase3_symmetry_mitigation_mode).strip().lower()
    if phase3_symmetry_mitigation_mode_key not in {"off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"}:
        raise ValueError(
            "phase3_symmetry_mitigation_mode must be one of {'off','verify_only','postselect_diag_v1','projector_renorm_v1'}."
        )
    phase3_lifetime_cost_mode_key = str(phase3_lifetime_cost_mode).strip().lower()
    if phase3_lifetime_cost_mode_key not in {"off", "phase3_v1"}:
        raise ValueError("phase3_lifetime_cost_mode must be one of {'off','phase3_v1'}.")
    phase3_runtime_split_mode_key = str(phase3_runtime_split_mode).strip().lower()
    if phase3_runtime_split_mode_key not in {"off", "shortlist_pauli_children_v1"}:
        raise ValueError(
            "phase3_runtime_split_mode must be one of {'off','shortlist_pauli_children_v1'}."
        )
    pool_key_input = None if adapt_pool is None else str(adapt_pool).strip().lower()
    adapt_spsa_params = {
        "a": float(adapt_spsa_a),
        "c": float(adapt_spsa_c),
        "alpha": float(adapt_spsa_alpha),
        "gamma": float(adapt_spsa_gamma),
        "A": float(adapt_spsa_A),
        "avg_last": int(adapt_spsa_avg_last),
        "eval_repeats": int(adapt_spsa_eval_repeats),
        "eval_agg": str(adapt_spsa_eval_agg_key),
        "callback_every": int(adapt_spsa_callback_every),
        "progress_every_s": float(adapt_spsa_progress_every_s),
    }
    t0 = time.perf_counter()
    hf_bits = "N/A"
    if adapt_ref_json is not None and psi_ref_override is not None:
        raise ValueError("Provide at most one of adapt_ref_json or psi_ref_override.")
    adapt_ref_import: dict[str, Any] | None = None
    adapt_ref_meta: Mapping[str, Any] | None = None
    if adapt_ref_json is not None:
        nq_total_expected = (
            int(2 * int(num_sites) + int(num_sites) * int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding))))
            if problem_key == "hh"
            else int(2 * int(num_sites))
        )
        psi_ref_override, adapt_ref_meta = _load_adapt_initial_state(
            Path(adapt_ref_json),
            int(nq_total_expected),
        )
        adapt_ref_vqe = adapt_ref_meta.get("adapt_vqe", {})
        if isinstance(adapt_ref_vqe, Mapping):
            ref_depth_raw = adapt_ref_vqe.get("ansatz_depth")
            try:
                ref_depth_val = int(ref_depth_raw)
                if ref_depth_val >= 0:
                    adapt_ref_base_depth = int(ref_depth_val)
            except (TypeError, ValueError):
                pass
        adapt_ref_import = {
            "path": str(Path(adapt_ref_json)),
            "initial_state_source": adapt_ref_meta.get("initial_state_source"),
            "settings": adapt_ref_meta.get("settings", {}),
            "adapt_vqe": adapt_ref_meta.get("adapt_vqe", {}),
            "adapt_ref_base_depth": int(adapt_ref_base_depth),
        }

```

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

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/time_propagation/local_checkpoint_fit.py
```py
"""Local checkpoint-fit dynamics utilities for fixed-seed HH hardware screens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


@dataclass(frozen=True)
class LocalPauliAnsatzSpec:
    num_qubits: int
    reps: int
    single_axes: tuple[str, ...] = ("y",)
    entangler_axes: tuple[str, ...] = ("xx", "zz")
    entangler_edges: tuple[tuple[int, int], ...] | None = None
    repr_mode: str = "JW"


@dataclass(frozen=True)
class CheckpointFitConfig:
    optimizer_method: str = "L-BFGS-B"
    maxiter: int = 60
    gtol: float = 1e-8
    ftol: float = 1e-12
    angle_bound: float = math.pi
    param_shift: float = math.pi / 2.0
    coefficient_tolerance: float = 1e-12


@dataclass(frozen=True)
class CheckpointFitStepResult:
    theta: np.ndarray
    state: np.ndarray
    fidelity: float
    objective: float
    solver_row: dict[str, Any]


@dataclass(frozen=True)
class CheckpointFitTrajectoryResult:
    times: np.ndarray
    theta_history: np.ndarray
    states: tuple[np.ndarray, ...]
    solver_rows: tuple[dict[str, Any], ...]


_MATH_NORMALIZE_STATE = r"\hat{\psi}=\psi/\|\psi\|"


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    vec = np.asarray(psi, dtype=complex).reshape(-1)
    nrm = float(np.linalg.norm(vec))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return vec / nrm


_MATH_CHAIN_EDGES = r"\mathcal{E}_{\mathrm{chain}}=\{(q,q+1)\ |\ q=0,\dots,n_q-2\}"


def default_chain_edges(num_qubits: int) -> tuple[tuple[int, int], ...]:
    nq = int(num_qubits)
    if nq < 1:
        raise ValueError("num_qubits must be positive.")
    return tuple((int(q), int(q + 1)) for q in range(nq - 1))


_MATH_PAULI_WORD = r"P(\{(q,\sigma_q)\})=\bigotimes_{j=n_q-1}^{0}\sigma_j,\ \sigma_j=e\ \text{unless assigned}"


def _pauli_word(num_qubits: int, ops: dict[int, str]) -> str:
    nq = int(num_qubits)
    chars = ["e"] * nq
    for qubit, axis in ops.items():
        q = int(qubit)
        if q < 0 or q >= nq:
            raise ValueError(f"qubit index {q} out of range for nq={nq}.")
        sym = str(axis).strip().lower()
        if sym not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported Pauli axis {axis!r}.")
        chars[nq - 1 - q] = sym
    return "".join(chars)


_MATH_MONOMIAL_TERM = r"H_k=P_k,\ \ U_k(\theta_k)=\exp(-i\theta_k P_k)"


def _monomial_term(
    *,
    num_qubits: int,
    label: str,
    ops: dict[int, str],
    repr_mode: str,
) -> AnsatzTerm:
    word = _pauli_word(int(num_qubits), dict(ops))
    poly = PauliPolynomial(str(repr_mode), [PauliTerm(int(num_qubits), ps=word, pc=1.0)])
    return AnsatzTerm(label=str(label), polynomial=poly)


_MATH_BUILD_LOCAL_ANSA = (
    r"\mathcal{A}=\prod_{\ell=1}^{r}\left[\prod_{q,a\in S}\exp(-i\theta_{\ell,q,a}\sigma_a^{(q)})"
    r"\prod_{(i,j),b\in E}\exp(-i\theta_{\ell,i,j,b}\sigma_b^{(i)}\sigma_b^{(j)})\right]"
)


def build_local_pauli_ansatz_terms(spec: LocalPauliAnsatzSpec) -> tuple[AnsatzTerm, ...]:
    nq = int(spec.num_qubits)
    reps = int(spec.reps)
    if nq < 1:
        raise ValueError("num_qubits must be positive.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    single_axes = tuple(str(axis).strip().lower() for axis in spec.single_axes)
    entangler_axes = tuple(str(axis).strip().lower() for axis in spec.entangler_axes)
    if any(axis not in {"x", "y", "z"} for axis in single_axes):
        raise ValueError("single_axes must contain only x/y/z.")
    if any(axis not in {"xx", "yy", "zz"} for axis in entangler_axes):
        raise ValueError("entangler_axes must contain only xx/yy/zz.")
    edges = tuple(spec.entangler_edges or default_chain_edges(int(nq)))
    terms: list[AnsatzTerm] = []
    for rep_idx in range(reps):
        for axis in single_axes:
            for qubit in range(nq):
                terms.append(
                    _monomial_term(
                        num_qubits=nq,
                        label=f"local_{axis}(q={qubit})_rep{rep_idx + 1}",
                        ops={int(qubit): str(axis)},
                        repr_mode=str(spec.repr_mode),
                    )
                )
        for axis in entangler_axes:
            pauli_axis = str(axis)[0]
            for q0, q1 in edges:
                terms.append(
                    _monomial_term(
                        num_qubits=nq,
                        label=f"local_{axis}(q={int(q0)},{int(q1)})_rep{rep_idx + 1}",
                        ops={int(q0): pauli_axis, int(q1): pauli_axis},
                        repr_mode=str(spec.repr_mode),
                    )
                )
    return tuple(terms)


_MATH_STATE_FIDELITY = r"F(\psi,\phi)=|\langle\phi|\psi\rangle|^2"


def state_fidelity(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    lhs = _normalize_state(psi_a)
    rhs = _normalize_state(psi_b)
    return float(abs(np.vdot(rhs, lhs)) ** 2)


_MATH_FIDELITY_OBJECTIVE = (
    r"\mathcal{L}(\theta)=1-F(\psi(\theta),\psi_{\mathrm{target}}),"
    r"\quad \partial_k \psi(\theta)=\frac{\psi(\theta+s e_k)-\psi(\theta-s e_k)}{2\sin s}"
)


def _evaluate_fidelity_objective(
    theta: np.ndarray,
    *,
    executor: CompiledAnsatzExecutor,
    reference_state: np.ndarray,
    target_state: np.ndarray,
    param_shift: float,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    psi = _normalize_state(executor.prepare_state(theta_vec, reference_state))
    target = _normalize_state(target_state)
    overlap = np.vdot(target, psi)
    fidelity = float(abs(overlap) ** 2)
    objective = float(max(0.0, 1.0 - fidelity))
    if int(theta_vec.size) == 0:
        return objective, np.zeros(0, dtype=float), psi, fidelity

    shift = float(param_shift)
    denom = 2.0 * math.sin(shift)
    if abs(denom) <= 1e-15:
        raise ValueError("param_shift leads to zero parameter-shift denominator.")

    grad = np.zeros(int(theta_vec.size), dtype=float)
    for idx in range(int(theta_vec.size)):
        theta_plus = np.asarray(theta_vec, dtype=float).copy()
        theta_minus = np.asarray(theta_vec, dtype=float).copy()
        theta_plus[idx] += shift
        theta_minus[idx] -= shift
        psi_plus = _normalize_state(executor.prepare_state(theta_plus, reference_state))
        psi_minus = _normalize_state(executor.prepare_state(theta_minus, reference_state))
        dpsi = (psi_plus - psi_minus) / denom
        grad[idx] = float(-2.0 * np.real(np.conjugate(overlap) * np.vdot(target, dpsi)))
    return objective, grad, psi, fidelity


def _try_import_scipy_minimize() -> Any:
    try:
        from scipy.optimize import minimize  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Checkpoint fitting requires scipy.optimize.minimize. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return minimize


_MATH_FIT_CHECKPOINT = r"\theta^\star(t)=\arg\min_\theta 1-|\langle\psi_{\mathrm{target}}(t)|U(\theta)|\psi_{\mathrm{ref}}\rangle|^2"


def fit_checkpoint_target_state(
    reference_state: np.ndarray,
    target_state: np.ndarray,
    terms: Sequence[AnsatzTerm],
    *,
    config: CheckpointFitConfig,
    theta_init: np.ndarray | None = None,
    executor: CompiledAnsatzExecutor | None = None,
) -> CheckpointFitStepResult:
    minimize = _try_import_scipy_minimize()
    ref = _normalize_state(reference_state)
    target = _normalize_state(target_state)
    compiled = executor or CompiledAnsatzExecutor(
        list(terms),
        coefficient_tolerance=float(config.coefficient_tolerance),
    )
    npar = int(len(terms))
    theta0 = (
        np.zeros(npar, dtype=float)
        if theta_init is None
        else np.asarray(theta_init, dtype=float).reshape(-1)
    )
    if int(theta0.size) != npar:
        raise ValueError(f"theta_init length {int(theta0.size)} does not match term count {npar}.")

    eval_counter = {"fresh_calls": 0}
    cache: dict[str, Any] = {}

    def _evaluate(theta_raw: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
        theta_vec = np.asarray(theta_raw, dtype=float).reshape(-1)
        cached_theta = cache.get("theta", None)
        if isinstance(cached_theta, np.ndarray) and np.array_equal(theta_vec, cached_theta):
            return (
                float(cache["objective"]),
                np.asarray(cache["gradient"], dtype=float),
                np.asarray(cache["state"], dtype=complex),
                float(cache["fidelity"]),
            )
        objective, gradient, state, fidelity = _evaluate_fidelity_objective(
            theta_vec,
            executor=compiled,
            reference_state=ref,
            target_state=target,
            param_shift=float(config.param_shift),
        )
        cache["theta"] = np.asarray(theta_vec, dtype=float).copy()
        cache["objective"] = float(objective)
        cache["gradient"] = np.asarray(gradient, dtype=float).copy()
        cache["state"] = np.asarray(state, dtype=complex).copy()
        cache["fidelity"] = float(fidelity)
        eval_counter["fresh_calls"] = int(eval_counter["fresh_calls"]) + 1
        return objective, gradient, state, fidelity

    if state_fidelity(ref, target) >= 1.0 - 1e-14 and np.allclose(theta0, 0.0):
        objective, gradient, state, fidelity = _evaluate(theta0)
        return CheckpointFitStepResult(
            theta=np.asarray(theta0, dtype=float).copy(),
            state=np.asarray(state, dtype=complex).copy(),
            fidelity=float(fidelity),
            objective=float(objective),
            solver_row={
                "success": True,
                "status": 0,
                "message": "Exact reference-target match at initialization.",
                "nit": 0,
                "nfev": int(eval_counter["fresh_calls"]),
                "njev": 0,
                "optimizer_method": str(config.optimizer_method),
            },
        )

    bounds = [(-float(config.angle_bound), float(config.angle_bound))] * npar
    result = minimize(
        lambda x: _evaluate(x)[0],
        theta0,
        jac=lambda x: _evaluate(x)[1],
        method=str(config.optimizer_method),
        bounds=bounds,
        options={
            "maxiter": int(config.maxiter),
            "gtol": float(config.gtol),
            "ftol": float(config.ftol),
        },
    )
    objective, gradient, state, fidelity = _evaluate(np.asarray(result.x, dtype=float))
    return CheckpointFitStepResult(
        theta=np.asarray(result.x, dtype=float).reshape(-1),
        state=np.asarray(state, dtype=complex).copy(),
        fidelity=float(fidelity),
        objective=float(objective),
        solver_row={
            "success": bool(result.success),
            "status": int(getattr(result, "status", 0)),
            "message": str(getattr(result, "message", "")),
            "nit": int(getattr(result, "nit", 0)),
            "nfev": int(getattr(result, "nfev", eval_counter["fresh_calls"])),
            "njev": int(getattr(result, "njev", eval_counter["fresh_calls"])),
            "optimizer_method": str(config.optimizer_method),
            "objective": float(objective),
            "gradient_norm": float(np.linalg.norm(np.asarray(gradient, dtype=float))),
        },
    )


_MATH_FIT_TRAJECTORY = r"\theta^\star(t_n)\leftarrow \mathrm{warm\_start}(\theta^\star(t_{n-1}))"


def fit_checkpoint_trajectory(
    reference_state: np.ndarray,
    target_states: Sequence[np.ndarray],
    times: Sequence[float],
    terms: Sequence[AnsatzTerm],
    *,
    config: CheckpointFitConfig,
    theta_init: np.ndarray | None = None,
) -> CheckpointFitTrajectoryResult:
    time_vec = np.asarray(times, dtype=float).reshape(-1)
    if int(time_vec.size) != int(len(target_states)):
        raise ValueError("times and target_states must have the same length.")
    if int(time_vec.size) < 1:
        raise ValueError("Need at least one target checkpoint.")

    compiled = CompiledAnsatzExecutor(
        list(terms),
        coefficient_tolerance=float(config.coefficient_tolerance),
    )
    npar = int(len(terms))
    theta_prev = (
        np.zeros(npar, dtype=float)
        if theta_init is None
        else np.asarray(theta_init, dtype=float).reshape(-1)
    )
    if int(theta_prev.size) != npar:
        raise ValueError(f"theta_init length {int(theta_prev.size)} does not match term count {npar}.")

    states: list[np.ndarray] = []
    theta_rows: list[np.ndarray] = []
    solver_rows: list[dict[str, Any]] = []
    for idx, target_state in enumerate(target_states):
        fit_result = fit_checkpoint_target_state(
            reference_state,
            np.asarray(target_state, dtype=complex),
            terms,
            config=config,
            theta_init=theta_prev,
            executor=compiled,
        )
        theta_prev = np.asarray(fit_result.theta, dtype=float).copy()
        states.append(np.asarray(fit_result.state, dtype=complex))
        theta_rows.append(np.asarray(fit_result.theta, dtype=float))
        solver_rows.append(
            {
                "time": float(time_vec[idx]),
                "fidelity": float(fit_result.fidelity),
                "objective": float(fit_result.objective),
                **dict(fit_result.solver_row),
            }
        )

    return CheckpointFitTrajectoryResult(
        times=np.asarray(time_vec, dtype=float),
        theta_history=np.asarray(theta_rows, dtype=float),
        states=tuple(np.asarray(state, dtype=complex) for state in states),
        solver_rows=tuple(dict(row) for row in solver_rows),
    )


__all__ = [
    "CheckpointFitConfig",
    "CheckpointFitStepResult",
    "CheckpointFitTrajectoryResult",
    "LocalPauliAnsatzSpec",
    "build_local_pauli_ansatz_terms",
    "default_chain_edges",
    "fit_checkpoint_target_state",
    "fit_checkpoint_trajectory",
    "state_fidelity",
]

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_replay.py
```py
#!/usr/bin/env python3
"""Replay controllers for HH ADAPT-family continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.hh_continuation_types import (
    Phase2OptimizerMemoryAdapter,
    QNSPSARefreshPlan,
    ReplayPhaseTelemetry,
    ReplayPlan,
)


@dataclass(frozen=True)
class ReplayControllerConfig:
    freeze_fraction: float = 0.2
    unfreeze_fraction: float = 0.3
    full_fraction: float = 0.5
    trust_radius_initial: float = 0.1
    trust_radius_growth: float = 2.0
    trust_radius_max: float = 0.4
    qn_spsa_refresh_every: int = 0
    qn_spsa_refresh_mode: str = "diag_rms_grad"
    symmetry_mitigation_mode: str = "off"


class RestrictedAnsatzView:
    """Ansatz view that masks parameters outside an active set."""

    def __init__(
        self,
        *,
        base_ansatz: Any,
        base_point: np.ndarray,
        active_indices: Sequence[int],
    ) -> None:
        self._base_ansatz = base_ansatz
        self._base_point = np.asarray(base_point, dtype=float).reshape(-1)
        self._active = [int(i) for i in active_indices]
        self.num_parameters = int(len(self._active))

    def _merge(self, x: np.ndarray) -> np.ndarray:
        merged = np.array(self._base_point, copy=True)
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        for k, idx in enumerate(self._active):
            merged[idx] = float(x_arr[k])
        return merged

    def prepare_state(self, x: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        return self._base_ansatz.prepare_state(self._merge(x), psi_ref)

    def parameter_labels(self) -> list[str]:
        if hasattr(self._base_ansatz, "parameter_labels"):
            labels = list(getattr(self._base_ansatz, "parameter_labels")())
        else:
            labels = [f"theta_{i}" for i in range(int(self._base_point.size))]
        return [str(labels[i]) for i in self._active]


def build_replay_plan(
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
    total = max(3, int(maxiter))
    freeze_steps = max(1, int(round(float(cfg.freeze_fraction) * total)))
    unfreeze_steps = max(1, int(round(float(cfg.unfreeze_fraction) * total)))
    full_steps = max(1, total - freeze_steps - unfreeze_steps)
    trust_initial = float(cfg.trust_radius_initial)
    trust_growth = float(max(cfg.trust_radius_growth, 1.0))
    trust_max = float(max(cfg.trust_radius_max, trust_initial))
    trust = [
        trust_initial,
        min(trust_max, trust_initial * trust_growth),
        trust_max,
    ]
    return ReplayPlan(
        continuation_mode=str(continuation_mode),
        seed_policy_resolved=str(seed_policy_resolved),
        handoff_state_kind=str(handoff_state_kind),
        freeze_scaffold_steps=int(freeze_steps),
        unfreeze_steps=int(unfreeze_steps),
        full_replay_steps=int(full_steps),
        trust_radius_initial=float(trust_initial),
        trust_radius_growth=float(trust_growth),
        trust_radius_max=float(trust_max),
        scaffold_block_indices=[int(i) for i in scaffold_block_indices],
        residual_block_indices=[int(i) for i in residual_block_indices],
        qn_spsa_refresh_every=int(max(0, cfg.qn_spsa_refresh_every)),
        trust_radius_schedule=trust,
        optimizer_memory_source="unavailable",
        optimizer_memory_reused=False,
        refresh_mode=(
            str(cfg.qn_spsa_refresh_mode)
            if int(max(0, cfg.qn_spsa_refresh_every)) > 0
            else "disabled"
        ),
        symmetry_mitigation_mode=str(symmetry_mitigation_mode),
        generator_ids=[str(x) for x in list(generator_ids or [])],
        motif_reference_ids=[str(x) for x in list(motif_reference_ids or [])],
    )


def _run_phase(
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
    view = RestrictedAnsatzView(
        base_ansatz=ansatz,
        base_point=np.asarray(full_theta_seed, dtype=float),
        active_indices=active_indices,
    )
    x0 = np.asarray(full_theta_seed, dtype=float)[active_indices] if active_indices else np.zeros(0, dtype=float)
    result = vqe_minimize_fn(
        h_poly,
        view,
        psi_ref,
        restarts=int(restarts),
        seed=int(seed),
        initial_point=x0,
        use_initial_point_first_restart=True,
        method=str(method),
        maxiter=int(maxiter),
        progress_every_s=float(progress_every_s),
        track_history=True,
        optimizer_memory=(dict(optimizer_memory) if isinstance(optimizer_memory, Mapping) else None),
        spsa_refresh_every=(int(refresh_plan.refresh_every) if bool(refresh_plan.enabled) else 0),
        spsa_precondition_mode=(str(refresh_plan.mode) if bool(refresh_plan.enabled) else "none"),
        **kwargs,
    )
    theta_opt = getattr(result, "theta", None)
    if theta_opt is None:
        theta_opt = getattr(result, "x")
    x_opt = np.asarray(theta_opt, dtype=float).reshape(-1)
    full = np.asarray(full_theta_seed, dtype=float).copy()
    for k, idx in enumerate(active_indices):
        full[idx] = float(x_opt[k])

    e_before = float(result.restart_summaries[0]["best_energy"]) if getattr(result, "restart_summaries", None) else float(result.energy)
    e_after = float(result.energy)
    d_before = float(abs(e_before - exact_energy)) if exact_energy is not None else None
    d_after = float(abs(e_after - exact_energy)) if exact_energy is not None else None
    tel = ReplayPhaseTelemetry(
        phase_name=str(phase_name),
        nfev=int(getattr(result, "nfev", 0)),
        nit=int(getattr(result, "nit", 0)),
        success=bool(getattr(result, "success", False)),
        energy_before=float(e_before),
        energy_after=float(e_after),
        delta_abs_before=d_before,
        delta_abs_after=d_after,
        active_count=int(len(active_indices)),
        frozen_count=int(len(full) - len(active_indices)),
        optimizer_memory_reused=bool(
            isinstance(optimizer_memory, Mapping) and bool(optimizer_memory.get("reused", False))
        ),
        optimizer_memory_source=str(
            optimizer_memory.get("source", "unavailable")
            if isinstance(optimizer_memory, Mapping)
            else "unavailable"
        ),
        qn_spsa_refresh_points=[
            int(x) for x in getattr(result, "optimizer_memory", {}) .get("refresh_points", [])
        ] if hasattr(result, "optimizer_memory") and isinstance(getattr(result, "optimizer_memory"), Mapping) else [],
    )
    return full, tel, result


def _refresh_plan_for_phase(
    *,
    phase_name: str,
    method: str,
    cfg: ReplayControllerConfig,
) -> QNSPSARefreshPlan:
    if str(method).strip().lower() != "spsa":
        return QNSPSARefreshPlan(enabled=False, refresh_every=0, mode="disabled", skip_reason="method_not_spsa")
    if str(phase_name) != "constrained_unfreeze":
        return QNSPSARefreshPlan(enabled=False, refresh_every=0, mode="disabled", skip_reason="phase_not_refreshed")
    cadence = int(max(0, cfg.qn_spsa_refresh_every))
    if cadence <= 0:
        return QNSPSARefreshPlan(enabled=False, refresh_every=0, mode="disabled", skip_reason="refresh_disabled")
    return QNSPSARefreshPlan(
        enabled=True,
        refresh_every=int(cadence),
        mode=str(cfg.qn_spsa_refresh_mode),
    )


def _seed_full_optimizer_memory(
    *,
    adapter: Phase2OptimizerMemoryAdapter,
    incoming_memory: Mapping[str, Any] | None,
    total_parameters: int,
    scaffold_block_size: int,
) -> tuple[dict[str, Any], str, bool]:
    total_n = int(max(0, total_parameters))
    scaffold_n = int(max(0, min(scaffold_block_size, total_n)))
    incoming_n = int(incoming_memory.get("parameter_count", 0)) if isinstance(incoming_memory, Mapping) else 0
    if not isinstance(incoming_memory, Mapping):
        return (
            adapter.unavailable(method="SPSA", parameter_count=total_n, reason="missing_handoff_optimizer_memory"),
            "missing_handoff_optimizer_memory",
            False,
        )
    if incoming_n == total_n:
        state = adapter.from_result(
            type("MemoryCarrier", (), {"optimizer_memory": dict(incoming_memory)})(),
            method=str(incoming_memory.get("optimizer", "SPSA")),
            parameter_count=int(total_n),
            source="handoff_full",
        )
        state["reused"] = bool(state.get("available", False))
        return state, "handoff_full", bool(state.get("reused", False))
    if incoming_n == scaffold_n:
        base = adapter.unavailable(method="SPSA", parameter_count=total_n, reason="replay_block_expansion")
        scaffold_state = adapter.from_result(
            type("MemoryCarrier", (), {"optimizer_memory": dict(incoming_memory)})(),
            method=str(incoming_memory.get("optimizer", "SPSA")),
            parameter_count=int(scaffold_n),
            source="handoff_scaffold",
        )
        merged = adapter.merge_active(
            base,
            active_indices=list(range(scaffold_n)),
            active_state=scaffold_state,
            source="handoff_scaffold_expand",
        )
        merged["reused"] = bool(scaffold_state.get("available", False))
        return merged, "handoff_scaffold_expand", bool(merged.get("reused", False))
    normalized = adapter.from_result(
        type("MemoryCarrier", (), {"optimizer_memory": dict(incoming_memory)})(),
        method=str(incoming_memory.get("optimizer", "SPSA")),
        parameter_count=int(total_n),
        source="handoff_resized",
    )
    normalized["reused"] = bool(normalized.get("available", False))
    return normalized, "handoff_resized", bool(normalized.get("reused", False))


def _run_phase_replay_controller(
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
    extra = dict(kwargs or {})
    theta_cur = np.asarray(seed_theta, dtype=float).reshape(-1).copy()
    npar = int(theta_cur.size)
    scaffold_n = max(0, min(int(scaffold_block_size), npar))
    residual_indices = [i for i in range(scaffold_n, npar)]
    scaffold_indices = [i for i in range(scaffold_n)]
    plan = build_replay_plan(
        continuation_mode=str(continuation_mode),
        seed_policy_resolved=str(seed_policy_resolved),
        handoff_state_kind=str(handoff_state_kind),
        scaffold_block_indices=scaffold_indices,
        residual_block_indices=residual_indices,
        maxiter=int(maxiter),
        cfg=cfg,
        symmetry_mitigation_mode=str(symmetry_mitigation_mode),
        generator_ids=list(generator_ids or []),
        motif_reference_ids=list(motif_reference_ids or []),
    )
    history: list[dict[str, Any]] = []
    adapter = Phase2OptimizerMemoryAdapter()
    full_memory, memory_source, memory_reused = _seed_full_optimizer_memory(
        adapter=adapter,
        incoming_memory=incoming_optimizer_memory,
        total_parameters=int(npar),
        scaffold_block_size=int(scaffold_n),
    )
    plan = ReplayPlan(
        **{
            **plan.__dict__,
            "optimizer_memory_source": str(memory_source),
            "optimizer_memory_reused": bool(memory_reused),
        }
    )
    refresh_points_total: list[int] = []
    refresh_plans: list[QNSPSARefreshPlan] = []

    phase_specs = [
        ("seed_burn_in", residual_indices, int(plan.freeze_scaffold_steps), 1, int(seed)),
        (
            "constrained_unfreeze",
            scaffold_indices[-max(0, min(len(scaffold_indices), max(1, len(scaffold_indices) // 3))):] + residual_indices,
            int(plan.unfreeze_steps),
            1,
            int(seed) + 1,
        ),
        ("full_replay", list(range(npar)), int(plan.full_replay_steps), int(restarts), int(seed) + 2),
    ]

    last_result: Any = None
    for phase_name, active_indices, phase_steps, phase_restarts, phase_seed in phase_specs:
        active_memory = adapter.select_active(
            full_memory,
            active_indices=list(active_indices),
            source=f"{continuation_mode}.{phase_name}.active_subset",
        )
        refresh_plan = _refresh_plan_for_phase(
            phase_name=str(phase_name),
            method=str(method),
            cfg=cfg,
        )
        refresh_plans.append(refresh_plan)
        if (not active_indices) or int(phase_steps) <= 0:
            tel = ReplayPhaseTelemetry(
                phase_name=str(phase_name),
                nfev=0,
                nit=0,
                success=True,
                energy_before=float("nan"),
                energy_after=float("nan"),
                delta_abs_before=None,
                delta_abs_after=None,
                active_count=int(len(active_indices)),
                frozen_count=int(npar - len(active_indices)),
                optimizer_memory_reused=bool(
                    isinstance(active_memory, Mapping) and bool(active_memory.get("reused", False))
                ),
                optimizer_memory_source=str(
                    active_memory.get("source", "unavailable")
                    if isinstance(active_memory, Mapping)
                    else "unavailable"
                ),
                qn_spsa_refresh_points=[],
            )
            history.append(tel.__dict__)
            continue
        theta_cur, tel, result = _run_phase(
            phase_name=str(phase_name),
            vqe_minimize_fn=vqe_minimize_fn,
            h_poly=h_poly,
            ansatz=ansatz,
            psi_ref=psi_ref,
            active_indices=list(active_indices),
            full_theta_seed=theta_cur,
            restarts=int(phase_restarts),
            seed=int(phase_seed),
            maxiter=int(phase_steps),
            method=str(method),
            progress_every_s=float(progress_every_s),
            exact_energy=exact_energy,
            kwargs=extra,
            optimizer_memory=active_memory,
            refresh_plan=refresh_plan,
        )
        last_result = result
        merged_memory = adapter.from_result(
            result,
            method=str(method),
            parameter_count=int(len(active_indices)),
            source=f"{continuation_mode}.{phase_name}.result",
        )
        full_memory = adapter.merge_active(
            full_memory,
            active_indices=list(active_indices),
            active_state=merged_memory,
            source=f"{continuation_mode}.{phase_name}.merge",
        )
        refresh_points = [
            int(x)
            for x in (
                getattr(result, "optimizer_memory", {}).get("refresh_points", [])
                if hasattr(result, "optimizer_memory") and isinstance(getattr(result, "optimizer_memory"), Mapping)
                else []
            )
        ]
        refresh_points_total.extend(x for x in refresh_points if x not in refresh_points_total)
        history.append(tel.__dict__)

    if last_result is None:
        class _ReplayNoopResult:
            energy = float("nan")
            nfev = 0
            nit = 0
            success = True
            message = "no_active_replay_phases"

        last_result = _ReplayNoopResult()

    replay_meta = {
        "replay_phase_config": {
            "continuation_mode": str(plan.continuation_mode),
            "seed_policy_resolved": str(seed_policy_resolved),
            "handoff_state_kind": str(handoff_state_kind),
            "freeze_scaffold_steps": int(plan.freeze_scaffold_steps),
            "unfreeze_steps": int(plan.unfreeze_steps),
            "full_replay_steps": int(plan.full_replay_steps),
            "trust_radius_initial": float(plan.trust_radius_initial),
            "trust_radius_growth": float(plan.trust_radius_growth),
            "trust_radius_max": float(plan.trust_radius_max),
            "scaffold_block_indices": [int(i) for i in plan.scaffold_block_indices],
            "residual_block_indices": [int(i) for i in plan.residual_block_indices],
            "qn_spsa_refresh_every": int(plan.qn_spsa_refresh_every),
            "trust_radius_schedule": [float(x) for x in plan.trust_radius_schedule],
            "optimizer_memory_source": str(plan.optimizer_memory_source),
            "optimizer_memory_reused": bool(plan.optimizer_memory_reused),
            "symmetry_mitigation_mode": str(plan.symmetry_mitigation_mode),
            "generator_ids": [str(x) for x in plan.generator_ids],
            "motif_reference_ids": [str(x) for x in plan.motif_reference_ids],
            "optimizer_memory": dict(full_memory),
            "residual_zero_initialized": True,
            "qn_spsa_refresh": {
                "enabled": any(bool(rp.enabled) for rp in refresh_plans),
                "refresh_every": int(plan.qn_spsa_refresh_every),
                "mode": str(plan.refresh_mode),
                "refresh_points": [int(x) for x in refresh_points_total],
                "phase_plans": [dict(rp.__dict__) for rp in refresh_plans],
            },
        },
        "result": {
            "energy": float(getattr(last_result, "energy", float("nan"))),
            "nfev": int(getattr(last_result, "nfev", 0)),
            "nit": int(getattr(last_result, "nit", 0)),
            "success": bool(getattr(last_result, "success", False)),
            "message": str(getattr(last_result, "message", "")),
        },
    }
    return theta_cur, history, replay_meta


def run_phase1_replay(
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
    """Built-in math expression:
    replay = burn_in(residual) -> constrained_unfreeze -> full
    """
    phase1_cfg = ReplayControllerConfig(
        freeze_fraction=float(cfg.freeze_fraction),
        unfreeze_fraction=float(cfg.unfreeze_fraction),
        full_fraction=float(cfg.full_fraction),
        trust_radius_initial=float(cfg.trust_radius_initial),
        trust_radius_growth=float(cfg.trust_radius_growth),
        trust_radius_max=float(cfg.trust_radius_max),
        qn_spsa_refresh_every=0,
        qn_spsa_refresh_mode="disabled",
    )
    return _run_phase_replay_controller(
        continuation_mode="phase1_v1",
        vqe_minimize_fn=vqe_minimize_fn,
        h_poly=h_poly,
        ansatz=ansatz,
        psi_ref=psi_ref,
        seed_theta=seed_theta,
        scaffold_block_size=scaffold_block_size,
        seed_policy_resolved=seed_policy_resolved,
        handoff_state_kind=handoff_state_kind,
        cfg=phase1_cfg,
        restarts=restarts,
        seed=seed,
        maxiter=maxiter,
        method=method,
        progress_every_s=progress_every_s,
        exact_energy=exact_energy,
        kwargs=kwargs,
        incoming_optimizer_memory=None,
        symmetry_mitigation_mode="off",
        generator_ids=None,
        motif_reference_ids=None,
    )


def run_phase2_replay(
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
    return _run_phase_replay_controller(
        continuation_mode="phase2_v1",
        vqe_minimize_fn=vqe_minimize_fn,
        h_poly=h_poly,
        ansatz=ansatz,
        psi_ref=psi_ref,
        seed_theta=seed_theta,
        scaffold_block_size=scaffold_block_size,
        seed_policy_resolved=seed_policy_resolved,
        handoff_state_kind=handoff_state_kind,
        cfg=cfg,
        restarts=restarts,
        seed=seed,
        maxiter=maxiter,
        method=method,
        progress_every_s=progress_every_s,
        exact_energy=exact_energy,
        kwargs=kwargs,
        incoming_optimizer_memory=incoming_optimizer_memory,
        symmetry_mitigation_mode=str(cfg.symmetry_mitigation_mode),
        generator_ids=None,
        motif_reference_ids=None,
    )


def run_phase3_replay(
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
    return _run_phase_replay_controller(
        continuation_mode="phase3_v1",
        vqe_minimize_fn=vqe_minimize_fn,
        h_poly=h_poly,
        ansatz=ansatz,
        psi_ref=psi_ref,
        seed_theta=seed_theta,
        scaffold_block_size=scaffold_block_size,
        seed_policy_resolved=seed_policy_resolved,
        handoff_state_kind=handoff_state_kind,
        cfg=cfg,
        restarts=restarts,
        seed=seed,
        maxiter=maxiter,
        method=method,
        progress_every_s=progress_every_s,
        exact_energy=exact_energy,
        kwargs=kwargs,
        incoming_optimizer_memory=incoming_optimizer_memory,
        symmetry_mitigation_mode=str(cfg.symmetry_mitigation_mode),
        generator_ids=list(generator_ids or []),
        motif_reference_ids=list(motif_reference_ids or []),
    )

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_hh_continuation_scoring.py
```py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_generators import build_generator_metadata
from pipelines.hardcoded.hh_continuation_scoring import (
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
from pipelines.hardcoded.hh_continuation_symmetry import build_symmetry_spec
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm


def _term(label: str) -> object:
    return type(
        "_DummyAnsatzTerm",
        (),
        {"label": str(label), "polynomial": PauliPolynomial("JW", [PauliTerm(1, ps=str(label), pc=1.0)])},
    )()


def _feat(
    *,
    gradient_signed: float = 0.4,
    metric_proxy: float = 0.5,
    sigma_hat: float = 0.0,
    refit_window_indices: list[int] | None = None,
    family_repeat_cost: float = 0.0,
    stage_gate_open: bool = True,
    cfg: SimpleScoreConfig | None = None,
) -> object:
    cfg_use = cfg or SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.0, z_alpha=0.0)
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    return build_candidate_features(
        stage_name="core",
        candidate_label="cand",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=float(gradient_signed),
        metric_proxy=float(metric_proxy),
        sigma_hat=float(sigma_hat),
        refit_window_indices=list(refit_window_indices or []),
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=len(refit_window_indices or [])),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=bool(stage_gate_open),
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg_use,
        family_repeat_cost=float(family_repeat_cost),
    )


def test_simple_v1_prefers_higher_gradient_with_equal_costs() -> None:
    cfg = SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.0, z_alpha=0.0)
    feat_a = _feat(gradient_signed=0.4, metric_proxy=0.5, cfg=cfg)
    feat_b = _feat(gradient_signed=0.2, metric_proxy=0.5, cfg=cfg)
    assert float(feat_a.simple_score or 0.0) > float(feat_b.simple_score or 0.0)


def test_stage_gate_blocks_score() -> None:
    feat = _feat(stage_gate_open=False)
    assert feat.simple_score == float("-inf")


def test_simple_v1_uses_g_lcb_not_g_abs() -> None:
    cfg = SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.0, z_alpha=10.0)
    feat = _feat(gradient_signed=0.4, metric_proxy=0.5, sigma_hat=0.03, cfg=cfg)
    assert float(feat.g_lcb) == pytest.approx(0.1)
    assert float(feat.simple_score or 0.0) == pytest.approx(0.5 * 0.1 * 0.1 / 0.5)


def test_family_repeat_cost_lowers_screen_score() -> None:
    cfg = SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.2, z_alpha=0.0)
    feat_a = _feat(gradient_signed=0.4, metric_proxy=0.5, family_repeat_cost=0.0, cfg=cfg)
    feat_b = _feat(gradient_signed=0.4, metric_proxy=0.5, family_repeat_cost=2.0, cfg=cfg)
    assert float(feat_a.simple_score or 0.0) > float(feat_b.simple_score or 0.0)


def test_measurement_cache_reuse_accounting() -> None:
    cache = MeasurementCacheAudit(nominal_shots_per_group=10)
    first = cache.estimate(["a", "b"])
    assert first.groups_new == 2
    cache.commit(["a", "b"])
    second = cache.estimate(["a", "b", "c"])
    assert second.groups_reused == 2
    assert second.groups_new == 1
    summary = cache.summary()
    assert str(summary["plan_version"]) == "phase1_grouped_label_reuse"


def test_measurement_cache_clone_isolated_from_parent_and_sibling() -> None:
    cache = MeasurementCacheAudit(nominal_shots_per_group=10)
    cache.commit(["a"])
    child_a = cache.clone()
    child_b = cache.clone()

    child_a.commit(["b"])
    child_b.commit(["c"])

    parent_stats = cache.estimate(["a", "b", "c"])
    child_a_stats = child_a.estimate(["a", "b", "c"])
    child_b_stats = child_b.estimate(["a", "b", "c"])

    assert parent_stats.groups_reused == 1
    assert child_a_stats.groups_reused == 2
    assert child_b_stats.groups_reused == 2


def test_measurement_cache_snapshot_roundtrip() -> None:
    cache = MeasurementCacheAudit(nominal_shots_per_group=7)
    cache.commit(["alpha", "beta"])
    restored = MeasurementCacheAudit.from_snapshot(cache.snapshot())

    assert restored.summary() == cache.summary()
    assert restored.estimate(["alpha", "beta", "gamma"]).groups_reused == 2
    assert restored.estimate(["alpha", "beta", "gamma"]).groups_new == 1


def test_trust_region_drop_matches_newton_branch() -> None:
    got = trust_region_drop(0.4, 2.0, 1.0, 1.0)
    assert got == pytest.approx(0.04)


def test_full_v2_uses_reduced_fields() -> None:
    cfg = FullScoreConfig(z_alpha=0.0, rho=1.0, gamma_N=1.0, wD=0.0, wG=0.0, wC=0.0, wc=0.0)
    feat = _feat(gradient_signed=0.5, metric_proxy=0.5, refit_window_indices=[0])
    feat = type(feat)(
        **{
            **feat.__dict__,
            "novelty": 0.5,
            "h_eff": 2.0,
            "F_red": 1.0,
            "h_raw": 2.0,
            "ridge_used": cfg.lambda_H,
        }
    )
    score, fallback = full_v2_score(feat, cfg)
    assert score == pytest.approx(0.03125)
    assert fallback == "append_exact_reduced_path"


def test_full_v2_zeroes_metric_collapse() -> None:
    cfg = FullScoreConfig(z_alpha=0.0, rho=1.0, gamma_N=1.0, wD=0.0, wG=0.0, wC=0.0, wc=0.0)
    feat = _feat(gradient_signed=0.5, metric_proxy=0.5, refit_window_indices=[0])
    feat = type(feat)(
        **{
            **feat.__dict__,
            "novelty": 0.0,
            "h_eff": 0.0,
            "F_red": cfg.metric_floor,
            "h_raw": 0.0,
            "ridge_used": cfg.lambda_H,
            "curvature_mode": "append_exact_metric_collapse_v1",
        }
    )
    score, fallback = full_v2_score(feat, cfg)
    assert score == 0.0
    assert fallback == "reduced_metric_collapse"


def test_full_v2_ignores_motif_bonus_in_active_score() -> None:
    cfg = FullScoreConfig(z_alpha=0.0, rho=1.0, gamma_N=1.0, wD=0.0, wG=0.0, wC=0.0, wc=0.0)
    feat = _feat(gradient_signed=0.5, metric_proxy=0.5, refit_window_indices=[0])
    feat_base = type(feat)(
        **{
            **feat.__dict__,
            "novelty": 1.0,
            "h_eff": 1.0,
            "F_red": 1.0,
            "h_raw": 1.0,
            "ridge_used": cfg.lambda_H,
            "motif_bonus": 0.0,
        }
    )
    feat_bonus = type(feat_base)(**{**feat_base.__dict__, "motif_bonus": 10.0})
    score_a, _ = full_v2_score(feat_base, cfg)
    score_b, _ = full_v2_score(feat_bonus, cfg)
    assert score_a == pytest.approx(score_b)


def test_build_full_candidate_features_emits_reduced_path_fields() -> None:
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    h_poly = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])
    h_compiled = compile_polynomial_action(h_poly)
    selected_ops = [_term("x")]
    theta = np.asarray([0.2], dtype=float)
    executor = CompiledAnsatzExecutor(selected_ops, pauli_action_cache={})
    psi_state = executor.prepare_state(theta, psi_ref)
    hpsi_state = apply_compiled_polynomial(psi_state, h_compiled)

    base = _feat(
        gradient_signed=0.3,
        metric_proxy=0.3,
        refit_window_indices=[0],
        cfg=SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.0),
    )
    novelty_oracle = Phase2NoveltyOracle()
    scaffold_context = novelty_oracle.prepare_scaffold_context(
        selected_ops=selected_ops,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        hpsi_state=hpsi_state,
        refit_window_indices=[0],
        pauli_action_cache={},
    )
    feat = build_full_candidate_features(
        base_feature=base,
        candidate_term=_term("y"),
        cfg=FullScoreConfig(shortlist_size=2),
        novelty_oracle=novelty_oracle,
        curvature_oracle=Phase2CurvatureOracle(),
        scaffold_context=scaffold_context,
        h_compiled=h_compiled,
        compiled_cache={},
        pauli_action_cache={},
        optimizer_memory=None,
    )
    assert 0.0 <= float(feat.novelty or 0.0) <= 1.0
    assert feat.refit_window_indices == [0]
    assert feat.F_raw is not None and feat.F_raw >= 0.0
    assert feat.F_red is not None and feat.F_red > 0.0
    assert feat.Q_window is not None
    assert feat.H_window_hessian is not None
    assert feat.h_eff is not None
    assert feat.full_v2_score is not None


def test_shortlist_only_expensive_scoring_calls_oracles_for_shortlist() -> None:
    class _CountingNovelty(Phase2NoveltyOracle):
        def __init__(self) -> None:
            self.calls = 0

        def estimate(self, *args, **kwargs):
            self.calls += 1
            return super().estimate(*args, **kwargs)

    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    h_poly = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])
    h_compiled = compile_polynomial_action(h_poly)
    hpsi_state = apply_compiled_polynomial(psi_ref, h_compiled)
    novelty = _CountingNovelty()
    scaffold_context = novelty.prepare_scaffold_context(
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi_ref,
        psi_state=psi_ref,
        h_compiled=h_compiled,
        hpsi_state=hpsi_state,
        refit_window_indices=[],
        pauli_action_cache={},
    )

    cheap_records = []
    for idx, grad in enumerate([0.9, 0.8, 0.3, 0.2]):
        feat = _feat(
            gradient_signed=float(grad),
            metric_proxy=1.0,
            refit_window_indices=[],
            cfg=SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0),
        )
        cheap_records.append(
            {
                "feature": feat,
                "simple_score": float(feat.simple_score or 0.0),
                "candidate_pool_index": idx,
                "position_id": 0,
                "candidate_term": _term("x"),
            }
        )
    shortlisted = shortlist_records(cheap_records, cfg=FullScoreConfig(shortlist_fraction=0.5, shortlist_size=2))
    for rec in shortlisted:
        build_full_candidate_features(
            base_feature=rec["feature"],
            candidate_term=rec["candidate_term"],
            cfg=FullScoreConfig(shortlist_fraction=0.5, shortlist_size=2),
            novelty_oracle=novelty,
            curvature_oracle=Phase2CurvatureOracle(),
            scaffold_context=scaffold_context,
            h_compiled=h_compiled,
            compiled_cache={},
            pauli_action_cache={},
            optimizer_memory=None,
        )
    assert len(shortlisted) == 2
    assert novelty.calls == 2


def test_remaining_evaluations_proxy_uses_remaining_depth_mode() -> None:
    got = remaining_evaluations_proxy(current_depth=2, max_depth=6, mode="remaining_depth")
    assert got == pytest.approx(5.0)


def test_lifetime_weight_components_are_zero_when_mode_off() -> None:
    cfg = FullScoreConfig(lifetime_cost_mode="off")
    feat = _feat(
        gradient_signed=0.5,
        metric_proxy=0.5,
        refit_window_indices=[0],
        cfg=SimpleScoreConfig(),
    )
    feat = type(feat)(
        **{
            **feat.__dict__,
            "remaining_evaluations_proxy": 5.0,
            "lifetime_cost_mode": "off",
        }
    )
    comps = lifetime_weight_components(feat, cfg)
    assert comps["remaining_evaluations_proxy"] == pytest.approx(5.0)
    assert comps["total"] == pytest.approx(0.0)


def test_family_repeat_cost_from_history_uses_consecutive_streak() -> None:
    history = [
        {"candidate_family": "a"},
        {"candidate_family": "b"},
        {"candidate_family": "b"},
    ]
    assert family_repeat_cost_from_history(history_rows=history, candidate_family="a") == pytest.approx(0.0)
    assert family_repeat_cost_from_history(history_rows=history, candidate_family="b") == pytest.approx(2.0)


def test_build_candidate_features_carries_generator_and_symmetry_metadata() -> None:
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(6, ps="eyeexy", pc=1.0),
            PauliTerm(6, ps="eyeeyx", pc=-1.0),
        ],
    )
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    meta = build_generator_metadata(
        label="macro_candidate",
        polynomial=poly,
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="macro_candidate",
        candidate_family="paop_lf_std",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.4,
        metric_proxy=0.4,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=2, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["macro"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
        generator_metadata=meta.__dict__,
        symmetry_spec=sym.__dict__,
        symmetry_mode="phase3_shared_spec",
        symmetry_mitigation_mode="verify_only",
        current_depth=0,
        max_depth=3,
        lifetime_cost_mode="phase3_v1",
        remaining_evaluations_proxy_mode="remaining_depth",
    )
    assert feat.generator_id == meta.generator_id
    assert feat.template_id == meta.template_id
    assert feat.is_macro_generator is True
    assert feat.symmetry_mode == "phase3_shared_spec"
    assert feat.symmetry_mitigation_mode == "verify_only"
    assert feat.remaining_evaluations_proxy == pytest.approx(4.0)

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/time_propagation/cfqm_schemes.py
```py
"""CFQM (commutator-free quasi-Magnus) scheme registry.

Public API
----------
get_cfqm_scheme(scheme_id: str) -> dict

Accepted scheme ids (case-insensitive):
- CF4:2: ``"cfqm4"``, ``"cf4:2"``, ``"cf4"``
- CF6:5Opt: ``"cfqm6"``, ``"cf6:5opt"``, ``"cf6"``
"""

from __future__ import annotations

from typing import Any


_DEFAULT_TOL = 1e-12


_SCHEME_DATA: dict[str, dict[str, Any]] = {
    "CF4:2": {
        "name": "CF4:2",
        "order": 4,
        "c": [
            0.2113248654051871,
            0.7886751345948129,
        ],
        "a": [
            [-0.0386751345948129, 0.5386751345948129],
            [0.5386751345948129, -0.0386751345948129],
        ],
        "_expected_row_sums": [
            0.5,
            0.5,
        ],
    },
    "CF6:5Opt": {
        "name": "CF6:5Opt",
        "order": 6,
        "c": [
            0.0694318442029737123880,
            0.3300094782075718675987,
            0.6699905217924281324013,
            0.9305681557970262876120,
        ],
        "a": [
            [-0.0025014052514919785, 0.0086390299226631599, -0.0241007202550846312, 0.1893630955839134498],
            [0.0118125500877527529, -0.0524780954434914029, 0.4364266743440699121, -0.0207973857888688969],
            [-0.0039494320625783987, -0.0424143111368839665, -0.0424143111368839665, -0.0039494320625783987],
            [-0.0207973857888688969, 0.4364266743440699121, -0.0524780954434914029, 0.0118125500877527529],
            [0.1893630955839134498, -0.0241007202550846312, 0.0086390299226631599, -0.0025014052514919785],
        ],
        "_expected_row_sums": [
            0.1714,
            0.37496374319946236513,
            -0.09272748639892473026,
            0.37496374319946236513,
            0.1714,
        ],
        "_expected_col_sums": [
            0.1739274225687269286865,
            0.3260725774312730713135,
            0.3260725774312730713135,
            0.1739274225687269286865,
        ],
    },
}


_SCHEME_ALIASES = {
    "cfqm4": "CF4:2",
    "cf4:2": "CF4:2",
    "cf4": "CF4:2",
    "cfqm6": "CF6:5Opt",
    "cf6:5opt": "CF6:5Opt",
    "cf6": "CF6:5Opt",
}


def _norm_key(scheme_id: str) -> str:
    return str(scheme_id).strip().lower()


def _close(lhs: float, rhs: float, tol: float) -> bool:
    return abs(float(lhs) - float(rhs)) <= float(tol)


def _row_sums(a: list[list[float]]) -> list[float]:
    return [float(sum(row)) for row in a]


def _col_sums(a: list[list[float]]) -> list[float]:
    if not a:
        return []
    m = len(a[0])
    out = [0.0] * m
    for row in a:
        for j, coeff in enumerate(row):
            out[j] += float(coeff)
    return [float(x) for x in out]


def validate_scheme(scheme: dict[str, Any], *, tol: float = _DEFAULT_TOL) -> None:
    """Validate CFQM scheme data and invariants."""
    required = ("name", "order", "c", "a", "s_static")
    for key in required:
        if key not in scheme:
            raise ValueError(f"Scheme missing required field: {key!r}")

    name = str(scheme["name"])
    order = int(scheme["order"])
    c = scheme["c"]
    a = scheme["a"]
    s_static = scheme["s_static"]

    if order <= 0:
        raise ValueError(f"{name}: order must be positive.")
    if not isinstance(c, list) or not c:
        raise ValueError(f"{name}: c must be a non-empty list.")
    if not isinstance(a, list) or not a:
        raise ValueError(f"{name}: a must be a non-empty list of rows.")

    m = len(c)
    s = len(a)
    if not isinstance(s_static, list) or len(s_static) != s:
        raise ValueError(f"{name}: s_static length must equal stage count.")

    for idx, node in enumerate(c):
        node_f = float(node)
        if node_f < -tol or node_f > 1.0 + tol:
            raise ValueError(f"{name}: node c[{idx}]={node_f} not in [0,1].")

    for k, row in enumerate(a):
        if not isinstance(row, list) or len(row) != m:
            raise ValueError(f"{name}: row a[{k}] must have length {m}.")

    total = float(sum(sum(float(x) for x in row) for row in a))
    if not _close(total, 1.0, tol):
        raise ValueError(f"{name}: sum(all a)={total} violates invariant == 1.")

    computed_rows = _row_sums(a)
    for k, (given_row_sum, computed_row_sum) in enumerate(zip(s_static, computed_rows)):
        if not _close(float(given_row_sum), float(computed_row_sum), tol):
            raise ValueError(
                f"{name}: s_static[{k}]={given_row_sum} does not match row-sum {computed_row_sum}."
            )

    expected_rows = scheme.get("_expected_row_sums")
    if expected_rows is not None:
        if len(expected_rows) != len(computed_rows):
            raise ValueError(f"{name}: expected row-sum length mismatch.")
        for k, (expected, computed) in enumerate(zip(expected_rows, computed_rows)):
            if not _close(float(expected), float(computed), tol):
                raise ValueError(
                    f"{name}: row-sum[{k}]={computed} does not match expected {expected}."
                )

    expected_cols = scheme.get("_expected_col_sums")
    if expected_cols is not None:
        computed_cols = _col_sums(a)
        if len(expected_cols) != len(computed_cols):
            raise ValueError(f"{name}: expected column-sum length mismatch.")
        for j, (expected, computed) in enumerate(zip(expected_cols, computed_cols)):
            if not _close(float(expected), float(computed), tol):
                raise ValueError(
                    f"{name}: col-sum[{j}]={computed} does not match expected {expected}."
                )


def get_cfqm_scheme(scheme_id: str) -> dict[str, Any]:
    """Return validated CFQM scheme data for *scheme_id*.

    Returned fields:
    - name: str
    - order: int
    - c: list[float]
    - a: list[list[float]]
    - s_static: list[float]
    """
    key = _norm_key(scheme_id)
    canonical = _SCHEME_ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(sorted(_SCHEME_ALIASES))
        raise ValueError(f"Unknown CFQM scheme_id={scheme_id!r}. Supported ids: {supported}.")

    data = _SCHEME_DATA[canonical]
    c = [float(x) for x in data["c"]]
    a = [[float(v) for v in row] for row in data["a"]]
    out: dict[str, Any] = {
        "name": str(data["name"]),
        "order": int(data["order"]),
        "c": c,
        "a": a,
        "s_static": _row_sums(a),
    }

    if "_expected_row_sums" in data:
        out["_expected_row_sums"] = [float(x) for x in data["_expected_row_sums"]]
    if "_expected_col_sums" in data:
        out["_expected_col_sums"] = [float(x) for x in data["_expected_col_sums"]]

    validate_scheme(out)
    out.pop("_expected_row_sums", None)
    out.pop("_expected_col_sums", None)
    return out


```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_hh_adapt_beam_search.py
```py
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.adapt_pipeline import _BeamBranchState, _run_hardcoded_adapt_vqe
from pipelines.hardcoded.hh_continuation_scoring import MeasurementCacheAudit
from pipelines.hardcoded.hh_continuation_stage_control import StageController, StageControllerConfig
from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian


def _make_branch() -> _BeamBranchState:
    stage = StageController(StageControllerConfig())
    stage.start_with_seed()
    measure_cache = MeasurementCacheAudit(nominal_shots_per_group=4)
    measure_cache.commit(["parent_group"])
    return _BeamBranchState(
        branch_id=11,
        parent_branch_id=None,
        depth_local=2,
        terminated=False,
        stop_reason=None,
        selected_ops=[SimpleNamespace(label="op_a"), SimpleNamespace(label="op_b")],
        theta=np.array([0.1, 0.2], dtype=float),
        energy_current=-1.25,
        available_indices={1, 2, 3},
        selection_counts=np.array([0, 1, 0, 2], dtype=np.int64),
        history=[{"depth": 1, "nested": {"path": ["parent"]}}],
        phase1_stage=stage,
        phase1_residual_opened=False,
        phase1_last_probe_reason="append_only",
        phase1_last_positions_considered=[2],
        phase1_last_trough_detected=False,
        phase1_last_trough_probe_triggered=False,
        phase1_last_selected_score=0.5,
        phase1_features_history=[{"nested": {"scores": [1.0]}}],
        phase1_stage_events=[{"meta": {"reason": "seed_complete"}}],
        phase1_measure_cache=measure_cache,
        phase2_optimizer_memory={"nested": {"weights": [1.0, 2.0]}, "slots": {"0": {"mean": 0.1}}},
        phase2_last_shortlist_records=[{"ids": [1], "feature": {"simple_score": 1.0}}],
        phase2_last_batch_selected=False,
        phase2_last_batch_penalty_total=0.0,
        phase2_last_optimizer_memory_reused=False,
        phase2_last_optimizer_memory_source="parent",
        phase2_last_shortlist_eval_records=[{"eval": {"kept": True}}],
        drop_prev_delta_abs=0.25,
        drop_plateau_hits=1,
        eps_energy_low_streak=0,
        phase3_split_events=[{"meta": {"selected": ["x"]}}],
        phase3_runtime_split_summary={"counts": {"selected": 1}, "selected_child_labels": ["x"]},
        phase3_motif_usage={"used": {"motif_a": 1}},
        phase3_rescue_history=[{"rescue": {"winner": "x"}}],
        nfev_total_local=17,
    )


def _hh_h() -> object:
    return build_hubbard_holstein_hamiltonian(
        dims=2,
        J=1.0,
        U=2.0,
        omega0=1.0,
        g=1.0,
        n_ph_max=1,
        boson_encoding="binary",
        v_t=None,
        v0=0.0,
        t_eval=None,
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
    )


def test_beam_branch_clone_for_child_isolates_branch_owned_state() -> None:
    parent = _make_branch()
    child = parent.clone_for_child(branch_id=12)

    child.available_indices.remove(2)
    child.selection_counts[1] = 99
    child.theta[0] = 9.0
    child.history[0]["nested"]["path"].append("child")
    child.phase1_features_history[0]["nested"]["scores"].append(2.0)
    child.phase1_stage_events[0]["meta"]["reason"] = "child_only"
    child.phase1_measure_cache.commit(["child_group"])
    child.phase2_optimizer_memory["nested"]["weights"].append(3.0)
    child.phase2_last_shortlist_records[0]["ids"].append(2)
    child.phase2_last_shortlist_eval_records[0]["eval"]["kept"] = False
    child.phase3_split_events[0]["meta"]["selected"].append("y")
    child.phase3_runtime_split_summary["counts"]["selected"] = 7
    child.phase3_motif_usage["used"]["motif_a"] = 9
    child.phase3_rescue_history[0]["rescue"]["winner"] = "child"
    child.phase1_stage.begin_core()

    assert child.branch_id == 12
    assert child.parent_branch_id == parent.branch_id
    assert child.selected_ops is not parent.selected_ops
    assert child.selected_ops[0] is parent.selected_ops[0]
    assert parent.available_indices == {1, 2, 3}
    assert parent.selection_counts.tolist() == [0, 1, 0, 2]
    assert parent.theta.tolist() == [0.1, 0.2]
    assert parent.history[0]["nested"]["path"] == ["parent"]
    assert parent.phase1_features_history[0]["nested"]["scores"] == [1.0]
    assert parent.phase1_stage_events[0]["meta"]["reason"] == "seed_complete"
    assert parent.phase1_measure_cache.summary()["groups_known"] == 1.0
    assert parent.phase2_optimizer_memory["nested"]["weights"] == [1.0, 2.0]
    assert parent.phase2_last_shortlist_records[0]["ids"] == [1]
    assert parent.phase2_last_shortlist_eval_records[0]["eval"]["kept"] is True
    assert parent.phase3_split_events[0]["meta"]["selected"] == ["x"]
    assert parent.phase3_runtime_split_summary["counts"]["selected"] == 1
    assert parent.phase3_motif_usage["used"]["motif_a"] == 1
    assert parent.phase3_rescue_history[0]["rescue"]["winner"] == "x"
    assert parent.phase1_stage.stage_name == "seed"


def test_beam_branch_sibling_clones_diverge_independently() -> None:
    parent = _make_branch()
    child_a = parent.clone_for_child(branch_id=21)
    child_b = parent.clone_for_child(branch_id=22)

    child_a.available_indices.remove(1)
    child_b.available_indices.remove(3)
    child_a.selection_counts[0] = 5
    child_b.selection_counts[0] = 8
    child_a.phase2_optimizer_memory["nested"]["weights"].append(10.0)
    child_b.phase2_optimizer_memory["nested"]["weights"].append(20.0)
    child_a.phase1_measure_cache.commit(["a_only"])
    child_b.phase1_measure_cache.commit(["b_only"])
    child_a.phase1_stage.begin_core()

    assert parent.available_indices == {1, 2, 3}
    assert child_a.available_indices == {2, 3}
    assert child_b.available_indices == {1, 2}
    assert parent.selection_counts.tolist() == [0, 1, 0, 2]
    assert child_a.selection_counts.tolist() == [5, 1, 0, 2]
    assert child_b.selection_counts.tolist() == [8, 1, 0, 2]
    assert parent.phase2_optimizer_memory["nested"]["weights"] == [1.0, 2.0]
    assert child_a.phase2_optimizer_memory["nested"]["weights"] == [1.0, 2.0, 10.0]
    assert child_b.phase2_optimizer_memory["nested"]["weights"] == [1.0, 2.0, 20.0]
    assert parent.phase1_measure_cache.summary()["groups_known"] == 1.0
    assert child_a.phase1_measure_cache.summary()["groups_known"] == 2.0
    assert child_b.phase1_measure_cache.summary()["groups_known"] == 2.0
    assert parent.phase1_stage.stage_name == "seed"
    assert child_a.phase1_stage.stage_name == "core"
    assert child_b.phase1_stage.stage_name == "seed"


def test_true_beam_defaults_are_exposed_and_winner_history_stays_singleton() -> None:
    diagnostics: dict[str, object] = {}
    payload, _psi = _run_hardcoded_adapt_vqe(
        h_poly=_hh_h(),
        num_sites=2,
        ordering="blocked",
        problem="hh",
        adapt_pool="paop_lf_std",
        t=1.0,
        u=2.0,
        dv=0.0,
        boundary="open",
        omega0=1.0,
        g_ep=1.0,
        n_ph_max=1,
        boson_encoding="binary",
        max_depth=1,
        eps_grad=1e-2,
        eps_energy=1e-6,
        maxiter=5,
        seed=7,
        allow_repeats=True,
        finite_angle_fallback=False,
        finite_angle=0.1,
        finite_angle_min_improvement=0.0,
        adapt_continuation_mode="phase3_v1",
        adapt_beam_live_branches=3,
        diagnostics_out=diagnostics,
    )

    beam_policy = diagnostics["beam_policy"]
    beam_search = diagnostics["beam_search"]
    assert beam_policy["beam_enabled"] is True
    assert int(beam_policy["live_branches_effective"]) == 3
    assert int(beam_policy["children_per_parent_effective"]) == 2
    assert int(beam_policy["terminated_keep_effective"]) == 3
    assert beam_search["beam_enabled"] is True
    assert beam_search["finalist_count"] >= 1
    assert isinstance(beam_search["rounds"], list)
    assert len(beam_search["rounds"]) >= 1
    round0 = beam_search["rounds"][0]
    assert int(round0["frontier_input_count"]) == 1
    assert int(round0["parents_expanded_count"]) == 1
    assert int(round0["children_materialized_count"]) == int(round0["active_children_raw_count"]) + int(
        round0["round_terminals_raw_count"]
    )
    assert int(round0["active_children_unique_count"]) <= int(round0["active_children_raw_count"])
    assert int(round0["frontier_kept_count"]) <= int(beam_policy["live_branches_effective"])
    assert int(round0["terminal_kept_count"]) <= int(beam_policy["terminated_keep_effective"])
    assert payload["adapt_beam_enabled"] is True
    assert int(payload["adapt_beam_live_branches"]) == 3
    assert int(payload["adapt_beam_children_per_parent"]) == 2
    assert int(payload["adapt_beam_terminated_keep"]) == 3
    assert isinstance(payload["operators"], list)
    assert isinstance(payload["optimal_point"], list)
    assert len(payload["optimal_point"]) == len(payload["operators"])
    for row in payload.get("history", []):
        assert row["selected_positions"] == [row["selected_position"]]
        assert bool(row["batch_selected"]) is False

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/compiled_polynomial.py
```py
"""Compiled PauliPolynomial helpers (exyz convention).

These utilities compile and apply PauliPolynomial operators using the shared
compiled-Pauli backend in ``src.quantum.pauli_actions``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_compiled_pauli,
    compile_pauli_action_exyz,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial


@dataclass(frozen=True)
class CompiledPolynomialTerm:
    coeff: complex
    action: CompiledPauliAction | None


@dataclass(frozen=True)
class CompiledPolynomialAction:
    nq: int
    terms: tuple[CompiledPolynomialTerm, ...]


_MATH_COMPILE_POLYNOMIAL_ACTION = (
    r"C(H)=\{(c_\ell,A_\ell)\}_\ell,\ "
    r"A_\ell=\mathrm{compile\_pauli\_action\_exyz}(\ell,n_q),\ "
    r"H=\sum_\ell c_\ell P_\ell"
)


def compile_polynomial_action(
    poly: PauliPolynomial,
    *,
    tol: float = 1e-15,
    pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
) -> CompiledPolynomialAction:
    """Compile a PauliPolynomial into reusable term actions."""
    terms = poly.return_polynomial()
    if not terms:
        raise ValueError("Cannot compile empty PauliPolynomial: unable to infer qubit count.")

    nq = int(terms[0].nqubit())
    id_label = "e" * nq
    coeff_by_label: dict[str, complex] = {}

    for term in terms:
        term_nq = int(term.nqubit())
        if term_nq != nq:
            raise ValueError(f"Inconsistent term qubit count: expected {nq}, got {term_nq}.")
        label = str(term.pw2strng())
        if len(label) != nq:
            raise ValueError(f"Invalid Pauli label length for '{label}': expected {nq}.")
        coeff_by_label[label] = coeff_by_label.get(label, 0.0 + 0.0j) + complex(term.p_coeff)

    cache = pauli_action_cache if pauli_action_cache is not None else {}
    compiled_terms: list[CompiledPolynomialTerm] = []
    for label, coeff in coeff_by_label.items():
        coeff_c = complex(coeff)
        if abs(coeff_c) <= float(tol):
            continue
        if label == id_label:
            compiled_terms.append(CompiledPolynomialTerm(coeff=coeff_c, action=None))
            continue
        action = cache.get(label)
        if action is None:
            action = compile_pauli_action_exyz(label, nq)
            cache[label] = action
        compiled_terms.append(CompiledPolynomialTerm(coeff=coeff_c, action=action))

    return CompiledPolynomialAction(nq=nq, terms=tuple(compiled_terms))


_MATH_APPLY_COMPILED_POLYNOMIAL = r"H|\psi\rangle=\sum_j c_j P_j|\psi\rangle"


def apply_compiled_polynomial(psi: np.ndarray, compiled: CompiledPolynomialAction) -> np.ndarray:
    """Apply a compiled PauliPolynomial action to a statevector."""
    psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
    expected_dim = 1 << int(compiled.nq)
    if psi_vec.size != expected_dim:
        raise ValueError(
            f"Statevector length mismatch: got {psi_vec.size}, expected {expected_dim} for nq={compiled.nq}."
        )

    out = np.zeros_like(psi_vec, dtype=complex)
    for term in compiled.terms:
        if term.action is None:
            out += term.coeff * psi_vec
        else:
            out += term.coeff * apply_compiled_pauli(psi_vec, term.action)
    return out


_MATH_ENERGY_VIA_ONE_APPLY = r"E=\operatorname{Re}\langle\psi|H|\psi\rangle,\ H|\psi\rangle\text{ computed once}"


def energy_via_one_apply(
    psi: np.ndarray, compiled_h: CompiledPolynomialAction,
) -> tuple[float, np.ndarray]:
    """Compute energy via one compiled Hamiltonian apply and return (E, Hpsi)."""
    psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
    hpsi = apply_compiled_polynomial(psi_vec, compiled_h)
    energy = float(np.real(np.vdot(psi_vec, hpsi)))
    return energy, hpsi


_MATH_ADAPT_COMMUTATOR_GRAD = r"g=\frac{dE}{d\theta}\big|_{\theta=0}=2\,\operatorname{Im}\langle H\psi|A\psi\rangle"


def adapt_commutator_grad_from_hpsi(Hpsi: np.ndarray, Apsi: np.ndarray) -> float:
    """Return the signed ADAPT commutator gradient from precomputed vectors."""
    hpsi_vec = np.asarray(Hpsi, dtype=complex).reshape(-1)
    apsi_vec = np.asarray(Apsi, dtype=complex).reshape(-1)
    if hpsi_vec.size != apsi_vec.size:
        raise ValueError(f"Vector size mismatch: {hpsi_vec.size} != {apsi_vec.size}.")
    return float(2.0 * np.imag(np.vdot(hpsi_vec, apsi_vec)))


__all__ = [
    "CompiledPolynomialTerm",
    "CompiledPolynomialAction",
    "compile_polynomial_action",
    "apply_compiled_polynomial",
    "energy_via_one_apply",
    "adapt_commutator_grad_from_hpsi",
]

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_local_checkpoint_fit.py
```py
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.time_propagation.local_checkpoint_fit import (
    CheckpointFitConfig,
    LocalPauliAnsatzSpec,
    build_local_pauli_ansatz_terms,
    fit_checkpoint_target_state,
    fit_checkpoint_trajectory,
)


def test_build_local_pauli_ansatz_terms_counts_chain_terms() -> None:
    terms = build_local_pauli_ansatz_terms(
        LocalPauliAnsatzSpec(
            num_qubits=4,
            reps=2,
            single_axes=("y",),
            entangler_axes=("zz",),
        )
    )
    assert len(terms) == 2 * (4 + 3)
    assert str(terms[0].label) == "local_y(q=0)_rep1"
    assert str(terms[-1].label) == "local_zz(q=2,3)_rep2"


def test_fit_checkpoint_target_state_recovers_one_qubit_rotation() -> None:
    terms = build_local_pauli_ansatz_terms(
        LocalPauliAnsatzSpec(
            num_qubits=1,
            reps=1,
            single_axes=("y",),
            entangler_axes=(),
        )
    )
    executor = CompiledAnsatzExecutor(list(terms))
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    target_theta = np.asarray([0.23], dtype=float)
    psi_target = executor.prepare_state(target_theta, psi_ref)

    result = fit_checkpoint_target_state(
        psi_ref,
        psi_target,
        terms,
        config=CheckpointFitConfig(maxiter=40),
    )

    assert float(result.fidelity) >= 1.0 - 1e-10
    assert float(result.objective) <= 1e-10
    assert np.allclose(result.theta, target_theta, atol=1e-6)


def test_fit_checkpoint_trajectory_warm_starts_across_times() -> None:
    terms = build_local_pauli_ansatz_terms(
        LocalPauliAnsatzSpec(
            num_qubits=1,
            reps=1,
            single_axes=("y",),
            entangler_axes=(),
        )
    )
    executor = CompiledAnsatzExecutor(list(terms))
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    theta_targets = [0.0, 0.1, 0.2]
    target_states = [executor.prepare_state(np.asarray([theta], dtype=float), psi_ref) for theta in theta_targets]

    result = fit_checkpoint_trajectory(
        psi_ref,
        target_states,
        [0.0, 0.1, 0.2],
        terms,
        config=CheckpointFitConfig(maxiter=30),
    )

    assert result.theta_history.shape == (3, 1)
    assert len(result.states) == 3
    assert all(float(row["fidelity"]) >= 1.0 - 1e-10 for row in result.solver_rows)

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_generators.py
```py
#!/usr/bin/env python3
"""Generator metadata helpers for HH continuation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any, Mapping, Sequence

from pipelines.hardcoded.hh_continuation_types import GeneratorMetadata, GeneratorSplitEvent
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _polynomial_signature(poly: Any, *, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    items: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in generator polynomial: {coeff}")
        items.append((str(term.pw2strng()), float(round(coeff.real, 12))))
    items.sort()
    return tuple(items)


def _support_qubits(poly: Any) -> list[int]:
    support: set[int] = set()
    for term in poly.return_polynomial():
        word = str(term.pw2strng())
        nq = int(term.nqubit())
        for idx, ch in enumerate(word):
            if ch == "e":
                continue
            support.add(int(nq - 1 - idx))
    return sorted(int(q) for q in support)


def _qubit_to_site(
    qubit: int,
    *,
    num_sites: int,
    ordering: str,
    qpb: int,
) -> int:
    q = int(qubit)
    n_sites = int(num_sites)
    fermion_qubits = 2 * n_sites
    if q < fermion_qubits:
        ordering_key = str(ordering).strip().lower()
        if ordering_key == "interleaved":
            return int(q // 2)
        return int(q % n_sites)
    return int((q - fermion_qubits) // int(max(1, qpb)))


def _support_sites(
    support_qubits: Sequence[int],
    *,
    num_sites: int,
    ordering: str,
    qpb: int,
) -> list[int]:
    out = {
        _qubit_to_site(int(q), num_sites=int(num_sites), ordering=str(ordering), qpb=int(qpb))
        for q in support_qubits
    }
    return sorted(int(x) for x in out)


def _relative_site_offsets(sites: Sequence[int]) -> list[int]:
    if not sites:
        return []
    site_min = min(int(x) for x in sites)
    return [int(int(x) - site_min) for x in sites]


def _template_id(
    *,
    family_id: str,
    support_site_offsets: Sequence[int],
    n_poly_terms: int,
    has_boson_support: bool,
    has_fermion_support: bool,
    is_macro_generator: bool,
) -> str:
    parts = [
        str(family_id),
        "macro" if bool(is_macro_generator) else "atomic",
        f"terms{int(n_poly_terms)}",
        f"sites{','.join(str(int(x)) for x in support_site_offsets)}",
        f"bos{int(bool(has_boson_support))}",
        f"ferm{int(bool(has_fermion_support))}",
    ]
    return "|".join(parts)


def _serialize_polynomial_terms(poly: Any, *, tol: float = 1e-12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        out.append(
            {
                "pauli_exyz": str(term.pw2strng()),
                "coeff_re": float(coeff.real),
                "coeff_im": float(coeff.imag),
                "nq": int(term.nqubit()),
            }
        )
    return out


def rebuild_polynomial_from_serialized_terms(
    serialized_terms: Sequence[Mapping[str, Any]],
) -> PauliPolynomial:
    pauli_terms: list[PauliTerm] = []
    nq_expected: int | None = None
    for raw in serialized_terms:
        if not isinstance(raw, Mapping):
            continue
        nq = int(raw.get("nq", 0))
        label = str(raw.get("pauli_exyz", ""))
        coeff = complex(float(raw.get("coeff_re", 0.0)), float(raw.get("coeff_im", 0.0)))
        if nq <= 0 or label == "":
            continue
        if nq_expected is None:
            nq_expected = int(nq)
        elif int(nq) != int(nq_expected):
            raise ValueError("Serialized runtime-split terms use inconsistent nq values.")
        pauli_terms.append(PauliTerm(int(nq), ps=label, pc=coeff))
    if nq_expected is None or not pauli_terms:
        raise ValueError("Serialized runtime-split terms are missing or invalid.")
    return PauliPolynomial("JW", list(pauli_terms))


def build_generator_metadata(
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
    signature = _polynomial_signature(polynomial)
    support_qubits = _support_qubits(polynomial)
    support_sites = _support_sites(
        support_qubits,
        num_sites=int(num_sites),
        ordering=str(ordering),
        qpb=int(qpb),
    )
    support_site_offsets = _relative_site_offsets(support_sites)
    has_fermion_support = any(int(q) < 2 * int(num_sites) for q in support_qubits)
    has_boson_support = any(int(q) >= 2 * int(num_sites) for q in support_qubits)
    n_poly_terms = int(len(list(polynomial.return_polynomial())))
    is_macro = bool(n_poly_terms > 1 and str(split_policy) != "deliberate_split")
    template_id = _template_id(
        family_id=str(family_id),
        support_site_offsets=support_site_offsets,
        n_poly_terms=int(n_poly_terms),
        has_boson_support=bool(has_boson_support),
        has_fermion_support=bool(has_fermion_support),
        is_macro_generator=bool(is_macro),
    )
    digest = hashlib.sha1(
        (
            f"{family_id}|{template_id}|{signature}|{split_policy}|{parent_generator_id or ''}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    return GeneratorMetadata(
        generator_id=f"gen:{digest}",
        family_id=str(family_id),
        template_id=str(template_id),
        candidate_label=str(label),
        support_qubits=[int(x) for x in support_qubits],
        support_sites=[int(x) for x in support_sites],
        support_site_offsets=[int(x) for x in support_site_offsets],
        is_macro_generator=bool(is_macro),
        split_policy=str(split_policy),
        parent_generator_id=(str(parent_generator_id) if parent_generator_id is not None else None),
        symmetry_spec=(dict(symmetry_spec) if isinstance(symmetry_spec, Mapping) else None),
        compile_metadata={
            "num_polynomial_terms": int(n_poly_terms),
            "signature_size": int(len(signature)),
            "has_boson_support": bool(has_boson_support),
            "has_fermion_support": bool(has_fermion_support),
            "support_size": int(len(support_qubits)),
            "serialized_terms_exyz": _serialize_polynomial_terms(polynomial),
        },
    )


def build_pool_generator_registry(
    *,
    terms: Sequence[Any],
    family_ids: Sequence[str],
    num_sites: int,
    ordering: str,
    qpb: int,
    symmetry_specs: Sequence[Mapping[str, Any] | None] | None = None,
    split_policy: str = "preserve",
) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    sym_specs = list(symmetry_specs) if symmetry_specs is not None else [None] * len(list(terms))
    for idx, term in enumerate(terms):
        family_id = str(family_ids[idx] if idx < len(family_ids) else "unknown")
        symmetry_spec = sym_specs[idx] if idx < len(sym_specs) else None
        meta = build_generator_metadata(
            label=str(term.label),
            polynomial=term.polynomial,
            family_id=str(family_id),
            num_sites=int(num_sites),
            ordering=str(ordering),
            qpb=int(qpb),
            split_policy=str(split_policy),
            symmetry_spec=symmetry_spec,
        )
        registry[str(term.label)] = asdict(meta)
    return registry


def build_runtime_split_children(
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
    serialized = _serialize_polynomial_terms(polynomial, tol=float(tol))
    total_children = int(len(serialized))
    if total_children <= 1:
        return []
    out: list[dict[str, Any]] = []
    parent_generator_id = None
    if isinstance(parent_generator_metadata, Mapping) and parent_generator_metadata.get("generator_id") is not None:
        parent_generator_id = str(parent_generator_metadata.get("generator_id"))
    child_limit = total_children if max_children is None or int(max_children) <= 0 else min(total_children, int(max_children))
    for child_index, term_info in enumerate(serialized[:child_limit]):
        child_poly = rebuild_polynomial_from_serialized_terms([term_info])
        child_label = (
            f"{str(parent_label)}::split[{int(child_index)}]::{str(term_info.get('pauli_exyz', ''))}"
        )
        child_meta = asdict(
            build_generator_metadata(
                label=str(child_label),
                polynomial=child_poly,
                family_id=str(family_id),
                num_sites=int(num_sites),
                ordering=str(ordering),
                qpb=int(qpb),
                split_policy="deliberate_split",
                parent_generator_id=parent_generator_id,
                symmetry_spec=symmetry_spec,
            )
        )
        compile_metadata = dict(child_meta.get("compile_metadata", {}))
        compile_metadata["runtime_split"] = {
            "mode": str(split_mode),
            "parent_label": str(parent_label),
            "child_index": int(child_index),
            "child_count": int(total_children),
        }
        compile_metadata["serialized_terms_exyz"] = [dict(term_info)]
        child_meta["compile_metadata"] = compile_metadata
        out.append(
            {
                "child_label": str(child_label),
                "child_polynomial": child_poly,
                "child_generator_metadata": dict(child_meta),
                "child_index": int(child_index),
                "child_count": int(total_children),
                "parent_label": str(parent_label),
            }
        )
    return out


def selected_generator_metadata_for_labels(
    labels: Sequence[str],
    registry: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in labels:
        meta = registry.get(str(label))
        if isinstance(meta, Mapping):
            out.append(dict(meta))
    return out


def build_split_event(
    *,
    parent_generator_id: str,
    child_generator_ids: Sequence[str],
    reason: str,
    split_mode: str,
) -> dict[str, Any]:
    event = GeneratorSplitEvent(
        parent_generator_id=str(parent_generator_id),
        child_generator_ids=[str(x) for x in child_generator_ids],
        reason=str(reason),
        split_mode=str(split_mode),
    )
    return asdict(event)

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/exact_bench/hh_fixed_seed_budgeted_projected_dynamics.py
```py
#!/usr/bin/env python3
"""Budgeted fixed-seed projected real-time dynamics sweep for HH."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    get_plt,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)
from docs.reports.qiskit_circuit_report import ops_to_circuit, transpile_circuit_metrics
from pipelines.hardcoded.hh_vqe_from_adapt_family import build_replay_sequence_from_input_json
from src.quantum.drives_time_potential import build_gaussian_sinusoid_density_drive
from src.quantum.time_propagation import (
    ProjectedRealTimeConfig,
    expectation_total_hamiltonian,
    run_exact_driven_reference,
    run_projected_real_time_trajectory,
    state_fidelity,
)


_DEFAULT_FIXED_SEED_JSON = REPO_ROOT / "artifacts" / "json" / "l2_hh_open_direct_adapt_phase3_paoplf_u4_g05_nph2.json"
_DEFAULT_OUTPUT_JSON = REPO_ROOT / "artifacts" / "json" / "hh_fixed_seed_budgeted_projected_dynamics.json"
_DEFAULT_OUTPUT_CSV = REPO_ROOT / "artifacts" / "json" / "hh_fixed_seed_budgeted_projected_dynamics.csv"
_DEFAULT_OUTPUT_PDF = REPO_ROOT / "artifacts" / "pdf" / "hh_fixed_seed_budgeted_projected_dynamics.pdf"
_DEFAULT_RUN_ROOT = REPO_ROOT / "artifacts" / "json" / "hh_fixed_seed_budgeted_projected_dynamics_runs"
_DEFAULT_TAG = "hh_fixed_seed_budgeted_projected_dynamics"
_DEFAULT_BACKEND_NAME = "FakeGuadalupeV2"
_DEFAULT_REPRESENTATIVE_PREFIXES = (1, 4, 8, 11, 12, 13, 15)


@dataclass(frozen=True)
class SweepConfig:
    fixed_seed_json: Path
    output_json: Path
    output_csv: Path
    output_pdf: Path
    run_root: Path
    tag: str
    backend_name: str
    use_fake_backend: bool
    circuit_optimization_level: int
    circuit_seed_transpiler: int
    max_cx_budget: int
    t_final: float
    num_times: int
    reference_steps: int
    ode_substeps: int
    tangent_eps: float
    lambda_reg: float
    svd_rcond: float
    drive_A: float
    drive_omega: float
    drive_tbar: float
    drive_phi: float
    drive_pattern: str
    drive_t0: float
    drive_time_sampling: str
    prefix_limit: int
    representative_prefixes: tuple[int, ...]
    skip_pdf: bool


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded: dict[str, Any] = {}
            for key in fieldnames:
                raw = _jsonable(row.get(key))
                encoded[key] = json.dumps(raw, sort_keys=True) if isinstance(raw, (dict, list)) else raw
            writer.writerow(encoded)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    text = str(raw).strip()
    if text == "":
        return ()
    seen: set[int] = set()
    out: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if item == "":
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("prefix lists must contain positive integers.")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _collect_hardcoded_terms_exyz(h_poly: Any) -> tuple[list[str], dict[str, complex]]:
    coeff_map: dict[str, complex] = {}
    native_order: list[str] = []
    for term in h_poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if label not in coeff_map:
            native_order.append(label)
            coeff_map[label] = coeff
        else:
            coeff_map[label] = coeff_map[label] + coeff
    return native_order, coeff_map


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = {
        "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    }
    out = mats[str(label)[0]]
    for ch in str(label)[1:]:
        out = np.kron(out, mats[ch])
    return out


def _build_hamiltonian_matrix(coeff_map_exyz: Mapping[str, complex]) -> np.ndarray:
    if not coeff_map_exyz:
        return np.zeros((1, 1), dtype=complex)
    nq = len(next(iter(coeff_map_exyz)))
    dim = 1 << int(nq)
    hmat = np.zeros((dim, dim), dtype=complex)
    for label, coeff in coeff_map_exyz.items():
        hmat += complex(coeff) * _pauli_matrix_exyz(str(label))
    return hmat


def _build_drive_provider(
    *,
    num_sites: int,
    nq_total: int,
    ordering: str,
    cfg: SweepConfig,
) -> tuple[Any, dict[str, Any]]:
    drive = build_gaussian_sinusoid_density_drive(
        n_sites=int(num_sites),
        nq_total=int(nq_total),
        indexing=str(ordering),
        A=float(cfg.drive_A),
        omega=float(cfg.drive_omega),
        tbar=float(cfg.drive_tbar),
        phi=float(cfg.drive_phi),
        pattern_mode=str(cfg.drive_pattern),
        include_identity=False,
        coeff_tol=0.0,
    )
    return drive.coeff_map_exyz, {
        "A": float(cfg.drive_A),
        "omega": float(cfg.drive_omega),
        "tbar": float(cfg.drive_tbar),
        "phi": float(cfg.drive_phi),
        "pattern": str(cfg.drive_pattern),
        "t0": float(cfg.drive_t0),
        "time_sampling": str(cfg.drive_time_sampling),
    }


def _transpile_theta_history(
    *,
    terms: Sequence[Any],
    theta_history: np.ndarray,
    num_qubits: int,
    cfg: SweepConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_row: dict[str, Any] | None = None
    final_row: dict[str, Any] | None = None
    for time_idx, theta in enumerate(np.asarray(theta_history, dtype=float)):
        metrics = transpile_circuit_metrics(
            ops_to_circuit(
                terms,
                np.asarray(theta, dtype=float),
                num_qubits=int(num_qubits),
            ),
            backend_name=str(cfg.backend_name),
            use_fake_backend=bool(cfg.use_fake_backend),
            optimization_level=int(cfg.circuit_optimization_level),
            seed_transpiler=int(cfg.circuit_seed_transpiler),
        )
        tx = metrics.get("transpiled", {})
        row = {
            "time_index": int(time_idx),
            "count_2q": int(tx.get("count_2q", 0)),
            "count_1q": int(tx.get("count_1q", 0)),
            "depth": int(tx.get("depth", 0)),
            "size": int(tx.get("size", 0)),
        }
        rows.append(row)
        final_row = dict(row)
        if max_row is None or (
            int(row["count_2q"]),
            int(row["depth"]),
            int(row["time_index"]),
        ) > (
            int(max_row["count_2q"]),
            int(max_row["depth"]),
            int(max_row["time_index"]),
        ):
            max_row = dict(row)
    return rows, (max_row or {}), (final_row or {})


def _candidate_row(
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
    trajectory: list[dict[str, Any]] = []
    energy_errors: list[float] = []
    fidelities: list[float] = []
    for idx, time_val in enumerate(np.asarray(times, dtype=float)):
        psi_sur = np.asarray(projected_result.states[idx], dtype=complex).reshape(-1)
        psi_exact = np.asarray(exact_states[idx], dtype=complex).reshape(-1)
        energy_sur = expectation_total_hamiltonian(
            psi_sur,
            hmat_static,
            drive_coeff_provider_exyz=drive_provider,
            t_physical=float(cfg.drive_t0) + float(time_val),
        )
        fidelity = state_fidelity(psi_sur, psi_exact)
        energy_error = float(abs(float(energy_sur) - float(exact_energies[idx])))
        energy_errors.append(float(energy_error))
        fidelities.append(float(fidelity))
        trajectory.append(
            {
                "time": float(time_val),
                "energy_total_exact": float(exact_energies[idx]),
                "energy_total_projected": float(energy_sur),
                "abs_energy_total_error": float(energy_error),
                "fidelity": float(fidelity),
                "theta_norm": float(projected_result.trajectory_rows[idx]["theta_norm"]),
                "condition_number": float(projected_result.trajectory_rows[idx]["condition_number"]),
                "state_norm": float(projected_result.trajectory_rows[idx]["state_norm"]),
            }
        )
    tx_rows, tx_max, tx_final = _transpile_theta_history(
        terms=replay_terms,
        theta_history=np.asarray(projected_result.theta_history, dtype=float),
        num_qubits=int(np.asarray(projected_result.states[0]).size.bit_length() - 1),
        cfg=cfg,
    )
    row = {
        "prefix_k": int(prefix_k),
        "term_count": int(len(replay_terms)),
        "max_transpiled_count_2q": int(tx_max.get("count_2q", 0)),
        "max_transpiled_depth": int(tx_max.get("depth", 0)),
        "final_transpiled_count_2q": int(tx_final.get("count_2q", 0)),
        "final_transpiled_depth": int(tx_final.get("depth", 0)),
        "final_fidelity": float(fidelities[-1]),
        "min_fidelity": float(min(fidelities)),
        "max_abs_energy_total_error": float(max(energy_errors)),
        "final_abs_energy_total_error": float(energy_errors[-1]),
        "budget_pass": bool(int(tx_max.get("count_2q", 0)) <= int(cfg.max_cx_budget)),
    }
    return row, {
        "prefix_k": int(prefix_k),
        "row": dict(row),
        "trajectory": trajectory,
        "solver_rows": [dict(item) for item in projected_result.trajectory_rows],
        "transpile_rows": tx_rows,
    }


def _best_row(rows: Sequence[Mapping[str, Any]], *, key: str) -> dict[str, Any] | None:
    valid = [dict(row) for row in rows if row.get("budget_pass", False)]
    if not valid:
        return None
    if str(key) == "error":
        return dict(
            min(
                valid,
                key=lambda row: (
                    float(row.get("max_abs_energy_total_error", float("inf"))),
                    int(row.get("max_transpiled_count_2q", 10**9)),
                    int(row.get("prefix_k", 10**9)),
                ),
            )
        )
    return dict(
        min(
            valid,
            key=lambda row: (
                int(row.get("max_transpiled_count_2q", 10**9)),
                float(row.get("max_abs_energy_total_error", float("inf"))),
                int(row.get("prefix_k", 10**9)),
            ),
        )
    )


def write_summary_pdf(
    *,
    cfg: SweepConfig,
    payload: Mapping[str, Any],
    candidate_details: Mapping[int, Mapping[str, Any]],
    seed_payload: Mapping[str, Any],
) -> None:
    if bool(cfg.skip_pdf):
        return
    require_matplotlib()
    plt = get_plt()
    PdfPages = get_PdfPages()
    command = current_command_string()
    rows = [dict(row) for row in payload.get("rows", [])]
    representative = [int(k) for k in payload.get("representative_prefixes", [])]

    with PdfPages(Path(cfg.output_pdf)) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein (HH)",
            ansatz="Fixed-seed projected real-time prefix ladder (one layer, dynamics-only)",
            drive_enabled=True,
            t=float(seed_payload["settings"]["t"]),
            U=float(seed_payload["settings"]["u"]),
            dv=float(seed_payload["settings"]["dv"]),
            extra={
                "fixed_seed_json": str(cfg.fixed_seed_json),
                "g_ep": float(seed_payload["settings"]["g_ep"]),
                "omega0": float(seed_payload["settings"]["omega0"]),
                "n_ph_max": int(seed_payload["settings"]["n_ph_max"]),
                "budget_scope": "dynamics_only",
                "max_cx_budget": int(cfg.max_cx_budget),
                "t_final": float(cfg.t_final),
                "num_times": int(cfg.num_times),
                "reference_steps": int(cfg.reference_steps),
                "ode_substeps": int(cfg.ode_substeps),
                "backend_name": str(cfg.backend_name),
            },
            command=command,
        )

        fig = plt.figure(figsize=(11.0, 8.5))
        ax = fig.add_subplot(111)
        table_rows = [
            [
                str(int(row["prefix_k"])),
                str(int(row["max_transpiled_count_2q"])),
                str(int(row["max_transpiled_depth"])),
                f"{float(row['final_fidelity']):.6f}",
                f"{float(row['min_fidelity']):.6f}",
                f"{float(row['max_abs_energy_total_error']):.6e}",
            ]
            for row in rows
        ]
        render_compact_table(
            ax,
            title="Prefix scoreboard",
            col_labels=["K", "Max 2Q", "Max depth", "Final F", "Min F", "Max |dE|"],
            rows=table_rows or [["(none)", "", "", "", "", ""]],
            fontsize=7,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for key, ylabel, title in (
            ("energy_total_projected", "energy", "Projected vs exact total energy"),
            ("fidelity", "fidelity", "Projected fidelity to exact trajectory"),
        ):
            fig, ax = plt.subplots(figsize=(11.0, 8.5))
            exact_detail = candidate_details.get(int(representative[0]), None) if representative else None
            if exact_detail is not None and key == "energy_total_projected":
                times = [float(item["time"]) for item in exact_detail["trajectory"]]
                exact_vals = [float(item["energy_total_exact"]) for item in exact_detail["trajectory"]]
                ax.plot(times, exact_vals, color="black", linewidth=2.0, label="exact")
            for prefix_k in representative:
                detail = candidate_details.get(int(prefix_k), None)
                if detail is None:
                    continue
                times = [float(item["time"]) for item in detail["trajectory"]]
                values = [float(item[key]) for item in detail["trajectory"]]
                ax.plot(times, values, linewidth=1.5, label=f"K={int(prefix_k)}")
            ax.set_title(title)
            ax.set_xlabel("time")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        summary_lines = [
            "Budgeted projected dynamics summary",
            "",
            f"fixed_seed_json: {cfg.fixed_seed_json}",
            f"backend_name: {cfg.backend_name}",
            f"max_cx_budget: {cfg.max_cx_budget}",
            f"best_by_error_under_budget: {payload.get('summary', {}).get('best_by_error_under_budget', {}).get('prefix_k', 'n/a')}",
            f"best_by_cx_under_budget: {payload.get('summary', {}).get('best_by_cx_under_budget', {}).get('prefix_k', 'n/a')}",
            f"representative_prefixes: {', '.join(str(x) for x in representative) if representative else 'none'}",
        ]
        render_text_page(pdf, summary_lines, fontsize=10, line_spacing=0.03, max_line_width=110)


def run_sweep(cfg: SweepConfig) -> dict[str, Any]:
    replay_data = build_replay_sequence_from_input_json(Path(cfg.fixed_seed_json))
    seed_payload = replay_data["payload"]
    initial_state = np.asarray(replay_data["initial_state"], dtype=complex).reshape(-1)
    replay_terms_full = list(replay_data["replay_terms"])
    h_poly = replay_data["h_poly"]
    num_qubits = int(replay_data["nq"])
    seed_settings = seed_payload["settings"]
    _ordered_labels_exyz, coeff_map_exyz = _collect_hardcoded_terms_exyz(h_poly)
    hmat_static = _build_hamiltonian_matrix(coeff_map_exyz)
    drive_provider, drive_profile = _build_drive_provider(
        num_sites=int(seed_settings["L"]),
        nq_total=int(num_qubits),
        ordering=str(seed_settings["ordering"]),
        cfg=cfg,
    )
    exact_ref = run_exact_driven_reference(
        initial_state,
        hmat_static,
        t_final=float(cfg.t_final),
        num_times=int(cfg.num_times),
        reference_steps=int(cfg.reference_steps),
        drive_coeff_provider_exyz=drive_provider,
        drive_t0=float(cfg.drive_t0),
        time_sampling=str(cfg.drive_time_sampling),
    )

    prefix_max = min(int(cfg.prefix_limit), int(len(replay_terms_full)))
    prefixes = list(range(1, prefix_max + 1))
    rows: list[dict[str, Any]] = []
    candidate_details: dict[int, dict[str, Any]] = {}

    for prefix_k in prefixes:
        run_dir = Path(cfg.run_root) / f"K{int(prefix_k):02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        projected_cfg = ProjectedRealTimeConfig(
            t_final=float(cfg.t_final),
            num_times=int(cfg.num_times),
            ode_substeps=int(cfg.ode_substeps),
            tangent_eps=float(cfg.tangent_eps),
            lambda_reg=float(cfg.lambda_reg),
            svd_rcond=float(cfg.svd_rcond),
        )
        projected = run_projected_real_time_trajectory(
            initial_state,
            replay_terms_full[:prefix_k],
            hmat_static,
            config=projected_cfg,
            drive_coeff_provider_exyz=drive_provider,
            drive_t0=float(cfg.drive_t0),
        )
        row, detail = _candidate_row(
            prefix_k=int(prefix_k),
            times=np.asarray(exact_ref.times, dtype=float),
            exact_states=exact_ref.states,
            exact_energies=np.asarray(exact_ref.energies_total, dtype=float),
            projected_result=projected,
            replay_terms=replay_terms_full[:prefix_k],
            hmat_static=hmat_static,
            drive_provider=drive_provider,
            cfg=cfg,
        )
        rows.append(dict(row))
        candidate_details[int(prefix_k)] = {
            **dict(detail),
            "run_dir": str(run_dir),
            "term_labels": [str(term.label) for term in replay_terms_full[:prefix_k]],
        }
        _write_json(run_dir / "candidate.json", candidate_details[int(prefix_k)])

    feasible = [dict(row) for row in rows if bool(row.get("budget_pass", False))]
    representative = [
        int(prefix)
        for prefix in cfg.representative_prefixes
        if int(prefix) in candidate_details
    ]
    payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_fixed_seed_budgeted_projected_dynamics",
        "model_family": "Hubbard-Holstein (HH)",
        "scope": {
            "local_only": True,
            "noiseless_only": True,
            "budget_scope": "dynamics_only",
            "surrogate_method": "projected_real_time",
        },
        "settings": asdict(cfg),
        "seed": {
            "fixed_seed_json": str(cfg.fixed_seed_json),
            "family_info": dict(replay_data["family_info"]),
            "family_resolved": str(replay_data["family_resolved"]),
            "term_count": int(len(replay_terms_full)),
            "pool_meta": dict(replay_data["pool_meta"]),
        },
        "drive_profile": dict(drive_profile),
        "artifacts": {
            "output_json": str(cfg.output_json),
            "output_csv": str(cfg.output_csv),
            "output_pdf": str(cfg.output_pdf),
            "run_root": str(cfg.run_root),
        },
        "exact_reference": {
            "reference_steps": int(cfg.reference_steps),
            "time_sampling": str(cfg.drive_time_sampling),
            "trajectory": [dict(row) for row in exact_ref.trajectory_rows],
        },
        "rows": rows,
        "representative_prefixes": representative,
        "summary": {
            "feasible_count": int(len(feasible)),
            "best_by_error_under_budget": _best_row(rows, key="error"),
            "best_by_cx_under_budget": _best_row(rows, key="cx"),
        },
    }
    _write_json(Path(cfg.output_json), payload)
    _write_csv(Path(cfg.output_csv), rows)
    write_summary_pdf(
        cfg=cfg,
        payload=payload,
        candidate_details=candidate_details,
        seed_payload=seed_payload,
    )
    return payload


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a budgeted fixed-seed HH projected real-time dynamics sweep over "
            "one-layer ADAPT-prefix surrogates."
        )
    )
    parser.add_argument("--fixed-seed-json", type=Path, default=_DEFAULT_FIXED_SEED_JSON)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=_DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-pdf", type=Path, default=_DEFAULT_OUTPUT_PDF)
    parser.add_argument("--run-root", type=Path, default=_DEFAULT_RUN_ROOT)
    parser.add_argument("--tag", type=str, default=_DEFAULT_TAG)
    parser.add_argument("--backend-name", type=str, default=_DEFAULT_BACKEND_NAME)
    parser.add_argument("--circuit-use-fake-backend", action="store_true")
    parser.add_argument("--circuit-optimization-level", type=int, default=3)
    parser.add_argument("--circuit-seed-transpiler", type=int, default=11)
    parser.add_argument("--max-cx-budget", type=int, default=300)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--num-times", type=int, default=41)
    parser.add_argument("--reference-steps", type=int, default=256)
    parser.add_argument("--ode-substeps", type=int, default=4)
    parser.add_argument("--tangent-eps", type=float, default=1e-6)
    parser.add_argument("--lambda-reg", type=float, default=1e-8)
    parser.add_argument("--svd-rcond", type=float, default=1e-12)
    parser.add_argument("--drive-A", type=float, default=1.0)
    parser.add_argument("--drive-omega", type=float, default=1.0)
    parser.add_argument("--drive-tbar", type=float, default=5.0)
    parser.add_argument("--drive-phi", type=float, default=0.0)
    parser.add_argument("--drive-pattern", type=str, default="staggered")
    parser.add_argument("--drive-t0", type=float, default=0.0)
    parser.add_argument("--drive-time-sampling", type=str, default="midpoint")
    parser.add_argument("--prefix-limit", type=int, default=15)
    parser.add_argument(
        "--representative-prefixes",
        type=str,
        default=",".join(str(x) for x in _DEFAULT_REPRESENTATIVE_PREFIXES),
    )
    parser.add_argument("--skip-pdf", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> SweepConfig:
    args = build_cli_parser().parse_args(list(argv) if argv is not None else None)
    if int(args.circuit_optimization_level) < 0 or int(args.circuit_optimization_level) > 3:
        raise ValueError("--circuit-optimization-level must be in {0,1,2,3}.")
    if int(args.max_cx_budget) <= 0:
        raise ValueError("--max-cx-budget must be positive.")
    if int(args.num_times) < 2:
        raise ValueError("--num-times must be >= 2.")
    if int(args.reference_steps) < 1:
        raise ValueError("--reference-steps must be >= 1.")
    if int(args.ode_substeps) < 1:
        raise ValueError("--ode-substeps must be >= 1.")
    if int(args.prefix_limit) < 1:
        raise ValueError("--prefix-limit must be >= 1.")
    representative = _parse_int_tuple(str(args.representative_prefixes))
    return SweepConfig(
        fixed_seed_json=Path(args.fixed_seed_json),
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
        output_pdf=Path(args.output_pdf),
        run_root=Path(args.run_root),
        tag=str(args.tag),
        backend_name=str(args.backend_name),
        use_fake_backend=bool(args.circuit_use_fake_backend),
        circuit_optimization_level=int(args.circuit_optimization_level),
        circuit_seed_transpiler=int(args.circuit_seed_transpiler),
        max_cx_budget=int(args.max_cx_budget),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        reference_steps=int(args.reference_steps),
        ode_substeps=int(args.ode_substeps),
        tangent_eps=float(args.tangent_eps),
        lambda_reg=float(args.lambda_reg),
        svd_rcond=float(args.svd_rcond),
        drive_A=float(args.drive_A),
        drive_omega=float(args.drive_omega),
        drive_tbar=float(args.drive_tbar),
        drive_phi=float(args.drive_phi),
        drive_pattern=str(args.drive_pattern),
        drive_t0=float(args.drive_t0),
        drive_time_sampling=str(args.drive_time_sampling),
        prefix_limit=int(args.prefix_limit),
        representative_prefixes=representative,
        skip_pdf=bool(args.skip_pdf),
    )


def main(argv: Sequence[str] | None = None) -> int:
    cfg = parse_args(argv)
    if not bool(cfg.use_fake_backend):
        cfg = SweepConfig(**{**asdict(cfg), "use_fake_backend": True})
    run_sweep(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_types.py
```py
#!/usr/bin/env python3
"""Shared continuation datamodel for HH ADAPT -> replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class CandidateFeatures:
    stage_name: str
    candidate_label: str
    candidate_family: str
    candidate_pool_index: int
    position_id: int
    append_position: int
    positions_considered: list[int]
    g_signed: float
    g_abs: float
    g_lcb: float
    sigma_hat: float
    F_metric: float
    metric_proxy: float
    novelty: float | None
    curvature_mode: str
    novelty_mode: str
    refit_window_indices: list[int]
    compiled_position_cost_proxy: dict[str, float]
    measurement_cache_stats: dict[str, float]
    leakage_penalty: float
    stage_gate_open: bool
    leakage_gate_open: bool
    trough_probe_triggered: bool
    trough_detected: bool
    simple_score: float | None
    score_version: str
    F_raw: float | None = None
    Q_window: list[list[float]] | None = None
    q_window: list[float] | None = None
    h_raw: float | None = None
    b_mixed: list[float] | None = None
    H_window_hessian: list[list[float]] | None = None
    M_window: list[list[float]] | None = None
    h_eff: float | None = None
    F_red: float | None = None
    q_reduced: list[float] | None = None
    ridge_used: float | None = None
    h_hat: float | None = None
    b_hat: list[float] | None = None
    H_window: list[list[float]] | None = None
    depth_cost: float = 0.0
    new_group_cost: float = 0.0
    new_shot_cost: float = 0.0
    opt_dim_cost: float = 0.0
    reuse_count_cost: float = 0.0
    family_repeat_cost: float = 0.0
    full_v2_score: float | None = None
    shortlist_rank: int | None = None
    shortlist_size: int | None = None
    actual_fallback_mode: str = "screen_only"
    compatibility_penalty_total: float = 0.0
    generator_id: str | None = None
    template_id: str | None = None
    is_macro_generator: bool = False
    parent_generator_id: str | None = None
    runtime_split_mode: str = "off"
    runtime_split_parent_label: str | None = None
    runtime_split_child_index: int | None = None
    runtime_split_child_count: int | None = None
    generator_metadata: dict[str, Any] | None = None
    symmetry_spec: dict[str, Any] | None = None
    symmetry_mode: str = "none"
    symmetry_mitigation_mode: str = "off"
    motif_metadata: dict[str, Any] | None = None
    motif_bonus: float = 0.0
    motif_source: str = "none"
    remaining_evaluations_proxy: float = 0.0
    remaining_evaluations_proxy_mode: str = "none"
    lifetime_cost_mode: str = "off"
    lifetime_weight_components: dict[str, float] = field(default_factory=dict)
    placeholder_hooks: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasurementPlan:
    plan_version: str
    group_keys: list[str]
    nominal_shots_per_group: int
    grouping_mode: str


@dataclass(frozen=True)
class MeasurementCacheStats:
    groups_total: int
    groups_reused: int
    groups_new: int
    shots_reused: float
    shots_new: float
    reuse_count_cost: float


@dataclass(frozen=True)
class CompileCostEstimate:
    new_pauli_actions: float
    new_rotation_steps: float
    position_shift_span: float
    refit_active_count: float
    proxy_total: float


@dataclass(frozen=True)
class ScaffoldFingerprintLite:
    selected_operator_labels: list[str]
    selected_generator_ids: list[str]
    num_parameters: int
    generator_family: str
    continuation_mode: str
    compiled_pauli_cache_size: int
    measurement_plan_version: str
    post_prune: bool
    split_event_count: int = 0
    motif_record_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PruneDecision:
    index: int
    label: str
    accepted: bool
    energy_before: float
    energy_after: float
    regression: float
    reason: str


@dataclass(frozen=True)
class ReplayPlan:
    continuation_mode: str
    seed_policy_resolved: str
    handoff_state_kind: str
    freeze_scaffold_steps: int
    unfreeze_steps: int
    full_replay_steps: int
    trust_radius_initial: float
    trust_radius_growth: float
    trust_radius_max: float
    scaffold_block_indices: list[int]
    residual_block_indices: list[int]
    qn_spsa_refresh_every: int
    trust_radius_schedule: list[float]
    optimizer_memory_source: str = "unavailable"
    optimizer_memory_reused: bool = False
    refresh_mode: str = "disabled"
    symmetry_mitigation_mode: str = "off"
    generator_ids: list[str] = field(default_factory=list)
    motif_reference_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayPhaseTelemetry:
    phase_name: str
    nfev: int
    nit: int
    success: bool
    energy_before: float
    energy_after: float
    delta_abs_before: float | None
    delta_abs_after: float | None
    active_count: int
    frozen_count: int
    optimizer_memory_reused: bool = False
    optimizer_memory_source: str = "unavailable"
    qn_spsa_refresh_points: list[int] = field(default_factory=list)
    residual_zero_initialized: bool = True


class NoveltyOracle(Protocol):
    def estimate(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:  # pragma: no cover - interface
        ...


class CurvatureOracle(Protocol):
    def estimate(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:  # pragma: no cover - interface
        ...


class OptimizerMemoryAdapter(Protocol):
    def unavailable(self, *, method: str, parameter_count: int, reason: str) -> dict[str, Any]:
        ...

    def from_result(
        self,
        result: Any,
        *,
        method: str,
        parameter_count: int,
        source: str,
    ) -> dict[str, Any]:
        ...

    def remap_insert(
        self,
        state: Mapping[str, Any] | None,
        *,
        position_id: int,
        count: int = 1,
    ) -> dict[str, Any]:
        ...

    def remap_remove(
        self,
        state: Mapping[str, Any] | None,
        *,
        indices: Sequence[int],
    ) -> dict[str, Any]:
        ...

    def select_active(
        self,
        state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        source: str,
    ) -> dict[str, Any]:
        ...

    def merge_active(
        self,
        base_state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        active_state: Mapping[str, Any] | None,
        source: str,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class QNSPSARefreshPlan:
    enabled: bool = False
    refresh_every: int = 0
    mode: str = "disabled"
    skip_reason: str = ""
    refresh_points: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class MotifMetadata:
    enabled: bool = False
    motif_tags: list[str] = field(default_factory=list)
    motif_ids: list[str] = field(default_factory=list)
    motif_source: str = "none"
    tiled_from_num_sites: int | None = None
    target_num_sites: int | None = None
    boundary_behavior: str | None = None
    transfer_mode: str = "exact_match_v1"


@dataclass(frozen=True)
class SymmetrySpec:
    spec_version: str = "phase3_symmetry_v1"
    particle_number_mode: str = "preserving"
    spin_sector_mode: str = "preserving"
    phonon_number_mode: str = "not_conserved"
    leakage_risk: float = 0.0
    mitigation_eligible: bool = False
    grouping_eligible: bool = True
    hard_guard: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeneratorMetadata:
    generator_id: str
    family_id: str
    template_id: str
    candidate_label: str
    support_qubits: list[int]
    support_sites: list[int]
    support_site_offsets: list[int]
    is_macro_generator: bool
    split_policy: str
    parent_generator_id: str | None = None
    symmetry_spec: dict[str, Any] | None = None
    compile_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratorSplitEvent:
    parent_generator_id: str
    child_generator_ids: list[str]
    reason: str
    split_mode: str


@dataclass(frozen=True)
class MotifRecord:
    motif_id: str
    family_id: str
    template_id: str
    source_num_sites: int
    relative_order: int
    support_site_offsets: list[int]
    mean_theta: float
    mean_abs_theta: float
    sign_hint: int
    generator_ids: list[str]
    symmetry_spec: dict[str, Any] | None = None
    boundary_behavior: str = "interior_only"
    source_tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MotifLibrary:
    library_version: str
    source_tag: str
    source_num_sites: int
    ordering: str
    boson_encoding: str
    source_tags: list[str] = field(default_factory=list)
    records: list[MotifRecord] = field(default_factory=list)


@dataclass(frozen=True)
class RescueDiagnostic:
    enabled: bool = False
    triggered: bool = False
    reason: str = "disabled"
    shortlisted_labels: list[str] = field(default_factory=list)
    selected_label: str | None = None
    selected_position: int | None = None
    overlap_gain: float = 0.0


class Phase2OptimizerMemoryAdapter:
    """Deterministic remapping adapter for persisted optimizer memory."""

    _VECTOR_KEYS = (
        "preconditioner_diag",
        "grad_sq_ema",
    )

    def unavailable(self, *, method: str, parameter_count: int, reason: str) -> dict[str, Any]:
        return {
            "version": "phase2_optimizer_memory_v1",
            "optimizer": str(method),
            "parameter_count": int(max(0, parameter_count)),
            "available": False,
            "reason": str(reason),
            "source": "unavailable",
            "reused": False,
            "preconditioner_diag": [1.0] * int(max(0, parameter_count)),
            "grad_sq_ema": [0.0] * int(max(0, parameter_count)),
            "history_tail": [],
            "refresh_points": [],
            "remap_events": [],
        }

    def from_result(
        self,
        result: Any,
        *,
        method: str,
        parameter_count: int,
        source: str,
    ) -> dict[str, Any]:
        raw = getattr(result, "optimizer_memory", None)
        if not isinstance(raw, Mapping):
            return self.unavailable(
                method=str(method),
                parameter_count=int(parameter_count),
                reason="optimizer_memory_missing",
            )
        state = self._normalize(raw, parameter_count=int(parameter_count))
        state["source"] = str(source)
        return state

    def remap_insert(
        self,
        state: Mapping[str, Any] | None,
        *,
        position_id: int,
        count: int = 1,
    ) -> dict[str, Any]:
        base = self._normalize(state, parameter_count=self._parameter_count(state))
        n = int(base["parameter_count"])
        pos = max(0, min(int(position_id), n))
        add_n = int(max(0, count))
        for key, default in (("preconditioner_diag", 1.0), ("grad_sq_ema", 0.0)):
            vec = list(base.get(key, []))
            base[key] = vec[:pos] + ([float(default)] * add_n) + vec[pos:]
        base["parameter_count"] = int(n + add_n)
        self._append_remap_event(base, {"op": "insert", "position_id": int(pos), "count": int(add_n)})
        return base

    def remap_remove(
        self,
        state: Mapping[str, Any] | None,
        *,
        indices: Sequence[int],
    ) -> dict[str, Any]:
        base = self._normalize(state, parameter_count=self._parameter_count(state))
        n = int(base["parameter_count"])
        drop = sorted({int(i) for i in indices if 0 <= int(i) < n})
        keep = [i for i in range(n) if i not in set(drop)]
        for key in self._VECTOR_KEYS:
            vec = list(base.get(key, []))
            base[key] = [float(vec[i]) for i in keep]
        base["parameter_count"] = int(len(keep))
        self._append_remap_event(base, {"op": "remove", "indices": [int(i) for i in drop]})
        return base

    def select_active(
        self,
        state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        source: str,
    ) -> dict[str, Any]:
        base = self._normalize(state, parameter_count=self._parameter_count(state))
        n = int(base["parameter_count"])
        active = [int(i) for i in active_indices if 0 <= int(i) < n]
        out = {
            **base,
            "parameter_count": int(len(active)),
            "preconditioner_diag": [float(base["preconditioner_diag"][i]) for i in active],
            "grad_sq_ema": [float(base["grad_sq_ema"][i]) for i in active],
            "source": str(source),
            "reused": bool(base.get("available", False) and len(active) > 0),
            "active_indices": [int(i) for i in active],
        }
        self._append_remap_event(out, {"op": "select_active", "active_indices": [int(i) for i in active]})
        return out

    def merge_active(
        self,
        base_state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        active_state: Mapping[str, Any] | None,
        source: str,
    ) -> dict[str, Any]:
        base = self._normalize(base_state, parameter_count=self._parameter_count(base_state))
        active_norm = self._normalize(active_state, parameter_count=len(list(active_indices)))
        n = int(base["parameter_count"])
        active = [int(i) for i in active_indices if 0 <= int(i) < n]
        for key, default in (("preconditioner_diag", 1.0), ("grad_sq_ema", 0.0)):
            vec = list(base.get(key, [float(default)] * n))
            active_vec = list(active_norm.get(key, []))
            for k, idx in enumerate(active):
                if k < len(active_vec):
                    vec[idx] = float(active_vec[k])
            base[key] = vec
        base["source"] = str(source)
        base["available"] = bool(base.get("available", False) or active_norm.get("available", False))
        base["reused"] = bool(active_norm.get("reused", False))
        refresh = list(base.get("refresh_points", []))
        refresh.extend(int(x) for x in active_norm.get("refresh_points", []) if int(x) not in refresh)
        base["refresh_points"] = refresh
        self._append_remap_event(base, {"op": "merge_active", "active_indices": [int(i) for i in active]})
        return base

    def _parameter_count(self, state: Mapping[str, Any] | None) -> int:
        if isinstance(state, Mapping) and state.get("parameter_count") is not None:
            return int(max(0, int(state.get("parameter_count", 0))))
        if isinstance(state, Mapping):
            for key in self._VECTOR_KEYS:
                raw = state.get(key, None)
                if isinstance(raw, Sequence):
                    return int(len(list(raw)))
        return 0

    def _normalize(self, state: Mapping[str, Any] | None, *, parameter_count: int) -> dict[str, Any]:
        n = int(max(0, parameter_count))
        if not isinstance(state, Mapping):
            return self.unavailable(method="unknown", parameter_count=n, reason="missing_state")
        out = {
            "version": str(state.get("version", "phase2_optimizer_memory_v1")),
            "optimizer": str(state.get("optimizer", "unknown")),
            "parameter_count": int(n),
            "available": bool(state.get("available", False)),
            "reason": str(state.get("reason", "")),
            "source": str(state.get("source", "")),
            "reused": bool(state.get("reused", False)),
            "history_tail": [dict(x) for x in state.get("history_tail", []) if isinstance(x, Mapping)][-32:],
            "refresh_points": [int(x) for x in state.get("refresh_points", [])],
            "remap_events": [dict(x) for x in state.get("remap_events", []) if isinstance(x, Mapping)][-32:],
        }
        for key, default in (("preconditioner_diag", 1.0), ("grad_sq_ema", 0.0)):
            raw = list(state.get(key, [])) if isinstance(state.get(key, []), Sequence) else []
            vec = [float(default)] * n
            for i in range(min(n, len(raw))):
                vec[i] = float(raw[i])
            out[key] = vec
        return out

    def _append_remap_event(self, state: dict[str, Any], event: Mapping[str, Any]) -> None:
        events = [dict(x) for x in state.get("remap_events", []) if isinstance(x, Mapping)]
        events.append({str(k): v for k, v in event.items()})
        state["remap_events"] = events[-32:]

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_hh_continuation_replay.py
```py
from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_replay import (
    ReplayControllerConfig,
    build_replay_plan,
    run_phase1_replay,
    run_phase2_replay,
    run_phase3_replay,
)


class _DummyAnsatz:
    def __init__(self, npar: int) -> None:
        self.num_parameters = int(npar)

    def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        # Deterministic, normalized return for tests.
        out = np.array(psi_ref, copy=True)
        out[0] = complex(1.0 + 0.0 * float(np.sum(theta)))
        return out / np.linalg.norm(out)


@dataclass
class _DummyRes:
    x: np.ndarray
    energy: float
    nfev: int
    nit: int
    success: bool
    message: str
    best_restart: int = 0
    restart_summaries: list[dict[str, float]] | None = None
    optimizer_memory: dict[str, object] | None = None


def _fake_vqe_minimize(_h, ansatz, _psi_ref, **kwargs):
    x0 = np.asarray(kwargs.get("initial_point", np.zeros(int(ansatz.num_parameters))), dtype=float)
    x = np.array(x0, copy=True)
    e = float(np.sum(x**2))
    return _DummyRes(
        x=x,
        energy=e,
        nfev=5,
        nit=3,
        success=True,
        message="ok",
        best_restart=0,
        restart_summaries=[{"best_energy": e + 0.1}],
        optimizer_memory={
            "version": "phase2_spsa_diag_memory_v1",
            "optimizer": "SPSA",
            "parameter_count": int(x.size),
            "available": True,
            "source": "dummy",
            "reused": True,
            "preconditioner_diag": [1.0] * int(x.size),
            "grad_sq_ema": [0.1] * int(x.size),
            "history_tail": [],
            "refresh_points": [2] if int(kwargs.get("spsa_refresh_every", 0)) > 0 else [],
            "remap_events": [],
        },
    )


def test_build_replay_plan_splits_steps() -> None:
    cfg = ReplayControllerConfig(freeze_fraction=0.2, unfreeze_fraction=0.3, full_fraction=0.5)
    plan = build_replay_plan(
        continuation_mode="phase1_v1",
        seed_policy_resolved="residual_only",
        handoff_state_kind="prepared_state",
        scaffold_block_indices=[0, 1],
        residual_block_indices=[2, 3, 4],
        maxiter=100,
        cfg=cfg,
    )
    assert plan.freeze_scaffold_steps > 0
    assert plan.unfreeze_steps > 0
    assert plan.full_replay_steps > 0
    assert len(plan.trust_radius_schedule) == 3
    assert plan.handoff_state_kind == "prepared_state"
    assert plan.scaffold_block_indices == [0, 1]
    assert plan.residual_block_indices == [2, 3, 4]


def test_run_phase1_replay_emits_phase_history() -> None:
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    theta, hist, meta = run_phase1_replay(
        vqe_minimize_fn=_fake_vqe_minimize,
        h_poly=None,
        ansatz=_DummyAnsatz(6),
        psi_ref=psi_ref,
        seed_theta=np.zeros(6, dtype=float),
        scaffold_block_size=2,
        seed_policy_resolved="residual_only",
        handoff_state_kind="prepared_state",
        cfg=ReplayControllerConfig(),
        restarts=2,
        seed=7,
        maxiter=60,
        method="SPSA",
        progress_every_s=60.0,
        exact_energy=None,
        kwargs={},
    )
    assert len(theta) == 6
    assert len(hist) == 3
    assert hist[0]["phase_name"] == "seed_burn_in"
    assert hist[1]["phase_name"] == "constrained_unfreeze"
    assert hist[2]["phase_name"] == "full_replay"
    assert "replay_phase_config" in meta
    cfg_out = meta["replay_phase_config"]
    assert cfg_out["handoff_state_kind"] == "prepared_state"
    assert cfg_out["scaffold_block_indices"] == [0, 1]
    assert cfg_out["residual_block_indices"] == [2, 3, 4, 5]
    assert "qn_spsa_refresh_every" in cfg_out


def test_run_phase2_replay_reuses_memory_and_logs_refresh() -> None:
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    theta, hist, meta = run_phase2_replay(
        vqe_minimize_fn=_fake_vqe_minimize,
        h_poly=None,
        ansatz=_DummyAnsatz(6),
        psi_ref=psi_ref,
        seed_theta=np.zeros(6, dtype=float),
        scaffold_block_size=2,
        seed_policy_resolved="scaffold_plus_zero",
        handoff_state_kind="reference_state",
        cfg=ReplayControllerConfig(qn_spsa_refresh_every=2),
        restarts=2,
        seed=7,
        maxiter=60,
        method="SPSA",
        progress_every_s=60.0,
        exact_energy=None,
        kwargs={},
        incoming_optimizer_memory={
            "version": "phase2_optimizer_memory_v1",
            "optimizer": "SPSA",
            "parameter_count": 2,
            "available": True,
            "source": "handoff",
            "reused": True,
            "preconditioner_diag": [1.0, 1.0],
            "grad_sq_ema": [0.1, 0.1],
            "history_tail": [],
            "refresh_points": [],
            "remap_events": [],
        },
    )
    assert len(theta) == 6
    assert len(hist) == 3
    cfg_out = meta["replay_phase_config"]
    assert cfg_out["continuation_mode"] == "phase2_v1"
    assert cfg_out["optimizer_memory_reused"] is True
    assert cfg_out["optimizer_memory_source"] in {"handoff_scaffold_expand", "handoff_full", "handoff_resized"}
    assert cfg_out["qn_spsa_refresh"]["enabled"] is True
    assert cfg_out["qn_spsa_refresh"]["refresh_points"] == [2]
    assert cfg_out["optimizer_memory"]["parameter_count"] == 6


def test_run_phase2_replay_missing_optimizer_memory_degrades_gracefully() -> None:
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    theta, hist, meta = run_phase2_replay(
        vqe_minimize_fn=_fake_vqe_minimize,
        h_poly=None,
        ansatz=_DummyAnsatz(6),
        psi_ref=psi_ref,
        seed_theta=np.zeros(6, dtype=float),
        scaffold_block_size=2,
        seed_policy_resolved="scaffold_plus_zero",
        handoff_state_kind="reference_state",
        cfg=ReplayControllerConfig(),
        restarts=2,
        seed=7,
        maxiter=60,
        method="SPSA",
        progress_every_s=60.0,
        exact_energy=None,
        kwargs={},
        incoming_optimizer_memory=None,
    )
    assert len(theta) == 6
    assert len(hist) == 3
    cfg_out = meta["replay_phase_config"]
    assert cfg_out["continuation_mode"] == "phase2_v1"
    assert cfg_out["optimizer_memory_source"] == "missing_handoff_optimizer_memory"
    assert cfg_out["optimizer_memory_reused"] is False
    assert cfg_out["optimizer_memory"]["parameter_count"] == 6
    assert cfg_out["optimizer_memory"]["available"] is True


def test_run_phase3_replay_emits_generator_motif_and_symmetry_fields() -> None:
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    theta, hist, meta = run_phase3_replay(
        vqe_minimize_fn=_fake_vqe_minimize,
        h_poly=None,
        ansatz=_DummyAnsatz(6),
        psi_ref=psi_ref,
        seed_theta=np.zeros(6, dtype=float),
        scaffold_block_size=2,
        seed_policy_resolved="scaffold_plus_zero",
        handoff_state_kind="reference_state",
        cfg=ReplayControllerConfig(
            qn_spsa_refresh_every=2,
            qn_spsa_refresh_mode="diag_rms_grad",
            symmetry_mitigation_mode="verify_only",
        ),
        restarts=2,
        seed=7,
        maxiter=60,
        method="SPSA",
        progress_every_s=60.0,
        exact_energy=None,
        kwargs={},
        incoming_optimizer_memory={
            "version": "phase2_optimizer_memory_v1",
            "optimizer": "SPSA",
            "parameter_count": 2,
            "available": True,
            "source": "handoff",
            "reused": True,
            "preconditioner_diag": [1.0, 1.0],
            "grad_sq_ema": [0.1, 0.1],
            "history_tail": [],
            "refresh_points": [],
            "remap_events": [],
        },
        generator_ids=["gen:a", "gen:b"],
        motif_reference_ids=["motif:1"],
    )
    assert len(theta) == 6
    assert len(hist) == 3
    cfg_out = meta["replay_phase_config"]
    assert cfg_out["continuation_mode"] == "phase3_v1"
    assert cfg_out["symmetry_mitigation_mode"] == "verify_only"
    assert cfg_out["generator_ids"] == ["gen:a", "gen:b"]
    assert cfg_out["motif_reference_ids"] == ["motif:1"]
    assert cfg_out["qn_spsa_refresh"]["enabled"] is True


def test_run_phase3_replay_skips_empty_seed_burn_in_phase() -> None:
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    theta, hist, meta = run_phase3_replay(
        vqe_minimize_fn=_fake_vqe_minimize,
        h_poly=None,
        ansatz=_DummyAnsatz(6),
        psi_ref=psi_ref,
        seed_theta=np.zeros(6, dtype=float),
        scaffold_block_size=6,
        seed_policy_resolved="residual_only",
        handoff_state_kind="prepared_state",
        cfg=ReplayControllerConfig(),
        restarts=2,
        seed=7,
        maxiter=60,
        method="SPSA",
        progress_every_s=60.0,
        exact_energy=None,
        kwargs={},
        incoming_optimizer_memory=None,
        generator_ids=["gen:a"],
        motif_reference_ids=[],
    )
    assert len(theta) == 6
    assert len(hist) == 3
    assert hist[0]["phase_name"] == "seed_burn_in"
    assert hist[0]["active_count"] == 0
    assert hist[0]["nfev"] == 0
    assert hist[1]["phase_name"] == "constrained_unfreeze"
    assert hist[1]["active_count"] > 0
    assert meta["result"]["success"] is True

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/handoff_state_bundle.py
```py
#!/usr/bin/env python3
"""Reusable handoff-state bundle writer for hardcoded HH workflows."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HandoffStateBundleConfig:
    L: int
    t: float
    U: float
    dv: float
    omega0: float
    g_ep: float
    n_ph_max: int
    boson_encoding: str
    ordering: str
    boundary: str
    sector_n_up: int
    sector_n_dn: int


def build_handoff_settings_manifest(
    cfg: HandoffStateBundleConfig,
    *,
    adapt_pool: str | None = None,
) -> dict[str, Any]:
    out = {
        "L": int(cfg.L),
        "problem": "hh",
        "t": float(cfg.t),
        "u": float(cfg.U),
        "dv": float(cfg.dv),
        "omega0": float(cfg.omega0),
        "g_ep": float(cfg.g_ep),
        "n_ph_max": int(cfg.n_ph_max),
        "boson_encoding": str(cfg.boson_encoding),
        "ordering": str(cfg.ordering),
        "boundary": str(cfg.boundary),
        "sector_n_up": int(cfg.sector_n_up),
        "sector_n_dn": int(cfg.sector_n_dn),
    }
    if adapt_pool is not None:
        out["adapt_pool"] = str(adapt_pool)
    return out


def _statevector_to_amplitudes_qn_to_q0(
    psi_state: np.ndarray,
    *,
    cutoff: float,
) -> dict[str, dict[str, float]]:
    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    nq_total = int(round(math.log2(int(psi.size))))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(psi):
        if abs(amp) <= float(cutoff):
            continue
        out[format(idx, f"0{nq_total}b")] = {
            "re": float(np.real(amp)),
            "im": float(np.imag(amp)),
        }
    return out


def write_handoff_state_bundle(
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
    """Write an adapt_json-compatible HH handoff bundle."""

    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(psi))
    if norm <= 0.0:
        raise ValueError("psi_state must be non-zero.")
    psi = psi / norm
    nq_total = int(round(math.log2(int(psi.size))))
    amps = _statevector_to_amplitudes_qn_to_q0(psi, cutoff=float(amplitude_cutoff))

    adapt_vqe_block: dict[str, Any] = {
        "energy": float(energy),
        "abs_delta_e": float(delta_E_abs),
        "relative_error_abs": float(relative_error_abs),
    }
    if adapt_operators is not None and adapt_optimal_point is not None:
        adapt_vqe_block["operators"] = list(adapt_operators)
        adapt_vqe_block["optimal_point"] = [float(x) for x in adapt_optimal_point]
        adapt_vqe_block["ansatz_depth"] = int(len(adapt_operators))
        adapt_vqe_block["num_parameters"] = int(len(adapt_optimal_point))
    if adapt_pool_type is not None:
        adapt_vqe_block["pool_type"] = str(adapt_pool_type)
    if pre_prune_scaffold is not None:
        adapt_vqe_block["pre_prune_scaffold"] = dict(pre_prune_scaffold)
    if prune_summary is not None:
        adapt_vqe_block["prune_summary"] = dict(prune_summary)

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settings": build_handoff_settings_manifest(cfg, adapt_pool=settings_adapt_pool),
        "adapt_vqe": adapt_vqe_block,
        "ground_state": {
            "exact_energy": float(exact_energy),
            "exact_energy_filtered": float(exact_energy),
            "filtered_sector": {
                "n_up": int(cfg.sector_n_up),
                "n_dn": int(cfg.sector_n_dn),
            },
            "method": "staged_handoff_bundle",
        },
        "initial_state": {
            "source": str(source),
            "nq_total": nq_total,
            "amplitudes_qn_to_q0": amps,
            "amplitude_cutoff": float(amplitude_cutoff),
            "norm": float(np.linalg.norm(psi)),
            **(
                {"handoff_state_kind": str(handoff_state_kind)}
                if handoff_state_kind is not None
                else {}
            ),
        },
        "exact": {
            "E_exact_sector": float(exact_energy),
        },
    }
    continuation_block: dict[str, Any] = {}
    if continuation_mode is not None:
        continuation_block["mode"] = str(continuation_mode)
    if continuation_scaffold is not None:
        continuation_block["scaffold"] = dict(continuation_scaffold)
    if optimizer_memory is not None:
        continuation_block["optimizer_memory"] = dict(optimizer_memory)
    if selected_generator_metadata is not None:
        continuation_block["selected_generator_metadata"] = [dict(x) for x in selected_generator_metadata]
    if generator_split_events is not None:
        continuation_block["generator_split_events"] = [dict(x) for x in generator_split_events]
    if motif_library is not None:
        continuation_block["motif_library"] = dict(motif_library)
    if motif_usage is not None:
        continuation_block["motif_usage"] = dict(motif_usage)
    if symmetry_mitigation is not None:
        continuation_block["symmetry_mitigation"] = dict(symmetry_mitigation)
    if rescue_history is not None:
        continuation_block["rescue_history"] = [dict(x) for x in rescue_history]
    if replay_contract is not None:
        continuation_block["replay_contract"] = dict(replay_contract)
    if replay_contract_hint is not None:
        continuation_block["replay_contract_hint"] = dict(replay_contract_hint)
    if continuation_block:
        payload["continuation"] = continuation_block
    if isinstance(vqe_payload, dict):
        payload["vqe"] = dict(vqe_payload)
    if isinstance(meta, dict):
        payload["meta"] = dict(meta)
    if isinstance(seed_provenance, dict):
        payload["seed_provenance"] = dict(seed_provenance)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_scoring.py
```py
#!/usr/bin/env python3
"""Scoring and reduced-path derivative accounting for HH continuation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.hh_continuation_types import (
    CandidateFeatures,
    CompileCostEstimate,
    CurvatureOracle,
    MeasurementCacheStats,
    MeasurementPlan,
    NoveltyOracle,
)
from pipelines.hardcoded.hh_continuation_motifs import motif_bonus_for_generator
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.pauli_actions import apply_compiled_pauli


@dataclass(frozen=True)
class SimpleScoreConfig:
    lambda_F: float = 1.0
    lambda_compile: float = 0.05
    lambda_measure: float = 0.02
    lambda_leak: float = 0.0
    z_alpha: float = 0.0
    wD: float = 0.0
    wG: float = 0.0
    wC: float = 0.0
    wc: float = 0.0
    depth_ref: float = 1.0
    group_ref: float = 1.0
    shot_ref: float = 1.0
    family_ref: float = 1.0
    lifetime_cost_mode: str = "off"
    score_version: str = "append_screen_v1"


@dataclass(frozen=True)
class FullScoreConfig:
    z_alpha: float = 0.0
    lambda_F: float = 1.0
    lambda_H: float = 1e-6
    rho: float = 0.25
    eta_L: float = 0.0
    gamma_N: float = 1.0
    wD: float = 0.2
    wG: float = 0.15
    wC: float = 0.15
    wP: float = 0.1
    wc: float = 0.1
    depth_ref: float = 1.0
    group_ref: float = 1.0
    shot_ref: float = 1.0
    optdim_ref: float = 1.0
    reuse_ref: float = 1.0
    family_ref: float = 1.0
    novelty_eps: float = 1e-6
    shortlist_fraction: float = 0.2
    shortlist_size: int = 12
    batch_target_size: int = 2
    batch_size_cap: int = 3
    batch_near_degenerate_ratio: float = 0.9
    compat_overlap_weight: float = 0.4
    compat_comm_weight: float = 0.2
    compat_curv_weight: float = 0.2
    compat_sched_weight: float = 0.2
    leakage_cap: float = 1e6
    lifetime_cost_mode: str = "off"
    remaining_evaluations_proxy_mode: str = "none"
    lifetime_weight: float = 0.05
    motif_bonus_weight: float = 0.0
    metric_floor: float = 1e-12
    reduced_metric_collapse_rel_tol: float = 1e-8
    ridge_growth_factor: float = 10.0
    ridge_max_steps: int = 12
    score_version: str = "append_full_v1"


@dataclass(frozen=True)
class _ScaffoldDerivativeContext:
    psi_state: np.ndarray
    hpsi_state: np.ndarray
    refit_window_indices: tuple[int, ...]
    dpsi_window: tuple[np.ndarray, ...]
    tangents_window: tuple[np.ndarray, ...]
    Q_window: np.ndarray
    H_window_hessian: np.ndarray


class Phase1CompileCostOracle:
    """Built-in math expression:
    D_proxy = n_new_pauli + n_rot + shift_span + active_count
    """

    def estimate(
        self,
        *,
        candidate_term_count: int,
        position_id: int,
        append_position: int,
        refit_active_count: int,
    ) -> CompileCostEstimate:
        new_pauli_actions = float(max(1, int(candidate_term_count)))
        new_rotation_steps = float(max(1, int(candidate_term_count)))
        position_shift_span = float(abs(int(append_position) - int(position_id)))
        refit_active = float(max(0, int(refit_active_count)))
        total = float(new_pauli_actions + new_rotation_steps + position_shift_span + refit_active)
        return CompileCostEstimate(
            new_pauli_actions=new_pauli_actions,
            new_rotation_steps=new_rotation_steps,
            position_shift_span=position_shift_span,
            refit_active_count=refit_active,
            proxy_total=total,
        )


class MeasurementCacheAudit:
    """Phase 1 accounting-only grouped reuse tracker."""

    def __init__(
        self,
        nominal_shots_per_group: int = 1,
        *,
        plan_version: str = "phase1_grouped_label_reuse",
        grouping_mode: str = "grouped_label_reuse",
    ) -> None:
        self._seen_groups: set[str] = set()
        self._nominal_shots = int(max(1, nominal_shots_per_group))
        self._plan_version = str(plan_version)
        self._grouping_mode = str(grouping_mode)

    def clone(self) -> "MeasurementCacheAudit":
        cloned = MeasurementCacheAudit(
            nominal_shots_per_group=int(self._nominal_shots),
            plan_version=str(self._plan_version),
            grouping_mode=str(self._grouping_mode),
        )
        cloned._seen_groups = set(self._seen_groups)
        return cloned

    def snapshot(self) -> dict[str, Any]:
        return {
            "seen_groups": sorted(str(x) for x in self._seen_groups),
            "nominal_shots_per_group": int(self._nominal_shots),
            "plan_version": str(self._plan_version),
            "grouping_mode": str(self._grouping_mode),
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "MeasurementCacheAudit":
        cloned = cls(
            nominal_shots_per_group=int(snapshot.get("nominal_shots_per_group", 1)),
            plan_version=str(snapshot.get("plan_version", "phase1_grouped_label_reuse")),
            grouping_mode=str(snapshot.get("grouping_mode", "grouped_label_reuse")),
        )
        cloned._seen_groups = {
            str(x)
            for x in snapshot.get("seen_groups", [])
            if str(x) != ""
        }
        return cloned

    def plan_for(self, group_keys: Iterable[str]) -> MeasurementPlan:
        keys = [str(k) for k in group_keys if str(k) != ""]
        unique_keys: list[str] = []
        for key in keys:
            if key not in unique_keys:
                unique_keys.append(key)
        return MeasurementPlan(
            plan_version=str(self._plan_version),
            group_keys=list(unique_keys),
            nominal_shots_per_group=int(self._nominal_shots),
            grouping_mode=str(self._grouping_mode),
        )

    def estimate(self, group_keys: Iterable[str]) -> MeasurementCacheStats:
        plan = self.plan_for(group_keys)
        unique_keys = list(plan.group_keys)

        groups_total = int(len(unique_keys))
        groups_reused = 0
        for key in unique_keys:
            if key in self._seen_groups:
                groups_reused += 1
        groups_new = int(groups_total - groups_reused)
        shots_reused = float(groups_reused * self._nominal_shots)
        shots_new = float(groups_new * self._nominal_shots)
        reuse_count_cost = float(groups_new)
        return MeasurementCacheStats(
            groups_total=groups_total,
            groups_reused=int(groups_reused),
            groups_new=int(groups_new),
            shots_reused=shots_reused,
            shots_new=shots_new,
            reuse_count_cost=reuse_count_cost,
        )

    def commit(self, group_keys: Iterable[str]) -> None:
        for key in group_keys:
            key_s = str(key)
            if key_s != "":
                self._seen_groups.add(key_s)

    def summary(self) -> dict[str, float]:
        return {
            "groups_known": float(len(self._seen_groups)),
            "nominal_shots_per_group": float(self._nominal_shots),
            "plan_version": str(self._plan_version),
            "grouping_mode": str(self._grouping_mode),
        }


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _replace_feature(feat: CandidateFeatures, **updates: Any) -> CandidateFeatures:
    return CandidateFeatures(**{**feat.__dict__, **updates})


def normalize(value: float, ref: float) -> float:
    denom = float(ref)
    if not math.isfinite(denom) or denom <= 0.0:
        return float(max(0.0, value))
    return float(max(0.0, value) / denom)


def trust_region_drop(g_lcb: float, h_eff: float, F: float, rho: float) -> float:
    if float(g_lcb) <= 0.0:
        return 0.0
    F_pos = float(max(float(F), 1e-12))
    h_eff_pos = float(max(0.0, h_eff))
    alpha_max = float(rho) / float(math.sqrt(F_pos))
    if h_eff_pos > 0.0:
        alpha_newton = float(g_lcb) / h_eff_pos
        if alpha_newton <= alpha_max:
            return float(0.5 * float(g_lcb) * float(g_lcb) / h_eff_pos)
    return float(float(g_lcb) * alpha_max - 0.5 * h_eff_pos * alpha_max * alpha_max)


def remaining_evaluations_proxy(
    *,
    current_depth: int | None,
    max_depth: int | None,
    mode: str,
) -> float:
    mode_key = str(mode).strip().lower()
    if mode_key == "none":
        return 0.0
    depth_now = 0 if current_depth is None else int(max(0, current_depth))
    depth_cap = depth_now if max_depth is None else int(max(depth_now, max_depth))
    if mode_key == "remaining_depth":
        return float(max(1, depth_cap - depth_now + 1))
    raise ValueError("remaining_evaluations_proxy_mode must be 'none' or 'remaining_depth'")


def family_repeat_cost_from_history(
    *,
    history_rows: Sequence[Mapping[str, Any]],
    candidate_family: str,
) -> float:
    fam = str(candidate_family).strip()
    if fam == "":
        return 0.0
    tail = [row for row in history_rows if isinstance(row, Mapping) and row.get("candidate_family") is not None]
    if not tail:
        return 0.0
    if str(tail[-1].get("candidate_family", "")).strip() != fam:
        return 0.0
    streak = 0
    for row in reversed(tail):
        if str(row.get("candidate_family", "")).strip() != fam:
            break
        streak += 1
    return float(streak)


def lifetime_weight_components(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> dict[str, float]:
    rem = float(max(0.0, feat.remaining_evaluations_proxy))
    if str(cfg.lifetime_cost_mode).strip().lower() == "off":
        return {
            "remaining_evaluations_proxy": float(rem),
            "depth_life": 0.0,
            "total": 0.0,
        }
    depth_life = float(max(1.0, rem) * normalize(float(feat.depth_cost), float(cfg.depth_ref)))
    return {
        "remaining_evaluations_proxy": float(rem),
        "depth_life": float(depth_life),
        "total": float(depth_life),
    }


def _depth_life_cost(feat: CandidateFeatures, cfg: FullScoreConfig) -> float:
    base = normalize(float(feat.depth_cost), float(cfg.depth_ref))
    if str(cfg.lifetime_cost_mode).strip().lower() == "off":
        return float(base)
    rem = float(max(1.0, feat.remaining_evaluations_proxy))
    return float(rem * base)


def _score_denominator(feat: CandidateFeatures, cfg: FullScoreConfig) -> float:
    return float(
        1.0
        + float(cfg.wD) * float(_depth_life_cost(feat, cfg))
        + float(cfg.wG) * normalize(float(feat.new_group_cost), float(cfg.group_ref))
        + float(cfg.wC) * normalize(float(feat.new_shot_cost), float(cfg.shot_ref))
        + float(cfg.wc) * normalize(float(feat.family_repeat_cost), float(cfg.family_ref))
    )


# ---------------------------------------------------------------------------
# Active append-only scoring surface
# ---------------------------------------------------------------------------

def _screen_denominator(feat: CandidateFeatures, cfg: SimpleScoreConfig) -> float:
    wD = float(cfg.wD if cfg.wD != 0.0 else cfg.lambda_compile)
    wG = float(cfg.wG if cfg.wG != 0.0 else cfg.lambda_measure)
    wC = float(cfg.wC if cfg.wC != 0.0 else cfg.lambda_measure)
    wc = float(cfg.wc if cfg.wc != 0.0 else cfg.lambda_measure)
    depth_life = normalize(float(feat.depth_cost), float(cfg.depth_ref))
    if str(cfg.lifetime_cost_mode).strip().lower() != "off":
        depth_life *= float(max(1.0, feat.remaining_evaluations_proxy))
    return float(
        1.0
        + wD * depth_life
        + wG * normalize(float(feat.new_group_cost), float(cfg.group_ref))
        + wC * normalize(float(feat.new_shot_cost), float(cfg.shot_ref))
        + wc * normalize(float(feat.family_repeat_cost), float(cfg.family_ref))
    )


def simple_v1_score(
    feat: CandidateFeatures,
    cfg: SimpleScoreConfig,
) -> float:
    if not bool(feat.stage_gate_open):
        return float("-inf")
    g_lcb = float(max(0.0, feat.g_lcb))
    F_raw = float(max(float(feat.F_raw if feat.F_raw is not None else feat.F_metric), 1e-12))
    if g_lcb <= 0.0:
        return 0.0
    lamF = float(max(float(cfg.lambda_F), 1e-12))
    delta_e_screen = 0.5 * g_lcb * g_lcb / (lamF * F_raw)
    return float(delta_e_screen / float(_screen_denominator(feat, cfg)))


def full_v2_score(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> tuple[float, str]:
    if not bool(feat.stage_gate_open):
        return float("-inf"), "blocked_stage"
    g_lcb = float(max(0.0, feat.g_lcb))
    if g_lcb <= 0.0:
        return 0.0, "nonpositive_gradient"

    F_red_raw = feat.F_red if feat.F_red is not None else feat.F_raw
    if F_red_raw is None:
        return 0.0, "missing_reduced_path_metric"
    h_eff = float(feat.h_eff if feat.h_eff is not None else (feat.h_raw if feat.h_raw is not None else 0.0))
    F_red = float(max(float(F_red_raw), float(cfg.metric_floor)))
    novelty = 1.0 if feat.novelty is None else float(min(1.0, max(0.0, feat.novelty)))
    if str(feat.curvature_mode).startswith("append_exact_metric_collapse") or novelty <= 0.0:
        return 0.0, "reduced_metric_collapse"
    delta_e = trust_region_drop(g_lcb, float(max(0.0, h_eff)), F_red, float(cfg.rho))
    if delta_e <= 0.0:
        return 0.0, "nonpositive_trust_region_drop"
    score = (float(novelty) ** float(cfg.gamma_N)) * float(delta_e) / float(_score_denominator(feat, cfg))
    if feat.ridge_used is not None and float(feat.ridge_used) > float(max(cfg.lambda_H, 0.0)):
        return float(score), "append_exact_reduced_path_ridge_grown"
    if len(feat.refit_window_indices) == 0:
        return float(score), "append_exact_empty_window"
    return float(score), "append_exact_reduced_path"


# ---------------------------------------------------------------------------
# Shortlist ranking helpers
# ---------------------------------------------------------------------------

def shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    score_key: str = "simple_score",
) -> list[dict[str, Any]]:
    ranked = sorted(
        [dict(rec) for rec in records],
        key=lambda rec: (
            -float(rec.get(score_key, float("-inf"))),
            -float(rec.get("simple_score", float("-inf"))),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return []
    total = int(len(ranked))
    target = int(max(1, min(total, cfg.shortlist_size, math.ceil(float(cfg.shortlist_fraction) * total))))
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(ranked[:target], start=1):
        updated = dict(rec)
        feat = updated.get("feature", None)
        if isinstance(feat, CandidateFeatures):
            updated["feature"] = _replace_feature(
                feat,
                shortlist_rank=int(idx),
                shortlist_size=int(target),
            )
        out.append(updated)
    return out


# ---------------------------------------------------------------------------
# Exact append-only reduced-path derivative helpers
# ---------------------------------------------------------------------------

def _executor_for_terms(
    terms: Sequence[Any],
    *,
    pauli_action_cache: dict[str, Any] | None,
) -> CompiledAnsatzExecutor:
    return CompiledAnsatzExecutor(
        list(terms),
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        pauli_action_cache=pauli_action_cache,
    )


def _rotation_triplet(vec: np.ndarray, step: Any, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vec_arr = np.asarray(vec, dtype=complex).reshape(-1)
    coeff = float(step.coeff_real)
    pvec = apply_compiled_pauli(vec_arr, step.action)
    phi = float(theta) * coeff
    c = math.cos(phi)
    s = math.sin(phi)
    u_vec = c * vec_arr - 1j * s * pvec
    d_vec = -coeff * s * vec_arr - 1j * coeff * c * pvec
    s_vec = -(coeff * coeff) * u_vec
    return np.asarray(u_vec, dtype=complex), np.asarray(d_vec, dtype=complex), np.asarray(s_vec, dtype=complex)


def _horizontal_tangent(psi_state: np.ndarray, dpsi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    dpsi_vec = np.asarray(dpsi, dtype=complex).reshape(-1)
    overlap = complex(np.vdot(psi, dpsi_vec))
    return np.asarray(dpsi_vec - overlap * psi, dtype=complex)


def _tangent_overlap_matrix(tangents: Sequence[np.ndarray]) -> np.ndarray:
    n = int(len(tangents))
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = float(np.real(np.vdot(tangents[i], tangents[j])))
            out[i, j] = val
            out[j, i] = val
    return out


def _energy_hessian_entry(
    *,
    dpsi_left: np.ndarray,
    dpsi_right: np.ndarray,
    d2psi: np.ndarray,
    hpsi_state: np.ndarray,
    hdpsi_right: np.ndarray,
) -> float:
    return float(
        2.0
        * np.real(
            np.vdot(np.asarray(d2psi, dtype=complex), np.asarray(hpsi_state, dtype=complex))
            + np.vdot(np.asarray(dpsi_left, dtype=complex), np.asarray(hdpsi_right, dtype=complex))
        )
    )


def _propagate_executor_derivatives(
    *,
    executor: CompiledAnsatzExecutor,
    theta: np.ndarray,
    psi_ref: np.ndarray,
    active_indices: Sequence[int],
) -> tuple[np.ndarray, list[np.ndarray], list[list[np.ndarray]]]:
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    active = [int(i) for i in active_indices]
    psi = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
    n_active = int(len(active))
    dpsi = [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
    d2psi = [[np.zeros_like(psi, dtype=complex) for _ in range(n_active)] for __ in range(n_active)]
    if n_active == 0:
        return executor.prepare_state(theta_vec, psi), dpsi, d2psi

    active_map = {int(global_idx): int(local_idx) for local_idx, global_idx in enumerate(active)}
    plans = list(getattr(executor, "_plans", []))
    if len(plans) != int(theta_vec.size):
        raise ValueError(f"theta length mismatch: got {theta_vec.size}, expected {len(plans)}.")

    for global_idx, plan in enumerate(plans):
        theta_k = float(theta_vec[global_idx])
        local = active_map.get(int(global_idx), None)
        for step in getattr(plan, "steps", ()):  # pragma: no branch - tuple in normal path
            old_psi = psi
            old_dpsi = dpsi
            old_d2psi = d2psi

            psi_u, psi_d, psi_s = _rotation_triplet(old_psi, step, theta_k)
            psi = psi_u

            next_dpsi: list[np.ndarray] = []
            d_old: list[np.ndarray] = []
            for idx in range(n_active):
                vec_u, vec_d, _vec_s = _rotation_triplet(old_dpsi[idx], step, theta_k)
                next_dpsi.append(vec_u)
                d_old.append(vec_d)
            if local is not None:
                next_dpsi[int(local)] = np.asarray(next_dpsi[int(local)] + psi_d, dtype=complex)

            next_d2psi: list[list[np.ndarray]] = [
                [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
                for __ in range(n_active)
            ]
            for row in range(n_active):
                for col in range(n_active):
                    vec_u, _vec_d, _vec_s = _rotation_triplet(old_d2psi[row][col], step, theta_k)
                    updated = vec_u
                    if local is not None:
                        if row == int(local):
                            updated = np.asarray(updated + d_old[col], dtype=complex)
                        if col == int(local):
                            updated = np.asarray(updated + d_old[row], dtype=complex)
                        if row == int(local) and col == int(local):
                            updated = np.asarray(updated + psi_s, dtype=complex)
                    next_d2psi[row][col] = np.asarray(updated, dtype=complex)

            dpsi = next_dpsi
            d2psi = next_d2psi

    return np.asarray(psi, dtype=complex), dpsi, d2psi


def _propagate_append_candidate(
    *,
    candidate_term: Any,
    psi_state: np.ndarray,
    window_dpsi: Sequence[np.ndarray],
    pauli_action_cache: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    cand_exec = _executor_for_terms([candidate_term], pauli_action_cache=pauli_action_cache)
    plan = list(getattr(cand_exec, "_plans", []))
    if not plan:
        zero = np.zeros_like(np.asarray(psi_state, dtype=complex).reshape(-1), dtype=complex)
        return zero, zero, [np.zeros_like(zero) for _ in window_dpsi]
    steps = list(getattr(plan[0], "steps", ()))
    psi = np.asarray(psi_state, dtype=complex).reshape(-1).copy()
    cand_dpsi = np.zeros_like(psi, dtype=complex)
    cand_d2psi = np.zeros_like(psi, dtype=complex)
    win_dpsi = [np.asarray(vec, dtype=complex).reshape(-1).copy() for vec in window_dpsi]
    cand_win_d2 = [np.zeros_like(psi, dtype=complex) for _ in window_dpsi]

    for step in steps:
        old_psi = psi
        old_cand_dpsi = cand_dpsi
        old_cand_d2psi = cand_d2psi
        old_win_dpsi = win_dpsi
        old_cand_win_d2 = cand_win_d2

        psi_u, psi_d, psi_s = _rotation_triplet(old_psi, step, 0.0)
        cand_u, cand_d, _cand_s = _rotation_triplet(old_cand_dpsi, step, 0.0)
        cand2_u, _cand2_d, _cand2_s = _rotation_triplet(old_cand_d2psi, step, 0.0)

        psi = psi_u
        cand_dpsi = np.asarray(cand_u + psi_d, dtype=complex)
        cand_d2psi = np.asarray(cand2_u + cand_d + cand_d + psi_s, dtype=complex)

        next_win_dpsi: list[np.ndarray] = []
        next_cand_win_d2: list[np.ndarray] = []
        for idx, win_vec in enumerate(old_win_dpsi):
            win_u, win_d, _win_s = _rotation_triplet(win_vec, step, 0.0)
            cross_u, _cross_d, _cross_s = _rotation_triplet(old_cand_win_d2[idx], step, 0.0)
            next_win_dpsi.append(np.asarray(win_u, dtype=complex))
            next_cand_win_d2.append(np.asarray(cross_u + win_d, dtype=complex))
        win_dpsi = next_win_dpsi
        cand_win_d2 = next_cand_win_d2

    return cand_dpsi, cand_d2psi, cand_win_d2


def _regularized_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    base_ridge: float,
    growth_factor: float,
    max_steps: int,
    require_pd: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    mat = np.asarray(matrix, dtype=float)
    vec = np.asarray(rhs, dtype=float).reshape(-1)
    n = int(mat.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float), float(max(base_ridge, 0.0)), np.zeros((0, 0), dtype=float)
    eye = np.eye(n, dtype=float)
    ridge = float(max(base_ridge, 0.0))
    if ridge == 0.0:
        ridge = 1e-12
    mat_sym = 0.5 * (mat + mat.T)
    for _ in range(int(max(1, max_steps))):
        trial = mat_sym + ridge * eye
        try:
            if require_pd:
                np.linalg.cholesky(trial)
            sol = np.linalg.solve(trial, vec)
            return np.asarray(sol, dtype=float), float(ridge), np.asarray(trial, dtype=float)
        except Exception:
            ridge *= float(max(growth_factor, 2.0))
    trial = mat_sym + ridge * eye
    if require_pd:
        np.linalg.cholesky(trial)
    sol = np.linalg.solve(trial, vec)
    return np.asarray(sol, dtype=float), float(ridge), np.asarray(trial, dtype=float)


class Phase2NoveltyOracle:
    """Exact ordered-state tangent context for append-only reduced-path scoring."""

    def prepare_scaffold_context(
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
        inherited_window = [int(i) for i in refit_window_indices]
        psi_current = np.asarray(psi_state, dtype=complex).reshape(-1)
        hpsi_current = np.asarray(hpsi_state, dtype=complex).reshape(-1)
        if not inherited_window:
            return _ScaffoldDerivativeContext(
                psi_state=psi_current,
                hpsi_state=hpsi_current,
                refit_window_indices=tuple(),
                dpsi_window=tuple(),
                tangents_window=tuple(),
                Q_window=np.zeros((0, 0), dtype=float),
                H_window_hessian=np.zeros((0, 0), dtype=float),
            )

        executor = _executor_for_terms(selected_ops, pauli_action_cache=pauli_action_cache)
        _psi_final, dpsi_window, d2psi_window = _propagate_executor_derivatives(
            executor=executor,
            theta=np.asarray(theta, dtype=float),
            psi_ref=np.asarray(psi_ref, dtype=complex),
            active_indices=inherited_window,
        )
        tangents_window = [
            _horizontal_tangent(psi_current, dpsi_vec)
            for dpsi_vec in dpsi_window
        ]
        q_window = _tangent_overlap_matrix(tangents_window)
        hdpsi_window = [
            apply_compiled_polynomial(np.asarray(dpsi_vec, dtype=complex), h_compiled)
            for dpsi_vec in dpsi_window
        ]
        m = int(len(inherited_window))
        hess = np.zeros((m, m), dtype=float)
        for row in range(m):
            for col in range(m):
                hess[row, col] = _energy_hessian_entry(
                    dpsi_left=dpsi_window[row],
                    dpsi_right=dpsi_window[col],
                    d2psi=d2psi_window[row][col],
                    hpsi_state=hpsi_current,
                    hdpsi_right=hdpsi_window[col],
                )
        hess = 0.5 * (hess + hess.T)
        return _ScaffoldDerivativeContext(
            psi_state=psi_current,
            hpsi_state=hpsi_current,
            refit_window_indices=tuple(inherited_window),
            dpsi_window=tuple(np.asarray(x, dtype=complex) for x in dpsi_window),
            tangents_window=tuple(np.asarray(x, dtype=complex) for x in tangents_window),
            Q_window=np.asarray(q_window, dtype=float),
            H_window_hessian=np.asarray(hess, dtype=float),
        )

    def estimate(
        self,
        *,
        scaffold_context: _ScaffoldDerivativeContext,
        candidate_label: str,
        candidate_term: Any,
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
        novelty_eps: float = 1e-6,
    ) -> Mapping[str, Any]:
        del compiled_cache, novelty_eps  # reduced-path novelty is completed after curvature correction.
        cand_dpsi, cand_d2psi, cand_window_d2 = _propagate_append_candidate(
            candidate_term=candidate_term,
            psi_state=scaffold_context.psi_state,
            window_dpsi=list(scaffold_context.dpsi_window),
            pauli_action_cache=pauli_action_cache,
        )
        cand_tangent = _horizontal_tangent(scaffold_context.psi_state, cand_dpsi)
        q_window = np.asarray(
            [
                float(np.real(np.vdot(tang_j, cand_tangent)))
                for tang_j in scaffold_context.tangents_window
            ],
            dtype=float,
        )
        F_raw = float(max(0.0, np.real(np.vdot(cand_tangent, cand_tangent))))
        return {
            "novelty_mode": "append_exact_tangent_context_v1",
            "candidate_dpsi": np.asarray(cand_dpsi, dtype=complex),
            "candidate_d2psi": np.asarray(cand_d2psi, dtype=complex),
            "candidate_window_d2": [np.asarray(x, dtype=complex) for x in cand_window_d2],
            "candidate_tangent": np.asarray(cand_tangent, dtype=complex),
            "F_raw": float(F_raw),
            "Q_window": np.asarray(scaffold_context.Q_window, dtype=float),
            "q_window": np.asarray(q_window, dtype=float),
        }


class Phase2CurvatureOracle:
    """Exact analytic Hessian blocks for the append-only reduced path."""

    def estimate(
        self,
        *,
        base_feature: CandidateFeatures,
        novelty_info: Mapping[str, Any],
        scaffold_context: _ScaffoldDerivativeContext,
        h_compiled: CompiledPolynomialAction,
        cfg: FullScoreConfig,
        optimizer_memory: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        del optimizer_memory
        F_raw = float(max(0.0, novelty_info.get("F_raw", base_feature.F_raw or base_feature.F_metric)))
        q_window = np.asarray(novelty_info.get("q_window", []), dtype=float).reshape(-1)
        Q_window = np.asarray(novelty_info.get("Q_window", scaffold_context.Q_window), dtype=float)
        cand_dpsi = np.asarray(novelty_info.get("candidate_dpsi"), dtype=complex).reshape(-1)
        cand_d2psi = np.asarray(novelty_info.get("candidate_d2psi"), dtype=complex).reshape(-1)
        cand_window_d2 = [
            np.asarray(x, dtype=complex).reshape(-1)
            for x in novelty_info.get("candidate_window_d2", [])
        ]
        hdpsi_candidate = apply_compiled_polynomial(cand_dpsi, h_compiled)
        h_raw = _energy_hessian_entry(
            dpsi_left=cand_dpsi,
            dpsi_right=cand_dpsi,
            d2psi=cand_d2psi,
            hpsi_state=scaffold_context.hpsi_state,
            hdpsi_right=hdpsi_candidate,
        )

        b_mixed = np.zeros(len(scaffold_context.refit_window_indices), dtype=float)
        for idx, dpsi_window in enumerate(scaffold_context.dpsi_window):
            if idx >= len(cand_window_d2):
                break
            b_mixed[idx] = _energy_hessian_entry(
                dpsi_left=dpsi_window,
                dpsi_right=cand_dpsi,
                d2psi=cand_window_d2[idx],
                hpsi_state=scaffold_context.hpsi_state,
                hdpsi_right=hdpsi_candidate,
            )

        H_window = np.asarray(scaffold_context.H_window_hessian, dtype=float)
        if H_window.size == 0:
            h_eff = float(h_raw)
            F_red = float(max(F_raw, float(cfg.metric_floor)))
            q_reduced = np.zeros(0, dtype=float)
            novelty = 1.0
            ridge_used = float(max(cfg.lambda_H, 0.0))
            M_window = np.zeros((0, 0), dtype=float)
            mode = "append_exact_empty_window"
        else:
            minv_b, ridge_used, M_window = _regularized_solve(
                H_window,
                b_mixed,
                base_ridge=float(max(cfg.lambda_H, 0.0)),
                growth_factor=float(max(cfg.ridge_growth_factor, 2.0)),
                max_steps=int(max(1, cfg.ridge_max_steps)),
                require_pd=True,
            )
            h_eff = float(h_raw - float(b_mixed.T @ minv_b))
            F_red_exact = float(
                F_raw
                - 2.0 * float(q_window.T @ minv_b)
                + float(minv_b.T @ Q_window @ minv_b)
            )
            F_red = float(max(F_red_exact, float(cfg.metric_floor)))
            q_reduced = np.asarray(q_window - Q_window @ minv_b, dtype=float)
            collapse_floor = max(
                float(cfg.metric_floor),
                float(cfg.reduced_metric_collapse_rel_tol) * float(max(F_raw, float(cfg.metric_floor))),
            )
            metric_collapse = bool(F_red_exact <= collapse_floor)
            if metric_collapse:
                novelty = 0.0
                mode = "append_exact_metric_collapse_v1"
            else:
                qsol, _nov_ridge, _Qreg = _regularized_solve(
                    Q_window,
                    q_reduced,
                    base_ridge=float(max(cfg.novelty_eps, 0.0)),
                    growth_factor=float(max(cfg.ridge_growth_factor, 2.0)),
                    max_steps=int(max(1, cfg.ridge_max_steps)),
                    require_pd=True,
                )
                novelty_raw = 1.0 - float(q_reduced.T @ qsol) / float(F_red)
                novelty = float(min(1.0, max(0.0, novelty_raw)))
                mode = (
                    "append_exact_window_hessian_ridge_grown_v1"
                    if float(ridge_used) > float(max(cfg.lambda_H, 0.0))
                    else "append_exact_window_hessian_v1"
                )

        return {
            "h_raw": float(h_raw),
            "b_mixed": [float(x) for x in b_mixed.tolist()],
            "H_window_hessian": [[float(x) for x in row] for row in H_window.tolist()],
            "M_window": [[float(x) for x in row] for row in np.asarray(M_window, dtype=float).tolist()],
            "h_eff": float(h_eff),
            "F_red": float(F_red),
            "q_reduced": [float(x) for x in q_reduced.tolist()],
            "novelty": float(novelty),
            "ridge_used": float(ridge_used),
            "curvature_mode": str(mode),
        }


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_full_candidate_features(
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
    novelty_info = novelty_oracle.estimate(
        scaffold_context=scaffold_context,
        candidate_label=str(base_feature.candidate_label),
        candidate_term=candidate_term,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
        novelty_eps=float(cfg.novelty_eps),
    )
    curvature_info = curvature_oracle.estimate(
        base_feature=base_feature,
        novelty_info=novelty_info,
        scaffold_context=scaffold_context,
        h_compiled=h_compiled,
        cfg=cfg,
        optimizer_memory=optimizer_memory,
    )
    feat = _replace_feature(
        base_feature,
        novelty=float(curvature_info.get("novelty", 1.0)),
        novelty_mode=str(novelty_info.get("novelty_mode", "append_exact_tangent_context_v1")),
        curvature_mode=str(curvature_info.get("curvature_mode", "append_exact_window_hessian_v1")),
        F_metric=float(max(0.0, novelty_info.get("F_raw", base_feature.F_metric))),
        metric_proxy=float(max(0.0, novelty_info.get("F_raw", base_feature.metric_proxy))),
        F_raw=float(max(0.0, novelty_info.get("F_raw", base_feature.F_raw or base_feature.F_metric))),
        Q_window=[[float(x) for x in row] for row in np.asarray(novelty_info.get("Q_window", np.zeros((0, 0), dtype=float)), dtype=float).tolist()],
        q_window=[float(x) for x in np.asarray(novelty_info.get("q_window", []), dtype=float).tolist()],
        h_raw=float(curvature_info.get("h_raw", 0.0)),
        b_mixed=[float(x) for x in curvature_info.get("b_mixed", [])],
        H_window_hessian=[[float(x) for x in row] for row in curvature_info.get("H_window_hessian", [])],
        M_window=[[float(x) for x in row] for row in curvature_info.get("M_window", [])],
        h_eff=float(curvature_info.get("h_eff", 0.0)),
        F_red=float(curvature_info.get("F_red", novelty_info.get("F_raw", 0.0))),
        q_reduced=[float(x) for x in curvature_info.get("q_reduced", [])],
        ridge_used=float(curvature_info.get("ridge_used", max(cfg.lambda_H, 0.0))),
        h_hat=float(curvature_info.get("h_raw", 0.0)),
        b_hat=[float(x) for x in curvature_info.get("b_mixed", [])],
        H_window=[[float(x) for x in row] for row in curvature_info.get("H_window_hessian", [])],
        score_version=str(cfg.score_version),
        placeholder_hooks={
            **dict(base_feature.placeholder_hooks),
            "novelty_oracle": True,
            "curvature_oracle": True,
            "full_v2_score": True,
        },
    )
    if isinstance(base_feature.generator_metadata, Mapping) and isinstance(motif_library, Mapping):
        motif_bonus, motif_meta = motif_bonus_for_generator(
            generator_metadata=base_feature.generator_metadata,
            motif_library=motif_library,
            target_num_sites=int(max(0, target_num_sites or 0)),
        )
        feat = _replace_feature(
            feat,
            motif_bonus=float(motif_bonus),
            motif_source=(
                str(motif_library.get("source_tag", "payload"))
                if bool(motif_bonus) else str(feat.motif_source)
            ),
            motif_metadata=(dict(motif_meta) if isinstance(motif_meta, Mapping) else feat.motif_metadata),
        )
    feat = _replace_feature(
        feat,
        lifetime_weight_components=dict(lifetime_weight_components(feat, cfg)),
        lifetime_cost_mode=str(cfg.lifetime_cost_mode),
        remaining_evaluations_proxy_mode=str(cfg.remaining_evaluations_proxy_mode),
    )
    score, fallback_mode = full_v2_score(feat, cfg)
    return _replace_feature(
        feat,
        full_v2_score=float(score),
        actual_fallback_mode=str(fallback_mode),
    )


def build_candidate_features(
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
    """Built-in math expression:
    g_lcb = max(|g| - z_alpha * sigma_hat, 0)
    """
    g_abs = float(abs(float(gradient_signed)))
    g_lcb = max(g_abs - float(cfg.z_alpha) * float(max(0.0, sigma_hat)), 0.0)
    raw_metric = float(max(0.0, metric_proxy))
    remaining_eval_proxy = remaining_evaluations_proxy(
        current_depth=current_depth,
        max_depth=max_depth,
        mode=str(remaining_evaluations_proxy_mode),
    )
    feat = CandidateFeatures(
        stage_name=str(stage_name),
        candidate_label=str(candidate_label),
        candidate_family=str(candidate_family),
        candidate_pool_index=int(candidate_pool_index),
        position_id=int(position_id),
        append_position=int(append_position),
        positions_considered=[int(x) for x in positions_considered],
        g_signed=float(gradient_signed),
        g_abs=float(g_abs),
        g_lcb=float(g_lcb),
        sigma_hat=float(max(0.0, sigma_hat)),
        F_metric=float(raw_metric),
        metric_proxy=float(raw_metric),
        novelty=None,
        curvature_mode="append_screen_only",
        novelty_mode="none",
        refit_window_indices=[int(i) for i in refit_window_indices],
        compiled_position_cost_proxy={
            "new_pauli_actions": float(compile_cost.new_pauli_actions),
            "new_rotation_steps": float(compile_cost.new_rotation_steps),
            "position_shift_span": float(compile_cost.position_shift_span),
            "refit_active_count": float(compile_cost.refit_active_count),
            "proxy_total": float(compile_cost.proxy_total),
        },
        measurement_cache_stats={
            "groups_total": float(measurement_stats.groups_total),
            "groups_reused": float(measurement_stats.groups_reused),
            "groups_new": float(measurement_stats.groups_new),
            "shots_reused": float(measurement_stats.shots_reused),
            "shots_new": float(measurement_stats.shots_new),
            "reuse_count_cost": float(measurement_stats.reuse_count_cost),
        },
        leakage_penalty=float(max(0.0, leakage_penalty)),
        stage_gate_open=bool(stage_gate_open),
        leakage_gate_open=bool(leakage_gate_open),
        trough_probe_triggered=bool(trough_probe_triggered),
        trough_detected=bool(trough_detected),
        simple_score=None,
        score_version=str(cfg.score_version),
        F_raw=float(raw_metric),
        depth_cost=float(compile_cost.new_pauli_actions + compile_cost.new_rotation_steps),
        new_group_cost=float(measurement_stats.groups_new),
        new_shot_cost=float(measurement_stats.shots_new),
        opt_dim_cost=float(len(refit_window_indices)),
        reuse_count_cost=float(measurement_stats.reuse_count_cost),
        family_repeat_cost=float(max(0.0, family_repeat_cost)),
        generator_id=(
            str(generator_metadata.get("generator_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("generator_id") is not None
            else None
        ),
        template_id=(
            str(generator_metadata.get("template_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("template_id") is not None
            else None
        ),
        is_macro_generator=bool(generator_metadata.get("is_macro_generator", False)) if isinstance(generator_metadata, Mapping) else False,
        parent_generator_id=(
            str(generator_metadata.get("parent_generator_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("parent_generator_id") is not None
            else None
        ),
        generator_metadata=(dict(generator_metadata) if isinstance(generator_metadata, Mapping) else None),
        symmetry_spec=(dict(symmetry_spec) if isinstance(symmetry_spec, Mapping) else None),
        symmetry_mode=str(symmetry_mode),
        symmetry_mitigation_mode=str(symmetry_mitigation_mode),
        motif_metadata=(dict(motif_metadata) if isinstance(motif_metadata, Mapping) else None),
        motif_bonus=float(max(0.0, motif_bonus)),
        motif_source=str(motif_source),
        remaining_evaluations_proxy=float(remaining_eval_proxy),
        remaining_evaluations_proxy_mode=str(remaining_evaluations_proxy_mode),
        lifetime_cost_mode=str(lifetime_cost_mode),
        lifetime_weight_components={
            "remaining_evaluations_proxy": float(remaining_eval_proxy),
        },
        placeholder_hooks={
            "novelty_oracle": False,
            "curvature_oracle": False,
            "full_v2_score": False,
            "qn_spsa_refresh": False,
            "motif_metadata": False,
            "symmetry_metadata": bool(isinstance(symmetry_spec, Mapping)),
        },
    )
    score = simple_v1_score(feat, cfg)
    return _replace_feature(feat, simple_score=float(score))


# ---------------------------------------------------------------------------
# Legacy compatibility helpers (inactive in active append-only path)
# ---------------------------------------------------------------------------

def _pauli_labels_from_term(term: Any) -> list[str]:
    labels: list[str] = []
    if term is None or not hasattr(term, "polynomial"):
        return labels
    for poly_term in term.polynomial.return_polynomial():
        labels.append(str(poly_term.pw2strng()))
    return labels


def _support_set(term: Any) -> set[int]:
    support: set[int] = set()
    labels = _pauli_labels_from_term(term)
    for label in labels:
        for idx, ch in enumerate(str(label)):
            if ch != "e":
                support.add(int(idx))
    return support


def _pauli_strings_commute(lhs: str, rhs: str) -> bool:
    anticomm = 0
    for a, b in zip(str(lhs), str(rhs)):
        if a == "e" or b == "e" or a == b:
            continue
        anticomm += 1
    return bool((anticomm % 2) == 0)


def _polynomials_commute(term_a: Any, term_b: Any) -> bool:
    labels_a = _pauli_labels_from_term(term_a)
    labels_b = _pauli_labels_from_term(term_b)
    if not labels_a or not labels_b:
        return True
    for lhs in labels_a:
        for rhs in labels_b:
            if not _pauli_strings_commute(lhs, rhs):
                return False
    return True


def compatibility_penalty(
    *,
    record_a: Mapping[str, Any],
    record_b: Mapping[str, Any],
    cfg: FullScoreConfig,
    psi_state: np.ndarray | None = None,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> dict[str, float]:
    del compiled_cache, pauli_action_cache
    feat_a = record_a.get("feature")
    feat_b = record_b.get("feature")
    term_a = record_a.get("candidate_term")
    term_b = record_b.get("candidate_term")
    if not isinstance(feat_a, CandidateFeatures) or not isinstance(feat_b, CandidateFeatures):
        return {"support_overlap": 0.0, "noncommutation": 0.0, "cross_curvature": 0.0, "schedule": 0.0, "total": 0.0}

    supp_a = _support_set(term_a)
    supp_b = _support_set(term_b)
    union = len(supp_a | supp_b)
    support_overlap = 0.0 if union == 0 else float(len(supp_a & supp_b) / union)
    noncomm = 0.0 if _polynomials_commute(term_a, term_b) else 1.0

    cross_curv = 0.0
    vec_a = np.asarray(
        feat_a.q_reduced if feat_a.q_reduced is not None else (feat_a.b_mixed if feat_a.b_mixed is not None else []),
        dtype=float,
    )
    vec_b = np.asarray(
        feat_b.q_reduced if feat_b.q_reduced is not None else (feat_b.b_mixed if feat_b.b_mixed is not None else []),
        dtype=float,
    )
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom > 0.0:
        cross_curv = float(min(1.0, abs(float(vec_a @ vec_b)) / denom))
    elif psi_state is not None:
        cross_curv = float(support_overlap)

    win_a = set(int(i) for i in feat_a.refit_window_indices)
    win_b = set(int(i) for i in feat_b.refit_window_indices)
    union_w = len(win_a | win_b)
    schedule = 0.0 if union_w == 0 else float(len(win_a & win_b) / union_w)
    total = (
        float(cfg.compat_overlap_weight) * float(support_overlap)
        + float(cfg.compat_comm_weight) * float(noncomm)
        + float(cfg.compat_curv_weight) * float(cross_curv)
        + float(cfg.compat_sched_weight) * float(schedule)
    )
    return {
        "support_overlap": float(support_overlap),
        "noncommutation": float(noncomm),
        "cross_curvature": float(cross_curv),
        "schedule": float(schedule),
        "total": float(total),
    }


class CompatibilityPenaltyOracle:
    def __init__(
        self,
        *,
        cfg: FullScoreConfig,
        psi_state: np.ndarray | None = None,
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.psi_state = None if psi_state is None else np.asarray(psi_state, dtype=complex)
        self.compiled_cache = compiled_cache
        self.pauli_action_cache = pauli_action_cache

    def penalty(self, record_a: Mapping[str, Any], record_b: Mapping[str, Any]) -> dict[str, float]:
        return compatibility_penalty(
            record_a=record_a,
            record_b=record_b,
            cfg=self.cfg,
            psi_state=self.psi_state,
            compiled_cache=self.compiled_cache,
            pauli_action_cache=self.pauli_action_cache,
        )


def greedy_batch_select(
    ranked_records: Sequence[Mapping[str, Any]],
    compat_oracle: CompatibilityPenaltyOracle,
    cfg: FullScoreConfig,
) -> tuple[list[dict[str, Any]], float]:
    del compat_oracle, cfg
    ranked = sorted(
        [dict(rec) for rec in ranked_records],
        key=lambda rec: (
            -float(rec.get("full_v2_score", float("-inf"))),
            -float(rec.get("simple_score", float("-inf"))),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return [], 0.0
    return [dict(ranked[0])], 0.0

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/time_propagation/cfqm_propagator.py
```py
"""CFQM macro-step propagation utilities.

This module implements one CFQM macro-step for statevector propagation:

    psi(t + dt) = exp(-i dt * Omega_1) ... exp(-i dt * Omega_s) psi(t)

with stage operators Omega_k assembled from static and sampled drive
coefficient maps.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class CFQMConfig:
    """Configuration for CFQM stage propagation backends."""

    backend: str = "expm_multiply_sparse"
    coeff_drop_abs_tol: float = 0.0
    normalize: bool = False
    sparse_min_dim: int = 64
    norm_drift_logger: Callable[..., None] | None = None
    emit_inner_order_warning: bool = True


def _cfg_get(config: object, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _build_dense_stage_matrix_via_repo_utility(
    stage_coeff_map: Mapping[str, complex | float],
    ordered_labels: list[str],
) -> np.ndarray:
    """Build dense stage matrix using existing repo Pauli-sum utility."""
    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.qubitization_module import PauliTerm
    from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

    if not ordered_labels:
        raise ValueError("ordered_labels must be non-empty.")
    nq = len(ordered_labels[0])
    for label in ordered_labels:
        if len(label) != nq:
            raise ValueError("All ordered_labels must have equal length.")
    ordered_set = set(ordered_labels)
    for label in stage_coeff_map:
        if label not in ordered_set:
            raise ValueError(f"Stage label {label!r} absent from ordered_labels.")

    pol = PauliPolynomial("JW")
    any_nonzero = False
    for label in ordered_labels:
        coeff = stage_coeff_map.get(label)
        if coeff is None:
            continue
        coeff_c = complex(coeff)
        if coeff_c == 0.0:
            continue
        pol.add_term(PauliTerm(nq, ps=label, pc=coeff_c))
        any_nonzero = True

    if not any_nonzero:
        dim = 1 << nq
        return np.zeros((dim, dim), dtype=complex)
    return hamiltonian_matrix(pol, tol=0.0)


def _build_sparse_stage_matrix_via_compiled_actions(
    stage_coeff_map: Mapping[str, complex | float],
    ordered_labels: list[str],
):
    """Build sparse stage matrix without dense materialization.

    Reuses the existing compiled Pauli-action utility from the hardcoded
    pipeline to preserve Pauli conventions exactly.
    """
    from scipy.sparse import csc_matrix, coo_matrix
    from src.quantum.pauli_actions import compile_pauli_action_exyz

    if not ordered_labels:
        raise ValueError("ordered_labels must be non-empty.")
    nq = len(ordered_labels[0])
    for label in ordered_labels:
        if len(label) != nq:
            raise ValueError("All ordered_labels must have equal length.")
    ordered_set = set(ordered_labels)
    for label in stage_coeff_map:
        if label not in ordered_set:
            raise ValueError(f"Stage label {label!r} absent from ordered_labels.")

    dim = 1 << nq
    row_base = np.arange(dim, dtype=np.int64)
    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []

    for label in ordered_labels:
        coeff = stage_coeff_map.get(label)
        if coeff is None:
            continue
        coeff_c = complex(coeff)
        if coeff_c == 0.0:
            continue
        action = compile_pauli_action_exyz(label, nq)
        row_chunks.append(row_base)
        col_chunks.append(np.asarray(action.perm, dtype=np.int64))
        data_chunks.append(coeff_c * np.asarray(action.phase, dtype=complex))

    if not data_chunks:
        return csc_matrix((dim, dim), dtype=complex)

    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    data = np.concatenate(data_chunks)
    h_csc = coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsc()
    h_csc.sum_duplicates()
    h_csc.eliminate_zeros()
    return h_csc


def _apply_stage_pauli_suzuki2(
    psi: np.ndarray,
    stage_coeff_map: Mapping[str, complex | float],
    dt: float,
    ordered_labels: list[str],
) -> np.ndarray:
    """Apply symmetric Suzuki-2 over Pauli terms in deterministic order."""
    from src.quantum.pauli_actions import apply_exp_term, compile_pauli_action_exyz

    nq = len(ordered_labels[0])
    compiled = {label: compile_pauli_action_exyz(label, nq) for label in ordered_labels}
    half = 0.5 * float(dt)
    out = np.array(psi, copy=True)

    for label in ordered_labels:
        alpha = complex(stage_coeff_map.get(label, 0.0 + 0.0j))
        if alpha == 0.0:
            continue
        out = apply_exp_term(out, compiled[label], alpha, half)
    for label in reversed(ordered_labels):
        alpha = complex(stage_coeff_map.get(label, 0.0 + 0.0j))
        if alpha == 0.0:
            continue
        out = apply_exp_term(out, compiled[label], alpha, half)
    return out


def apply_stage_exponential(
    psi: np.ndarray,
    stage_coeff_map: dict[str, complex | float],
    dt: float,
    ordered_labels: list[str],
    backend_config: object,
) -> np.ndarray:
    """Apply exp(-i * dt * H_stage) to a statevector."""
    if abs(float(dt)) <= 1e-15 or not stage_coeff_map:
        return np.array(psi, copy=True)

    backend_key = str(_cfg_get(backend_config, "backend", "expm_multiply_sparse")).strip().lower()
    scheme_name = str(_cfg_get(backend_config, "scheme_name", "")).strip().lower()

    if backend_key in {"dense", "dense_expm"}:
        from scipy.linalg import expm

        h_stage = _build_dense_stage_matrix_via_repo_utility(stage_coeff_map, ordered_labels)
        return np.asarray(expm((-1j * float(dt)) * h_stage) @ psi, dtype=complex)

    if backend_key in {"expm_multiply_sparse", "sparse_expm_multiply", "sparse"}:
        from scipy.sparse.linalg import expm_multiply

        h_stage = _build_sparse_stage_matrix_via_compiled_actions(stage_coeff_map, ordered_labels)
        return np.asarray(expm_multiply((-1j * float(dt)) * h_stage, psi), dtype=complex)

    if backend_key == "pauli_suzuki2":
        emit_warning = bool(_cfg_get(backend_config, "emit_inner_order_warning", True))
        if emit_warning and (scheme_name in {"cf4:2", "cf6:5opt"} or scheme_name.startswith("cfqm")):
            warnings.warn(
                "Inner Suzuki-2 makes overall method 2nd order; use expm_multiply_sparse/dense_expm for true CFQM order.",
                RuntimeWarning,
                stacklevel=2,
            )
        return np.asarray(
            _apply_stage_pauli_suzuki2(
                psi=psi,
                stage_coeff_map=stage_coeff_map,
                dt=float(dt),
                ordered_labels=ordered_labels,
            ),
            dtype=complex,
        )

    if backend_key == "auto":
        from scipy.sparse.linalg import expm_multiply

        sparse_min_dim = int(_cfg_get(backend_config, "sparse_min_dim", 64))
        if int(psi.size) < sparse_min_dim:
            from scipy.linalg import expm

            h_stage_dense = _build_dense_stage_matrix_via_repo_utility(stage_coeff_map, ordered_labels)
            return np.asarray(expm((-1j * float(dt)) * h_stage_dense) @ psi, dtype=complex)
        h_sparse = _build_sparse_stage_matrix_via_compiled_actions(stage_coeff_map, ordered_labels)
        return np.asarray(expm_multiply((-1j * float(dt)) * h_sparse, psi), dtype=complex)

    raise ValueError(
        f"Unsupported backend {backend_key!r}. "
        "Use one of {'dense_expm','expm_multiply_sparse','pauli_suzuki2','auto'}."
    )


def cfqm_step(
    psi: np.ndarray,
    t_abs: float,
    dt: float,
    static_coeff_map: dict[str, complex | float],
    drive_coeff_provider: Callable[[float], Mapping[str, complex | float]] | None,
    ordered_labels: list[str],
    scheme: dict,
    config: object,
) -> np.ndarray:
    """Advance one CFQM macro-step.

    Drive-map labels not present in ``ordered_labels`` are ignored so that
    stage assembly remains deterministic and never introduces out-of-order
    terms.
    """
    if not ordered_labels:
        raise ValueError("ordered_labels must be non-empty.")
    dt_f = float(dt)
    if not np.isfinite(dt_f) or dt_f <= 0.0:
        raise ValueError(f"cfqm_step requires dt > 0 and finite; got dt={dt_f}.")

    dim = int(psi.size)
    nq = int(round(math.log2(dim)))
    if (1 << nq) != dim:
        raise ValueError("psi size must be a power of two.")
    for label in ordered_labels:
        if len(label) != nq:
            raise ValueError("All labels must match statevector qubit count.")

    c_nodes = [float(x) for x in scheme["c"]]
    a_rows = [[float(v) for v in row] for row in scheme["a"]]
    s_static = [float(v) for v in scheme["s_static"]]

    m_nodes = len(c_nodes)
    s_stages = len(a_rows)
    if len(s_static) != s_stages:
        raise ValueError("scheme.s_static length must match number of stage rows.")
    for k, row in enumerate(a_rows):
        if len(row) != m_nodes:
            raise ValueError(f"scheme.a row {k} length mismatch: expected {m_nodes}.")

    ordered_set = set(ordered_labels)
    static_map = {str(lbl): complex(coeff) for lbl, coeff in static_coeff_map.items()}
    for label in static_map:
        if label not in ordered_set:
            raise ValueError(f"Static label {label!r} is absent from ordered_labels.")

    unknown_label_policy = str(_cfg_get(config, "unknown_label_policy", "warn_ignore")).strip().lower()
    if unknown_label_policy not in {"warn_ignore", "ignore", "strict"}:
        raise ValueError(
            "unknown_label_policy must be one of {'warn_ignore','ignore','strict'}."
        )
    unknown_label_warn_abs_tol = max(
        0.0,
        float(_cfg_get(config, "unknown_label_warn_abs_tol", 1e-14)),
    )
    warned_labels_obj = _cfg_get(config, "unknown_label_warned_labels", None)
    if warned_labels_obj is None:
        warned_labels: set[str] = set()
    else:
        if not hasattr(warned_labels_obj, "__contains__") or not hasattr(warned_labels_obj, "add"):
            raise ValueError("unknown_label_warned_labels must support membership and add().")
        warned_labels = warned_labels_obj

    drive_maps: list[dict[str, complex]] = []
    for c_j in c_nodes:
        t_node = float(t_abs) + float(c_j) * dt_f
        if drive_coeff_provider is None:
            raw_map: Mapping[str, complex | float] = {}
        else:
            raw = drive_coeff_provider(float(t_node))
            raw_map = {} if raw is None else raw

        node_map: dict[str, complex] = {}
        for label, coeff in raw_map.items():
            lbl = str(label)
            coeff_c = complex(coeff)
            if not np.isfinite(coeff_c.real) or not np.isfinite(coeff_c.imag):
                raise ValueError(
                    "Non-finite drive coefficient detected: "
                    f"label={lbl!r}, time={float(t_node):.16g}, coeff={coeff_c!r}."
                )
            if lbl not in ordered_set:
                # Do not introduce labels outside deterministic ordered_labels.
                mag = abs(coeff_c)
                if mag <= unknown_label_warn_abs_tol:
                    continue
                if unknown_label_policy == "strict":
                    raise ValueError(
                        "Unknown drive label absent from ordered_labels: "
                        f"label={lbl!r}, time={float(t_node):.16g}, coeff={coeff_c!r}."
                    )
                if unknown_label_policy == "warn_ignore" and lbl not in warned_labels:
                    warnings.warn(
                        "Ignoring unknown drive label absent from ordered_labels: "
                        f"label={lbl!r}, time={float(t_node):.16g}, |coeff|={mag:.3e}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned_labels.add(lbl)
                continue
            node_map[lbl] = coeff_c
        drive_maps.append(node_map)

    coeff_drop_tol = float(_cfg_get(config, "coeff_drop_abs_tol", 0.0))
    stage_maps: list[dict[str, complex]] = []

    for k in range(s_stages):
        stage_map: dict[str, complex] = {}

        w_static = float(s_static[k])
        for label in ordered_labels:
            coeff0 = static_map.get(label)
            if coeff0 is None:
                continue
            scaled = complex(w_static) * coeff0
            if scaled != 0.0:
                stage_map[label] = scaled

        for j in range(m_nodes):
            w = float(a_rows[k][j])
            if w == 0.0:
                continue
            for label, coeff_drive in drive_maps[j].items():
                incr = complex(w) * complex(coeff_drive)
                # A=0 invariance guard:
                # if increment is exactly zero, do not insert new labels.
                if incr == 0.0 and label not in stage_map:
                    continue
                stage_map[label] = stage_map.get(label, 0.0 + 0.0j) + incr

        if coeff_drop_tol > 0.0:
            for label in list(stage_map):
                if abs(stage_map[label]) < coeff_drop_tol:
                    del stage_map[label]

        stage_maps.append(stage_map)

    backend_cfg = {
        "backend": str(_cfg_get(config, "backend", "expm_multiply_sparse")),
        "sparse_min_dim": int(_cfg_get(config, "sparse_min_dim", 64)),
        "scheme_name": str(scheme.get("name", "")),
        "emit_inner_order_warning": bool(_cfg_get(config, "emit_inner_order_warning", True)),
    }

    psi_next = np.asarray(psi, dtype=complex)
    # Rightmost exponential acts first on statevectors => descending stage index.
    for k in range(s_stages - 1, -1, -1):
        psi_next = apply_stage_exponential(
            psi=psi_next,
            stage_coeff_map=stage_maps[k],
            dt=dt_f,
            ordered_labels=ordered_labels,
            backend_config=backend_cfg,
        )

    if bool(_cfg_get(config, "normalize", False)):
        norm_before = float(np.linalg.norm(psi_next))
        if norm_before <= 0.0:
            raise ValueError("Encountered zero-norm state in cfqm_step.")
        norm_drift = abs(norm_before - 1.0)
        logger = _cfg_get(config, "norm_drift_logger", None)
        if callable(logger):
            logger(
                event="cfqm_norm_drift",
                t_abs=float(t_abs),
                dt=dt_f,
                norm_before=norm_before,
                norm_drift=norm_drift,
            )
        psi_next = psi_next / norm_before

    return np.asarray(psi_next, dtype=complex)

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_rescue.py
```py
#!/usr/bin/env python3
"""Simulator-only overlap-guided rescue helpers for HH continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class RescueConfig:
    enabled: bool = False
    simulator_only: bool = True
    recent_drop_patience: int = 2
    weak_drop_threshold: float = 1e-6
    shortlist_flat_ratio: float = 0.95
    max_candidates: int = 6
    min_overlap_gain: float = 1e-7


def should_trigger_rescue(
    *,
    enabled: bool,
    exact_state_available: bool,
    residual_opened: bool,
    trough_detected: bool,
    history: Sequence[Mapping[str, Any]],
    shortlist_records: Sequence[Mapping[str, Any]],
    cfg: RescueConfig,
) -> tuple[bool, str]:
    if not bool(enabled):
        return False, "disabled"
    if not bool(exact_state_available):
        return False, "exact_state_unavailable"
    if not (bool(residual_opened) or bool(trough_detected)):
        return False, "residual_not_open_or_no_trough"
    need = int(max(1, cfg.recent_drop_patience))
    recent = [row for row in history if isinstance(row, Mapping)][-need:]
    if len(recent) < need:
        return False, "insufficient_history"
    if any(float(row.get("delta_abs_drop_from_prev", 1.0)) > float(cfg.weak_drop_threshold) for row in recent):
        return False, "drop_not_flat"
    if len(shortlist_records) < 2:
        return False, "shortlist_too_small"
    top = float(shortlist_records[0].get("full_v2_score", shortlist_records[0].get("simple_score", 0.0)))
    second = float(shortlist_records[1].get("full_v2_score", shortlist_records[1].get("simple_score", 0.0)))
    if top <= 0.0:
        return False, "nonpositive_shortlist"
    if second < float(cfg.shortlist_flat_ratio) * top:
        return False, "shortlist_not_flat"
    return True, "flat_drop_and_shortlist"


def rank_rescue_candidates(
    *,
    records: Sequence[Mapping[str, Any]],
    overlap_gain_fn: Callable[[Mapping[str, Any]], float],
    cfg: RescueConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for rec in list(records)[: int(max(1, cfg.max_candidates))]:
        gain = float(overlap_gain_fn(rec))
        ranked.append(
            {
                **dict(rec),
                "overlap_gain": float(gain),
            }
        )
    ranked = sorted(
        ranked,
        key=lambda rec: (
            -float(rec.get("overlap_gain", 0.0)),
            -float(rec.get("full_v2_score", rec.get("simple_score", float("-inf")))),
            -float(rec.get("simple_score", float("-inf"))),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return None, {"executed": False, "reason": "no_candidates", "ranked": []}
    best = dict(ranked[0])
    if float(best.get("overlap_gain", 0.0)) <= float(cfg.min_overlap_gain):
        return None, {
            "executed": True,
            "reason": "insufficient_overlap_gain",
            "ranked": [dict(x) for x in ranked],
        }
    return best, {
        "executed": True,
        "reason": "selected",
        "ranked": [dict(x) for x in ranked],
    }

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_projected_real_time.py
```py
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.time_propagation.projected_real_time import (
    ProjectedRealTimeConfig,
    build_tangent_vectors,
    run_exact_driven_reference,
    run_projected_real_time_trajectory,
    solve_mclachlan_step,
    state_fidelity,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _basis(dim: int, idx: int) -> np.ndarray:
    out = np.zeros(dim, dtype=complex)
    out[int(idx)] = 1.0
    return out


def _term(label: str, *, nq: int = 1, coeff: complex = 1.0) -> AnsatzTerm:
    return AnsatzTerm(
        label=f"term_{label}",
        polynomial=PauliPolynomial("JW", [PauliTerm(int(nq), ps=str(label), pc=complex(coeff))]),
    )


def test_build_tangent_vectors_matches_single_pauli_derivative() -> None:
    psi_ref = _basis(2, 0)
    psi, tangents = build_tangent_vectors(
        psi_ref,
        [_term("x")],
        np.array([0.0], dtype=float),
        tangent_eps=1e-7,
    )

    assert np.allclose(psi, psi_ref)
    assert len(tangents) == 1
    assert np.allclose(np.asarray(tangents[0]), np.array([0.0, -1.0j], dtype=complex), atol=1e-5)


def test_solve_mclachlan_step_solves_regularized_system() -> None:
    tangent = np.array([0.0, 1.0], dtype=complex)
    hpsi = np.array([0.0, 1.0j], dtype=complex)
    theta_dot, diag = solve_mclachlan_step([tangent], hpsi, lambda_reg=1e-8, svd_rcond=1e-12)

    assert theta_dot.shape == (1,)
    assert abs(float(theta_dot[0]) - 0.99999999) < 1e-6
    assert diag["regularization_used"] is True
    assert diag["solve_mode"] == "solve"


def test_run_projected_real_time_trajectory_zero_hamiltonian_keeps_state() -> None:
    psi_ref = _basis(2, 0)
    result = run_projected_real_time_trajectory(
        psi_ref,
        [_term("x")],
        np.zeros((2, 2), dtype=complex),
        config=ProjectedRealTimeConfig(t_final=0.5, num_times=5, ode_substeps=2),
    )

    assert result.theta_history.shape == (5, 1)
    assert np.allclose(result.theta_history, 0.0)
    for state in result.states:
        assert np.allclose(state, psi_ref)
    assert all(abs(float(row["state_norm"]) - 1.0) < 1e-10 for row in result.trajectory_rows)


def test_run_exact_driven_reference_static_matches_analytic_state() -> None:
    psi_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    hmat = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    result = run_exact_driven_reference(
        psi_plus,
        hmat,
        t_final=0.5,
        num_times=3,
        reference_steps=8,
        drive_coeff_provider_exyz=None,
    )

    expected = np.array([np.exp(-0.5j), np.exp(0.5j)], dtype=complex) / np.sqrt(2.0)
    assert result.times.shape == (3,)
    assert state_fidelity(result.states[-1], expected) > 1.0 - 1e-12
    assert all(abs(float(row["state_norm"]) - 1.0) < 1e-10 for row in result.trajectory_rows)

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_hh_continuation_generators.py
```py
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_generators import (
    build_generator_metadata,
    build_pool_generator_registry,
    build_runtime_split_children,
    build_split_event,
)
from pipelines.hardcoded.hh_continuation_symmetry import build_symmetry_spec
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm


def _term(label: str, poly: PauliPolynomial):
    return type("_DummyAnsatzTerm", (), {"label": str(label), "polynomial": poly})()


def _macro_poly() -> PauliPolynomial:
    return PauliPolynomial(
        "JW",
        [
            PauliTerm(6, ps="eyeexy", pc=1.0),
            PauliTerm(6, ps="eyeeyx", pc=-1.0),
        ],
    )


def test_build_generator_metadata_is_stable_for_same_structure() -> None:
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    first = build_generator_metadata(
        label="cand",
        polynomial=_macro_poly(),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    second = build_generator_metadata(
        label="cand",
        polynomial=_macro_poly(),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    assert first.generator_id == second.generator_id
    assert first.template_id == second.template_id
    assert first.support_site_offsets == [0, 1]
    assert first.is_macro_generator is True


def test_pool_registry_carries_symmetry_metadata() -> None:
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    registry = build_pool_generator_registry(
        terms=[_term("macro", _macro_poly())],
        family_ids=["paop_lf_std"],
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_specs=[sym.__dict__],
    )
    meta = registry["macro"]
    assert meta["family_id"] == "paop_lf_std"
    assert meta["is_macro_generator"] is True
    assert meta["symmetry_spec"]["mitigation_eligible"] is True


def test_deliberate_split_marks_child_metadata() -> None:
    meta = build_generator_metadata(
        label="child",
        polynomial=_macro_poly(),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_policy="deliberate_split",
        parent_generator_id="gen:parent",
    )
    assert meta.is_macro_generator is False
    assert meta.parent_generator_id == "gen:parent"
    assert meta.split_policy == "deliberate_split"


def test_build_split_event_keeps_parent_child_provenance() -> None:
    event = build_split_event(
        parent_generator_id="gen:parent",
        child_generator_ids=["gen:c1", "gen:c2"],
        reason="compiled_depth_cap",
        split_mode="selective",
    )
    assert event["parent_generator_id"] == "gen:parent"
    assert event["child_generator_ids"] == ["gen:c1", "gen:c2"]
    assert event["reason"] == "compiled_depth_cap"


def test_build_runtime_split_children_emits_serialized_single_term_children() -> None:
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    parent_meta = build_generator_metadata(
        label="macro",
        polynomial=_macro_poly(),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    children = build_runtime_split_children(
        parent_label="macro",
        polynomial=_macro_poly(),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=sym.__dict__,
    )
    assert len(children) == 2
    assert children[0]["child_label"].startswith("macro::split[0]::")
    assert children[1]["child_label"].startswith("macro::split[1]::")
    for idx, child in enumerate(children):
        meta = child["child_generator_metadata"]
        compile_meta = meta["compile_metadata"]
        assert meta["parent_generator_id"] == parent_meta.generator_id
        assert meta["split_policy"] == "deliberate_split"
        assert meta["is_macro_generator"] is False
        assert compile_meta["runtime_split"]["mode"] == "shortlist_pauli_children_v1"
        assert compile_meta["runtime_split"]["parent_label"] == "macro"
        assert compile_meta["runtime_split"]["child_index"] == idx
        assert compile_meta["runtime_split"]["child_count"] == 2
        assert len(compile_meta["serialized_terms_exyz"]) == 1

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/compiled_ansatz.py
```py
"""Compiled ansatz execution helpers (exyz convention).

This module provides a wrapper executor that prepares ansatz states from
``AnsatzTerm``-like inputs using shared compiled Pauli actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_exp_term,
    compile_pauli_action_exyz,
)

if TYPE_CHECKING:  # pragma: no cover
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm


@dataclass(frozen=True)
class CompiledRotationStep:
    coeff_real: float
    action: CompiledPauliAction


@dataclass(frozen=True)
class CompiledPolynomialRotationPlan:
    nq: int | None
    steps: tuple[CompiledRotationStep, ...]


class CompiledAnsatzExecutor:
    """Compiled-action ansatz executor for lists of AnsatzTerm-like objects."""

    _MATH_INIT = (
        r"\exp(-i\theta_k H_k)\approx\prod_j \exp(-i\theta_k c_{k,j} P_{k,j}),"
        r"\quad P_{k,j}\mapsto (\mathrm{perm},\mathrm{phase})"
    )

    def __init__(
        self,
        terms: Sequence["AnsatzTerm"],
        *,
        coefficient_tolerance: float = 1e-12,
        ignore_identity: bool = True,
        sort_terms: bool = True,
        pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
    ):
        self.coefficient_tolerance = float(coefficient_tolerance)
        self.ignore_identity = bool(ignore_identity)
        self.sort_terms = bool(sort_terms)
        self.terms = list(terms)

        self.pauli_action_cache = (
            pauli_action_cache if pauli_action_cache is not None else {}
        )
        self._plans: list[CompiledPolynomialRotationPlan] = []
        self.nq: int | None = None

        for term in self.terms:
            plan = self._compile_polynomial_plan(getattr(term, "polynomial"))
            if plan.nq is not None:
                if self.nq is None:
                    self.nq = int(plan.nq)
                elif int(plan.nq) != int(self.nq):
                    raise ValueError(
                        f"Inconsistent ansatz qubit counts: saw {plan.nq} after {self.nq}."
                    )
            self._plans.append(plan)

    _MATH_COMPILE_POLYNOMIAL_PLAN = (
        r"H=\sum_j c_j P_j,\ \text{ordered as in apply\_exp\_pauli\_polynomial},"
        r"\ \text{skip }|c_j|<\epsilon,\ \text{optional skip }I"
    )

    def _compile_polynomial_plan(self, poly: Any) -> CompiledPolynomialRotationPlan:
        poly_terms = list(poly.return_polynomial())
        if not poly_terms:
            return CompiledPolynomialRotationPlan(nq=None, steps=tuple())

        nq = int(poly_terms[0].nqubit())
        id_label = "e" * nq
        ordered = list(poly_terms)
        if self.sort_terms:
            ordered.sort(key=lambda t: t.pw2strng())

        compiled_steps: list[CompiledRotationStep] = []
        for pauli_term in ordered:
            label = str(pauli_term.pw2strng())
            coeff = complex(pauli_term.p_coeff)

            if int(pauli_term.nqubit()) != nq:
                raise ValueError(
                    f"Inconsistent polynomial term qubit count: expected {nq}, got {pauli_term.nqubit()}."
                )
            if len(label) != nq:
                raise ValueError(f"Invalid Pauli label length for '{label}': expected {nq}.")
            if abs(coeff) < self.coefficient_tolerance:
                continue
            if self.ignore_identity and label == id_label:
                continue
            if abs(coeff.imag) > self.coefficient_tolerance:
                raise ValueError(f"Non-negligible imaginary coefficient in term {label}: {coeff}.")

            action = self.pauli_action_cache.get(label)
            if action is None:
                action = compile_pauli_action_exyz(label, nq)
                self.pauli_action_cache[label] = action
            compiled_steps.append(CompiledRotationStep(coeff_real=float(coeff.real), action=action))

        return CompiledPolynomialRotationPlan(nq=nq, steps=tuple(compiled_steps))

    _MATH_PREPARE_STATE = (
        r"|\psi(\theta)\rangle=\prod_k \exp(-i\theta_k H_k)|\psi_{\mathrm{ref}}\rangle"
        r"\approx \prod_k \prod_j \exp(-i\theta_k c_{k,j}P_{k,j})|\psi_{\mathrm{ref}}\rangle"
    )

    def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        """Prepare ansatz state using compiled Pauli actions."""
        theta_vec = np.asarray(theta, dtype=float).reshape(-1)
        if int(theta_vec.size) != int(len(self._plans)):
            raise ValueError(
                f"theta length mismatch: got {theta_vec.size}, expected {len(self._plans)}."
            )

        psi = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
        if self.nq is not None:
            expected_dim = 1 << int(self.nq)
            if int(psi.size) != int(expected_dim):
                raise ValueError(
                    f"psi_ref length mismatch: got {psi.size}, expected {expected_dim} for nq={self.nq}."
                )

        for k, poly_plan in enumerate(self._plans):
            dt = float(theta_vec[k])
            if dt == 0.0:
                continue
            for step in poly_plan.steps:
                psi = apply_exp_term(
                    psi,
                    step.action,
                    coeff=complex(step.coeff_real),
                    dt=dt,
                    tol=self.coefficient_tolerance,
                )
        return psi


__all__ = ["CompiledAnsatzExecutor"]

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

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_vqe_from_adapt_family.py
(lines 780-1139: Replay contract extraction, handoff_state_kind inference, and replay seed-policy logic that define existing replay/export semantics for ADAPT-family continuation.)
```py
    theta = np.asarray(theta_vals, dtype=float)
    return labels, theta


def _inject_replay_terms_from_payload(
    label_to_term: dict[str, AnsatzTerm],
    payload: Mapping[str, Any] | None,
) -> None:
    if not isinstance(payload, Mapping):
        return
    continuation = payload.get("continuation", {})
    if not isinstance(continuation, Mapping):
        return
    raw_selected_meta = continuation.get("selected_generator_metadata", [])
    if not isinstance(raw_selected_meta, Sequence):
        return
    for raw_meta in raw_selected_meta:
        if not isinstance(raw_meta, Mapping):
            continue
        lbl = str(raw_meta.get("candidate_label", "")).strip()
        if lbl == "" or lbl in label_to_term:
            continue
        compile_meta = raw_meta.get("compile_metadata", {})
        serialized_terms = (
            compile_meta.get("serialized_terms_exyz", [])
            if isinstance(compile_meta, Mapping)
            else []
        )
        if not isinstance(serialized_terms, Sequence):
            continue
        try:
            poly = rebuild_polynomial_from_serialized_terms(serialized_terms)
        except Exception:
            continue
        label_to_term[lbl] = AnsatzTerm(label=str(lbl), polynomial=poly)


def _build_replay_terms_from_adapt_labels(
    family_pool: Sequence[AnsatzTerm],
    adapt_labels: Sequence[str],
    payload: Mapping[str, Any] | None = None,
) -> list[AnsatzTerm]:
    label_to_term: dict[str, AnsatzTerm] = {}
    duplicate_labels: list[str] = []
    for term in family_pool:
        lbl = str(term.label)
        if lbl in label_to_term:
            duplicate_labels.append(lbl)
            continue
        label_to_term[lbl] = term
    if duplicate_labels:
        dup_preview = sorted(set(duplicate_labels))[:8]
        raise ValueError(
            "Resolved family pool has duplicate operator labels; replay mapping is ambiguous. "
            f"Examples: {dup_preview}"
        )

    _inject_replay_terms_from_payload(label_to_term, payload)

    replay_terms: list[AnsatzTerm] = []
    missing: list[str] = []
    for lbl in adapt_labels:
        term = label_to_term.get(str(lbl), None)
        if term is None:
            missing.append(str(lbl))
            continue
        replay_terms.append(term)
    if missing:
        miss_preview = missing[:8]
        raise ValueError(
            "ADAPT operators are not present in the resolved replay family pool. "
            f"Missing examples: {miss_preview}"
        )
    return replay_terms


def _build_replay_seed_theta(adapt_theta: np.ndarray, *, reps: int) -> np.ndarray:
    if int(reps) < 1:
        raise ValueError("reps must be >= 1 for replay seed construction.")
    base = np.asarray(adapt_theta, dtype=float).reshape(-1)
    if int(base.size) == 0:
        raise ValueError("adapt_theta must be non-empty for replay seed construction.")
    if not np.all(np.isfinite(base)):
        raise ValueError("adapt_theta contains non-finite values.")
    return np.tile(base, int(reps)).astype(float, copy=False)


# ── Replay seed policy (v2 contract) ────────────────────────────────────
REPLAY_SEED_POLICIES = {"auto", "scaffold_plus_zero", "residual_only", "tile_adapt"}
REPLAY_CONTRACT_VERSION = 2

# Handoff-state kind constants
_PREPARED_STATE = "prepared_state"
_REFERENCE_STATE = "reference_state"

# Sources that map unambiguously to reference_state in legacy payloads.
_LEGACY_REFERENCE_SOURCES = {"hf", "exact"}
# Source suffixes that map to prepared_state in legacy staged exports.
_LEGACY_PREPARED_FINAL_SUFFIXES = ("_final",)
# Source values that map to prepared_state in legacy payloads.
_LEGACY_PREPARED_SOURCES = {"adapt_vqe", "seed_refine_vqe"}


def _coerce_bool(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        sval = value.strip().lower()
        if sval in {"1", "true", "yes", "on", "y"}:
            return True
        if sval in {"0", "false", "no", "off", "n"}:
            return False
    raise ValueError(f"{path} must be a boolean-like value.")


def _extract_replay_contract(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return parsed replay contract from continuation when present.

    Modern staged artifacts treat continuation.replay_contract as authoritative.
    If the field is present but malformed, this raises a ValueError rather than
    silently falling back to legacy metadata.
    """
    continuation = payload.get("continuation", None)
    if not isinstance(continuation, Mapping):
        return None

    raw_contract = continuation.get("replay_contract", None)
    if raw_contract is None:
        return None
    if not isinstance(raw_contract, Mapping):
        raise ValueError("continuation.replay_contract must be a JSON object/map.")

    version_raw = raw_contract.get("contract_version", raw_contract.get("version", None))
    if version_raw is None:
        raise ValueError("continuation.replay_contract must include contract_version.")
    try:
        version = int(version_raw)
    except Exception as exc:
        raise ValueError("continuation.replay_contract.contract_version must be int.") from exc
    if version != int(REPLAY_CONTRACT_VERSION):
        raise ValueError(
            f"Unsupported continuation.replay_contract.contract_version={version}; "
            f"expected {REPLAY_CONTRACT_VERSION}."
        )

    # family
    raw_family = raw_contract.get("generator_family")
    if raw_family is None:
        raise ValueError("continuation.replay_contract must include generator_family.")

    if isinstance(raw_family, str):
        raw_family_str = str(raw_family).strip().lower()
        requested = raw_family_str if raw_family_str == "match_adapt" else _canonical_family(raw_family_str)
        if requested is None:
            raise ValueError(
                "continuation.replay_contract.generator_family must be canonical HH family ID or "
                "match_adapt when specified as a string."
            )
        resolved = None if requested == "match_adapt" else requested
        fallback_family = None
        resolution_source = "continuation.replay_contract.generator_family"
        fallback_used = False
    elif isinstance(raw_family, Mapping):
        raw_requested = raw_family.get("requested", None)
        raw_resolved = raw_family.get("resolved", None)
        has_resolved = "resolved" in raw_family

        if raw_requested is None:
            requested = None
        else:
            requested_raw = str(raw_requested).strip().lower()
            requested = requested_raw if requested_raw == "match_adapt" else _canonical_family(requested_raw)
            if requested is None:
                raise ValueError(
                    "continuation.replay_contract.generator_family.requested must be a canonical HH family ID "
                    "or match_adapt."
                )

        if has_resolved:
            if raw_resolved is None:
                raise ValueError(
                    "continuation.replay_contract.generator_family.resolved must be present when 'requested' is supplied."
                )
            resolved = _canonical_family(raw_resolved)
            if resolved is None:
                raise ValueError(
                    "continuation.replay_contract.generator_family.resolved must be canonical HH family ID."
                )
        else:
            resolved = None

        if requested == "match_adapt" and not has_resolved:
            raise ValueError(
                "continuation.replay_contract.generator_family.requested='match_adapt' requires "
                "resolved canonical family."
            )

        if requested is None and resolved is not None:
            # Compatibility-friendly form: map-only contracts omit requested.
            requested = "match_adapt"

        if resolved is None and requested is not None and requested != "match_adapt":
            resolved = requested

        fallback_family = _canonical_family(raw_family.get("fallback_family")) if raw_family.get("fallback_family") is not None else None
        if raw_family.get("fallback_family") is not None and fallback_family is None:
            raise ValueError("continuation.replay_contract.generator_family.fallback_family is invalid.")

        resolution_source = str(raw_family.get("resolution_source", "continuation.replay_contract.generator_family"))

        if "fallback_used" in raw_family:
            fallback_used = _coerce_bool(raw_family.get("fallback_used"), "continuation.replay_contract.generator_family.fallback_used")
        else:
            fallback_used = False

        if fallback_used and fallback_family is None:
            raise ValueError(
                "continuation.replay_contract.generator_family.fallback_used requires fallback_family."
            )

        if requested is not None and requested != "match_adapt" and resolved is not None and resolved != requested:
            if not fallback_used:
                raise ValueError(
                    "continuation.replay_contract.generator_family is inconsistent: "
                    "resolved must match requested unless fallback_used is true."
                )
    else:
        raise ValueError("continuation.replay_contract.generator_family must be a family string or object.")

    if resolved is None:
        raise ValueError("continuation.replay_contract.generator_family.resolved is required.")

    # seed policies
    seed_policy_requested = _canonical_seed_policy(raw_contract.get("seed_policy_requested"))
    if seed_policy_requested is None:
        raise ValueError("continuation.replay_contract.seed_policy_requested is required and must be one of "
                         f"{sorted(REPLAY_SEED_POLICIES)}.")
    seed_policy_resolved = _canonical_seed_policy(raw_contract.get("seed_policy_resolved"))
    if seed_policy_resolved is None:
        raise ValueError("continuation.replay_contract.seed_policy_resolved is required and must be one of "
                         f"{sorted(REPLAY_SEED_POLICIES)}.")
    if seed_policy_requested != "auto" and seed_policy_resolved != seed_policy_requested:
        raise ValueError(
            "continuation.replay_contract seed policy is inconsistent: "
            f"requested='{seed_policy_requested}' resolved='{seed_policy_resolved}'."
        )

    # state-kind
    handoff_state_kind = str(raw_contract.get("handoff_state_kind", "")).strip().lower()
    if handoff_state_kind not in {_PREPARED_STATE, _REFERENCE_STATE}:
        raise ValueError(
            "continuation.replay_contract.handoff_state_kind must be one of "
            f"{{{_PREPARED_STATE}, {_REFERENCE_STATE}}}."
        )

    if "continuation_mode" not in raw_contract:
        raise ValueError("continuation.replay_contract must include continuation_mode.")
    continuation_mode = str(raw_contract.get("continuation_mode")).strip().lower()
    if continuation_mode not in {"legacy", "phase1_v1", "phase2_v1", "phase3_v1"}:
        raise ValueError(
            "continuation.replay_contract.continuation_mode must be one of "
            "{'legacy', 'phase1_v1', 'phase2_v1', 'phase3_v1'}."
        )

    return {
        "contract_version": int(version),
        "generator_family": {
            "requested": str(requested),
            "resolved": str(resolved),
            "resolution_source": str(resolution_source),
            "fallback_family": fallback_family,
            "fallback_used": bool(fallback_used),
        },
        "seed_policy_requested": str(seed_policy_requested),
        "seed_policy_resolved": str(seed_policy_resolved),
        "handoff_state_kind": str(handoff_state_kind),
        "continuation_mode": str(continuation_mode),
        "provenance_source": str(raw_contract.get("provenance_source", "explicit")),
    }


def _canonical_seed_policy(raw: Any) -> str | None:
    if raw is None:
        return None
    val = str(raw).strip().lower()
    return val if val in REPLAY_SEED_POLICIES else None


def _infer_handoff_state_kind(
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    """Return (handoff_state_kind, provenance_source).

    provenance_source is one of:
      "contract"           – "continuation.replay_contract.handoff_state_kind" was present.
      "explicit"          – ``initial_state.handoff_state_kind`` was present.
      "inferred_source"   – inferred from ``initial_state.source`` legacy field.
      "ambiguous"         – could not resolve; caller must raise.
    """
    contract = _extract_replay_contract(payload)
    if contract is not None:
        return str(contract["handoff_state_kind"]), "contract"

    init = payload.get("initial_state", {})
    if not isinstance(init, Mapping):
        init = {}

    explicit = init.get("handoff_state_kind", None)
    if isinstance(explicit, str) and explicit in {_PREPARED_STATE, _REFERENCE_STATE}:
        return str(explicit), "explicit"

    # Legacy inference from initial_state.source
    source_raw = init.get("source", None)
    if isinstance(source_raw, str):
        source = str(source_raw).strip().lower()
        if source in _LEGACY_REFERENCE_SOURCES:
            return _REFERENCE_STATE, "inferred_source"
        if source in _LEGACY_PREPARED_SOURCES:
            return _PREPARED_STATE, "inferred_source"
        # Staged final exports use suffixes like A_probe_final, B_medium_final
        for suffix in _LEGACY_PREPARED_FINAL_SUFFIXES:
            if source.endswith(suffix):
                return _PREPARED_STATE, "inferred_source"
        # Warm-start exports
        if "warm_start" in source:
            return _PREPARED_STATE, "inferred_source"

    return "ambiguous", "ambiguous"


def _build_replay_seed_theta_policy(
    adapt_theta: np.ndarray,
    *,
    reps: int,
    policy: str,
    handoff_state_kind: str,
) -> tuple[np.ndarray, str]:
    """Build replay seed theta according to the given policy.

    Returns (seed_theta, resolved_policy_name).
    """
    base = np.asarray(adapt_theta, dtype=float).reshape(-1)
    if int(base.size) == 0:
        raise ValueError("adapt_theta must be non-empty for replay seed construction.")
    if not np.all(np.isfinite(base)):
        raise ValueError("adapt_theta contains non-finite values.")
    if int(reps) < 1:
        raise ValueError("reps must be >= 1 for replay seed construction.")

    adapt_depth = int(base.size)
    total_params = adapt_depth * int(reps)

    if policy == "auto":
        if handoff_state_kind == _PREPARED_STATE:
            resolved = "residual_only"
        elif handoff_state_kind == _REFERENCE_STATE:
            resolved = "scaffold_plus_zero"
        else:

```

(lines 1724-2119: build_replay_sequence_from_input_json, family/replay ansatz context builders, and run() entry surfaces used to reconstruct matched-family replay from staged ADAPT outputs.)
```py
def build_replay_sequence_from_input_json(
    adapt_input_json: Path,
    *,
    generator_family: str = "match_adapt",
    fallback_family: str = "full_meta",
    legacy_paop_key: str = "paop_lf_std",
    replay_continuation_mode: str | None = "phase3_v1",
) -> dict[str, Any]:
    psi_ref, payload = _read_input_state_and_payload(Path(adapt_input_json))
    cfg = _build_sequence_cfg_from_payload(
        Path(adapt_input_json),
        payload,
        generator_family=str(generator_family),
        fallback_family=str(fallback_family),
        legacy_paop_key=str(legacy_paop_key),
        replay_continuation_mode=replay_continuation_mode,
    )
    family_info = _resolve_family(cfg, payload)
    h_poly = _build_hh_hamiltonian(cfg)
    e_exact = _resolve_exact_energy_from_payload(payload)
    if e_exact is None:
        e_exact = float(
            exact_ground_energy_sector_hh(
                h_poly,
                num_sites=int(cfg.L),
                num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
                ordering=str(cfg.ordering),
                n_ph_max=int(cfg.n_ph_max),
                boson_encoding=str(cfg.boson_encoding),
                t=float(cfg.t),
                U=float(cfg.u),
                dv=float(cfg.dv),
                omega0=float(cfg.omega0),
                g_ep=float(cfg.g_ep),
                boundary=str(cfg.boundary),
            )
        )
    replay_ctx = build_replay_ansatz_context(
        cfg,
        payload_in=payload,
        psi_ref=psi_ref,
        h_poly=h_poly,
        family_info=family_info,
        e_exact=float(e_exact),
    )
    return {
        "cfg": cfg,
        "payload": payload,
        "initial_state": np.asarray(psi_ref, dtype=complex).reshape(-1).copy(),
        "h_poly": h_poly,
        "family_info": dict(replay_ctx["family_info"]),
        "replay_terms": list(replay_ctx["replay_terms"]),
        "adapt_labels": list(replay_ctx["adapt_labels"]),
        "seed_theta": np.asarray(replay_ctx["seed_theta"], dtype=float).copy(),
        "family_resolved": str(replay_ctx["family_resolved"]),
        "family_terms_count": int(replay_ctx["family_terms_count"]),
        "pool_meta": dict(replay_ctx["pool_meta"]),
        "nq": int(replay_ctx["nq"]),
    }


def build_family_ansatz_context(
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
    family_key = _canonical_family(family)
    if family_key is None:
        raise ValueError(f"Unsupported generator family: {family}")

    if terms_override is None:
        terms, pool_meta = _build_pool_for_family(cfg, family=family_key, h_poly=h_poly)
        family_terms_count = int(len(terms))
    else:
        terms = list(terms_override)
        pool_meta = dict(pool_meta_override or {"family": family_key})
        family_terms_count = int(
            family_terms_count_override if family_terms_count_override is not None else len(terms)
        )
    if int(len(terms)) <= 0:
        raise ValueError(f"Generator family '{family_key}' materialized an empty ansatz term list.")

    nq = int(2 * int(cfg.L) + int(cfg.L) * int(boson_qubits_per_site(int(cfg.n_ph_max), str(cfg.boson_encoding))))
    ansatz = PoolTermwiseAnsatz(terms=list(terms), reps=int(cfg.reps), nq=nq)
    if seed_theta is None:
        seed_theta_arr = np.zeros(int(ansatz.num_parameters), dtype=float)
    else:
        seed_theta_arr = np.asarray(seed_theta, dtype=float).reshape(-1)
    if int(seed_theta_arr.size) != int(ansatz.num_parameters):
        raise ValueError(
            "Internal family parameter mismatch: "
            f"seed size {int(seed_theta_arr.size)} != ansatz.num_parameters {int(ansatz.num_parameters)}."
        )

    psi_seed = np.asarray(ansatz.prepare_state(seed_theta_arr, psi_ref), dtype=complex).reshape(-1)
    seed_energy = float(expval_pauli_polynomial(psi_seed, h_poly))
    seed_delta_abs = float(abs(seed_energy - float(e_exact)))
    seed_relative_abs = float(seed_delta_abs / max(abs(float(e_exact)), 1e-14))
    family_info = dict(
        family_info_override
        or {
            "requested": str(family_key),
            "resolved": str(family_key),
            "resolution_source": "explicit_family",
            "fallback_family": None,
            "fallback_used": False,
            "warning": None,
        }
    )
    return {
        "family_info": family_info,
        "family_resolved": str(family_key),
        "family_terms_count": int(family_terms_count),
        "pool_meta": dict(pool_meta),
        "terms": list(terms),
        "ansatz": ansatz,
        "nq": int(nq),
        "seed_theta": np.asarray(seed_theta_arr, dtype=float).copy(),
        "psi_seed": np.asarray(psi_seed, dtype=complex).reshape(-1).copy(),
        "seed_energy": float(seed_energy),
        "seed_delta_abs": float(seed_delta_abs),
        "seed_relative_abs": float(seed_relative_abs),
    }


def build_replay_ansatz_context(
    cfg: RunConfig,
    *,
    payload_in: Mapping[str, Any],
    psi_ref: np.ndarray,
    h_poly: Any,
    family_info: Mapping[str, Any],
    e_exact: float,
) -> dict[str, Any]:
    adapt_labels_raw, adapt_theta_raw = _extract_adapt_operator_theta_sequence(payload_in)
    handoff_state_kind, provenance_source = _infer_handoff_state_kind(payload_in)
    contract = _extract_replay_contract(payload_in)
    family_effective = dict(family_info)
    family_resolved_requested = str(family_effective["resolved"])
    adapt_labels, adapt_theta, logical_pair_seed_expansion = _expand_product_labels_and_theta_for_family(
        adapt_labels_raw,
        adapt_theta_raw,
        family=family_resolved_requested,
    )
    effective_seed_policy = str(cfg.replay_seed_policy)
    contract_seed_policy: str | None = None
    contract_seed_policy_resolved: str | None = None

    if contract is not None:
        contract_seed_policy = str(contract["seed_policy_requested"])
        contract_seed_policy_resolved = str(contract["seed_policy_resolved"])
        if effective_seed_policy == "auto":
            effective_seed_policy = contract_seed_policy
        elif contract_seed_policy != effective_seed_policy:
            # Explicit CLI override wins; keep CLI requested and validate contract consistency later.
            pass

    if provenance_source == "ambiguous" and effective_seed_policy == "auto":
        raise ValueError(
            "Cannot resolve replay seed policy 'auto': input JSON has no "
            "initial_state.handoff_state_kind and initial_state.source could not "
            "be mapped unambiguously to reference_state or prepared_state. "
            "Use an explicit --replay-seed-policy (scaffold_plus_zero, residual_only, "
            "or tile_adapt) to proceed."
        )

    seed_theta, resolved_seed_policy = _build_replay_seed_theta_policy(
        adapt_theta,
        reps=int(cfg.reps),
        policy=str(effective_seed_policy),
        handoff_state_kind=str(handoff_state_kind),
    )

    if contract is not None and effective_seed_policy == str(contract_seed_policy):
        if contract_seed_policy_resolved is not None and resolved_seed_policy != contract_seed_policy_resolved:
            raise ValueError(
                "Replay seed policy mismatch: contract resolved policy "
                f"'{contract_seed_policy_resolved}' conflicts with payload policy "
                f"'{resolved_seed_policy}' for handoff_state_kind='{handoff_state_kind}'."
            )
    family_resolved = str(family_effective["resolved"])
    try:
        replay_terms, pool_meta, family_terms_count = _build_replay_terms_for_family(
            cfg,
            family=family_resolved,
            h_poly=h_poly,
            adapt_labels=adapt_labels,
            payload=payload_in,
        )
    except ValueError as exc:
        missing_label_error = "ADAPT operators are not present in the resolved replay family pool."
        fallback_family = family_effective.get("fallback_family", None)
        can_retry_full_meta = (
            str(exc).startswith(missing_label_error)
            and isinstance(fallback_family, str)
            and str(fallback_family) == "full_meta"
            and family_resolved != "full_meta"
        )
        if not can_retry_full_meta:
            raise
        replay_terms, pool_meta, family_terms_count = _build_replay_terms_for_family(
            cfg,
            family="full_meta",
            h_poly=h_poly,
            adapt_labels=adapt_labels,
            payload=payload_in,
        )
        family_effective["resolved"] = "full_meta"
        family_effective["resolution_source"] = "fallback_family_missing_labels"
        family_effective["fallback_used"] = True
        warning = family_effective.get("warning", None)
        retry_warning = (
            "Resolved replay family could not represent the ADAPT-selected labels; "
            "retrying replay with fallback family 'full_meta'."
        )
        family_effective["warning"] = retry_warning if warning in (None, "") else f"{warning} {retry_warning}"
        family_resolved = "full_meta"

    family_ctx = build_family_ansatz_context(
        cfg,
        psi_ref=psi_ref,
        h_poly=h_poly,
        family=str(family_resolved),
        e_exact=float(e_exact),
        seed_theta=np.asarray(seed_theta, dtype=float),
        terms_override=replay_terms,
        pool_meta_override=pool_meta,
        family_terms_count_override=int(family_terms_count),
        family_info_override=family_effective,
    )
    return {
        "adapt_labels": list(adapt_labels),
        "adapt_theta": np.asarray(adapt_theta, dtype=float).copy(),
        "source_adapt_labels": list(adapt_labels_raw),
        "source_adapt_theta": np.asarray(adapt_theta_raw, dtype=float).copy(),
        "logical_pair_seed_expansion": logical_pair_seed_expansion,
        "handoff_state_kind": str(handoff_state_kind),
        "provenance_source": str(provenance_source),
        "resolved_seed_policy": str(resolved_seed_policy),
        "replay_terms": list(replay_terms),
        **family_ctx,
    }


def run(cfg: RunConfig, diagnostics_out: dict[str, Any] | None = None) -> dict[str, Any]:
    logger = RunLogger(cfg.output_log)
    logger.log(f"Loading ADAPT input JSON: {cfg.adapt_input_json}")
    psi_ref, payload_in = _read_input_state_and_payload(cfg.adapt_input_json)

    family_info = _resolve_family(cfg, payload_in)
    contract = _extract_replay_contract(payload_in)
    if family_info.get("warning"):
        logger.log(f"FAMILY WARNING: {family_info['warning']}")
    logger.log(
        f"Generator family resolved: requested={family_info['requested']} "
        f"resolved={family_info['resolved']} source={family_info['resolution_source']}"
    )

    logger.log("Building HH Hamiltonian.")
    h_poly = _build_hh_hamiltonian(cfg)
    e_exact_payload = _resolve_exact_energy_from_payload(payload_in)
    if e_exact_payload is not None:
        e_exact = float(e_exact_payload)
        logger.log(f"Using exact sector energy from input payload: E_exact={e_exact:.12f}")
    else:
        logger.log("Computing exact sector energy via ED (payload exact unavailable).")
        e_exact = float(
            exact_ground_energy_sector_hh(
                h_poly,
                num_sites=int(cfg.L),
                num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
                n_ph_max=int(cfg.n_ph_max),
                boson_encoding=str(cfg.boson_encoding),
                indexing=str(cfg.ordering),
            )
        )
        logger.log(f"Computed exact sector energy via ED: E_exact={e_exact:.12f}")

    replay_ctx = build_replay_ansatz_context(
        cfg,
        payload_in=payload_in,
        psi_ref=psi_ref,
        h_poly=h_poly,
        family_info=family_info,
        e_exact=float(e_exact),
    )
    family_info = dict(replay_ctx["family_info"])
    adapt_labels = [str(x) for x in replay_ctx["adapt_labels"]]
    adapt_theta = np.asarray(replay_ctx["adapt_theta"], dtype=float)
    handoff_state_kind = str(replay_ctx["handoff_state_kind"])
    provenance_source = str(replay_ctx["provenance_source"])
    if provenance_source not in {"explicit", "contract"}:
        logger.log(
            f"PROVENANCE WARNING: handoff_state_kind inferred as '{handoff_state_kind}' "
            f"from legacy metadata (source='{provenance_source}'). "
            "Consider regenerating input JSON with explicit handoff_state_kind."
        )
    logger.log(
        f"Provenance: handoff_state_kind={handoff_state_kind} "
        f"provenance_source={provenance_source} "
        f"replay_seed_policy={cfg.replay_seed_policy}"
    )
    seed_theta = np.asarray(replay_ctx["seed_theta"], dtype=float)
    resolved_seed_policy = str(replay_ctx["resolved_seed_policy"])
    family_resolved = str(replay_ctx["family_resolved"])
    family_terms_count = int(replay_ctx["family_terms_count"])
    pool_meta = dict(replay_ctx["pool_meta"])
    replay_terms = list(replay_ctx["replay_terms"])
    ansatz = replay_ctx["ansatz"]
    nq = int(replay_ctx["nq"])
    if bool(family_info.get("fallback_used", False)) and str(family_info.get("resolved")) != str(family_resolved):
        family_info["resolved"] = str(family_resolved)
    if str(family_info.get("resolved")) == "full_meta" and str(family_info.get("resolution_source")) == "fallback_family_missing_labels":
        logger.log(
            "Replay family fallback applied: initial resolved family could not represent "
            "the ADAPT-selected labels; using full_meta for replay reconstruction."
        )
    logger.log(
        f"Pool built: family={family_resolved} family_terms={family_terms_count} "
        f"adapt_depth={len(adapt_labels)} replay_terms={len(replay_terms)} npar={ansatz.num_parameters}"
    )
    psi_seed = np.asarray(replay_ctx["psi_seed"], dtype=complex).reshape(-1)
    seed_energy = float(replay_ctx["seed_energy"])
    seed_delta_abs = float(replay_ctx["seed_delta_abs"])
    seed_relative_abs = float(replay_ctx["seed_relative_abs"])
    logger.log(
        f"Seed baseline (policy={resolved_seed_policy}): "
        f"E={seed_energy:.12f} |DeltaE|={seed_delta_abs:.6e}"
    )

    progress_tail: list[dict[str, Any]] = []
    run_t0 = time.perf_counter()
    wall_hit = False
    contract_mode = None if contract is None else str(contract.get("continuation_mode", "legacy"))
    replay_mode = _resolve_replay_continuation_mode(
        cfg.replay_continuation_mode if cfg.replay_continuation_mode is not None else contract_mode
    )
    incoming_optimizer_memory = None
    incoming_generator_metadata = None
    incoming_motif_library = None
    if isinstance(payload_in.get("continuation"), Mapping):
        incoming_optimizer_memory = payload_in.get("continuation", {}).get("optimizer_memory", None)
        incoming_generator_metadata = payload_in.get("continuation", {}).get("selected_generator_metadata", None)
        incoming_motif_library = payload_in.get("continuation", {}).get("motif_library", None)

    def _progress_logger(ev: dict[str, Any]) -> None:
        nonlocal progress_tail
        row = dict(ev)
        e_cur = row.get("energy_current", None)
        e_best = row.get("energy_best_global", None)
        if isinstance(e_cur, (int, float)):
            row["delta_abs_current"] = float(abs(float(e_cur) - float(e_exact)))
        if isinstance(e_best, (int, float)):
            row["delta_abs_best"] = float(abs(float(e_best) - float(e_exact)))
        progress_tail.append(row)
        if len(progress_tail) > 200:
            progress_tail = progress_tail[-200:]
        if str(row.get("event", "")) in {"heartbeat", "restart_end", "run_end", "early_stop_triggered"}:
            logger.log(
                f"VQE {row.get('event')} elapsed_s={float(row.get('elapsed_s', 0.0)):.1f} "
                f"nfev={int(row.get('nfev_so_far', 0))} "
                f"delta_abs_best={row.get('delta_abs_best')}"
            )

    def _early_stop_checker(ev: dict[str, Any]) -> bool:
        nonlocal wall_hit
        elapsed = float(ev.get("elapsed_s", 0.0))
        if elapsed >= float(cfg.wallclock_cap_s):
            wall_hit = True
            return True
        return False

    common_opt_kwargs = {
        "spsa_a": float(cfg.spsa_a),
        "spsa_c": float(cfg.spsa_c),
        "spsa_alpha": float(cfg.spsa_alpha),
        "spsa_gamma": float(cfg.spsa_gamma),
        "spsa_A": float(cfg.spsa_A),
        "spsa_avg_last": int(cfg.spsa_avg_last),
        "spsa_eval_repeats": int(cfg.spsa_eval_repeats),
        "spsa_eval_agg": str(cfg.spsa_eval_agg),
        "energy_backend": str(cfg.energy_backend),
    }
    replay_phase_history: list[dict[str, Any]] = []
    replay_phase_config: dict[str, Any] = {}
    if replay_mode == "legacy":
        vqe_res = vqe_minimize(
            h_poly,

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_stage_control.py
```py
#!/usr/bin/env python3
"""Stage and position-probe policy for HH continuation Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StageControllerConfig:
    plateau_patience: int = 2
    weak_drop_threshold: float = 1e-9
    probe_margin_ratio: float = 1.0
    max_probe_positions: int = 6
    append_admit_threshold: float = 0.05
    family_repeat_patience: int = 2


@dataclass(frozen=True)
class PositionProbeDecision:
    should_probe: bool
    reason: str
    positions: list[int]


def allowed_positions(
    *,
    n_params: int,
    append_position: int,
    active_window_indices: Iterable[int],
    max_positions: int,
) -> list[int]:
    positions = [int(append_position)]
    if int(n_params) <= 0:
        return [0]

    positions.append(0)
    for idx in active_window_indices:
        positions.append(int(idx))

    out: list[int] = []
    for p in positions:
        p_clamped = max(0, min(int(append_position), int(p)))
        if p_clamped not in out:
            out.append(p_clamped)
        if len(out) >= int(max_positions):
            break
    return out


def detect_trough(
    *,
    append_score: float,
    best_non_append_score: float,
    best_non_append_g_lcb: float,
    margin_ratio: float,
    append_admit_threshold: float,
) -> bool:
    if float(best_non_append_g_lcb) <= 0.0:
        return False
    if float(best_non_append_score) >= float(margin_ratio) * float(append_score):
        return True
    return (
        float(append_score) < float(append_admit_threshold)
        and float(best_non_append_score) >= float(append_admit_threshold)
    )


def should_probe_positions(
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
    if str(stage_name) == "residual":
        return False, "residual_stage"
    if int(drop_plateau_hits) >= int(cfg.plateau_patience):
        return True, "drop_plateau"
    if float(max_grad) < float(eps_grad) and bool(finite_angle_flat):
        return True, "eps_grad_flat"
    if bool(repeated_family_flat):
        return True, "family_repeat_flat"
    return False, "default_append_only"


class StageController:
    def __init__(self, cfg: StageControllerConfig) -> None:
        self.cfg = cfg
        self._stage = "core"

    def clone(self) -> "StageController":
        cloned = StageController(self.cfg)
        cloned._stage = str(self._stage)
        return cloned

    def snapshot(self) -> dict[str, str | StageControllerConfig]:
        return {
            "cfg": self.cfg,
            "stage": str(self._stage),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, object]) -> "StageController":
        cfg = snapshot.get("cfg")
        if not isinstance(cfg, StageControllerConfig):
            raise TypeError("StageController snapshot missing StageControllerConfig.")
        out = cls(cfg)
        out._stage = str(snapshot.get("stage", "core"))
        return out

    @property
    def stage_name(self) -> str:
        return str(self._stage)

    def start_with_seed(self) -> None:
        self._stage = "seed"

    def begin_core(self) -> None:
        self._stage = "core"

    def resolve_stage_transition(
        self,
        *,
        drop_plateau_hits: int,
        trough_detected: bool,
        residual_opened: bool,
    ) -> tuple[str, str]:
        if self._stage == "seed":
            self._stage = "core"
            return self._stage, "seed_complete"
        if self._stage == "core":
            if int(drop_plateau_hits) >= int(self.cfg.plateau_patience) and (not bool(trough_detected)):
                self._stage = "residual"
                return self._stage, "plateau_without_trough"
            return self._stage, "stay_core"
        if self._stage == "residual":
            if bool(residual_opened):
                return self._stage, "stay_residual"
            return self._stage, "residual_closed"
        return self._stage, "unknown_stage"

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/operator_pools/vlf_sq.py
```py
from __future__ import annotations

from typing import Any

from src.quantum.hubbard_latex_python_pairs import (
    boson_displacement_operator,
    boson_operator,
    boson_qubits_per_site,
    jw_number_operator,
    mode_index,
    phonon_qubit_indices_for_site,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.operator_pools.polaron_paop import (
    _clean_poly,
    _distance_1d,
    _mul_clean,
    _normalize_poly,
    _to_signature,
)

__all__ = ["build_vlf_sq_pool", "make_vlf_sq_pool"]

_MATH_SHIFTED_DENSITY = "δn_i := n_i - nbar I"
_MATH_VLF_SHELL = "G_r^VLF := Σ_{dist(i,j)=r} δn_i P_j"
_MATH_SQ = "G^SQ := Σ_i 1/2 (X_i P_i + P_i X_i) = Σ_i i[(b_i^†)^2 - b_i^2]"
_MATH_DENS_SQ = "G^(n)_SQ := Σ_i δn_i · 1/2 (X_i P_i + P_i X_i)"


def _family_flags(name: str) -> tuple[str, bool, bool, bool]:
    mode = str(name).strip().lower()
    if mode not in {"vlf_only", "sq_only", "vlf_sq", "sq_dens_only", "vlf_sq_dens"}:
        raise ValueError(
            "VLF/SQ family name must be one of vlf_only, sq_only, vlf_sq, sq_dens_only, vlf_sq_dens."
        )
    include_vlf = mode in {"vlf_only", "vlf_sq", "vlf_sq_dens"}
    include_sq = mode in {"sq_only", "vlf_sq", "vlf_sq_dens"}
    include_dens_sq = mode in {"sq_dens_only", "vlf_sq_dens"}
    return mode, include_vlf, include_sq, include_dens_sq


# Math: shells(periodic/open) := { r | 0 <= r <= r_max_effective and ∃(i,j) with dist(i,j)=r }
def _shells_for_radius(*, num_sites: int, periodic: bool, shell_radius: int | None) -> list[int]:
    if int(num_sites) <= 0:
        return []
    if bool(periodic):
        max_possible = int(num_sites) // 2
    else:
        max_possible = max(0, int(num_sites) - 1)
    if shell_radius is None:
        return list(range(max_possible + 1))
    cap = min(max(0, int(shell_radius)), max_possible)
    return list(range(cap + 1))


# Math: I := e^{\otimes nq}
def _identity_poly(nq: int) -> PauliPolynomial:
    return PauliPolynomial("JW", [PauliTerm(int(nq), ps="e" * int(nq), pc=1.0)])


# Math: build_vlf_sq_pool(name) -> {prefixed macro generators, metadata}
def build_vlf_sq_pool(
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
    mode, include_vlf, include_sq, include_dens_sq = _family_flags(name)
    n_sites = int(num_sites)
    if n_sites <= 0:
        return [], {
            "family": mode,
            "nbar": 0.0,
            "shell_radius": None if shell_radius is None else int(shell_radius),
            "shells": [],
            "parameter_count": 0,
            "sq_parameterization": "global_shared",
            "density_conditioned_sq": bool(include_dens_sq),
            "math_contract": {
                "shifted_density": _MATH_SHIFTED_DENSITY,
                "vlf_shell": _MATH_VLF_SHELL,
                "sq": _MATH_SQ,
                "dens_sq": _MATH_DENS_SQ,
            },
        }

    n_ph_max_i = int(n_ph_max)
    boson_encoding_i = str(boson_encoding)
    ordering_i = str(ordering)
    periodic = str(boundary).strip().lower() == "periodic"
    total_electrons = int(num_particles[0]) + int(num_particles[1]) if num_particles else 0
    nbar = (float(total_electrons) / float(n_sites)) if total_electrons > 0 else 1.0
    if nbar <= 0.0:
        nbar = 1.0

    nq = 2 * n_sites + n_sites * boson_qubits_per_site(n_ph_max_i, boson_encoding_i)
    id_poly = _identity_poly(nq)
    phonon_qubit_cache: dict[int, tuple[int, ...]] = {}
    number_cache: dict[int, PauliPolynomial] = {}
    p_cache: dict[int, PauliPolynomial] = {}
    x_cache: dict[int, PauliPolynomial] = {}
    sq_cache: dict[int, PauliPolynomial] = {}

    def local_qubits(site: int) -> tuple[int, ...]:
        key = int(site)
        if key not in phonon_qubit_cache:
            phonon_qubit_cache[key] = tuple(
                phonon_qubit_indices_for_site(
                    key,
                    n_sites=n_sites,
                    qpb=boson_qubits_per_site(n_ph_max_i, boson_encoding_i),
                    fermion_qubits=2 * n_sites,
                )
            )
        return phonon_qubit_cache[key]

    def n_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in number_cache:
            up = mode_index(key, 0, indexing=ordering_i, n_sites=n_sites)
            dn = mode_index(key, 1, indexing=ordering_i, n_sites=n_sites)
            number_cache[key] = jw_number_operator("JW", nq, up) + jw_number_operator("JW", nq, dn)
        return number_cache[key]

    def shifted_density(site: int) -> PauliPolynomial:
        return n_i(int(site)) + ((-float(nbar)) * id_poly)

    def b_i(site: int) -> PauliPolynomial:
        return boson_operator(
            "JW",
            nq,
            local_qubits(int(site)),
            which="b",
            n_ph_max=n_ph_max_i,
            encoding=boson_encoding_i,
        )

    def bdag_i(site: int) -> PauliPolynomial:
        return boson_operator(
            "JW",
            nq,
            local_qubits(int(site)),
            which="bdag",
            n_ph_max=n_ph_max_i,
            encoding=boson_encoding_i,
        )

    def p_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in p_cache:
            p_cache[key] = _clean_poly((1j * bdag_i(key)) + ((-1j) * b_i(key)), float(prune_eps))
        return p_cache[key]

    def x_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in x_cache:
            x_cache[key] = boson_displacement_operator(
                "JW",
                nq,
                local_qubits(int(key)),
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
        return x_cache[key]

    def sq_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in sq_cache:
            xp = _mul_clean(x_i(key), p_i(key), float(prune_eps), enforce_real=False)
            px = _mul_clean(p_i(key), x_i(key), float(prune_eps), enforce_real=False)
            sq_cache[key] = _clean_poly(0.5 * (xp + px), float(prune_eps))
        return sq_cache[key]

    shells = _shells_for_radius(num_sites=n_sites, periodic=periodic, shell_radius=shell_radius)
    raw_pool: list[tuple[str, PauliPolynomial]] = []

    if include_vlf:
        for shell in shells:
            shell_poly = PauliPolynomial("JW")
            pair_count = 0
            for i_site in range(n_sites):
                for j_site in range(n_sites):
                    if _distance_1d(i_site, j_site, n_sites, periodic) != int(shell):
                        continue
                    shell_poly += _mul_clean(shifted_density(i_site), p_i(j_site), float(prune_eps))
                    pair_count += 1
            shell_poly = _clean_poly(shell_poly, float(prune_eps))
            if shell_poly.return_polynomial():
                raw_pool.append((f"vlf_shell(r={shell})", _normalize_poly(shell_poly, str(normalization))))

    if include_sq:
        sq_poly = PauliPolynomial("JW")
        for site in range(n_sites):
            sq_poly += sq_i(site)
        sq_poly = _clean_poly(sq_poly, float(prune_eps))
        if sq_poly.return_polynomial():
            raw_pool.append(("sq_global", _normalize_poly(sq_poly, str(normalization))))

    if include_dens_sq:
        dens_sq_poly = PauliPolynomial("JW")
        for site in range(n_sites):
            dens_sq_poly += _mul_clean(shifted_density(site), sq_i(site), float(prune_eps))
        dens_sq_poly = _clean_poly(dens_sq_poly, float(prune_eps))
        if dens_sq_poly.return_polynomial():
            raw_pool.append(("dens_sq_global", _normalize_poly(dens_sq_poly, str(normalization))))

    if mode == "sq_only" and not raw_pool:
        raise ValueError("sq_only produced no surviving squeeze generators; n_ph_max may be too small.")
    if mode == "sq_dens_only" and not raw_pool:
        raise ValueError("sq_dens_only produced no surviving density-conditioned squeeze generators; n_ph_max may be too small.")

    dedup: list[tuple[str, PauliPolynomial]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for label, poly in raw_pool:
        sig = _to_signature(poly)
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append((f"{mode}:{label}", poly))

    meta = {
        "family": mode,
        "nbar": float(nbar),
        "shell_radius": None if shell_radius is None else int(shell_radius),
        "shells": list(shells if include_vlf else []),
        "parameter_count": int(len(dedup)),
        "sq_parameterization": "global_shared" if include_sq or include_dens_sq else "off",
        "density_conditioned_sq": bool(include_dens_sq),
        "math_contract": {
            "shifted_density": _MATH_SHIFTED_DENSITY,
            "vlf_shell": _MATH_VLF_SHELL,
            "sq": _MATH_SQ,
            "dens_sq": _MATH_DENS_SQ,
        },
    }
    return dedup, meta


def make_vlf_sq_pool(
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
    pool, _meta = build_vlf_sq_pool(
        name,
        num_sites=int(num_sites),
        num_particles=tuple(num_particles) if num_particles is not None else (),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        shell_radius=None if shell_radius is None else int(shell_radius),
        prune_eps=float(prune_eps),
        normalization=str(normalization),
    )
    return pool

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_pruning.py
```py
#!/usr/bin/env python3
"""Prune-before-replay helpers for HH continuation Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pipelines.hardcoded.hh_continuation_types import PruneDecision


@dataclass(frozen=True)
class PruneConfig:
    max_candidates: int = 6
    min_candidates: int = 2
    fraction_candidates: float = 0.25
    max_regression: float = 1e-8


def rank_prune_candidates(
    *,
    theta: np.ndarray,
    labels: list[str],
    marginal_proxy_benefit: list[float] | None,
    max_candidates: int,
    min_candidates: int,
    fraction_candidates: float,
) -> list[int]:
    n = int(theta.size)
    if n <= 0:
        return []
    target = int(np.ceil(float(fraction_candidates) * float(n)))
    target = max(int(min_candidates), target)
    target = min(int(max_candidates), target, n)
    benefits = list(marginal_proxy_benefit) if marginal_proxy_benefit is not None else []

    def _benefit_key(i: int) -> float:
        if i >= len(benefits):
            return float("inf")
        val = float(benefits[i])
        if not np.isfinite(val):
            return float("inf")
        return float(val)

    order = sorted(
        range(n),
        key=lambda i: (
            abs(float(theta[i])),
            _benefit_key(int(i)),
            str(labels[i]),
        ),
    )
    return [int(i) for i in order[:target]]


def apply_pruning(
    *,
    theta: np.ndarray,
    labels: list[str],
    candidate_indices: list[int],
    eval_with_removal: Callable[..., tuple[float, np.ndarray]],
    energy_before: float,
    max_regression: float,
) -> tuple[np.ndarray, list[str], list[PruneDecision], float]:
    cur_theta = np.asarray(theta, dtype=float).copy()
    cur_labels = list(labels)
    decisions: list[PruneDecision] = []
    cur_energy = float(energy_before)
    removed_so_far = 0

    for idx0 in candidate_indices:
        idx = int(idx0) - int(removed_so_far)
        if idx < 0 or idx >= len(cur_labels):
            continue
        try:
            trial_energy, trial_theta = eval_with_removal(idx, cur_theta, list(cur_labels))
        except TypeError:
            trial_energy, trial_theta = eval_with_removal(idx, cur_theta)
        regression = float(trial_energy - cur_energy)
        accepted = bool(regression <= float(max_regression))
        reason = "accepted" if accepted else "regression_exceeded"
        label = str(cur_labels[idx])
        decisions.append(
            PruneDecision(
                index=int(idx),
                label=label,
                accepted=bool(accepted),
                energy_before=float(cur_energy),
                energy_after=float(trial_energy),
                regression=float(regression),
                reason=str(reason),
            )
        )
        if accepted:
            cur_theta = np.asarray(trial_theta, dtype=float).copy()
            del cur_labels[idx]
            cur_energy = float(trial_energy)
            removed_so_far += 1

    return cur_theta, cur_labels, decisions, float(cur_energy)


def post_prune_refit(
    *,
    theta: np.ndarray,
    refit_fn: Callable[[np.ndarray], tuple[np.ndarray, float]],
) -> tuple[np.ndarray, float]:
    return refit_fn(np.asarray(theta, dtype=float))

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/pipelines/hardcoded/hh_continuation_symmetry.py
```py
#!/usr/bin/env python3
"""Shared symmetry metadata and verify-only mitigation hooks for HH continuation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from pipelines.hardcoded.hh_continuation_types import SymmetrySpec


_LOW_RISK_FAMILIES = {
    "paop",
    "paop_min",
    "paop_std",
    "paop_full",
    "paop_lf",
    "paop_lf_std",
    "paop_lf2_std",
    "paop_lf_full",
    "uccsd_paop_lf_full",
    "uccsd",
    "hva",
    "core",
}

_ALLOWED_SYMMETRY_MODES = {"off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"}


def normalize_phase3_symmetry_mitigation_mode(mode: str | None) -> str:
    mode_key = "off" if mode is None else str(mode).strip().lower()
    if mode_key == "":
        return "off"
    if mode_key not in _ALLOWED_SYMMETRY_MODES:
        raise ValueError(
            "phase3_symmetry_mitigation_mode must be one of "
            f"{sorted(_ALLOWED_SYMMETRY_MODES)}."
        )
    return str(mode_key)


def build_symmetry_spec(
    *,
    family_id: str,
    mitigation_mode: str = "off",
) -> SymmetrySpec:
    family_key = str(family_id).strip().lower()
    mitigation_mode_key = normalize_phase3_symmetry_mitigation_mode(mitigation_mode)
    if family_key in _LOW_RISK_FAMILIES:
        leakage_risk = 0.0
    elif family_key in {"residual", "full_meta", "all_hh_meta_v1", "full_hamiltonian"}:
        leakage_risk = 0.1
    else:
        leakage_risk = 0.2
    tags = ["fermion_number", "spin_sector"]
    if mitigation_mode_key in {"postselect_diag_v1", "projector_renorm_v1"}:
        tags.append("active_symmetry_requested")
    return SymmetrySpec(
        particle_number_mode="preserving",
        spin_sector_mode="preserving",
        phonon_number_mode="not_conserved",
        leakage_risk=float(leakage_risk),
        mitigation_eligible=bool(mitigation_mode_key != "off"),
        grouping_eligible=bool(leakage_risk <= 0.2),
        hard_guard=False,
        tags=list(tags),
    )


def symmetry_spec_to_dict(spec: SymmetrySpec | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if isinstance(spec, SymmetrySpec):
        return asdict(spec)
    if isinstance(spec, Mapping):
        return dict(spec)
    return None


def leakage_penalty_from_spec(spec: SymmetrySpec | Mapping[str, Any] | None) -> float:
    if isinstance(spec, SymmetrySpec):
        return float(spec.leakage_risk)
    if isinstance(spec, Mapping):
        return float(spec.get("leakage_risk", 0.0))
    return 0.0


def verify_symmetry_sequence(
    *,
    generator_metadata: Sequence[Mapping[str, Any]],
    mitigation_mode: str,
) -> dict[str, Any]:
    mode = normalize_phase3_symmetry_mitigation_mode(mitigation_mode)
    if mode == "off":
        return {
            "mode": "off",
            "executed": False,
            "active": False,
            "passed": True,
            "high_risk_count": 0,
            "max_leakage_risk": 0.0,
        }
    leakage_values = []
    for meta in generator_metadata:
        spec = meta.get("symmetry_spec", None) if isinstance(meta, Mapping) else None
        leakage_values.append(float(leakage_penalty_from_spec(spec)))
    max_risk = float(max(leakage_values) if leakage_values else 0.0)
    high_risk_count = int(sum(1 for val in leakage_values if float(val) > 0.5))
    return {
        "mode": str(mode),
        "executed": True,
        "active": bool(mode in {"postselect_diag_v1", "projector_renorm_v1"}),
        "passed": bool(high_risk_count == 0),
        "high_risk_count": int(high_risk_count),
        "max_leakage_risk": float(max_risk),
    }

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

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/test/test_staged_export_replay_roundtrip.py
```py
"""Round-trip test: staged export -> canonical replay metadata validation.

Verifies that state bundles emitted by the upgraded staged exporter satisfy
the canonical replay parser contract in ``hh_vqe_from_adapt_family.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_vqe_from_adapt_family import (
    _extract_adapt_operator_theta_sequence,
    _extract_replay_contract,
    _infer_handoff_state_kind,
    _resolve_family_from_metadata,
)
from pipelines.hardcoded.handoff_state_bundle import (
    HandoffStateBundleConfig,
    write_handoff_state_bundle,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal staged-export payload that mirrors what
# write_handoff_state_bundle now emits for a final stage export.
# ---------------------------------------------------------------------------

def _make_staged_export_payload(
    *,
    L: int = 2,
    operators: list[str] | None = None,
    optimal_point: list[float] | None = None,
    pool_type: str | None = None,
) -> dict:
    if operators is None:
        operators = ["op_a", "op_b", "op_c"]
    if optimal_point is None:
        optimal_point = [0.1, -0.2, 0.3]
    if pool_type is None:
        pool_type = "pool_a"

    nq = 2 * L + L  # minimal: 2L fermion + L boson bits for n_ph_max=1/binary
    dim = 1 << nq
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    # Build amplitudes dict
    amps = {}
    for idx in range(dim):
        amp = psi[idx]
        if abs(amp) > 1e-14:
            amps[format(idx, f"0{nq}b")] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}

    return {
        "generated_utc": "2026-03-06T00:00:00Z",
        "settings": {
            "L": L,
            "problem": "hh",
            "t": 1.0,
            "u": 4.0,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.5,
            "n_ph_max": 1,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "sector_n_up": (L + 1) // 2,
            "sector_n_dn": L // 2,
        },
        "adapt_vqe": {
            "energy": -1.0,
            "abs_delta_e": 0.01,
            "relative_error_abs": 0.001,
            "operators": operators,
            "optimal_point": optimal_point,
            "ansatz_depth": len(operators),
            "num_parameters": len(optimal_point),
            "pool_type": pool_type,
        },
        "initial_state": {
            "source": "A_probe_final",
            "nq_total": nq,
            "amplitudes_qn_to_q0": amps,
            "amplitude_cutoff": 1e-14,
            "norm": 1.0,
            "handoff_state_kind": "prepared_state",
        },
        "ground_state": {
            "exact_energy": -2.0,
            "exact_energy_filtered": -2.0,
            "filtered_sector": {"n_up": (L + 1) // 2, "n_dn": L // 2},
            "method": "staged_handoff_bundle",
        },
        "exact": {"E_exact_sector": -2.0},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCanonicalReplayFieldsPresent:
    """Verify exported payload satisfies the canonical replay parser."""

    def test_extract_succeeds_for_staged_export(self) -> None:
        payload = _make_staged_export_payload()
        labels, theta = _extract_adapt_operator_theta_sequence(payload)
        assert labels == ["op_a", "op_b", "op_c"]
        assert np.allclose(theta, [0.1, -0.2, 0.3])

    def test_operators_and_optimal_point_length_match(self) -> None:
        payload = _make_staged_export_payload(operators=["a", "b"], optimal_point=[0.1, 0.2])
        labels, theta = _extract_adapt_operator_theta_sequence(payload)
        assert len(labels) == len(theta)

    def test_pool_type_resolves_pool_a(self) -> None:
        payload = _make_staged_export_payload(pool_type="pool_a")
        fam, src = _resolve_family_from_metadata(payload)
        assert fam == "pool_a"
        assert src == "adapt_vqe.pool_type"

    def test_pool_type_resolves_pool_b(self) -> None:
        payload = _make_staged_export_payload(pool_type="pool_b")
        fam, src = _resolve_family_from_metadata(payload)
        assert fam == "pool_b"
        assert src == "adapt_vqe.pool_type"

    def test_seed_provenance_is_ignored_for_replay_family_resolution(self) -> None:
        payload = _make_staged_export_payload(pool_type="pool_a")
        payload["seed_provenance"] = {
            "warm_ansatz": "hh_hva_ptw",
            "refine_family": "uccsd_otimes_paop_lf_std",
            "refine_family_kind": "uccsd_paop_product",
            "refine_paop_motif_families": ["paop_lf_std"],
            "refine_reps": 2,
        }
        fam, src = _resolve_family_from_metadata(payload)
        assert fam == "pool_a"
        assert src == "adapt_vqe.pool_type"

    def test_ansatz_depth_matches_operators(self) -> None:
        ops = ["x", "y", "z", "w"]
        payload = _make_staged_export_payload(operators=ops, optimal_point=[0.1] * 4)
        assert payload["adapt_vqe"]["ansatz_depth"] == len(ops)
        assert payload["adapt_vqe"]["num_parameters"] == len(ops)

    def test_settings_has_required_hh_keys(self) -> None:
        for L in [2, 3, 4, 5]:
            payload = _make_staged_export_payload(L=L)
            settings = payload["settings"]
            for key in ("L", "t", "u", "omega0", "g_ep", "n_ph_max",
                        "boson_encoding", "ordering", "boundary",
                        "sector_n_up", "sector_n_dn"):
                assert key in settings, f"Missing settings key {key} for L={L}"
            assert settings["L"] == L

    def test_initial_state_has_amplitudes(self) -> None:
        payload = _make_staged_export_payload()
        assert "amplitudes_qn_to_q0" in payload["initial_state"]
        assert "nq_total" in payload["initial_state"]


class TestStagedExportRejectsIncomplete:
    """Verify the replay parser rejects incomplete staged exports."""

    def test_missing_operators_rejected(self) -> None:
        payload = _make_staged_export_payload()
        del payload["adapt_vqe"]["operators"]
        with pytest.raises(ValueError, match="adapt_vqe.operators"):
            _extract_adapt_operator_theta_sequence(payload)

    def test_missing_optimal_point_rejected(self) -> None:
        payload = _make_staged_export_payload()
        del payload["adapt_vqe"]["optimal_point"]
        with pytest.raises(ValueError, match="adapt_vqe.optimal_point"):
            _extract_adapt_operator_theta_sequence(payload)

    def test_length_mismatch_rejected(self) -> None:
        payload = _make_staged_export_payload(operators=["a"], optimal_point=[0.1, 0.2])
        with pytest.raises(ValueError, match="Length mismatch"):
            _extract_adapt_operator_theta_sequence(payload)


class TestArbitraryLRoundTrip:
    """Verify round-trip for L=2,3,4,5."""

    @pytest.mark.parametrize("L", [2, 3, 4, 5])
    def test_roundtrip_for_L(self, L: int) -> None:
        n_ops = L + 1
        ops = [f"op_{i}" for i in range(n_ops)]
        theta = [float(i) * 0.1 for i in range(n_ops)]
        payload = _make_staged_export_payload(L=L, operators=ops, optimal_point=theta)

        labels, theta_out = _extract_adapt_operator_theta_sequence(payload)
        assert labels == ops
        assert np.allclose(theta_out, theta)

        fam, src = _resolve_family_from_metadata(payload)
        assert fam == "pool_a"

    @pytest.mark.parametrize("L", [2, 3, 4, 5])
    def test_settings_sector_half_filling(self, L: int) -> None:
        payload = _make_staged_export_payload(L=L)
        s = payload["settings"]
        assert s["sector_n_up"] == (L + 1) // 2
        assert s["sector_n_dn"] == L // 2


class TestWriteStateBundleRoundTrip:
    """Test the active handoff bundle writer produces replay-compatible JSON."""

    def test_write_and_read_back(self, tmp_path: Path) -> None:
        """Write a state bundle via write_handoff_state_bundle and verify it round-trips."""
        cfg = HandoffStateBundleConfig(
            L=2,
            t=1.0,
            U=4.0,
            dv=0.0,
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            sector_n_up=1,
            sector_n_dn=1,
        )

        nq = 2 * 2 + 2  # L=2, n_ph_max=1, binary
        psi = np.zeros(1 << nq, dtype=complex)
        psi[0] = 1.0

        out_path = tmp_path / "stage_export.json"
        write_handoff_state_bundle(
            path=out_path,
            psi_state=psi,
            cfg=cfg,
            source="A_probe_final",
            exact_energy=-2.0,
            energy=-1.9,
            delta_E_abs=0.1,
            relative_error_abs=0.05,
            meta={"run_id": "A_probe", "budget_name": "probe"},
            adapt_operators=["op_x", "op_y"],
            adapt_optimal_point=[0.1, -0.2],
            adapt_pool_type="pool_a",
            settings_adapt_pool="pool_a",
            continuation_mode="phase1_v1",
            continuation_scaffold={"num_parameters": 2, "post_prune": True},
            optimizer_memory={"version": "phase2_optimizer_memory_v1", "parameter_count": 2, "available": True},
            selected_generator_metadata=[
                {
                    "generator_id": "gen:1",
                    "family_id": "paop_lf_std",
                    "template_id": "paop_lf_std|macro|terms2|sites0,1|bos0|ferm1",
                    "candidate_label": "op_x",
                    "support_qubits": [0, 1],
                    "support_sites": [0, 1],
                    "support_site_offsets": [0, 1],
                    "is_macro_generator": True,
                    "split_policy": "preserve",
                }
            ],
            motif_library={
                "library_version": "phase3_motif_library_v1",
                "source_tag": "unit_test",
                "source_num_sites": 2,
                "ordering": "blocked",
                "boson_encoding": "binary",
                "records": [
                    {
                        "motif_id": "motif:1",
                        "family_id": "paop_lf_std",
                        "template_id": "paop_lf_std|macro|terms2|sites0,1|bos0|ferm1",
                        "source_num_sites": 2,
                        "relative_order": 0,
                        "support_site_offsets": [0, 1],
                        "mean_theta": 0.1,
                        "mean_abs_theta": 0.1,
                        "sign_hint": 1,
                        "generator_ids": ["gen:1"],
                    }
                ],
            },
            motif_usage={"source_tag": "unit_test", "selected_count": 1},
            symmetry_mitigation={"mode": "verify_only", "executed": True, "passed": True},
            rescue_history=[{"enabled": False, "triggered": False, "reason": "disabled"}],
            pre_prune_scaffold={"operators": ["op_x", "op_y", "op_z"]},
            prune_summary={"executed": True, "accepted_count": 1},
        )

        # Read back and validate
        payload = json.loads(out_path.read_text(encoding="utf-8"))

        # Canonical replay fields
        labels, theta = _extract_adapt_operator_theta_sequence(payload)
        assert labels == ["op_x", "op_y"]
        assert np.allclose(theta, [0.1, -0.2])
        assert payload["adapt_vqe"]["ansatz_depth"] == 2
        assert payload["adapt_vqe"]["num_parameters"] == 2
        assert payload["adapt_vqe"]["pool_type"] == "pool_a"

        # Family resolution
        fam, src = _resolve_family_from_metadata(payload)
        assert fam == "pool_a"

        # Settings
        assert payload["settings"]["L"] == 2
        assert payload["settings"]["sector_n_up"] == 1
        assert payload["settings"]["sector_n_dn"] == 1
        assert payload["settings"]["adapt_pool"] == "pool_a"
        assert payload["ground_state"]["exact_energy_filtered"] == pytest.approx(-2.0)
        assert payload["ground_state"]["filtered_sector"] == {"n_up": 1, "n_dn": 1}

        # Initial state
        assert "amplitudes_qn_to_q0" in payload["initial_state"]
        assert payload["initial_state"]["nq_total"] == nq
        assert payload["continuation"]["mode"] == "phase1_v1"
        assert payload["continuation"]["optimizer_memory"]["parameter_count"] == 2
        assert payload["continuation"]["selected_generator_metadata"][0]["generator_id"] == "gen:1"
        assert payload["continuation"]["motif_library"]["records"][0]["motif_id"] == "motif:1"
        assert payload["continuation"]["symmetry_mitigation"]["mode"] == "verify_only"
        assert "seed_provenance" not in payload

    def test_write_and_read_back_with_seed_provenance(self, tmp_path: Path) -> None:
        cfg = HandoffStateBundleConfig(
            L=2,
            t=1.0,
            U=4.0,
            dv=0.0,
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            sector_n_up=1,
            sector_n_dn=1,
        )
        psi = np.zeros(1 << 6, dtype=complex)
        psi[0] = 1.0
        out_path = tmp_path / "stage_export_seed.json"
        write_handoff_state_bundle(
            path=out_path,
            psi_state=psi,
            cfg=cfg,
            source="adapt_vqe",
            exact_energy=-2.0,
            energy=-1.9,
            delta_E_abs=0.1,
            relative_error_abs=0.05,
            seed_provenance={
                "warm_ansatz": "hh_hva_ptw",
                "refine_family": "uccsd_otimes_paop_lf_std",
                "refine_family_kind": "uccsd_paop_product",
                "refine_paop_motif_families": ["paop_lf_std"],
                "refine_reps": 2,
            },
        )

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["seed_provenance"]["warm_ansatz"] == "hh_hva_ptw"
        assert payload["seed_provenance"]["refine_family"] == "uccsd_otimes_paop_lf_std"

    def test_write_with_contract_is_parseable(self, tmp_path: Path) -> None:
        cfg = HandoffStateBundleConfig(
            L=2,
            t=1.0,
            U=4.0,
            dv=0.0,
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            sector_n_up=1,
            sector_n_dn=1,
        )

        nq = 2 * 2 + 2
        psi = np.zeros(1 << nq, dtype=complex)
        psi[0] = 1.0

        out_path = tmp_path / "with_contract.json"
        write_handoff_state_bundle(
            path=out_path,
            psi_state=psi,
            cfg=cfg,
            source="A_probe_final",
            exact_energy=-2.0,
            energy=-1.9,
            delta_E_abs=0.1,
            relative_error_abs=0.05,
            adapt_operators=["op_x", "op_y"],
            adapt_optimal_point=[0.1, -0.2],
            adapt_pool_type="pool_a",
            settings_adapt_pool="pool_a",
            continuation_mode="phase1_v1",
            replay_contract={
                "contract_version": 2,
                "generator_family": {
                    "requested": "match_adapt",
                    "resolved": "paop_lf_std",
                    "resolution_source": "selected_generator_metadata.family_id",
                    "fallback_family": "full_meta",
                    "fallback_used": False,
                },
                "seed_policy_requested": "auto",
                "seed_policy_resolved": "residual_only",
                "handoff_state_kind": "prepared_state",
                "continuation_mode": "phase1_v1",
            },
        )

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert "continuation" in payload
        contract = _extract_replay_contract(payload)
        assert contract is not None
        assert contract["generator_family"]["resolved"] == "paop_lf_std"

    def test_write_sparse_bundle_without_continuation_preserves_legacy_load(self, tmp_path: Path) -> None:
        cfg = HandoffStateBundleConfig(
            L=2,
            t=1.0,
            U=4.0,
            dv=0.0,
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            sector_n_up=1,
            sector_n_dn=1,
        )

        nq = 2 * 2 + 2
        psi = np.zeros(1 << nq, dtype=complex)
        psi[0] = 1.0

        out_path = tmp_path / "legacy_shape.json"
        write_handoff_state_bundle(
            path=out_path,
            psi_state=psi,
            cfg=cfg,
            source="A_probe_final",
            exact_energy=-2.0,
            energy=-1.9,
            delta_E_abs=0.1,
            relative_error_abs=0.05,
            adapt_operators=["op_x"],
            adapt_optimal_point=[0.1],
            adapt_pool_type="pool_a",
            settings_adapt_pool="pool_a",
        )

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert "continuation" not in payload

        labels, theta = _extract_adapt_operator_theta_sequence(payload)
        assert labels == ["op_x"]
        assert np.allclose(theta, [0.1])

        fam, src = _resolve_family_from_metadata(payload)
        assert fam == "pool_a"
        assert src == "adapt_vqe.pool_type"
        assert payload["settings"]["adapt_pool"] == "pool_a"
        assert payload["ground_state"]["exact_energy_filtered"] == pytest.approx(-2.0)

        kind, provenance = _infer_handoff_state_kind(payload)
        assert kind == "prepared_state"
        assert provenance == "inferred_source"


# ---------------------------------------------------------------------------
# Provenance tests
# ---------------------------------------------------------------------------


class TestStagedExportProvenance:
    """Verify staged exports carry explicit handoff_state_kind provenance."""

    def test_staged_export_has_prepared_state_kind(self) -> None:
        payload = _make_staged_export_payload()
        assert payload["initial_state"]["handoff_state_kind"] == "prepared_state"

    def test_provenance_is_explicit(self) -> None:
        payload = _make_staged_export_payload()
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "prepared_state"
        assert src == "explicit"

    @pytest.mark.parametrize("L", [2, 3, 4, 5])
    def test_provenance_present_for_all_L(self, L: int) -> None:
        payload = _make_staged_export_payload(L=L)
        assert "handoff_state_kind" in payload["initial_state"]
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "prepared_state"
        assert src == "explicit"

    def test_legacy_payload_without_provenance_infers_from_source(self) -> None:
        """Old payloads without handoff_state_kind can be inferred from source."""
        payload = _make_staged_export_payload()
        del payload["initial_state"]["handoff_state_kind"]
        kind, src = _infer_handoff_state_kind(payload)
        # Source is "A_probe_final" which ends with "_final" -> prepared_state
        assert kind == "prepared_state"
        assert src == "inferred_source"

    def test_write_state_bundle_stamps_provenance(self, tmp_path: Path) -> None:
        """Verify write_handoff_state_bundle stamps handoff_state_kind."""
        cfg = HandoffStateBundleConfig(
            L=2,
            t=1.0,
            U=4.0,
            dv=0.0,
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            sector_n_up=1,
            sector_n_dn=1,
        )

        nq = 2 * 2 + 2
        psi = np.zeros(1 << nq, dtype=complex)
        psi[0] = 1.0

        out_path = tmp_path / "provenance_test.json"
        write_handoff_state_bundle(
            path=out_path,
            psi_state=psi,
            cfg=cfg,
            source="A_probe_final",
            exact_energy=-2.0,
            energy=-1.9,
            delta_E_abs=0.1,
            relative_error_abs=0.05,
            handoff_state_kind="prepared_state",
            adapt_operators=["op_x"],
            adapt_optimal_point=[0.1],
            adapt_pool_type="pool_a",
        )

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["initial_state"]["handoff_state_kind"] == "prepared_state"
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "prepared_state"
        assert src == "explicit"

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/time_propagation/__init__.py
```py
"""Time-propagation utilities."""

from src.quantum.time_propagation.cfqm_propagator import CFQMConfig, cfqm_step
from src.quantum.time_propagation.cfqm_schemes import get_cfqm_scheme
from src.quantum.time_propagation.local_checkpoint_fit import (
    CheckpointFitConfig,
    CheckpointFitStepResult,
    CheckpointFitTrajectoryResult,
    LocalPauliAnsatzSpec,
    build_local_pauli_ansatz_terms,
    default_chain_edges,
    fit_checkpoint_target_state,
    fit_checkpoint_trajectory,
)
from src.quantum.time_propagation.projected_real_time import (
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

__all__ = [
    "CFQMConfig",
    "CheckpointFitConfig",
    "CheckpointFitStepResult",
    "CheckpointFitTrajectoryResult",
    "ExactDrivenReferenceResult",
    "LocalPauliAnsatzSpec",
    "ProjectedRealTimeConfig",
    "ProjectedRealTimeResult",
    "build_tangent_vectors",
    "build_local_pauli_ansatz_terms",
    "cfqm_step",
    "default_chain_edges",
    "expectation_total_hamiltonian",
    "fit_checkpoint_target_state",
    "fit_checkpoint_trajectory",
    "get_cfqm_scheme",
    "run_exact_driven_reference",
    "run_projected_real_time_trajectory",
    "solve_mclachlan_step",
    "state_fidelity",
]

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/operator_pools/polaron_paop.py
```py
"""Polaron-adapted operator pool for Hubbard-Holstein (HH) ADAPT-VQE.

This module exposes a lightweight, composable PAOP pool builder using the
existing Pauli-layer operators from the repository math stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from src.quantum.hubbard_latex_python_pairs import (
    bravais_nearest_neighbor_edges,
    boson_operator,
    boson_displacement_operator,
    boson_qubits_per_site,
    jw_number_operator,
    mode_index,
    phonon_qubit_indices_for_site,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial, fermion_minus_operator, fermion_plus_operator
from src.quantum.qubitization_module import PauliTerm


@dataclass(frozen=True)
class PhononMotifSpec:
    label: str
    family: str
    poly: PauliPolynomial
    sites: tuple[int, ...]
    bonds: tuple[tuple[int, int], ...]
    uses_sq: bool


def _to_signature(poly: PauliPolynomial, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    items: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"PAOP generator has non-negligible imaginary term: {coeff}")
        items.append((str(term.pw2strng()), float(round(coeff.real, 12))))
    items.sort()
    return tuple(items)


def _clean_poly(poly: PauliPolynomial, prune_eps: float) -> PauliPolynomial:
    """Drop tiny coefficients and enforce purely-real Pauli coefficients."""
    return _prune_poly(poly, prune_eps, enforce_real=True)


def _prune_poly(poly: PauliPolynomial, prune_eps: float, *, enforce_real: bool) -> PauliPolynomial:
    """Drop tiny coefficients; optionally enforce purely-real Pauli coefficients."""
    terms = poly.return_polynomial()
    if not terms:
        return PauliPolynomial("JW")
    nq = int(terms[0].nqubit())
    cleaned = PauliPolynomial("JW")
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(prune_eps):
            continue
        if enforce_real and abs(coeff.imag) > 1e-10:
            raise ValueError(f"PAOP generator has non-negligible imaginary coefficient: {coeff}")
        coeff_out: complex | float
        if enforce_real:
            coeff_out = float(coeff.real)
        else:
            coeff_out = complex(coeff)
        cleaned.add_term(PauliTerm(nq, ps=str(term.pw2strng()), pc=coeff_out))
    cleaned._reduce()
    return cleaned


def _normalize_poly(poly: PauliPolynomial, mode: str) -> PauliPolynomial:
    mode_key = str(mode).strip().lower()
    if mode_key == "none":
        return poly
    terms = poly.return_polynomial()
    if not terms:
        return poly

    if mode_key == "maxcoeff":
        max_coeff = max(abs(complex(term.p_coeff)) for term in terms)
        if max_coeff <= 0.0:
            return poly
        return (1.0 / max_coeff) * poly

    if mode_key == "fro":
        norm = math.sqrt(sum(abs(complex(term.p_coeff)) ** 2 for term in terms))
        if norm <= 0.0:
            return poly
        return (1.0 / norm) * poly

    raise ValueError(f"Unknown PAOP normalization '{mode_key}'. Use none|fro|maxcoeff.")


def _append_operator(
    pool: list[tuple[str, PauliPolynomial]],
    label: str,
    poly: PauliPolynomial,
    split_paulis: bool,
    prune_eps: float,
) -> None:
    poly = _clean_poly(poly, prune_eps)
    if not poly.return_polynomial():
        return
    if not split_paulis:
        pool.append((label, poly))
        return

    for term_idx, term in enumerate(poly.return_polynomial()):
        coeff = complex(term.p_coeff)
        if abs(coeff) <= prune_eps:
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"PAOP generator has non-negligible imaginary coefficient: {coeff}")
        sub_label = f"{label}[{term_idx}]_{term.pw2strng()}"
        single = PauliPolynomial("JW", [PauliTerm(int(term.nqubit()), ps=str(term.pw2strng()), pc=float(coeff.real))])
        pool.append((sub_label, single))


def _mul_clean(
    left: PauliPolynomial,
    right: PauliPolynomial,
    prune_eps: float,
    *,
    enforce_real: bool = True,
) -> PauliPolynomial:
    """(AB)_clean := clean(A * B) after each nontrivial multiplication."""
    return _prune_poly(left * right, float(prune_eps), enforce_real=bool(enforce_real))


def _distance_1d(i: int, j: int, n_sites: int, periodic: bool) -> int:
    dist = abs(int(i) - int(j))
    if periodic and n_sites > 0:
        period = int(n_sites)
        dist = min(dist, period - dist)
    return int(dist)


def _word_from_qubit_letters(nq: int, letters: dict[int, str]) -> str:
    word = ["e"] * int(nq)
    for qubit, letter in letters.items():
        q = int(qubit)
        if q < 0 or q >= int(nq):
            raise ValueError(f"Qubit index {q} out of range for nq={nq}")
        idx = int(nq) - 1 - q
        word[idx] = str(letter)
    return "".join(word)


def jw_current_hop(nq: int, p: int, q: int) -> PauliPolynomial:
    r"""Build Hermitian odd hopping channel in JW form.

    J_{pq} = i (c^†_p c_q - c^†_q c_p)
          = 1/2 * (X_hi Z_{lo+1..hi-1} Y_lo - Y_hi Z_{lo+1..hi-1} X_lo)
    """
    p_i = int(p)
    q_i = int(q)
    nq_i = int(nq)
    if p_i == q_i:
        return PauliPolynomial("JW")
    if p_i < 0 or p_i >= nq_i or q_i < 0 or q_i >= nq_i:
        raise ValueError(f"jw_current_hop indices out of range: p={p_i}, q={q_i}, nq={nq_i}")

    lo = min(p_i, q_i)
    hi = max(p_i, q_i)
    z_letters = {k: "z" for k in range(lo + 1, hi)}

    xy = dict(z_letters)
    xy[hi] = "x"
    xy[lo] = "y"

    yx = dict(z_letters)
    yx[hi] = "y"
    yx[lo] = "x"

    out = PauliPolynomial("JW")
    out.add_term(PauliTerm(nq_i, ps=_word_from_qubit_letters(nq_i, xy), pc=0.5))
    out.add_term(PauliTerm(nq_i, ps=_word_from_qubit_letters(nq_i, yx), pc=-0.5))
    out._reduce()
    if p_i > q_i:
        return (-1.0) * out
    return out


def _drop_terms_with_identity_on_qubits(poly: PauliPolynomial, qubits: tuple[int, ...]) -> PauliPolynomial:
    terms = poly.return_polynomial()
    if not terms:
        return poly
    nq = int(terms[0].nqubit())
    keep = PauliPolynomial("JW")
    qidx = [int(q) for q in qubits]
    for term in terms:
        word = str(term.pw2strng())
        if all(word[nq - 1 - q] == "e" for q in qidx):
            continue
        keep.add_term(PauliTerm(nq, ps=word, pc=complex(term.p_coeff)))
    keep._reduce()
    return keep


def make_phonon_motifs(
    family: str,
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    boundary: str,
    prune_eps: float = 0.0,
    normalization: str = "none",
) -> list[PhononMotifSpec]:
    family_key = str(family).strip().lower()
    if family_key not in {"paop_lf_std", "paop_lf2_std", "paop_bond_disp_std"}:
        raise ValueError(
            "Phonon motif family must be one of paop_lf_std, paop_lf2_std, paop_bond_disp_std."
        )

    n_sites = int(num_sites)
    if n_sites <= 0:
        return []

    n_ph_max_i = int(n_ph_max)
    boson_encoding_i = str(boson_encoding)
    periodic = str(boundary).strip().lower() == "periodic"
    qpb = boson_qubits_per_site(n_ph_max_i, boson_encoding_i)
    nq = 2 * n_sites + n_sites * qpb
    phonon_qubits = tuple(range(2 * n_sites, nq))
    repr_mode = "JW"

    phonon_qubit_cache: dict[int, tuple[int, ...]] = {}
    b_cache: dict[int, PauliPolynomial] = {}
    bdag_cache: dict[int, PauliPolynomial] = {}
    p_cache: dict[int, PauliPolynomial] = {}
    delta_p_cache: dict[tuple[int, int], PauliPolynomial] = {}
    delta_p_power_cache: dict[tuple[int, int, int], PauliPolynomial] = {}
    bond_p_sum_cache: dict[tuple[int, int], PauliPolynomial] = {}

    def local_qubits(site: int) -> tuple[int, ...]:
        key = int(site)
        if key not in phonon_qubit_cache:
            phonon_qubit_cache[key] = tuple(
                phonon_qubit_indices_for_site(
                    key,
                    n_sites=n_sites,
                    qpb=qpb,
                    fermion_qubits=2 * n_sites,
                )
            )
        return phonon_qubit_cache[key]

    def b_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in b_cache:
            b_cache[key] = boson_operator(
                repr_mode,
                nq,
                local_qubits(key),
                which="b",
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
        return b_cache[key]

    def bdag_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in bdag_cache:
            bdag_cache[key] = boson_operator(
                repr_mode,
                nq,
                local_qubits(key),
                which="bdag",
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
        return bdag_cache[key]

    def p_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in p_cache:
            p_cache[key] = _clean_poly((1j * bdag_i(key)) + ((-1j) * b_i(key)), prune_eps)
        return p_cache[key]

    def delta_p_ij(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in delta_p_cache:
            delta_p_cache[key] = _clean_poly(p_i(key[0]) + ((-1.0) * p_i(key[1])), prune_eps)
        return delta_p_cache[key]

    def delta_p_power(i_site: int, j_site: int, exponent: int) -> PauliPolynomial:
        power = int(exponent)
        if power < 1:
            raise ValueError("delta_p_power exponent must be >= 1")
        key = (int(i_site), int(j_site), power)
        if key in delta_p_power_cache:
            return delta_p_power_cache[key]
        base = delta_p_ij(i_site, j_site)
        if power == 1:
            delta_p_power_cache[key] = base
            return base
        acc = base
        for _ in range(1, power):
            acc = _mul_clean(acc, base, prune_eps)
        delta_p_power_cache[key] = acc
        return acc

    def bond_p_sum(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in bond_p_sum_cache:
            bond_p_sum_cache[key] = _clean_poly(p_i(key[0]) + p_i(key[1]), prune_eps)
        return bond_p_sum_cache[key]

    motifs: list[PhononMotifSpec] = []

    def _append_motif(
        *,
        label: str,
        poly: PauliPolynomial,
        sites: tuple[int, ...],
        bonds: tuple[tuple[int, int], ...] = (),
        uses_sq: bool = False,
        drop_phonon_identity: bool = False,
    ) -> None:
        poly_out = poly
        if drop_phonon_identity:
            poly_out = _drop_terms_with_identity_on_qubits(poly_out, phonon_qubits)
        poly_out = _clean_poly(poly_out, prune_eps)
        poly_out = _normalize_poly(poly_out, normalization)
        poly_out = _clean_poly(poly_out, prune_eps)
        if not poly_out.return_polynomial():
            return
        bonds_canon = tuple(sorted({tuple(sorted((int(i), int(j)))) for i, j in bonds}))
        motifs.append(
            PhononMotifSpec(
                label=str(label),
                family=family_key,
                poly=poly_out,
                sites=tuple(sorted({int(site) for site in sites})),
                bonds=bonds_canon,
                uses_sq=bool(uses_sq),
            )
        )

    for site in range(n_sites):
        _append_motif(
            label=f"p(site={site})",
            poly=p_i(site),
            sites=(site,),
        )

    edges = bravais_nearest_neighbor_edges(n_sites, pbc=periodic)
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        _append_motif(
            label=f"delta_p({i},{j})",
            poly=delta_p_ij(i, j),
            sites=(i, j),
            bonds=((i, j),),
        )

    if family_key == "paop_lf2_std":
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_motif(
                label=f"delta_p2({i},{j})",
                poly=delta_p_power(i, j, 2),
                sites=(i, j),
                bonds=((i, j),),
                drop_phonon_identity=True,
            )

    if family_key == "paop_bond_disp_std":
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_motif(
                label=f"bond_p_sum({i},{j})",
                poly=bond_p_sum(i, j),
                sites=(i, j),
                bonds=((i, j),),
            )

    return motifs


def _make_paop_core(
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
    n_sites = int(num_sites)
    if n_sites <= 0:
        return []

    n_ph_max_i = int(n_ph_max)
    boson_encoding_i = str(boson_encoding)
    ordering_i = str(ordering)
    boundary_i = str(boundary).strip().lower()
    periodic = boundary_i == "periodic"
    total_electrons = int(num_particles[0]) + int(num_particles[1]) if num_particles else 0
    nbar = (float(total_electrons) / float(n_sites)) if total_electrons > 0 else 1.0
    if nbar <= 0.0:
        nbar = 1.0

    nq = 2 * n_sites + n_sites * boson_qubits_per_site(n_ph_max_i, boson_encoding_i)
    repr_mode = "JW"
    id_label = "e" * nq
    id_poly = PauliPolynomial(repr_mode, [PauliTerm(nq, ps=id_label, pc=1.0)])

    number_cache: dict[int, PauliPolynomial] = {}
    doublon_cache: dict[int, PauliPolynomial] = {}
    phonon_qubit_cache: dict[int, tuple[int, ...]] = {}
    b_cache: dict[int, PauliPolynomial] = {}
    bdag_cache: dict[int, PauliPolynomial] = {}
    p_cache: dict[int, PauliPolynomial] = {}
    x_cache: dict[int, PauliPolynomial] = {}
    sq_cache: dict[int, PauliPolynomial] = {}
    hopping_cache: dict[tuple[int, int], PauliPolynomial] = {}
    current_cache: dict[tuple[int, int], PauliPolynomial] = {}
    delta_p_cache: dict[tuple[int, int], PauliPolynomial] = {}
    delta_p_power_cache: dict[tuple[int, int, int], PauliPolynomial] = {}
    bond_p_sum_cache: dict[tuple[int, int], PauliPolynomial] = {}
    bond_sq_sum_cache: dict[tuple[int, int], PauliPolynomial] = {}
    pair_sq_cache: dict[tuple[int, int], PauliPolynomial] = {}
    pool: list[tuple[str, PauliPolynomial]] = []
    phonon_qubits = tuple(range(2 * n_sites, nq))

    def local_qubits(site: int) -> tuple[int, ...]:
        key = int(site)
        if key not in phonon_qubit_cache:
            phonon_qubit_cache[key] = tuple(
                phonon_qubit_indices_for_site(
                    key,
                    n_sites=n_sites,
                    qpb=boson_qubits_per_site(n_ph_max_i, boson_encoding_i),
                    fermion_qubits=2 * n_sites,
                )
            )
        return phonon_qubit_cache[key]

    def n_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in number_cache:
            up = mode_index(int(key), 0, indexing=ordering_i, n_sites=n_sites)
            down = mode_index(int(key), 1, indexing=ordering_i, n_sites=n_sites)
            n_up = jw_number_operator(repr_mode, nq, up)
            n_dn = jw_number_operator(repr_mode, nq, down)
            cached = n_up + n_dn
            number_cache[key] = cached
        return number_cache[key]

    def b_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in b_cache:
            b_cache[key] = boson_operator(
                repr_mode,
                nq,
                local_qubits(key),
                which="b",
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
        return b_cache[key]

    def bdag_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in bdag_cache:
            bdag_cache[key] = boson_operator(
                repr_mode,
                nq,
                local_qubits(key),
                which="bdag",
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
        return bdag_cache[key]

    def p_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in p_cache:
            # P = i (b^† - b)
            p_cache[key] = _clean_poly((1j * bdag_i(key)) + (-1j * b_i(key)), prune_eps)
        return p_cache[key]

    def x_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in x_cache:
            x_cache[key] = boson_displacement_operator(
                repr_mode,
                nq,
                local_qubits(key),
                n_ph_max=n_ph_max_i,
                encoding=boson_encoding_i,
            )
        return x_cache[key]

    def sq_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in sq_cache:
            b2 = _mul_clean(b_i(key), b_i(key), prune_eps, enforce_real=False)
            bdag2 = _mul_clean(bdag_i(key), bdag_i(key), prune_eps, enforce_real=False)
            sq_cache[key] = _clean_poly((1j * bdag2) + ((-1j) * b2), prune_eps)
        return sq_cache[key]

    def shifted_density(site: int) -> PauliPolynomial:
        n_site = n_i(site)
        if abs(nbar) < 1e-15:
            return n_site
        return n_site + ((-nbar) * id_poly)

    def doublon_i(site: int) -> PauliPolynomial:
        key = int(site)
        if key not in doublon_cache:
            up = mode_index(key, 0, indexing=ordering_i, n_sites=n_sites)
            down = mode_index(key, 1, indexing=ordering_i, n_sites=n_sites)
            n_up = jw_number_operator(repr_mode, nq, up)
            n_dn = jw_number_operator(repr_mode, nq, down)
            doublon_cache[key] = _mul_clean(n_up, n_dn, prune_eps)
        return doublon_cache[key]

    def k_ij(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in hopping_cache:
            hopping = PauliPolynomial(repr_mode)
            for spin in (0, 1):
                i_spin = mode_index(key[0], spin, indexing=ordering_i, n_sites=n_sites)
                j_spin = mode_index(key[1], spin, indexing=ordering_i, n_sites=n_sites)
                term_ij = fermion_plus_operator(repr_mode, nq, i_spin) * fermion_minus_operator(repr_mode, nq, j_spin)
                term_ji = fermion_plus_operator(repr_mode, nq, j_spin) * fermion_minus_operator(repr_mode, nq, i_spin)
                hopping += term_ij
                hopping += term_ji
            hopping_cache[key] = _clean_poly(hopping, prune_eps)
        return hopping_cache[key]

    def j_ij(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in current_cache:
            current = PauliPolynomial(repr_mode)
            for spin in (0, 1):
                i_spin = mode_index(key[0], spin, indexing=ordering_i, n_sites=n_sites)
                j_spin = mode_index(key[1], spin, indexing=ordering_i, n_sites=n_sites)
                current += jw_current_hop(nq, i_spin, j_spin)
            current_cache[key] = _clean_poly(current, prune_eps)
        return current_cache[key]

    def delta_p_ij(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in delta_p_cache:
            delta_p_cache[key] = _clean_poly(p_i(key[0]) + ((-1.0) * p_i(key[1])), prune_eps)
        return delta_p_cache[key]

    def delta_p_power(i_site: int, j_site: int, exponent: int) -> PauliPolynomial:
        power = int(exponent)
        if power < 1:
            raise ValueError("delta_p_power exponent must be >= 1")
        key = (int(i_site), int(j_site), power)
        if key in delta_p_power_cache:
            return delta_p_power_cache[key]
        base = delta_p_ij(i_site, j_site)
        if power == 1:
            delta_p_power_cache[key] = base
            return base
        acc = base
        for _ in range(1, power):
            acc = _mul_clean(acc, base, prune_eps)
        delta_p_power_cache[key] = acc
        return acc

    def bond_p_sum(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in bond_p_sum_cache:
            bond_p_sum_cache[key] = _clean_poly(p_i(key[0]) + p_i(key[1]), prune_eps)
        return bond_p_sum_cache[key]

    def bond_sq_sum(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in bond_sq_sum_cache:
            bond_sq_sum_cache[key] = _clean_poly(sq_i(key[0]) + sq_i(key[1]), prune_eps)
        return bond_sq_sum_cache[key]

    def pair_sq_ij(i_site: int, j_site: int) -> PauliPolynomial:
        key = (int(i_site), int(j_site))
        if key not in pair_sq_cache:
            pair_create = _mul_clean(bdag_i(key[0]), bdag_i(key[1]), prune_eps, enforce_real=False)
            pair_annih = _mul_clean(b_i(key[0]), b_i(key[1]), prune_eps, enforce_real=False)
            pair_sq_cache[key] = _clean_poly((1j * pair_create) + ((-1j) * pair_annih), prune_eps)
        return pair_sq_cache[key]

    # (A) local conditional displacement dressing
    if include_disp:
        for site in range(n_sites):
            _append_operator(
                pool,
                f"paop_disp(site={site})",
                _normalize_poly(shifted_density(site) * p_i(site), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (B) legacy local doublon dressing
    if include_doublon:
        for site in range(n_sites):
            _append_operator(
                pool,
                f"paop_dbl(site={site})",
                _normalize_poly(shifted_density(site) * doublon_i(site), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    edges = bravais_nearest_neighbor_edges(n_sites, pbc=periodic) if (include_hopdrag or include_curdrag or include_hop2 or include_curdrag3 or include_hop4 or include_bond_disp or include_hop_sq or include_pair_sq) else []

    # (C) dressed hopping K_{ij}(P_i - P_j)
    if include_hopdrag:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_operator(
                pool,
                f"paop_hopdrag({i},{j})",
                _normalize_poly(_mul_clean(k_ij(i, j), delta_p_ij(i, j), prune_eps), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (D) LF-leading odd channel J_{ij}(P_i - P_j)
    if include_curdrag:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_operator(
                pool,
                f"paop_curdrag({i},{j})",
                _normalize_poly(_mul_clean(j_ij(i, j), delta_p_ij(i, j), prune_eps), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (E) LF second-order even channel K_{ij}(P_i - P_j)^2
    if include_hop2:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            hop2_poly = _mul_clean(k_ij(i, j), delta_p_power(i, j, 2), prune_eps)
            if drop_hop2_phonon_identity:
                hop2_poly = _drop_terms_with_identity_on_qubits(hop2_poly, phonon_qubits)
            _append_operator(
                pool,
                f"paop_hop2({i},{j})",
                _normalize_poly(hop2_poly, normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (E3) LF third-order odd current channel J_{ij}(P_i - P_j)^3
    if include_curdrag3:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_operator(
                pool,
                f"paop_curdrag3({i},{j})",
                _normalize_poly(_mul_clean(j_ij(i, j), delta_p_power(i, j, 3), prune_eps), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (E4) LF fourth-order even hopping channel K_{ij}(P_i - P_j)^4
    if include_hop4:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            hop4_poly = _mul_clean(k_ij(i, j), delta_p_power(i, j, 4), prune_eps)
            hop4_poly = _drop_terms_with_identity_on_qubits(hop4_poly, phonon_qubits)
            _append_operator(
                pool,
                f"paop_hop4({i},{j})",
                _normalize_poly(hop4_poly, normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (Bdisp) bond-conditioned symmetric displacement K_{ij}(P_i + P_j)
    if include_bond_disp:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_operator(
                pool,
                f"paop_bond_disp({i},{j})",
                _normalize_poly(_mul_clean(k_ij(i, j), bond_p_sum(i, j), prune_eps), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (HSQ) hopping-conditioned local squeeze K_{ij}(S_i + S_j)
    if include_hop_sq:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_operator(
                pool,
                f"paop_hop_sq({i},{j})",
                _normalize_poly(_mul_clean(k_ij(i, j), bond_sq_sum(i, j), prune_eps), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (PSQ) two-mode phonon pair squeeze i(b_i^† b_j^† - b_i b_j)
    if include_pair_sq:
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            _append_operator(
                pool,
                f"paop_pair_sq({i},{j})",
                _normalize_poly(pair_sq_ij(i, j), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # (S) local phonon squeeze generator S_i and density-conditioned variants
    if include_sq:
        for site in range(n_sites):
            _append_operator(
                pool,
                f"paop_sq(site={site})",
                _normalize_poly(sq_i(site), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    if include_dens_sq:
        for site in range(n_sites):
            _append_operator(
                pool,
                f"paop_dens_sq(site={site})",
                _normalize_poly(_mul_clean(shifted_density(site), sq_i(site), prune_eps), normalization),
                split_paulis=split_paulis,
                prune_eps=prune_eps,
            )

    # Optional radius-R extension for cloud dressing
    if include_extended_cloud and cloud_radius >= 0:
        radius = int(cloud_radius)
        for i_site in range(n_sites):
            for j_site in range(n_sites):
                if i_site == j_site:
                    continue
                if _distance_1d(i_site, j_site, n_sites, periodic) > radius:
                    continue
                _append_operator(
                    pool,
                    f"paop_cloud_p(site={i_site}->phonon={j_site})",
                    _normalize_poly(shifted_density(i_site) * p_i(j_site), normalization),
                    split_paulis=split_paulis,
                    prune_eps=prune_eps,
                )
                if include_cloud_x:
                    _append_operator(
                        pool,
                        f"paop_cloud_x(site={i_site}->phonon={j_site})",
                        _normalize_poly(shifted_density(i_site) * x_i(j_site), normalization),
                        split_paulis=split_paulis,
                        prune_eps=prune_eps,
                    )

    if include_cloud_sq and cloud_radius >= 0:
        radius = int(cloud_radius)
        for i_site in range(n_sites):
            for j_site in range(n_sites):
                if i_site == j_site:
                    continue
                if _distance_1d(i_site, j_site, n_sites, periodic) > radius:
                    continue
                _append_operator(
                    pool,
                    f"paop_cloud_sq(site={i_site}->phonon={j_site})",
                    _normalize_poly(_mul_clean(shifted_density(i_site), sq_i(j_site), prune_eps), normalization),
                    split_paulis=split_paulis,
                    prune_eps=prune_eps,
                )

    # (F) LF doublon-conditioned phonon translation D_i p_j / D_i x_j
    if (include_doublon_translation_p or include_doublon_translation_x) and cloud_radius >= 0:
        radius = int(cloud_radius)
        for i_site in range(n_sites):
            for j_site in range(n_sites):
                if _distance_1d(i_site, j_site, n_sites, periodic) > radius:
                    continue
                if include_doublon_translation_p:
                    _append_operator(
                        pool,
                        f"paop_dbl_p(site={i_site}->phonon={j_site})",
                        _normalize_poly(doublon_i(i_site) * p_i(j_site), normalization),
                        split_paulis=split_paulis,
                        prune_eps=prune_eps,
                    )

    if (include_doublon_sq or include_doublon_translation_x) and cloud_radius >= 0:
        radius = int(cloud_radius)
        for i_site in range(n_sites):
            for j_site in range(n_sites):
                if _distance_1d(i_site, j_site, n_sites, periodic) > radius:
                    continue
                if include_doublon_sq:
                    _append_operator(
                        pool,
                        f"paop_dbl_sq(site={i_site}->phonon={j_site})",
                        _normalize_poly(_mul_clean(doublon_i(i_site), sq_i(j_site), prune_eps), normalization),
                        split_paulis=split_paulis,
                        prune_eps=prune_eps,
                    )
                if include_doublon_translation_x:
                    _append_operator(
                        pool,
                        f"paop_dbl_x(site={i_site}->phonon={j_site})",
                        _normalize_poly(doublon_i(i_site) * x_i(j_site), normalization),
                        split_paulis=split_paulis,
                        prune_eps=prune_eps,
                    )

    # Keep deterministic ordering and drop exact duplicates
    dedup: list[tuple[str, PauliPolynomial]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for label, poly in pool:
        sig = _to_signature(poly)
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append((f"{pool_name}:{label}", poly))
    return dedup


def make_pool(
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
    """Build PAOP pools for HH.

    Names accepted:
      - paop (alias to paop_std)
      - paop_min
      - paop_std
      - paop_full
      - paop_lf (alias to paop_lf_std)
      - paop_lf_std
      - paop_lf2_std
      - paop_lf3_std
      - paop_lf4_std
      - paop_lf_full
      - paop_sq_std
      - paop_sq_full
      - paop_bond_disp_std
      - paop_hop_sq_std
      - paop_pair_sq_std
    """
    mode = str(name).strip().lower()
    if mode == "paop":
        mode = "paop_std"
    if mode == "paop_lf":
        mode = "paop_lf_std"

    if mode not in {
        "paop_min",
        "paop_std",
        "paop_full",
        "paop_lf_std",
        "paop_lf2_std",
        "paop_lf3_std",
        "paop_lf4_std",
        "paop_lf_full",
        "paop_sq_std",
        "paop_sq_full",
        "paop_bond_disp_std",
        "paop_hop_sq_std",
        "paop_pair_sq_std",
    }:
        raise ValueError(
            "PAOP pool name must be one of paop, paop_min, paop_std, paop_full, "
            "paop_lf, paop_lf_std, paop_lf2_std, paop_lf3_std, paop_lf4_std, "
            "paop_lf_full, paop_sq_std, paop_sq_full, paop_bond_disp_std, paop_hop_sq_std, paop_pair_sq_std."
        )

    include_disp = True
    include_doublon = mode == "paop_full"
    include_hopdrag = mode in {
        "paop_std",
        "paop_full",
        "paop_lf_std",
        "paop_lf2_std",
        "paop_lf3_std",
        "paop_lf4_std",
        "paop_lf_full",
        "paop_sq_std",
        "paop_sq_full",
        "paop_bond_disp_std",
        "paop_hop_sq_std",
        "paop_pair_sq_std",
    }
    include_curdrag = mode in {
        "paop_lf_std",
        "paop_lf2_std",
        "paop_lf3_std",
        "paop_lf4_std",
        "paop_lf_full",
        "paop_sq_std",
        "paop_sq_full",
        "paop_bond_disp_std",
        "paop_hop_sq_std",
        "paop_pair_sq_std",
    }
    include_hop2 = mode in {"paop_lf2_std", "paop_lf3_std", "paop_lf4_std", "paop_lf_full"}
    include_curdrag3 = mode in {"paop_lf3_std", "paop_lf4_std"}
    include_hop4 = mode in {"paop_lf4_std"}
    include_bond_disp = mode in {"paop_bond_disp_std"}
    include_hop_sq = mode in {"paop_hop_sq_std"}
    include_pair_sq = mode in {"paop_pair_sq_std"}
    drop_hop2_phonon_identity = include_hop2
    include_extended = mode in {"paop_full", "paop_lf_full"}
    include_cloud_x = mode in {"paop_full", "paop_lf_full"}
    include_dbl_p = mode == "paop_lf_full"
    include_dbl_x = mode == "paop_lf_full"
    include_sq = mode in {"paop_sq_std", "paop_sq_full"}
    include_dens_sq = mode in {"paop_sq_std", "paop_sq_full"}
    include_cloud_sq = mode == "paop_sq_full"
    include_dbl_sq = mode == "paop_sq_full"
    radius = max(0, int(paop_r))
    if (include_extended or include_cloud_sq or include_dbl_sq) and radius == 0:
        radius = 1

    return _make_paop_core(
        num_sites=int(num_sites),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        num_particles=tuple(num_particles) if num_particles is not None else (),
        include_disp=include_disp,
        include_doublon=include_doublon,
        include_hopdrag=include_hopdrag,
        include_curdrag=include_curdrag,
        include_hop2=include_hop2,
        include_curdrag3=include_curdrag3,
        include_hop4=include_hop4,
        include_bond_disp=include_bond_disp,
        include_hop_sq=include_hop_sq,
        include_pair_sq=include_pair_sq,
        drop_hop2_phonon_identity=drop_hop2_phonon_identity,
        include_extended_cloud=include_extended,
        cloud_radius=radius,
        include_cloud_x=include_cloud_x,
        include_doublon_translation_p=include_dbl_p,
        include_doublon_translation_x=include_dbl_x,
        include_sq=include_sq,
        include_dens_sq=include_dens_sq,
        include_cloud_sq=include_cloud_sq,
        include_doublon_sq=include_dbl_sq,
        split_paulis=bool(paop_split_paulis),
        prune_eps=float(paop_prune_eps),
        normalization=str(paop_normalization),
        pool_name=mode,
    )

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/src/quantum/time_propagation/projected_real_time.py
```py
"""Projected real-time dynamics utilities for fixed-generator HH surrogates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.vqe_latex_python_pairs import apply_exp_pauli_polynomial


_EXPM_SPARSE_MIN_DIM: int = 64


@dataclass(frozen=True)
class ProjectedRealTimeConfig:
    t_final: float
    num_times: int
    ode_substeps: int = 4
    tangent_eps: float = 1e-6
    lambda_reg: float = 1e-8
    svd_rcond: float = 1e-12
    coefficient_tolerance: float = 1e-12
    sort_terms: bool = True


@dataclass(frozen=True)
class ProjectedRealTimeResult:
    times: np.ndarray
    theta_history: np.ndarray
    states: tuple[np.ndarray, ...]
    trajectory_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ExactDrivenReferenceResult:
    times: np.ndarray
    states: tuple[np.ndarray, ...]
    energies_total: np.ndarray
    trajectory_rows: tuple[dict[str, Any], ...]


_MATH_NORMALIZE_STATE = r"\hat{\psi}=\psi/\|\psi\|"


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    vec = np.asarray(psi, dtype=complex).reshape(-1)
    nrm = float(np.linalg.norm(vec))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return vec / nrm


_MATH_PAULI_MATRIX_EXYZ = r"P(\ell)=\bigotimes_{q=n-1}^{0}\sigma_{\ell_q}"


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = {
        "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    }
    if str(label) == "":
        raise ValueError("Pauli label must be non-empty.")
    out = mats[str(label)[0]]
    for ch in str(label)[1:]:
        out = np.kron(out, mats[ch])
    return out


_MATH_DRIVE_IS_Z_TYPE = r"\ell\ \mathrm{diagonal}\iff \forall q,\ \ell_q\in\{e,z\}"


def _is_all_z_type(label: str) -> bool:
    return all(ch in {"e", "z"} for ch in str(label))


_MATH_DRIVE_DIAGONAL = r"d(\mathrm{idx})=\sum_\ell c_\ell \prod_{q:\ell_q=z}(-1)^{((\mathrm{idx}\gg q)\&1)}"


def _build_drive_diagonal(
    drive_map: Mapping[str, complex],
    *,
    dim: int,
    nq: int,
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    idx = np.arange(int(dim), dtype=np.int64)
    diag = np.zeros(int(dim), dtype=complex)
    local_cache = {} if cache is None else cache
    for label, coeff in drive_map.items():
        coeff_c = complex(coeff)
        if abs(coeff_c) <= 1e-15:
            continue
        eig = local_cache.get(str(label), None)
        if eig is None:
            eig = np.ones(int(dim), dtype=np.float64)
            for q in range(int(nq)):
                if str(label)[int(nq) - 1 - q] == "z":
                    eig *= 1.0 - 2.0 * ((idx >> q) & 1).astype(np.float64)
            local_cache[str(label)] = eig
        diag += coeff_c * eig
    return diag


_MATH_TERM_POLY = r"G_j=\mathrm{poly}(term_j)"


def _term_polynomial(term: Any) -> Any:
    if hasattr(term, "polynomial"):
        return term.polynomial
    return term


_MATH_APPLY_ORDERED_TERMS = r"U(\theta)|\psi_0\rangle=\prod_{j=1}^{K}\widetilde{\exp}(-i\theta_j G_j)|\psi_0\rangle"


def _apply_ordered_terms(
    reference_state: np.ndarray,
    terms: Sequence[Any],
    theta: np.ndarray,
    *,
    coefficient_tolerance: float,
    sort_terms: bool,
) -> np.ndarray:
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    if int(theta_vec.size) != int(len(terms)):
        raise ValueError(
            f"theta length {int(theta_vec.size)} does not match term count {int(len(terms))}."
        )
    psi = _normalize_state(reference_state)
    for idx, term in enumerate(terms):
        psi = apply_exp_pauli_polynomial(
            psi,
            _term_polynomial(term),
            float(theta_vec[idx]),
            coefficient_tolerance=float(coefficient_tolerance),
            sort_terms=bool(sort_terms),
        )
    return _normalize_state(psi)


_MATH_TANGENT_VECTORS = r"\partial_j|\psi(\theta)\rangle=U_K\cdots U_{j+1}\,\partial_{\theta_j}(U_j)\,U_{j-1}\cdots U_1|\psi_0\rangle"


def build_tangent_vectors(
    reference_state: np.ndarray,
    terms: Sequence[Any],
    theta: np.ndarray,
    *,
    tangent_eps: float = 1e-6,
    coefficient_tolerance: float = 1e-12,
    sort_terms: bool = True,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    if int(theta_vec.size) != int(len(terms)):
        raise ValueError(
            f"theta length {int(theta_vec.size)} does not match term count {int(len(terms))}."
        )
    eps = float(tangent_eps)
    if eps <= 0.0:
        raise ValueError("tangent_eps must be > 0.")
    if not terms:
        return _normalize_state(reference_state), ()

    prefix_states: list[np.ndarray] = [_normalize_state(reference_state)]
    for idx, term in enumerate(terms):
        prefix_states.append(
            apply_exp_pauli_polynomial(
                prefix_states[-1],
                _term_polynomial(term),
                float(theta_vec[idx]),
                coefficient_tolerance=float(coefficient_tolerance),
                sort_terms=bool(sort_terms),
            )
        )

    tangents: list[np.ndarray] = []
    for idx, term in enumerate(terms):
        psi_before = prefix_states[idx]
        poly = _term_polynomial(term)
        psi_plus = apply_exp_pauli_polynomial(
            psi_before,
            poly,
            float(theta_vec[idx] + eps),
            coefficient_tolerance=float(coefficient_tolerance),
            sort_terms=bool(sort_terms),
        )
        psi_minus = apply_exp_pauli_polynomial(
            psi_before,
            poly,
            float(theta_vec[idx] - eps),
            coefficient_tolerance=float(coefficient_tolerance),
            sort_terms=bool(sort_terms),
        )
        tangent = (psi_plus - psi_minus) / (2.0 * eps)
        for tail_idx in range(idx + 1, int(len(terms))):
            tangent = apply_exp_pauli_polynomial(
                tangent,
                _term_polynomial(terms[tail_idx]),
                float(theta_vec[tail_idx]),
                coefficient_tolerance=float(coefficient_tolerance),
                sort_terms=bool(sort_terms),
            )
        tangents.append(np.asarray(tangent, dtype=complex).reshape(-1))
    return _normalize_state(prefix_states[-1]), tuple(tangents)


_MATH_SOLVE_MCLACHLAN = r"A_{ij}=\Re\langle \partial_i\psi|\partial_j\psi\rangle,\ C_i=\Im\langle \partial_i\psi|H|\psi\rangle,\ \dot{\theta}=(A+\lambda I)^+C"


def solve_mclachlan_step(
    tangents: Sequence[np.ndarray],
    hpsi: np.ndarray,
    *,
    lambda_reg: float = 1e-8,
    svd_rcond: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not tangents:
        return np.zeros(0, dtype=float), {
            "condition_number": 1.0,
            "regularization": float(lambda_reg),
            "regularization_used": bool(lambda_reg > 0.0),
            "solve_mode": "empty",
            "residual_norm": 0.0,
        }

    tangent_mat = np.column_stack([np.asarray(vec, dtype=complex).reshape(-1) for vec in tangents])
    hpsi_vec = np.asarray(hpsi, dtype=complex).reshape(-1)
    amat = np.real(np.conjugate(tangent_mat).T @ tangent_mat)
    cvec = np.imag(np.conjugate(tangent_mat).T @ hpsi_vec)
    cond = float(np.linalg.cond(amat))
    reg = float(max(0.0, lambda_reg))
    solve_mode = "solve"
    system = amat + reg * np.eye(int(amat.shape[0]), dtype=float)
    try:
        theta_dot = np.linalg.solve(system, cvec)
    except np.linalg.LinAlgError:
        solve_mode = "pinv"
        theta_dot = np.linalg.pinv(system, rcond=float(svd_rcond)) @ cvec
    theta_dot = np.asarray(theta_dot, dtype=float).reshape(-1)
    if not np.all(np.isfinite(theta_dot)):
        solve_mode = "pinv"
        theta_dot = np.asarray(
            np.linalg.pinv(system, rcond=float(svd_rcond)) @ cvec,
            dtype=float,
        ).reshape(-1)
    residual = np.asarray(system @ theta_dot - cvec, dtype=float).reshape(-1)
    return theta_dot, {
        "condition_number": float(cond),
        "regularization": float(reg),
        "regularization_used": bool(reg > 0.0),
        "solve_mode": str(solve_mode),
        "residual_norm": float(np.linalg.norm(residual)),
        "matrix_rank": int(np.linalg.matrix_rank(system)),
    }


_MATH_TOTAL_H_ACTION = r"H(t)|\psi\rangle=(H_{\mathrm{static}}+H_{\mathrm{drive}}(t))|\psi\rangle"


def _apply_total_hamiltonian(
    psi: np.ndarray,
    hmat_static: np.ndarray,
    *,
    drive_coeff_provider_exyz: Any | None,
    t_physical: float,
    drive_diag_cache: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
    hpsi = np.asarray(hmat_static, dtype=complex) @ psi_vec
    if drive_coeff_provider_exyz is None:
        return hpsi, None
    drive_map = {
        str(label): complex(coeff)
        for label, coeff in dict(drive_coeff_provider_exyz(float(t_physical))).items()
        if abs(complex(coeff)) > 1e-15
    }
    if not drive_map:
        return hpsi, None
    nq = int(round(math.log2(int(psi_vec.size))))
    if all(_is_all_z_type(label) for label in drive_map):
        diag = _build_drive_diagonal(
            drive_map,
            dim=int(psi_vec.size),
            nq=int(nq),
            cache=drive_diag_cache,
        )
        return hpsi + diag * psi_vec, np.asarray(diag, dtype=complex)

    h_drive = np.zeros((int(psi_vec.size), int(psi_vec.size)), dtype=complex)
    for label, coeff in drive_map.items():
        h_drive += complex(coeff) * _pauli_matrix_exyz(label)
    return hpsi + h_drive @ psi_vec, None


_MATH_RHS = r"\dot{\theta}(t)=\mathcal{P}_{\theta}(H(t),|\psi(\theta)\rangle)"


def _rhs_theta_dot(
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
    psi, tangents = build_tangent_vectors(
        reference_state,
        terms,
        theta,
        tangent_eps=float(tangent_eps),
        coefficient_tolerance=float(coefficient_tolerance),
        sort_terms=bool(sort_terms),
    )
    hpsi, drive_diag = _apply_total_hamiltonian(
        psi,
        hmat_static,
        drive_coeff_provider_exyz=drive_coeff_provider_exyz,
        t_physical=float(drive_t0) + float(t_rel),
        drive_diag_cache=drive_diag_cache,
    )
    theta_dot, solve_diag = solve_mclachlan_step(
        tangents,
        hpsi,
        lambda_reg=float(lambda_reg),
        svd_rcond=float(svd_rcond),
    )
    diagnostics = dict(solve_diag)
    diagnostics["state_norm"] = float(np.linalg.norm(psi))
    diagnostics["theta_norm"] = float(np.linalg.norm(np.asarray(theta, dtype=float)))
    diagnostics["theta_dot_norm"] = float(np.linalg.norm(theta_dot))
    diagnostics["drive_diag_used"] = bool(drive_diag is not None)
    return theta_dot, psi, diagnostics


_MATH_RUN_PROJECTED_RT = r"\theta_{n+1}=\mathrm{RK4}(\dot{\theta},\Delta t),\ |\psi_n\rangle=U(\theta_n)|\psi_0\rangle"


def run_projected_real_time_trajectory(
    reference_state: np.ndarray,
    terms: Sequence[Any],
    hmat_static: np.ndarray,
    *,
    config: ProjectedRealTimeConfig,
    drive_coeff_provider_exyz: Any | None = None,
    drive_t0: float = 0.0,
    theta_init: np.ndarray | None = None,
) -> ProjectedRealTimeResult:
    cfg = config
    if float(cfg.t_final) < 0.0:
        raise ValueError("t_final must be >= 0.")
    if int(cfg.num_times) < 2:
        raise ValueError("num_times must be >= 2.")
    if int(cfg.ode_substeps) < 1:
        raise ValueError("ode_substeps must be >= 1.")
    theta = (
        np.zeros(int(len(terms)), dtype=float)
        if theta_init is None
        else np.asarray(theta_init, dtype=float).reshape(-1)
    )
    if int(theta.size) != int(len(terms)):
        raise ValueError(
            f"theta_init length {int(theta.size)} does not match term count {int(len(terms))}."
        )
    times = np.linspace(0.0, float(cfg.t_final), int(cfg.num_times))
    drive_diag_cache: dict[str, np.ndarray] = {}
    states: list[np.ndarray] = []
    theta_rows: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for time_idx, t_rel in enumerate(times):
        theta_dot_now, psi_now, diag_now = _rhs_theta_dot(
            float(t_rel),
            theta,
            reference_state=np.asarray(reference_state, dtype=complex),
            terms=terms,
            hmat_static=np.asarray(hmat_static, dtype=complex),
            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
            drive_t0=float(drive_t0),
            tangent_eps=float(cfg.tangent_eps),
            lambda_reg=float(cfg.lambda_reg),
            svd_rcond=float(cfg.svd_rcond),
            coefficient_tolerance=float(cfg.coefficient_tolerance),
            sort_terms=bool(cfg.sort_terms),
            drive_diag_cache=drive_diag_cache,
        )
        states.append(np.asarray(psi_now, dtype=complex))
        theta_rows.append(np.asarray(theta, dtype=float).copy())
        rows.append(
            {
                "time": float(t_rel),
                "state_norm": float(diag_now["state_norm"]),
                "theta_norm": float(diag_now["theta_norm"]),
                "theta_dot_norm": float(diag_now["theta_dot_norm"]),
                "condition_number": float(diag_now["condition_number"]),
                "regularization": float(diag_now["regularization"]),
                "regularization_used": bool(diag_now["regularization_used"]),
                "solve_mode": str(diag_now["solve_mode"]),
                "residual_norm": float(diag_now["residual_norm"]),
                "matrix_rank": int(diag_now["matrix_rank"]),
                "drive_diag_used": bool(diag_now["drive_diag_used"]),
            }
        )
        if time_idx >= int(times.size) - 1:
            continue
        dt = float(times[time_idx + 1] - times[time_idx]) / float(cfg.ode_substeps)
        t_step = float(t_rel)
        for sub_idx in range(int(cfg.ode_substeps)):
            t_sub = t_step + float(sub_idx) * dt
            k1, _psi1, _diag1 = _rhs_theta_dot(
                t_sub,
                theta,
                reference_state=np.asarray(reference_state, dtype=complex),
                terms=terms,
                hmat_static=np.asarray(hmat_static, dtype=complex),
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                drive_t0=float(drive_t0),
                tangent_eps=float(cfg.tangent_eps),
                lambda_reg=float(cfg.lambda_reg),
                svd_rcond=float(cfg.svd_rcond),
                coefficient_tolerance=float(cfg.coefficient_tolerance),
                sort_terms=bool(cfg.sort_terms),
                drive_diag_cache=drive_diag_cache,
            )
            k2, _psi2, _diag2 = _rhs_theta_dot(
                t_sub + 0.5 * dt,
                theta + 0.5 * dt * k1,
                reference_state=np.asarray(reference_state, dtype=complex),
                terms=terms,
                hmat_static=np.asarray(hmat_static, dtype=complex),
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                drive_t0=float(drive_t0),
                tangent_eps=float(cfg.tangent_eps),
                lambda_reg=float(cfg.lambda_reg),
                svd_rcond=float(cfg.svd_rcond),
                coefficient_tolerance=float(cfg.coefficient_tolerance),
                sort_terms=bool(cfg.sort_terms),
                drive_diag_cache=drive_diag_cache,
            )
            k3, _psi3, _diag3 = _rhs_theta_dot(
                t_sub + 0.5 * dt,
                theta + 0.5 * dt * k2,
                reference_state=np.asarray(reference_state, dtype=complex),
                terms=terms,
                hmat_static=np.asarray(hmat_static, dtype=complex),
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                drive_t0=float(drive_t0),
                tangent_eps=float(cfg.tangent_eps),
                lambda_reg=float(cfg.lambda_reg),
                svd_rcond=float(cfg.svd_rcond),
                coefficient_tolerance=float(cfg.coefficient_tolerance),
                sort_terms=bool(cfg.sort_terms),
                drive_diag_cache=drive_diag_cache,
            )
            k4, _psi4, _diag4 = _rhs_theta_dot(
                t_sub + dt,
                theta + dt * k3,
                reference_state=np.asarray(reference_state, dtype=complex),
                terms=terms,
                hmat_static=np.asarray(hmat_static, dtype=complex),
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                drive_t0=float(drive_t0),
                tangent_eps=float(cfg.tangent_eps),
                lambda_reg=float(cfg.lambda_reg),
                svd_rcond=float(cfg.svd_rcond),
                coefficient_tolerance=float(cfg.coefficient_tolerance),
                sort_terms=bool(cfg.sort_terms),
                drive_diag_cache=drive_diag_cache,
            )
            theta = np.asarray(
                theta + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4),
                dtype=float,
            ).reshape(-1)

    return ProjectedRealTimeResult(
        times=np.asarray(times, dtype=float),
        theta_history=np.asarray(theta_rows, dtype=float),
        states=tuple(np.asarray(state, dtype=complex) for state in states),
        trajectory_rows=tuple(dict(row) for row in rows),
    )


_MATH_EXACT_REFERENCE = r"|\psi_{\mathrm{exact}}(t)\rangle=\prod_{k=0}^{N-1}\exp(-i\Delta t\,H(t_k))|\psi_0\rangle"


def run_exact_driven_reference(
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
    if float(t_final) < 0.0:
        raise ValueError("t_final must be >= 0.")
    if int(num_times) < 2:
        raise ValueError("num_times must be >= 2.")
    if int(reference_steps) < 1:
        raise ValueError("reference_steps must be >= 1.")
    sampling = str(time_sampling).strip().lower()
    if sampling not in {"midpoint", "left", "right"}:
        raise ValueError("time_sampling must be one of {'midpoint','left','right'}.")

    psi0 = _normalize_state(initial_state)
    h_static = np.asarray(hmat_static, dtype=complex)
    dim = int(psi0.size)
    nq = int(round(math.log2(int(dim))))
    times = np.linspace(0.0, float(t_final), int(num_times))
    drive_diag_cache: dict[str, np.ndarray] = {}

    use_sparse = int(dim) >= int(_EXPM_SPARSE_MIN_DIM)
    h_static_sparse = None
    sparse_diags = None
    expm_multiply = None
    dense_expm = None
    if use_sparse:
        from scipy.sparse import csc_matrix, diags
        from scipy.sparse.linalg import expm_multiply as scipy_expm_multiply

        h_static_sparse = csc_matrix(h_static)
        sparse_diags = diags
        expm_multiply = scipy_expm_multiply
    else:
        from scipy.linalg import expm as scipy_dense_expm

        dense_expm = scipy_dense_expm

    states: list[np.ndarray] = []
    energies: list[float] = []
    rows: list[dict[str, Any]] = []

    for t_rel in times:
        if abs(float(t_rel)) <= 1e-15:
            psi = np.asarray(psi0, dtype=complex).copy()
        elif drive_coeff_provider_exyz is None:
            evals, evecs = np.linalg.eigh(h_static)
            coeffs = np.conjugate(evecs).T @ psi0
            psi = evecs @ (np.exp(-1.0j * float(t_rel) * evals) * coeffs)
        else:
            psi = np.asarray(psi0, dtype=complex).copy()
            dt = float(t_rel) / float(reference_steps)
            for step_idx in range(int(reference_steps)):
                if sampling == "midpoint":
                    t_sample = float(drive_t0) + (float(step_idx) + 0.5) * dt
                elif sampling == "left":
                    t_sample = float(drive_t0) + float(step_idx) * dt
                else:
                    t_sample = float(drive_t0) + (float(step_idx) + 1.0) * dt
                drive_map = {
                    str(label): complex(coeff)
                    for label, coeff in dict(drive_coeff_provider_exyz(float(t_sample))).items()
                    if abs(complex(coeff)) > 1e-15
                }
                if drive_map and not all(_is_all_z_type(label) for label in drive_map):
                    h_drive = np.zeros_like(h_static)
                    for label, coeff in drive_map.items():
                        h_drive += complex(coeff) * _pauli_matrix_exyz(label)
                    h_total_dense = h_static + h_drive
                    if dense_expm is None:
                        from scipy.linalg import expm as scipy_dense_expm

                        dense_expm = scipy_dense_expm
                    psi = dense_expm(-1.0j * dt * h_total_dense) @ psi
                elif use_sparse and h_static_sparse is not None and sparse_diags is not None and expm_multiply is not None:
                    diag = (
                        _build_drive_diagonal(
                            drive_map,
                            dim=int(dim),
                            nq=int(nq),
                            cache=drive_diag_cache,
                        )
                        if drive_map
                        else np.zeros(int(dim), dtype=complex)
                    )
                    h_total_sparse = h_static_sparse if not np.any(diag) else h_static_sparse + sparse_diags(diag, format="csc")
                    psi = expm_multiply((-1.0j * dt) * h_total_sparse, psi)
                else:
                    if dense_expm is None:
                        from scipy.linalg import expm as scipy_dense_expm

                        dense_expm = scipy_dense_expm
                    diag = (
                        _build_drive_diagonal(
                            drive_map,
                            dim=int(dim),
                            nq=int(nq),
                            cache=drive_diag_cache,
                        )
                        if drive_map
                        else np.zeros(int(dim), dtype=complex)
                    )
                    h_total_dense = h_static + np.diag(diag)
                    psi = dense_expm(-1.0j * dt * h_total_dense) @ psi
        psi = _normalize_state(psi)
        hpsi_total, _drive_diag = _apply_total_hamiltonian(
            psi,
            h_static,
            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
            t_physical=float(drive_t0) + float(t_rel),
            drive_diag_cache=drive_diag_cache,
        )
        energy = float(np.real(np.vdot(psi, hpsi_total)))
        states.append(np.asarray(psi, dtype=complex))
        energies.append(float(energy))
        rows.append(
            {
                "time": float(t_rel),
                "state_norm": float(np.linalg.norm(psi)),
                "energy_total_exact": float(energy),
            }
        )

    return ExactDrivenReferenceResult(
        times=np.asarray(times, dtype=float),
        states=tuple(np.asarray(state, dtype=complex) for state in states),
        energies_total=np.asarray(energies, dtype=float),
        trajectory_rows=tuple(dict(row) for row in rows),
    )


_MATH_TOTAL_ENERGY = r"E(t)=\Re\langle\psi|H(t)|\psi\rangle"


def expectation_total_hamiltonian(
    psi: np.ndarray,
    hmat_static: np.ndarray,
    *,
    drive_coeff_provider_exyz: Any | None = None,
    t_physical: float = 0.0,
) -> float:
    hpsi, _drive_diag = _apply_total_hamiltonian(
        psi,
        hmat_static,
        drive_coeff_provider_exyz=drive_coeff_provider_exyz,
        t_physical=float(t_physical),
        drive_diag_cache={},
    )
    return float(np.real(np.vdot(np.asarray(psi, dtype=complex).reshape(-1), hpsi)))


_MATH_STATE_FIDELITY = r"F(\psi,\phi)=|\langle\psi|\phi\rangle|^2"


def state_fidelity(psi_left: np.ndarray, psi_right: np.ndarray) -> float:
    left = _normalize_state(psi_left)
    right = _normalize_state(psi_right)
    return float(abs(np.vdot(left, right)) ** 2)


__all__ = [
    "ExactDrivenReferenceResult",
    "ProjectedRealTimeConfig",
    "ProjectedRealTimeResult",
    "build_tangent_vectors",
    "expectation_total_hamiltonian",
    "run_exact_driven_reference",
    "run_projected_real_time_trajectory",
    "solve_mclachlan_step",
    "state_fidelity",
]

```
</file_contents>
<meta prompt 1 = "[Architect]">
You are producing an implementation-ready technical plan. The implementer will work from your plan without asking clarifying questions, so every design decision must be resolved, every touched component must be identified, and every behavioral change must be specified precisely.

Your job:
1. Analyze the requested change against the provided code — identify the relevant architecture, constraints, data flow, and extension points.
2. Decide whether this is best solved by a targeted change or a broader refactor, and justify that decision.
3. Produce a plan detailed enough that an engineer can implement it file-by-file without making design decisions of their own.

Hard constraints:
- Do not write production code, patches, diffs, or copy-paste-ready implementations.
- Stay in analysis and architecture mode only.
- Use illustrative snippets, interface shapes, sample signatures, state/data shapes, or pseudocode when they communicate the design more precisely than prose. Keep them partial — enough to remove ambiguity, not enough to copy-paste.

─── ANALYSIS ───

Current-state analysis (always include):
- Map the existing responsibilities, type relationships, ownership, data flow, and mutation points relevant to the request.
- Identify existing code that should be reused or extended — never duplicate what already exists without justification.
- Note hard constraints: API contracts, protocol conformances, state ownership rules, thread/actor isolation, persistence schemas, UI update mechanisms.
- When multiple subsystems interact, trace the call chain end-to-end and identify each transformation boundary.

─── DESIGN ───

Design standards — apply uniformly to every aspect of the plan:

1. New and modified components/types: For each, specify:
   - The name, kind (for example: class, interface, enum, record, service, module, controller), and why that kind fits the codebase and language.
   - The fields/properties/state it owns, including data shape, mutability, and ownership/lifecycle semantics.
   - Key callable interfaces or signatures, including inputs, outputs, and whether execution is synchronous/asynchronous or can fail.
   - Contracts it implements, extends, composes with, or depends on.
   - For closed sets of variants (for example enums, tagged unions, discriminated unions): all cases/variants and any attached data.
   - Where the component lives (file path) and who creates/owns its instances.

2. State and data flow: For each state change the plan introduces or modifies:
   - What triggers the change (user action, callback, notification, timer, stream event).
   - The exact path the data travels: source → transformations → destination.
   - Thread/actor/queue context at each step.
   - How downstream consumers observe the change (published property, delegate, notification, binding, callback).
   - What happens if the change arrives out of order, is duplicated, or is dropped.

3. API and interface changes: For each modified public/internal interface:
   - The before and after signatures (or new signature if additive).
   - Every call site that must be updated, grouped by file.
   - Backward-compatibility strategy if the interface is used by external consumers or persisted data.

4. Persistence and serialization: When the plan touches stored data:
   - Schema changes with exact field names, types, and defaults.
   - Migration strategy: how existing data is read, transformed, and re-persisted.
   - What happens when new code reads old data and when old code reads new data (if rollback is possible).

5. Concurrency and lifecycle:
   - Specify the execution model and safety boundaries for each new/modified component: thread affinity, event-loop/runtime constraints, isolation boundaries, queue/worker discipline, or thread-safety expectations as applicable.
   - Identify potential races, leaked references/resources, or lifecycle mismatches introduced by the change.
   - When operations are asynchronous, specify cancellation/abort behavior and what state remains after interruption.

6. Error handling and edge cases:
   - For each operation that can fail, specify what failures are possible and how they propagate.
   - Describe degraded-mode behavior: what the user sees, what state is preserved, what recovery is available.
   - Identify boundary conditions: empty collections, missing/null/optional values, first-run states, interrupted operations.

7. Algorithmic and logic-heavy work (include whenever the change involves non-trivial control flow, state machines, data transformations, or performance-sensitive paths):
   - Describe the algorithm step-by-step: inputs, outputs, invariants, and data structures.
   - Cover edge cases, failure modes, and performance characteristics (time/space complexity if relevant).
   - Explain why this approach over the most plausible alternatives.

8. Avoid unnecessary complexity:
   - Do not add layers, abstractions, or indirection without a concrete benefit identified in the plan.
   - Do not create parallel code paths — unify where possible.
   - Reuse existing patterns unless those patterns are themselves the problem.

─── OUTPUT ───

Structure your response as:

1. **Summary** — One paragraph: what changes, why, and the high-level approach.

2. **Current-state analysis** — How the relevant code works today. Trace the data/control flow end-to-end. Identify what is reusable and what is blocking.

3. **Design** — The core of the plan. Apply every applicable standard from above. Organize by logical component or subsystem, not by standard number. Each component section should cover types, state flow, interfaces, persistence, concurrency, and error handling as relevant to that component.

4. **File-by-file impact** — For every file that changes, list:
   - What changes (added/modified/removed types, methods, properties).
   - Why (which design decision drives this change).
   - Dependencies on other changes in this plan (ordering constraints).

5. **Trade-offs and alternatives** — What was considered and rejected, and why. Include the cost/benefit of the chosen approach vs. the runner-up.

6. **Risks and migration** — Breaking changes, rollback concerns, data migration, feature flags, and incremental delivery strategy if the change is large.

7. **Implementation order** — A numbered sequence of steps. Each step should be independently compilable and testable where possible. Call out steps that must be atomic (landed together).

Response discipline:
- Be specific to the provided code — reference actual type names, file paths, method names, and property names.
- Make every assumption explicit.
- Flag unknowns that must be validated during implementation, with a suggested validation approach.
- When a design decision has a non-obvious rationale, explain it in one sentence.
- Do not pad with generic advice. Every sentence should convey information the implementer needs.

Please proceed with your analysis based on the following <user instructions>
</meta prompt 1>
<user_instructions>
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
