# Artifact Manifest

Bundle root:

`/root/DL-project/lean_single_step_formalization/experiments/reports/local_certification_report_20260604`

## Report

- `report.md`: consolidated report written on 2026-06-04.
- `manual_audit_details.md`: case-level manual audit details, including false-invalid interpretation.

## Previous Docs

- `docs/research_report_adaptive_adversarial_gv.md`: previous adaptive adversarial GV report.
- `docs/local_formalization_audit_report.md`: previous local formalization audit report.
- `docs/manual_local_validity_labels_codex55_v2.md`: manual local-validity labels.
- `docs/manual_local_validity_labels_codex55_v2.jsonl`: machine-readable manual labels.

## Prompts

- `prompts/adaptive_adversarial_gv_v2`: current v2 prompt folder.

Important prompt files:

- `step_decompose.md`: no `implicit_dependency`; premises must come from explicit sources.
- `generator_formalize.md`: faithful generator; reports compile ok/fail.
- `verifier_review_compile_ok.md`: compile-ok branch verifier.
- `verifier_review_compile_fail.md`: compile-fail branch verifier.
- `generator_repair.md`: repair prompt.
- `annotation_alignment.md`: annotation alignment scoring.

## Scripts

- `scripts/run_adversarial_game_gv_v2.py`: v2 GV runner.
- `scripts/run_stepd_gv_alignment_batches_v2.py`: OPC false-step v2 batch/alignment runner.
- `scripts/build_opc_positive_stepd_controls.py`: positive-control construction runner.
- `scripts/run_positive_gv_v2_no_baseline.py`: positive-control GV runner without baseline.
- `scripts/run_structured_stepd_hacker_gv_v2.py`: structured hacker runner.
- `scripts/run_missing_premise_usecase_gv_v2.py`: missing-premise use-case runner.

## Runs

Primary positive-control artifacts:

- `runs/opc_positive_stepd_controls_codex55_high_50_002`
- `runs/positive_gv_v2_no_baseline_codex55_high_50_001`

Primary false-step comparison:

- `runs/codex55_high_opc_stepd_gv_v2_50cases_001`

Missing-premise use cases:

- `runs/missing_premise_usecase_gv_v2_codex55_high_001`
- `runs/missing_premise_usecase_gv_v2_codex55_high_10groups_001`

Structured hacker pre-fix:

- `runs/structured_stepd_hacker_gv_v2_codex55_high_attached_case0_001`
- `runs/structured_stepd_hacker_gv_v2_codex55_high_attached_case1_001`
- `runs/structured_stepd_hacker_gv_v2_codex55_high_attached_case2_001`

Structured hacker noimplicit:

- `runs/structured_stepd_hacker_gv_v2_noimplicit_codex55_high_case0_001`
- `runs/structured_stepd_hacker_gv_v2_noimplicit_codex55_high_case1_001`
- `runs/structured_stepd_hacker_gv_v2_noimplicit_codex55_high_case2_001`
- `runs/structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset3_001`
- `runs/structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset4_001`
- `runs/structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset5_001`
- `runs/structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset6_001`
- `runs/structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset7_001`
- `runs/structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset8_001`
- `runs/structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset9_001`
- `runs/structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset10_001`
- `runs/structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset11_001`
- `runs/structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset12_001`

Partial model comparison:

- `runs/deepseek_v4pro_thinking_opc_stepd_gv_v2_50cases_001`
