# Lean Local Formalization Audit Report

Date: 2026-06-04

## Takeaway

Lean is useful here because it turns an ambiguous natural-language proof step
into an explicit local mathematical object: variables, assumptions, conclusion,
and dependency on external lemmas. This does not prove the whole original
solution correct. It tells us whether the selected target step, as decomposed by
`step_d`, is locally faithful and satisfiable.

The important distinction is:

- **Local validity**: the proposition constructed from `target_step` and
  `step_decomposition` is mathematically valid under its stated setting.
- **Global faithfulness**: that local proposition also serves the original
  problem statement without semantic drift.
- **Annotation alignment**: the verifier diagnosis matches the OPC human
  invalid annotation.

Current annotation scoring only measures the third item. It can mark a good
local formalization as weak when the selected target is locally valid but
globally mismatched, or when Lean finds a different concrete local error than
the human note.

## Completed Codex Result

Run:
`lean_single_step_formalization/experiments/runs/codex55_high_opc_stepd_gv_v2_50cases_001`

| metric | value |
|---|---:|
| cases | 50 |
| baseline invalid | 48 |
| GV v2 invalid | 47 |
| GV v2 valid | 3 |
| baseline annotation avg | 8.38 |
| GV v2 annotation avg | 8.04 |
| weak cases | 5 |

The only baseline/GV verdict disagreement is
`005_5_OPC_best_of_n_BMOSL_2017_5_25`: baseline says invalid, GV says valid.
Under the local-validity interpretation, GV is preferable: Lean constructs the
local `mth powers` interpolation proposition, while the true problem is the
global mismatch with `perfect powers of m`.

## Weak / Valid Audit

Manual labels are recorded in:

- `manual_local_validity_labels_codex55_v2.md`
- `manual_local_validity_labels_codex55_v2.jsonl`

| case | GV verdict | classification | interpretation |
|---|---|---|---|
| `003_3_OPC_best_of_n_BMOSL_2018_12_13` | invalid | metric mismatch | GV finds a concrete invalid inequality transfer. Human annotation says the inequality is not sharp enough for equality. Both identify invalidity, but diagnosis differs, so annotation alignment undercounts GV. |
| `004_4_OPC_best_of_n_BMOSL_2019_9_19` | valid | underformalized assumption | Lean accepts the tangent-chord step because the exact tangent-chord equality is passed as an external theorem. This exposes the dependency, but does not independently verify the geometric orientation. |
| `005_1_OPC_best_of_n_BMOSL_2019_9_21` | valid | local valid under convention | Step decomposition uses directed angles modulo pi. Lean proves the directed-angle version. Human annotation appears to use supplementary ordinary-angle semantics. |
| `005_5_OPC_best_of_n_BMOSL_2017_5_25` | valid | local valid, global mismatch | Target step says `mth powers` and Lean proves that local proposition. Original problem needs `powers of m`. This is not a local Lean failure; it shows the need for a global bridge check. |
| `010_4_OPC_best_of_n_BMOSL_2017_22_49` | invalid | metric mismatch | GV gives a concrete counterexample to the convexity maximization claim. Human annotation only says computational steps were skipped. The Lean-aided diagnosis is more specific than the annotation. |

Summary of the audit:

- Manual labels: `local_valid_global_valid` 1, `local_valid_global_mismatch`
  1, `underformalized_assumption` 1, `metric_mismatch` 2,
  `true_system_error` 0.
- `005_5` is the cleanest evidence that Lean separates local correctness from
  global semantic faithfulness.
- `005_1` shows the importance of angle convention.
- `004_4` shows a real limitation: if the key external theorem is assumed in
  exactly the desired form, Lean verifies faithfulness to the local theorem
  rather than the theorem's geometric applicability.
- `003_3` and `010_4` show annotation-metric mismatch: GV can find a valid
  local reason for invalidity that is not the human annotator's stated reason.

## Preliminary DeepSeek Result

Run still in progress:
`lean_single_step_formalization/experiments/runs/deepseek_v4pro_thinking_opc_stepd_gv_v2_50cases_001`

Current completed case-result files at the time of this report: 19/50.

| metric | value |
|---|---:|
| completed cases | 19 |
| baseline invalid | 19 |
| GV v2 invalid | 16 |
| GV v2 valid | 3 |
| baseline annotation avg | 8.84 |
| GV v2 annotation avg | 7.63 |
| weak cases | 4 |

DeepSeek preliminary weak/valid cases:

- `002_2_OPC_best_of_n_USAMO_2015_5_7`
- `003_3_OPC_best_of_n_BMOSL_2018_12_13`
- `003_5_OPC_best_of_n_BMOSL_2019_9_15`
- `004_2_OPC_best_of_n_BMOSL_2019_9_17`

The DeepSeek run should be treated as model-comparison evidence only after all
50 cases finish.

## What This Supports

The strongest claim we can make is:

> Lean-assisted GV makes the local proof obligation auditable. It can tell
> whether a selected step is locally formalizable, whether the generated Lean is
> faithful to that local step, and whether a compile failure corresponds to a
> real mathematical obstruction.

The claim we should not make is:

> Lean proves the entire original proof correct or incorrect.

The system is currently best described as a **local formalization and error
localization tool**, not a full proof verifier.

## Next Experiments

1. **Manual local-validity audit**
   Label all `gv_valid` and `weak` cases as:
   `local_valid_global_valid`, `local_valid_global_mismatch`,
   `underformalized_assumption`, `metric_mismatch`, or `true_system_error`.

2. **Bridge-obligation variant**
   Add a second verifier mode that checks whether the local target proposition
   actually satisfies the original problem requirement. This should flip cases
   like `mth powers` from local-valid to global-invalid.

3. **External theorem audit**
   For compile-ok geometry cases, inspect whether Lean proved the key geometry
   relation or accepted it as an assumed theorem. This directly addresses cases
   like tangent-chord orientation.

4. **Model comparison**
   Finish the DeepSeek 50-case run and compare Codex vs DeepSeek on:
   valid/invalid distribution, weak count, compile-ok rate, repair success, and
   runtime.

5. **Metric revision**
   Report both annotation alignment and local-validity score. Annotation
   alignment alone is not a fair measure for this task.
