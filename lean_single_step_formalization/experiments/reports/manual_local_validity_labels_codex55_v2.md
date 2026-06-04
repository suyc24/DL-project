# Manual Local Validity Labels

Run: `codex55_high_opc_stepd_gv_v2_50cases_001`

Scope: all Codex cases where `gv_v2` is `valid` or the automatic annotation
alignment marked the case as `weak`.

## Summary

| label | count |
|---|---:|
| `local_valid_global_valid` | 1 |
| `local_valid_global_mismatch` | 1 |
| `underformalized_assumption` | 1 |
| `metric_mismatch` | 2 |
| `true_system_error` | 0 |

Interpretation: none of the five audited cases is a clear local-formalization
system error. The weak scores mostly come from annotation-metric mismatch or
from the fact that local validity is different from global problem faithfulness.

## Labels

| case | GV | manual label | local judgment | note |
|---|---|---|---|---|
| `003_3_OPC_best_of_n_BMOSL_2018_12_13` | invalid | `metric_mismatch` | invalid | GV catches a concrete inequality-transfer error; annotation emphasizes non-sharpness/equality-case failure. |
| `004_4_OPC_best_of_n_BMOSL_2019_9_19` | valid | `underformalized_assumption` | valid under assumed tangent-chord theorem | Local step is faithful if the exact tangent-chord theorem is accepted; orientation applicability is not independently checked. |
| `005_1_OPC_best_of_n_BMOSL_2019_9_21` | valid | `local_valid_global_valid` | valid under directed angles mod pi | The step_d convention uses directed angles modulo pi, so GV valid is correct locally. |
| `005_5_OPC_best_of_n_BMOSL_2017_5_25` | valid | `local_valid_global_mismatch` | valid | Lean proves the local `mth powers` proposition; original problem needs `powers of m`. This is a bridge failure, not local Lean failure. |
| `010_4_OPC_best_of_n_BMOSL_2017_22_49` | invalid | `metric_mismatch` | invalid | GV gives a counterexample to the convexity maximization claim; annotation says computational steps were skipped. |

## Conclusion

For the Codex 50-case run, the manual audit supports the local-formalization
claim: Lean is useful because it exposes what local proposition is being checked
and whether that proposition is valid. The automatic annotation score is not
enough by itself, because it penalizes locally valid steps and different but
mathematically meaningful invalidity diagnoses.
