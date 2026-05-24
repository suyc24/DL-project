# v2.5 Hazard Base/Big/Ensemble Comparison

Hard-dev common rows; clean eval still held out.

| score | budget mode | FP/TP budget | threshold | recall | step FP/TP | first-trigger FP/TP | first-trigger precision | correct false-stop | pre-FHIS false-stop | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| big_platt | step | 2.0 | 0.1121 | 0.740 | 2.000 | 1.118 | 0.472 | 0.579 | 0.283 | 188 | 376 | 66 |
| big_platt | first_trigger | 2.0 | 0.0396 | 0.898 | 2.855 | 1.973 | 0.336 | 0.824 | 0.469 | 228 | 651 | 26 |
| base_platt | step | 2.0 | 0.0679 | 0.811 | 1.995 | 1.347 | 0.426 | 0.604 | 0.370 | 206 | 411 | 48 |
| base_platt | first_trigger | 2.0 | 0.0541 | 0.870 | 2.326 | 2.000 | 0.333 | 0.717 | 0.488 | 221 | 514 | 33 |
| avg_platt | step | 2.0 | 0.0899 | 0.807 | 1.956 | 1.216 | 0.451 | 0.635 | 0.331 | 205 | 401 | 49 |
| avg_platt | first_trigger | 2.0 | 0.0478 | 0.906 | 2.726 | 1.892 | 0.346 | 0.761 | 0.480 | 230 | 627 | 24 |
| max_platt | step | 2.0 | 0.1214 | 0.791 | 1.990 | 1.168 | 0.461 | 0.610 | 0.319 | 201 | 400 | 53 |
| max_platt | first_trigger | 2.0 | 0.0575 | 0.898 | 2.689 | 1.944 | 0.340 | 0.761 | 0.484 | 228 | 613 | 26 |
| min_platt | step | 2.0 | 0.0550 | 0.803 | 2.000 | 1.357 | 0.424 | 0.597 | 0.362 | 204 | 408 | 50 |
| min_platt | first_trigger | 2.0 | 0.0396 | 0.898 | 2.702 | 1.945 | 0.340 | 0.805 | 0.469 | 228 | 616 | 26 |
| prod_or | step | 2.0 | 0.1728 | 0.807 | 1.956 | 1.216 | 0.451 | 0.635 | 0.331 | 205 | 401 | 49 |
| prod_or | first_trigger | 2.0 | 0.0934 | 0.906 | 2.735 | 1.901 | 0.345 | 0.767 | 0.480 | 230 | 629 | 24 |

## Takeaway

- `base_platt` is the best single model for strict step-level FP/TP<=2: recall 0.811 vs `big_platt` 0.740.
- `avg_platt` is the best current user-like first-trigger FP/TP<=2 high-recall point: recall 0.906, first-trigger FP/TP 1.892, correct false-stop 0.761.
- Simple ensembles improve the first-trigger operating point, but still stop too many correct traces for an online policy.
- Current best operating choices: `base_platt` threshold 0.0679 for strict step-level budget, or `avg_platt` threshold 0.0478 for first-trigger budget.
