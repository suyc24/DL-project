# v2.5 Hazard All Tradeoff Summary

Hard-dev common rows across all pulled models. Clean eval remains held out and unused for threshold selection.

## Individual Model Quality

| model | best epoch | val AUPRC | hard AUROC | hard AUPRC | Brier |
|---|---:|---:|---:|---:|---:|
| big_platt | 4 | 0.8237 | 0.8167 | 0.5774 | 0.1013 |
| base_platt | 13 | 0.8523 | 0.8301 | 0.5790 | 0.1016 |
| trade_neg3_pos5 | 42 | 0.8584 | 0.8058 | 0.5385 | 0.1053 |
| trade_neg4_pos4 | 25 | 0.8556 | 0.8090 | 0.5288 | 0.1068 |

## FP/TP <= 2 Operating Points

| score | mode | threshold | recall | step FP/TP | first-trigger FP/TP | first-trigger precision | correct false-stop | pre-FHIS false-stop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| big_platt | step | 0.1121 | 0.740 | 2.000 | 1.118 | 0.472 | 0.579 | 0.283 |
| big_platt | first_trigger | 0.0396 | 0.898 | 2.855 | 1.973 | 0.336 | 0.824 | 0.469 |
| base_platt | step | 0.0679 | 0.811 | 1.995 | 1.347 | 0.426 | 0.604 | 0.370 |
| base_platt | first_trigger | 0.0541 | 0.870 | 2.326 | 2.000 | 0.333 | 0.717 | 0.488 |
| trade_neg3_pos5 | step | 0.0792 | 0.732 | 1.914 | 1.119 | 0.472 | 0.516 | 0.315 |
| trade_neg3_pos5 | first_trigger | 0.0573 | 0.890 | 2.765 | 1.955 | 0.338 | 0.792 | 0.480 |
| trade_neg4_pos4 | step | 0.0709 | 0.768 | 1.985 | 1.316 | 0.432 | 0.604 | 0.358 |
| trade_neg4_pos4 | first_trigger | 0.0607 | 0.835 | 2.557 | 1.961 | 0.338 | 0.717 | 0.469 |
| avg_big_base | step | 0.0899 | 0.807 | 1.956 | 1.216 | 0.451 | 0.635 | 0.331 |
| avg_big_base | first_trigger | 0.0478 | 0.906 | 2.726 | 1.892 | 0.346 | 0.761 | 0.480 |
| avg_all4 | step | 0.0762 | 0.839 | 1.986 | 1.234 | 0.448 | 0.642 | 0.350 |
| avg_all4 | first_trigger | 0.0534 | 0.913 | 2.685 | 1.893 | 0.346 | 0.780 | 0.480 |
| max_all4 | step | 0.1227 | 0.807 | 1.990 | 1.198 | 0.455 | 0.629 | 0.327 |
| max_all4 | first_trigger | 0.0667 | 0.894 | 2.678 | 1.972 | 0.336 | 0.774 | 0.480 |
| min_all4 | step | 0.0581 | 0.756 | 1.964 | 1.197 | 0.455 | 0.572 | 0.311 |
| min_all4 | first_trigger | 0.0396 | 0.898 | 2.702 | 1.945 | 0.340 | 0.805 | 0.469 |

## Correct-False-Stop Caps at First-Trigger FP/TP <= 2

| score | cap | threshold | recall | first-trigger FP/TP | correct false-stop | pre-FHIS false-stop |
|---|---:|---:|---:|---:|---:|---:|
| avg_big_base | 0.5 | 0.1166 | 0.760 | 0.984 | 0.497 | 0.287 |
| max_all4 | 0.5 | 0.1807 | 0.756 | 0.953 | 0.484 | 0.283 |
| avg_all4 | 0.5 | 0.1179 | 0.744 | 0.882 | 0.465 | 0.264 |
| avg_all4 | 0.6 | 0.0839 | 0.811 | 1.159 | 0.585 | 0.331 |
| max_all4 | 0.6 | 0.1350 | 0.803 | 1.092 | 0.597 | 0.307 |
| avg_big_base | 0.6 | 0.0965 | 0.795 | 1.144 | 0.597 | 0.315 |
| base_platt | 0.7 | 0.0550 | 0.866 | 1.923 | 0.698 | 0.476 |
| avg_big_base | 0.7 | 0.0616 | 0.862 | 1.475 | 0.686 | 0.398 |
| max_all4 | 0.7 | 0.0850 | 0.854 | 1.471 | 0.698 | 0.394 |
| avg_all4 | 0.8 | 0.0534 | 0.913 | 1.893 | 0.780 | 0.480 |
| avg_big_base | 0.8 | 0.0478 | 0.906 | 1.892 | 0.761 | 0.480 |
| max_all4 | 0.8 | 0.0667 | 0.894 | 1.972 | 0.774 | 0.480 |

## Takeaway

- The negative-weight tradeoff runs lower false stops at moderate thresholds, but they do not beat `avg_big_base` for high-recall first-trigger FP/TP<=2.
- Best current first-trigger FP/TP<=2 point remains `avg_big_base`: recall 0.906, correct false-stop 0.761.
- Best strict step-level FP/TP<=2 point remains `base_platt`: recall 0.811, correct false-stop 0.604.
- If correct false-stop must stay <=0.7, the best recall is about 0.866; reaching 0.90 recall currently requires accepting about 0.76 correct false-stop on this hard-dev split.
