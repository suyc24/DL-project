# v2.5 Trace-Loss Comparison Summary

Hard-dev comparison on common rows across base, tradeoff, and trace-loss models. Clean eval remains held out and unused for threshold selection.

- Common rows: 1552
- Common traces: 376
- FHIS-positive step rows: 254
- Negative step rows: 1298

## Individual / Ensemble Quality

| score | kind | best epoch | val AUPRC | hard AUROC | hard AUPRC | Brier |
|---|---|---:|---:|---:|---:|---:|
| avg_all4 | ensemble |  |  | 0.8316 | 0.5774 | 0.1005 |
| avg_all6 | ensemble |  |  | 0.8258 | 0.5610 | 0.1032 |
| avg_base_trace2 | ensemble |  |  | 0.8186 | 0.5440 | 0.1065 |
| avg_big_base | ensemble |  |  | 0.8316 | 0.5924 | 0.0987 |
| avg_trace2 | ensemble |  |  | 0.8052 | 0.5155 | 0.1115 |
| max_all6 | ensemble |  |  | 0.8282 | 0.5718 | 0.1070 |
| min_all6 | ensemble |  |  | 0.7970 | 0.5087 | 0.1154 |
| base_platt | model | 13 | 0.8523 | 0.8301 | 0.5790 | 0.1016 |
| big_platt | model | 4 | 0.8237 | 0.8167 | 0.5774 | 0.1013 |
| trace_c05_pre03_pos05 | model | 10 | 0.8580 | 0.8086 | 0.5247 | 0.1097 |
| trace_c10_pre05_pos08 | model | 8 | 0.8613 | 0.7964 | 0.4969 | 0.1170 |
| trade_neg3_pos5 | model | 42 | 0.8584 | 0.8058 | 0.5385 | 0.1053 |
| trade_neg4_pos4 | model | 25 | 0.8556 | 0.8090 | 0.5288 | 0.1068 |

## FP/TP <= 2 Operating Points

| score | mode | threshold | recall | step FP/TP | first-trigger FP/TP | first-trigger precision | correct false-stop | pre-FHIS false-stop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| avg_all4 | first_trigger | 0.0534 | 0.913 | 2.685 | 1.893 | 0.346 | 0.780 | 0.480 |
| avg_big_base | first_trigger | 0.0478 | 0.906 | 2.726 | 1.892 | 0.346 | 0.761 | 0.480 |
| avg_all6 | first_trigger | 0.0583 | 0.898 | 2.610 | 1.944 | 0.340 | 0.748 | 0.484 |
| min_all6 | first_trigger | 0.0396 | 0.898 | 2.702 | 1.945 | 0.340 | 0.805 | 0.469 |
| big_platt | first_trigger | 0.0396 | 0.898 | 2.855 | 1.973 | 0.336 | 0.824 | 0.469 |
| max_all6 | first_trigger | 0.0810 | 0.894 | 2.533 | 1.754 | 0.363 | 0.736 | 0.461 |
| trade_neg3_pos5 | first_trigger | 0.0573 | 0.890 | 2.765 | 1.955 | 0.338 | 0.792 | 0.480 |
| avg_base_trace2 | first_trigger | 0.0609 | 0.882 | 2.455 | 1.951 | 0.339 | 0.686 | 0.496 |
| trace_c10_pre05_pos08 | first_trigger | 0.0660 | 0.882 | 2.558 | 2.000 | 0.333 | 0.704 | 0.504 |
| avg_trace2 | first_trigger | 0.0633 | 0.870 | 2.489 | 1.990 | 0.334 | 0.673 | 0.504 |
| base_platt | first_trigger | 0.0541 | 0.870 | 2.326 | 2.000 | 0.333 | 0.717 | 0.488 |
| trace_c05_pre03_pos05 | first_trigger | 0.0589 | 0.866 | 2.368 | 1.900 | 0.345 | 0.610 | 0.500 |
| trade_neg4_pos4 | first_trigger | 0.0607 | 0.835 | 2.557 | 1.961 | 0.338 | 0.717 | 0.469 |
| avg_all4 | step | 0.0762 | 0.839 | 1.986 | 1.234 | 0.448 | 0.642 | 0.350 |
| avg_all6 | step | 0.0799 | 0.819 | 1.976 | 1.183 | 0.458 | 0.597 | 0.339 |
| base_platt | step | 0.0679 | 0.811 | 1.995 | 1.347 | 0.426 | 0.604 | 0.370 |
| max_all6 | step | 0.1350 | 0.811 | 1.961 | 1.148 | 0.465 | 0.610 | 0.323 |
| avg_base_trace2 | step | 0.0771 | 0.807 | 1.971 | 1.163 | 0.462 | 0.547 | 0.346 |
| avg_big_base | step | 0.0899 | 0.807 | 1.956 | 1.216 | 0.451 | 0.635 | 0.331 |
| avg_trace2 | step | 0.0773 | 0.803 | 1.966 | 1.047 | 0.489 | 0.503 | 0.335 |
| trace_c10_pre05_pos08 | step | 0.0856 | 0.799 | 1.926 | 0.947 | 0.514 | 0.465 | 0.315 |
| min_all6 | step | 0.0541 | 0.783 | 1.995 | 1.265 | 0.442 | 0.585 | 0.339 |
| trade_neg4_pos4 | step | 0.0709 | 0.768 | 1.985 | 1.316 | 0.432 | 0.604 | 0.358 |
| trace_c05_pre03_pos05 | step | 0.0727 | 0.752 | 2.000 | 1.128 | 0.470 | 0.478 | 0.339 |
| big_platt | step | 0.1121 | 0.740 | 2.000 | 1.118 | 0.472 | 0.579 | 0.283 |
| trade_neg3_pos5 | step | 0.0792 | 0.732 | 1.914 | 1.119 | 0.472 | 0.516 | 0.315 |

## Correct-False-Stop Caps at First-Trigger FP/TP <= 2

| score | cap | threshold | recall | first-trigger FP/TP | correct false-stop | pre-FHIS false-stop |
|---|---:|---:|---:|---:|---:|---:|
| trace_c10_pre05_pos08 | 0.5 | 0.0770 | 0.831 | 1.304 | 0.497 | 0.402 |
| avg_trace2 | 0.5 | 0.0782 | 0.799 | 1.047 | 0.497 | 0.335 |
| trace_c05_pre03_pos05 | 0.5 | 0.0702 | 0.772 | 1.143 | 0.491 | 0.346 |
| avg_all6 | 0.5 | 0.1021 | 0.760 | 0.953 | 0.497 | 0.287 |
| max_all6 | 0.5 | 0.1807 | 0.760 | 0.976 | 0.497 | 0.287 |
| avg_trace2 | 0.6 | 0.0689 | 0.854 | 1.482 | 0.591 | 0.429 |
| trace_c10_pre05_pos08 | 0.6 | 0.0720 | 0.850 | 1.664 | 0.591 | 0.457 |
| avg_base_trace2 | 0.6 | 0.0696 | 0.846 | 1.311 | 0.597 | 0.386 |
| trace_c05_pre03_pos05 | 0.6 | 0.0613 | 0.839 | 1.670 | 0.597 | 0.457 |
| avg_all6 | 0.6 | 0.0799 | 0.819 | 1.183 | 0.597 | 0.339 |
| avg_base_trace2 | 0.7 | 0.0609 | 0.882 | 1.951 | 0.686 | 0.496 |
| trace_c10_pre05_pos08 | 0.7 | 0.0664 | 0.878 | 1.952 | 0.698 | 0.496 |
| avg_trace2 | 0.7 | 0.0633 | 0.870 | 1.990 | 0.673 | 0.504 |
| trace_c05_pre03_pos05 | 0.7 | 0.0589 | 0.866 | 1.900 | 0.610 | 0.500 |
| base_platt | 0.7 | 0.0550 | 0.866 | 1.923 | 0.698 | 0.476 |
| avg_all4 | 0.8 | 0.0534 | 0.913 | 1.893 | 0.780 | 0.480 |
| avg_big_base | 0.8 | 0.0478 | 0.906 | 1.892 | 0.761 | 0.480 |
| avg_all6 | 0.8 | 0.0583 | 0.898 | 1.944 | 0.748 | 0.484 |
| max_all6 | 0.8 | 0.0810 | 0.894 | 1.754 | 0.736 | 0.461 |
| min_all6 | 0.8 | 0.0405 | 0.890 | 1.909 | 0.786 | 0.465 |

## Takeaway

- Best strict step FP/TP<=2 point in this expanded comparison is `avg_all4`: recall 0.839, correct false-stop 0.642.
- Best first-trigger FP/TP<=2 point is `avg_all4`: recall 0.913, correct false-stop 0.780.
- Best trace-loss first-trigger point is `trace_c10_pre05_pos08`: recall 0.882, correct false-stop 0.704.
- With correct false-stop capped at <=0.7, best recall is `avg_base_trace2` at 0.882.
- Trace-level regularization gives a cleaner medium-high recall candidate, but it does not dominate the previous high-recall ensemble; held-out clean-eval transfer is still required before online intervention.
