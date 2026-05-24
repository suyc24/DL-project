# v2.5 Hazard Detector Threshold Summary

Data: v2.5 high-confidence train-expansion FHIS labels only. Clean eval remains held out and was not used for model selection or threshold selection.

## hazard_recall_big_platt

- best_epoch: 4; best_val_auprc: 0.8237
- hard_dev rows: 1552 = 254 positives + 1298 negatives
- hard_dev AUROC: 0.8167; AUPRC: 0.5774; Brier: 0.1013

| budget mode | FP/TP budget | threshold | recall | step FP/TP | first-trigger FP/TP | first-trigger precision | correct false-stop | pre-FHIS false-stop | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| step | 1.0 | 0.2418 | 0.587 | 0.960 | 0.781 | 0.561 | 0.340 | 0.201 | 149 | 143 | 105 |
| first_trigger | 1.0 | 0.1443 | 0.732 | 1.683 | 0.858 | 0.538 | 0.491 | 0.240 | 186 | 313 | 68 |
| step | 1.5 | 0.1807 | 0.677 | 1.488 | 0.846 | 0.542 | 0.434 | 0.228 | 172 | 256 | 82 |
| first_trigger | 1.5 | 0.0582 | 0.839 | 2.460 | 1.483 | 0.403 | 0.698 | 0.382 | 213 | 524 | 41 |
| step | 2.0 | 0.1121 | 0.740 | 2.000 | 1.118 | 0.472 | 0.579 | 0.283 | 188 | 376 | 66 |
| first_trigger | 2.0 | 0.0396 | 0.898 | 2.855 | 1.973 | 0.336 | 0.824 | 0.469 | 228 | 651 | 26 |
| step | 3.0 | 0.0374 | 0.906 | 2.943 | 2.143 | 0.318 | 0.824 | 0.500 | 230 | 677 | 24 |
| first_trigger | 3.0 | 0.0316 | 0.925 | 3.362 | 2.888 | 0.257 | 0.893 | 0.583 | 235 | 790 | 19 |

## hazard_recall_big_isotonic

- best_epoch: 4; best_val_auprc: 0.8237
- hard_dev rows: 1552 = 254 positives + 1298 negatives
- hard_dev AUROC: 0.8029; AUPRC: 0.5015; Brier: 0.1026

| budget mode | FP/TP budget | threshold | recall | step FP/TP | first-trigger FP/TP | first-trigger precision | correct false-stop | pre-FHIS false-stop | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| step | 1.0 | 0.2388 | 0.421 | 0.505 | 0.444 | 0.692 | 0.176 | 0.110 | 107 | 54 | 147 |
| first_trigger | 1.0 | 0.2388 | 0.421 | 0.505 | 0.444 | 0.692 | 0.176 | 0.110 | 107 | 54 | 147 |
| step | 1.5 | 0.2388 | 0.421 | 0.505 | 0.444 | 0.692 | 0.176 | 0.110 | 107 | 54 | 147 |
| first_trigger | 1.5 | 0.1429 | 0.760 | 2.016 | 1.158 | 0.463 | 0.597 | 0.295 | 193 | 389 | 61 |
| step | 2.0 | 0.2388 | 0.421 | 0.505 | 0.444 | 0.692 | 0.176 | 0.110 | 107 | 54 | 147 |
| first_trigger | 2.0 | 0.0385 | 0.898 | 2.864 | 1.973 | 0.336 | 0.824 | 0.469 | 228 | 653 | 26 |
| step | 3.0 | 0.0385 | 0.898 | 2.864 | 1.973 | 0.336 | 0.824 | 0.469 | 228 | 653 | 26 |
| first_trigger | 3.0 | 0.0294 | 0.917 | 3.116 | 2.414 | 0.293 | 0.855 | 0.535 | 233 | 726 | 21 |

## Current Takeaway

- Platt is the better calibrated candidate for this run: hard-dev AUROC/AUPRC are higher than isotonic, and the threshold grid is smoother.
- Best strict step-level point under FP/TP<=2 is Platt threshold 0.1121: recall 0.740, step FP/TP 2.000, first-trigger FP/TP 1.118.
- Best trace/first-trigger budget point under FP/TP<=2 is Platt threshold 0.0396: recall 0.898, first-trigger FP/TP 1.973, first-trigger precision 0.336; this is much closer to the user objective, but correct-trace false-stop is high at 0.824.
- Do not run online intervention yet; next useful work is either label held-out clean eval for unbiased threshold validation or train/ensemble variants that reduce correct-trace false stops at the same recall.
