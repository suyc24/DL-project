# Probe v2 Fast Meta-Calibrator Report

Date: 2026-05-16 07:00 CST heartbeat.

Pure-Python logistic meta-calibrator, trained on val_inner, threshold selected on calibration, fixed evaluation on original hard-dev. This quick grid uses component scores from v2.2 p15, v2.2 p20, and v2.3-low.

## Top Runs

| feature_set | pos_weight | l2 | selection_mode | selected_threshold | cal_recall | cal_fp_per_tp | cal_first_fp_per_tp | hard_recall | hard_precision | hard_fp_per_tp | hard_correct_false_stop | hard_pre_fhis_stop | hard_first_precision | hard_first_fp_per_tp | hard_triggered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scores | 4 | 0.0010 | first_trigger | 0.1644 | 0.8264 | 2.2353 | 1.9545 | 0.8523 | 0.3406 | 1.9356 | 0.6689 | 0.3966 | 0.3607 | 1.7727 | 305 |
| scores | 4 | 0.0100 | first_trigger | 0.1666 | 0.8264 | 2.2353 | 1.9545 | 0.8523 | 0.3406 | 1.9356 | 0.6689 | 0.3966 | 0.3607 | 1.7727 | 305 |
| scores | 6 | 0.0010 | first_trigger | 0.2222 | 0.8264 | 2.2353 | 1.9545 | 0.8523 | 0.3406 | 1.9356 | 0.6689 | 0.3966 | 0.3607 | 1.7727 | 305 |
| scores | 6 | 0.0100 | first_trigger | 0.2248 | 0.8264 | 2.2353 | 1.9545 | 0.8523 | 0.3406 | 1.9356 | 0.6689 | 0.3966 | 0.3607 | 1.7727 | 305 |
| scores | 2 | 0.0010 | first_trigger | 0.0951 | 0.8264 | 2.2353 | 1.9545 | 0.8523 | 0.3401 | 1.9406 | 0.6689 | 0.4008 | 0.3574 | 1.7982 | 305 |
| scores | 2 | 0.0100 | first_trigger | 0.0967 | 0.8264 | 2.2353 | 1.9545 | 0.8523 | 0.3401 | 1.9406 | 0.6689 | 0.4008 | 0.3574 | 1.7982 | 305 |
| scores | 1 | 0.0010 | first_trigger | 0.0562 | 0.8125 | 2.2308 | 2.0000 | 0.8481 | 0.3466 | 1.8856 | 0.6623 | 0.3882 | 0.3663 | 1.7297 | 303 |
| scores | 1 | 0.0100 | first_trigger | 0.0574 | 0.8125 | 2.2308 | 2.0000 | 0.8481 | 0.3466 | 1.8856 | 0.6623 | 0.3882 | 0.3663 | 1.7297 | 303 |
| wide | 1 | 0.0010 | step | 0.0739 | 0.7778 | 1.9821 | 1.9836 | 0.8354 | 0.3960 | 1.5253 | 0.5430 | 0.3797 | 0.3922 | 1.5495 | 283 |
| wide | 1 | 0.0010 | first_trigger | 0.0739 | 0.7778 | 1.9821 | 1.9836 | 0.8354 | 0.3960 | 1.5253 | 0.5430 | 0.3797 | 0.3922 | 1.5495 | 283 |
| wide | 2 | 0.0010 | step | 0.1324 | 0.7778 | 1.9821 | 2.0847 | 0.8354 | 0.4008 | 1.4949 | 0.5364 | 0.3840 | 0.3901 | 1.5636 | 282 |
| wide | 2 | 0.0100 | step | 0.1308 | 0.7778 | 1.9821 | 2.0333 | 0.8354 | 0.3984 | 1.5101 | 0.5364 | 0.3840 | 0.3901 | 1.5636 | 282 |

## Baseline Context

- v2.2 p15 calibration-selected first-trigger: hard recall 0.8481, precision 0.3602, FP/TP 1.7761, first-trigger precision 0.3831.

- max(v2.2 p15, v2.3-low) calibration-selected first-trigger: hard recall 0.8565, precision 0.3355, FP/TP 1.9803, first-trigger precision 0.3537.


## Interpretation

- Best fast meta-calibrator: feature_set=scores, pos_weight=4, l2=0.001, selection=first_trigger; hard recall 0.8523, FP/TP 1.9356, first-trigger FP/TP 1.7727.

- If this does not exceed the simple calibrated max ensemble by a meaningful margin, the next step should stay focused on targeted data expansion rather than score-combination tricks.

