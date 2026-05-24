# Probe v2 Ensemble Recall-Budget Analysis

Date: 2026-05-16 05:30 CST heartbeat.

Input score files were pulled from the remote hard-dev reports and filtered to the original hard-dev subset by excluding `example_type` prefixes `recovered` and `maxed`.

Common evaluated rows: 1522; example types: `{'strong_prefhis_negative': 402, 'fhis_positive': 203, 'strong_correct_negative': 784, 'mined_hard_negative': 99, 'hard_fhis_false_negative': 34}`.

Important caveat: thresholds below are selected on hard-dev, so this is a model-selection analysis, not a deploy-ready calibration. The next proper step is to export calibration/validation score details or reserve another split for the ensemble threshold.

## Selected Results

| model | budget_mode | threshold | recall | precision | fp_per_tp | correct_trace_false_stop_rate | pre_fhis_false_stop_rate | first_trigger_precision | first_trigger_fp_per_tp | triggered_traces |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v22_p15 | step | 0.0816 | 0.8650 | 0.3361 | 1.9756 | 0.6755 | 0.4093 | 0.3581 | 1.7928 | 310 |
| v22_p15 | first_trigger | 0.0739 | 0.8692 | 0.3219 | 2.1068 | 0.6887 | 0.4388 | 0.3376 | 1.9623 | 314 |
| v22_p20 | step | 0.0659 | 0.8439 | 0.3344 | 1.9900 | 0.6291 | 0.4262 | 0.3423 | 1.9216 | 298 |
| v22_p20 | first_trigger | 0.0648 | 0.8565 | 0.3322 | 2.0099 | 0.6424 | 0.4430 | 0.3333 | 2.0000 | 303 |
| v23_low | step | 0.0737 | 0.8228 | 0.3374 | 1.9641 | 0.6424 | 0.4430 | 0.3106 | 2.2198 | 293 |
| v23_low | first_trigger | 0.0768 | 0.8101 | 0.3536 | 1.8281 | 0.6358 | 0.3966 | 0.3471 | 1.8812 | 291 |
| cal_avg_v22p15_0.50_v23low_0.50 | step | 0.0763 | 0.8734 | 0.3333 | 2.0000 | 0.6821 | 0.4177 | 0.3526 | 1.8364 | 312 |
| cal_avg_v22p15_0.50_v23low_0.50 | first_trigger | 0.0763 | 0.8734 | 0.3333 | 2.0000 | 0.6821 | 0.4177 | 0.3526 | 1.8364 | 312 |
| rank_avg_v22p15_0.50_v23low_0.50 | step | 0.5845 | 0.8692 | 0.3377 | 1.9612 | 0.6623 | 0.4388 | 0.3333 | 2.0000 | 306 |
| rank_avg_v22p15_0.50_v23low_0.50 | first_trigger | 0.5845 | 0.8692 | 0.3377 | 1.9612 | 0.6623 | 0.4388 | 0.3333 | 2.0000 | 306 |

## Top 10 By Recall Under Budget

| model | budget_mode | threshold | recall | precision | fp_per_tp | correct_trace_false_stop_rate | pre_fhis_false_stop_rate | first_trigger_precision | first_trigger_fp_per_tp | triggered_traces |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cal_avg_v22p15_0.50_v23low_0.50 | step | 0.0763 | 0.8734 | 0.3333 | 2.0000 | 0.6821 | 0.4177 | 0.3526 | 1.8364 | 312 |
| cal_avg_v22p15_0.50_v23low_0.50 | first_trigger | 0.0763 | 0.8734 | 0.3333 | 2.0000 | 0.6821 | 0.4177 | 0.3526 | 1.8364 | 312 |
| cal_avg_v22p15_0.45_v23low_0.55 | first_trigger | 0.0756 | 0.8734 | 0.3328 | 2.0048 | 0.6887 | 0.4262 | 0.3450 | 1.8981 | 313 |
| cal_avg_v22p15_0.55_v23low_0.45 | first_trigger | 0.0754 | 0.8734 | 0.3265 | 2.0628 | 0.7020 | 0.4262 | 0.3449 | 1.8991 | 316 |
| cal_avg_v22p15_0.60_v23low_0.40 | first_trigger | 0.0745 | 0.8734 | 0.3224 | 2.1014 | 0.7086 | 0.4346 | 0.3396 | 1.9444 | 318 |
| cal_avg_v22p15_0.40_v23low_0.60 | first_trigger | 0.0740 | 0.8734 | 0.3281 | 2.0483 | 0.7020 | 0.4346 | 0.3365 | 1.9717 | 315 |
| rank_avg_v22p15_0.70_v23low_0.30 | first_trigger | 0.5742 | 0.8692 | 0.3259 | 2.0680 | 0.6556 | 0.4430 | 0.3377 | 1.9615 | 308 |
| rank_avg_v22p15_0.65_v23low_0.35 | step | 0.5816 | 0.8692 | 0.3339 | 1.9951 | 0.6623 | 0.4388 | 0.3377 | 1.9615 | 308 |
| rank_avg_v22p15_0.65_v23low_0.35 | first_trigger | 0.5816 | 0.8692 | 0.3339 | 1.9951 | 0.6623 | 0.4388 | 0.3377 | 1.9615 | 308 |
| v22_p15 | first_trigger | 0.0739 | 0.8692 | 0.3219 | 2.1068 | 0.6887 | 0.4388 | 0.3376 | 1.9623 | 314 |

## Current Judgment

- The best hard-dev point in this grid is the simple 50/50 calibrated-score average of v2.2 p15 and v2.3-low; it reaches 0.8734 recall but uses hard-dev threshold selection and is therefore optimistic.

- The robust/simple 50/50 calibrated average remains useful: recall 0.8734 at FP/TP 2.0, with first-trigger FP/TP 1.8364.

- v2.2 p20 does not help; adding more positive loss pressure alone is not the right lever.

- Next high-value step: make the ensemble deployable by either exporting validation/calibration score details from both component models and selecting the threshold there, or by training a tiny held-out meta-calibrator on validation scores.

