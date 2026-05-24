# Probe v2 Ensemble Calibration-Split Threshold Report

Date: 2026-05-16 05:30 CST heartbeat.

This report selects model/ensemble thresholds on `calibration_score_details.jsonl` and evaluates the fixed threshold on `hard_dev_score_details.jsonl`. Rows with `example_type` prefixes `recovered` or `maxed` are excluded to keep the original hard-dev target comparable.

- val_inner: 866 common original rows; example types `{'strong_prefhis_negative': 227, 'fhis_positive': 112, 'hard_fhis_false_negative': 18, 'strong_correct_negative': 468, 'mined_hard_negative': 41}`.
- calibration: 1149 common original rows; example types `{'hard_fhis_false_negative': 21, 'strong_prefhis_negative': 270, 'mined_hard_negative': 58, 'fhis_positive': 123, 'strong_correct_negative': 677}`.
- hard_dev: 1522 common original rows; example types `{'strong_prefhis_negative': 402, 'fhis_positive': 203, 'strong_correct_negative': 784, 'mined_hard_negative': 99, 'hard_fhis_false_negative': 34}`.

## Selected Models

| model | selection_mode | selected_threshold | cal_recall | cal_fp_per_tp | cal_first_fp_per_tp | hard_recall | hard_precision | hard_fp_per_tp | hard_correct_false_stop | hard_pre_fhis_stop | hard_first_precision | hard_first_fp_per_tp | hard_triggered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v22_p15 | step | 0.1074 | 0.7847 | 2.0000 | 1.7656 | 0.8228 | 0.3757 | 1.6615 | 0.5629 | 0.3544 | 0.3964 | 1.5225 | 280 |
| v22_p15 | first_trigger | 0.0931 | 0.8194 | 2.1695 | 1.8923 | 0.8481 | 0.3602 | 1.7761 | 0.6225 | 0.3713 | 0.3831 | 1.6106 | 295 |
| v22_p20 | step | 0.0784 | 0.7778 | 1.9107 | 1.7424 | 0.8059 | 0.3760 | 1.6597 | 0.5166 | 0.3713 | 0.3875 | 1.5810 | 271 |
| v22_p20 | first_trigger | 0.0730 | 0.7986 | 2.0609 | 1.9683 | 0.8228 | 0.3618 | 1.7641 | 0.5629 | 0.3755 | 0.3830 | 1.6111 | 282 |
| v23_low | step | 0.0798 | 0.7639 | 1.9818 | 1.8871 | 0.8017 | 0.3592 | 1.7842 | 0.6225 | 0.3840 | 0.3554 | 1.8137 | 287 |
| v23_low | first_trigger | 0.0798 | 0.7639 | 1.9818 | 1.8871 | 0.8017 | 0.3592 | 1.7842 | 0.6225 | 0.3840 | 0.3554 | 1.8137 | 287 |
| cal_avg_v22p15_0.50_v23low_0.50 | step | 0.1016 | 0.7639 | 2.0000 | 1.7656 | 0.8101 | 0.3757 | 1.6615 | 0.5695 | 0.3586 | 0.3849 | 1.5981 | 278 |
| cal_avg_v22p15_0.50_v23low_0.50 | first_trigger | 0.0838 | 0.8125 | 2.2393 | 2.0000 | 0.8481 | 0.3436 | 1.9104 | 0.6623 | 0.3966 | 0.3597 | 1.7798 | 303 |
| rank_avg_v22p15_0.50_v23low_0.50 | step | 0.7247 | 0.7500 | 1.9444 | 1.7302 | 0.7553 | 0.4377 | 1.2849 | 0.4172 | 0.2911 | 0.4568 | 1.1892 | 243 |
| rank_avg_v22p15_0.50_v23low_0.50 | first_trigger | 0.6420 | 0.8472 | 2.2787 | 2.0000 | 0.8143 | 0.3669 | 1.7254 | 0.6159 | 0.3629 | 0.3806 | 1.6273 | 289 |
| cal_max_v22p15_v23low | step | 0.1189 | 0.7778 | 1.9911 | 1.7538 | 0.8228 | 0.3659 | 1.7333 | 0.5894 | 0.3671 | 0.3825 | 1.6147 | 285 |
| cal_max_v22p15_v23low | first_trigger | 0.0931 | 0.8403 | 2.2562 | 2.0000 | 0.8565 | 0.3355 | 1.9803 | 0.7020 | 0.4008 | 0.3537 | 1.8273 | 311 |

## Top 12 Calibration-Selected Runs By Hard-Dev Recall

| model | selection_mode | selected_threshold | cal_recall | cal_fp_per_tp | cal_first_fp_per_tp | hard_recall | hard_precision | hard_fp_per_tp | hard_correct_false_stop | hard_pre_fhis_stop | hard_first_precision | hard_first_fp_per_tp | hard_triggered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cal_max_v22p15_v23low | first_trigger | 0.0931 | 0.8403 | 2.2562 | 2.0000 | 0.8565 | 0.3355 | 1.9803 | 0.7020 | 0.4008 | 0.3537 | 1.8273 | 311 |
| cal_avg_v22p15_0.90_v23low_0.10 | first_trigger | 0.0896 | 0.8194 | 2.2119 | 1.9077 | 0.8523 | 0.3531 | 1.8317 | 0.6424 | 0.3755 | 0.3779 | 1.6460 | 299 |
| cal_avg_v22p15_0.85_v23low_0.15 | first_trigger | 0.0879 | 0.8194 | 2.2288 | 1.9077 | 0.8523 | 0.3501 | 1.8564 | 0.6490 | 0.3797 | 0.3733 | 1.6786 | 300 |
| cal_avg_v22p15_0.80_v23low_0.20 | first_trigger | 0.0862 | 0.8194 | 2.2627 | 1.9844 | 0.8523 | 0.3471 | 1.8812 | 0.6556 | 0.3882 | 0.3654 | 1.7364 | 301 |
| cal_avg_v22p15_0.75_v23low_0.25 | first_trigger | 0.0845 | 0.8264 | 2.2353 | 1.9538 | 0.8523 | 0.3406 | 1.9356 | 0.6623 | 0.3966 | 0.3597 | 1.7798 | 303 |
| cal_avg_v22p15_0.60_v23low_0.40 | first_trigger | 0.0829 | 0.8264 | 2.2437 | 2.0000 | 0.8523 | 0.3412 | 1.9307 | 0.6689 | 0.4008 | 0.3574 | 1.7982 | 305 |
| cal_avg_v22p15_0.55_v23low_0.45 | first_trigger | 0.0819 | 0.8264 | 2.2269 | 2.0000 | 0.8523 | 0.3401 | 1.9406 | 0.6755 | 0.4008 | 0.3562 | 1.8073 | 306 |
| v22_p15 | first_trigger | 0.0931 | 0.8194 | 2.1695 | 1.8923 | 0.8481 | 0.3602 | 1.7761 | 0.6225 | 0.3713 | 0.3831 | 1.6106 | 295 |
| cal_avg_v22p15_1.00_v23low_0.00 | first_trigger | 0.0931 | 0.8194 | 2.1695 | 1.8923 | 0.8481 | 0.3602 | 1.7761 | 0.6225 | 0.3713 | 0.3831 | 1.6106 | 295 |
| cal_avg_v22p15_0.95_v23low_0.05 | first_trigger | 0.0914 | 0.8194 | 2.2119 | 1.9077 | 0.8481 | 0.3545 | 1.8209 | 0.6358 | 0.3755 | 0.3771 | 1.6518 | 297 |
| cal_avg_v22p15_0.70_v23low_0.30 | first_trigger | 0.0873 | 0.8194 | 2.1949 | 1.9091 | 0.8481 | 0.3508 | 1.8507 | 0.6424 | 0.3840 | 0.3691 | 1.7091 | 298 |
| cal_avg_v22p15_0.50_v23low_0.50 | first_trigger | 0.0838 | 0.8125 | 2.2393 | 2.0000 | 0.8481 | 0.3436 | 1.9104 | 0.6623 | 0.3966 | 0.3597 | 1.7798 | 303 |

## Interpretation

- Once threshold selection is moved to the calibration split, the simple v2.2/v2.3-low ensemble no longer clearly beats v2.2 p15 on hard-dev. The apparent 0.8734 hard-dev gain from direct threshold search should be treated as optimistic model selection signal, not deploy-ready calibration.

- The strongest calibration-selected hard-dev points remain around 0.86-0.87 recall with first-trigger precision near the 1/3 floor.

- This reinforces the earlier bottleneck diagnosis: architecture/weight tweaks are nearly saturated; the next real gain likely needs more targeted hard-positive labels or a separately trained meta-calibrator with a larger held-out score set.

