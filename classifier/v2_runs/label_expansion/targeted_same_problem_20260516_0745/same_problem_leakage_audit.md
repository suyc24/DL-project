# Same-Problem Expansion Leakage Audit

Date: 2026-05-16 09:15 CST heartbeat.

This audit maps the same-problem expansion candidates onto the current v2.2 p15 score-detail splits. It does not create labels.

## Summary

- total_candidates: `80`
- trace_split_counts: `{'not_in_score_details': 12, 'hard_dev': 68}`
- problem_split_counts: `{'hard_dev': 80}`
- leakage_risk_counts: `{'hard_dev_problem': 80}`
- bucket_by_leakage: `{'hard_dev_problem': {'same_problem_unlabeled_needs_annotation': 10, 'same_problem_high_conf_wrong_positive': 30, 'same_problem_recovered_error_positive': 2, 'seed_strict_missed_positive': 30, 'same_problem_high_conf_correct_negative': 8}}`

## Guidance

- Candidates whose problem appears in `hard_dev` must not be used to claim improvement on the existing hard-dev set. They are useful for a stress-training/diagnostic model only, or require a new independent evaluation split.

- If the user approves labeling the 10 unlabeled traces, keep v2.4 reports explicit: train-with-targeted-hard-cases is a targeted repair experiment, not an unbiased hard-dev comparison.

- For a clean metric after v2.4, create a new problem-disjoint eval set from rough-unknown/new generations or hold out all strict-miss seed problems from evaluation.

## Hard-Dev Problem Candidates By Bucket

| bucket | count |
| --- | ---: |
| same_problem_unlabeled_needs_annotation | 10 |
| same_problem_high_conf_wrong_positive | 30 |
| same_problem_recovered_error_positive | 2 |
| seed_strict_missed_positive | 30 |
| same_problem_high_conf_correct_negative | 8 |
