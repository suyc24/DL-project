# Clean Eval Candidate Audit

Date: 2026-05-16 10:45 CST heartbeat.

This audit identifies already prepared trace candidates whose problem ids are disjoint from current v2.2 p15 val/calibration/hard-dev splits plus the strict-miss same-problem repair set. It does not create labels.

## Summary

- split_problem_counts: `{'val_inner': 186, 'calibration': 93, 'hard_dev': 117}`
- contaminated_problem_count: `396`
- total_candidate_rows: `76`
- clean_problem_disjoint_rows: `40`
- candidate_counts_by_source: `{'rough_unknown_60_preview': 60, 'quality_pilot_traces': 8, 'quality_pilot_lowtemp_traces': 8}`
- clean_counts_by_source: `{'rough_unknown_60_preview': 38, 'quality_pilot_traces': 2}`
- clean_rough_unknown_count: `38`

## Top Candidate Rows

| source | trace_id | problem_id | steps | rough_correct | final_answer |
| --- | --- | --- | ---: | --- | --- |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-119::sample-0` | `OE_TO_maths_en_COMP-119` | 1 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-119::sample-1` | `OE_TO_maths_en_COMP-119` | 1 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-119::sample-2` | `OE_TO_maths_en_COMP-119` | 12 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-137::sample-1` | `OE_TO_maths_en_COMP-137` | 2 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-205::sample-0` | `OE_TO_maths_en_COMP-205` | 10 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-205::sample-1` | `OE_TO_maths_en_COMP-205` | 5 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-205::sample-2` | `OE_TO_maths_en_COMP-205` | 10 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-205::sample-3` | `OE_TO_maths_en_COMP-205` | 5 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-333::sample-0` | `OE_TO_maths_en_COMP-333` | 5 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-333::sample-1` | `OE_TO_maths_en_COMP-333` | 5 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-333::sample-3` | `OE_TO_maths_en_COMP-333` | 6 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-340::sample-2` | `OE_TO_maths_en_COMP-340` | 7 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-371::sample-3` | `OE_TO_maths_en_COMP-371` | 2 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-387::sample-1` | `OE_TO_maths_en_COMP-387` | 6 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-387::sample-2` | `OE_TO_maths_en_COMP-387` | 6 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-396::sample-0` | `OE_TO_maths_en_COMP-396` | 3 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-438::sample-3` | `OE_TO_maths_en_COMP-438` | 6 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-456::sample-1` | `OE_TO_maths_en_COMP-456` | 4 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-456::sample-3` | `OE_TO_maths_en_COMP-456` | 5 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-461::sample-2` | `OE_TO_maths_en_COMP-461` | 1 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-478::sample-1` | `OE_TO_maths_en_COMP-478` | 5 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-524::sample-0` | `OE_TO_maths_en_COMP-524` | 4 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-539::sample-0` | `OE_TO_maths_en_COMP-539` | 6 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-539::sample-1` | `OE_TO_maths_en_COMP-539` | 6 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-544::sample-0` | `OE_TO_maths_en_COMP-544` | 3 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-544::sample-1` | `OE_TO_maths_en_COMP-544` | 5 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-544::sample-2` | `OE_TO_maths_en_COMP-544` | 5 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-544::sample-3` | `OE_TO_maths_en_COMP-544` | 5 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-561::sample-0` | `OE_TO_maths_en_COMP-561` | 3 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-561::sample-1` | `OE_TO_maths_en_COMP-561` | 3 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-561::sample-2` | `OE_TO_maths_en_COMP-561` | 5 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-570::sample-1` | `OE_TO_maths_en_COMP-570` | 6 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-577::sample-0` | `OE_TO_maths_en_COMP-577` | 2 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-577::sample-1` | `OE_TO_maths_en_COMP-577` | 2 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-590::sample-0` | `OE_TO_maths_en_COMP-590` | 4 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-590::sample-1` | `OE_TO_maths_en_COMP-590` | 6 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-590::sample-3` | `OE_TO_maths_en_COMP-590` | 3 | None |  |
| rough_unknown_60_preview | `OE_TO_maths_en_COMP-65::sample-3` | `OE_TO_maths_en_COMP-65` | 2 | None |  |
| quality_pilot_traces | `OE_TO_maths_en_COMP-504::sample-0` | `OE_TO_maths_en_COMP-504` | 4 | True | \[ \boxed{\frac{1}{2}} \] |
| quality_pilot_traces | `OE_TO_maths_en_COMP-504::sample-1` | `OE_TO_maths_en_COMP-504` | 4 | True | \[ \boxed{\frac{1}{2}} \] |

## Recommendation

- Use these candidates for Track B clean generalization labeling after explicit labeling approval.

- Prioritize rough-unknown candidates first, because they are likely to contain uncertain/incorrect traces.

- Keep targeted same-problem repair labels out of the clean eval threshold/evaluation path.

