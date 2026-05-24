# Targeted Labeling Approval Request

Date: 2026-05-16 08:30 CST

## Why Approval Is Needed

The next useful Probe Classifier v2 step is to label 10 local same-problem traces identified from strict missed positives. A safety review blocked persisting these labels because it requires explicit user approval before Codex/OpenAI adjudicates local trace contents and saves new FHIS labels.

## Requested Approval Scope

Approve Codex/OpenAI adjudication and label persistence for only these prepared local files:

```text
classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/annotation_candidate_preview.jsonl
classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/same_problem_candidate_queue.jsonl
```

The intended output files are:

```text
classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/manual_labels_high.jsonl
classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/manual_label_traces.jsonl
classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/manual_label_summary.json
```

## Planned Labeling Policy

For each candidate trace, record:

- `final_correct`
- `first_invalid_step`, or null if no harmful invalid step exists
- `error_type`
- concise reason
- confidence

Use only high-confidence labels for training. Ambiguous cases should be placed in an adjudication queue, not forced into training.

## Planned Training Use After Approval

1. Persist the 10 labels and matching full traces.
2. Extract hidden states remotely using only `/root/shared-nvme/models/Qwen2.5-Math-7B-Instruct`.
3. Build a v2.4 targeted expansion set that treats these as hard positives/negatives, while keeping hard-dev problem leakage explicit.
4. Prefer a fresh problem-disjoint evaluation or separate diagnostic split before claiming improvement.

## Current Non-Approved State

No new manual label file has been written. Existing safe artifacts are only candidate queues and reports.
