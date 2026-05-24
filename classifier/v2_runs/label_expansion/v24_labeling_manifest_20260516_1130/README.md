# v2.4 Labeling Manifest and Execution Checklist

Date: 2026-05-16 11:30 CST

This manifest does not create labels. It lists the exact candidate batches that need explicit user approval before Codex/OpenAI adjudication and label persistence.

## Batches

- Track A repair batch: 10 traces from `/home/suyc24/Python/DL project/classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/annotation_candidate_preview.jsonl`. These overlap hard-dev problems and are for targeted repair only.

- Track B clean eval batch: 40 problem-disjoint traces from `/home/suyc24/Python/DL project/classifier/v2_runs/label_expansion/clean_eval_candidates_20260516_1045/clean_eval_candidate_queue.jsonl`. These are for independent evaluation and must stay out of training/threshold selection.

## Approval Sentence Needed

A sufficient user approval would be: “I approve Codex/OpenAI adjudication of the v2.4 repair and clean-eval candidate traces listed in `classifier/v2_runs/label_expansion/v24_labeling_manifest_20260516_1130/manifest.json`, and I approve saving the resulting FHIS labels in the repository.”

## After Approval

1. Create repair labels and clean-eval labels.

2. Extract hidden states only for repair/train additions, not clean eval unless needed for held-out scoring.

3. Train one v2.4 targeted-repair hazard GRU using v2.2 p15 settings.

4. Report Track A diagnostic separately from Track B clean evaluation.

## Files

```text
classifier/v2_runs/label_expansion/v24_labeling_manifest_20260516_1130/manifest.json
classifier/v2_runs/label_expansion/v24_labeling_manifest_20260516_1130/repair_batch_trace_ids.txt
classifier/v2_runs/label_expansion/v24_labeling_manifest_20260516_1130/clean_eval_batch_trace_ids.txt
```
