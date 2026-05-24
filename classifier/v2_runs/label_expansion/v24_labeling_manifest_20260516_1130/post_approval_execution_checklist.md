# v2.4 Post-Approval Execution Checklist

Date: 2026-05-16 12:15 CST

This is a dry-run checklist. It intentionally does not create labels or start training before explicit approval.

## Preconditions

- User explicitly approves Codex/OpenAI adjudication and persistence of FHIS labels for the listed v2.4 repair and clean-eval candidates.
- Remote model exists at `/root/shared-nvme/models/Qwen2.5-Math-7B-Instruct`.
- Track A repair and Track B clean eval remain separated.

## Local Files

```text
classifier/v2_runs/label_expansion/v24_labeling_manifest_20260516_1130/manifest.json
classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/annotation_candidate_preview.jsonl
classifier/v2_runs/label_expansion/clean_eval_candidates_20260516_1045/clean_eval_candidate_queue.jsonl
```

## Step 1: Persist Labels After Approval

Expected outputs:

```text
classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/manual_labels_high.jsonl
classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/manual_label_traces.jsonl
classifier/v2_runs/label_expansion/clean_eval_candidates_20260516_1045/clean_eval_labels.jsonl
classifier/v2_runs/label_expansion/clean_eval_candidates_20260516_1045/clean_eval_traces.jsonl
```

Rules:

- Repair labels may enter training only with `example_type` marking them as targeted/leaky.
- Clean eval labels must not enter training, threshold selection, or model selection.
- Ambiguous labels should go to an adjudication queue, not training/eval.

## Step 2: Sync Repair Training Additions To Remote

Only sync repair labels/traces for hidden-state extraction. Clean eval traces should be synced only if held-out scoring requires hidden states later.

```bash
rsync -av classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/manual_labels_high.jsonl   root@REMOTE:/root/shared-nvme/DL-project/classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/
rsync -av classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/manual_label_traces.jsonl   root@REMOTE:/root/shared-nvme/DL-project/classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/
```

## Step 3: Extract Hidden States On Remote

Use only exact Qwen2.5-Math-7B-Instruct:

```text
/root/shared-nvme/models/Qwen2.5-Math-7B-Instruct
```

Expected artifact name:

```text
classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/manual_step_hidden_states.pt
```

## Step 4: Build v2.4 Targeted Repair Dataset

Merge base v2.2 maxed12 data plus approved repair additions. Do not merge clean eval.

Expected directory:

```text
classifier/v2_runs/bad_step_v24_targeted_repair/
```

## Step 5: Train One Conservative Variant

Start with v2.2 p15 Platt hazard GRU settings. Avoid broad sweeps until repair signal is validated.

## Step 6: Report Separately

Track A diagnostic:

- strict-miss repair recall
- same-problem repair positives recall
- repair hard-negative FP/TP
- pre-FHIS false-stop rate
- correct-trace false-stop rate

Track B clean eval:

- fixed threshold from calibration only
- problem-disjoint clean eval recall/precision
- FP/TP and first-trigger FP/TP
- step-1/2 bad-step recall

## Stop Conditions

Do not proceed to online intervention unless Track B shows calibrated high recall under the FP budget. Do not claim current hard-dev improvement from Track A.
