# Probe Classifier v2.4 Targeted Repair Evaluation Protocol

Date: 2026-05-16 10:00 CST

## Current State

The best honest offline frontier is still roughly 0.85-0.86 hard-dev recall under the user budget (`FP/TP <= 2`, first-trigger precision near or above 1/3). Score fusion and a lightweight meta-calibrator did not materially improve this frontier. The current bottleneck is data/model coverage for early plausible-looking wrong setup steps.

A same-problem expansion queue exists for strict hard-dev misses:

```text
classifier/v2_runs/label_expansion/targeted_same_problem_20260516_0745/
```

Leakage audit result: all 80 same-problem candidates belong to current hard-dev problems. They are useful for targeted repair, but not for claiming unbiased hard-dev improvement.

## Two Separate Tracks

### Track A: Targeted Repair Diagnostic

Goal: answer whether adding same-problem hard positives/negatives teaches the detector to catch the known failure modes.

Allowed training additions after explicit labeling approval:

- 10 same-problem unlabeled candidates once adjudicated.
- Existing same-problem high-confidence positives and correct negatives.
- False-stop hard negatives from `targeted_hard_cases_20260516_0615`.

Evaluation wording:

- Report as targeted repair / stress-test only.
- Do not describe current hard-dev gains as generalization.
- Always include leakage warning: target problems overlap hard-dev.

Useful metrics:

- Recall on strict missed seed positives.
- Recall on newly labeled same-problem positives.
- FP/TP on same-problem correct negatives and false-stop queues.
- Whether pre-FHIS false-stop rate worsens.

### Track B: Clean Generalization Evaluation

Goal: produce a credible claim that v2.4 improves the detector beyond known hard-dev misses.

Required isolation:

- Exclude all strict-miss seed problem ids from clean evaluation.
- Prefer a fresh problem-disjoint eval set, not used for targeted repair training.
- Select thresholds on calibration data only, not on clean eval.

Candidate sources:

- Prepared rough-unknown queue under `classifier/v2_runs/label_expansion/rough_unknown_unlabeled_60_*` after explicit labeling approval.
- New generations from exact Qwen2.5-Math-7B-Instruct for problem ids not in train/calibration/hard-dev if available.
- Existing holdout labels only if the tensor/features are real tensors and not Git LFS pointers.

Minimum clean eval size target:

- At least 100 wrong traces with valid first invalid step.
- At least 100 correct traces / no-FHIS traces.
- Prefer 300-500 labeled traces before trusting a small recall movement.

Report metrics:

- Step-level recall and precision.
- FP/TP at selected threshold.
- First-trigger precision and first-trigger FP/TP.
- Correct-trace false-stop rate.
- Pre-FHIS false-stop rate.
- Recall on step 1/2 bad steps separately.

## Recommended Next Action After Approval

1. Persist labels for the 10 same-problem candidates.
2. Extract their hidden states on remote with exact model path:

```text
/root/shared-nvme/models/Qwen2.5-Math-7B-Instruct
```

3. Build v2.4 targeted repair dataset, clearly marking `example_type` as targeted/leaky.
4. Train one conservative hazard GRU variant with v2.2 p15 settings, not a broad architecture sweep.
5. Evaluate Track A diagnostic.
6. Only after Track A confirms the direction, build Track B clean eval with new problem-disjoint labels.

## Approval Still Needed

No new FHIS labels should be written until the user explicitly approves Codex/OpenAI adjudication of the 10 local trace candidates and label persistence, or provides an offline/non-OpenAI labeler.
