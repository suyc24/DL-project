# High-Precision Bad-Step Detector v2.1 Plan

Date: 2026-05-15

## Goal

Train a probe that is optimized for safe test-time intervention, not generic row-level accuracy. The deployable object should be a high-precision bad-step detector with a separate calibration layer:

```text
hidden/scalar/text/trace features -> bad-step score -> calibrated trigger threshold
```

The target positive is still the first harmful invalid step, but the operational target is stricter: trigger only when the posterior risk is high enough that retrying is better than letting the original step continue.

## Current Status

The best current online policy is:

```text
probe: classifier/v2_runs/sweep_scalars/scalars_c5_p4/probe.joblib
threshold: 0.8
retry_prompt_style: strict_step
model: /root/shared-nvme/models/Qwen2.5-Math-7B-Instruct
```

The corrected 30-problem matched comparison shows a real positive signal:

| Policy | Accepted | Abstained | Rough Solve All | Rough Solve Answered | Flagged | Restart Trace |
|---|---:|---:|---:|---:|---:|---:|
| t08 default retry | 14/30 | 16/30 | 0.2333 | 0.5000 | 33 | 16 |
| t08 strict retry | 18/30 | 12/30 | 0.3333 | 0.5556 | 29 | 12 |

This is enough to justify continued online work, but not enough to call the classifier solved. The remaining bottleneck is high-cost false triggering and weak calibration around the intervention threshold.

## Five Bottlenecks To Address

### 1. Label Noise From Trace-Level Supervision

Problem: many current labels are effectively trace-level first-invalid labels. Pre-FHIS steps in wrong traces are negative, but some are already suspicious or incomplete. Correct traces may contain benign gaps. This makes a binary step label noisy near the boundary.

Action:

- Keep high-confidence existing labels as the main dataset.
- Add a v2.1 adjudication queue for ambiguous/high-impact cases.
- Use `valid`, `benign_gap`, `harmful_invalid`, `after_invalid`, and `ambiguous` internally; train the main binary head only on `harmful_invalid` vs strong negatives.
- Downweight or exclude `ambiguous` and low-confidence labels from calibration.

### 2. Insufficient Hard Negatives

Problem: the classifier has enough easy negatives, but online intervention is dominated by mistakes on plausible-looking correct steps and pre-FHIS steps.

Action:

- Mine hard negatives from:
  - high-score correct-trace false positives,
  - high-score non-FHIS steps before the true FHIS,
  - retry outputs that were flagged but led to worse online behavior,
  - same-problem correct/wrong branches where a wrong trace diverges after a shared-looking prefix.
- Give these examples explicit higher negative weight during detector training.
- Build a hard-negative dev set that is never used for threshold fitting.

### 3. Wrong Optimization Target

Problem: AUROC/AUPRC can improve while the online trigger remains unsafe. The useful operating point is high precision at a deployable threshold.

Action:

- Report threshold curves for:
  - bad-step precision,
  - FHIS recall,
  - correct-trace false-stop rate,
  - pre-FHIS false-stop rate,
  - per-trace first-trigger precision,
  - online acceptance/solve rate when available.
- Select detector checkpoints by high-precision constraints first, then recall.
- Treat `recall@0.90` as a diagnostic, not the deployment objective.

### 4. Offline-Online Distribution Shift

Problem: retry prompts produce a different hidden-state distribution from first-pass traces.

Action:

- Add online retry artifacts to the training/evaluation manifest.
- Label retry candidates as:
  - `retry_fixed`,
  - `retry_still_bad`,
  - `retry_degenerate`,
  - `retry_false_alarm`.
- Keep a separate calibration split containing online retry examples.
- Do not deploy a threshold unless it is stable on both first-pass and retry-distribution calibration sets.

### 5. Weak Classifier Architecture

Problem: the current scalar layerwise MLP is useful, but it treats each step mostly independently and has no explicit hazard/calibration design.

Action:

Train in stages:

1. **Detector A: high-precision scalar layerwise MLP**
   - Start from `scalars_c5_p4`.
   - Reweight strong correct/pre-FHIS negatives and hard negatives.
   - Add focal or asymmetric loss to focus on confident false positives.
   - Save raw logits, not just probabilities.

2. **Detector B: calibrated detector**
   - Fit temperature scaling or isotonic calibration on a held-out calibration split.
   - Calibrate separately for first-pass and retry distributions if needed.
   - Export a threshold table: precision-targeted thresholds at 0.80, 0.85, 0.90, 0.95 estimated precision.

3. **Detector C: causal hazard model**
   - Use per-step encoded features plus a causal GRU/Transformer over prior steps.
   - Predict `p(FHIS at step k | steps <= k)`.
   - Compare against Detector B only after Detector B has a stable calibrated baseline.

## Data Plan

Initial v2.1 target sizes:

| Split | Positives | Strong Negatives | Hard Negatives | Notes |
|---|---:|---:|---:|---|
| train | 1k-3k | 8k-25k | 1k-3k | can reuse existing high-confidence labels plus mined cases |
| calibration | 200-500 | 1k-3k | 300-800 | problem-disjoint; includes online retry distribution |
| hard dev | 200-500 | 1k-3k | 500-1k | no threshold fitting |
| dataset-disjoint test | 300-800 | 2k-8k | 300-800 | MATH/Olympiad held out |

Priority for new self-labeling:

1. Existing high-score false positives and false negatives from `v2_error_analysis.py`.
2. Matched30 online retries where strict/default disagree or restart.
3. Same-problem branches from the original 4 samples per OlympiadBench problem.
4. MATH Level 5 holdout traces for dataset-disjoint calibration/test.

## Near-Term Execution

1. Build `classifier/v2_runs/bad_step_v21/manifest.jsonl` with explicit example type, source, original label, v2.1 training label, weight, and split.
2. Produce `adjudication_queue.jsonl` for cases that should be manually/self-labeled before training.
3. Train Detector A using the existing layerwise/scalar architecture with stricter hard-negative weights.
4. Fit calibration on held-out logits and write `calibration_report.json`.
5. Run offline intervention metrics at precision-targeted thresholds.
6. Only then run online probe-retry with the best calibrated threshold and `strict_step` retry prompt.

