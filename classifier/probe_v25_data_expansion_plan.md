# Probe Classifier v2.5 Data Expansion Plan

Date: 2026-05-16

Goal: add 2000 high-quality FHIS trace labels and build a cleaner, more natural held-out evaluation set for the high-recall bad-step detector.

## Data Policy

- Training expansion labels may include a mixture of natural samples and hard-case samples.
- Natural clean evaluation labels must be problem-disjoint from all training, calibration, hard-dev, and targeted repair data.
- Clean evaluation labels must not be used for training, threshold selection, architecture selection, or calibration.
- Track B clean eval from v2.4 remains held out.

## Proposed Split

- `train_expansion`: 2000 new high-quality trace labels.
- `clean_eval_natural`: 300-500 new high-quality trace labels from natural generation.
- `clean_eval_natural` should use complete problem groups where possible, so all samples for a problem stay in the same split.

## Sampling Recipe

1. Preserve existing labels and avoid duplicate `trace_id`.
2. Use unlabeled existing generated traces first.
3. Generate additional natural traces from OlympiadBench using the exact Qwen2.5-Math-7B-Instruct model.
4. Use new sample ids or batch ids so repeated generations for an old problem do not collide with old `sample-0..3` trace ids.
5. Keep at least 300 natural-eval labels problem-disjoint from every known labeled/training problem.

## Label Quality

- Use the existing canonical FHIS prompt and JSON schema.
- Keep only `confidence=high` for model training.
- Save medium/low labels separately for audit, but do not train on them.
- For wrong final answers without a determinable first invalid step, keep the row but mark it non-train-usable.

## Evaluation

After labeling:

- Rebuild hidden-state features for the high-confidence train expansion.
- Train a recall-oriented hazard GRU variant and a larger variant.
- Calibrate separately on a clean calibration split.
- Evaluate on:
  - current hard-dev for continuity only;
  - new `clean_eval_natural` for unbiased deployment-like estimates;
  - targeted repair slice for strict-miss diagnostics only.

Promotion criterion:

- Prefer recall improvement while keeping `FP/TP <= 2`.
- Report correct-trace false-stop, pre-FHIS false-stop, and first-trigger precision.
