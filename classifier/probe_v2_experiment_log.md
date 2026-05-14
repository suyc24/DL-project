# Probe Classifier v2 Experiment Log

Date: 2026-05-13

## Current Objective

Improve the FHIS probe from a good offline/ranking classifier into a classifier that is useful for test-time intervention. The main bottleneck is online false stopping: at high FHIS recall, v1 flags too many correct or pre-FHIS steps.

## Remote Setup

Remote workspace:

```text
/root/shared-nvme/DL-project
```

Environment:

```text
conda env: fhis-v2
GPU: RTX 4090
```

Key uploaded artifacts:

```text
data_generation/qwen25_fhis/features/step_hidden_states_codex_clean.pt
data_generation/qwen25_fhis/outputs/generated_traces.jsonl
data_generation/qwen25_fhis/labels/fhis_labels.jsonl
data_generation/qwen25_fhis/results/hidden_mlp_probe.joblib
classifier/v2_error_analysis.py
src/fhis/train_probe_v2.py
```

## Baselines

### v1 hidden MLP

Remote error analysis output:

```text
classifier/v2_runs/error_analysis_mlp_test
```

Key test metrics:

| Metric | Value |
|---|---:|
| AUROC | 0.8738 |
| AUPRC | 0.5214 |
| recall_0.90 threshold | 0.3003 |
| FHIS step recall | 0.9006 |
| observable non-FHIS step FPR | 0.3308 |
| step precision | 0.3037 |
| correct-trace false stop rate | 0.8112 |
| pre-FHIS stop rate on wrong traces | 0.4152 |

### v2 layerwise/ranking baseline

Remote artifact:

```text
classifier/v2_runs/hidden_layerwise_probe_v2.joblib
classifier/v2_runs/layerwise_metrics.json
classifier/v2_runs/error_analysis_layerwise_v2_full
```

Key test metrics:

| Metric | Value |
|---|---:|
| AUROC | 0.8681 |
| AUPRC | 0.5284 |
| recall_0.90 threshold | 0.1433 |
| FHIS step recall | 0.9006 |
| observable non-FHIS step FPR | 0.3224 |
| step precision | 0.3092 |
| correct-trace false stop rate | 0.8112 |
| pre-FHIS stop rate on wrong traces | 0.4152 |

Conclusion: layerwise pooling plus ranking alone only marginally improves FPR and does not solve correct-trace false stops.

## Weighted Sweep

Remote summary:

```text
classifier/v2_runs/sweep_summary.json
```

Best non-scalar deployable variant:

| Variant | AUROC | AUPRC | r90 FPR | r90 precision | r90 correct false stop | r90 pre-FHIS stop | fixed-0.5 recall | fixed-0.5 FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| correct5_prefhis2 | 0.8769 | 0.5369 | 0.3205 | 0.3105 | 0.6224 | 0.4971 | 0.6199 | 0.0890 |

Best first scalar variant:

| Variant | AUROC | AUPRC | r90 FPR | r90 precision | r90 correct false stop | r90 pre-FHIS stop | fixed-0.5 recall | fixed-0.5 FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| correct3_prefhis2_scalars | 0.8811 | 0.5382 | 0.3037 | 0.3222 | 0.6993 | 0.4152 | 0.6257 | 0.1040 |

Takeaway: weighting correct-trace negatives substantially reduces correct-trace false stops at conservative thresholds. Adding scalar step metadata improves high-recall separation.

## Focused Scalar Sweep

Remote summary:

```text
classifier/v2_runs/sweep_scalars_summary.json
```

Most useful candidates:

| Variant | AUROC | AUPRC | r80 FPR | r80 correct false stop | r90 FPR | r90 precision | r90 correct false stop | fixed-0.5 FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scalars_c3_p2 | 0.8811 | 0.5382 | 0.2024 | 0.4965 | 0.3037 | 0.3222 | 0.6993 | 0.1040 |
| scalars_c5_p4 | 0.8823 | 0.5544 | 0.1893 | 0.4545 | 0.3046 | 0.3215 | 0.6853 | 0.0694 |
| scalars_c5_p2_wide | 0.8824 | 0.5860 | 0.2174 | 0.4545 | 0.3112 | 0.3169 | 0.6224 | 0.0656 |

Recommended current candidate:

```text
classifier/v2_runs/sweep_scalars/scalars_c5_p4/probe.joblib
```

`scalars_c5_p4` is the best balance if the target is high-recall routing with lower false stops. `scalars_c5_p2_wide` is attractive if AUPRC and correct-trace false stops matter more than the last point of r90 FPR.

## Label Audit

Local hard-case audit files:

```text
classifier/v2_runs/label_adjudication/hardcase_trace_ids.txt
classifier/v2_runs/label_adjudication/hardcase_sources.jsonl
classifier/v2_runs/label_adjudication/hardcase_labels.jsonl
classifier/v2_runs/label_adjudication/hardcase_label_audit_summary.json
```

Audit sample: 20 hard cases selected from high-score correct-trace false positives, high-score non-FHIS negatives, low-score FHIS positives, and rough/Codex conflicts.

Result:

| Category | Completed | Changed vs old label |
|---|---:|---:|
| rough_codex_conflict | 4 | 0 |
| scalar_fp | 8 | 0 |
| scalar_hn | 4 | 1 |
| scalar_fn | 4 | 0 |
| total | 20 | 1 |

The only changed case was `OE_TO_maths_en_COMP-30::sample-2`, where the new label moved FHIS from step 2 to step 4. Manual inspection suggests the old step-2 label is more consistent with the existing FHIS definition because step 2 already makes a false counting claim that is plausibly used downstream.

Conclusion: the existing high-confidence Codex labels are reasonably stable. Do not blindly relabel or overwrite the dataset. Use targeted adjudication for disputed cases and consider downweighting ambiguous traces rather than treating a single relabel as ground truth.

## Code Changes

Implemented:

```text
classifier/v2_error_analysis.py
src/fhis/train_probe_v2.py
src/fhis/online_router.py
src/fhis/score_candidate_pool.py
src/fhis/probe_retry_router.py
```

Highlights:

- Added v2 layerwise/ranking probe training.
- Added online-prefix error analysis and hard-case queue generation.
- Added sample weighting for correct-trace negatives and pre-FHIS negatives.
- Added scalar step metadata support.
- Updated router scoring so scalar probes can receive `step_index` and `step_length_chars` online while old probes remain compatible.

## Current Judgment

Full relabeling is not the next best use of effort. The better path is:

1. Keep the original high-confidence labels as the main truth set.
2. Add a small adjudication workflow for disagreement cases.
3. Train with explicit correct-trace and pre-FHIS negative weighting.
4. Use scalar metadata online, because it gave the clearest separation gain.
5. Evaluate intervention policies at multiple operating points, especially recall 0.80 and 0.90, instead of deploying a single high-recall threshold.

The best v2 models improve the situation but still do not make probe-only retry safe at recall 0.90. A practical test-time intervention should either use a more conservative threshold, or route flagged steps to an external verifier/secondary check before forcing a retry.

## Heartbeat Follow-Up: Online Smoke Prep

Date: 2026-05-13, heartbeat `probe-v2-experiment-follow-up`

Added a probe-retry smoke config for the current recommended scalar candidate:

```text
data_generation/qwen25_fhis/configs/probe_retry_v2_scalars_c5_p4_smoke.yaml
```

Configuration:

- Probe: `classifier/v2_runs/sweep_scalars/scalars_c5_p4/probe.joblib`
- Threshold: `0.5`
- Problems: `data_generation/qwen25_fhis/outputs/online_smoke_10_problems.jsonl`
- Output: `classifier/v2_runs/online_smoke/scalars_c5_p4_retry_t05_results.jsonl`
- Retry policy: 1 whole-trace attempt, 2 attempts per suspicious step, accept best after retries within margin 0.05.

The threshold is intentionally conservative. Offline `fixed_0_5` for `scalars_c5_p4` had FHIS recall 0.5848 and observable non-FHIS FPR 0.0694, which is more appropriate for a first probe-only retry smoke than the high-recall r90 operating point.

The remote SSH gateway rejected the password session during this heartbeat, so the smoke was not started remotely. Next attempt should:

```bash
cd /root/shared-nvme/DL-project
# sync local router scalar-metadata patches if not already present
mkdir -p classifier/v2_runs/online_smoke
nohup bash -lc 'PYTHONUNBUFFERED=1 conda run -n fhis-v2 python -m fhis.probe_retry_router --config data_generation/qwen25_fhis/configs/probe_retry_v2_scalars_c5_p4_smoke.yaml --limit 10 && conda run -n fhis-v2 python -m fhis.evaluate_online --config data_generation/qwen25_fhis/configs/probe_retry_v2_scalars_c5_p4_smoke.yaml --output classifier/v2_runs/online_smoke/scalars_c5_p4_retry_t05_summary.json' > classifier/v2_runs/online_smoke/scalars_c5_p4_retry_t05.log 2>&1 &
```

## Heartbeat Follow-Up: Remote Smoke Attempt

Date: 2026-05-13, heartbeat `probe-v2-experiment-follow-up`

Remote SSH succeeded on this heartbeat. Synced the minimum remote pieces needed for scalar probe-retry smoke:

```text
src/fhis/probe_retry_router.py
src/fhis/online_router.py scalar probe load/scoring patch
data_generation/qwen25_fhis/configs/probe_retry_v2_scalars_c5_p4_smoke.yaml
classifier/v2_runs/online_smoke/problems_10_from_traces.jsonl
```

The original remote smoke problem files were Git LFS pointers, so the remote smoke set was regenerated from `generated_traces.jsonl` as `classifier/v2_runs/online_smoke/problems_10_from_traces.jsonl`.

Remote compile passed:

```bash
conda run -n fhis-v2 python -m py_compile \
  src/fhis/online_router.py \
  src/fhis/probe_retry_router.py \
  src/fhis/train_probe_v2.py \
  src/fhis/evaluate_online.py
```

## Heartbeat Follow-Up: Blocker Recheck

Date: 2026-05-14, heartbeat `probe-v2-experiment-follow-up`

Checked local and remote state.

Local:

- No local FHIS/probe training or smoke process is running.
- No local `Qwen2.5-Math-7B-Instruct` cache was found under `/home/suyc24` or `/mnt/c/Users/77838`.

Remote:

- No remote FHIS/probe training or smoke process is running.
- Available remote model directories are still:
  - `/root/shared-nvme/models/Qwen2.5-Coder-0.5B-Instruct`
  - `/root/shared-nvme/models/Qwen2.5-Coder-1.5B-Instruct`
  - `/root/shared-nvme/models/Qwen3-4B`
  - `/root/shared-nvme/models/StepFun-Formalizer-7B`
- Required model directory is still missing:
  - `/root/shared-nvme/models/Qwen2.5-Math-7B-Instruct`
- Remote network check to `huggingface.co:443` still fails with `[Errno 101] Network is unreachable`.
- Existing smoke log confirms the previous smoke attempt failed while loading `Qwen/Qwen2.5-Math-7B-Instruct` from Hugging Face, before any usable online result was produced.

Decision: keep the online probe-retry smoke blocked. Do not retry until either the exact local model directory exists on the remote server or Hugging Face connectivity is restored and the exact model can be downloaded.

The smoke run was attempted and failed before inference because the remote server cannot load `Qwen/Qwen2.5-Math-7B-Instruct`:

```text
[Errno 101] Network is unreachable while requesting Hugging Face config.json
OSError: Can't load the configuration of 'Qwen/Qwen2.5-Math-7B-Instruct'
```

Remote model cache inspection found only:

```text
/root/shared-nvme/models/Qwen2.5-Coder-0.5B-Instruct
/root/shared-nvme/models/Qwen2.5-Coder-1.5B-Instruct
/root/shared-nvme/models/Qwen3-4B
/root/shared-nvme/models/StepFun-Formalizer-7B
```

No local `Qwen2.5-Math-7B-Instruct` cache was available. Do not run the v2 probe-retry smoke against Qwen3/Qwen-Coder as a substitute, because the hidden-state probe was trained on Qwen2.5-Math-7B layers and hidden geometry. Next useful step is to make the correct model available on the remote server, or run the online smoke in an environment that already has that exact model cached.

## Heartbeat Follow-Up: Model Transfer Plan

Date: 2026-05-13, heartbeat `probe-v2-experiment-follow-up`

Local cache search also found no `Qwen2.5-Math-7B-Instruct` snapshot under the visible Linux/Windows user paths. Since both remote download and local transfer are currently blocked by missing model files, added a dedicated setup note:

```text
classifier/qwen25_math7b_remote_setup.md
```

That file records:

- Why Qwen3/Qwen-Coder/StepFun substitutes should not be used for this hidden-state probe.
- The prepared remote smoke config and output paths.
- How to download the exact model on remote if Hugging Face access returns.
- How to transfer a local cached snapshot if one becomes available.
- A local-files-only readiness check before re-running smoke.

The heartbeat cadence was reduced from hourly to every 6 hours while this external model-availability blocker remains.

## Mirror Download And Online Smoke

Date: 2026-05-14

The remote server could reach the Hugging Face mirror but not direct Hugging Face:

```text
https://hf-mirror.com reachable
https://huggingface.co timed out
```

Downloaded the exact required model through the mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com conda run -n fhis-v2 hf download \
  Qwen/Qwen2.5-Math-7B-Instruct \
  --local-dir /root/shared-nvme/models/Qwen2.5-Math-7B-Instruct
```

Download result:

```text
model dir: /root/shared-nvme/models/Qwen2.5-Math-7B-Instruct
model size: 15G
files: 4 safetensors shards plus config/tokenizer files
remote disk after download: 42G used, 8.1G free on /root/shared-nvme
```

Readiness check passed with local files only:

```text
Qwen2Config qwen2
Qwen2Tokenizer
```

Updated the remote smoke config:

```yaml
model:
  name: /root/shared-nvme/models/Qwen2.5-Math-7B-Instruct
```

Ran the prepared scalar probe smoke:

```bash
conda run -n fhis-v2 python -m fhis.probe_retry_router \
  --config data_generation/qwen25_fhis/configs/probe_retry_v2_scalars_c5_p4_smoke.yaml \
  --limit 10
```

The generator completed all 10 problems. The original chained evaluation command failed because `evaluate_online.py` expected `paths.online_results`, while the probe-retry config stores the output path under `probe_retry.probe_retry_results`. Re-ran evaluation with an explicit results path:

```bash
conda run -n fhis-v2 python -m fhis.evaluate_online \
  --config data_generation/qwen25_fhis/configs/probe_retry_v2_scalars_c5_p4_smoke.yaml \
  --results classifier/v2_runs/online_smoke/scalars_c5_p4_retry_t05_results.jsonl \
  --output classifier/v2_runs/online_smoke/scalars_c5_p4_retry_t05_summary.json
```

Smoke summary:

| Metric | Value |
|---|---:|
| num_problems | 10 |
| accepted | 4 |
| abstained | 6 |
| answer_rate | 0.4 |
| rough_solve_rate_all | 0.2 |
| rough_solve_rate_answered | 0.5 |
| generated_steps | 45 |
| attempts | 10 |
| avg_attempts_per_problem | 1.0 |

Probe/retry behavior:

| Count | Value |
|---|---:|
| decisions | 45 |
| flagged decisions | 15 |
| retry_step actions | 11 |
| restart_trace actions | 6 |
| parse_bad decisions | 2 |

Interpretation: the exact-model blocker is resolved and the smoke route works, but threshold `0.5` still causes aggressive step retries. Several retries either failed parsing or generated low-quality/degenerate continuations, leading to a high abstain rate. The next highest-value experiment is not another model download; it is a policy/threshold sweep for online routing, likely comparing `threshold=0.7/0.8/0.9`, fewer retry attempts, and stronger retry prompt constraints against a no-probe or majority baseline on the same 10-problem smoke set.

## Online Threshold Sweep

Date: 2026-05-14

Ran a small online policy sweep for the current best scalar probe `scalars_c5_p4` on the same 10-problem smoke set. All runs used the exact local model path:

```text
/root/shared-nvme/models/Qwen2.5-Math-7B-Instruct
```

Remote artifacts:

```text
classifier/v2_runs/online_policy_sweep/scalars_c5_p4_t07_results.jsonl
classifier/v2_runs/online_policy_sweep/scalars_c5_p4_t07_summary.json
classifier/v2_runs/online_policy_sweep/scalars_c5_p4_t08_results.jsonl
classifier/v2_runs/online_policy_sweep/scalars_c5_p4_t08_summary.json
classifier/v2_runs/online_policy_sweep/scalars_c5_p4_t09_results.jsonl
classifier/v2_runs/online_policy_sweep/scalars_c5_p4_t09_summary.json
classifier/v2_runs/online_policy_sweep/scalars_c5_p4_threshold_sweep_summary.json
```

Policy details:

- Thresholds: `0.7`, `0.8`, `0.9`.
- `max_step_attempts=2`.
- `accept_best_after_max_step_attempts=true`.
- `accept_best_margin=0.0`, stricter than the first `t05` smoke margin of `0.05`.

Summary:

| Threshold | Accepted | Abstained | Answer Rate | Rough Solve All | Rough Solve Answered | Decisions | Flagged | retry_step | restart_trace | parse_bad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 4 | 6 | 0.4 | 0.2 | 0.5 | 45 | 15 | 11 | 6 | 2 |
| 0.7 | 3 | 7 | 0.3 | 0.3 | 1.0 | 41 | 15 | 9 | 7 | 1 |
| 0.8 | 7 | 3 | 0.7 | 0.4 | 0.5714 | 47 | 10 | 8 | 3 | 1 |
| 0.9 | 10 | 0 | 1.0 | 0.3 | 0.3 | 54 | 0 | 0 | 0 | 0 |

Interpretation:

- `threshold=0.8` is the best current online point on this smoke set: it improves rough solve rate all from `0.2` at `t05` to `0.4`, while cutting restarts from `6` to `3` and abstentions from `6` to `3`.
- `threshold=0.7` is too aggressive for this policy. It still flags 15 decisions and has the worst abstain count.
- `threshold=0.9` is a useful control: it triggers no probe intervention at all on this smoke set, accepts every trace, but rough solve rate drops to `0.3`. This suggests probe intervention is helpful when calibrated, but over-intervention causes abstention and retry degeneration.

Current recommendation: use `scalars_c5_p4` around threshold `0.8` for the next online experiment, and focus the next change on retry quality rather than classifier training. The retry prompt currently often asks the model to redo a step but still permits long, repetitive, or malformed continuations. The next policy test should keep `threshold=0.8` and compare retry prompt constraints, for example shorter retry generation (`max_step_new_tokens=256`) and a stricter instruction to output exactly one step without restating previous steps.

## Retry Output Length Variant

Date: 2026-05-14

Tested a retry-quality variant at the best current threshold point:

```text
variant: scalars_c5_p4_t08_tok256
generator model: /root/shared-nvme/models/Qwen2.5-Math-7B-Instruct
threshold: 0.8
max_step_new_tokens: 256
max_step_attempts: 2
accept_best_after_max_step_attempts: true
accept_best_margin: 0.0
```

Remote artifacts:

```text
data_generation/qwen25_fhis/configs/probe_retry_v2_scalars_c5_p4_t08_tok256_smoke.yaml
classifier/v2_runs/online_policy_sweep/scalars_c5_p4_t08_tok256_results.jsonl
classifier/v2_runs/online_policy_sweep/scalars_c5_p4_t08_tok256_summary.json
classifier/v2_runs/online_policy_sweep/scalars_c5_p4_t08_tok256_stats.json
classifier/v2_runs/online_policy_sweep/scalars_c5_p4_t08_tok256.log
```

Result compared with the previous `t08` run:

| Variant | Accepted | Abstained | Answer Rate | Rough Solve All | Rough Solve Answered | Decisions | Flagged | retry_step | restart_trace | parse_bad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t08 | 7 | 3 | 0.7 | 0.4 | 0.5714 | 47 | 10 | 8 | 3 | 1 |
| t08_tok256 | 4 | 6 | 0.4 | 0.2 | 0.5 | 50 | 11 | 6 | 6 | 2 |

Interpretation: simply shortening `max_step_new_tokens` to 256 is worse. It reduces some retry continuations but increases abstention and restarts, likely because the corrected step is more often incomplete or still high-scoring after retry. The next retry-quality experiment should not reduce the token limit further. A better next test is a stricter retry prompt and/or an acceptance-policy change:

1. Prompt change: instruct retry generation to output exactly one complete mathematical step, no preamble, no restatement of previous accepted steps, no final answer unless the step naturally finishes the solution.
2. Acceptance change: after one retry at threshold `0.8`, accept the lowest-score parseable candidate if it improves over the original by a meaningful margin, instead of restarting whenever it remains just above threshold.
3. Optional decoding change: lower retry-only temperature, while keeping normal first-pass generation unchanged.

