# Qwen2.5-Math-7B Remote Setup For Probe v2 Smoke

This is the current blocker for the Probe Classifier v2 online smoke test.

The v2 probe was trained on hidden states from:

```text
Qwen/Qwen2.5-Math-7B-Instruct
```

Do not substitute Qwen3, Qwen-Coder, or StepFun-Formalizer for the smoke test. The classifier depends on the hidden-state geometry of this exact model and selected layers `[6, 13, 20, 27]`.

## Current Remote State

Remote workspace:

```text
/root/shared-nvme/DL-project
```

Prepared smoke config:

```text
data_generation/qwen25_fhis/configs/probe_retry_v2_scalars_c5_p4_smoke.yaml
```

Prepared smoke problem file on remote:

```text
classifier/v2_runs/online_smoke/problems_10_from_traces.jsonl
```

Prepared probe:

```text
classifier/v2_runs/sweep_scalars/scalars_c5_p4/probe.joblib
```

Known remote model directories available as of 2026-05-13:

```text
/root/shared-nvme/models/Qwen2.5-Coder-0.5B-Instruct
/root/shared-nvme/models/Qwen2.5-Coder-1.5B-Instruct
/root/shared-nvme/models/Qwen3-4B
/root/shared-nvme/models/StepFun-Formalizer-7B
```

Available as of 2026-05-14 after mirror download:

```text
/root/shared-nvme/models/Qwen2.5-Math-7B-Instruct
```

Resolved on 2026-05-14:

- The exact model directory is now available after mirror download.
- Direct `huggingface.co:443` access still times out, but `hf-mirror.com` worked.
- Local-files-only config/tokenizer readiness check passed.
- The first `scalars_c5_p4` smoke generated 10 results and was evaluated. Summary: accepted 4, abstained 6, answer rate 0.4, rough solve rate all 0.2.

## Option A: Download On Remote

Use this if remote Hugging Face access is restored.

```bash
conda run -n fhis-v2 huggingface-cli download \
  Qwen/Qwen2.5-Math-7B-Instruct \
  --local-dir /root/shared-nvme/models/Qwen2.5-Math-7B-Instruct \
  --local-dir-use-symlinks False
```

Then edit the remote smoke config so `model.name` points to the local directory:

```yaml
model:
  name: /root/shared-nvme/models/Qwen2.5-Math-7B-Instruct
```



## Option C: Download Through Hugging Face Mirror

This path succeeded on 2026-05-14 while direct `huggingface.co` access timed out:

```bash
HF_ENDPOINT=https://hf-mirror.com conda run -n fhis-v2 hf download \
  Qwen/Qwen2.5-Math-7B-Instruct \
  --local-dir /root/shared-nvme/models/Qwen2.5-Math-7B-Instruct
```

Post-download state:

```text
model size: 15G
remote disk free on /root/shared-nvme: about 8.1G
readiness check: passed
```

The prepared smoke config should now use the local model path:

```yaml
model:
  name: /root/shared-nvme/models/Qwen2.5-Math-7B-Instruct
```

## Option B: Transfer From A Machine With Cache

Use this if another machine already has the exact model snapshot.

Find the local snapshot:

```bash
find ~/.cache/huggingface/hub -path '*models--Qwen--Qwen2.5-Math-7B-Instruct*' -maxdepth 5
```

Copy the resolved snapshot directory to remote:

```bash
rsync -avP -e "ssh -p 2233" \
  /path/to/Qwen2.5-Math-7B-Instruct/ \
  'root@ackcs-00gjgu50'@ssh.bj8.bz1.paratera.com:/root/shared-nvme/models/Qwen2.5-Math-7B-Instruct/
```

If `rsync` is not available, use SFTP in multiple chunks. Avoid pasting model files through the interactive terminal.

## Readiness Check

On remote:

```bash
cd /root/shared-nvme/DL-project
conda run -n fhis-v2 python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
path = "/root/shared-nvme/models/Qwen2.5-Math-7B-Instruct"
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
print(type(cfg).__name__, cfg.model_type)
print(type(tok).__name__)
PY
```

Only after this passes should the online smoke be retried.

## Smoke Command

On remote, after the readiness check:

```bash
cd /root/shared-nvme/DL-project
mkdir -p classifier/v2_runs/online_smoke
rm -f \
  classifier/v2_runs/online_smoke/scalars_c5_p4_retry_t05_results.jsonl \
  classifier/v2_runs/online_smoke/scalars_c5_p4_retry_t05_summary.json \
  classifier/v2_runs/online_smoke/scalars_c5_p4_retry_t05.log

nohup bash -lc 'PYTHONUNBUFFERED=1 conda run -n fhis-v2 python -m fhis.probe_retry_router --config data_generation/qwen25_fhis/configs/probe_retry_v2_scalars_c5_p4_smoke.yaml --limit 10 && conda run -n fhis-v2 python -m fhis.evaluate_online --config data_generation/qwen25_fhis/configs/probe_retry_v2_scalars_c5_p4_smoke.yaml --output classifier/v2_runs/online_smoke/scalars_c5_p4_retry_t05_summary.json' \
  > classifier/v2_runs/online_smoke/scalars_c5_p4_retry_t05.log 2>&1 &
```

Expected outputs:

```text
classifier/v2_runs/online_smoke/scalars_c5_p4_retry_t05_results.jsonl
classifier/v2_runs/online_smoke/scalars_c5_p4_retry_t05_summary.json
classifier/v2_runs/online_smoke/scalars_c5_p4_retry_t05.log
```
