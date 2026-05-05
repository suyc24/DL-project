# Qwen2.5 FHIS data generation

This folder contains the first-stage data generation pipeline for AG-SFV. It
collects Qwen2.5-Math reasoning traces, labels first harmful incorrect steps
with local Codex, extracts step-boundary hidden states, and trains the first
probe.

The full dataset card is in
[`docs/qwen25_fhis_dataset.md`](../docs/qwen25_fhis_dataset.md).

Default target:

- Dataset: `Hothan/OlympiadBench`
- Subset: `OE_TO_maths_en_COMP`
- Model: `Qwen/Qwen2.5-Math-7B-Instruct`
- Problems: `200`
- Samples per problem: `4`
- Context: `4096`, with `3072` max generated tokens
- Expected accuracy: about `41.6%` from the Qwen2.5-Math official CoT pass@1
  table on OlympiadBench, so the wrong-trace rate should be high enough for
  probe training.

Run from the repository root:

```bash
python data_generation/generate_olympiadbench_traces.py \
  --config data_generation/recommended_config.yaml
```

For a small connectivity and formatting check:

```bash
python data_generation/generate_olympiadbench_traces.py \
  --config data_generation/recommended_config.yaml \
  --limit 2
```

Outputs are written under `data_generation/outputs/` by default:

- `problems.jsonl`
- `generated_traces.jsonl`
- `summary.json`

Use `--resume` to skip problems that already have generated traces in the output
JSONL. If `generated_traces.jsonl` already exists and `--resume` is not set, the
script stops by default. Use `--overwrite` only when you intentionally want to
replace the existing trace file.

Label with local Codex:

```bash
python data_generation/label_with_local_codex.py \
  --traces data_generation/outputs/generated_traces.jsonl \
  --output data_generation/labels/fhis_labels.jsonl \
  --schema data_generation/local_codex_label_schema.json \
  --resume
```

Extract hidden states and train the probe:

```bash
python -m fhis.extract_hidden_states_transformers \
  --config data_generation/qwen25_probe_config.yaml

python -m fhis.train_probe \
  --config data_generation/qwen25_probe_config.yaml
```

Generated artifacts under `outputs/`, `labels/`, `features/`, and `results/`
are intentionally ignored by git.
