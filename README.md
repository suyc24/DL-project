# FHIS Verification Signal v0

This repository is a first runnable scaffold for the v0 experiment in
`refine-logs/FINAL_PROPOSAL.md`: test whether Qwen hidden states at structured
reasoning step boundaries can identify the first harmful invalid step (FHIS).

The pipeline is intentionally modular:

1. Sample a small MATH subset.
2. Generate structured reasoning traces with vLLM.
3. Label traces with a first-pass annotator.
4. Extract step-boundary hidden states.
5. Train probes and compare simple baselines.

vLLM is used for high-throughput generation. Hidden-state extraction follows the
current vLLM hidden-state connector pattern: `extract_hidden_states` speculative
method plus `ExampleHiddenStatesConnector`, which writes token-level hidden
states to safetensors files. Because this vLLM feature is version-sensitive, a
transformers fallback extractor is included only for local debugging and smoke
tests.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For GPT-based annotation:

```bash
export OPENAI_API_KEY=...
```

## Minimal Run

```bash
python -m fhis.math_dataset --config configs/experiment.yaml --limit 30
python -m fhis.vllm_generate --config configs/experiment.yaml --limit 30
python -m fhis.label_with_openai --config configs/experiment.yaml --limit 20
python -m fhis.extract_hidden_states_vllm --config configs/experiment.yaml --limit 20
python -m fhis.train_probe --config configs/experiment.yaml
```

For a CPU-only smoke test of parsing and metrics:

```bash
pytest
```

## Main Artifacts

- `data/generated_traces.jsonl`
- `data/fhis_labels.jsonl`
- `data/step_hidden_states.pt`
- `results/probe_metrics.json`
- `results/layer_sweep.csv`
- `figures/recall_at_k.png`
- `figures/top_budget_coverage.png`

## Notes

- Splits are problem-disjoint.
- Wrong traces label only the FHIS step as positive; later steps are excluded.
- Correct high-confidence traces label all steps as negative.
- Hidden-state features concatenate selected layer states at each step end.
