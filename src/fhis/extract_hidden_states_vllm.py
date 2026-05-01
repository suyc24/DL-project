from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoTokenizer

from fhis.config import load_config, selected_layers_for_model
from fhis.io import read_jsonl
from fhis.labeling import fhis_step_label
from fhis.steps import extract_steps, step_end_token_indices, steps_as_dicts, steps_from_dicts


def mean_step_logprob(tokenizer: Any, completion: str, trace: dict[str, Any], step: dict[str, Any]) -> float:
    token_logprobs = trace.get("token_logprobs") or []
    if not token_logprobs:
        return float("nan")
    encoded = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    values: list[float] = []
    for i, (start, end) in enumerate(offsets):
        if i >= len(token_logprobs):
            break
        if start >= int(step["start_char"]) and end <= int(step["end_char"]):
            value = token_logprobs[i]
            if value is not None:
                values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def build_feature_rows(
    trace: dict[str, Any],
    label: dict[str, Any],
    hidden_states: torch.Tensor,
    tokenizer: Any,
    layer_ids: list[int],
    keep_confidence: str,
) -> list[dict[str, Any]]:
    step_dicts = trace.get("steps") or steps_as_dicts(extract_steps(trace["completion"]))
    steps = steps_from_dicts(step_dicts)
    step_token_indices = step_end_token_indices(tokenizer, trace["prompt"], trace["completion"], steps)
    rows: list[dict[str, Any]] = []
    for step_dict in step_dicts:
        step_index = int(step_dict["index"])
        y = fhis_step_label(label, step_index, keep_confidence=keep_confidence)
        if y is None or step_index not in step_token_indices:
            continue
        token_index = min(step_token_indices[step_index], hidden_states.shape[0] - 1)
        per_layer = hidden_states[token_index].float().cpu()
        feature = per_layer.reshape(-1)
        rows.append(
            {
                "trace_id": trace["trace_id"],
                "problem_id": trace["problem_id"],
                "subject": trace.get("subject"),
                "level": trace.get("level"),
                "step_index": step_index,
                "step_text": step_dict.get("text", ""),
                "label": int(y),
                "trace_final_correct": bool(label.get("final_correct", False)),
                "layer_ids": layer_ids,
                "token_index": int(token_index),
                "feature": feature,
                "baselines": {
                    "step_index": float(step_index),
                    "step_length_chars": float(len(step_dict.get("text", ""))),
                    "mean_token_logprob": mean_step_logprob(
                        tokenizer, trace["completion"], trace, step_dict
                    ),
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract vLLM step-end hidden states using the vLLM hidden-state connector."
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--traces", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    config = load_config(args.config)
    model_cfg = config["model"]
    layer_ids = selected_layers_for_model(config)
    keep_confidence = config["labels"].get("keep_confidence", "high")
    traces_path = args.traces or config["paths"]["generated_traces"]
    labels_path = args.labels or config["paths"]["labels"]
    output_path = Path(args.output or config["paths"]["hidden_states"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = {row["trace_id"]: row for row in read_jsonl(labels_path)}
    traces = [row for row in read_jsonl(traces_path) if row["trace_id"] in labels]
    if args.limit is not None:
        traces = traces[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    full_prompts = [trace["prompt"] + trace["completion"] for trace in traces]
    feature_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        llm = LLM(
            model=model_cfg["name"],
            tensor_parallel_size=int(model_cfg.get("tensor_parallel_size", 1)),
            dtype=model_cfg.get("dtype", "auto"),
            gpu_memory_utilization=float(model_cfg.get("gpu_memory_utilization", 0.9)),
            max_model_len=int(model_cfg.get("max_model_len", 8192)),
            trust_remote_code=True,
            attention_config=(
                {"backend": model_cfg["attention_backend"]}
                if model_cfg.get("attention_backend")
                else None
            ),
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": layer_ids,
                    }
                },
            },
            kv_transfer_config={
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": tmpdir,
                },
            },
        )
        outputs = llm.generate(full_prompts, SamplingParams(max_tokens=1, temperature=0.0))

        for trace, output in tqdm(
            zip(traces, outputs, strict=True), total=len(traces), desc="extracting states"
        ):
            hidden_states_path = (getattr(output, "kv_transfer_params", None) or {}).get(
                "hidden_states_path"
            )
            if not hidden_states_path:
                raise RuntimeError(
                    "vLLM did not return kv_transfer_params.hidden_states_path. "
                    "Check that your vLLM version supports extract_hidden_states."
                )
            with safe_open(hidden_states_path, framework="pt") as f:
                hidden_states = f.get_tensor("hidden_states")
            feature_rows.extend(
                build_feature_rows(
                    trace,
                    labels[trace["trace_id"]],
                    hidden_states,
                    tokenizer,
                    layer_ids,
                    keep_confidence,
                )
            )

    torch.save({"rows": feature_rows, "layer_ids": layer_ids, "model": model_cfg["name"]}, output_path)
    print(f"Wrote {len(feature_rows)} labeled step features to {output_path}")


if __name__ == "__main__":
    main()
