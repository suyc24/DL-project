from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fhis.config import load_config, selected_layers_for_model
from fhis.io import append_jsonl, read_jsonl
from fhis.online_router import load_probe, score_feature, step_metadata_row
from fhis.prompting import apply_qwen_chat_template, build_user_prompt
from fhis.steps import extract_steps, step_end_token_indices


def score_completion_steps(
    *,
    model: Any,
    tokenizer: Any,
    probe: Any,
    layer_ids: list[int],
    prompt: str,
    completion: str,
) -> dict[str, Any]:
    import torch

    steps = extract_steps(completion)
    if not steps:
        return {
            "parse_ok": False,
            "num_steps": 0,
            "step_scores": [],
            "max_score": None,
            "mean_score": None,
            "top2_mean_score": None,
        }
    indices = step_end_token_indices(tokenizer, prompt, completion, steps)
    if not indices:
        return {
            "parse_ok": False,
            "num_steps": len(steps),
            "step_scores": [],
            "max_score": None,
            "mean_score": None,
            "top2_mean_score": None,
        }

    encoded = tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    seq_len = outputs.hidden_states[0].shape[1]

    step_scores: list[dict[str, Any]] = []
    for step in steps:
        token_index = indices.get(step.index)
        if token_index is None:
            continue
        token_index = min(token_index, seq_len - 1)
        per_layer = torch.stack(
            [outputs.hidden_states[layer_id + 1][0, token_index] for layer_id in layer_ids],
            dim=0,
        )
        score = score_feature(
            probe,
            per_layer.float().cpu().reshape(-1),
            row=step_metadata_row(step.index, step.text),
        )
        step_scores.append({"step_index": step.index, "score": score})

    scores = [row["score"] for row in step_scores]
    if not scores:
        return {
            "parse_ok": False,
            "num_steps": len(steps),
            "step_scores": [],
            "max_score": None,
            "mean_score": None,
            "top2_mean_score": None,
        }
    top_scores = sorted(scores, reverse=True)[:2]
    return {
        "parse_ok": True,
        "num_steps": len(steps),
        "scored_steps": len(scores),
        "step_scores": step_scores,
        "max_score": max(scores),
        "mean_score": sum(scores) / len(scores),
        "top2_mean_score": sum(top_scores) / len(top_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score generated candidate pools with FHIS probe.")
    parser.add_argument("--config", default="data_generation/qwen25_fhis/configs/probe_retry_mlp_rate05_balanced.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--problems", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    config = load_config(args.config)
    model_cfg = config["model"]
    layer_ids = selected_layers_for_model(config)
    router_cfg = config["router"]
    probe = load_probe(router_cfg["probe_model"])

    rows = list(read_jsonl(args.input))
    problem_map = {}
    if args.problems:
        problem_map = {
            str(problem.get("problem_id", problem.get("id", "unknown"))): problem
            for problem in read_jsonl(args.problems)
        }
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.resume:
        done = {str(row.get("problem_id")) for row in read_jsonl(args.output)}
        rows = [row for row in rows if str(row.get("problem_id")) not in done]
    elif Path(args.output).exists():
        Path(args.output).unlink()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    dtype_name = str(model_cfg.get("dtype", "auto"))
    torch_dtype: Any = "auto"
    if dtype_name == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype_name == "float16":
        torch_dtype = torch.float16
    elif dtype_name == "float32":
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    for row in rows:
        problem = problem_map.get(str(row.get("problem_id")), row)
        problem_text = str(problem.get("problem", ""))
        prompt = (
            apply_qwen_chat_template(
                tokenizer,
                build_user_prompt(problem_text),
                enable_thinking=bool(model_cfg.get("enable_thinking", True)),
            )
            if problem_text
            else ""
        )
        scored_samples = []
        for sample in row.get("samples", []):
            completion = str(sample.get("completion") or "")
            scored = dict(sample)
            scored["probe_risk"] = score_completion_steps(
                model=model,
                tokenizer=tokenizer,
                probe=probe,
                layer_ids=layer_ids,
                prompt=prompt,
                completion=completion,
            )
            scored_samples.append(scored)
        output_row = dict(row)
        output_row["samples"] = scored_samples
        append_jsonl(args.output, [output_row])
        print(json.dumps({"problem_id": row.get("problem_id"), "samples": len(scored_samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
