from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer

from fhis.config import load_config
from fhis.io import append_jsonl, read_jsonl
from fhis.prompting import apply_qwen_chat_template, build_user_prompt
from fhis.steps import (
    extract_final_answer,
    extract_steps,
    rough_answer_match,
    steps_as_dicts,
)


def chosen_logprob(logprob_row: Any, token_id: int) -> float | None:
    if logprob_row is None:
        return None
    value = None
    if isinstance(logprob_row, dict):
        value = logprob_row.get(token_id)
        if value is None and str(token_id) in logprob_row:
            value = logprob_row[str(token_id)]
    if value is None and isinstance(logprob_row, dict) and logprob_row:
        value = next(iter(logprob_row.values()))
    if value is None:
        return None
    if hasattr(value, "logprob"):
        return float(value.logprob)
    return float(value)


def build_trace_rows(
    problem: dict[str, Any],
    prompt: str,
    completion_outputs: Iterable[Any],
    generation_config: dict[str, Any],
    model_name: str,
    completion_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_index, output in enumerate(completion_outputs):
        completion = completion_prefix + output.text
        steps = extract_steps(completion)
        final_answer = extract_final_answer(completion)
        token_ids = list(getattr(output, "token_ids", []) or [])
        logprob_rows = list(getattr(output, "logprobs", []) or [])
        token_logprobs = [
            chosen_logprob(logprob_rows[i], token_id) if i < len(logprob_rows) else None
            for i, token_id in enumerate(token_ids)
        ]
        trace_id = f"{problem['problem_id']}::sample-{sample_index}"
        rows.append(
            {
                "trace_id": trace_id,
                "problem_id": problem["problem_id"],
                "dataset": problem.get("dataset"),
                "subset": problem.get("subset"),
                "model_name": model_name,
                "sample_index": sample_index,
                "subject": problem["subject"],
                "level": problem["level"],
                "source_split": problem.get("source_split"),
                "generation_config": generation_config,
                "completion_prefix": completion_prefix,
                "problem": problem["problem"],
                "reference_solution": problem["reference_solution"],
                "reference_answer": problem["reference_answer"],
                "prompt": prompt,
                "completion": completion,
                "steps": steps_as_dicts(steps),
                "final_answer": final_answer,
                "rough_final_correct": rough_answer_match(final_answer, problem["reference_answer"]),
                "token_ids": token_ids,
                "token_logprobs": token_logprobs,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structured MATH traces with vLLM.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--problems", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    config = load_config(args.config)
    model_cfg = config["model"]
    gen_cfg = config["generation"]
    problems_path = args.problems or config["paths"]["problems"]
    output_path = args.output or config["paths"]["generated_traces"]
    problems = list(read_jsonl(problems_path))
    if args.limit is not None:
        problems = problems[: args.limit]

    done_problem_ids: set[str] = set()
    if args.resume:
        for row in read_jsonl(output_path):
            done_problem_ids.add(str(row["problem_id"]))
        problems = [p for p in problems if str(p["problem_id"]) not in done_problem_ids]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    completion_prefix = str(gen_cfg.get("completion_prefix", ""))
    base_prompts = [
        apply_qwen_chat_template(
            tokenizer,
            build_user_prompt(p["problem"]),
            enable_thinking=bool(model_cfg.get("enable_thinking", True)),
        )
        for p in problems
    ]
    prompts = [prompt + completion_prefix for prompt in base_prompts]

    llm_kwargs = {
        "model": model_cfg["name"],
        "tensor_parallel_size": int(model_cfg.get("tensor_parallel_size", 1)),
        "dtype": model_cfg.get("dtype", "auto"),
        "gpu_memory_utilization": float(model_cfg.get("gpu_memory_utilization", 0.9)),
        "max_model_len": int(model_cfg.get("max_model_len", 8192)),
        "trust_remote_code": True,
    }
    if model_cfg.get("attention_backend"):
        llm_kwargs["attention_config"] = {"backend": model_cfg["attention_backend"]}
    llm = LLM(**llm_kwargs)
    temperature = float(gen_cfg.get("temperature", 0.7))
    n_samples = int(gen_cfg.get("n_samples_per_problem", 3))
    if temperature == 0.0 and n_samples != 1:
        print("temperature=0 uses greedy sampling in vLLM; forcing n_samples_per_problem=1")
        n_samples = 1

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=float(gen_cfg.get("top_p", 0.95)),
        n=n_samples,
        max_tokens=int(gen_cfg.get("max_new_tokens", 4096)),
        logprobs=int(gen_cfg.get("logprobs", 1)),
    )
    generation_config = {
        "temperature": temperature,
        "top_p": float(gen_cfg.get("top_p", 0.95)),
        "n_samples_per_problem": n_samples,
        "max_new_tokens": int(gen_cfg.get("max_new_tokens", 4096)),
        "logprobs": int(gen_cfg.get("logprobs", 1)),
        "completion_prefix": completion_prefix,
    }

    outputs = llm.generate(prompts, sampling_params)
    for problem, base_prompt, request_output in tqdm(
        zip(problems, base_prompts, outputs, strict=True),
        total=len(problems),
        desc="writing traces",
    ):
        rows = build_trace_rows(
            problem,
            base_prompt,
            request_output.outputs,
            generation_config,
            model_cfg["name"],
            completion_prefix,
        )
        append_jsonl(output_path, rows)

    print(f"Wrote traces for {len(problems)} problems to {output_path}")


if __name__ == "__main__":
    main()
