from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"Could not find repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fhis.io import append_jsonl, read_jsonl, write_jsonl  # noqa: E402
from fhis.prompting import apply_qwen_chat_template, build_user_prompt  # noqa: E402
from fhis.steps import (  # noqa: E402
    extract_final_answer,
    extract_steps,
    rough_answer_match,
    steps_as_dicts,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyyaml. Install project dependencies with "
            '`pip install -e ".[dev]"` or `pip install pyyaml`.'
        ) from exc

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def as_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def rough_any_answer_match(predicted: str | None, references: list[str]) -> bool | None:
    if predicted is None or not references:
        return None
    return any(rough_answer_match(predicted, reference) for reference in references)


def choose_problem_text(item: dict[str, Any]) -> str:
    context = str(item.get("context") or "").strip()
    question = str(item.get("question") or "").strip()
    if context:
        return f"{context}\n\n{question}".strip()
    return question


def load_olympiadbench_problems(
    config: dict[str, Any],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset_cfg = config["dataset"]
    seed = int(config.get("seed", 0))
    target = int(dataset_cfg.get("target_problems", 200))
    if limit is not None:
        target = min(target, int(limit))

    ds = load_dataset(
        dataset_cfg["hf_name"],
        dataset_cfg["subset"],
        split=dataset_cfg.get("split", "train"),
    )

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(ds):
        problem = choose_problem_text(dict(item))
        if not problem:
            continue
        rows.append(
            {
                "problem_id": f"{dataset_cfg['subset']}-{idx}",
                "source_id": item.get("id", idx),
                "dataset": dataset_cfg["hf_name"],
                "subset": dataset_cfg["subset"],
                "source_split": dataset_cfg.get("split", "train"),
                "problem": problem,
                "reference_solution": item.get("solution"),
                "reference_answer": as_answer_list(item.get("final_answer")),
                "question_type": item.get("question_type"),
                "subject": item.get("subject"),
                "subfield": item.get("subfield"),
                "language": item.get("language"),
                "difficulty": item.get("difficulty"),
                "answer_type": item.get("answer_type"),
                "is_multiple_answer": item.get("is_multiple_answer"),
                "unit": item.get("unit"),
            }
        )

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:target]


def chosen_logprob(logprob_row: Any, token_id: int) -> float | None:
    if logprob_row is None:
        return None
    value = None
    if isinstance(logprob_row, dict):
        value = logprob_row.get(token_id)
        if value is None:
            value = logprob_row.get(str(token_id))
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
                "dataset": problem["dataset"],
                "subset": problem["subset"],
                "source_id": problem["source_id"],
                "model_name": model_name,
                "sample_index": sample_index,
                "generation_config": generation_config,
                "completion_prefix": completion_prefix,
                "problem": problem["problem"],
                "reference_solution": problem["reference_solution"],
                "reference_answer": problem["reference_answer"],
                "prompt": prompt,
                "completion": completion,
                "steps": steps_as_dicts(steps),
                "final_answer": final_answer,
                "rough_final_correct": rough_any_answer_match(
                    final_answer,
                    problem["reference_answer"],
                ),
                "token_ids": token_ids,
                "token_logprobs": token_logprobs,
            }
        )
    return rows


def summarize_traces(path: str | Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    correct = [row for row in rows if row.get("rough_final_correct") is True]
    wrong = [row for row in rows if row.get("rough_final_correct") is False]
    unknown = [row for row in rows if row.get("rough_final_correct") is None]
    parseable_steps = [row for row in rows if row.get("steps")]
    return {
        "num_traces": len(rows),
        "num_problems": len({row.get("problem_id") for row in rows}),
        "rough_correct": len(correct),
        "rough_wrong": len(wrong),
        "rough_unknown": len(unknown),
        "rough_accuracy": (len(correct) / len(correct + wrong)) if correct or wrong else None,
        "step_parse_rate": (len(parseable_steps) / len(rows)) if rows else None,
    }


def generate_traces(config: dict[str, Any], problems: list[dict[str, Any]], resume: bool) -> None:
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_cfg = config["model"]
    gen_cfg = config["generation"]
    output_path = config["paths"]["generated_traces"]

    if resume:
        done_problem_ids = {str(row["problem_id"]) for row in read_jsonl(output_path)}
        problems = [p for p in problems if str(p["problem_id"]) not in done_problem_ids]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    completion_prefix = str(gen_cfg.get("completion_prefix", ""))
    base_prompts = [
        apply_qwen_chat_template(
            tokenizer,
            build_user_prompt(problem["problem"]),
            enable_thinking=bool(model_cfg.get("enable_thinking", True)),
        )
        for problem in problems
    ]
    prompts = [prompt + completion_prefix for prompt in base_prompts]

    llm = LLM(
        model=model_cfg["name"],
        tensor_parallel_size=int(model_cfg.get("tensor_parallel_size", 1)),
        dtype=model_cfg.get("dtype", "auto"),
        gpu_memory_utilization=float(model_cfg.get("gpu_memory_utilization", 0.9)),
        max_model_len=int(model_cfg.get("max_model_len", 8192)),
        enforce_eager=bool(model_cfg.get("enforce_eager", False)),
        trust_remote_code=True,
    )

    temperature = float(gen_cfg.get("temperature", 0.7))
    n_samples = int(gen_cfg.get("n_samples_per_problem", 4))
    if temperature == 0.0 and n_samples != 1:
        print("temperature=0 uses greedy sampling in vLLM; forcing n_samples_per_problem=1")
        n_samples = 1

    generation_config = {
        "temperature": temperature,
        "top_p": float(gen_cfg.get("top_p", 0.95)),
        "n_samples_per_problem": n_samples,
        "max_new_tokens": int(gen_cfg.get("max_new_tokens", 4096)),
        "logprobs": int(gen_cfg.get("logprobs", 5)),
        "completion_prefix": completion_prefix,
    }
    sampling_params = SamplingParams(
        temperature=generation_config["temperature"],
        top_p=generation_config["top_p"],
        n=generation_config["n_samples_per_problem"],
        max_tokens=generation_config["max_new_tokens"],
        logprobs=generation_config["logprobs"],
    )

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate low-accuracy OlympiadBench CoT traces for AG-SFV."
    )
    parser.add_argument("--config", default="data_generation/qwen25_fhis/configs/recommended.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    paths = config["paths"]

    if args.generate_only:
        problems = list(read_jsonl(paths["problems"]))
        if args.limit is not None:
            problems = problems[: args.limit]
    else:
        problems = load_olympiadbench_problems(config, limit=args.limit)
        write_jsonl(paths["problems"], problems)
        print(f"Wrote {len(problems)} problems to {paths['problems']}")

    if args.prepare_only:
        return

    traces_path = Path(paths["generated_traces"])
    if traces_path.exists() and not args.resume:
        if not args.overwrite:
            raise SystemExit(
                f"{traces_path} already exists. Use --resume to continue or "
                "--overwrite to replace it."
            )
        traces_path.unlink()

    generate_traces(config, problems, resume=args.resume)
    summary = summarize_traces(paths["generated_traces"])
    summary.update(
        {
            "model": config["model"]["name"],
            "dataset": config["dataset"]["hf_name"],
            "subset": config["dataset"]["subset"],
        }
    )
    write_json(paths["summary"], summary)
    print(f"Wrote summary to {paths['summary']}")


if __name__ == "__main__":
    main()
