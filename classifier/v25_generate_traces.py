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
    extract_reference_answer,
    extract_steps,
    rough_answer_match,
    steps_as_dicts,
)


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


def choose_olympiadbench_problem_text(item: dict[str, Any]) -> str:
    context = str(item.get("context") or "").strip()
    question = str(item.get("question") or "").strip()
    if context:
        return f"{context}\n\n{question}".strip()
    return question


def load_olympiadbench(
    subset: str,
    split: str,
    seed: int,
    limit: int | None,
    exclude_problem_ids: set[str],
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("Hothan/OlympiadBench", subset, split=split)
    rows: list[dict[str, Any]] = []
    for idx, item_raw in enumerate(ds):
        item = dict(item_raw)
        problem_id = f"{subset}-{idx}"
        if problem_id in exclude_problem_ids:
            continue
        problem = choose_olympiadbench_problem_text(item)
        if not problem:
            continue
        rows.append(
            {
                "problem_id": problem_id,
                "source_id": item.get("id", idx),
                "dataset": "Hothan/OlympiadBench",
                "subset": subset,
                "source_split": split,
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
    return rows[:limit] if limit is not None else rows


def load_hendrycks_math(
    subjects: list[str],
    split: str,
    seed: int,
    limit: int | None,
    levels: set[str] | None,
    exclude_problem_ids: set[str],
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    for subject in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
        for idx, item_raw in enumerate(ds):
            item = dict(item_raw)
            level = str(item.get("level") or "")
            if levels and level not in levels:
                continue
            problem_id = f"hendrycks_math_{split}_{subject}_{idx}"
            if problem_id in exclude_problem_ids:
                continue
            solution = str(item.get("solution") or "")
            reference = extract_reference_answer(solution)
            if not reference:
                continue
            rows.append(
                {
                    "problem_id": problem_id,
                    "source_id": idx,
                    "dataset": "EleutherAI/hendrycks_math",
                    "subset": subject,
                    "source_split": split,
                    "problem": str(item.get("problem") or "").strip(),
                    "reference_solution": solution,
                    "reference_answer": [reference],
                    "question_type": None,
                    "subject": item.get("type") or subject,
                    "subfield": subject,
                    "language": "en",
                    "difficulty": level,
                    "answer_type": None,
                    "is_multiple_answer": False,
                    "unit": None,
                }
            )
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit] if limit is not None else rows


def read_problem_ids(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    ids = set()
    for row in read_jsonl(p):
        if "problem_id" in row:
            ids.add(str(row["problem_id"]))
    return ids


def load_existing_problem_rows(traces_path: str | Path, seed: int, limit: int | None) -> list[dict[str, Any]]:
    seen: set[str] = set()
    problems: list[dict[str, Any]] = []
    for trace in read_jsonl(traces_path):
        pid = str(trace.get("problem_id"))
        if not pid or pid in seen:
            continue
        seen.add(pid)
        problems.append(
            {
                "problem_id": pid,
                "source_id": trace.get("source_id"),
                "dataset": trace.get("dataset"),
                "subset": trace.get("subset"),
                "source_split": trace.get("source_split", "train"),
                "problem": trace.get("problem"),
                "reference_solution": trace.get("reference_solution"),
                "reference_answer": trace.get("reference_answer") or [],
                "question_type": trace.get("question_type"),
                "subject": trace.get("subject"),
                "subfield": trace.get("subfield"),
                "language": trace.get("language"),
                "difficulty": trace.get("difficulty"),
                "answer_type": trace.get("answer_type"),
                "is_multiple_answer": trace.get("is_multiple_answer"),
                "unit": trace.get("unit"),
            }
        )
    rng = random.Random(seed)
    rng.shuffle(problems)
    return problems[:limit] if limit is not None else problems


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
    batch_id: str,
) -> list[dict[str, Any]]:
    rows = []
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
        trace_id = f"{problem['problem_id']}::{batch_id}-sample-{sample_index}"
        rows.append(
            {
                "trace_id": trace_id,
                "problem_id": problem["problem_id"],
                "dataset": problem["dataset"],
                "subset": problem["subset"],
                "source_id": problem["source_id"],
                "model_name": model_name,
                "sample_index": sample_index,
                "generation_batch_id": batch_id,
                "generation_config": generation_config,
                "completion_prefix": completion_prefix,
                "problem": problem["problem"],
                "reference_solution": problem["reference_solution"],
                "reference_answer": problem["reference_answer"],
                "prompt": prompt,
                "completion": completion,
                "steps": steps_as_dicts(steps),
                "final_answer": final_answer,
                "rough_final_correct": rough_any_answer_match(final_answer, problem["reference_answer"]),
                "token_ids": token_ids,
                "token_logprobs": token_logprobs,
            }
        )
    return rows


def summarize_traces(path: str | Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    return {
        "num_traces": len(rows),
        "num_problems": len({row.get("problem_id") for row in rows}),
        "rough_correct": sum(row.get("rough_final_correct") is True for row in rows),
        "rough_wrong": sum(row.get("rough_final_correct") is False for row in rows),
        "rough_unknown": sum(row.get("rough_final_correct") is None for row in rows),
        "step_parseable": sum(bool(row.get("steps")) for row in rows),
    }


def generate_vllm(
    args: argparse.Namespace,
    problems: list[dict[str, Any]],
    base_prompts: list[str],
    completion_prefix: str,
    generation_config: dict[str, Any],
) -> None:
    from tqdm import tqdm
    from vllm import LLM, SamplingParams

    prompts = [prompt + completion_prefix for prompt in base_prompts]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.samples_per_problem,
        max_tokens=args.max_new_tokens,
        logprobs=args.logprobs,
    )
    outputs = llm.generate(prompts, sampling_params)
    for problem, base_prompt, request_output in tqdm(
        zip(problems, base_prompts, outputs, strict=True),
        total=len(problems),
        desc="writing v2.5 traces",
    ):
        append_jsonl(
            args.output_traces,
            build_trace_rows(
                problem,
                base_prompt,
                request_output.outputs,
                generation_config,
                args.model,
                completion_prefix,
                args.batch_id,
            ),
        )


class TransformersOutput:
    def __init__(self, text: str):
        self.text = text
        self.token_ids = []
        self.logprobs = []


def generate_transformers(
    args: argparse.Namespace,
    problems: list[dict[str, Any]],
    base_prompts: list[str],
    completion_prefix: str,
    generation_config: dict[str, Any],
) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    for problem, base_prompt in tqdm(
        zip(problems, base_prompts, strict=True),
        total=len(problems),
        desc="generating transformers v2.5 traces",
    ):
        prompt = base_prompt + completion_prefix
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p if args.temperature > 0 else None,
                num_return_sequences=args.samples_per_problem,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs = []
        prompt_len = int(inputs["input_ids"].shape[-1])
        for row in generated:
            text = tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
            outputs.append(TransformersOutput(text))
        append_jsonl(
            args.output_traces,
            build_trace_rows(
                problem,
                base_prompt,
                outputs,
                generation_config,
                args.model,
                completion_prefix,
                args.batch_id,
            ),
        )


def generate(args: argparse.Namespace, problems: list[dict[str, Any]]) -> None:
    from transformers import AutoTokenizer

    if args.resume and Path(args.output_traces).exists():
        done = {str(row["problem_id"]) for row in read_jsonl(args.output_traces)}
        problems = [problem for problem in problems if str(problem["problem_id"]) not in done]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    completion_prefix = args.completion_prefix
    base_prompts = [
        apply_qwen_chat_template(
            tokenizer,
            build_user_prompt(problem["problem"]),
            enable_thinking=not args.disable_thinking,
        )
        for problem in problems
    ]
    generation_config = {
        "backend": args.backend,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n_samples_per_problem": args.samples_per_problem,
        "max_new_tokens": args.max_new_tokens,
        "logprobs": args.logprobs if args.backend == "vllm" else 0,
        "completion_prefix": completion_prefix,
    }
    if args.backend == "vllm":
        generate_vllm(args, problems, base_prompts, completion_prefix, generation_config)
    else:
        generate_transformers(args, problems, base_prompts, completion_prefix, generation_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unique-id v2.5 FHIS traces.")
    parser.add_argument("--source", choices=["existing-traces", "olympiadbench", "hendrycks_math"], required=True)
    parser.add_argument("--existing-traces")
    parser.add_argument("--olympiadbench-subset", default="OE_TO_maths_en_COMP")
    parser.add_argument("--hendrycks-subject", action="append", default=[])
    parser.add_argument("--hendrycks-split", default="test")
    parser.add_argument("--hendrycks-level", action="append", default=[])
    parser.add_argument("--exclude-problems-jsonl", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--limit-problems", type=int, default=None)
    parser.add_argument("--problems-out", required=True)
    parser.add_argument("--output-traces", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--model", default="/root/shared-nvme/models/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--samples-per-problem", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--completion-prefix", default="Step 1:")
    args = parser.parse_args()

    exclude_ids: set[str] = set()
    for path in args.exclude_problems_jsonl:
        exclude_ids |= read_problem_ids(path)

    if args.source == "existing-traces":
        if not args.existing_traces:
            raise SystemExit("--existing-traces is required for source=existing-traces")
        problems = load_existing_problem_rows(args.existing_traces, args.seed, args.limit_problems)
    elif args.source == "olympiadbench":
        problems = load_olympiadbench(
            args.olympiadbench_subset,
            "train",
            args.seed,
            args.limit_problems,
            exclude_ids,
        )
    else:
        subjects = args.hendrycks_subject or [
            "algebra",
            "number_theory",
            "precalculus",
            "intermediate_algebra",
            "geometry",
            "counting_and_probability",
        ]
        problems = load_hendrycks_math(
            subjects,
            args.hendrycks_split,
            args.seed,
            args.limit_problems,
            set(args.hendrycks_level) or None,
            exclude_ids,
        )

    write_jsonl(args.problems_out, problems)
    if not args.prepare_only:
        generate(args, problems)
    summary = {
        "source": args.source,
        "batch_id": args.batch_id,
        "num_problems": len(problems),
        "problems_out": args.problems_out,
        "output_traces": args.output_traces,
    }
    trace_summary = summarize_traces(args.output_traces)
    summary.update({f"generated_{key}": value for key, value in trace_summary.items()})
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
