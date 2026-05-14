from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fhis.config import load_config
from fhis.io import append_jsonl, read_jsonl
from fhis.prompting import apply_qwen_chat_template, build_user_prompt
from fhis.steps import canonical_answer, extract_final_answer, rough_answer_match


@dataclass(frozen=True)
class MajoritySample:
    sample_index: int
    answer: str | None
    canonical_answer: str | None
    rough_correct: bool | None
    completion: str


@dataclass(frozen=True)
class MajorityResult:
    problem_id: str
    status: str
    voted_answer: str | None
    voted_canonical_answer: str | None
    vote_count: int
    num_parseable_answers: int
    num_samples: int
    rough_correct: bool | None
    samples: list[dict[str, Any]]


def answer_matches(predicted: str | None, references: Any) -> bool | None:
    if predicted is None:
        return None
    if references is None:
        return None
    if isinstance(references, list):
        refs = [str(reference) for reference in references if reference is not None]
        if not refs:
            return None
        return any(rough_answer_match(predicted, reference) for reference in refs)
    return rough_answer_match(predicted, str(references))


def choose_majority(samples: list[MajoritySample]) -> tuple[str | None, str | None, int]:
    parseable = [sample for sample in samples if sample.canonical_answer]
    if not parseable:
        return None, None, 0
    counts = Counter(sample.canonical_answer for sample in parseable)
    best_count = max(counts.values())
    for sample in parseable:
        if counts[sample.canonical_answer] == best_count:
            return sample.answer, sample.canonical_answer, best_count
    raise AssertionError("unreachable")


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("status") == "accepted"]
    abstained = [row for row in rows if row.get("status") == "abstained"]
    correct = [row for row in accepted if row.get("rough_correct") is True]
    known = [row for row in accepted if row.get("rough_correct") is not None]
    sample0_correct = [
        row
        for row in rows
        if row.get("samples") and row["samples"][0].get("rough_correct") is True
    ]
    any_sample_correct = [
        row
        for row in rows
        if any(sample.get("rough_correct") is True for sample in row.get("samples", []))
    ]
    vote_counts = Counter(row.get("vote_count") for row in accepted)
    return {
        "num_problems": len(rows),
        "accepted": len(accepted),
        "abstained": len(abstained),
        "answer_rate": len(accepted) / len(rows) if rows else None,
        "rough_solve_rate_all": len(correct) / len(rows) if rows else None,
        "rough_solve_rate_answered": len(correct) / len(known) if known else None,
        "sample0_correct": len(sample0_correct),
        "sample0_solve_rate_all": len(sample0_correct) / len(rows) if rows else None,
        "oracle_any_sample_correct": len(any_sample_correct),
        "oracle_any_sample_solve_rate_all": len(any_sample_correct) / len(rows) if rows else None,
        "vote_count_distribution": dict(sorted(vote_counts.items())),
        "num_samples_per_problem": rows[0].get("num_samples") if rows else None,
    }


def write_result(problem: dict[str, Any], samples: list[MajoritySample], output_path: str) -> None:
    voted_answer, voted_canonical_answer, vote_count = choose_majority(samples)
    status = "accepted" if voted_answer is not None else "abstained"
    result = MajorityResult(
        problem_id=str(problem.get("problem_id", problem.get("id", "unknown"))),
        status=status,
        voted_answer=voted_answer,
        voted_canonical_answer=voted_canonical_answer,
        vote_count=vote_count,
        num_parseable_answers=sum(sample.answer is not None for sample in samples),
        num_samples=len(samples),
        rough_correct=answer_matches(voted_answer, problem.get("reference_answer")),
        samples=[asdict(sample) for sample in samples],
    )
    append_jsonl(output_path, [asdict(result)])


def run_transformers_backend(
    *,
    model_cfg: dict[str, Any],
    tokenizer: Any,
    problems: list[dict[str, Any]],
    prompts: list[str],
    completion_prefix: str,
    output_path: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    seed: int | None,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, set_seed

    if seed is not None:
        set_seed(seed)

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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    do_sample = temperature > 0
    for start in range(0, len(problems), batch_size):
        batch_problems = problems[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        for problem_index, problem in enumerate(batch_problems):
            samples: list[MajoritySample] = []
            first_output = problem_index * n_samples
            for sample_index, output in enumerate(outputs[first_output : first_output + n_samples]):
                generated = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                completion = completion_prefix + generated
                answer = extract_final_answer(completion)
                samples.append(
                    MajoritySample(
                        sample_index=sample_index,
                        answer=answer,
                        canonical_answer=canonical_answer(answer),
                        rough_correct=answer_matches(answer, problem.get("reference_answer")),
                        completion=completion,
                    )
                )
            write_result(problem, samples, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate n samples and answer by majority vote.")
    parser.add_argument("--config", default="data_generation/qwen25_fhis/configs/majority_vote.yaml")
    parser.add_argument("--problems", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config = load_config(args.config)
    seed = int(config["seed"]) if config.get("seed") is not None else None
    model_cfg = config["model"]
    gen_cfg = config.get("generation", {})
    paths = config.get("paths", {})
    problems_path = args.problems or paths["problems"]
    output_path = args.output or paths["majority_vote_results"]
    summary_path = args.summary or paths.get("majority_vote_summary")

    problems = list(read_jsonl(problems_path))
    if args.limit is not None:
        problems = problems[: args.limit]
    if args.resume:
        done = {str(row.get("problem_id")) for row in read_jsonl(output_path)}
        problems = [problem for problem in problems if str(problem.get("problem_id")) not in done]
    elif Path(output_path).exists():
        Path(output_path).unlink()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    completion_prefix = str(gen_cfg.get("completion_prefix", "Step 1:"))
    base_prompts = [
        apply_qwen_chat_template(
            tokenizer,
            build_user_prompt(str(problem["problem"])),
            enable_thinking=bool(model_cfg.get("enable_thinking", True)),
        )
        for problem in problems
    ]
    prompts = [prompt + completion_prefix for prompt in base_prompts]

    temperature = float(gen_cfg.get("temperature", 0.7))
    n_samples = int(gen_cfg.get("n_samples_per_problem", 3))
    top_p = float(gen_cfg.get("top_p", 0.95))
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 2048))
    batch_size = int(gen_cfg.get("batch_size", 1))

    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError:
        LLM = None
        SamplingParams = None

    if LLM is not None and SamplingParams is not None:
        llm = LLM(
            model=model_cfg["name"],
            tensor_parallel_size=int(model_cfg.get("tensor_parallel_size", 1)),
            dtype=model_cfg.get("dtype", "auto"),
            gpu_memory_utilization=float(model_cfg.get("gpu_memory_utilization", 0.9)),
            max_model_len=int(model_cfg.get("max_model_len", 4096)),
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            n=n_samples,
            max_tokens=max_new_tokens,
            seed=seed,
        )

        outputs = llm.generate(prompts, sampling_params)
        for problem, request_output in zip(problems, outputs, strict=True):
            samples = []
            for sample_index, output in enumerate(request_output.outputs):
                completion = completion_prefix + output.text
                answer = extract_final_answer(completion)
                samples.append(
                    MajoritySample(
                        sample_index=sample_index,
                        answer=answer,
                        canonical_answer=canonical_answer(answer),
                        rough_correct=answer_matches(answer, problem.get("reference_answer")),
                        completion=completion,
                    )
                )
            write_result(problem, samples, output_path)
    else:
        run_transformers_backend(
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            problems=problems,
            prompts=prompts,
            completion_prefix=completion_prefix,
            output_path=output_path,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
            seed=seed,
        )

    rows = list(read_jsonl(output_path))
    summary = summarize_results(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary_path:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
