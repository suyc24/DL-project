from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fhis.config import load_config
from fhis.io import append_jsonl, read_jsonl
from fhis.prompting import apply_qwen_chat_template
from fhis.steps import canonical_answer, extract_final_answer, rough_answer_match


@dataclass(frozen=True)
class AdjudicationResult:
    problem_id: str
    status: str
    answer: str | None
    canonical_answer: str | None
    rough_correct: bool | None
    prompt_candidate_answers: list[dict[str, Any]]
    completion: str


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


def sample_risk(sample: dict[str, Any], metric: str) -> float:
    risk = sample.get("probe_risk") or {}
    value = risk.get(metric)
    return float(value) if value is not None else 9.0


def trim(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def select_candidate_groups(
    samples: list[dict[str, Any]],
    *,
    max_answers: int,
    risk_metric: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        answer = sample.get("canonical_answer")
        if answer:
            groups.setdefault(str(answer), []).append(sample)
    ranked = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), min(sample_risk(sample, risk_metric) for sample in item[1])),
    )
    # Also make sure the lowest-risk answer is considered even if it has one vote.
    lowest = min(groups.items(), key=lambda item: min(sample_risk(sample, risk_metric) for sample in item[1]), default=None)
    selected = ranked[:max_answers]
    if lowest is not None and lowest[0] not in {answer for answer, _ in selected}:
        selected = selected[:-1] + [lowest] if len(selected) >= max_answers else selected + [lowest]

    payload = []
    for answer, answer_samples in selected:
        best_sample = min(answer_samples, key=lambda sample: sample_risk(sample, risk_metric))
        payload.append(
            {
                "answer": answer,
                "votes": len(answer_samples),
                "min_probe_risk": sample_risk(best_sample, risk_metric),
                "representative_completion": best_sample.get("completion", ""),
            }
        )
    return payload


def build_adjudication_prompt(
    problem: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    max_solution_chars: int,
    prompt_style: str,
) -> str:
    rendered = []
    for idx, candidate in enumerate(candidates, start=1):
        if prompt_style == "answers_only":
            rendered.append(
                f"""Candidate {idx}: {candidate['answer']}
Votes: {candidate['votes']}
Verifier risk score: {candidate['min_probe_risk']:.6f}
"""
            )
        else:
            rendered.append(
                f"""Candidate {idx}
Final answer: {candidate['answer']}
Number of independent samples with this answer: {candidate['votes']}
Verifier risk score for representative solution: {candidate['min_probe_risk']:.6f}
Representative solution:
{trim(str(candidate['representative_completion']), max_solution_chars)}
"""
            )
    candidates_text = "\n".join(rendered)
    if prompt_style == "answers_only":
        return f"""You are given a math problem and several candidate final answers.

The verifier risk score is a learned signal: lower usually means the solution path was safer, but it can be wrong. Votes count independent samples with the same answer, but the majority can be wrong.

Solve the original problem independently. Use the candidate answers only as a checklist of plausible outcomes. If one candidate is correct, output that answer exactly; if none is correct, output your own answer.

Problem:
{str(problem['problem']).strip()}

Candidate final answers:
{candidates_text}

Write a concise verification, then end with exactly:
Final Answer: [answer]
"""
    return f"""You are given a math problem and several independently generated candidate solutions.

The verifier risk score is a learned signal: lower usually means safer, but it can be wrong. Do not blindly follow the majority vote or the verifier score.

Your task is to solve the original problem and decide which final answer is correct. You may use the candidate solutions as hints, but check the mathematics yourself.

Problem:
{str(problem['problem']).strip()}

Candidate solutions:
{candidates_text}

Output a short verification and then:

Final Answer: [answer]
"""


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("status") == "accepted"]
    correct = [row for row in accepted if row.get("rough_correct") is True]
    known = [row for row in accepted if row.get("rough_correct") is not None]
    return {
        "num_problems": len(rows),
        "accepted": len(accepted),
        "abstained": len(rows) - len(accepted),
        "answer_rate": len(accepted) / len(rows) if rows else None,
        "rough_solve_rate_all": len(correct) / len(rows) if rows else None,
        "rough_solve_rate_answered": len(correct) / len(known) if known else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adjudicate candidate answers with verifier-ranked hints.")
    parser.add_argument("--config", default="data_generation/qwen25_fhis/configs/majority_vote_n8.yaml")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-answers", type=int, default=4)
    parser.add_argument("--risk-metric", default="max_score")
    parser.add_argument("--max-solution-chars", type=int, default=1200)
    parser.add_argument("--prompt-style", choices=["solutions", "answers_only"], default="solutions")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    config = load_config(args.config)
    model_cfg = config["model"]
    gen_cfg = config.get("generation", {})
    problems = {
        str(row.get("problem_id", row.get("id", "unknown"))): row
        for row in read_jsonl(args.problems)
    }
    rows = list(read_jsonl(args.candidates))
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

    temperature = float(gen_cfg.get("adjudication_temperature", 0.0))
    do_sample = temperature > 0
    for row in rows:
        problem = problems[str(row["problem_id"])]
        candidates = select_candidate_groups(
            row.get("samples", []),
            max_answers=args.max_answers,
            risk_metric=args.risk_metric,
        )
        if not candidates:
            result = AdjudicationResult(
                problem_id=str(row["problem_id"]),
                status="abstained",
                answer=None,
                canonical_answer=None,
                rough_correct=None,
                prompt_candidate_answers=[],
                completion="",
            )
            append_jsonl(args.output, [asdict(result)])
            continue
        prompt = apply_qwen_chat_template(
            tokenizer,
            build_adjudication_prompt(
                problem,
                candidates,
                max_solution_chars=args.max_solution_chars,
                prompt_style=args.prompt_style,
            ),
            enable_thinking=bool(model_cfg.get("enable_thinking", True)),
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=int(gen_cfg.get("adjudication_max_new_tokens", 1024)),
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=float(gen_cfg.get("top_p", 0.95)) if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        answer = extract_final_answer(completion)
        result = AdjudicationResult(
            problem_id=str(row["problem_id"]),
            status="accepted" if answer is not None else "abstained",
            answer=answer,
            canonical_answer=canonical_answer(answer),
            rough_correct=answer_matches(answer, problem.get("reference_answer")),
            prompt_candidate_answers=[
                {key: value for key, value in candidate.items() if key != "representative_completion"}
                for candidate in candidates
            ],
            completion=completion,
        )
        append_jsonl(args.output, [asdict(result)])

    output_rows = list(read_jsonl(args.output))
    summary = summarize(output_rows)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
