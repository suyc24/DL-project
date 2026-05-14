from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fhis.config import load_config, selected_layers_for_model
from fhis.io import append_jsonl, read_jsonl
from fhis.online_router import load_probe
from fhis.prompting import STEP_PROMPT, apply_qwen_chat_template, build_user_prompt
from fhis.score_candidate_pool import score_completion_steps
from fhis.steps import canonical_answer, extract_final_answer, extract_steps, rough_answer_match


@dataclass(frozen=True)
class RepairResult:
    problem_id: str
    status: str
    answer: str | None
    canonical_answer: str | None
    rough_correct: bool | None
    source: str
    base_answer: str | None
    base_risk: float | None
    pivot_step: int | None
    num_repairs: int
    completion: str
    candidates: list[dict[str, Any]]


def answer_matches(predicted: str | None, references: Any) -> bool | None:
    if predicted is None or references is None:
        return None
    if isinstance(references, list):
        refs = [str(reference) for reference in references if reference is not None]
        if not refs:
            return None
        return any(rough_answer_match(predicted, reference) for reference in refs)
    return rough_answer_match(predicted, str(references))


def finite(value: Any, default: float = 9.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def sample_metric(sample: dict[str, Any], metric: str) -> float:
    return finite((sample.get("probe_risk") or {}).get(metric))


def best_sample_for_answer(samples: list[dict[str, Any]], answer: str, metric: str) -> dict[str, Any] | None:
    same = [sample for sample in samples if sample.get("canonical_answer") == answer]
    if not same:
        return None
    return min(same, key=lambda sample: sample_metric(sample, metric))


def choose_base_sample(row: dict[str, Any], metric: str) -> dict[str, Any] | None:
    samples = [sample for sample in row.get("samples", []) if sample.get("canonical_answer")]
    if not samples:
        return None
    voted = row.get("voted_canonical_answer")
    if voted:
        base = best_sample_for_answer(samples, str(voted), metric)
        if base is not None:
            return base
    return min(samples, key=lambda sample: sample_metric(sample, metric))


def find_pivot_step(sample: dict[str, Any], threshold: float, fallback: str) -> int | None:
    scores = (sample.get("probe_risk") or {}).get("step_scores") or []
    if not scores:
        return None
    above = [row for row in scores if finite(row.get("score"), 0.0) >= threshold]
    if above:
        return int(min(above, key=lambda row: int(row.get("step_index", 999))).get("step_index"))
    if fallback == "max":
        return int(max(scores, key=lambda row: finite(row.get("score"), 0.0)).get("step_index"))
    return None


def prefix_before_step(completion: str, pivot_step: int) -> tuple[str, list[str]]:
    steps = extract_steps(completion)
    accepted = [step for step in steps if step.index < pivot_step]
    rendered = "\n\n".join(f"Step {step.index}: {step.text}" for step in accepted)
    return rendered, [step.text for step in accepted]


def build_repair_prompt(problem: dict[str, Any], accepted_prefix: str, pivot_step: int) -> str:
    previous = accepted_prefix.strip() if accepted_prefix.strip() else "(none)"
    prompt = f"""{STEP_PROMPT.format(problem=str(problem["problem"]).strip())}

We are repairing a previous solution using a learned step verifier.
The following prefix is the part to preserve:
{previous}

The next step in the previous solution was flagged as likely wrong or unhelpful.
Continue from Step {pivot_step} with a fresh derivation. Do not copy the rejected continuation.
Keep the same exact Step k format and finish with "Final Answer:".

Step {pivot_step}:"""
    return prompt


def generate_repairs(
    *,
    model: Any,
    tokenizer: Any,
    model_cfg: dict[str, Any],
    repair_cfg: dict[str, Any],
    problem: dict[str, Any],
    accepted_prefix: str,
    pivot_step: int,
) -> list[str]:
    import torch

    prompt = apply_qwen_chat_template(
        tokenizer,
        build_repair_prompt(problem, accepted_prefix, pivot_step),
        enable_thinking=bool(model_cfg.get("enable_thinking", True)),
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    temperature = float(repair_cfg.get("temperature", 0.8))
    do_sample = temperature > 0
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=int(repair_cfg.get("max_new_tokens", 1400)),
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=float(repair_cfg.get("top_p", 0.95)) if do_sample else None,
            num_return_sequences=int(repair_cfg.get("num_repairs", 4)),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = encoded["input_ids"].shape[1]
    completions = []
    prefix = accepted_prefix.strip()
    if prefix:
        prefix = prefix + "\n\n"
    for output in outputs:
        suffix = tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
        if suffix.lower().startswith(f"step {pivot_step}:"):
            completion = prefix + suffix
        else:
            completion = prefix + f"Step {pivot_step}: " + suffix
        completions.append(completion)
    return completions


def select_answer(candidates: list[dict[str, Any]], metric: str, vote_weight: float, risk_weight: float) -> dict[str, Any] | None:
    parseable = [candidate for candidate in candidates if candidate.get("canonical_answer")]
    if not parseable:
        return None
    groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in parseable:
        groups.setdefault(str(candidate["canonical_answer"]), []).append(candidate)
    total = sum(len(group) for group in groups.values())
    best_key: tuple[float, int, float] | None = None
    best_candidate: dict[str, Any] | None = None
    for answer, group in groups.items():
        risk = min(sample_metric(candidate, metric) for candidate in group)
        vote_frac = len(group) / total if total else 0.0
        score = vote_weight * vote_frac - risk_weight * risk
        representative = min(group, key=lambda candidate: sample_metric(candidate, metric))
        key = (score, len(group), -risk)
        if best_key is None or key > best_key:
            best_key = key
            best_candidate = representative
    return best_candidate


class VerifierGuidedRepair:
    def __init__(self, config: dict[str, Any]) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.config = config
        self.model_cfg = config["model"]
        self.repair_cfg = config.get("verifier_guided_repair", {})
        self.layer_ids = selected_layers_for_model(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg["name"], trust_remote_code=True)
        dtype_name = str(self.model_cfg.get("dtype", "auto"))
        torch_dtype: Any = "auto"
        if dtype_name == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype_name == "float16":
            torch_dtype = torch.float16
        elif dtype_name == "float32":
            torch_dtype = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg["name"],
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()
        router_cfg = config["router"]
        self.probe = load_probe(router_cfg["probe_model"])
        self.threshold = float(router_cfg.get("threshold", getattr(self.probe, "decision_threshold", 0.5)))

    def base_prompt(self, problem: dict[str, Any]) -> str:
        return apply_qwen_chat_template(
            self.tokenizer,
            build_user_prompt(str(problem["problem"])),
            enable_thinking=bool(self.model_cfg.get("enable_thinking", True)),
        )

    def score_candidate(self, problem: dict[str, Any], completion: str) -> dict[str, Any]:
        answer = extract_final_answer(completion)
        prompt = self.base_prompt(problem)
        return {
            "answer": answer,
            "canonical_answer": canonical_answer(answer),
            "completion": completion,
            "probe_risk": score_completion_steps(
                model=self.model,
                tokenizer=self.tokenizer,
                probe=self.probe,
                layer_ids=self.layer_ids,
                prompt=prompt,
                completion=completion,
            ),
        }

    def solve(self, row: dict[str, Any], problem: dict[str, Any]) -> RepairResult:
        metric = str(self.repair_cfg.get("risk_metric", "top2_mean_score"))
        base = choose_base_sample(row, metric)
        original_candidates = [dict(sample, source="original") for sample in row.get("samples", [])]
        if base is None:
            selected = select_answer(
                original_candidates,
                metric=metric,
                vote_weight=float(self.repair_cfg.get("vote_weight", 1.0)),
                risk_weight=float(self.repair_cfg.get("risk_weight", 1.0)),
            )
            answer = selected.get("answer") if selected else None
            return RepairResult(
                problem_id=str(row.get("problem_id")),
                status="accepted" if answer else "abstained",
                answer=answer,
                canonical_answer=canonical_answer(answer),
                rough_correct=answer_matches(answer, problem.get("reference_answer")),
                source="original_no_base",
                base_answer=None,
                base_risk=None,
                pivot_step=None,
                num_repairs=0,
                completion=selected.get("completion", "") if selected else "",
                candidates=original_candidates,
            )

        pivot = find_pivot_step(
            base,
            threshold=float(self.repair_cfg.get("pivot_threshold", self.threshold)),
            fallback=str(self.repair_cfg.get("pivot_fallback", "max")),
        )
        repairs: list[dict[str, Any]] = []
        if pivot is not None:
            accepted_prefix, _ = prefix_before_step(str(base.get("completion", "")), pivot)
            for completion in generate_repairs(
                model=self.model,
                tokenizer=self.tokenizer,
                model_cfg=self.model_cfg,
                repair_cfg=self.repair_cfg,
                problem=problem,
                accepted_prefix=accepted_prefix,
                pivot_step=pivot,
            ):
                scored = self.score_candidate(problem, completion)
                scored["source"] = "repair"
                repairs.append(scored)

        candidates = original_candidates + repairs
        selected = select_answer(
            candidates,
            metric=metric,
            vote_weight=float(self.repair_cfg.get("vote_weight", 1.0)),
            risk_weight=float(self.repair_cfg.get("risk_weight", 1.0)),
        )
        answer = selected.get("answer") if selected else None
        return RepairResult(
            problem_id=str(row.get("problem_id")),
            status="accepted" if answer else "abstained",
            answer=answer,
            canonical_answer=canonical_answer(answer),
            rough_correct=answer_matches(answer, problem.get("reference_answer")),
            source=str(selected.get("source", "none")) if selected else "none",
            base_answer=base.get("answer"),
            base_risk=sample_metric(base, metric),
            pivot_step=pivot,
            num_repairs=len(repairs),
            completion=str(selected.get("completion", "")) if selected else "",
            candidates=[
                {
                    "source": candidate.get("source"),
                    "answer": candidate.get("answer"),
                    "canonical_answer": candidate.get("canonical_answer"),
                    "risk": sample_metric(candidate, metric),
                }
                for candidate in candidates
            ],
        )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("status") == "accepted"]
    correct = [row for row in accepted if row.get("rough_correct") is True]
    source_counts = Counter(row.get("source") for row in accepted)
    return {
        "num_problems": len(rows),
        "accepted": len(accepted),
        "abstained": len(rows) - len(accepted),
        "rough_correct": len(correct),
        "rough_solve_rate_all": len(correct) / len(rows) if rows else None,
        "source_counts": dict(sorted(source_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Use verifier-localized pivots to repair candidate solutions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = list(read_jsonl(args.candidates))
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.resume:
        done = {str(row.get("problem_id")) for row in read_jsonl(args.output)}
        rows = [row for row in rows if str(row.get("problem_id")) not in done]
    elif Path(args.output).exists():
        Path(args.output).unlink()

    problems = {
        str(problem.get("problem_id", problem.get("id", "unknown"))): problem
        for problem in read_jsonl(args.problems)
    }
    runner = VerifierGuidedRepair(config)
    for row in rows:
        problem = problems[str(row["problem_id"])]
        result = runner.solve(row, problem)
        append_jsonl(args.output, [asdict(result)])
        print(json.dumps({"problem_id": result.problem_id, "source": result.source, "correct": result.rough_correct}, ensure_ascii=False))

    out_rows = list(read_jsonl(args.output))
    summary = summarize(out_rows)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
