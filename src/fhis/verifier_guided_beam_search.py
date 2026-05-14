from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fhis.config import load_config, selected_layers_for_model
from fhis.io import append_jsonl, read_jsonl
from fhis.online_router import BoundaryStoppingCriteria, first_boundary, load_probe, score_feature
from fhis.probe_retry_router import append_step_suffix, has_final_answer_signal
from fhis.prompting import apply_qwen_chat_template, build_user_prompt
from fhis.steps import canonical_answer, extract_final_answer, extract_steps, rough_answer_match, step_end_token_indices


@dataclass(frozen=True)
class Beam:
    completion: str
    next_step: int
    step_scores: list[float]
    rank_score: float
    finished: bool = False
    answer: str | None = None


@dataclass(frozen=True)
class BeamResult:
    problem_id: str
    status: str
    answer: str | None
    canonical_answer: str | None
    rough_correct: bool | None
    completion: str
    beams: list[dict[str, Any]]


def answer_matches(predicted: str | None, references: Any) -> bool | None:
    if predicted is None or references is None:
        return None
    if isinstance(references, list):
        refs = [str(reference) for reference in references if reference is not None]
        if not refs:
            return None
        return any(rough_answer_match(predicted, reference) for reference in refs)
    return rough_answer_match(predicted, str(references))


def beam_rank(step_scores: list[float], risk_weight: float, max_weight: float) -> float:
    if not step_scores:
        return 0.0
    mean_score = sum(step_scores) / len(step_scores)
    return risk_weight * mean_score + max_weight * max(step_scores)


def truncate_at_boundary(text: str, step_index: int) -> tuple[str, str | None]:
    boundary = first_boundary(text, step_index)
    if not boundary:
        return text, None
    marker = boundary.group(1)
    return text[: boundary.start()], marker


def strip_duplicate_marker(text: str, step_index: int) -> str:
    return re.sub(
        rf"^\s*(?:#{{1,6}}\s*)?(?:\*\*)?Step\s+{step_index}\s*(?::|\*\*:)\s*",
        "",
        text,
        count=1,
        flags=re.I | re.S,
    ).strip()


class VerifierGuidedBeamSearch:
    def __init__(self, config: dict[str, Any]) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.config = config
        self.model_cfg = config["model"]
        self.online_cfg = config.get("online", {})
        self.search_cfg = config.get("verifier_guided_search", {})
        self.layer_ids = selected_layers_for_model(config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg["name"],
            trust_remote_code=True,
        )
        device = str(self.online_cfg.get("device", "auto"))
        dtype_name = str(self.model_cfg.get("dtype", "auto"))
        torch_dtype: Any = "auto"
        if dtype_name == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype_name == "float16":
            torch_dtype = torch.float16
        elif dtype_name == "float32":
            torch_dtype = torch.float32
        device_map = device if device == "auto" or device.startswith("cuda") else None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg["name"],
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device_map,
        )
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()

        router_cfg = config["router"]
        self.probe = load_probe(router_cfg["probe_model"])
        threshold = router_cfg.get("threshold")
        if threshold is None:
            threshold = getattr(self.probe, "decision_threshold", 0.5)
        self.threshold = float(threshold)

    def base_prompt(self, problem: str) -> str:
        return apply_qwen_chat_template(
            self.tokenizer,
            build_user_prompt(problem),
            enable_thinking=bool(self.model_cfg.get("enable_thinking", True)),
        )

    def generate_step_candidates(self, prompt: str, completion: str, step_index: int) -> list[str]:
        from transformers import StoppingCriteriaList
        import torch

        full_text = prompt + completion
        encoded = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        stopper = BoundaryStoppingCriteria(
            self.tokenizer,
            prompt_len=encoded["input_ids"].shape[1],
            current_step=step_index,
        ).stopper
        temperature = float(self.search_cfg.get("temperature", self.online_cfg.get("temperature", 0.7)))
        do_sample = temperature > 0
        branch_factor = int(self.search_cfg.get("branch_factor", 4))
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=int(self.search_cfg.get("max_step_new_tokens", self.online_cfg.get("max_step_new_tokens", 384))),
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=float(self.search_cfg.get("top_p", self.online_cfg.get("top_p", 0.95))) if do_sample else None,
                num_return_sequences=branch_factor,
                stopping_criteria=StoppingCriteriaList([stopper]),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        return [
            self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            for output in outputs
        ]

    def generate_final_answer(self, prompt: str, completion: str) -> str:
        import torch

        if "Final Answer:" not in completion:
            completion = completion.rstrip() + "\n\nFinal Answer:"
        encoded = self.tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        temperature = float(self.search_cfg.get("final_temperature", 0.0))
        do_sample = temperature > 0
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=int(self.search_cfg.get("max_final_new_tokens", self.online_cfg.get("max_final_new_tokens", 192))),
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=float(self.search_cfg.get("top_p", self.online_cfg.get("top_p", 0.95))) if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return completion + self.tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)

    def step_score(self, prompt: str, completion: str, step_index: int) -> float | None:
        import torch

        steps = extract_steps(completion)
        target = next((step for step in steps if step.index == step_index), None)
        if target is None:
            return None
        indices = step_end_token_indices(self.tokenizer, prompt, completion, [target])
        token_index = indices.get(step_index)
        if token_index is None:
            return None
        encoded = self.tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
        token_index = min(token_index, outputs.hidden_states[0].shape[1] - 1)
        per_layer = torch.stack(
            [outputs.hidden_states[layer_id + 1][0, token_index] for layer_id in self.layer_ids],
            dim=0,
        )
        return score_feature(self.probe, per_layer.float().cpu().reshape(-1))

    def finalize_beam(self, prompt: str, beam: Beam) -> Beam:
        completion = self.generate_final_answer(prompt, beam.completion)
        answer = extract_final_answer(completion)
        return Beam(
            completion=completion,
            next_step=beam.next_step,
            step_scores=beam.step_scores,
            rank_score=beam.rank_score,
            finished=answer is not None,
            answer=answer,
        )

    def solve_problem(self, problem: dict[str, Any]) -> BeamResult:
        prompt = self.base_prompt(str(problem["problem"]))
        beam_width = int(self.search_cfg.get("beam_width", 4))
        max_steps = int(self.search_cfg.get("max_steps", self.online_cfg.get("max_steps", 24)))
        risk_weight = float(self.search_cfg.get("risk_weight", 1.0))
        max_weight = float(self.search_cfg.get("max_risk_weight", 0.5))
        keep_above_threshold = int(self.search_cfg.get("keep_above_threshold", 1))
        completion_prefix = str(self.online_cfg.get("completion_prefix", "Step 1:"))

        beams = [
            Beam(
                completion=completion_prefix,
                next_step=1,
                step_scores=[],
                rank_score=0.0,
            )
        ]
        finals: list[Beam] = []

        for _ in range(max_steps):
            candidates: list[Beam] = []
            for beam in beams:
                if beam.finished:
                    finals.append(beam)
                    continue
                step_index = beam.next_step
                generated_candidates = self.generate_step_candidates(prompt, beam.completion, step_index)
                for generated in generated_candidates:
                    suffix, boundary_marker = truncate_at_boundary(generated, step_index)
                    suffix = strip_duplicate_marker(suffix, step_index)
                    if not suffix:
                        continue
                    candidate_completion = append_step_suffix(beam.completion, suffix)
                    score = self.step_score(prompt, candidate_completion, step_index)
                    if score is None or not math.isfinite(score):
                        continue
                    step_scores = beam.step_scores + [score]
                    rank = beam_rank(step_scores, risk_weight=risk_weight, max_weight=max_weight)
                    if boundary_marker and boundary_marker.lower().startswith("final answer"):
                        candidate_completion = candidate_completion.rstrip() + "\n\nFinal Answer:"
                        final_answer = self.generate_final_answer(prompt, candidate_completion)
                        answer = extract_final_answer(final_answer)
                        candidates.append(
                            Beam(final_answer, step_index + 1, step_scores, rank, answer is not None, answer)
                        )
                    elif has_final_answer_signal(candidate_completion):
                        answer = extract_final_answer(candidate_completion)
                        candidates.append(
                            Beam(candidate_completion, step_index + 1, step_scores, rank, answer is not None, answer)
                        )
                    else:
                        next_completion = candidate_completion.rstrip() + f"\n\nStep {step_index + 1}:"
                        candidates.append(
                            Beam(next_completion, step_index + 1, step_scores, rank)
                        )

            if not candidates:
                break

            below = [beam for beam in candidates if beam.step_scores and beam.step_scores[-1] <= self.threshold]
            above = [beam for beam in candidates if not beam.step_scores or beam.step_scores[-1] > self.threshold]
            pool = sorted(below, key=lambda beam: beam.rank_score)
            if len(pool) < beam_width and keep_above_threshold > 0:
                pool.extend(sorted(above, key=lambda beam: beam.rank_score)[:keep_above_threshold])
            beams = sorted(pool, key=lambda beam: beam.rank_score)[:beam_width]
            finals.extend([beam for beam in beams if beam.finished])
            beams = [beam for beam in beams if not beam.finished]
            if not beams and finals:
                break

        if not finals:
            finals = [self.finalize_beam(prompt, beam) for beam in sorted(beams, key=lambda b: b.rank_score)[:beam_width]]

        final_candidates = [beam for beam in finals if beam.answer]
        if not final_candidates:
            best = min(finals or beams, key=lambda beam: beam.rank_score)
            answer = extract_final_answer(best.completion)
            return BeamResult(
                problem_id=str(problem.get("problem_id", problem.get("id", "unknown"))),
                status="abstained" if answer is None else "accepted",
                answer=answer,
                canonical_answer=canonical_answer(answer),
                rough_correct=answer_matches(answer, problem.get("reference_answer")),
                completion=best.completion,
                beams=[asdict(beam) for beam in finals],
            )

        grouped: dict[str, list[Beam]] = {}
        for beam in final_candidates:
            canonical = canonical_answer(beam.answer)
            if canonical:
                grouped.setdefault(canonical, []).append(beam)
        if not grouped:
            best = min(final_candidates, key=lambda beam: beam.rank_score)
        else:
            # Prefer consensus among low-risk beams; break ties by minimum risk.
            best_answer, best_group = max(
                grouped.items(),
                key=lambda item: (
                    len(item[1]),
                    -sum(beam.rank_score for beam in item[1]) / len(item[1]),
                    -min(beam.rank_score for beam in item[1]),
                ),
            )
            best = min(best_group, key=lambda beam: beam.rank_score)

        answer = best.answer
        return BeamResult(
            problem_id=str(problem.get("problem_id", problem.get("id", "unknown"))),
            status="accepted" if answer is not None else "abstained",
            answer=answer,
            canonical_answer=canonical_answer(answer),
            rough_correct=answer_matches(answer, problem.get("reference_answer")),
            completion=best.completion,
            beams=[asdict(beam) for beam in sorted(final_candidates, key=lambda beam: beam.rank_score)[:beam_width]],
        )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("status") == "accepted"]
    correct = [row for row in accepted if row.get("rough_correct") is True]
    known = [row for row in accepted if row.get("rough_correct") is not None]
    return {
        "num_problems": len(rows),
        "accepted": len(accepted),
        "abstained": len(rows) - len(accepted),
        "rough_correct": len(correct),
        "rough_solve_rate_all": len(correct) / len(rows) if rows else None,
        "rough_solve_rate_answered": len(correct) / len(known) if known else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifier-guided step-level beam search.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--problems", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})
    problems_path = args.problems or paths["problems"]
    rows = list(read_jsonl(problems_path))
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.resume:
        done = {str(row.get("problem_id")) for row in read_jsonl(args.output)}
        rows = [row for row in rows if str(row.get("problem_id", row.get("id", "unknown"))) not in done]
    elif Path(args.output).exists():
        Path(args.output).unlink()

    search = VerifierGuidedBeamSearch(config)
    for problem in rows:
        result = search.solve_problem(problem)
        append_jsonl(args.output, [asdict(result)])
        print(json.dumps({"problem_id": result.problem_id, "status": result.status, "correct": result.rough_correct}, ensure_ascii=False))

    out_rows = list(read_jsonl(args.output))
    summary = summarize(out_rows)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
