from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from fhis.config import load_config, selected_layers_for_model
from fhis.io import append_jsonl, read_jsonl
from fhis.online_router import (
    BoundaryStoppingCriteria,
    first_boundary,
    load_probe,
    score_feature,
    step_metadata_row,
)
from fhis.prompting import STEP_PROMPT, apply_qwen_chat_template, build_user_prompt
from fhis.steps import extract_final_answer, extract_steps, step_end_token_indices


@dataclass(frozen=True)
class ProbeRetryDecision:
    attempt: int
    step_index: int
    step_attempt: int
    feedback_used: bool
    parse_ok: bool
    step_text: str | None
    probe_score: float | None
    flagged_for_retry: bool | None
    action: str


@dataclass(frozen=True)
class ProbeRetryResult:
    problem_id: str
    status: str
    attempts_used: int
    answer: str | None
    completion: str
    decisions: list[dict[str, Any]]


def strip_redundant_step_marker(text: str, step_index: int) -> str:
    pattern = re.compile(
        rf"^\s*(?:#{{1,6}}\s*)?(?:\*\*)?Step\s+{step_index}\s*(?::\*\*|\*\*:|:)\s*",
        flags=re.I | re.S,
    )
    return pattern.sub("", text, count=1).lstrip()


def append_step_suffix(step_prefix_completion: str, step_suffix: str) -> str:
    prefix = step_prefix_completion.rstrip()
    suffix = step_suffix.strip()
    if not suffix:
        return prefix
    if prefix.endswith(":"):
        return f"{prefix} {suffix}"
    return f"{prefix}{suffix}"


def has_final_answer_marker(text: str) -> bool:
    return re.search(r"\bFinal\s+Answer\s*:", text, flags=re.I) is not None


def has_final_answer_signal(text: str) -> bool:
    if has_final_answer_marker(text):
        return True
    if not re.search(r"\bfinal\s+answer\s+(?:is|=)", text, flags=re.I):
        return False
    return extract_final_answer(text) is not None


class ProbeRetryRouter:
    """Online router that uses the probe itself as the rejection signal.

    If a generated step receives a score above the threshold, the router feeds
    the model a short critique prompt and asks it to redo only that step. After
    max_step_attempts rejected/invalid local tries, the whole solution attempt is
    restarted from scratch. No Lean/formalizer call is made.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.config = config
        self.model_cfg = config["model"]
        self.online_cfg = config.get("online", {})
        self.retry_cfg = config.get("probe_retry", {})
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

    def retry_prompt(
        self,
        problem: dict[str, Any],
        accepted_steps: list[str],
        step_index: int,
        rejected_step: str,
    ) -> str:
        max_chars = int(self.retry_cfg.get("max_rejected_step_chars", 1200))
        rejected = rejected_step.strip()
        if len(rejected) > max_chars:
            rejected = rejected[:max_chars].rstrip() + " ..."
        previous = "\n".join(
            f"Step {idx}: {text}" for idx, text in enumerate(accepted_steps, start=1)
        )
        if not previous:
            previous = "(none)"
        prompt = f"""{STEP_PROMPT.format(problem=str(problem["problem"]).strip())}

Accepted solution so far:
{previous}

The previous draft of Step {step_index} is likely incorrect:
Step {step_index}: {rejected}

Redo Step {step_index} from the accepted solution so far.
Output only the corrected Step {step_index}. Do not output Step {step_index + 1} or the final answer.

Step {step_index}:"""
        return apply_qwen_chat_template(
            self.tokenizer,
            prompt,
            enable_thinking=bool(self.model_cfg.get("enable_thinking", True)),
        )

    def generate_text(self, prompt: str, completion: str, current_step: int) -> str:
        from transformers import StoppingCriteriaList
        import torch

        full_text = prompt + completion
        encoded = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        stopper = BoundaryStoppingCriteria(
            self.tokenizer,
            prompt_len=encoded["input_ids"].shape[1],
            current_step=current_step,
        ).stopper
        temperature = float(self.online_cfg.get("temperature", 0.7))
        do_sample = temperature > 0
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=int(self.online_cfg.get("max_step_new_tokens", 512)),
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=float(self.online_cfg.get("top_p", 0.95)) if do_sample else None,
                stopping_criteria=StoppingCriteriaList([stopper]),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def step_feature(self, prompt: str, completion: str, step_index: int) -> Any:
        import torch

        steps = extract_steps(completion)
        target = next((step for step in steps if step.index == step_index), None)
        if target is None:
            raise ValueError(f"Could not parse Step {step_index} from completion")
        indices = step_end_token_indices(self.tokenizer, prompt, completion, [target])
        if step_index not in indices:
            raise ValueError(f"Could not locate token boundary for Step {step_index}")
        token_index = indices[step_index]
        encoded = self.tokenizer(
            prompt + completion,
            return_tensors="pt",
            add_special_tokens=False,
        )
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
        token_index = min(token_index, outputs.hidden_states[0].shape[1] - 1)
        per_layer = torch.stack(
            [outputs.hidden_states[layer_id + 1][0, token_index] for layer_id in self.layer_ids],
            dim=0,
        )
        return per_layer.float().cpu().reshape(-1)

    def generate_final_answer(self, prompt: str, completion: str) -> str:
        import torch

        if "Final Answer:" not in completion:
            completion = completion.rstrip() + "\n\nFinal Answer:"
        full_text = prompt + completion
        encoded = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        temperature = float(self.online_cfg.get("temperature", 0.7))
        do_sample = temperature > 0
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=int(self.online_cfg.get("max_final_new_tokens", 256)),
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=float(self.online_cfg.get("top_p", 0.95)) if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[1] :]
        return completion + self.tokenizer.decode(generated, skip_special_tokens=True)

    def solve_problem(self, problem: dict[str, Any]) -> ProbeRetryResult:
        problem_id = str(problem.get("problem_id", problem.get("id", "unknown")))
        prompt = self.base_prompt(str(problem["problem"]))
        max_attempts = int(self.online_cfg.get("retry_budget", 3))
        max_step_attempts = int(self.retry_cfg.get("max_step_attempts", 3))
        max_steps = int(self.online_cfg.get("max_steps", 32))
        accept_best_after_retries = bool(
            self.retry_cfg.get("accept_best_after_max_step_attempts", False)
        )
        accept_best_margin = float(self.retry_cfg.get("accept_best_margin", 0.0))
        completion_prefix = str(self.online_cfg.get("completion_prefix", "Step 1:"))
        all_decisions: list[dict[str, Any]] = []
        last_completion = ""

        for attempt in range(1, max_attempts + 1):
            completion = completion_prefix
            accepted_steps: list[str] = []
            terminated = False
            for step_index in range(1, max_steps + 1):
                step_prefix_completion = completion
                boundary_marker = f"Step {step_index + 1}:"
                rejected_step = ""
                best_candidate: dict[str, Any] | None = None

                for step_attempt in range(1, max_step_attempts + 1):
                    feedback_used = step_attempt > 1
                    if feedback_used:
                        generation_prompt = self.retry_prompt(
                            problem,
                            accepted_steps,
                            step_index,
                            rejected_step,
                        )
                        generated = self.generate_text(generation_prompt, "", step_index)
                    else:
                        generated = self.generate_text(prompt, step_prefix_completion, step_index)

                    boundary = first_boundary(generated, step_index)
                    if boundary:
                        step_suffix = generated[: boundary.start()]
                        boundary_marker = boundary.group(1)
                    else:
                        step_suffix = generated
                        boundary_marker = f"Step {step_index + 1}:"
                    if feedback_used:
                        step_suffix = strip_redundant_step_marker(step_suffix, step_index)
                        candidate_completion = append_step_suffix(
                            step_prefix_completion,
                            step_suffix,
                        )
                    else:
                        candidate_completion = step_prefix_completion + step_suffix.rstrip()

                    steps = extract_steps(candidate_completion)
                    if not steps or steps[-1].index != step_index:
                        rejected_step = step_suffix.strip() or generated.strip()
                        action = (
                            "restart_trace"
                            if step_attempt >= max_step_attempts
                            else "retry_step_parse_failed"
                        )
                        if (
                            action == "restart_trace"
                            and accept_best_after_retries
                            and best_candidate is not None
                            and best_candidate["had_boundary"]
                            and best_candidate["score"] <= self.threshold + accept_best_margin
                        ):
                            action = "accept_best_after_parse_failures"
                        all_decisions.append(
                            asdict(
                                ProbeRetryDecision(
                                    attempt=attempt,
                                    step_index=step_index,
                                    step_attempt=step_attempt,
                                    feedback_used=feedback_used,
                                    parse_ok=False,
                                    step_text=None,
                                    probe_score=None,
                                    flagged_for_retry=None,
                                    action=action,
                                )
                            )
                        )
                        if action == "accept_best_after_parse_failures":
                            completion = str(best_candidate["completion"])
                            accepted_steps.append(str(best_candidate["step_text"]))
                            boundary_marker = str(best_candidate["boundary_marker"])
                            break
                        if step_attempt >= max_step_attempts:
                            terminated = True
                            break
                        continue

                    current_step_text = steps[-1].text
                    feature = self.step_feature(prompt, candidate_completion, step_index)
                    score = score_feature(
                        self.probe,
                        feature,
                        row=step_metadata_row(step_index, current_step_text),
                    )
                    if best_candidate is None or score < float(best_candidate["score"]):
                        best_candidate = {
                            "completion": candidate_completion,
                            "step_text": current_step_text,
                            "score": score,
                            "boundary_marker": boundary_marker,
                            "had_boundary": boundary is not None,
                        }
                    flagged = score >= self.threshold
                    action = "retry_step" if flagged else "accept_step"
                    if flagged and step_attempt >= max_step_attempts:
                        action = "restart_trace"
                        if (
                            accept_best_after_retries
                            and best_candidate is not None
                            and best_candidate["had_boundary"]
                            and best_candidate["score"] <= self.threshold + accept_best_margin
                        ):
                            action = "accept_best_after_retries"
                    all_decisions.append(
                        asdict(
                            ProbeRetryDecision(
                                attempt=attempt,
                                step_index=step_index,
                                step_attempt=step_attempt,
                                feedback_used=feedback_used,
                                parse_ok=True,
                                step_text=current_step_text,
                                probe_score=score,
                                flagged_for_retry=flagged,
                                action=action,
                            )
                        )
                    )

                    if action == "accept_best_after_retries":
                        completion = str(best_candidate["completion"])
                        accepted_steps.append(str(best_candidate["step_text"]))
                        boundary_marker = str(best_candidate["boundary_marker"])
                        break

                    if flagged:
                        rejected_step = current_step_text
                        if step_attempt >= max_step_attempts:
                            terminated = True
                            break
                        continue

                    completion = candidate_completion
                    accepted_steps.append(current_step_text)
                    break

                if terminated:
                    break
                if not accepted_steps or len(accepted_steps) < step_index:
                    terminated = True
                    break

                if boundary_marker.lower().startswith("final answer"):
                    completion = self.generate_final_answer(prompt, completion)
                    answer = extract_final_answer(completion)
                    return ProbeRetryResult(
                        problem_id=problem_id,
                        status="accepted",
                        attempts_used=attempt,
                        answer=answer,
                        completion=completion,
                        decisions=all_decisions,
                    )
                if has_final_answer_signal(completion):
                    answer = extract_final_answer(completion)
                    return ProbeRetryResult(
                        problem_id=problem_id,
                        status="accepted",
                        attempts_used=attempt,
                        answer=answer,
                        completion=completion,
                        decisions=all_decisions,
                    )

                completion = completion.rstrip() + f"\n\nStep {step_index + 1}:"

            last_completion = completion
            if not terminated:
                completion = self.generate_final_answer(prompt, completion)
                answer = extract_final_answer(completion)
                return ProbeRetryResult(
                    problem_id=problem_id,
                    status="accepted",
                    attempts_used=attempt,
                    answer=answer,
                    completion=completion,
                    decisions=all_decisions,
                )

        return ProbeRetryResult(
            problem_id=problem_id,
            status="abstained",
            attempts_used=max_attempts,
            answer=None,
            completion=last_completion,
            decisions=all_decisions,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Online probe-only step retry without Lean intervention."
    )
    parser.add_argument("--config", default="data_generation/qwen25_fhis/configs/probe_retry.yaml")
    parser.add_argument("--problems", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    problems_path = args.problems or config["paths"]["problems"]
    output_path = args.output or config["paths"]["probe_retry_results"]
    problems = list(read_jsonl(problems_path))
    if args.limit is not None:
        problems = problems[: args.limit]

    router = ProbeRetryRouter(config)
    for problem in problems:
        result = router.solve_problem(problem)
        append_jsonl(output_path, [asdict(result)])
        print(json.dumps(asdict(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
