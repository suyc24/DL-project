from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fhis.config import load_config, selected_layers_for_model
from fhis.formalizer import FormalizationRequest, build_formalizer
from fhis.io import append_jsonl, read_jsonl
from fhis.lean_verify import verify_lean_code
from fhis.prompting import apply_qwen_chat_template, build_user_prompt
from fhis.steps import extract_final_answer, extract_steps, step_end_token_indices


BOUNDARY_RE = re.compile(r"\n\s*(Step\s+(\d+)\s*:|Final Answer\s*:)", flags=re.I)


@dataclass(frozen=True)
class StepDecision:
    attempt: int
    step_index: int
    step_retry: int
    step_text: str
    probe_score: float
    routed_to_lean: bool
    verification_status: str | None
    lean_code: str | None = None
    lean_stdout: str | None = None
    lean_stderr: str | None = None


@dataclass(frozen=True)
class OnlineResult:
    problem_id: str
    status: str
    attempts_used: int
    answer: str | None
    completion: str
    decisions: list[dict[str, Any]]


class BoundaryStoppingCriteria:
    def __init__(self, tokenizer: Any, prompt_len: int, current_step: int) -> None:
        from transformers import StoppingCriteria

        class _Stopper(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_: Any) -> bool:
                text = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
                for match in BOUNDARY_RE.finditer(text):
                    marker = match.group(1).lower()
                    if marker.startswith("final answer"):
                        return True
                    step_num = match.group(2)
                    if step_num is not None and int(step_num) > current_step:
                        return True
                return False

        self.stopper = _Stopper()


def first_boundary(text: str, current_step: int) -> re.Match[str] | None:
    for match in BOUNDARY_RE.finditer(text):
        marker = match.group(1).lower()
        if marker.startswith("final answer"):
            return match
        step_num = match.group(2)
        if step_num is not None and int(step_num) > current_step:
            return match
    return None


def load_probe(path: str | Path) -> Any:
    import joblib

    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def score_feature(probe: Any, feature: torch.Tensor) -> float:
    x = feature.float().cpu().numpy()[None, :]
    if hasattr(probe, "predict_proba"):
        return float(probe.predict_proba(x)[0, 1])
    if hasattr(probe, "predict_scores"):
        return float(probe.predict_scores(x)[0])
    raise TypeError("Probe object must expose predict_proba or predict_scores")


class OnlineStepRouter:
    def __init__(self, config: dict[str, Any]) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.config = config
        self.model_cfg = config["model"]
        self.online_cfg = config.get("online", {})
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
        self.formalizer = build_formalizer(config)

    def base_prompt(self, problem: str) -> str:
        return apply_qwen_chat_template(
            self.tokenizer,
            build_user_prompt(problem),
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

    def step_feature(self, prompt: str, completion: str, step_index: int) -> torch.Tensor:
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

    def solve_problem(self, problem: dict[str, Any]) -> OnlineResult:
        problem_id = str(problem.get("problem_id", problem.get("id", "unknown")))
        prompt = self.base_prompt(str(problem["problem"]))
        max_attempts = int(self.online_cfg.get("retry_budget", 3))
        step_retry_budget = int(self.online_cfg.get("step_retry_budget", 2))
        max_steps = int(self.online_cfg.get("max_steps", 32))
        completion_prefix = str(self.online_cfg.get("completion_prefix", "Step 1:"))
        all_decisions: list[dict[str, Any]] = []
        last_completion = ""

        for attempt in range(1, max_attempts + 1):
            completion = completion_prefix
            accepted_steps: list[str] = []
            terminated = False
            for step_index in range(1, max_steps + 1):
                step_prefix_completion = completion
                accepted_current_step = False
                boundary_marker = f"Step {step_index + 1}:"

                for step_retry in range(step_retry_budget + 1):
                    generated = self.generate_text(prompt, step_prefix_completion, step_index)
                    boundary = first_boundary(generated, step_index)
                    if boundary:
                        step_suffix = generated[: boundary.start()]
                        boundary_marker = boundary.group(1)
                    else:
                        step_suffix = generated
                        boundary_marker = f"Step {step_index + 1}:"

                    candidate_completion = step_prefix_completion + step_suffix.rstrip()
                    steps = extract_steps(candidate_completion)
                    if not steps or steps[-1].index != step_index:
                        if step_retry >= step_retry_budget:
                            terminated = True
                        continue

                    current_step_text = steps[-1].text
                    feature = self.step_feature(prompt, candidate_completion, step_index)
                    score = score_feature(self.probe, feature)
                    routed = score >= self.threshold
                    verification_status: str | None = None
                    lean_code: str | None = None
                    lean_stdout: str | None = None
                    lean_stderr: str | None = None

                    if routed:
                        request = FormalizationRequest(
                            problem=str(problem["problem"]),
                            previous_steps=accepted_steps,
                            current_step_index=step_index,
                            current_step=current_step_text,
                        )
                        lean_code = self.formalizer.formalize(request)
                        lean_result = verify_lean_code(
                            lean_code,
                            workdir=self.config.get("lean", {}).get("workdir"),
                            executable=str(self.config.get("lean", {}).get("executable", "lean")),
                            timeout_s=float(self.config.get("lean", {}).get("timeout_s", 30.0)),
                            keep_file=bool(self.config.get("lean", {}).get("keep_files", False)),
                        )
                        verification_status = lean_result.status
                        lean_stdout = lean_result.stdout
                        lean_stderr = lean_result.stderr

                    decision = StepDecision(
                        attempt=attempt,
                        step_index=step_index,
                        step_retry=step_retry,
                        step_text=current_step_text,
                        probe_score=score,
                        routed_to_lean=routed,
                        verification_status=verification_status,
                        lean_code=lean_code,
                        lean_stdout=lean_stdout,
                        lean_stderr=lean_stderr,
                    )
                    all_decisions.append(asdict(decision))

                    if routed and verification_status != "proved":
                        if step_retry >= step_retry_budget:
                            terminated = True
                            break
                        continue

                    completion = candidate_completion
                    accepted_steps.append(current_step_text)
                    accepted_current_step = True
                    break

                if terminated:
                    break
                if not accepted_current_step:
                    terminated = True
                    break

                if boundary_marker.lower().startswith("final answer"):
                    completion = self.generate_final_answer(prompt, completion)
                    answer = extract_final_answer(completion)
                    return OnlineResult(
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
                return OnlineResult(
                    problem_id=problem_id,
                    status="accepted",
                    attempts_used=attempt,
                    answer=answer,
                    completion=completion,
                    decisions=all_decisions,
                )

        return OnlineResult(
            problem_id=problem_id,
            status="abstained",
            attempts_used=max_attempts,
            answer=None,
            completion=last_completion,
            decisions=all_decisions,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Online prefix-causal probe routing to selective Lean verification."
    )
    parser.add_argument("--config", default="data_generation/qwen25_fhis/configs/online_verify.yaml")
    parser.add_argument("--problems", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    problems_path = args.problems or config["paths"]["problems"]
    output_path = args.output or config["paths"]["online_results"]
    problems = list(read_jsonl(problems_path))
    if args.limit is not None:
        problems = problems[: args.limit]

    router = OnlineStepRouter(config)
    for problem in problems:
        result = router.solve_problem(problem)
        append_jsonl(output_path, [asdict(result)])
        print(json.dumps(asdict(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
