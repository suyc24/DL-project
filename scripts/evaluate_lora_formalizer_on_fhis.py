from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fhis.semantic_lean_verify import SemanticLeanVerifier  # noqa: E402
from no_template_formalizer_experiment import (  # noqa: E402
    Case,
    build_direct_prompt,
    build_repair_prompt,
    build_task,
    extract_common_code,
    lean_error_text,
)
from qwen3_lora_formalizer_experiment import add_lora, generate_text  # noqa: E402


@dataclass(frozen=True)
class RealStepCase:
    case_id: str
    expected_status: str
    trace_id: str
    problem_id: str
    problem: str
    prior_steps: list[str]
    current_step: str
    label_reason: str | None
    label_confidence: str | None
    source: str

    def as_case(self) -> Case:
        return Case(
            case_id=self.case_id,
            expected_status=self.expected_status,
            problem=self.problem,
            prior_steps=self.prior_steps,
            current_step=self.current_step,
        )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def step_texts(trace: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for step in trace.get("steps") or []:
        if isinstance(step, dict):
            out.append(str(step.get("text") or ""))
        else:
            out.append(str(step))
    return out


def build_cases(
    labels_path: str | Path,
    traces_path: str | Path,
    *,
    seed: int,
    n_invalid: int,
    n_valid: int,
    max_prior_steps: int,
) -> list[RealStepCase]:
    rng = random.Random(seed)
    labels = read_jsonl(labels_path)
    traces = {row["trace_id"]: row for row in read_jsonl(traces_path)}

    invalid: list[RealStepCase] = []
    valid: list[RealStepCase] = []
    for label in labels:
        if label.get("confidence") != "high":
            continue
        trace = traces.get(label.get("trace_id"))
        if trace is None:
            continue
        steps = step_texts(trace)
        if not steps:
            continue
        problem = str(trace.get("problem") or "")
        if label.get("final_correct") is False and label.get("first_invalid_step"):
            idx = int(label["first_invalid_step"])
            if 1 <= idx <= len(steps):
                priors = steps[max(0, idx - 1 - max_prior_steps) : idx - 1]
                invalid.append(
                    RealStepCase(
                        case_id=f"{label['trace_id']}::step-{idx}",
                        expected_status="invalid",
                        trace_id=label["trace_id"],
                        problem_id=str(label.get("problem_id") or trace.get("problem_id")),
                        problem=problem,
                        prior_steps=priors,
                        current_step=steps[idx - 1],
                        label_reason=label.get("reason"),
                        label_confidence=label.get("confidence"),
                        source="first_invalid_step",
                    )
                )
        elif label.get("final_correct") is True and not label.get("first_invalid_step"):
            candidate_indices = list(range(1, len(steps) + 1))
            rng.shuffle(candidate_indices)
            for idx in candidate_indices[: min(2, len(candidate_indices))]:
                priors = steps[max(0, idx - 1 - max_prior_steps) : idx - 1]
                valid.append(
                    RealStepCase(
                        case_id=f"{label['trace_id']}::step-{idx}",
                        expected_status="valid",
                        trace_id=label["trace_id"],
                        problem_id=str(label.get("problem_id") or trace.get("problem_id")),
                        problem=problem,
                        prior_steps=priors,
                        current_step=steps[idx - 1],
                        label_reason=label.get("reason"),
                        label_confidence=label.get("confidence"),
                        source="correct_trace_step",
                    )
                )

    rng.shuffle(invalid)
    rng.shuffle(valid)
    selected = invalid[:n_invalid] + valid[:n_valid]
    rng.shuffle(selected)
    return selected


def load_model(model_path: str, adapter_path: str | None, dtype: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype_map[dtype],
        low_cpu_mem_usage=True,
    )
    model.to(device=device, dtype=dtype_map[dtype])
    if adapter_path:
        payload = torch.load(adapter_path, map_location="cpu")
        config = payload["config"]
        add_lora(
            model,
            int(config["r"]),
            int(config["alpha"]),
            float(config["dropout"]),
            tuple(config["target_names"]),
        )
        model.to(device=device, dtype=dtype_map[dtype])
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        print(json.dumps({"adapter": adapter_path, "missing": len(missing), "unexpected": len(unexpected)}))
    model.eval()
    return tokenizer, model


def evaluate_case(
    case: RealStepCase,
    tokenizer: Any,
    model: Any,
    verifier: SemanticLeanVerifier,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    formal_case = case.as_case()
    raw = generate_text(tokenizer, model, build_direct_prompt(formal_case), args.max_new_tokens, "FINAL_COMMON_CODE:\n")
    code = extract_common_code("FINAL_COMMON_CODE:\n" + raw)
    repair_raw = None
    if code is None:
        return {
            **asdict(case),
            "status": "extraction_failed",
            "matches_expected": False,
            "seconds": round(time.time() - started, 3),
            "raw_generation": raw,
        }
    result = verifier.verify(build_task(formal_case, code))
    for _ in range(args.repair_rounds):
        if result.status in {"valid", "invalid"}:
            break
        repair_raw = generate_text(
            tokenizer,
            model,
            build_repair_prompt(formal_case, code, lean_error_text(result)),
            args.max_new_tokens,
            "FINAL_COMMON_CODE:\n",
        )
        repaired = extract_common_code("FINAL_COMMON_CODE:\n" + repair_raw)
        if repaired is None or repaired == code:
            break
        code = repaired
        result = verifier.verify(build_task(formal_case, code))

    return {
        **asdict(case),
        "status": result.status,
        "reason": result.reason,
        "matches_expected": result.status == case.expected_status,
        "seconds": round(time.time() - started, 3),
        "code": code,
        "raw_generation": raw,
        "repair_raw": repair_raw,
        "prove_stdout": result.prove.stdout[-1200:] if result.prove else None,
        "refute_stdout": result.refute.stdout[-1200:] if result.refute else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = sorted({str(row.get("status")) for row in rows})
    decisive = [row for row in rows if row.get("status") in {"valid", "invalid"}]
    return {
        "total": len(rows),
        "decisive": len(decisive),
        "decisive_rate": round(len(decisive) / len(rows), 4) if rows else 0.0,
        "matches_expected": sum(1 for row in rows if row.get("matches_expected") is True),
        "agreement_rate": round(sum(1 for row in rows if row.get("matches_expected") is True) / len(rows), 4) if rows else 0.0,
        "decisive_agreement_rate": round(sum(1 for row in decisive if row.get("matches_expected") is True) / len(decisive), 4) if decisive else 0.0,
        "statuses": {status: sum(1 for row in rows if row.get("status") == status) for status in statuses},
        "by_expected": {
            expected: {
                "total": sum(1 for row in rows if row.get("expected_status") == expected),
                "decisive": sum(1 for row in rows if row.get("expected_status") == expected and row.get("status") in {"valid", "invalid"}),
                "matches": sum(1 for row in rows if row.get("expected_status") == expected and row.get("matches_expected") is True),
            }
            for expected in ["valid", "invalid"]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3 LoRA Lean formalizer on real FHIS steps.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-invalid", type=int, default=50)
    parser.add_argument("--n-valid", type=int, default=50)
    parser.add_argument("--max-prior-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--repair-rounds", type=int, default=1)
    parser.add_argument("--lean-executable", default="lean")
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    cases = build_cases(
        args.labels,
        args.traces,
        seed=args.seed,
        n_invalid=args.n_invalid,
        n_valid=args.n_valid,
        max_prior_steps=args.max_prior_steps,
    )
    tokenizer, model = load_model(args.model, args.adapter, args.dtype, args.device)
    verifier = SemanticLeanVerifier(args.lean_executable, timeout_s=args.timeout_s)
    rows = [evaluate_case(case, tokenizer, model, verifier, args) for case in cases]
    payload = {
        "config": {
            "model": args.model,
            "adapter": args.adapter,
            "labels": args.labels,
            "traces": args.traces,
            "n_invalid": args.n_invalid,
            "n_valid": args.n_valid,
            "max_prior_steps": args.max_prior_steps,
            "repair_rounds": args.repair_rounds,
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
