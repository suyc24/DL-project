from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from fhis.config import load_config
from fhis.io import read_jsonl
from fhis.steps import rough_answer_match


def answer_matches(predicted: str | None, references: Any) -> bool | None:
    if predicted is None:
        return None
    if references is None:
        return None
    if isinstance(references, list):
        return any(rough_answer_match(predicted, str(reference)) for reference in references)
    return rough_answer_match(predicted, str(references))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize online selective-verification results.")
    parser.add_argument("--config", default="data_generation/qwen25_fhis/configs/online_verify.yaml")
    parser.add_argument("--results", default=None)
    parser.add_argument("--problems", default=None)
    parser.add_argument(
        "--output",
        default="data_generation/qwen25_fhis/results/online_verify_summary.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results_path = args.results or config["paths"].get(
        "online_results",
        config["paths"].get("probe_retry_results"),
    )
    if results_path is None:
        raise KeyError("config.paths must define online_results or probe_retry_results")
    problems_path = args.problems or config["paths"]["problems"]
    problems = {
        str(row.get("problem_id", row.get("id", "unknown"))): row for row in read_jsonl(problems_path)
    }
    rows = list(read_jsonl(results_path))

    accepted = [row for row in rows if row.get("status") == "accepted"]
    abstained = [row for row in rows if row.get("status") == "abstained"]
    correctness = []
    correct_by_retry_source = {
        "fresh_sample_no_retry": 0,
        "step_retry_same_trace": 0,
        "whole_problem_restart": 0,
    }
    for row in accepted:
        problem = problems.get(str(row.get("problem_id")))
        references = problem.get("reference_answer") if problem else None
        is_correct = answer_matches(row.get("answer"), references)
        correctness.append(is_correct)
        if is_correct is True:
            decisions = row.get("decisions") or []
            attempts_used = int(row.get("attempts_used", 0) or 0)
            had_restart = attempts_used > 1 or any(
                str(decision.get("action")) == "restart_trace" for decision in decisions
            )
            had_step_retry = any(
                str(decision.get("action")) == "retry_step"
                or decision.get("feedback_used") is True
                for decision in decisions
            )
            if had_restart:
                correct_by_retry_source["whole_problem_restart"] += 1
            elif had_step_retry:
                correct_by_retry_source["step_retry_same_trace"] += 1
            else:
                correct_by_retry_source["fresh_sample_no_retry"] += 1

    routed_steps = 0
    generated_steps = 0
    verification_status_counts: Counter[str] = Counter()
    attempts = 0
    classifier_flags = 0
    step_retry_decisions = 0
    feedback_decisions = 0
    parse_failures = 0
    for row in rows:
        attempts += int(row.get("attempts_used", 0) or 0)
        decisions = row.get("decisions") or []
        generated_steps += len(decisions)
        for decision in decisions:
            classifier_flags += int(decision.get("flagged_for_retry") is True)
            step_retry_decisions += int(str(decision.get("action")) == "retry_step")
            feedback_decisions += int(decision.get("feedback_used") is True)
            parse_failures += int(decision.get("parse_ok") is False)
            if decision.get("routed_to_lean"):
                routed_steps += 1
                status = str(decision.get("verification_status"))
                verification_status_counts[status] += 1

    solved = sum(1 for item in correctness if item is True)
    known = sum(1 for item in correctness if item is not None)
    payload = {
        "num_problems": len(rows),
        "accepted": len(accepted),
        "abstained": len(abstained),
        "answer_rate": len(accepted) / len(rows) if rows else None,
        "rough_solve_rate_all": solved / len(rows) if rows else None,
        "rough_solve_rate_answered": solved / known if known else None,
        "correct_by_retry_source": correct_by_retry_source,
        "generated_steps": generated_steps,
        "lean_calls": routed_steps,
        "verification_rate": routed_steps / generated_steps if generated_steps else None,
        "classifier_flags": classifier_flags,
        "classifier_flag_rate": classifier_flags / generated_steps if generated_steps else None,
        "step_retry_decisions": step_retry_decisions,
        "feedback_decisions": feedback_decisions,
        "parse_failures": parse_failures,
        "attempts": attempts,
        "avg_attempts_per_problem": attempts / len(rows) if rows else None,
        "lean_calls_per_solved_problem": routed_steps / solved if solved else None,
        "verification_status": dict(verification_status_counts),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
