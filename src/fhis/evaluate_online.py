from __future__ import annotations

import argparse
import json
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
    results_path = args.results or config["paths"]["online_results"]
    problems_path = args.problems or config["paths"]["problems"]
    problems = {
        str(row.get("problem_id", row.get("id", "unknown"))): row for row in read_jsonl(problems_path)
    }
    rows = list(read_jsonl(results_path))

    accepted = [row for row in rows if row.get("status") == "accepted"]
    abstained = [row for row in rows if row.get("status") == "abstained"]
    correctness = []
    for row in accepted:
        problem = problems.get(str(row.get("problem_id")))
        references = problem.get("reference_answer") if problem else None
        correctness.append(answer_matches(row.get("answer"), references))

    routed_steps = 0
    generated_steps = 0
    proved = 0
    failed = 0
    formalization_failed = 0
    attempts = 0
    for row in rows:
        attempts += int(row.get("attempts_used", 0) or 0)
        decisions = row.get("decisions") or []
        generated_steps += len(decisions)
        for decision in decisions:
            if decision.get("routed_to_lean"):
                routed_steps += 1
                status = decision.get("verification_status")
                proved += int(status == "proved")
                failed += int(status == "failed")
                formalization_failed += int(status == "formalization_failed")

    solved = sum(1 for item in correctness if item is True)
    known = sum(1 for item in correctness if item is not None)
    payload = {
        "num_problems": len(rows),
        "accepted": len(accepted),
        "abstained": len(abstained),
        "answer_rate": len(accepted) / len(rows) if rows else None,
        "rough_solve_rate_all": solved / len(rows) if rows else None,
        "rough_solve_rate_answered": solved / known if known else None,
        "generated_steps": generated_steps,
        "lean_calls": routed_steps,
        "verification_rate": routed_steps / generated_steps if generated_steps else None,
        "attempts": attempts,
        "avg_attempts_per_problem": attempts / len(rows) if rows else None,
        "lean_calls_per_solved_problem": routed_steps / solved if solved else None,
        "verification_status": {
            "proved": proved,
            "failed": failed,
            "formalization_failed": formalization_failed,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
