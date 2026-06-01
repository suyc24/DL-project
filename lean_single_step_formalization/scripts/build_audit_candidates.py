#!/usr/bin/env python3
"""Build high-risk single-step audit candidates from scored CoT steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_metadata(paths: list[Path]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for idx, row in enumerate(read_jsonl(path)):
            pid = str(row.get("id") or row.get("problem_id") or f"problem_{idx:04d}")
            metadata[pid] = row
    return metadata


def build_selected_step_row(
    chain: dict[str, Any],
    step: dict[str, Any],
    score: dict[str, Any],
    *,
    context_before: int,
    context_after: int,
    include_final_answer: bool,
) -> dict[str, Any]:
    steps = chain.get("steps", [])
    step_by_index = {int(row["step_index"]): row for row in steps}
    ordered_indices = sorted(step_by_index)
    step_id = int(step["step_index"])
    selected_pos = ordered_indices.index(step_id)
    context_indices = ordered_indices[
        max(0, selected_pos - context_before) : selected_pos + context_after + 1
    ]
    return {
        "id": chain["id"],
        "question": chain["question"],
        "chain_id": chain["chain_index"],
        "step_id": step_id,
        "target_step": step["text"],
        "previous_steps": [
            step_by_index[i]["text"] for i in ordered_indices if i < step_id
        ],
        "context_steps": [
            {
                "step_id": i,
                "text": step_by_index[i]["text"],
                "is_selected": i == step_id,
            }
            for i in context_indices
        ],
        "final_answer": chain.get("final_answer") if include_final_answer else None,
        "selection_score": score,
    }


def candidate_reason(score: dict[str, Any]) -> str:
    reasons = []
    if int(score.get("risk", 0)) >= 4:
        reasons.append("high_risk")
    if int(score.get("verification_value", 0)) >= 4:
        reasons.append("high_verification_value")
    if int(score.get("lean_feasibility", 0)) >= 4:
        reasons.append("lean_feasible")
    if score.get("low_value"):
        reasons.append("low_value")
    return ",".join(reasons) or "top_ranked"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="run directory containing cot/ and selection/")
    parser.add_argument("--scores", default=None, help="step_scores.jsonl path; defaults under run dir")
    parser.add_argument("--output", default=None, help="audit_candidates.jsonl path")
    parser.add_argument("--selected-output", default=None, help="optional selected_steps-compatible output")
    parser.add_argument("--source-problems", nargs="*", default=[], help="optional original JSONL files with answer/difficulty metadata")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-risk", type=int, default=4)
    parser.add_argument("--min-value", type=int, default=4)
    parser.add_argument("--include-low-value", action="store_true")
    parser.add_argument("--include-final-answer", action="store_true")
    parser.add_argument("--context-before", type=int, default=5)
    parser.add_argument("--context-after", type=int, default=2)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    scores_path = Path(args.scores) if args.scores else run_dir / "selection" / "step_scores.jsonl"
    if not scores_path.exists():
        fallback = run_dir / "selection" / "value_selector_probe" / "step_scores.jsonl"
        if fallback.exists():
            scores_path = fallback
    output_path = Path(args.output) if args.output else run_dir / "selection" / "audit_candidates.jsonl"

    parsed_path = run_dir / "cot" / "cot_steps.jsonl"
    if not parsed_path.exists():
        raise FileNotFoundError(f"missing parsed CoT file: {parsed_path}")
    if not scores_path.exists():
        raise FileNotFoundError(f"missing step scores file: {scores_path}")

    parsed = read_jsonl(parsed_path)
    scores = read_jsonl(scores_path)
    chain_by_key = {(row["id"], row["chain_index"]): row for row in parsed}
    metadata = load_metadata([Path(p) for p in args.source_problems])

    candidates: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for score_record in scores:
        key = (score_record["id"], score_record["chain_id"])
        chain = chain_by_key.get(key)
        if not chain:
            continue
        step_by_id = {int(step["step_index"]): step for step in chain.get("steps", [])}
        ranked = sorted(
            score_record.get("scores", []),
            key=lambda row: (
                row.get("combined_score", -999),
                row.get("verification_value", 0),
                row.get("risk", 0),
                row.get("lean_feasibility", 0),
            ),
            reverse=True,
        )
        accepted: list[dict[str, Any]] = []
        for score in ranked:
            if int(score.get("step_id", -1)) not in step_by_id:
                continue
            if score.get("low_value") and not args.include_low_value:
                continue
            risk = int(score.get("risk", 0))
            value = int(score.get("verification_value", 0))
            if risk < args.min_risk and value < args.min_value and len(accepted) >= args.top_k:
                continue
            if risk < args.min_risk and value < args.min_value:
                continue
            accepted.append(score)
            if len(accepted) >= args.top_k:
                break
        if not accepted:
            accepted = [
                score for score in ranked
                if int(score.get("step_id", -1)) in step_by_id
                and (args.include_low_value or not score.get("low_value"))
            ][: args.top_k]

        meta = metadata.get(str(chain["id"]), {})
        for score in accepted:
            step = step_by_id[int(score["step_id"])]
            selected_row = build_selected_step_row(
                chain,
                step,
                score,
                context_before=args.context_before,
                context_after=args.context_after,
                include_final_answer=args.include_final_answer,
            )
            selected_rows.append(selected_row)
            candidates.append(
                {
                    **selected_row,
                    "source": meta.get("source"),
                    "difficulty": meta.get("difficulty") or meta.get("level"),
                    "subject": meta.get("subject") or meta.get("domain"),
                    "gold_answer": meta.get("answer"),
                    "model_final_answer": chain.get("final_answer"),
                    "candidate_reason": candidate_reason(score),
                    "original_cot": chain.get("original_text", ""),
                }
            )

    write_jsonl(candidates, output_path)
    if args.selected_output:
        write_jsonl(selected_rows, Path(args.selected_output))
    print(json.dumps({
        "candidates": len(candidates),
        "output": str(output_path),
        "selected_output": args.selected_output,
        "scores": str(scores_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
