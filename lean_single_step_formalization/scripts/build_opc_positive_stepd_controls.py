#!/usr/bin/env python3
"""Build positive-control OPC step decompositions from correct solutions.

This script does not modify existing runners. It reads OPC rows whose proof was
judged correct, extracts local proof-step candidates, asks a local LLM to
annotate whether each candidate is a faithful valid step, and then runs the v2
step decomposition prompt on accepted candidates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"

sys.path.insert(0, str(SCRIPT_DIR))
import run_adversarial_game_gv_v2 as gv2  # noqa: E402


ANNOTATION_SYSTEM_PROMPT = """\
You are a careful human-style olympiad proof annotator.

Return only JSON, with no Markdown.

Your task is to decide whether the target proof step is a correct positive-control
step from the supplied OPC solution. The full OPC solution was externally judged
correct, but you must still inspect the local target step.

Accept a candidate only when all of these are true:
- the target step is faithfully copied from the supplied solution;
- in the problem and previous/context proof, the target step is mathematically correct;
- the target step has a real local mathematical claim or derivation, not just a title,
  theorem restatement, or final boxed answer;
- ordinary olympiad conventions and standard named facts used in the solution are allowed.

Reject or mark uncertain if the step is too broad, mostly narrative, missing the
actual mathematical claim, or if local correctness cannot be judged from context.
Use confidence 4 or 5 for a clear valid candidate. Use confidence 1 or 2 only
when the candidate is invalid or genuinely uncertain.

Schema:
{
  "label": "valid | invalid | uncertain",
  "faithful_to_solution": true,
  "locally_correct": true,
  "self_contained_enough": true,
  "reason": "short reason",
  "confidence": 4
}
"""


MATH_MARKER_RE = re.compile(
    r"(\\\[|\\\(|\$|=|≤|≥|<|>|\\le|\\ge|\\equiv|\\mid|\\sum|\\prod|"
    r"pmod|mod|divisible|divides|factor|prime|integer|positive|"
    r"therefore|hence|thus|so|implies|follows|contradiction)",
    re.IGNORECASE,
)
BAD_START_RE = re.compile(
    r"^\s*(Here is|Proof\.|Solution\.|Theorem\.|We will show|We shall show|"
    r"It remains|This completes|Thus the proof|Therefore the proof|In conclusion)",
    re.IGNORECASE,
)
REASONING_WORD_RE = re.compile(
    r"\b(Since|Then|Hence|Therefore|Thus|So|But|If|Suppose|For|By|Using|"
    r"Applying|Observe|It follows|This gives|We get|we have)\b"
)
GEOMETRY_WORD_RE = re.compile(
    r"\b(triangle|circle|angle|perpendicular|parallel|circumcenter|midpoint|"
    r"line|tangent|chord|arc|orthocenter|altitude|bisector)\b",
    re.IGNORECASE,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_id(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    return value[:140] or "sample"


def is_correct_score(score: Any) -> bool:
    return isinstance(score, list) and 1 in score and 0 not in score


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_long_block(block: str) -> list[str]:
    if len(block) <= 1500:
        return [block]
    parts = re.split(
        r"(?<=[.!?。])\s+(?=(?:Since|Then|Hence|Therefore|Thus|Now|But|If|"
        r"Suppose|For|We|This|It|Finally|Conversely|So|By|Applying|Using)\b)",
        block,
    )
    return [part.strip() for part in parts if part.strip()]


def split_solution(solution: str) -> list[str]:
    solution = normalize_text(solution)
    raw_blocks = re.split(r"\n\s*(?:-{3,}|\*{3,})\s*\n|\n\s*\n+", solution)
    blocks: list[str] = []
    for raw in raw_blocks:
        block = raw.strip()
        if not block:
            continue
        blocks.extend(split_long_block(block))
    return blocks


def candidate_score(problem: str, text: str) -> float:
    score = 0.0
    if REASONING_WORD_RE.search(text):
        score += 2.5
    if "\\[" in text or "$$" in text:
        score += 2.0
    if re.search(r"(\\equiv|pmod|\\mid|divisible|divides)", text, re.IGNORECASE):
        score += 2.0
    if re.search(r"(\\le|\\ge|≤|≥|<|>|inequality)", text, re.IGNORECASE):
        score += 1.5
    if 180 <= len(text) <= 900:
        score += 1.5
    if 900 < len(text) <= 1300:
        score += 0.5
    if GEOMETRY_WORD_RE.search(problem + "\n" + text):
        score -= 0.75
    if text.strip().lower().startswith(("conclusion", "verification")):
        score -= 2.0
    return score


def looks_like_candidate(text: str) -> bool:
    stripped = text.strip()
    if not 120 <= len(stripped) <= 1700:
        return False
    if BAD_START_RE.search(stripped):
        return False
    if not MATH_MARKER_RE.search(stripped):
        return False
    if stripped.count("\n") == 0 and len(stripped.split()) < 18:
        return False
    return True


def make_context_steps(blocks: list[str], selected_index: int) -> list[dict[str, Any]]:
    start = max(0, selected_index - 3)
    end = min(len(blocks), selected_index + 2)
    return [
        {
            "step_id": idx + 1,
            "text": blocks[idx],
            "is_selected": idx == selected_index,
        }
        for idx in range(start, end)
    ]


def load_correct_rows(parquet_path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = pq.read_table(parquet_path).to_pylist()
    correct: list[tuple[int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        if not is_correct_score(row.get("score")):
            continue
        feedback = row.get("feedback") or []
        if feedback and not any("correct" in str(item).lower() for item in feedback):
            continue
        if not normalize_text(row.get("problem")) or not normalize_text(row.get("solution")):
            continue
        correct.append((idx, row))
    return correct


def build_candidates(correct_rows: list[tuple[int, dict[str, Any]]], pool_size: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order = 0
    for source_row_index, source in correct_rows:
        problem_id = str(source.get("problem_id") or f"row_{source_row_index}")
        problem = normalize_text(source.get("problem"))
        solution = normalize_text(source.get("solution"))
        blocks = split_solution(solution)
        for block_index, block in enumerate(blocks):
            if not looks_like_candidate(block):
                continue
            row_id = safe_id(f"opc_pos_{source_row_index:03d}_{block_index + 1:03d}_{problem_id}")
            candidate = {
                "id": row_id,
                "source_id": problem_id,
                "question": problem,
                "chain_id": source_row_index,
                "step_id": block_index + 1,
                "target_step": block,
                "previous_steps": blocks[:block_index],
                "context_steps": make_context_steps(blocks, block_index),
                "original_cot": solution,
                "mutated_cot": solution,
                "adversarial": False,
                "final_answer": None,
                "model_final_answer": None,
                "gold_answer": None,
                "gold_verdict": "valid",
                "gold_issue_type": "opc_correct_positive_control",
                "gold_diagnosis": "Positive control from an OPC proof judged correct; local Codex annotation accepts this target step as valid.",
                "candidate_score": candidate_score(problem, block),
                "candidate_order": order,
                "opc": {
                    "problem_id": problem_id,
                    "source_row_index": source_row_index,
                    "candidate_block_index": block_index,
                    "score": source.get("score"),
                    "feedback": source.get("feedback") or [],
                    "competition": source.get("competition"),
                    "category": source.get("category") or [],
                    "level": source.get("level"),
                    "year": source.get("year"),
                    "url": source.get("url"),
                    "annotation_target": block,
                    "annotation": {
                        "comment": "Positive-control local step selected from an OPC solution with score=[1].",
                        "selected_text": block,
                        "original_text": block,
                    },
                },
            }
            grouped.setdefault(problem_id, []).append(candidate)
            order += 1

    for rows in grouped.values():
        rows.sort(key=lambda item: (-float(item["candidate_score"]), int(item["candidate_order"])))

    problem_ids = sorted(grouped)
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < pool_size:
        added = False
        for problem_id in problem_ids:
            rows = grouped[problem_id]
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
                if len(selected) >= pool_size:
                    break
        if not added:
            break
        depth += 1
    return selected


def validate_positive_annotation(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("label") not in {"valid", "invalid", "uncertain"}:
        errors.append("label must be valid, invalid, or uncertain")
    for key in ("faithful_to_solution", "locally_correct", "self_contained_enough"):
        if not isinstance(payload.get(key), bool):
            errors.append(f"{key} must be boolean")
    if not isinstance(payload.get("reason"), str) or not payload.get("reason", "").strip():
        errors.append("reason must be a non-empty string")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or confidence < 1 or confidence > 5:
        errors.append("confidence must be a number from 1 to 5")
    return errors


def build_annotation_prompt(candidate: dict[str, Any]) -> str:
    context_text = "\n".join(
        f"{step['step_id']}. {'[TARGET] ' if step.get('is_selected') else ''}{step['text']}"
        for step in candidate.get("context_steps", [])
    )
    return (
        "# OPC positive-control candidate\n\n"
        f"## Candidate id\n{candidate['id']}\n\n"
        "## OPC external metadata\n"
        f"{json.dumps(candidate.get('opc') or {}, ensure_ascii=False, indent=2)}\n\n"
        "## Problem statement\n"
        f"{candidate['question']}\n\n"
        "## Local proof window\n"
        f"{context_text}\n\n"
        "## Target step\n"
        f"{candidate['target_step']}\n\n"
        "## Full correct OPC solution\n"
        f"{candidate['original_cot']}\n\n"
        "Return only the annotation JSON."
    )


def accepted_annotation(annotation: dict[str, Any], errors: list[str]) -> bool:
    return (
        not errors
        and annotation.get("label") == "valid"
        and annotation.get("faithful_to_solution") is True
        and annotation.get("locally_correct") is True
        and annotation.get("self_contained_enough") is True
        and float(annotation.get("confidence") or 0) >= 3
    )


def annotate_candidate(
    candidate: dict[str, Any],
    *,
    case_dir: Path,
    provider: str,
    model: str | None,
    mock: bool,
    llm_timeout: int,
    max_tokens: int,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
    codex_reasoning_effort: str,
    codex_sandbox: str,
    codex_cwd: str,
) -> tuple[dict[str, Any], str, list[str]]:
    annotation_dir = case_dir / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    result_path = annotation_dir / "positive_annotation.json"
    response_path = annotation_dir / "positive_annotation.response.txt"
    if result_path.exists() and response_path.exists():
        payload = read_json(result_path)
        return payload["annotation"], response_path.read_text(encoding="utf-8"), payload.get("errors") or []

    user_prompt = build_annotation_prompt(candidate)
    (annotation_dir / "positive_annotation.prompt.md").write_text(
        f"# System prompt\n\n{ANNOTATION_SYSTEM_PROMPT}\n\n# User prompt\n\n{user_prompt}",
        encoding="utf-8",
    )
    annotation, raw, errors = gv2.json_call_gv(
        system_prompt=ANNOTATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        provider=provider,
        model=model,
        mock_payload={
            "label": "valid",
            "faithful_to_solution": True,
            "locally_correct": True,
            "self_contained_enough": True,
            "reason": "mock valid annotation",
            "confidence": 4,
        },
        mock=mock,
        llm_timeout=llm_timeout,
        max_tokens=max_tokens,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
        codex_thread_file=str(case_dir / "threads" / "positive_annotation.thread"),
        call_label=f"{candidate['id']}-positive-annotation",
        validator=validate_positive_annotation,
    )
    response_path.write_text(raw, encoding="utf-8")
    write_json({"annotation": annotation, "errors": errors}, result_path)
    return annotation, raw, errors


def run_step_decomposition(
    candidate: dict[str, Any],
    annotation: dict[str, Any],
    *,
    case_dir: Path,
    provider: str,
    model: str | None,
    mock: bool,
    llm_timeout: int,
    max_tokens: int,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
    codex_reasoning_effort: str,
    codex_sandbox: str,
    codex_cwd: str,
) -> tuple[dict[str, Any], str, list[str]]:
    step_dir = case_dir / "step_decompose"
    result_path = step_dir / "step_decomposition.json"
    response_path = step_dir / "step_decompose.response.txt"
    if result_path.exists() and response_path.exists():
        payload = read_json(result_path)
        return payload["decomposition"], response_path.read_text(encoding="utf-8"), payload.get("errors") or []

    initial = {
        "verdict": "valid",
        "reason": "Positive-control local annotation accepted this OPC correct-proof step as faithful and locally correct.",
        "confidence": annotation.get("confidence") or 4,
    }
    return gv2.run_step_decomposition_for_row(
        candidate,
        out_dir=step_dir,
        initial=initial,
        provider=provider,
        model=model,
        mock=mock,
        llm_timeout=llm_timeout,
        max_tokens=max_tokens,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
        codex_thread_file=str(case_dir / "threads" / "step_decompose.thread"),
    )


def process_candidate(candidate: dict[str, Any], args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    case_dir = run_dir / "cases" / candidate["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "threads").mkdir(parents=True, exist_ok=True)
    write_json(candidate, case_dir / "candidate.json")
    result_path = case_dir / "case_result.json"
    if args.resume and result_path.exists():
        return read_json(result_path)

    started = time.time()
    try:
        annotation, _, annotation_errors = annotate_candidate(
            candidate,
            case_dir=case_dir,
            provider=args.llm_provider,
            model=args.model,
            mock=args.mock,
            llm_timeout=args.llm_timeout,
            max_tokens=args.annotation_max_tokens,
            reasoning=args.reasoning,
            openai_reasoning_effort=args.openai_reasoning_effort,
            codex_reasoning_effort=args.codex_reasoning_effort,
            codex_sandbox=gv2.agent_sandbox(args.llm_provider, args.codex_sandbox),
            codex_cwd=args.codex_cwd,
        )
        if not accepted_annotation(annotation, annotation_errors):
            record = {
                "status": "rejected_annotation",
                "case_id": candidate["id"],
                "candidate": candidate,
                "annotation": annotation,
                "annotation_errors": annotation_errors,
                "elapsed_sec": round(time.time() - started, 3),
            }
            write_json(record, result_path)
            return record

        step_decomposition, _, step_errors = run_step_decomposition(
            candidate,
            annotation,
            case_dir=case_dir,
            provider=args.llm_provider,
            model=args.model,
            mock=args.mock,
            llm_timeout=args.llm_timeout,
            max_tokens=args.stepd_max_tokens,
            reasoning=args.reasoning,
            openai_reasoning_effort=args.openai_reasoning_effort,
            codex_reasoning_effort=args.codex_reasoning_effort,
            codex_sandbox=gv2.agent_sandbox(args.llm_provider, args.codex_sandbox),
            codex_cwd=args.codex_cwd,
        )
        if step_errors:
            record = {
                "status": "rejected_stepd_errors",
                "case_id": candidate["id"],
                "candidate": candidate,
                "annotation": annotation,
                "annotation_errors": annotation_errors,
                "step_decomposition": step_decomposition,
                "step_decomposition_errors": step_errors,
                "elapsed_sec": round(time.time() - started, 3),
            }
            write_json(record, result_path)
            return record

        row = dict(candidate)
        row["manual_annotation"] = {
            "source": f"local_codex_{args.model or 'default'}_{args.codex_reasoning_effort}_treated_as_manual",
            "label": "valid",
            "annotation": annotation,
        }
        row["step_decomposition"] = step_decomposition
        record = {
            "status": "accepted",
            "case_id": candidate["id"],
            "row": row,
            "candidate": candidate,
            "annotation": annotation,
            "annotation_errors": annotation_errors,
            "step_decomposition": step_decomposition,
            "step_decomposition_errors": step_errors,
            "elapsed_sec": round(time.time() - started, 3),
        }
    except Exception as exc:
        record = {
            "status": "error",
            "case_id": candidate["id"],
            "candidate": candidate,
            "error": repr(exc),
            "elapsed_sec": round(time.time() - started, 3),
        }
    write_json(record, result_path)
    return record


def collect_existing_results(run_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted((run_dir / "cases").glob("*/case_result.json")):
        try:
            results.append(read_json(path))
        except Exception:
            continue
    return results


def write_outputs(run_dir: Path, candidates: list[dict[str, Any]], results: list[dict[str, Any]], max_examples: int) -> None:
    accepted = [row for row in results if row.get("status") == "accepted"]
    accepted.sort(key=lambda item: int((item.get("candidate") or {}).get("candidate_order", 10**9)))
    accepted = accepted[:max_examples]
    accepted_rows = [row["row"] for row in accepted]
    stepd_rows = [
        {
            "id": row["case_id"],
            "source_id": (row.get("candidate") or {}).get("source_id"),
            "target_step": (row.get("candidate") or {}).get("target_step"),
            "manual_annotation": row.get("annotation"),
            "step_decomposition": row.get("step_decomposition"),
        }
        for row in accepted
    ]
    write_jsonl(candidates, run_dir / "input" / "opc_positive_candidates.jsonl")
    write_jsonl(accepted_rows, run_dir / "input" / "opc_positive_valid_rows.jsonl")
    write_jsonl(stepd_rows, run_dir / "input" / "opc_positive_step_decompositions.jsonl")
    summary = {
        "run_id": run_dir.name,
        "candidate_pool": len(candidates),
        "processed": len(results),
        "accepted": len(accepted_rows),
        "rejected_annotation": sum(1 for row in results if row.get("status") == "rejected_annotation"),
        "rejected_stepd_errors": sum(1 for row in results if row.get("status") == "rejected_stepd_errors"),
        "errors": sum(1 for row in results if row.get("status") == "error"),
        "accepted_path": str(run_dir / "input" / "opc_positive_valid_rows.jsonl"),
        "step_decomposition_path": str(run_dir / "input" / "opc_positive_step_decompositions.jsonl"),
        "accepted_ids": [row["id"] for row in accepted_rows],
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(summary, run_dir / "summary.json")
    report_lines = [
        "# OPC Positive Step Decomposition Controls",
        "",
        f"- run_id: `{run_dir.name}`",
        f"- candidate_pool: {len(candidates)}",
        f"- processed: {len(results)}",
        f"- accepted: {len(accepted_rows)}",
        f"- rejected_annotation: {summary['rejected_annotation']}",
        f"- rejected_stepd_errors: {summary['rejected_stepd_errors']}",
        f"- errors: {summary['errors']}",
        f"- accepted_rows: `{summary['accepted_path']}`",
        f"- step_decompositions: `{summary['step_decomposition_path']}`",
        "",
        "Accepted cases:",
    ]
    for row in accepted_rows:
        problem_id = (row.get("opc") or {}).get("problem_id")
        report_lines.append(f"- `{row['id']}` from `{problem_id}`")
    (run_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="opc_positive_stepd_controls_codex55_high_50_001")
    parser.add_argument(
        "--parquet",
        default=str(ROOT / "data" / "external" / "opc" / "best_of_n.parquet"),
    )
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--candidate-pool", type=int, default=220)
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--llm-provider", default="codex")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--llm-timeout", type=int, default=900)
    parser.add_argument("--annotation-max-tokens", type=int, default=3000)
    parser.add_argument("--stepd-max-tokens", type=int, default=6000)
    parser.add_argument("--reasoning", action="store_true", default=None)
    parser.add_argument("--openai-reasoning-effort", default=None)
    parser.add_argument("--codex-reasoning-effort", default="high")
    parser.add_argument("--codex-sandbox", default="danger-full-access")
    parser.add_argument("--codex-cwd", default=str(ROOT.parent))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = Path(args.parquet)
    correct_rows = load_correct_rows(parquet_path)
    candidates = build_candidates(correct_rows, args.candidate_pool)
    write_json(
        {
            "run_id": args.run_id,
            "parquet": str(parquet_path),
            "correct_source_rows": len(correct_rows),
            "candidate_pool": len(candidates),
            "max_examples": args.max_examples,
            "provider": args.llm_provider,
            "model": args.model,
            "codex_reasoning_effort": args.codex_reasoning_effort,
            "parallel": args.parallel,
        },
        run_dir / "config.json",
    )
    write_jsonl(candidates, run_dir / "input" / "opc_positive_candidates.jsonl")

    existing = collect_existing_results(run_dir) if args.resume else []
    processed_ids = {row.get("case_id") for row in existing}
    accepted_count = sum(1 for row in existing if row.get("status") == "accepted")
    all_results = list(existing)
    if accepted_count >= args.max_examples:
        write_outputs(run_dir, candidates, all_results, args.max_examples)
        print(json.dumps(read_json(run_dir / "summary.json"), ensure_ascii=False, indent=2))
        return

    pending: dict[Any, dict[str, Any]] = {}
    remaining = [row for row in candidates if row["id"] not in processed_ids]
    cursor = 0
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        while accepted_count < args.max_examples and (cursor < len(remaining) or pending):
            while accepted_count + len(pending) < args.max_examples + args.parallel and cursor < len(remaining) and len(pending) < args.parallel:
                candidate = remaining[cursor]
                cursor += 1
                future = executor.submit(process_candidate, candidate, args, run_dir)
                pending[future] = candidate
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                candidate = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"status": "error", "case_id": candidate["id"], "candidate": candidate, "error": repr(exc)}
                all_results.append(result)
                if result.get("status") == "accepted":
                    accepted_count += 1
                write_outputs(run_dir, candidates, all_results, args.max_examples)
                print(
                    json.dumps(
                        {
                            "processed": len(all_results),
                            "accepted": accepted_count,
                            "last_case": result.get("case_id"),
                            "last_status": result.get("status"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    write_outputs(run_dir, candidates, all_results, args.max_examples)
    print(json.dumps(read_json(run_dir / "summary.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
