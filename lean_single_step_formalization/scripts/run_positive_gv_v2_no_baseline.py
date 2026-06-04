#!/usr/bin/env python3
"""Run v2 Lean/GV on accepted positive controls without baseline or step_d calls."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
DEFAULT_STEPS = (
    ROOT
    / "experiments"
    / "runs"
    / "opc_positive_stepd_controls_codex55_high_50_002"
    / "input"
    / "opc_positive_valid_rows.jsonl"
)

sys.path.insert(0, str(SCRIPT_DIR))
from make_adversarial_steps import safe_id  # noqa: E402
from run_loop import cfg_get, parse_reasoning, read_config, read_jsonl, write_json, write_jsonl  # noqa: E402
import run_adversarial_game_gv_v2 as gv2  # noqa: E402


def load_rows(paths: list[str], limit: int | None) -> list[dict[str, Any]]:
    expanded: list[Path] = []
    for item in paths:
        matches = [Path(path) for path in glob.glob(item)]
        expanded.extend(matches or [Path(item)])
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in sorted(expanded):
        for row in read_jsonl(path):
            row_id = str(row.get("id") or "")
            if not row_id or row_id in seen:
                continue
            if not isinstance(row.get("step_decomposition"), dict):
                raise ValueError(f"{row_id} is missing embedded step_decomposition")
            seen.add(row_id)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def manual_initial(row: dict[str, Any]) -> dict[str, Any]:
    annotation = ((row.get("manual_annotation") or {}).get("annotation") or {})
    confidence = annotation.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 4
    return {
        "verdict": "valid",
        "reason": (
            "Positive-control row: OPC source proof was judged correct and the local "
            "Codex-as-manual annotation marked this target step faithful, locally correct, "
            "and self-contained enough."
        ),
        "confidence": confidence,
    }


def final_generator_report(lean: dict[str, Any]) -> dict[str, Any]:
    events = lean.get("generator_events") or []
    if not events:
        return {}
    report = events[-1].get("report") if isinstance(events[-1], dict) else {}
    return report if isinstance(report, dict) else {}


def final_review_action(lean: dict[str, Any]) -> str | None:
    decisions = lean.get("verifier_decisions") or []
    if not decisions:
        return None
    decision = decisions[-1].get("decision") if isinstance(decisions[-1], dict) else {}
    if isinstance(decision, dict):
        return decision.get("action")
    return None


def review_prompts(lean: dict[str, Any]) -> list[str]:
    prompts: list[str] = []
    for item in lean.get("verifier_decisions") or []:
        if isinstance(item, dict) and item.get("review_prompt"):
            prompts.append(str(item["review_prompt"]))
    return prompts


def run_case(
    row: dict[str, Any],
    *,
    case_id: str,
    case_dir: Path,
    provider: str,
    model: str | None,
    mock: bool,
    llm_timeout: int,
    lean_max_tokens: int,
    judge_max_tokens: int,
    project_dir: Path,
    lean_timeout: int,
    repair_rounds: int,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
    codex_reasoning_effort: str,
    codex_sandbox: str,
    codex_cwd: str,
    resume: bool,
) -> dict[str, Any]:
    result_path = case_dir / "case_result.json"
    if resume and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(row, case_dir / "positive_row.json")
    write_jsonl([row], case_dir / "positive_row.jsonl")
    thread_dir = case_dir / "threads"
    thread_dir.mkdir(parents=True, exist_ok=True)

    initial = manual_initial(row)
    step_decomposition = row["step_decomposition"]
    started = time.time()
    lean = gv2.run_gv_lean_assist_v2(
        row,
        round_dir=case_dir,
        provider=provider,
        model=model,
        mock=mock,
        llm_timeout=llm_timeout,
        lean_max_tokens=lean_max_tokens,
        judge_max_tokens=judge_max_tokens,
        project_dir=project_dir,
        lean_timeout=lean_timeout,
        repair_rounds=repair_rounds,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
        generator_thread_file=str(thread_dir / "generator.thread"),
        verifier_thread_dir=thread_dir / "verifier_reviews",
        initial_override=initial,
        initial_errors_override=[],
        step_decomposition_override=step_decomposition,
        step_decomposition_errors_override=[],
    )
    judgment = lean.get("judgment") or {}
    final_report = final_generator_report(lean)
    record = {
        "case_id": case_id,
        "row_id": row.get("id"),
        "source_id": row.get("source_id") or (row.get("opc") or {}).get("problem_id"),
        "case_dir": str(case_dir),
        "row": row,
        "initial": initial,
        "step_decomposition": step_decomposition,
        "lean": lean,
        "gv_verdict": judgment.get("verdict"),
        "gv_reason": judgment.get("reason"),
        "gv_stage": lean.get("stage"),
        "final_compile_ok": final_report.get("compile_ok"),
        "final_compile_reason": final_report.get("reason"),
        "generator_attempts": len(lean.get("generator_events") or []),
        "verifier_reviews": len(lean.get("verifier_decisions") or []),
        "final_review_action": final_review_action(lean),
        "review_prompts": review_prompts(lean),
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(record, result_path)
    return record


def aggregate_summary(records: list[dict[str, Any]], group_summaries: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    verdicts = Counter(str(record.get("gv_verdict")) for record in records)
    stages = Counter(str(record.get("gv_stage")) for record in records)
    compile_status = Counter(str(record.get("final_compile_ok")) for record in records)
    review_actions = Counter(str(record.get("final_review_action")) for record in records)
    false_invalid = [record["case_id"] for record in records if record.get("gv_verdict") != "valid"]
    summary = {
        "cases": len(records),
        "groups": len(group_summaries),
        "gv_valid": verdicts.get("valid", 0),
        "gv_invalid": verdicts.get("invalid", 0),
        "gv_other": len(records) - verdicts.get("valid", 0) - verdicts.get("invalid", 0),
        "false_invalid": len(false_invalid),
        "false_invalid_case_ids": false_invalid,
        "final_compile_ok": compile_status.get("True", 0),
        "final_compile_fail": compile_status.get("False", 0),
        "final_compile_unknown": compile_status.get("None", 0),
        "verdict_counts": dict(verdicts),
        "stage_counts": dict(stages),
        "compile_status_counts": dict(compile_status),
        "review_action_counts": dict(review_actions),
        "avg_generator_attempts": (
            sum(int(record.get("generator_attempts") or 0) for record in records) / len(records)
            if records
            else 0
        ),
        "avg_verifier_reviews": (
            sum(int(record.get("verifier_reviews") or 0) for record in records) / len(records)
            if records
            else 0
        ),
        "group_summaries": group_summaries,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(summary, run_dir / "summary.json")
    return summary


def write_report(run_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Positive-Control GV v2 No-Baseline Run",
        "",
        f"- cases: {summary['cases']}",
        f"- gv_valid: {summary['gv_valid']}",
        f"- gv_invalid: {summary['gv_invalid']}",
        f"- false_invalid: {summary['false_invalid']}",
        f"- final_compile_ok: {summary['final_compile_ok']}",
        f"- final_compile_fail: {summary['final_compile_fail']}",
        f"- avg_generator_attempts: {summary['avg_generator_attempts']:.2f}",
        f"- avg_verifier_reviews: {summary['avg_verifier_reviews']:.2f}",
        "",
        "False-invalid cases:",
    ]
    for case_id in summary["false_invalid_case_ids"]:
        lines.append(f"- `{case_id}`")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", action="append", default=[str(DEFAULT_STEPS)])
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--parallel-cases", type=int, default=5)
    parser.add_argument("--llm-provider", choices=["openai", "codex"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--llm-timeout", type=int, default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--lean-max-tokens", type=int, default=None)
    parser.add_argument("--repair-rounds", type=int, default=3)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--lean-timeout", type=int, default=None)
    parser.add_argument("--reasoning", choices=["auto", "enabled", "disabled"], default=None)
    parser.add_argument("--openai-reasoning-effort", choices=["high", "max"], default=None)
    parser.add_argument("--codex-reasoning-effort", default=None)
    parser.add_argument("--codex-sandbox", default=None)
    parser.add_argument("--codex-cwd", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    provider = args.llm_provider or os.environ.get("LLM_PROVIDER") or cfg_get(config, "llm.provider", "codex")
    model = args.model or cfg_get(config, "llm.model", None)
    llm_timeout = args.llm_timeout if args.llm_timeout is not None else int(cfg_get(config, "llm.timeout", 900))
    lean_max_tokens = args.lean_max_tokens if args.lean_max_tokens is not None else int(cfg_get(config, "llm.lean_max_tokens", 4096))
    project_dir = Path(args.project_dir or cfg_get(config, "paths.lean_project_dir", "/root/mathlib4"))
    lean_timeout = args.lean_timeout if args.lean_timeout is not None else int(cfg_get(config, "lean.timeout", 120))
    reasoning = parse_reasoning(args.reasoning if args.reasoning is not None else cfg_get(config, "llm.reasoning", None))
    openai_reasoning_effort = args.openai_reasoning_effort or cfg_get(config, "llm.openai_reasoning_effort", None)
    codex_reasoning_effort = args.codex_reasoning_effort or os.environ.get("CODEX_REASONING_EFFORT") or cfg_get(config, "llm.codex_reasoning_effort", "high")
    codex_sandbox = args.codex_sandbox or os.environ.get("CODEX_SANDBOX") or cfg_get(config, "llm.codex_sandbox", "danger-full-access")
    codex_cwd = args.codex_cwd or cfg_get(config, "llm.codex_cwd", str(ROOT.parent))

    rows = load_rows(args.steps or [str(DEFAULT_STEPS)], args.max_cases)
    run_dir = RUNS_DIR / args.run_id
    input_dir = run_dir / "input"
    groups_dir = run_dir / "groups"
    input_dir.mkdir(parents=True, exist_ok=True)
    groups_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, input_dir / "positive_rows.jsonl")
    write_json(
        {
            "run_id": args.run_id,
            "runner": "run_positive_gv_v2_no_baseline.py",
            "prompt_dir": str(gv2.PROMPT_DIR),
            "steps": args.steps or [str(DEFAULT_STEPS)],
            "cases_loaded": len(rows),
            "max_cases": args.max_cases,
            "batch_size": args.batch_size,
            "parallel_cases": args.parallel_cases,
            "provider": "mock" if args.mock else provider,
            "model": "mock" if args.mock else model,
            "repair_rounds": args.repair_rounds,
            "codex_reasoning_effort": codex_reasoning_effort,
            "codex_sandbox": codex_sandbox,
            "codex_cwd": codex_cwd,
            "baseline": "skipped",
            "step_decomposition": "embedded positive-control step_d reused",
            "generator_single_thread": True,
            "verifier_review_thread": "one fresh thread file per review attempt",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        run_dir / "run_config.json",
    )

    all_records: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    for group_idx, start in enumerate(range(0, len(rows), args.batch_size), start=1):
        group_rows = rows[start : start + args.batch_size]
        if len(group_rows) < args.batch_size:
            print(f"Skipping partial group {group_idx}: only {len(group_rows)} rows.", flush=True)
            break
        group_dir = groups_dir / f"group_{group_idx:03d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(group_rows, group_dir / "positive_rows.jsonl")
        print(f"Starting group {group_idx:03d} with {len(group_rows)} cases.", flush=True)

        group_records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.parallel_cases) as executor:
            futures = []
            for local_idx, row in enumerate(group_rows, start=1):
                case_id = f"{group_idx:03d}_{local_idx}_{safe_id(str(row['id']))}"
                futures.append(
                    executor.submit(
                        run_case,
                        row,
                        case_id=case_id,
                        case_dir=group_dir / case_id,
                        provider=provider,
                        model=model,
                        mock=args.mock,
                        llm_timeout=llm_timeout,
                        lean_max_tokens=lean_max_tokens,
                        judge_max_tokens=args.judge_max_tokens,
                        project_dir=project_dir,
                        lean_timeout=lean_timeout,
                        repair_rounds=args.repair_rounds,
                        reasoning=reasoning,
                        openai_reasoning_effort=openai_reasoning_effort,
                        codex_reasoning_effort=codex_reasoning_effort,
                        codex_sandbox=codex_sandbox,
                        codex_cwd=codex_cwd,
                        resume=args.resume,
                    )
                )
            for future in as_completed(futures):
                record = future.result()
                group_records.append(record)
                print(
                    json.dumps(
                        {
                            "case_id": record["case_id"],
                            "gv_verdict": record.get("gv_verdict"),
                            "gv_stage": record.get("gv_stage"),
                            "final_compile_ok": record.get("final_compile_ok"),
                            "generator_attempts": record.get("generator_attempts"),
                            "verifier_reviews": record.get("verifier_reviews"),
                            "final_review_action": record.get("final_review_action"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        group_records.sort(key=lambda record: record["case_id"])
        all_records.extend(group_records)
        group_summary = {
            "group": group_idx,
            "cases": len(group_records),
            "gv_valid": sum(1 for record in group_records if record.get("gv_verdict") == "valid"),
            "gv_invalid": sum(1 for record in group_records if record.get("gv_verdict") == "invalid"),
            "final_compile_ok": sum(1 for record in group_records if record.get("final_compile_ok") is True),
            "final_compile_fail": sum(1 for record in group_records if record.get("final_compile_ok") is False),
            "case_ids": [record["case_id"] for record in group_records],
        }
        group_summaries.append(group_summary)
        write_json(group_summary, group_dir / "group_summary.json")
        summary = aggregate_summary(all_records, group_summaries, run_dir)
        write_report(run_dir, summary)
        print(json.dumps({"group_summary": group_summary, "cumulative": summary}, ensure_ascii=False), flush=True)

    summary = aggregate_summary(all_records, group_summaries, run_dir)
    write_report(run_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
