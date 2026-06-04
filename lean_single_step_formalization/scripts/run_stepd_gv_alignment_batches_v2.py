#!/usr/bin/env python3
"""Run v2 step_d -> baseline initial and split-review GV in parallel batches."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
DEFAULT_STEPS = "experiments/runs/opc_stepd_gv_alignment_50cases_construct_obligations_001/input/opc_rows_for_hacker.jsonl"

sys.path.insert(0, str(SCRIPT_DIR))
from make_adversarial_steps import safe_id
from run_loop import cfg_get, parse_reasoning, read_config, read_jsonl, write_json, write_jsonl
import run_adversarial_game_gv_v2 as gv2


def validate_alignment(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    score = payload.get("score")
    if not isinstance(score, (int, float)) or score < 0 or score > 10:
        errors.append("score must be a number from 0 to 10")
    if not isinstance(payload.get("aligned"), bool):
        errors.append("aligned must be boolean")
    if not isinstance(payload.get("reason"), str) or not payload.get("reason", "").strip():
        errors.append("reason must be non-empty string")
    return errors


def annotation_payload(row: dict[str, Any]) -> dict[str, Any]:
    opc = row.get("opc") or {}
    annotation = opc.get("annotation") or {}
    return {
        "gold_diagnosis": row.get("gold_diagnosis", ""),
        "annotation_comment": annotation.get("comment", ""),
        "annotation_target": opc.get("annotation_target", ""),
        "feedback": opc.get("feedback") or [],
        "problem_id": opc.get("problem_id") or row.get("source_id"),
    }


def build_alignment_prompt(row: dict[str, Any], method: str, judgment: dict[str, Any]) -> str:
    return (
        "# Alignment scoring input\n\n"
        f"## case_id\n{row.get('id')}\n\n"
        f"## method\n{method}\n\n"
        f"## human annotation\n{json.dumps(annotation_payload(row), ensure_ascii=False, indent=2)}\n\n"
        f"## verifier judgment\n{json.dumps(judgment, ensure_ascii=False, indent=2)}\n\n"
        "Return only the required JSON."
    )


def score_alignment(
    row: dict[str, Any],
    *,
    method: str,
    judgment: dict[str, Any],
    out_dir: Path,
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
    codex_thread_file: str | None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict = judgment.get("verdict")
    reason = str(judgment.get("reason", "")).strip()
    if verdict != "invalid" or not reason:
        result = {
            "score": 0,
            "aligned": False,
            "reason": "verifier did not return an invalid reason, so it cannot align with an invalid annotation.",
        }
        write_json(result, out_dir / f"{method}_annotation_alignment.json")
        return result

    system_prompt = gv2.load_prompt("annotation_alignment.md")
    user_prompt = build_alignment_prompt(row, method, judgment)
    (out_dir / f"{method}_annotation_alignment.prompt.md").write_text(
        f"# System prompt\n\n{system_prompt}\n\n# User prompt\n\n{user_prompt}",
        encoding="utf-8",
    )
    parsed, raw, errors = gv2.json_call_gv(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider,
        model=model,
        mock_payload={"score": 8, "aligned": True, "reason": "mock alignment"},
        mock=mock,
        llm_timeout=llm_timeout,
        max_tokens=max_tokens,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
        codex_thread_file=codex_thread_file,
        call_label=f"{row['id']}-{method}-annotation-alignment-v2",
        validator=validate_alignment,
    )
    if errors:
        parsed = {"score": 0, "aligned": False, "reason": "alignment scoring JSON failed validation: " + "; ".join(errors)}
    (out_dir / f"{method}_annotation_alignment.response.txt").write_text(raw, encoding="utf-8")
    write_json({"alignment": parsed, "errors": errors}, out_dir / f"{method}_annotation_alignment.json")
    return parsed


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
            if row_id in seen:
                continue
            seen.add(row_id)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def final_review_action(lean: dict[str, Any]) -> str:
    decisions = lean.get("verifier_decisions") or []
    if not decisions:
        return ""
    decision = decisions[-1].get("overridden_decision") or decisions[-1].get("decision") or {}
    return str(decision.get("action") or "")


def final_review_prompt(lean: dict[str, Any]) -> str:
    decisions = lean.get("verifier_decisions") or []
    if not decisions:
        return ""
    return str(decisions[-1].get("review_prompt") or "")


def is_lean_weak(record: dict[str, Any], weak_alignment_threshold: float) -> bool:
    lean = record.get("lean") or {}
    gv_alignment = record.get("gv_alignment") or {}
    gv_judgment = lean.get("judgment") or {}
    return not (
        lean.get("lean_used") is True
        and gv_judgment.get("verdict") == "invalid"
        and float(gv_alignment.get("score") or 0) >= weak_alignment_threshold
    )


def case_markdown(record: dict[str, Any]) -> str:
    row = record["row"]
    baseline_judgment = record["baseline"].get("judgment") or {}
    gv_judgment = record["lean"].get("judgment") or {}
    return (
        f"## {record['case_id']}\n\n"
        f"- source: `{row.get('source_id')}`\n"
        f"- annotation: {row.get('gold_diagnosis', '')}\n"
        f"- baseline: `{baseline_judgment.get('verdict')}`; alignment "
        f"{record['baseline_alignment'].get('score')}/10; {baseline_judgment.get('reason', '')}\n"
        f"- gv_v2: `{gv_judgment.get('verdict')}` stage `{record['lean'].get('stage')}` action "
        f"`{final_review_action(record['lean'])}` prompt `{final_review_prompt(record['lean'])}`; alignment "
        f"{record['gv_alignment'].get('score')}/10; {gv_judgment.get('reason', '')}\n"
        f"- weak: `{record.get('weak')}`\n"
    )


def write_group_report(group_dir: Path, group_records: list[dict[str, Any]], *, weak_alignment_threshold: float) -> dict[str, Any]:
    baseline_scores = [float(row["baseline_alignment"].get("score") or 0) for row in group_records]
    gv_scores = [float(row["gv_alignment"].get("score") or 0) for row in group_records]
    weak_records = [row for row in group_records if row.get("weak")]
    summaries = [row["lean"].get("verifier_decisions") or [] for row in group_records]
    final_prompts = [items[-1].get("review_prompt") for items in summaries if items]
    summary = {
        "cases": len(group_records),
        "baseline_invalid": sum(1 for row in group_records if (row["baseline"].get("judgment") or {}).get("verdict") == "invalid"),
        "gv_invalid": sum(1 for row in group_records if (row["lean"].get("judgment") or {}).get("verdict") == "invalid"),
        "gv_valid": sum(1 for row in group_records if (row["lean"].get("judgment") or {}).get("verdict") == "valid"),
        "baseline_alignment_avg": sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0,
        "gv_alignment_avg": sum(gv_scores) / len(gv_scores) if gv_scores else 0,
        "weak": len(weak_records),
        "weak_case_ids": [row["case_id"] for row in weak_records],
        "weak_alignment_threshold": weak_alignment_threshold,
        "final_review_compile_ok": sum(1 for name in final_prompts if name == "verifier_review_compile_ok.md"),
        "final_review_compile_fail": sum(1 for name in final_prompts if name == "verifier_review_compile_fail.md"),
    }
    write_json(summary, group_dir / "group_summary.json")
    write_jsonl(group_records, group_dir / "group_results.jsonl")
    md = [
        "# Group Report",
        "",
        f"- cases: {summary['cases']}",
        f"- baseline invalid: {summary['baseline_invalid']}",
        f"- GV v2 invalid: {summary['gv_invalid']}",
        f"- GV v2 valid: {summary['gv_valid']}",
        f"- baseline annotation alignment avg: {summary['baseline_alignment_avg']:.2f}/10",
        f"- GV v2 annotation alignment avg: {summary['gv_alignment_avg']:.2f}/10",
        f"- final review compile-ok: {summary['final_review_compile_ok']}",
        f"- final review compile-fail: {summary['final_review_compile_fail']}",
        f"- weak: {summary['weak']} ({', '.join(summary['weak_case_ids']) or 'none'})",
        "",
    ]
    for record in group_records:
        md.append(case_markdown(record))
        md.append("")
    (group_dir / "group_report.md").write_text("\n".join(md), encoding="utf-8")
    return summary


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
    verifier_initial_prompt: str,
    weak_alignment_threshold: float,
    resume: bool,
) -> dict[str, Any]:
    result_path = case_dir / "case_result.json"
    if resume and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    case_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl([row], case_dir / "adversarial_step.jsonl")
    thread_dir = case_dir / "threads"
    thread_dir.mkdir(parents=True, exist_ok=True)

    step_d, _, step_d_errors = gv2.run_step_decomposition_for_row(
        row,
        out_dir=case_dir / "step_decompose",
        initial={"verdict": "unknown", "reason": "baseline initial is run after step decomposition", "confidence": 0},
        provider=provider,
        model=model,
        mock=mock,
        llm_timeout=llm_timeout,
        max_tokens=judge_max_tokens,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=gv2.agent_sandbox(provider, codex_sandbox),
        codex_cwd=codex_cwd,
        codex_thread_file=str(thread_dir / "step_decompose.thread"),
    )

    baseline = gv2.run_baseline_verifier_judge_with_decomposition(
        row,
        round_dir=case_dir,
        step_decomposition=step_d,
        provider=provider,
        model=model,
        mock=mock,
        llm_timeout=llm_timeout,
        max_tokens=judge_max_tokens,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
        verifier_initial_prompt=verifier_initial_prompt,
    )
    baseline_judgment = baseline.get("judgment") or {}

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
        verifier_initial_prompt=verifier_initial_prompt,
        initial_override=baseline_judgment,
        initial_errors_override=baseline.get("errors") or [],
        step_decomposition_override=step_d,
        step_decomposition_errors_override=step_d_errors,
    )

    baseline_alignment = score_alignment(
        row,
        method="baseline",
        judgment=baseline_judgment,
        out_dir=case_dir / "alignment",
        provider=provider,
        model=model,
        mock=mock,
        llm_timeout=llm_timeout,
        max_tokens=judge_max_tokens,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
        codex_thread_file=str(thread_dir / "baseline_alignment.thread"),
    )
    gv_alignment = score_alignment(
        row,
        method="gv_v2",
        judgment=lean.get("judgment") or {},
        out_dir=case_dir / "alignment",
        provider=provider,
        model=model,
        mock=mock,
        llm_timeout=llm_timeout,
        max_tokens=judge_max_tokens,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
        codex_thread_file=str(thread_dir / "gv_v2_alignment.thread"),
    )

    record = {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "row": row,
        "step_decomposition": step_d,
        "step_decomposition_errors": step_d_errors,
        "baseline": baseline,
        "lean": lean,
        "baseline_alignment": baseline_alignment,
        "gv_alignment": gv_alignment,
    }
    record["weak"] = is_lean_weak(record, weak_alignment_threshold)
    write_json(record, result_path)
    return record


def aggregate_summary(records: list[dict[str, Any]], group_summaries: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    baseline_scores = [float(row["baseline_alignment"].get("score") or 0) for row in records]
    gv_scores = [float(row["gv_alignment"].get("score") or 0) for row in records]
    summary = {
        "cases": len(records),
        "groups": len(group_summaries),
        "baseline_invalid": sum(1 for row in records if (row["baseline"].get("judgment") or {}).get("verdict") == "invalid"),
        "gv_invalid": sum(1 for row in records if (row["lean"].get("judgment") or {}).get("verdict") == "invalid"),
        "gv_valid": sum(1 for row in records if (row["lean"].get("judgment") or {}).get("verdict") == "valid"),
        "baseline_alignment_avg": sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0,
        "gv_alignment_avg": sum(gv_scores) / len(gv_scores) if gv_scores else 0,
        "weak": sum(1 for row in records if row.get("weak")),
        "weak_case_ids": [row["case_id"] for row in records if row.get("weak")],
        "group_summaries": group_summaries,
    }
    write_json(summary, run_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", action="append", default=None, help="JSONL path or glob; repeatable")
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
    parser.add_argument("--verifier-initial-prompt", default="verifier_initial.md")
    parser.add_argument("--weak-alignment-threshold", type=float, default=4.0)
    parser.add_argument("--stop-weak-threshold", type=int, default=999)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

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

    rows = load_rows(args.steps or [DEFAULT_STEPS], args.max_cases)
    run_dir = RUNS_DIR / args.run_id
    input_dir = run_dir / "input"
    groups_dir = run_dir / "groups"
    input_dir.mkdir(parents=True, exist_ok=True)
    groups_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, input_dir / "steps.jsonl")
    write_json(
        {
            "run_id": args.run_id,
            "runner": "run_stepd_gv_alignment_batches_v2.py",
            "prompt_dir": str(gv2.PROMPT_DIR),
            "steps": args.steps or [DEFAULT_STEPS],
            "cases_loaded": len(rows),
            "max_cases": args.max_cases,
            "batch_size": args.batch_size,
            "parallel_cases": args.parallel_cases,
            "provider": "mock" if args.mock else provider,
            "model": "mock" if args.mock else model,
            "repair_rounds": args.repair_rounds,
            "weak_alignment_threshold": args.weak_alignment_threshold,
            "stop_weak_threshold": args.stop_weak_threshold,
            "codex_reasoning_effort": codex_reasoning_effort,
            "codex_sandbox": codex_sandbox,
            "codex_cwd": codex_cwd,
            "baseline_reuses_initial": True,
            "generator_single_thread": True,
            "verifier_review_thread": "one fresh thread file per review attempt",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        run_dir / "run_config.json",
    )
    if len(rows) < args.max_cases:
        print(f"Only {len(rows)} annotated rows loaded; requested {args.max_cases}.", flush=True)

    all_records: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    for group_idx, start in enumerate(range(0, len(rows), args.batch_size), start=1):
        group_rows = rows[start : start + args.batch_size]
        if len(group_rows) < args.batch_size:
            print(f"Skipping partial group {group_idx}: only {len(group_rows)} rows.", flush=True)
            break
        group_dir = groups_dir / f"group_{group_idx:03d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(group_rows, group_dir / "steps.jsonl")
        print(f"Starting group {group_idx:03d} with {len(group_rows)} cases.", flush=True)

        group_records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.parallel_cases) as executor:
            futures = []
            for local_idx, row in enumerate(group_rows, start=1):
                case_id = f"{group_idx:03d}_{local_idx}_{safe_id(row['id'])}"
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
                        verifier_initial_prompt=args.verifier_initial_prompt,
                        weak_alignment_threshold=args.weak_alignment_threshold,
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
                            "baseline_score": record["baseline_alignment"].get("score"),
                            "gv_v2_score": record["gv_alignment"].get("score"),
                            "gv_v2_verdict": (record["lean"].get("judgment") or {}).get("verdict"),
                            "gv_v2_stage": record["lean"].get("stage"),
                            "final_review_prompt": final_review_prompt(record["lean"]),
                            "weak": record.get("weak"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        group_records.sort(key=lambda row: row["case_id"])
        summary = write_group_report(group_dir, group_records, weak_alignment_threshold=args.weak_alignment_threshold)
        summary["group"] = group_idx
        summary["group_dir"] = str(group_dir)
        group_summaries.append(summary)
        all_records.extend(group_records)
        write_jsonl(all_records, run_dir / "all_case_results.jsonl")
        aggregate_summary(all_records, group_summaries, run_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        if summary["weak"] >= args.stop_weak_threshold:
            print(f"Stopping early: group {group_idx:03d} has {summary['weak']} weak cases.", flush=True)
            break

    final_summary = aggregate_summary(all_records, group_summaries, run_dir)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
