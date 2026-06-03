#!/usr/bin/env python3
"""Run model reflection on Lean single-step audit results."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"

sys.path.insert(0, str(SCRIPT_DIR))
from llm_client import call_llm

JSON_BLOCK_RE = re.compile(r"```\s*(?:json)?\s*\n(.*?)```", re.I | re.S)


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


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def extract_json_object(text: str) -> dict[str, Any]:
    block_match = JSON_BLOCK_RE.search(text)
    candidate = block_match.group(1).strip() if block_match else text.strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")
    parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("JSON root is not an object")
    return parsed


def issue_type(verification: dict[str, Any], wrapped: dict[str, Any], lean_code: str) -> str:
    if verification.get("ok") is False:
        return "compile_error"
    if "LLM call failed" in lean_code:
        return "generation_failed"
    if verification.get("dependency_mode") == "global_axiom_fallback":
        return "global_axiom"
    if verification.get("dependency_mode") == "local_missing_hypotheses":
        return "missing_hypothesis"
    if wrapped.get("low_value") is True:
        return "low_value"
    if re.search(r"\{?\s*P\d?\s+Q?\s*R?\s*C\s*:\s*Prop", lean_code):
        return "placeholder_prop"
    return "none"


def issue_found(issue: str) -> bool:
    return issue not in {"none"}


def build_user_prompt(row: dict[str, Any]) -> str:
    selected_context = "\n".join(
        f"{step['step_id']}. {'[选中] ' if step.get('is_selected') else ''}{step['text']}"
        for step in row.get("context_steps", [])
    )
    verification = row.get("verification", {})
    lean_excerpt = row.get("lean_code", "")
    if len(lean_excerpt) > 6000:
        lean_excerpt = lean_excerpt[:3000] + "\n\n...（中间省略）...\n\n" + lean_excerpt[-2500:]
    error_text = (verification.get("stdout") or "") + "\n" + (verification.get("stderr") or "")
    if len(error_text) > 5000:
        error_text = error_text[-5000:]
    return (
        f"题目：\n{row['question']}\n\n"
        f"CoT 上下文：\n{selected_context or '(none)'}\n\n"
        f"选中步骤：\n{row['target_step']}\n\n"
        f"模型最终答案：\n{row.get('model_final_answer') or row.get('final_answer') or '(unknown)'}\n\n"
        f"标准答案：\n{row.get('gold_answer') or '(unknown)'}\n\n"
        "wrapped_claim：\n"
        f"{json.dumps(row.get('wrapped_claim', {}), ensure_ascii=False, indent=2)}\n\n"
        "Lean 验证摘要：\n"
        f"{json.dumps(row.get('lean_summary', {}), ensure_ascii=False, indent=2)}\n\n"
        f"Lean 代码：\n```lean\n{lean_excerpt}\n```\n\n"
        f"Lean 输出/错误：\n```text\n{error_text}\n```\n"
    )


def validate_reflection(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    allowed = {"valid", "missing_premise", "too_strong", "wrong", "low_value", "formalization_issue"}
    if record.get("step_status") not in allowed:
        errors.append("invalid step_status")
    if not isinstance(record.get("issue_found"), bool):
        errors.append("issue_found must be boolean")
    for key in ["diagnosis", "revised_step", "impact_on_solution", "revised_final_answer"]:
        if key not in record or not isinstance(record.get(key), str):
            errors.append(f"{key} must be string")
    if not isinstance(record.get("added_premises"), list):
        errors.append("added_premises must be list")
    return errors


def fallback_reflection(row: dict[str, Any], reason: str) -> dict[str, Any]:
    issue = row.get("lean_summary", {}).get("issue_type", "none")
    status = {
        "compile_error": "formalization_issue",
        "generation_failed": "formalization_issue",
        "global_axiom": "missing_premise",
        "missing_hypothesis": "missing_premise",
        "low_value": "low_value",
        "placeholder_prop": "formalization_issue",
        "none": "valid",
    }.get(issue, "formalization_issue")
    return {
        "step_status": status,
        "issue_found": issue_found(issue),
        "diagnosis": f"自动保底反思：{reason}；Lean issue_type={issue}。",
        "revised_step": row.get("target_step", ""),
        "added_premises": [],
        "impact_on_solution": "需要人工进一步判断该局部问题是否影响后续推理。",
        "revised_final_answer": "",
    }


def load_audit_rows(run_dir: Path, candidates_path: Path | None, issue_only: bool) -> list[dict[str, Any]]:
    wrapped_rows = read_jsonl(run_dir / "wrapped_claims" / "wrapped_claims.jsonl")
    manifest_rows = read_jsonl(run_dir / "lean" / "lean_generation_manifest.jsonl")
    verification_rows = json.loads((run_dir / "verification" / "verification.json").read_text(encoding="utf-8"))
    candidates = read_jsonl(candidates_path) if candidates_path and candidates_path.exists() else []
    candidate_by_key = {
        (row["id"], row["chain_id"], row["step_id"]): row for row in candidates
    }
    manifest_by_key = {
        (row["id"], row["chain_id"], row["step_id"]): row for row in manifest_rows
    }
    verification_by_file = {str(Path(row["file"])): row for row in verification_rows}

    audit_rows: list[dict[str, Any]] = []
    for wrapped in wrapped_rows:
        key = (wrapped["id"], wrapped["chain_id"], wrapped["step_id"])
        manifest = manifest_by_key.get(key)
        if not manifest:
            continue
        lean_file = Path(manifest["lean_file"])
        lean_code = lean_file.read_text(encoding="utf-8") if lean_file.exists() else ""
        verification = verification_by_file.get(str(lean_file), {})
        issue = issue_type(verification, wrapped, lean_code)
        if issue_only and not issue_found(issue):
            continue
        candidate = candidate_by_key.get(key, {})
        lean_summary = {
            "ok": verification.get("ok"),
            "dependency_mode": verification.get("dependency_mode"),
            "issue_type": issue,
            "declared_axioms": verification.get("declared_axioms", []),
            "local_missing_hypotheses": verification.get("local_missing_hypotheses", []),
            "kernel_axioms": verification.get("kernel_axioms", []),
            "repair_rounds_used": manifest.get("repair_rounds_used"),
            "lean_file": str(lean_file),
        }
        audit_rows.append(
            {
                **wrapped,
                "gold_answer": candidate.get("gold_answer"),
                "model_final_answer": candidate.get("model_final_answer") or wrapped.get("final_answer"),
                "candidate_reason": candidate.get("candidate_reason"),
                "original_cot": candidate.get("original_cot"),
                "selection_score": candidate.get("selection_score") or wrapped.get("selection_score"),
                "lean_code": lean_code,
                "verification": verification,
                "lean_summary": lean_summary,
                "response_file": manifest.get("response_file"),
                "repair_history_file": manifest.get("repair_history_file"),
            }
        )
    return audit_rows


def reflect_one(
    row: dict[str, Any],
    *,
    system_prompt: str,
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
) -> dict[str, Any]:
    if mock:
        raw = json.dumps(fallback_reflection(row, "mock"), ensure_ascii=False, indent=2)
    else:
        raw = call_llm(
            system_prompt=system_prompt,
            user_prompt=build_user_prompt(row),
            provider=provider,
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=llm_timeout,
            retries=1,
            reasoning=reasoning,
            openai_reasoning_effort=openai_reasoning_effort,
            codex_reasoning_effort=codex_reasoning_effort,
            codex_sandbox=codex_sandbox,
            codex_cwd=codex_cwd,
            call_label=f"{row['id']}-c{row['chain_id']}-s{row['step_id']}-reflect",
        )
    try:
        parsed = extract_json_object(raw)
        errors = validate_reflection(parsed)
    except Exception as exc:
        parsed = fallback_reflection(row, str(exc))
        errors = [str(exc)]
    if errors:
        parsed = fallback_reflection(row, "; ".join(errors))
    return {
        "id": row["id"],
        "chain_id": row["chain_id"],
        "step_id": row["step_id"],
        "target_step": row["target_step"],
        "lean_summary": row["lean_summary"],
        "reflection": parsed,
        "reflection_valid": not errors,
        "reflection_errors": errors,
        "raw_response": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--provider", choices=["openai", "codex"], default="codex")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--llm-timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--issue-only", action="store_true", help="reflect only rows with Lean audit issues")
    parser.add_argument("--reasoning", choices=["auto", "enabled", "disabled"], default="auto")
    parser.add_argument("--openai-reasoning-effort", choices=["high", "max"], default=None)
    parser.add_argument("--codex-reasoning-effort", default="high")
    parser.add_argument("--codex-sandbox", default="danger-full-access")
    parser.add_argument("--codex-cwd", default=str(ROOT.parent))
    args = parser.parse_args()

    reasoning = None
    if args.reasoning == "enabled":
        reasoning = True
    elif args.reasoning == "disabled":
        reasoning = False

    run_dir = Path(args.run_dir)
    candidates_path = Path(args.candidates) if args.candidates else None
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "reflection"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = load_audit_rows(run_dir, candidates_path, args.issue_only)
    system_prompt = load_prompt("lean_feedback_reflect.md")

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                reflect_one,
                row,
                system_prompt=system_prompt,
                provider=args.provider,
                model=args.model,
                mock=args.mock,
                llm_timeout=args.llm_timeout,
                max_tokens=args.max_tokens,
                reasoning=reasoning,
                openai_reasoning_effort=args.openai_reasoning_effort,
                codex_reasoning_effort=args.codex_reasoning_effort,
                codex_sandbox=args.codex_sandbox,
                codex_cwd=args.codex_cwd,
            )
            for row in audit_rows
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (row["id"], row["chain_id"], row["step_id"]))

    for row in results:
        raw_path = out_dir / f"{row['id'].replace('/', '_')}_c{row['chain_id']}_s{row['step_id']}.response.txt"
        raw_path.write_text(row.pop("raw_response"), encoding="utf-8")
        row["reflection_response_file"] = str(raw_path)
    write_jsonl(results, out_dir / "reflections.jsonl")

    issue_count = sum(1 for row in results if row.get("lean_summary", {}).get("issue_type") != "none")
    helped_count = sum(
        1
        for row in results
        if row.get("reflection", {}).get("issue_found") is True
        and row.get("reflection", {}).get("step_status") not in {"valid", "formalization_issue"}
    )
    summary = {
        "reflections": len(results),
        "lean_issue_rows": issue_count,
        "reflection_valid": sum(1 for row in results if row.get("reflection_valid") is True),
        "reflection_invalid": sum(1 for row in results if row.get("reflection_valid") is not True),
        "actionable_feedback": helped_count,
        "output": str(out_dir / "reflections.jsonl"),
    }
    write_json(summary, out_dir / "reflection_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
