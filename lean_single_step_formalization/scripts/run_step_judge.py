#!/usr/bin/env python3
"""Evaluate targeted step judgments with optional wrapped/Lean evidence."""
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
from llm_client import LLMCallError, call_llm
from run_reflection import issue_found, issue_type

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


def safe_name(row: dict[str, Any], mode: str) -> str:
    raw = f"{row['id']}_c{row['chain_id']}_s{row['step_id']}_{mode}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")[:160] or "judgment"


def key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("id"), row.get("chain_id"), row.get("step_id")


def load_audit_evidence(run_dir: Path | None) -> dict[tuple[Any, Any, Any], dict[str, Any]]:
    if run_dir is None:
        return {}
    wrapped_path = run_dir / "wrapped_claims" / "wrapped_claims.jsonl"
    manifest_path = run_dir / "lean" / "lean_generation_manifest.jsonl"
    verification_path = run_dir / "verification" / "verification.json"
    wrapped_rows = read_jsonl(wrapped_path) if wrapped_path.exists() else []
    manifest_rows = read_jsonl(manifest_path) if manifest_path.exists() else []
    verification_rows = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else []
    manifest_by_key = {key(row): row for row in manifest_rows}
    verification_by_file = {str(Path(row["file"])): row for row in verification_rows}
    evidence: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for wrapped in wrapped_rows:
        manifest = manifest_by_key.get(key(wrapped), {})
        lean_file = Path(manifest["lean_file"]) if manifest.get("lean_file") else None
        lean_code = lean_file.read_text(encoding="utf-8") if lean_file and lean_file.exists() else ""
        verification = verification_by_file.get(str(lean_file), {}) if lean_file else {}
        issue = issue_type(verification, wrapped, lean_code) if verification or lean_code else "none"
        evidence[key(wrapped)] = {
            "wrapped_claim": wrapped.get("wrapped_claim", {}),
            "wrap_valid": wrapped.get("wrap_valid"),
            "wrap_errors": wrapped.get("wrap_errors", []),
            "low_value": wrapped.get("low_value"),
            "low_value_reason": wrapped.get("low_value_reason", ""),
            "lean_code": lean_code,
            "verification": verification,
            "lean_summary": {
                "ok": verification.get("ok"),
                "dependency_mode": verification.get("dependency_mode"),
                "issue_type": issue,
                "issue_found": issue_found(issue),
                "declared_axioms": verification.get("declared_axioms", []),
                "local_missing_hypotheses": verification.get("local_missing_hypotheses", []),
                "kernel_axioms": verification.get("kernel_axioms", []),
                "lean_file": str(lean_file) if lean_file else "",
            },
        }
    return evidence


def build_user_prompt(row: dict[str, Any], mode: str, evidence: dict[str, Any]) -> str:
    context = "\n".join(
        f"{step['step_id']}. {'[目标] ' if step.get('is_selected') else ''}{step['text']}"
        for step in row.get("context_steps", [])
    )
    parts = [
        f"评测模式：{mode}",
        f"题目：\n{row['question']}",
        f"CoT 上下文：\n{context or '(none)'}",
        f"目标步骤：\n{row['target_step']}",
        f"模型最终答案：\n{row.get('model_final_answer') or row.get('final_answer') or '(unknown)'}",
        f"标准答案：\n{row.get('gold_answer') or '(unknown)'}",
    ]
    if mode in {"wrapped_only", "lean_assisted"}:
        parts.append("wrapped_claim：\n" + json.dumps(evidence.get("wrapped_claim", {}), ensure_ascii=False, indent=2))
        if evidence.get("wrap_errors"):
            parts.append("wrapped_claim 校验问题：\n" + json.dumps(evidence.get("wrap_errors"), ensure_ascii=False, indent=2))
    if mode == "lean_assisted":
        lean_summary = evidence.get("lean_summary", {})
        verification = evidence.get("verification", {})
        lean_code = evidence.get("lean_code", "")
        error_text = (verification.get("stdout") or "") + "\n" + (verification.get("stderr") or "")
        if len(lean_code) > 5000:
            lean_code = lean_code[:2500] + "\n\n...（中间省略）...\n\n" + lean_code[-2000:]
        if len(error_text) > 5000:
            error_text = error_text[-5000:]
        parts.extend(
            [
                "Lean 验证摘要：\n" + json.dumps(lean_summary, ensure_ascii=False, indent=2),
                f"Lean 代码：\n```lean\n{lean_code}\n```",
                f"Lean 输出/错误：\n```text\n{error_text}\n```",
            ]
        )
    parts.append("请判断目标步骤是否可靠。")
    return "\n\n".join(parts)


def validate_judgment(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("verdict") not in {"valid", "invalid", "uncertain"}:
        errors.append("invalid verdict")
    allowed_issues = {
        "none",
        "missing_premise",
        "too_strong",
        "algebra_error",
        "inequality_direction",
        "quantifier_swap",
        "modular_condition",
        "boundary_case",
        "necessity_sufficiency",
        "formalization_issue",
        "other",
    }
    if record.get("issue_type") not in allowed_issues:
        errors.append("invalid issue_type")
    for field in ["reason", "suggested_revision"]:
        if not isinstance(record.get(field), str):
            errors.append(f"{field} must be string")
    try:
        confidence = int(record.get("confidence"))
        if confidence < 1 or confidence > 5:
            errors.append("confidence out of range")
    except Exception:
        errors.append("confidence must be integer")
    return errors


def fallback_judgment(mode: str, row: dict[str, Any], evidence: dict[str, Any], reason: str) -> dict[str, Any]:
    if mode == "lean_assisted":
        issue = evidence.get("lean_summary", {}).get("issue_type", "none")
        if issue in {"global_axiom", "missing_hypothesis", "placeholder_prop"}:
            verdict = "invalid"
        elif issue in {"compile_error", "generation_failed"}:
            verdict = "uncertain"
        else:
            verdict = "valid"
        issue_map = {
            "global_axiom": "missing_premise",
            "missing_hypothesis": "missing_premise",
            "placeholder_prop": "formalization_issue",
            "compile_error": "formalization_issue",
            "generation_failed": "formalization_issue",
            "none": "none",
        }
        return {
            "verdict": verdict,
            "issue_type": issue_map.get(issue, "other"),
            "reason": f"自动保底判断：{reason}；Lean issue_type={issue}。",
            "suggested_revision": row.get("target_step", ""),
            "confidence": 2,
        }
    return {
        "verdict": "uncertain",
        "issue_type": "other",
        "reason": f"自动保底判断：{reason}。",
        "suggested_revision": row.get("target_step", ""),
        "confidence": 1,
    }


def mock_judgment(mode: str, row: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    gold = row.get("gold_verdict")
    if mode == "baseline" and row.get("adversarial"):
        verdict = "valid"
    elif mode == "wrapped_only" and row.get("adversarial"):
        verdict = "uncertain"
    elif mode == "lean_assisted" and evidence.get("lean_summary", {}).get("issue_found"):
        verdict = "invalid"
    else:
        verdict = gold if gold in {"valid", "invalid"} else "uncertain"
    return {
        "verdict": verdict,
        "issue_type": row.get("gold_issue_type", "none") if verdict == "invalid" else "none",
        "reason": f"mock {mode} 判断。",
        "suggested_revision": row.get("gold_corrected_step") or row.get("target_step", ""),
        "confidence": 3,
    }


def judge_one(
    row_mode: tuple[dict[str, Any], str],
    *,
    system_prompt: str,
    evidence_by_key: dict[tuple[Any, Any, Any], dict[str, Any]],
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
    row, mode = row_mode
    evidence = evidence_by_key.get(key(row), {})
    if mock:
        raw = json.dumps(mock_judgment(mode, row, evidence), ensure_ascii=False, indent=2)
        parsed = mock_judgment(mode, row, evidence)
        errors: list[str] = []
    else:
        try:
            raw = call_llm(
                system_prompt=system_prompt,
                user_prompt=build_user_prompt(row, mode, evidence),
                provider=provider,
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=llm_timeout,
                retries=2,
                reasoning=reasoning,
                openai_reasoning_effort=openai_reasoning_effort,
                codex_reasoning_effort=codex_reasoning_effort,
                codex_sandbox=codex_sandbox,
                codex_cwd=codex_cwd,
                call_label=f"{row['id']}-c{row['chain_id']}-s{row['step_id']}-{mode}-judge",
            )
            parsed = extract_json_object(raw)
            errors = validate_judgment(parsed)
        except (LLMCallError, Exception) as exc:
            raw = str(exc)
            parsed = fallback_judgment(mode, row, evidence, str(exc))
            errors = [str(exc)]
    gold = row.get("gold_verdict")
    correct = parsed.get("verdict") == gold if gold in {"valid", "invalid", "uncertain"} else None
    return {
        "id": row.get("id"),
        "chain_id": row.get("chain_id"),
        "step_id": row.get("step_id"),
        "mode": mode,
        "gold_verdict": gold,
        "gold_issue_type": row.get("gold_issue_type"),
        "adversarial": row.get("adversarial"),
        "judgment": parsed,
        "correct": correct,
        "validation_errors": errors,
        "raw_response": raw,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    modes = sorted({row["mode"] for row in results})
    by_mode: dict[str, Any] = {}
    for mode in modes:
        rows = [row for row in results if row["mode"] == mode]
        labeled = [row for row in rows if row.get("correct") is not None]
        invalid_rows = [row for row in rows if row.get("gold_verdict") == "invalid"]
        valid_rows = [row for row in rows if row.get("gold_verdict") == "valid"]
        by_mode[mode] = {
            "count": len(rows),
            "accuracy": (sum(1 for row in labeled if row.get("correct")) / len(labeled)) if labeled else None,
            "invalid_detection_rate": (
                sum(1 for row in invalid_rows if row.get("judgment", {}).get("verdict") == "invalid") / len(invalid_rows)
            ) if invalid_rows else None,
            "valid_accept_rate": (
                sum(1 for row in valid_rows if row.get("judgment", {}).get("verdict") == "valid") / len(valid_rows)
            ) if valid_rows else None,
            "uncertain_rate": (
                sum(1 for row in rows if row.get("judgment", {}).get("verdict") == "uncertain") / len(rows)
            ) if rows else None,
            "invalid_json_or_fallback": sum(1 for row in rows if row.get("validation_errors")),
        }
    def result_map(mode: str) -> dict[tuple[Any, Any, Any], dict[str, Any]]:
        return {key(row): row for row in results if row["mode"] == mode}
    rescue = None
    if {"baseline", "wrapped_only", "lean_assisted"}.issubset(set(modes)):
        baseline = result_map("baseline")
        wrapped = result_map("wrapped_only")
        lean = result_map("lean_assisted")
        candidates = [
            k for k, row in baseline.items()
            if row.get("gold_verdict") == "invalid"
            and row.get("judgment", {}).get("verdict") != "invalid"
            and wrapped.get(k, {}).get("judgment", {}).get("verdict") != "invalid"
        ]
        rescued = [
            k for k in candidates
            if lean.get(k, {}).get("judgment", {}).get("verdict") == "invalid"
        ]
        rescue = {
            "denominator": len(candidates),
            "rescued": len(rescued),
            "rescue_rate": len(rescued) / len(candidates) if candidates else None,
        }
    return {"by_mode": by_mode, "lean_rescue": rescue}


def write_report(results: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    lines = ["# Targeted Step Judge Report", ""]
    lines.append("## Summary")
    lines.append("")
    for mode, row in summary["by_mode"].items():
        lines.append(
            f"- {mode}: count={row['count']}, accuracy={row['accuracy']}, "
            f"invalid_detection_rate={row['invalid_detection_rate']}, valid_accept_rate={row['valid_accept_rate']}"
        )
    if summary.get("lean_rescue"):
        rescue = summary["lean_rescue"]
        lines.append(
            f"- lean_rescue: rescued={rescue['rescued']}/{rescue['denominator']}, "
            f"rate={rescue['rescue_rate']}"
        )
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(key(row), []).append(row)
    for case_key, rows in grouped.items():
        first = rows[0]
        lines.append(f"### {case_key[0]} c{case_key[1]} s{case_key[2]}")
        lines.append("")
        lines.append(f"- gold: {first.get('gold_verdict')} / {first.get('gold_issue_type')}")
        for row in sorted(rows, key=lambda item: item["mode"]):
            judgment = row.get("judgment", {})
            reason = str(judgment.get("reason", "")).replace("\n", " ")
            if len(reason) > 240:
                reason = reason[:240] + "..."
            lines.append(
                f"- {row['mode']}: {judgment.get('verdict')} / {judgment.get('issue_type')} "
                f"(correct={row.get('correct')}, confidence={judgment.get('confidence')}) - {reason}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", required=True, help="JSONL with target steps and optional gold_verdict")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-run-dir", default=None, help="run_selected_audit.py output dir for wrapped/Lean evidence")
    parser.add_argument("--modes", nargs="+", choices=["baseline", "wrapped_only", "lean_assisted"], default=["baseline", "wrapped_only", "lean_assisted"])
    parser.add_argument("--llm-provider", choices=["openai", "codex"], default="codex")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--llm-timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--reasoning", choices=["auto", "enabled", "disabled"], default="auto")
    parser.add_argument("--openai-reasoning-effort", choices=["high", "max"], default=None)
    parser.add_argument("--codex-reasoning-effort", default="high")
    parser.add_argument("--codex-sandbox", default="read-only")
    parser.add_argument("--codex-cwd", default=str(ROOT.parent))
    args = parser.parse_args()

    reasoning = None
    if args.reasoning == "enabled":
        reasoning = True
    elif args.reasoning == "disabled":
        reasoning = False

    rows = read_jsonl(Path(args.steps))
    evidence = load_audit_evidence(Path(args.audit_run_dir)) if args.audit_run_dir else {}
    system_prompt = load_prompt("step_judge.md")
    tasks = [(row, mode) for row in rows for mode in args.modes]
    worker_kwargs = {
        "system_prompt": system_prompt,
        "evidence_by_key": evidence,
        "provider": args.llm_provider,
        "model": args.model,
        "mock": args.mock,
        "llm_timeout": args.llm_timeout,
        "max_tokens": args.max_tokens,
        "reasoning": reasoning,
        "openai_reasoning_effort": args.openai_reasoning_effort,
        "codex_reasoning_effort": args.codex_reasoning_effort,
        "codex_sandbox": args.codex_sandbox,
        "codex_cwd": args.codex_cwd,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(judge_one, task, **worker_kwargs) for task in tasks]
        results = [future.result() for future in futures]

    out_dir = Path(args.output_dir)
    response_dir = out_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    for row in results:
        response_path = response_dir / f"{safe_name(row, row['mode'])}.response.txt"
        response_path.write_text(row.pop("raw_response", ""), encoding="utf-8")
        row["response_file"] = str(response_path)
    write_jsonl(results, out_dir / "judgments.jsonl")
    summary = summarize(results)
    write_json(summary, out_dir / "summary.json")
    write_report(results, summary, out_dir / "report.md")
    print(json.dumps({**summary, "output_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
