#!/usr/bin/env python3
"""Run an adaptive adversarial single-step verification game."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
RUNS_DIR = ROOT / "experiments" / "runs"
PROMPTS_DIR = ROOT / "prompts"

sys.path.insert(0, str(SCRIPT_DIR))
from llm_client import LLMCallError, call_llm
from make_adversarial_steps import (
    build_hacker_prompt,
    build_invalid_row,
    candidate_thread_file,
    extract_json_object,
    mock_attack,
    safe_id,
    validate_attack,
)
from run_loop import (
    check_lean_code,
    cfg_get,
    default_lean_project_dir,
    existing_default_config,
    extract_lean_code,
    generate_lean_contracts,
    generate_wrapped_claims,
    parse_reasoning,
    read_config,
    read_jsonl,
    safe_lean_name,
    verify_lean_outputs,
    write_json,
    write_jsonl,
)
from run_step_judge import judge_one, load_audit_evidence, load_prompt as load_judge_prompt, summarize


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def json_call(
    *,
    system_prompt: str,
    user_prompt: str,
    provider: str,
    model: str | None,
    mock_payload: dict[str, Any] | None,
    mock: bool,
    llm_timeout: int,
    max_tokens: int,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
    codex_reasoning_effort: str,
    codex_sandbox: str,
    codex_cwd: str,
    call_label: str,
) -> tuple[dict[str, Any], str, list[str]]:
    if mock:
        payload = mock_payload or {}
        return payload, json.dumps(payload, ensure_ascii=False, indent=2), []
    try:
        raw = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
            call_label=call_label,
        )
        return extract_json_object(raw), raw, []
    except (LLMCallError, Exception) as exc:
        return {"verdict": "uncertain", "issue_type": "other", "reason": str(exc), "confidence": 1}, str(exc), [str(exc)]


def build_context_text(row: dict[str, Any]) -> str:
    return "\n".join(
        f"{step['step_id']}. {'[目标] ' if step.get('is_selected') else ''}{step['text']}"
        for step in row.get("context_steps", [])
    ) or "(none)"


def build_initial_judge_prompt(row: dict[str, Any]) -> str:
    return (
        f"题目：\n{row['question']}\n\n"
        f"CoT 上下文：\n{build_context_text(row)}\n\n"
        f"目标步骤：\n{row['target_step']}\n\n"
        f"模型最终答案：\n{row.get('model_final_answer') or row.get('final_answer') or '(unknown)'}\n\n"
        f"标准答案：\n{row.get('gold_answer') or '(unknown)'}\n\n"
        "请判断目标步骤本身是否可靠。"
    )


def run_initial_judge(
    row: dict[str, Any],
    *,
    label: str,
    round_dir: Path,
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
    mock_payload = {
        "verdict": "valid" if not row.get("adversarial") else "uncertain",
        "issue_type": "none",
        "reason": f"mock {label} initial judge",
        "should_try_lean": True,
        "lean_target": row.get("target_step", ""),
        "confidence": 3,
    }
    parsed, raw, errors = json_call(
        system_prompt=load_prompt("adversarial_judge_initial.md"),
        user_prompt=build_initial_judge_prompt(row),
        provider=provider,
        model=model,
        mock_payload=mock_payload,
        mock=mock,
        llm_timeout=llm_timeout,
        max_tokens=max_tokens,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
        call_label=f"{row['id']}-{label}-initial-judge",
    )
    out_dir = round_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "initial_judge.response.txt").write_text(raw, encoding="utf-8")
    write_json({"judgment": parsed, "errors": errors}, out_dir / "initial_judge.json")
    return {"mode": label, "stage": "initial", "judgment": parsed, "errors": errors}


def build_lean_formalize_prompt(row: dict[str, Any], initial: dict[str, Any], theorem_name: str) -> str:
    return (
        f"theorem 名字：{theorem_name}\n\n"
        f"题目：\n{row['question']}\n\n"
        f"CoT 上下文：\n{build_context_text(row)}\n\n"
        f"目标步骤：\n{row['target_step']}\n\n"
        f"lean_target：\n{initial.get('lean_target') or row.get('target_step')}\n\n"
        "请直接形式化这个目标步骤。"
    )


def build_lean_review_prompt(
    row: dict[str, Any],
    initial: dict[str, Any],
    code: str,
    check: dict[str, Any],
) -> str:
    evidence = {
        "ok": check.get("ok"),
        "dependency_mode": check.get("dependency_mode"),
        "declared_axioms": check.get("declared_axioms", []),
        "local_missing_hypotheses": check.get("local_missing_hypotheses", []),
        "kernel_axioms": check.get("kernel_axioms", []),
        "stdout": check.get("stdout", "")[-3000:],
        "stderr": check.get("stderr", "")[-3000:],
    }
    return (
        f"题目：\n{row['question']}\n\n"
        f"CoT 上下文：\n{build_context_text(row)}\n\n"
        f"目标步骤：\n{row['target_step']}\n\n"
        f"初步判断：\n{json.dumps(initial, ensure_ascii=False, indent=2)}\n\n"
        f"Lean 代码：\n```lean\n{code}\n```\n\n"
        f"Lean 结果：\n{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
        "请重新判断目标步骤本身是否可靠。"
    )


def build_lean_repair_prompt(code: str, check: dict[str, Any], decision: dict[str, Any]) -> str:
    error_text = (check.get("stdout") or "") + "\n" + (check.get("stderr") or "")
    return (
        "上一版 Lean 没有通过，但判断结果要求继续修复。\n\n"
        f"修复方向：\n{decision.get('repair_instruction', '')}\n\n"
        f"原 Lean 代码：\n```lean\n{code}\n```\n\n"
        f"Lean 输出/错误：\n```text\n{error_text[-6000:]}\n```\n\n"
        "请返回修复后的完整 Lean 文件。"
    )


def run_direct_lean_assist(
    row: dict[str, Any],
    initial: dict[str, Any],
    *,
    round_dir: Path,
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
) -> dict[str, Any]:
    out_dir = round_dir / "lean_direct"
    out_dir.mkdir(parents=True, exist_ok=True)
    theorem_name = safe_lean_name(row)
    system_formalize = load_prompt("adversarial_lean_formalize.md")
    if mock:
        code = f"import Mathlib\n\ntheorem {theorem_name} : True := by\n  trivial\n"
        raw = f"```lean\n{code}```"
    else:
        raw = call_llm(
            system_prompt=system_formalize,
            user_prompt=build_lean_formalize_prompt(row, initial, theorem_name),
            provider=provider,
            model=model,
            temperature=0.0,
            max_tokens=lean_max_tokens,
            timeout=llm_timeout,
            retries=2,
            reasoning=reasoning,
            openai_reasoning_effort=openai_reasoning_effort,
            codex_reasoning_effort=codex_reasoning_effort,
            codex_sandbox=codex_sandbox,
            codex_cwd=codex_cwd,
            call_label=f"{row['id']}-direct-lean",
        )
        code = extract_lean_code(raw)
    (out_dir / "lean.response.txt").write_text(raw, encoding="utf-8")

    decisions: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    final_judgment: dict[str, Any] | None = None
    final_raw = ""
    final_errors: list[str] = []
    for attempt in range(repair_rounds + 1):
        lean_path = out_dir / f"attempt_{attempt}.lean"
        lean_path.write_text(code, encoding="utf-8")
        check = check_lean_code(
            code,
            project_dir=project_dir,
            timeout=lean_timeout,
            theorem_name=theorem_name,
            stem=f"{safe_id(row['id'])}_direct_attempt_{attempt}",
        )
        checks.append(check)
        write_json(check, out_dir / f"attempt_{attempt}.check.json")
        if check.get("ok") is True:
            mock_payload = {
                "verdict": "valid",
                "issue_type": "none",
                "reason": "mock lean pass review",
                "lean_evidence": "mock",
                "confidence": 3,
            }
            final_judgment, final_raw, final_errors = json_call(
                system_prompt=load_prompt("adversarial_lean_pass_review.md"),
                user_prompt=build_lean_review_prompt(row, initial, code, check),
                provider=provider,
                model=model,
                mock_payload=mock_payload,
                mock=mock,
                llm_timeout=llm_timeout,
                max_tokens=judge_max_tokens,
                reasoning=reasoning,
                openai_reasoning_effort=openai_reasoning_effort,
                codex_reasoning_effort=codex_reasoning_effort,
                codex_sandbox=codex_sandbox,
                codex_cwd=codex_cwd,
                call_label=f"{row['id']}-lean-pass-review",
            )
            break

        mock_payload = {
            "action": "return_invalid" if row.get("adversarial") else "return_uncertain",
            "verdict": "invalid" if row.get("adversarial") else "uncertain",
            "issue_type": row.get("gold_issue_type", "other"),
            "reason": "mock lean fail decision",
            "repair_instruction": "",
            "confidence": 3,
        }
        decision, raw_decision, errors = json_call(
            system_prompt=load_prompt("adversarial_lean_fail_decide.md"),
            user_prompt=build_lean_review_prompt(row, initial, code, check),
            provider=provider,
            model=model,
            mock_payload=mock_payload,
            mock=mock,
            llm_timeout=llm_timeout,
            max_tokens=judge_max_tokens,
            reasoning=reasoning,
            openai_reasoning_effort=openai_reasoning_effort,
            codex_reasoning_effort=codex_reasoning_effort,
            codex_sandbox=codex_sandbox,
            codex_cwd=codex_cwd,
            call_label=f"{row['id']}-lean-fail-decide-{attempt}",
        )
        decision_record = {"attempt": attempt, "decision": decision, "raw": raw_decision, "errors": errors}
        decisions.append(decision_record)
        if decision.get("action") != "continue_repair" or attempt >= repair_rounds:
            final_judgment = {
                "verdict": decision.get("verdict", "uncertain"),
                "issue_type": decision.get("issue_type", "other"),
                "reason": decision.get("reason", ""),
                "lean_evidence": "Lean 未通过；fail_decide 阶段返回。",
                "confidence": decision.get("confidence", 1),
            }
            final_raw = raw_decision
            final_errors = errors
            break
        if mock:
            code = code
        else:
            repair_raw = call_llm(
                system_prompt=load_prompt("adversarial_lean_repair.md"),
                user_prompt=build_lean_repair_prompt(code, check, decision),
                provider=provider,
                model=model,
                temperature=0.0,
                max_tokens=lean_max_tokens,
                timeout=llm_timeout,
                retries=1,
                reasoning=reasoning,
                openai_reasoning_effort=openai_reasoning_effort,
                codex_reasoning_effort=codex_reasoning_effort,
                codex_sandbox=codex_sandbox,
                codex_cwd=codex_cwd,
                call_label=f"{row['id']}-direct-lean-repair-{attempt + 1}",
            )
            (out_dir / f"repair_{attempt + 1}.response.txt").write_text(repair_raw, encoding="utf-8")
            code = extract_lean_code(repair_raw)

    (out_dir / "final_review.response.txt").write_text(final_raw, encoding="utf-8")
    result = {
        "mode": "lean_assisted",
        "stage": "lean_review",
        "judgment": final_judgment or {"verdict": "uncertain", "issue_type": "other", "reason": "no final judgment", "confidence": 1},
        "errors": final_errors,
        "lean_used": True,
        "lean_checks": checks,
        "fail_decisions": decisions,
    }
    write_json(result, out_dir / "lean_assisted_result.json")
    return result


def build_hacker_feedback_prompt(
    candidate: dict[str, Any],
    previous_attack: dict[str, Any],
    judgments: list[dict[str, Any]],
    round_idx: int,
    round_status: str,
) -> str:
    compact_judgments = [
        {
            "mode": row["mode"],
            "verdict": row.get("judgment", {}).get("verdict"),
            "issue_type": row.get("judgment", {}).get("issue_type"),
            "reason": row.get("judgment", {}).get("reason"),
            "confidence": row.get("judgment", {}).get("confidence"),
        }
        for row in judgments
    ]
    base_prompt = build_hacker_prompt(candidate)
    if round_status == "too_obvious":
        diagnosis = "上一轮攻击已经被 baseline 和 Lean-assisted 同时识别，说明伪证太明显。"
    elif round_status == "lean_missed":
        diagnosis = "上一轮攻击骗过了 baseline，但 Lean-assisted 也没有识别；这不是目标样本。请让错误更适合被局部形式化证据暴露。"
    elif round_status == "wrapped_or_lean_rescue":
        diagnosis = "上一轮 wrapped-only 已经识别了错误，说明结构化包装本身就足够；请让错误更依赖 Lean 证据才能暴露。"
    elif round_status == "lean_weaker_than_baseline":
        diagnosis = "上一轮 baseline 识别了错误，但 Lean-assisted 没有识别；请避免让错误只靠自然语言检查明显可见。"
    else:
        diagnosis = "上一轮攻击不是目标样本。"
    return (
        f"{diagnosis}"
        "请在同一个原始目标步骤上重新生成一个更隐蔽的新 flawed_step。\n\n"
        f"上一轮轮次：{round_idx}\n"
        f"上一轮 flawed_step：\n{previous_attack.get('flawed_step', '')}\n\n"
        f"上一轮评测器反馈：\n{json.dumps(compact_judgments, ensure_ascii=False, indent=2)}\n\n"
        "改进要求：\n"
        "- 不要重复上一轮的错误形式。\n"
        "- 错误仍必须集中在目标步骤本身。\n"
        "- 让错误更像正常数学证明中的隐含前提、边界遗漏或细微等价偷换。\n\n"
        f"{base_prompt}"
    )


def call_hacker(
    candidate: dict[str, Any],
    *,
    round_idx: int,
    previous_attack: dict[str, Any] | None,
    previous_judgments: list[dict[str, Any]] | None,
    previous_status: str | None,
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
    codex_thread_dir: str,
) -> tuple[dict[str, Any], str, list[str]]:
    if mock:
        parsed = mock_attack(candidate, round_idx)
        return parsed, json.dumps(parsed, ensure_ascii=False, indent=2), []

    if previous_attack and previous_judgments:
        actual_system_prompt = load_prompt("adversarial_hacker_retry.md")
        user_prompt = build_hacker_feedback_prompt(
            candidate,
            previous_attack,
            previous_judgments,
            round_idx - 1,
            previous_status or "other",
        )
    else:
        actual_system_prompt = load_prompt("adversarial_hacker_init.md")
        user_prompt = build_hacker_prompt(candidate)

    try:
        raw = call_llm(
            system_prompt=actual_system_prompt,
            user_prompt=user_prompt,
            provider=provider,
            model=model,
            temperature=0.7,
            max_tokens=max_tokens,
            timeout=llm_timeout,
            retries=2,
            reasoning=reasoning,
            openai_reasoning_effort=openai_reasoning_effort,
            codex_reasoning_effort=codex_reasoning_effort,
            codex_sandbox=codex_sandbox,
            codex_cwd=codex_cwd,
            codex_thread_file=candidate_thread_file(candidate, codex_thread_dir),
            call_label=f"{candidate['id']}-c{candidate['chain_id']}-s{candidate['step_id']}-hack-r{round_idx}",
        )
        parsed = extract_json_object(raw)
        errors = validate_attack(parsed)
        return parsed, raw, errors
    except (LLMCallError, Exception) as exc:
        return {"attackable": False}, str(exc), [str(exc)]


def run_audit(
    row: dict[str, Any],
    *,
    round_dir: Path,
    provider: str,
    model: str | None,
    mock: bool,
    llm_timeout: int,
    wrap_max_tokens: int,
    lean_max_tokens: int,
    wrap_repair_rounds: int,
    project_dir: Path,
    lean_timeout: int,
    repair_rounds: int,
    skip_lean_check: bool,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
    codex_reasoning_effort: str,
    codex_sandbox: str,
    codex_cwd: str,
) -> dict[str, Any]:
    selection_dir = round_dir / "selection"
    wrapped_dir = round_dir / "wrapped_claims"
    lean_dir = round_dir / "lean"
    verification_dir = round_dir / "verification"
    write_jsonl([row], selection_dir / "steps_selected.jsonl")

    wrapped = generate_wrapped_claims(
        [row],
        provider=provider,
        model=model,
        mock=mock,
        out_dir=wrapped_dir,
        llm_timeout=llm_timeout,
        wrap_max_tokens=wrap_max_tokens,
        wrap_repair_rounds=wrap_repair_rounds,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
    )
    write_jsonl(wrapped, wrapped_dir / "wrapped_claims.jsonl")

    generated = generate_lean_contracts(
        wrapped,
        provider=provider,
        model=model,
        mock=mock,
        out_dir=lean_dir,
        llm_timeout=llm_timeout,
        lean_max_tokens=lean_max_tokens,
        project_dir=project_dir,
        lean_timeout=lean_timeout,
        repair_rounds=repair_rounds,
        skip_lean_check=skip_lean_check,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
    )
    write_jsonl(generated, lean_dir / "lean_generation_manifest.jsonl")

    verification = verify_lean_outputs(
        generated,
        project_dir=project_dir,
        timeout=lean_timeout,
        skip=skip_lean_check,
    )
    write_json(verification, verification_dir / "verification.json")
    summary = {
        "wrapped_claims": len(wrapped),
        "wrap_valid": sum(1 for item in wrapped if item.get("wrap_valid") is True),
        "lean_files": len(generated),
        "verified_ok": sum(1 for item in verification if item.get("ok") is True),
        "verified_failed": sum(1 for item in verification if item.get("ok") is False),
        "verified_skipped": sum(1 for item in verification if item.get("ok") is None),
        "complete_proofs": sum(
            1
            for item in verification
            if item.get("ok") is True and item.get("dependency_mode") == "complete"
        ),
        "local_missing_hypotheses": sum(
            1
            for item in verification
            if item.get("ok") is True and item.get("dependency_mode") == "local_missing_hypotheses"
        ),
        "global_axiom_fallbacks": sum(
            1
            for item in verification
            if item.get("ok") is True and item.get("dependency_mode") == "global_axiom_fallback"
        ),
    }
    write_json(summary, round_dir / "audit_summary.json")
    return summary


def run_judges(
    row: dict[str, Any],
    *,
    round_dir: Path,
    modes: list[str],
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
) -> list[dict[str, Any]]:
    evidence = load_audit_evidence(round_dir)
    system_prompt = load_judge_prompt("step_judge.md")
    results = []
    response_dir = round_dir / "judge" / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        result = judge_one(
            (row, mode),
            system_prompt=system_prompt,
            evidence_by_key=evidence,
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
        )
        response_path = response_dir / f"{safe_id(f'{row['id']}_{mode}')}.response.txt"
        response_path.write_text(result.pop("raw_response", ""), encoding="utf-8")
        result["response_file"] = str(response_path)
        results.append(result)
    write_jsonl(results, round_dir / "judge" / "judgments.jsonl")
    write_json(summarize(results), round_dir / "judge" / "summary.json")
    return results


def classify_round(judgments: list[dict[str, Any]]) -> str:
    by_row = {row["mode"]: row for row in judgments}
    by_mode = {mode: row.get("judgment", {}).get("verdict") for mode, row in by_row.items()}
    baseline = by_mode.get("baseline")
    lean = by_mode.get("lean_assisted")
    lean_used = bool(by_row.get("lean_assisted", {}).get("lean_used"))
    if baseline == "invalid" and lean == "invalid":
        return "too_obvious"
    if baseline != "invalid" and lean == "invalid":
        return "lean_rescue" if lean_used else "model_rescue_no_lean"
    if baseline != "invalid" and lean != "invalid":
        return "lean_missed"
    if baseline == "invalid" and lean != "invalid":
        return "lean_weaker_than_baseline"
    return "other"


def write_markdown_report(trace: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    lines = ["# Adaptive Adversarial Game Report", ""]
    lines.append("## Summary")
    lines.append("")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Rounds")
    lines.append("")
    for row in trace:
        lines.append(f"### {row['case_id']} round {row['round']}")
        lines.append("")
        lines.append(f"- status: {row['round_status']}")
        lines.append(f"- flaw_type: {row.get('attack', {}).get('flaw_type')}")
        lines.append(f"- flawed_step: {row.get('attack', {}).get('flawed_step')}")
        lines.append(f"- gold: {row.get('attack', {}).get('why_invalid')}")
        for judgment in row.get("judgments", []):
            payload = judgment.get("judgment", {})
            reason = str(payload.get("reason", "")).replace("\n", " ")
            if len(reason) > 260:
                reason = reason[:260] + "..."
            lines.append(
                f"- {judgment['mode']}: {payload.get('verdict')} / {payload.get('issue_type')} "
                f"(confidence={payload.get('confidence')}) - {reason}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=existing_default_config())
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--target-lean-rescues", type=int, default=0, help="stop the whole run after this many lean_rescue rounds; 0 disables")
    parser.add_argument("--continue-after-non-obvious", action="store_true", help="keep adapting even after lean_missed/wrapped_or_lean_rescue/lean_weaker statuses")
    parser.add_argument("--llm-provider", choices=["openai", "codex"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--llm-timeout", type=int, default=None)
    parser.add_argument("--hacker-max-tokens", type=int, default=8192)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--wrap-max-tokens", type=int, default=None)
    parser.add_argument("--lean-max-tokens", type=int, default=None)
    parser.add_argument("--wrap-repair-rounds", type=int, default=None)
    parser.add_argument("--repair-rounds", type=int, default=None)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--lean-timeout", type=int, default=None)
    parser.add_argument("--skip-lean-check", action="store_true")
    parser.add_argument("--reasoning", choices=["auto", "enabled", "disabled"], default=None)
    parser.add_argument("--openai-reasoning-effort", choices=["high", "max"], default=None)
    parser.add_argument("--codex-reasoning-effort", default=None)
    parser.add_argument("--codex-sandbox", default=None)
    parser.add_argument("--codex-cwd", default=None)
    args = parser.parse_args()

    config = read_config(args.config)
    provider = args.llm_provider or os.environ.get("LLM_PROVIDER") or cfg_get(config, "llm.provider", "codex")
    model = args.model or cfg_get(config, "llm.model", None)
    llm_timeout = args.llm_timeout if args.llm_timeout is not None else int(cfg_get(config, "llm.timeout", 900))
    wrap_max_tokens = args.wrap_max_tokens if args.wrap_max_tokens is not None else int(cfg_get(config, "llm.wrap_max_tokens", 4096))
    lean_max_tokens = args.lean_max_tokens if args.lean_max_tokens is not None else int(cfg_get(config, "llm.lean_max_tokens", 4096))
    wrap_repair_rounds = (
        args.wrap_repair_rounds
        if args.wrap_repair_rounds is not None
        else int(cfg_get(config, "run.wrap_repair_rounds", 2))
    )
    repair_rounds = args.repair_rounds if args.repair_rounds is not None else int(cfg_get(config, "lean.repair_rounds", 3))
    project_dir = Path(args.project_dir or cfg_get(config, "paths.lean_project_dir", default_lean_project_dir()))
    lean_timeout = args.lean_timeout if args.lean_timeout is not None else int(cfg_get(config, "lean.timeout", 120))
    skip_lean_check = args.skip_lean_check or bool(cfg_get(config, "lean.skip_check", False))
    reasoning = parse_reasoning(args.reasoning if args.reasoning is not None else cfg_get(config, "llm.reasoning", None))
    openai_reasoning_effort = args.openai_reasoning_effort or cfg_get(config, "llm.openai_reasoning_effort", None)
    codex_reasoning_effort = (
        args.codex_reasoning_effort
        or os.environ.get("CODEX_REASONING_EFFORT")
        or cfg_get(config, "llm.codex_reasoning_effort", "high")
    )
    codex_sandbox = args.codex_sandbox or os.environ.get("CODEX_SANDBOX") or cfg_get(config, "llm.codex_sandbox", "read-only")
    codex_cwd = args.codex_cwd or cfg_get(config, "llm.codex_cwd", str(REPO_ROOT))

    run_dir = RUNS_DIR / args.run_id
    input_dir = run_dir / "input"
    thread_dir = run_dir / "hacker_threads"
    rounds_dir = run_dir / "rounds"
    run_dir.mkdir(parents=True, exist_ok=True)
    thread_dir.mkdir(parents=True, exist_ok=True)
    rounds_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_jsonl(Path(args.candidates))
    random.Random(args.seed).shuffle(candidates)
    candidates = candidates[: args.limit]
    write_jsonl(candidates, input_dir / "candidates.jsonl")
    write_json(
        {
            "run_id": args.run_id,
            "candidates": args.candidates,
            "limit": args.limit,
            "max_rounds": args.max_rounds,
            "provider": "mock" if args.mock else provider,
            "model": "mock" if args.mock else model,
            "llm_timeout": llm_timeout,
            "project_dir": str(project_dir),
            "lean_timeout": lean_timeout,
            "skip_lean_check": skip_lean_check,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        run_dir / "run_config.json",
    )

    hacker_prompt = load_prompt("adversarial_hacker_init.md")
    trace: list[dict[str, Any]] = []
    for case_idx, candidate in enumerate(candidates, start=1):
        case_id = safe_id(f"{case_idx}_{candidate['id']}_c{candidate['chain_id']}_s{candidate['step_id']}")
        previous_attack: dict[str, Any] | None = None
        previous_judgments: list[dict[str, Any]] | None = None
        previous_status: str | None = None
        for round_idx in range(1, args.max_rounds + 1):
            round_dir = rounds_dir / case_id / f"round_{round_idx}"
            attack, raw_attack, attack_errors = call_hacker(
                candidate,
                round_idx=round_idx,
                previous_attack=previous_attack,
                previous_judgments=previous_judgments,
                previous_status=previous_status,
                system_prompt=hacker_prompt,
                provider=provider,
                model=model,
                mock=args.mock,
                llm_timeout=llm_timeout,
                max_tokens=args.hacker_max_tokens,
                reasoning=reasoning,
                openai_reasoning_effort=openai_reasoning_effort,
                codex_reasoning_effort=codex_reasoning_effort,
                codex_sandbox=codex_sandbox,
                codex_cwd=codex_cwd,
                codex_thread_dir=str(thread_dir),
            )
            round_dir.mkdir(parents=True, exist_ok=True)
            (round_dir / "hacker_response.txt").write_text(raw_attack, encoding="utf-8")
            write_json({"attack": attack, "errors": attack_errors}, round_dir / "attack.json")
            if attack_errors or not attack.get("attackable"):
                trace.append(
                    {
                        "case_id": case_id,
                        "round": round_idx,
                        "round_status": "hacker_failed",
                        "attack": attack,
                        "attack_errors": attack_errors,
                        "judgments": [],
                    }
                )
                break

            row = build_invalid_row(candidate, attack, round_idx)
            write_jsonl([row], round_dir / "adversarial_step.jsonl")
            baseline_result = run_initial_judge(
                row,
                label="baseline",
                round_dir=round_dir,
                provider=provider,
                model=model,
                mock=args.mock,
                llm_timeout=llm_timeout,
                max_tokens=args.judge_max_tokens,
                reasoning=reasoning,
                openai_reasoning_effort=openai_reasoning_effort,
                codex_reasoning_effort=codex_reasoning_effort,
                codex_sandbox=codex_sandbox,
                codex_cwd=codex_cwd,
            )
            lean_initial = run_initial_judge(
                row,
                label="lean_initial",
                round_dir=round_dir,
                provider=provider,
                model=model,
                mock=args.mock,
                llm_timeout=llm_timeout,
                max_tokens=args.judge_max_tokens,
                reasoning=reasoning,
                openai_reasoning_effort=openai_reasoning_effort,
                codex_reasoning_effort=codex_reasoning_effort,
                codex_sandbox=codex_sandbox,
                codex_cwd=codex_cwd,
            )
            initial_payload = lean_initial.get("judgment", {})
            if (
                initial_payload.get("verdict") == "invalid"
                or initial_payload.get("should_try_lean") is False
            ):
                lean_result = {
                    "mode": "lean_assisted",
                    "stage": "initial_only",
                    "judgment": {
                        "verdict": initial_payload.get("verdict", "uncertain"),
                        "issue_type": initial_payload.get("issue_type", "other"),
                        "reason": initial_payload.get("reason", ""),
                        "lean_evidence": "Lean 未运行：初始判断已返回 invalid 或 should_try_lean=false。",
                        "confidence": initial_payload.get("confidence", 1),
                    },
                    "errors": lean_initial.get("errors", []),
                    "lean_used": False,
                }
                write_json(lean_result, round_dir / "lean_assisted_result.json")
            else:
                lean_result = run_direct_lean_assist(
                    row,
                    initial_payload,
                    round_dir=round_dir,
                    provider=provider,
                    model=model,
                    mock=args.mock,
                    llm_timeout=llm_timeout,
                    lean_max_tokens=lean_max_tokens,
                    judge_max_tokens=args.judge_max_tokens,
                    project_dir=project_dir,
                    lean_timeout=lean_timeout,
                    repair_rounds=repair_rounds,
                    reasoning=reasoning,
                    openai_reasoning_effort=openai_reasoning_effort,
                    codex_reasoning_effort=codex_reasoning_effort,
                    codex_sandbox=codex_sandbox,
                    codex_cwd=codex_cwd,
                )
            judgments = [baseline_result, lean_initial, lean_result]
            write_jsonl(judgments, round_dir / "judgments.jsonl")
            round_status = classify_round(judgments)
            trace_row = {
                "case_id": case_id,
                "round": round_idx,
                "round_dir": str(round_dir),
                "round_status": round_status,
                "attack": attack,
                "judgments": judgments,
            }
            trace.append(trace_row)
            previous_attack = attack
            previous_judgments = judgments
            previous_status = round_status
            if args.target_lean_rescues and sum(
                1 for row in trace if row.get("round_status") == "lean_rescue"
            ) >= args.target_lean_rescues:
                break
            if round_status == "lean_rescue":
                break
            if not args.continue_after_non_obvious and round_status != "too_obvious":
                break
        if args.target_lean_rescues and sum(
            1 for row in trace if row.get("round_status") == "lean_rescue"
        ) >= args.target_lean_rescues:
            break

    write_jsonl(trace, run_dir / "game_trace.jsonl")
    summary = {
        "cases": len(candidates),
        "rounds": len(trace),
        "lean_rescue": sum(1 for row in trace if row.get("round_status") == "lean_rescue"),
        "model_rescue_no_lean": sum(1 for row in trace if row.get("round_status") == "model_rescue_no_lean"),
        "wrapped_or_lean_rescue": sum(1 for row in trace if row.get("round_status") == "wrapped_or_lean_rescue"),
        "too_obvious_rounds": sum(1 for row in trace if row.get("round_status") == "too_obvious"),
        "lean_missed": sum(1 for row in trace if row.get("round_status") == "lean_missed"),
        "lean_weaker_than_baseline": sum(1 for row in trace if row.get("round_status") == "lean_weaker_than_baseline"),
        "hacker_failed": sum(1 for row in trace if row.get("round_status") == "hacker_failed"),
        "target_lean_rescues": args.target_lean_rescues,
        "output": str(run_dir),
    }
    write_json(summary, run_dir / "summary.json")
    write_markdown_report(trace, summary, run_dir / "report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
