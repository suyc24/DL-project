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
ADAPTIVE_PROMPTS_DIR = PROMPTS_DIR / "adaptive_adversarial"
FORMAT_REPAIR_ROUNDS = 2
LEAN_BLOCK_RE = re.compile(r"```\s*(?:lean|lean4)?\s*\n(.*?)```", re.I | re.S)
CHECK_GLOBAL_PROMPT = "adversarial_check_global.md"
INITIAL_JUDGE_PROMPT = "adversarial_judge_initial.md"
CHECK_PROMPT_NAMES = {
    INITIAL_JUDGE_PROMPT,
    "adversarial_lean_formalize.md",
    "adversarial_lean_pass_review.md",
    "adversarial_lean_fail_decide.md",
    "adversarial_lean_repair.md",
    "format_repair_lean.md",
}

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


def load_adaptive_prompt(name: str) -> str:
    prompt = (ADAPTIVE_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()
    if name in CHECK_PROMPT_NAMES:
        global_prompt = (ADAPTIVE_PROMPTS_DIR / CHECK_GLOBAL_PROMPT).read_text(encoding="utf-8").strip()
        return f"{global_prompt}\n\n{prompt}"
    return prompt


def call_model(
    *,
    system_prompt: str,
    user_prompt: str,
    provider: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
    codex_reasoning_effort: str,
    codex_sandbox: str,
    codex_cwd: str,
    call_label: str,
    codex_thread_file: str | None = None,
) -> str:
    return call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
        codex_thread_file=codex_thread_file,
        call_label=call_label,
    )


def build_json_format_repair_prompt(original_user_prompt: str, raw: str, errors: list[str]) -> str:
    return (
        "原始任务：\n"
        f"{original_user_prompt}\n\n"
        "上一轮输出：\n"
        f"{raw}\n\n"
        "格式/结构错误：\n"
        f"{json.dumps(errors, ensure_ascii=False, indent=2)}\n\n"
        "请只返回修复后的 JSON 对象。"
    )


def build_lean_format_repair_prompt(original_user_prompt: str, raw: str, errors: list[str], theorem_name: str) -> str:
    return (
        f"指定 theorem 名字：{theorem_name}\n\n"
        "原始任务：\n"
        f"{original_user_prompt}\n\n"
        "上一轮输出：\n"
        f"{raw}\n\n"
        "格式错误：\n"
        f"{json.dumps(errors, ensure_ascii=False, indent=2)}\n\n"
        "请只返回修复后的结果：如果原任务应继续形式化，返回 Lean 代码块；如果已经发现目标步骤错误，返回 invalid JSON。"
    )


def extract_invalid_judgment(text: str) -> dict[str, Any] | None:
    try:
        payload = extract_json_object(text)
    except Exception:
        return None
    if payload.get("verdict") != "invalid":
        return None
    if not isinstance(payload.get("reason"), str) or not payload.get("reason", "").strip():
        return None
    try:
        confidence = int(payload.get("confidence", 3))
    except Exception:
        confidence = 3
    return {
        "verdict": "invalid",
        "reason": payload["reason"].strip(),
        "lean_evidence": str(payload.get("lean_evidence", "模型在 Lean 阶段直接发现目标步骤不可靠。")),
        "confidence": confidence,
    }


def extract_required_lean_code(text: str, theorem_name: str) -> str:
    if not LEAN_BLOCK_RE.search(text):
        raise ValueError("missing Lean code block")
    code = extract_lean_code(text)
    if theorem_name not in code:
        raise ValueError(f"Lean code must contain theorem name {theorem_name}")
    if "sorry" in code or "admit" in code:
        raise ValueError("Lean code must not contain sorry or admit")
    return code


def lean_code_call(
    *,
    system_prompt: str,
    user_prompt: str,
    theorem_name: str,
    provider: str,
    model: str | None,
    mock: bool,
    mock_code: str,
    llm_timeout: int,
    max_tokens: int,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
    codex_reasoning_effort: str,
    codex_sandbox: str,
    codex_cwd: str,
    call_label: str,
    codex_thread_file: str | None = None,
) -> tuple[str, str, list[str], dict[str, Any] | None]:
    if mock:
        raw = f"```lean\n{mock_code}```"
        return mock_code, raw, [], None
    try:
        raw = call_model(
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
            codex_thread_file=codex_thread_file,
            call_label=call_label,
        )
        current_raw = raw
        current_errors: list[str] = []
        for repair_idx in range(FORMAT_REPAIR_ROUNDS + 1):
            invalid_judgment = extract_invalid_judgment(current_raw)
            if invalid_judgment is not None:
                return "", current_raw, [], invalid_judgment
            try:
                code = extract_required_lean_code(current_raw, theorem_name)
                return code, current_raw, [], None
            except Exception as exc:
                current_errors = [str(exc)]
            if repair_idx >= FORMAT_REPAIR_ROUNDS:
                break
            current_raw = call_model(
                system_prompt=load_adaptive_prompt("format_repair_lean.md"),
                user_prompt=build_lean_format_repair_prompt(user_prompt, current_raw, current_errors, theorem_name),
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
                codex_thread_file=codex_thread_file,
                call_label=f"{call_label}-format-repair-{repair_idx + 1}",
            )
        invalid_judgment = extract_invalid_judgment(current_raw)
        if invalid_judgment is not None:
            return "", current_raw, [], invalid_judgment
        return extract_lean_code(current_raw), current_raw, current_errors, None
    except (LLMCallError, Exception) as exc:
        code = f"import Mathlib\n\n-- Lean 生成调用失败：{exc}\ntheorem {theorem_name} : False := by\n  exact False.elim ?missing\n"
        return code, str(exc), [str(exc)], None


def validate_judge_json(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("verdict") not in {"valid", "invalid"}:
        errors.append("verdict must be one of valid, invalid")
    if not isinstance(payload.get("reason"), str) or not payload.get("reason", "").strip():
        errors.append("reason must be non-empty string")
    if "confidence" not in payload:
        errors.append("confidence is required")
    return errors


def validate_fail_decide_json(payload: dict[str, Any]) -> list[str]:
    errors = validate_judge_json(payload)
    if payload.get("action") not in {"return_invalid", "continue_repair"}:
        errors.append("action must be one of return_invalid, continue_repair")
    if payload.get("action") == "continue_repair" and not str(payload.get("repair_instruction", "")).strip():
        errors.append("repair_instruction must be non-empty when action is continue_repair")
    return errors


def validate_attack_for_candidate(candidate: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    errors = validate_attack(payload)
    if payload.get("attackable") is False:
        return errors
    if not isinstance(payload.get("modified_cot"), str) or not payload.get("modified_cot", "").strip():
        errors.append("modified_cot must be non-empty string")
    return errors


def parse_json_with_format_repair(
    *,
    raw: str,
    validator: Any,
    original_user_prompt: str,
    provider: str,
    model: str | None,
    llm_timeout: int,
    max_tokens: int,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
    codex_reasoning_effort: str,
    codex_sandbox: str,
    codex_cwd: str,
    call_label: str,
    codex_thread_file: str | None = None,
) -> tuple[dict[str, Any], str, list[str]]:
    current_raw = raw
    current_errors: list[str] = []
    for repair_idx in range(FORMAT_REPAIR_ROUNDS + 1):
        try:
            parsed = extract_json_object(current_raw)
            current_errors = validator(parsed) if validator else []
            if not current_errors:
                return parsed, current_raw, []
        except Exception as exc:
            current_errors = [str(exc)]
        if repair_idx >= FORMAT_REPAIR_ROUNDS:
            break
        current_raw = call_model(
            system_prompt=load_adaptive_prompt("format_repair_json.md"),
            user_prompt=build_json_format_repair_prompt(original_user_prompt, current_raw, current_errors),
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
            codex_thread_file=codex_thread_file,
            call_label=f"{call_label}-format-repair-{repair_idx + 1}",
        )
    return {}, current_raw, current_errors


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
    validator: Any = validate_judge_json,
    codex_thread_file: str | None = None,
) -> tuple[dict[str, Any], str, list[str]]:
    if mock:
        payload = mock_payload or {}
        return payload, json.dumps(payload, ensure_ascii=False, indent=2), []
    try:
        raw = call_model(
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
            codex_thread_file=codex_thread_file,
            call_label=call_label,
        )
        parsed, final_raw, format_errors = parse_json_with_format_repair(
            raw=raw,
            validator=validator,
            original_user_prompt=user_prompt,
            provider=provider,
            model=model,
            llm_timeout=llm_timeout,
            max_tokens=max_tokens,
            reasoning=reasoning,
            openai_reasoning_effort=openai_reasoning_effort,
            codex_reasoning_effort=codex_reasoning_effort,
            codex_sandbox=codex_sandbox,
            codex_cwd=codex_cwd,
            codex_thread_file=codex_thread_file,
            call_label=call_label,
        )
        if not format_errors:
            return parsed, final_raw, []
        return {"verdict": "invalid", "reason": "; ".join(format_errors), "confidence": 1}, final_raw, format_errors
    except (LLMCallError, Exception) as exc:
        return {"verdict": "invalid", "reason": str(exc), "confidence": 1}, str(exc), [str(exc)]


def build_context_text(row: dict[str, Any]) -> str:
    if row.get("adversarial") and row.get("mutated_cot"):
        return str(row["mutated_cot"])
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
    codex_thread_file: str | None = None,
) -> dict[str, Any]:
    mock_payload = {
        "verdict": "valid" if not row.get("adversarial") else "invalid",
        "reason": f"mock {label} initial judge",
        "confidence": 3,
    }
    parsed, raw, errors = json_call(
        system_prompt=load_adaptive_prompt(INITIAL_JUDGE_PROMPT),
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
        codex_thread_file=codex_thread_file,
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
        f"初步判断：\n{json.dumps(initial, ensure_ascii=False, indent=2)}\n\n"
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
    codex_thread_file: str | None = None,
) -> dict[str, Any]:
    out_dir = round_dir / "lean_direct"
    out_dir.mkdir(parents=True, exist_ok=True)
    theorem_name = safe_lean_name(row)
    system_formalize = load_adaptive_prompt("adversarial_lean_formalize.md")
    mock_code = f"import Mathlib\n\ntheorem {theorem_name} : True := by\n  trivial\n"
    code, raw, lean_format_errors, formalize_invalid = lean_code_call(
        system_prompt=system_formalize,
        user_prompt=build_lean_formalize_prompt(row, initial, theorem_name),
        theorem_name=theorem_name,
        provider=provider,
        model=model,
        mock=mock,
        mock_code=mock_code,
        llm_timeout=llm_timeout,
        max_tokens=lean_max_tokens,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
        call_label=f"{row['id']}-direct-lean",
        codex_thread_file=codex_thread_file,
    )
    (out_dir / "lean.response.txt").write_text(raw, encoding="utf-8")

    decisions: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    format_errors: list[dict[str, Any]] = []
    if lean_format_errors:
        format_errors.append({"stage": "lean_formalize", "errors": lean_format_errors})
    if formalize_invalid is not None:
        result = {
            "mode": "lean_assisted",
            "stage": "formalize_return_invalid",
            "judgment": formalize_invalid,
            "errors": lean_format_errors,
            "lean_used": False,
            "lean_checks": [],
            "fail_decisions": [],
            "format_errors": format_errors,
        }
        write_json(result, out_dir / "lean_assisted_result.json")
        return result
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
                "reason": "mock lean pass review",
                "lean_evidence": "mock",
                "confidence": 3,
            }
            final_judgment, final_raw, final_errors = json_call(
                system_prompt=load_adaptive_prompt("adversarial_lean_pass_review.md"),
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
                codex_thread_file=codex_thread_file,
            )
            break

        mock_payload = {
            "action": "return_invalid",
            "verdict": "invalid",
            "reason": "mock lean fail decision",
            "repair_instruction": "",
            "confidence": 3,
        }
        decision, raw_decision, errors = json_call(
            system_prompt=load_adaptive_prompt("adversarial_lean_fail_decide.md"),
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
            validator=validate_fail_decide_json,
            codex_thread_file=codex_thread_file,
        )
        decision_record = {"attempt": attempt, "decision": decision, "raw": raw_decision, "errors": errors}
        decisions.append(decision_record)
        if decision.get("action") != "continue_repair" or attempt >= repair_rounds:
            final_judgment = {
                "verdict": decision.get("verdict", "invalid"),
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
            repair_prompt = build_lean_repair_prompt(code, check, decision)
            code, repair_raw, repair_format_errors, repair_invalid = lean_code_call(
                system_prompt=load_adaptive_prompt("adversarial_lean_repair.md"),
                user_prompt=repair_prompt,
                theorem_name=theorem_name,
                provider=provider,
                model=model,
                mock=mock,
                mock_code=code,
                llm_timeout=llm_timeout,
                max_tokens=lean_max_tokens,
                reasoning=reasoning,
                openai_reasoning_effort=openai_reasoning_effort,
                codex_reasoning_effort=codex_reasoning_effort,
                codex_sandbox=codex_sandbox,
                codex_cwd=codex_cwd,
                call_label=f"{row['id']}-direct-lean-repair-{attempt + 1}",
                codex_thread_file=codex_thread_file,
            )
            (out_dir / f"repair_{attempt + 1}.response.txt").write_text(repair_raw, encoding="utf-8")
            if repair_format_errors:
                format_errors.append({"stage": f"lean_repair_{attempt + 1}", "errors": repair_format_errors})
            if repair_invalid is not None:
                final_judgment = repair_invalid
                final_raw = repair_raw
                final_errors = repair_format_errors
                break

    (out_dir / "final_review.response.txt").write_text(final_raw, encoding="utf-8")
    result = {
        "mode": "lean_assisted",
        "stage": "lean_review",
        "judgment": final_judgment or {"verdict": "invalid", "reason": "no final judgment", "confidence": 1},
        "errors": final_errors,
        "lean_used": True,
        "lean_checks": checks,
        "fail_decisions": decisions,
        "format_errors": format_errors,
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
        "请基于同一个题目和目标步骤序号，重新生成一个更隐蔽的新攻击。\n\n"
        f"上一轮轮次：{round_idx}\n"
        f"上一轮 flawed_step 编号：\n{previous_attack.get('flawed_step', '')}\n\n"
        f"上一轮 modified_cot：\n{previous_attack.get('modified_cot', '')}\n\n"
        f"上一轮错误理由：\n{previous_attack.get('why_invalid', '')}\n\n"
        f"上一轮评测器反馈：\n{json.dumps(compact_judgments, ensure_ascii=False, indent=2)}\n\n"
        "改进要求：\n"
        "- 不要重复上一轮的错误形式。\n"
        "- CoT 只是参考，可以修改任意数量的 CoT 步骤，但要尽量少改。\n"
        "- flawed_step 必须是 modified_cot 中故意写错的 Step 编号。\n"
        "- 先模拟强文本 verifier 会如何逐字抓错；如果会被一两句话直接抓住，换方向。\n"
        "- 避免明显边界遗漏、必要/充分偷换、直接反例、简单算术错误、量词明面反转。\n"
        "- 禁止只改常数、系数、符号或模数作为主要攻击。\n"
        "- 优先制造更适合形式化暴露的错误：统一 witness/参数独立性/量词顺序/定义域或非零条件/分支覆盖/局部到全局接口缺失。\n\n"
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
        actual_system_prompt = load_adaptive_prompt("adversarial_hacker_retry.md")
        user_prompt = build_hacker_feedback_prompt(
            candidate,
            previous_attack,
            previous_judgments,
            round_idx - 1,
            previous_status or "other",
        )
    else:
        actual_system_prompt = load_adaptive_prompt("adversarial_hacker_init.md")
        user_prompt = build_hacker_prompt(candidate)

    thread_file = candidate_thread_file(candidate, codex_thread_dir)
    try:
        raw = call_model(
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
            codex_thread_file=thread_file,
            call_label=f"{candidate['id']}-c{candidate['chain_id']}-s{candidate['step_id']}-hack-r{round_idx}",
        )
        parsed, final_raw, errors = parse_json_with_format_repair(
            raw=raw,
            validator=lambda payload: validate_attack_for_candidate(candidate, payload),
            original_user_prompt=user_prompt,
            provider=provider,
            model=model,
            llm_timeout=llm_timeout,
            max_tokens=max_tokens,
            reasoning=reasoning,
            openai_reasoning_effort=openai_reasoning_effort,
            codex_reasoning_effort=codex_reasoning_effort,
            codex_sandbox=codex_sandbox,
            codex_cwd=codex_cwd,
            codex_thread_file=thread_file,
            call_label=f"{candidate['id']}-c{candidate['chain_id']}-s{candidate['step_id']}-hack-r{round_idx}",
        )
        if errors:
            return {"attackable": False}, final_raw, errors
        raw = final_raw
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
        attack = row.get("attack", {})
        lines.append("")
        lines.append("#### Hacker")
        lines.append("")
        lines.append(f"- flawed_step: {attack.get('flawed_step')}")
        if attack.get("modified_cot"):
            lines.append("- modified_cot:")
            lines.append("")
            lines.append("```text")
            lines.append(str(attack.get("modified_cot")))
            lines.append("```")
        lines.append(f"- why_invalid: {attack.get('why_invalid')}")
        lines.append("")
        lines.append("#### Judges")
        lines.append("")
        for judgment in row.get("judgments", []):
            if judgment.get("mode") == "lean_initial":
                continue
            payload = judgment.get("judgment", {})
            reason = str(payload.get("reason", "")).strip()
            lean_evidence = str(payload.get("lean_evidence", "")).strip()
            lines.append(
                f"- {judgment['mode']}: {payload.get('verdict')} (confidence={payload.get('confidence')})"
            )
            lines.append(f"  - reason: {reason}")
            if lean_evidence:
                lines.append(f"  - lean_evidence: {lean_evidence}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(trace: list[dict[str, Any]], *, cases: int, target_lean_rescues: int, run_dir: Path) -> dict[str, Any]:
    return {
        "cases": cases,
        "rounds": len(trace),
        "lean_rescue": sum(1 for row in trace if row.get("round_status") == "lean_rescue"),
        "model_rescue_no_lean": sum(1 for row in trace if row.get("round_status") == "model_rescue_no_lean"),
        "wrapped_or_lean_rescue": sum(1 for row in trace if row.get("round_status") == "wrapped_or_lean_rescue"),
        "too_obvious_rounds": sum(1 for row in trace if row.get("round_status") == "too_obvious"),
        "lean_missed": sum(1 for row in trace if row.get("round_status") == "lean_missed"),
        "lean_weaker_than_baseline": sum(1 for row in trace if row.get("round_status") == "lean_weaker_than_baseline"),
        "hacker_failed": sum(1 for row in trace if row.get("round_status") == "hacker_failed"),
        "target_lean_rescues": target_lean_rescues,
        "output": str(run_dir),
    }


def write_run_progress(trace: list[dict[str, Any]], *, cases: int, target_lean_rescues: int, run_dir: Path) -> dict[str, Any]:
    write_jsonl(trace, run_dir / "game_trace.jsonl")
    summary = build_summary(trace, cases=cases, target_lean_rescues=target_lean_rescues, run_dir=run_dir)
    write_json(summary, run_dir / "summary.json")
    write_markdown_report(trace, summary, run_dir / "report.md")
    return summary


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    parser.add_argument("--resume", action="store_true", help="continue an existing run-id from completed round files")
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
    codex_sandbox = args.codex_sandbox or os.environ.get("CODEX_SANDBOX") or cfg_get(config, "llm.codex_sandbox", "danger-full-access")
    codex_cwd = args.codex_cwd or cfg_get(config, "llm.codex_cwd", str(REPO_ROOT))

    run_dir = RUNS_DIR / args.run_id
    input_dir = run_dir / "input"
    thread_dir = run_dir / "hacker_threads"
    lean_thread_dir = run_dir / "lean_threads"
    rounds_dir = run_dir / "rounds"
    run_dir.mkdir(parents=True, exist_ok=True)
    thread_dir.mkdir(parents=True, exist_ok=True)
    lean_thread_dir.mkdir(parents=True, exist_ok=True)
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
            "hacker_thread_dir": str(thread_dir),
            "lean_thread_dir": str(lean_thread_dir),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        run_dir / "run_config.json",
    )

    hacker_prompt = load_adaptive_prompt("adversarial_hacker_init.md")
    trace: list[dict[str, Any]] = []
    for case_idx, candidate in enumerate(candidates, start=1):
        case_id = safe_id(f"{case_idx}_{candidate['id']}_c{candidate['chain_id']}_s{candidate['step_id']}")
        previous_attack: dict[str, Any] | None = None
        previous_judgments: list[dict[str, Any]] | None = None
        previous_status: str | None = None
        start_round = 1
        pending_attack: tuple[dict[str, Any], str, list[str]] | None = None
        case_done = False
        if args.resume:
            for resume_round in range(1, args.max_rounds + 1):
                resume_dir = rounds_dir / case_id / f"round_{resume_round}"
                attack_path = resume_dir / "attack.json"
                judgments_path = resume_dir / "judgments.jsonl"
                if judgments_path.exists():
                    attack_record = load_json(attack_path) if attack_path.exists() else {"attack": {}, "errors": []}
                    judgments = read_jsonl(judgments_path)
                    round_status = classify_round(judgments)
                    attack = attack_record.get("attack", {})
                    trace.append(
                        {
                            "case_id": case_id,
                            "round": resume_round,
                            "round_dir": str(resume_dir),
                            "round_status": round_status,
                            "attack": attack,
                            "judgments": judgments,
                            "resumed": True,
                        }
                    )
                    previous_attack = attack
                    previous_judgments = judgments
                    previous_status = round_status
                    start_round = resume_round + 1
                    if args.target_lean_rescues and sum(
                        1 for row in trace if row.get("round_status") == "lean_rescue"
                    ) >= args.target_lean_rescues:
                        case_done = True
                        break
                    if round_status == "lean_rescue":
                        case_done = True
                        break
                    if not args.continue_after_non_obvious and round_status != "too_obvious":
                        case_done = True
                        break
                    continue
                if attack_path.exists():
                    attack_record = load_json(attack_path)
                    raw_attack = (resume_dir / "hacker_response.txt").read_text(encoding="utf-8") if (resume_dir / "hacker_response.txt").exists() else ""
                    pending_attack = (attack_record.get("attack", {}), raw_attack, attack_record.get("errors", []))
                    start_round = resume_round
                    break
                break
        if case_done:
            if args.target_lean_rescues and sum(
                1 for row in trace if row.get("round_status") == "lean_rescue"
            ) >= args.target_lean_rescues:
                break
            continue

        for round_idx in range(start_round, args.max_rounds + 1):
            round_dir = rounds_dir / case_id / f"round_{round_idx}"
            lean_thread_file_path = lean_thread_dir / case_id / f"round_{round_idx}.thread"
            lean_thread_file_path.parent.mkdir(parents=True, exist_ok=True)
            lean_thread_file = str(lean_thread_file_path)
            if pending_attack is not None and round_idx == start_round:
                attack, raw_attack, attack_errors = pending_attack
                pending_attack = None
            else:
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
                write_run_progress(
                    trace,
                    cases=len(candidates),
                    target_lean_rescues=args.target_lean_rescues,
                    run_dir=run_dir,
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
                codex_thread_file=lean_thread_file,
            )
            initial_payload = lean_initial.get("judgment", {})
            if initial_payload.get("verdict") != "valid":
                lean_result = {
                    "mode": "lean_assisted",
                    "stage": "initial_only",
                    "judgment": {
                        "verdict": initial_payload.get("verdict", "invalid"),
                        "reason": initial_payload.get("reason", ""),
                        "lean_evidence": "Lean 未运行：初步判断没有认为目标步骤正确。",
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
                    codex_thread_file=lean_thread_file,
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
                "lean_thread_file": lean_thread_file,
            }
            trace.append(trace_row)
            write_run_progress(
                trace,
                cases=len(candidates),
                target_lean_rescues=args.target_lean_rescues,
                run_dir=run_dir,
            )
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

    summary = write_run_progress(
        trace,
        cases=len(candidates),
        target_lean_rescues=args.target_lean_rescues,
        run_dir=run_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
