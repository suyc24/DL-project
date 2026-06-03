#!/usr/bin/env python3
"""Run adaptive adversarial game with generator/verifier Lean-assisted threads."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
RUNS_DIR = ROOT / "experiments" / "runs"
PROMPT_DIR = ROOT / "prompts" / "adaptive_adversarial_gv"

sys.path.insert(0, str(SCRIPT_DIR))
import run_adversarial_game as base
from make_adversarial_steps import build_invalid_row, safe_id
from run_loop import cfg_get, parse_reasoning, read_config, read_jsonl, safe_lean_name, write_json, write_jsonl


def load_prompt(name: str) -> str:
    prompt = (PROMPT_DIR / name).read_text(encoding="utf-8").strip()
    if name in {
        "verifier_initial.md",
        "generator_formalize.md",
        "verifier_review.md",
        "generator_repair.md",
        "format_repair_lean.md",
    }:
        global_prompt = (PROMPT_DIR / "check_global.md").read_text(encoding="utf-8").strip()
        return f"{global_prompt}\n\n{prompt}"
    return prompt


def parse_json_with_repair(
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
    codex_thread_file: str | None,
    call_label: str,
) -> tuple[dict[str, Any], str, list[str]]:
    current_raw = raw
    current_errors: list[str] = []
    for repair_idx in range(base.FORMAT_REPAIR_ROUNDS + 1):
        try:
            parsed = base.extract_json_object(current_raw)
            current_errors = validator(parsed) if validator else []
            if not current_errors:
                return parsed, current_raw, []
        except Exception as exc:
            current_errors = [str(exc)]
        if repair_idx >= base.FORMAT_REPAIR_ROUNDS:
            break
        current_raw = base.call_model(
            system_prompt=load_prompt("format_repair_json.md"),
            user_prompt=base.build_json_format_repair_prompt(original_user_prompt, current_raw, current_errors),
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


def json_call_gv(
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
    codex_thread_file: str | None,
    call_label: str,
    validator: Any = base.validate_judge_json,
) -> tuple[dict[str, Any], str, list[str]]:
    if mock:
        payload = mock_payload or {}
        return payload, json.dumps(payload, ensure_ascii=False, indent=2), []
    try:
        raw = base.call_model(
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
        parsed, final_raw, errors = parse_json_with_repair(
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
        if not errors:
            return parsed, final_raw, []
        return {"verdict": "invalid", "reason": "; ".join(errors), "confidence": 1}, final_raw, errors
    except Exception as exc:
        return {"verdict": "invalid", "reason": str(exc), "confidence": 1}, str(exc), [str(exc)]


def run_baseline_verifier_judge(
    row: dict[str, Any],
    *,
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
    parsed, raw, errors = json_call_gv(
        system_prompt=load_prompt("verifier_initial.md"),
        user_prompt=base.build_initial_judge_prompt(row),
        provider=provider,
        model=model,
        mock_payload={
            "verdict": "valid" if not row.get("adversarial") else "invalid",
            "reason": "mock baseline verifier initial",
            "confidence": 3,
        },
        mock=mock,
        llm_timeout=llm_timeout,
        max_tokens=max_tokens,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
        codex_thread_file=None,
        call_label=f"{row['id']}-baseline-verifier-initial",
    )
    out_dir = round_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "initial_judge.response.txt").write_text(raw, encoding="utf-8")
    write_json({"judgment": parsed, "errors": errors}, out_dir / "initial_judge.json")
    return {"mode": "baseline", "stage": "initial", "judgment": parsed, "errors": errors}


def validate_verifier_decision(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("action") not in {"return_valid", "return_invalid", "request_repair"}:
        errors.append("action must be one of return_valid, return_invalid, request_repair")
    if payload.get("verdict") not in {"valid", "invalid"}:
        errors.append("verdict must be valid or invalid")
    if payload.get("action") == "return_valid" and payload.get("verdict") != "valid":
        errors.append("return_valid requires verdict=valid")
    if payload.get("action") == "return_invalid" and payload.get("verdict") != "invalid":
        errors.append("return_invalid requires verdict=invalid")
    if payload.get("action") == "request_repair" and not str(payload.get("repair_instruction", "")).strip():
        errors.append("request_repair requires repair_instruction")
    for key in ["reason", "lean_evidence"]:
        if not isinstance(payload.get(key), str) or not payload.get(key, "").strip():
            errors.append(f"{key} must be non-empty string")
    if "confidence" not in payload:
        errors.append("confidence is required")
    return errors


def validate_generator_report(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("action") not in {"generated", "return_invalid"}:
        errors.append("action must be generated or return_invalid")
    if payload.get("action") == "return_invalid":
        if payload.get("verdict") != "invalid":
            errors.append("return_invalid requires verdict=invalid")
        if not isinstance(payload.get("reason"), str) or not payload.get("reason", "").strip():
            errors.append("reason must be non-empty string")
    if payload.get("action") == "generated":
        for key in ["lean_file", "compile_command", "faithfulness_summary", "reason"]:
            if not isinstance(payload.get(key), str) or not payload.get(key, "").strip():
                errors.append(f"{key} must be non-empty string")
        if not isinstance(payload.get("compile_ok"), bool):
            errors.append("compile_ok must be boolean")
        for key in ["stdout_tail", "stderr_tail"]:
            if key in payload and not isinstance(payload.get(key), str):
                errors.append(f"{key} must be string")
    if "confidence" not in payload:
        errors.append("confidence is required")
    return errors


def generator_json_call_gv(
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
    generator_thread_file: str | None,
    call_label: str,
) -> tuple[dict[str, Any], str, list[str]]:
    if mock:
        payload = mock_payload or {}
        return payload, json.dumps(payload, ensure_ascii=False, indent=2), []
    try:
        raw = base.call_model(
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
            codex_thread_file=generator_thread_file,
            call_label=call_label,
        )
        parsed, final_raw, errors = parse_json_with_repair(
            raw=raw,
            validator=validate_generator_report,
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
            codex_thread_file=generator_thread_file,
            call_label=call_label,
        )
        if not errors:
            return parsed, final_raw, []
        return {"action": "return_invalid", "verdict": "invalid", "reason": "; ".join(errors), "confidence": 1}, final_raw, errors
    except Exception as exc:
        return {"action": "return_invalid", "verdict": "invalid", "reason": str(exc), "confidence": 1}, str(exc), [str(exc)]


def agent_sandbox(provider: str, sandbox: str) -> str:
    if provider == "codex" and sandbox in {"read-only", "workspace-write"}:
        return "danger-full-access"
    return sandbox


def build_generator_prompt(
    row: dict[str, Any],
    initial: dict[str, Any],
    theorem_name: str,
    workspace_dir: Path,
    lean_file: Path,
) -> str:
    return (
        f"theorem 名字：{theorem_name}\n\n"
        f"工作区目录：\n{workspace_dir}\n\n"
        f"建议 Lean 文件路径：\n{lean_file}\n\n"
        f"题目：\n{row['question']}\n\n"
        f"CoT 上下文：\n{base.build_context_text(row)}\n\n"
        f"目标步骤：\n{row['target_step']}\n\n"
        f"verifier 初步判断：\n{json.dumps(initial, ensure_ascii=False, indent=2)}\n\n"
        "请在工作区中写 Lean 文件并运行编译命令；最终只返回 JSON 报告，不要输出 Lean 代码。"
    )


def build_verifier_review_prompt(
    row: dict[str, Any],
    initial: dict[str, Any],
    workspace_dir: Path,
    generator_report: dict[str, Any],
    attempt: int,
) -> str:
    return (
        f"attempt：{attempt}\n\n"
        f"工作区目录：\n{workspace_dir}\n\n"
        f"题目：\n{row['question']}\n\n"
        f"CoT 上下文：\n{base.build_context_text(row)}\n\n"
        f"目标步骤：\n{row['target_step']}\n\n"
        f"verifier 初步判断：\n{json.dumps(initial, ensure_ascii=False, indent=2)}\n\n"
        f"generator 报告：\n{json.dumps(generator_report, ensure_ascii=False, indent=2)}\n\n"
        "请判断目标步骤是否严格可靠，或者是否需要 generator 修复 Lean。"
    )


def build_generator_repair_prompt(
    row: dict[str, Any],
    theorem_name: str,
    workspace_dir: Path,
    current_lean_file: Path,
    generator_report: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    return (
        f"theorem 名字：{theorem_name}\n\n"
        f"工作区目录：\n{workspace_dir}\n\n"
        f"当前 Lean 文件路径：\n{current_lean_file}\n\n"
        f"题目：\n{row['question']}\n\n"
        f"CoT 上下文：\n{base.build_context_text(row)}\n\n"
        f"目标步骤：\n{row['target_step']}\n\n"
        f"上一轮 generator 报告：\n{json.dumps(generator_report, ensure_ascii=False, indent=2)}\n\n"
        f"verifier 修复要求：\n{decision.get('repair_instruction', '')}\n\n"
        "请在工作区中修改 Lean 文件并运行编译命令；最终只返回 JSON 报告，不要输出 Lean 代码。"
    )


def run_gv_lean_assist(
    row: dict[str, Any],
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
    generator_thread_file: str | None,
    verifier_thread_file: str | None,
) -> dict[str, Any]:
    out_dir = round_dir / "lean_gv"
    out_dir.mkdir(parents=True, exist_ok=True)
    theorem_name = safe_lean_name(row)

    initial, raw_initial, initial_errors = json_call_gv(
        system_prompt=load_prompt("verifier_initial.md"),
        user_prompt=base.build_initial_judge_prompt(row),
        provider=provider,
        model=model,
        mock_payload={"verdict": "valid" if not row.get("adversarial") else "invalid", "reason": "mock verifier initial", "confidence": 3},
        mock=mock,
        llm_timeout=llm_timeout,
        max_tokens=judge_max_tokens,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
        codex_thread_file=verifier_thread_file,
        call_label=f"{row['id']}-gv-verifier-initial",
    )
    (out_dir / "verifier_initial.response.txt").write_text(raw_initial, encoding="utf-8")
    write_json({"judgment": initial, "errors": initial_errors}, out_dir / "verifier_initial.json")
    if initial.get("verdict") != "valid":
        result = {
            "mode": "lean_assisted",
            "stage": "verifier_initial_only",
            "judgment": {
                "verdict": initial.get("verdict", "invalid"),
                "reason": initial.get("reason", ""),
                "lean_evidence": "Lean 未运行：verifier 初步判断没有认为目标步骤正确。",
                "confidence": initial.get("confidence", 1),
            },
            "errors": initial_errors,
            "lean_used": False,
            "verifier_initial": initial,
            "generator_events": [],
            "verifier_decisions": [],
            "lean_checks": [],
            "generator_thread_file": generator_thread_file,
            "verifier_thread_file": verifier_thread_file,
        }
        write_json(result, out_dir / "lean_assisted_result.json")
        return result

    try:
        workspace_stem = str(round_dir.relative_to(ROOT))
    except ValueError:
        workspace_stem = str(round_dir)
    workspace_dir = project_dir / ".single_step_gv_workspaces" / safe_id(workspace_stem)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    lean_file = workspace_dir / f"{theorem_name}.lean"
    write_json(
        {
            "theorem_name": theorem_name,
            "workspace_dir": str(workspace_dir),
            "lean_file": str(lean_file),
            "question": row.get("question"),
            "target_step": row.get("target_step"),
        },
        workspace_dir / "task.json",
    )
    mock_report = {
        "action": "generated",
        "lean_file": str(lean_file),
        "compile_command": f"lake env lean {lean_file}",
        "compile_ok": True,
        "stdout_tail": "",
        "stderr_tail": "",
        "faithfulness_summary": "mock faithfulness summary",
        "reason": "mock generator report",
        "confidence": 3,
    }
    generator_report, raw_code, code_errors = generator_json_call_gv(
        system_prompt=load_prompt("generator_formalize.md"),
        user_prompt=build_generator_prompt(row, initial, theorem_name, workspace_dir, lean_file),
        provider=provider,
        model=model,
        mock=mock,
        mock_payload=mock_report,
        llm_timeout=llm_timeout,
        max_tokens=lean_max_tokens,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=agent_sandbox(provider, codex_sandbox),
        codex_cwd=str(project_dir),
        generator_thread_file=generator_thread_file,
        call_label=f"{row['id']}-gv-generator-formalize",
    )
    (out_dir / "generator_formalize.response.txt").write_text(raw_code, encoding="utf-8")
    generator_events = [{"stage": "formalize", "raw": raw_code, "errors": code_errors, "report": generator_report}]
    if generator_report.get("action") == "return_invalid":
        invalid_judgment = {
            "verdict": "invalid",
            "reason": generator_report.get("reason", ""),
            "lean_evidence": "generator 在工作区形式化阶段直接发现目标步骤不可靠。",
            "confidence": generator_report.get("confidence", 1),
        }
        result = {
            "mode": "lean_assisted",
            "stage": "generator_return_invalid",
            "judgment": invalid_judgment,
            "errors": code_errors,
            "lean_used": False,
            "verifier_initial": initial,
            "generator_events": generator_events,
            "verifier_decisions": [],
            "lean_checks": [],
            "workspace_dir": str(workspace_dir),
            "generator_thread_file": generator_thread_file,
            "verifier_thread_file": verifier_thread_file,
        }
        write_json(result, out_dir / "lean_assisted_result.json")
        return result

    verifier_decisions: list[dict[str, Any]] = []
    final_judgment: dict[str, Any] | None = None
    final_stage = "no_final"
    final_errors: list[str] = []
    for attempt in range(repair_rounds + 1):
        write_json(generator_report, out_dir / f"generator_report_{attempt}.json")
        decision, raw_decision, decision_errors = json_call_gv(
            system_prompt=load_prompt("verifier_review.md"),
            user_prompt=build_verifier_review_prompt(row, initial, workspace_dir, generator_report, attempt),
            provider=provider,
            model=model,
            mock_payload={
                "action": "return_invalid" if row.get("adversarial") else "return_valid",
                "verdict": "invalid" if row.get("adversarial") else "valid",
                "reason": "mock verifier review",
                "lean_evidence": "mock",
                "repair_instruction": "",
                "confidence": 3,
            },
            mock=mock,
            llm_timeout=llm_timeout,
            max_tokens=judge_max_tokens,
            reasoning=reasoning,
            openai_reasoning_effort=openai_reasoning_effort,
            codex_reasoning_effort=codex_reasoning_effort,
            codex_sandbox=agent_sandbox(provider, codex_sandbox),
            codex_cwd=str(project_dir),
            codex_thread_file=verifier_thread_file,
            call_label=f"{row['id']}-gv-verifier-review-{attempt}",
            validator=validate_verifier_decision,
        )
        decision_record = {"attempt": attempt, "decision": decision, "raw": raw_decision, "errors": decision_errors}
        if decision.get("action") == "return_valid" and generator_report.get("compile_ok") is not True:
            guard_reason = (
                "verifier attempted return_valid, but generator_report.compile_ok is not true; "
                "Lean-assisted cannot accept a step without a successful Lean compile."
            )
            decision_errors = [*decision_errors, guard_reason]
            if attempt < repair_rounds:
                decision = {
                    "action": "request_repair",
                    "verdict": "invalid",
                    "reason": guard_reason,
                    "lean_evidence": str(generator_report.get("stderr_tail") or generator_report.get("reason") or ""),
                    "repair_instruction": (
                        "请先确保工作区中的 Lean 文件真实存在，并且运行 `lake env lean <lean_file>` 成功；"
                        "如果无法编译或发现目标步骤错误，返回 return_invalid JSON。"
                    ),
                    "confidence": 1,
                }
                decision_record["overridden_decision"] = decision
                decision_record["errors"] = decision_errors
            else:
                decision = {
                    "action": "return_invalid",
                    "verdict": "invalid",
                    "reason": guard_reason,
                    "lean_evidence": str(generator_report.get("stderr_tail") or generator_report.get("reason") or ""),
                    "repair_instruction": "",
                    "confidence": 1,
                }
                decision_record["overridden_decision"] = decision
                decision_record["errors"] = decision_errors
        verifier_decisions.append(decision_record)
        (out_dir / f"verifier_review_{attempt}.response.txt").write_text(raw_decision, encoding="utf-8")
        if decision.get("action") in {"return_valid", "return_invalid"}:
            final_stage = f"verifier_{decision.get('action')}"
            final_judgment = {
                "verdict": decision.get("verdict", "invalid"),
                "reason": decision.get("reason", ""),
                "lean_evidence": decision.get("lean_evidence", ""),
                "confidence": decision.get("confidence", 1),
            }
            final_errors = decision_errors
            break
        if attempt >= repair_rounds:
            final_stage = "repair_budget_exhausted"
            final_judgment = {
                "verdict": "invalid",
                "reason": decision.get("reason", "verifier requested repair but repair budget was exhausted"),
                "lean_evidence": decision.get("lean_evidence", "Lean 修复轮数耗尽。"),
                "confidence": decision.get("confidence", 1),
            }
            final_errors = decision_errors
            break

        current_lean_file = Path(str(generator_report.get("lean_file") or lean_file))
        generator_report, raw_repair, repair_errors = generator_json_call_gv(
            system_prompt=load_prompt("generator_repair.md"),
            user_prompt=build_generator_repair_prompt(row, theorem_name, workspace_dir, current_lean_file, generator_report, decision),
            provider=provider,
            model=model,
            mock=mock,
            mock_payload=mock_report,
            llm_timeout=llm_timeout,
            max_tokens=lean_max_tokens,
            reasoning=reasoning,
            openai_reasoning_effort=openai_reasoning_effort,
            codex_reasoning_effort=codex_reasoning_effort,
            codex_sandbox=agent_sandbox(provider, codex_sandbox),
            codex_cwd=str(project_dir),
            generator_thread_file=generator_thread_file,
            call_label=f"{row['id']}-gv-generator-repair-{attempt + 1}",
        )
        (out_dir / f"generator_repair_{attempt + 1}.response.txt").write_text(raw_repair, encoding="utf-8")
        generator_events.append({"stage": f"repair_{attempt + 1}", "raw": raw_repair, "errors": repair_errors, "report": generator_report})
        if generator_report.get("action") == "return_invalid":
            final_stage = "generator_repair_return_invalid"
            final_judgment = {
                "verdict": "invalid",
                "reason": generator_report.get("reason", ""),
                "lean_evidence": "generator 在工作区修复阶段直接发现目标步骤不可靠。",
                "confidence": generator_report.get("confidence", 1),
            }
            final_errors = repair_errors
            break

    result = {
        "mode": "lean_assisted",
        "stage": final_stage,
        "judgment": final_judgment or {"verdict": "invalid", "reason": "no final judgment", "confidence": 1},
        "errors": final_errors,
        "lean_used": bool(generator_events),
        "verifier_initial": initial,
        "generator_events": generator_events,
        "verifier_decisions": verifier_decisions,
        "lean_checks": [],
        "workspace_dir": str(workspace_dir),
        "generator_thread_file": generator_thread_file,
        "verifier_thread_file": verifier_thread_file,
    }
    write_json(result, out_dir / "lean_assisted_result.json")
    return result


def run_configured_game(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(args.config)
    provider = args.llm_provider or os.environ.get("LLM_PROVIDER") or cfg_get(config, "llm.provider", "codex")
    model = args.model or cfg_get(config, "llm.model", None)
    llm_timeout = args.llm_timeout if args.llm_timeout is not None else int(cfg_get(config, "llm.timeout", 900))
    lean_max_tokens = args.lean_max_tokens if args.lean_max_tokens is not None else int(cfg_get(config, "llm.lean_max_tokens", 4096))
    repair_rounds = args.repair_rounds if args.repair_rounds is not None else int(cfg_get(config, "lean.repair_rounds", 3))
    project_dir = Path(args.project_dir or cfg_get(config, "paths.lean_project_dir", base.default_lean_project_dir()))
    lean_timeout = args.lean_timeout if args.lean_timeout is not None else int(cfg_get(config, "lean.timeout", 120))
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
    hacker_thread_dir = run_dir / "hacker_threads"
    gv_thread_dir = run_dir / "lean_gv_threads"
    rounds_dir = run_dir / "rounds"
    for path in [input_dir, hacker_thread_dir, gv_thread_dir, rounds_dir]:
        path.mkdir(parents=True, exist_ok=True)

    candidates = read_jsonl(Path(args.candidates))
    random.Random(args.seed).shuffle(candidates)
    candidates = candidates[: args.limit]
    write_jsonl(candidates, input_dir / "candidates.jsonl")
    write_json(
        {
            "run_id": args.run_id,
            "runner": "run_adversarial_game_gv.py",
            "prompt_dir": str(PROMPT_DIR),
            "baseline_prompt": "adaptive_adversarial_gv/verifier_initial.md",
            "candidates": args.candidates,
            "limit": args.limit,
            "max_rounds": args.max_rounds,
            "provider": "mock" if args.mock else provider,
            "model": "mock" if args.mock else model,
            "llm_timeout": llm_timeout,
            "codex_sandbox": codex_sandbox,
            "codex_agent_sandbox": agent_sandbox(provider, codex_sandbox),
            "project_dir": str(project_dir),
            "lean_timeout": lean_timeout,
            "hacker_thread_dir": str(hacker_thread_dir),
            "gv_thread_dir": str(gv_thread_dir),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        run_dir / "run_config.json",
    )

    trace: list[dict[str, Any]] = []
    for case_idx, candidate in enumerate(candidates, start=1):
        case_id = safe_id(f"{case_idx}_{candidate['id']}_c{candidate['chain_id']}_s{candidate['step_id']}")
        previous_attack: dict[str, Any] | None = None
        previous_judgments: list[dict[str, Any]] | None = None
        previous_status: str | None = None
        for round_idx in range(1, args.max_rounds + 1):
            round_dir = rounds_dir / case_id / f"round_{round_idx}"
            round_dir.mkdir(parents=True, exist_ok=True)
            gv_case_thread_dir = gv_thread_dir / case_id / f"round_{round_idx}"
            gv_case_thread_dir.mkdir(parents=True, exist_ok=True)
            generator_thread_file = str(gv_case_thread_dir / "generator.thread")
            verifier_thread_file = str(gv_case_thread_dir / "verifier.thread")

            attack, raw_attack, attack_errors = base.call_hacker(
                candidate,
                round_idx=round_idx,
                previous_attack=previous_attack,
                previous_judgments=previous_judgments,
                previous_status=previous_status,
                system_prompt="",
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
                codex_thread_dir=str(hacker_thread_dir),
            )
            (round_dir / "hacker_response.txt").write_text(raw_attack, encoding="utf-8")
            write_json({"attack": attack, "errors": attack_errors}, round_dir / "attack.json")
            if attack_errors or not attack.get("attackable"):
                trace.append(
                    {
                        "case_id": case_id,
                        "round": round_idx,
                        "round_dir": str(round_dir),
                        "round_status": "hacker_failed",
                        "attack": attack,
                        "attack_errors": attack_errors,
                        "judgments": [],
                    }
                )
                base.write_run_progress(trace, cases=len(candidates), target_lean_rescues=args.target_lean_rescues, run_dir=run_dir)
                break

            row = build_invalid_row(candidate, attack, round_idx)
            write_jsonl([row], round_dir / "adversarial_step.jsonl")
            baseline_result = run_baseline_verifier_judge(
                row,
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
            lean_result = run_gv_lean_assist(
                row,
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
                generator_thread_file=generator_thread_file,
                verifier_thread_file=verifier_thread_file,
            )
            verifier_initial = {
                "mode": "lean_initial",
                "stage": "initial",
                "judgment": lean_result.get("verifier_initial", {}),
                "errors": [],
            }
            judgments = [baseline_result, verifier_initial, lean_result]
            write_jsonl(judgments, round_dir / "judgments.jsonl")
            round_status = base.classify_round(judgments)
            trace_row = {
                "case_id": case_id,
                "round": round_idx,
                "round_dir": str(round_dir),
                "round_status": round_status,
                "attack": attack,
                "judgments": judgments,
                "generator_thread_file": generator_thread_file,
                "verifier_thread_file": verifier_thread_file,
            }
            trace.append(trace_row)
            base.write_run_progress(trace, cases=len(candidates), target_lean_rescues=args.target_lean_rescues, run_dir=run_dir)

            previous_attack = attack
            previous_judgments = judgments
            previous_status = round_status
            if args.target_lean_rescues and sum(1 for item in trace if item.get("round_status") == "lean_rescue") >= args.target_lean_rescues:
                break
            if round_status == "lean_rescue":
                break
            if not args.continue_after_non_obvious and round_status != "too_obvious":
                break
        if args.target_lean_rescues and sum(1 for item in trace if item.get("round_status") == "lean_rescue") >= args.target_lean_rescues:
            break

    return base.write_run_progress(trace, cases=len(candidates), target_lean_rescues=args.target_lean_rescues, run_dir=run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=base.existing_default_config())
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--target-lean-rescues", type=int, default=0)
    parser.add_argument("--continue-after-non-obvious", action="store_true")
    parser.add_argument("--llm-provider", choices=["openai", "codex"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--llm-timeout", type=int, default=None)
    parser.add_argument("--hacker-max-tokens", type=int, default=8192)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--lean-max-tokens", type=int, default=None)
    parser.add_argument("--repair-rounds", type=int, default=None)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--lean-timeout", type=int, default=None)
    parser.add_argument("--reasoning", choices=["auto", "enabled", "disabled"], default=None)
    parser.add_argument("--openai-reasoning-effort", choices=["high", "max"], default=None)
    parser.add_argument("--codex-reasoning-effort", default=None)
    parser.add_argument("--codex-sandbox", default=None)
    parser.add_argument("--codex-cwd", default=None)
    args = parser.parse_args()

    summary = run_configured_game(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Run directory: {summary['output']}")


if __name__ == "__main__":
    main()
