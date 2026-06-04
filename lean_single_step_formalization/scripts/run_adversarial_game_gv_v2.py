#!/usr/bin/env python3
"""V2 adaptive adversarial GV runner with split compile-ok/fail review prompts."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "prompts" / "adaptive_adversarial_gv_v2"

import run_adversarial_game_gv as gv


GLOBAL_PROMPT_NAMES = {
    "verifier_initial.md",
    "generator_formalize.md",
    "generator_repair.md",
    "verifier_review_compile_ok.md",
    "verifier_review_compile_fail.md",
}


def load_prompt(name: str) -> str:
    prompt = (PROMPT_DIR / name).read_text(encoding="utf-8").strip()
    if name in GLOBAL_PROMPT_NAMES:
        global_prompt = (PROMPT_DIR / "check_global.md").read_text(encoding="utf-8").strip()
        return f"{global_prompt}\n\n{prompt}"
    return prompt


# Reuse the existing implementation helpers without modifying the old runner.
gv.PROMPT_DIR = PROMPT_DIR
gv.load_prompt = load_prompt

agent_sandbox = gv.agent_sandbox
json_call_gv = gv.json_call_gv
run_step_decomposition_for_row = gv.run_step_decomposition_for_row
run_baseline_verifier_judge_with_decomposition = gv.run_baseline_verifier_judge_with_decomposition


def review_prompt_name(generator_report: dict[str, Any]) -> str:
    if generator_report.get("compile_ok") is True:
        return "verifier_review_compile_ok.md"
    return "verifier_review_compile_fail.md"


def review_thread_file(verifier_thread_dir: Path, attempt: int, prompt_name: str) -> str:
    stem = prompt_name.removesuffix(".md")
    return str(verifier_thread_dir / f"{attempt:02d}_{stem}.thread")


def override_bad_review_action(
    *,
    decision: dict[str, Any],
    generator_report: dict[str, Any],
    attempt: int,
    repair_rounds: int,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    compile_ok = generator_report.get("compile_ok") is True
    action = decision.get("action")

    if compile_ok and action == "return_invalid":
        reason = "compile-ok review must not return_invalid; it should request Lean repair if the Lean file is not faithful."
        errors.append(reason)
        if attempt < repair_rounds:
            return (
                {
                    "action": "request_repair",
                    "verdict": "invalid",
                    "unmatched_lean_parts": ["compile-ok reviewer returned return_invalid"],
                    "unmatched_natural_language_parts": [],
                    "reason": reason,
                    "lean_evidence": str(generator_report.get("lean_file") or ""),
                    "repair_instruction": "compile_ok=true 时只能审查 Lean 忠实性；请修复 Lean 与目标步骤/step_d 的不匹配，而不是直接判自然语言 invalid。",
                    "confidence": 1,
                },
                errors,
            )
    if not compile_ok and action == "return_valid":
        reason = "compile-fail review must not return_valid; an uncompiled Lean report can only justify invalid or request repair."
        errors.append(reason)
        if attempt < repair_rounds:
            return (
                {
                    "action": "request_repair",
                    "verdict": "invalid",
                    "unmatched_lean_parts": ["generator_report.compile_ok is false"],
                    "unmatched_natural_language_parts": [],
                    "reason": reason,
                    "lean_evidence": str(generator_report.get("stderr_tail") or generator_report.get("reason") or ""),
                    "repair_instruction": "compile_ok=false 时请判断失败理由；如果理由不足以说明原步骤 invalid，应要求 generator 继续修到忠实可编译或给出更具体的数学失败理由。",
                    "confidence": 1,
                },
                errors,
            )
    return decision, errors


def run_gv_lean_assist_v2(
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
    verifier_thread_dir: Path,
    verifier_initial_prompt: str = "verifier_initial.md",
    initial_override: dict[str, Any] | None = None,
    initial_errors_override: list[str] | None = None,
    step_decomposition_override: dict[str, Any] | None = None,
    step_decomposition_errors_override: list[str] | None = None,
) -> dict[str, Any]:
    out_dir = round_dir / "lean_gv_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    verifier_thread_dir.mkdir(parents=True, exist_ok=True)
    theorem_name = gv.safe_lean_name(row)

    initial = initial_override or {
        "verdict": "valid",
        "reason": "复用外部 initial；未提供时默认进入 Lean/GV。",
        "confidence": 1,
    }
    initial_errors = initial_errors_override or []
    (out_dir / "verifier_initial.reused.json").write_text(
        json.dumps({"judgment": initial, "errors": initial_errors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if step_decomposition_override is not None:
        step_decomposition = step_decomposition_override
        decomposition_errors = step_decomposition_errors_override or []
        gv.write_json(
            {
                "decomposition": step_decomposition,
                "errors": decomposition_errors,
                "source": "external_override",
            },
            out_dir / "step_decomposition.json",
        )
    else:
        step_decomposition, _, decomposition_errors = gv.run_step_decomposition_for_row(
            row,
            out_dir=out_dir,
            initial=initial,
            provider=provider,
            model=model,
            mock=mock,
            llm_timeout=llm_timeout,
            max_tokens=judge_max_tokens,
            reasoning=reasoning,
            openai_reasoning_effort=openai_reasoning_effort,
            codex_reasoning_effort=codex_reasoning_effort,
            codex_sandbox=agent_sandbox(provider, codex_sandbox),
            codex_cwd=str(project_dir),
            codex_thread_file=None,
        )

    try:
        workspace_stem = str(round_dir.relative_to(ROOT))
    except ValueError:
        workspace_stem = str(round_dir)
    workspace_dir = project_dir / ".single_step_gv_workspaces" / gv.safe_id(f"v2_{workspace_stem}")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    lean_file = workspace_dir / f"{theorem_name}.lean"
    gv.write_json(
        {
            "theorem_name": theorem_name,
            "workspace_dir": str(workspace_dir),
            "lean_file": str(lean_file),
            "question": row.get("question"),
            "target_step": row.get("target_step"),
            "step_decomposition": step_decomposition,
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

    generator_system_prompt = load_prompt("generator_formalize.md")
    generator_user_prompt = gv.build_generator_prompt(row, initial, step_decomposition, theorem_name, workspace_dir, lean_file)
    (out_dir / "generator_formalize.prompt.md").write_text(
        f"# System prompt\n\n{generator_system_prompt}\n\n# User prompt\n\n{generator_user_prompt}",
        encoding="utf-8",
    )
    generator_report, raw_code, code_errors = gv.generator_json_call_gv(
        system_prompt=generator_system_prompt,
        user_prompt=generator_user_prompt,
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
        call_label=f"{row['id']}-gv-v2-generator-formalize",
    )
    (out_dir / "generator_formalize.response.txt").write_text(raw_code, encoding="utf-8")
    generator_report = gv.normalize_generator_report(generator_report, lean_file=lean_file)
    if provider != "codex":
        generator_report = gv.materialize_api_lean_report(
            generator_report,
            lean_file=Path(str(generator_report.get("lean_file") or lean_file)),
            project_dir=project_dir,
            lean_timeout=lean_timeout,
        )
    generator_events = [{"stage": "formalize", "raw": raw_code, "errors": code_errors, "report": generator_report}]

    verifier_decisions: list[dict[str, Any]] = []
    final_judgment: dict[str, Any] | None = None
    final_stage = "no_final"
    final_errors: list[str] = []

    for attempt in range(repair_rounds + 1):
        gv.write_json(generator_report, out_dir / f"generator_report_{attempt}.json")
        prompt_name = review_prompt_name(generator_report)
        review_system_prompt = load_prompt(prompt_name)
        review_user_prompt = gv.build_verifier_review_prompt(row, initial, step_decomposition, workspace_dir, generator_report, attempt)
        review_thread = review_thread_file(verifier_thread_dir, attempt, prompt_name)
        (out_dir / f"verifier_review_{attempt}_{prompt_name.removesuffix('.md')}.prompt.md").write_text(
            f"# System prompt\n\n{review_system_prompt}\n\n# User prompt\n\n{review_user_prompt}",
            encoding="utf-8",
        )
        decision, raw_decision, decision_errors = gv.json_call_gv(
            system_prompt=review_system_prompt,
            user_prompt=review_user_prompt,
            provider=provider,
            model=model,
            mock_payload={
                "action": "return_valid" if generator_report.get("compile_ok") is True else "return_invalid",
                "verdict": "valid" if generator_report.get("compile_ok") is True else "invalid",
                "unmatched_lean_parts": [],
                "unmatched_natural_language_parts": [],
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
            codex_thread_file=review_thread,
            call_label=f"{row['id']}-gv-v2-verifier-review-{attempt}",
            validator=gv.validate_verifier_decision,
        )
        decision_record = {
            "attempt": attempt,
            "review_prompt": prompt_name,
            "verifier_thread_file": review_thread,
            "decision": decision,
            "raw": raw_decision,
            "errors": decision_errors,
        }
        overridden_decision, override_errors = override_bad_review_action(
            decision=decision,
            generator_report=generator_report,
            attempt=attempt,
            repair_rounds=repair_rounds,
        )
        if override_errors:
            decision = overridden_decision
            decision_errors = [*decision_errors, *override_errors]
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
        repair_system_prompt = load_prompt("generator_repair.md")
        repair_user_prompt = gv.build_generator_repair_prompt(
            row,
            step_decomposition,
            theorem_name,
            workspace_dir,
            current_lean_file,
            generator_report,
            decision,
        )
        (out_dir / f"generator_repair_{attempt + 1}.prompt.md").write_text(
            f"# System prompt\n\n{repair_system_prompt}\n\n# User prompt\n\n{repair_user_prompt}",
            encoding="utf-8",
        )
        generator_report, raw_repair, repair_errors = gv.generator_json_call_gv(
            system_prompt=repair_system_prompt,
            user_prompt=repair_user_prompt,
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
            call_label=f"{row['id']}-gv-v2-generator-repair-{attempt + 1}",
        )
        (out_dir / f"generator_repair_{attempt + 1}.response.txt").write_text(raw_repair, encoding="utf-8")
        generator_report = gv.normalize_generator_report(generator_report, lean_file=current_lean_file)
        if provider != "codex":
            generator_report = gv.materialize_api_lean_report(
                generator_report,
                lean_file=Path(str(generator_report.get("lean_file") or current_lean_file)),
                project_dir=project_dir,
                lean_timeout=lean_timeout,
            )
        generator_events.append({"stage": f"repair_{attempt + 1}", "raw": raw_repair, "errors": repair_errors, "report": generator_report})

    result = {
        "mode": "lean_assisted_v2",
        "stage": final_stage,
        "judgment": final_judgment or {"verdict": "invalid", "reason": "no final judgment", "confidence": 1},
        "errors": final_errors,
        "lean_used": bool(generator_events),
        "verifier_initial": initial,
        "verifier_initial_prompt": verifier_initial_prompt,
        "step_decomposition": step_decomposition,
        "step_decomposition_errors": decomposition_errors,
        "generator_events": generator_events,
        "verifier_decisions": verifier_decisions,
        "lean_checks": [],
        "workspace_dir": str(workspace_dir),
        "generator_thread_file": generator_thread_file,
        "verifier_thread_dir": str(verifier_thread_dir),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    result["outputs_md"] = str(gv.write_lean_gv_outputs_md(out_dir, result))
    gv.write_json(result, out_dir / "lean_assisted_result.json")
    return result
