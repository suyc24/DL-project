#!/usr/bin/env python3
"""Create adversarial single-step verification examples from audit candidates."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"

sys.path.insert(0, str(SCRIPT_DIR))
from llm_client import LLMCallError, call_llm

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


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")[:120] or "sample"


def candidate_thread_file(candidate: dict[str, Any], thread_dir: str | None) -> str | None:
    if not thread_dir:
        return None
    stem = safe_id(f"{candidate['id']}_c{candidate['chain_id']}_s{candidate['step_id']}")
    return str(Path(thread_dir) / f"{stem}.thread")


def build_hacker_prompt(candidate: dict[str, Any]) -> str:
    context = "\n".join(
        f"{step['step_id']}. {'[目标] ' if step.get('is_selected') else ''}{step['text']}"
        for step in candidate.get("context_steps", [])
    )
    score = candidate.get("selection_score") or {}
    return (
        f"题目：\n{candidate['question']}\n\n"
        f"CoT 上下文：\n{context or '(none)'}\n\n"
        f"目标步骤序号：{candidate['step_id']}\n"
        f"目标步骤原文：\n{candidate['target_step']}\n\n"
        f"模型最终答案：\n{candidate.get('model_final_answer') or candidate.get('final_answer') or '(unknown)'}\n\n"
        f"标准答案：\n{candidate.get('gold_answer') or '(unknown)'}\n\n"
        f"步骤风险评分：\n{json.dumps(score, ensure_ascii=False, indent=2)}\n\n"
        "请生成一个隐蔽的 flawed_step。"
    )


def mock_attack(candidate: dict[str, Any], idx: int) -> dict[str, Any]:
    original = str(candidate.get("target_step", ""))
    flawed = original
    flaw_type = "too_strong"
    why = "把原步骤的结论加强为必然成立，但上下文没有提供足够前提。"
    replacements = [
        (r"≥", ">=", "inequality_direction", "把不等式方向或强弱关系改错。"),
        (r"≤", ">=", "inequality_direction", "把不等式方向改错。"),
        (r">", "≥", "boundary_case", "把严格条件放宽，可能引入边界反例。"),
        (r"<", "≤", "boundary_case", "把严格条件放宽，可能引入边界反例。"),
        (r"存在", "任意", "quantifier_swap", "把存在性结论偷换成全称结论。"),
        (r"任意", "存在", "quantifier_swap", "把全称前提偷换成存在性前提。"),
        (r"mod 6", "mod 4", "modular_condition", "细微改变同余模数，原构造不再覆盖。"),
        (r"\\+84", "+85", "algebra_error", "把关键常数改错。"),
    ]
    for pattern, repl, kind, reason in replacements:
        if pattern in flawed:
            flawed = flawed.replace(pattern, repl, 1)
            flaw_type = kind
            why = reason
            break
    if flawed == original:
        flawed = original.rstrip("。") + "，并且该结论在所有相关情形下都自动成立。"
    return {
        "attackable": True,
        "flawed_step": flawed,
        "flaw_type": flaw_type,
        "why_invalid": why,
        "corrected_step": original,
        "changed_elements": ["最小改动目标步骤"],
        "stealth_strategy": "保留原句结构，只改变一个关键条件或结论强度。",
        "expected_lean_signal": "missing_premise",
        "difficulty_for_judge": 4,
    }


def validate_attack(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(record.get("attackable"), bool):
        errors.append("attackable must be boolean")
    if record.get("attackable") is False:
        return errors
    allowed_flaws = {
        "missing_premise",
        "too_strong",
        "algebra_error",
        "inequality_direction",
        "quantifier_swap",
        "modular_condition",
        "boundary_case",
        "necessity_sufficiency",
    }
    if record.get("flaw_type") not in allowed_flaws:
        errors.append("invalid flaw_type")
    for key in ["flawed_step", "why_invalid", "corrected_step", "stealth_strategy", "expected_lean_signal"]:
        if not isinstance(record.get(key), str) or not record.get(key, "").strip():
            errors.append(f"{key} must be non-empty string")
    if not isinstance(record.get("changed_elements"), list):
        errors.append("changed_elements must be list")
    if "difficulty_for_judge" in record:
        difficulty_keys = ["difficulty_for_judge"]
    else:
        difficulty_keys = ["difficulty_for_baseline", "difficulty_for_lean_assisted"]
    for key in difficulty_keys:
        try:
            difficulty = int(record.get(key))
            if difficulty < 1 or difficulty > 5:
                errors.append(f"{key} out of range")
        except Exception:
            errors.append(f"{key} must be integer")
    return errors


def replace_target_context(candidate: dict[str, Any], flawed_step: str) -> list[dict[str, Any]]:
    replaced = []
    for step in candidate.get("context_steps", []):
        row = dict(step)
        if row.get("is_selected"):
            row["original_text"] = row.get("text")
            row["text"] = flawed_step
        replaced.append(row)
    return replaced


def replace_target_in_cot(original_cot: str, original_step: str, flawed_step: str) -> str:
    if original_cot and original_step in original_cot:
        return original_cot.replace(original_step, flawed_step, 1)
    return original_cot


def build_invalid_row(candidate: dict[str, Any], attack: dict[str, Any], sample_index: int) -> dict[str, Any]:
    attack_id = f"{candidate['id']}_c{candidate['chain_id']}_s{candidate['step_id']}_adv{sample_index}"
    flawed_step = attack["flawed_step"].strip()
    original_step = candidate["target_step"]
    return {
        **candidate,
        "id": safe_id(attack_id),
        "source_id": candidate["id"],
        "source_chain_id": candidate["chain_id"],
        "source_step_id": candidate["step_id"],
        "adversarial": True,
        "gold_verdict": "invalid",
        "gold_issue_type": attack["flaw_type"],
        "gold_diagnosis": attack["why_invalid"],
        "gold_corrected_step": attack["corrected_step"],
        "target_step": flawed_step,
        "original_target_step": original_step,
        "context_steps": replace_target_context(candidate, flawed_step),
        "original_cot": candidate.get("original_cot", ""),
        "mutated_cot": replace_target_in_cot(candidate.get("original_cot", ""), original_step, flawed_step),
        "attack": attack,
    }


def build_valid_row(candidate: dict[str, Any]) -> dict[str, Any]:
    valid_id = f"{candidate['id']}_c{candidate['chain_id']}_s{candidate['step_id']}_valid"
    return {
        **candidate,
        "id": safe_id(valid_id),
        "source_id": candidate["id"],
        "source_chain_id": candidate["chain_id"],
        "source_step_id": candidate["step_id"],
        "adversarial": False,
        "gold_verdict": "valid",
        "gold_issue_type": "none",
        "gold_diagnosis": "原始候选步骤，作为有效样本的对照。仍建议后续由 Lean 或人工复核。",
        "gold_corrected_step": candidate["target_step"],
        "original_target_step": candidate["target_step"],
        "mutated_cot": candidate.get("original_cot", ""),
    }


def attack_one(
    idx_candidate: tuple[int, dict[str, Any]],
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
    codex_thread_id: str | None,
    codex_thread_dir: str | None,
) -> dict[str, Any]:
    idx, candidate = idx_candidate
    if mock:
        raw = json.dumps(mock_attack(candidate, idx), ensure_ascii=False, indent=2)
        parsed = mock_attack(candidate, idx)
        errors: list[str] = []
    else:
        try:
            raw = call_llm(
                system_prompt=system_prompt,
                user_prompt=build_hacker_prompt(candidate),
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
                codex_thread_id=codex_thread_id,
                codex_thread_file=candidate_thread_file(candidate, codex_thread_dir),
                call_label=f"{candidate['id']}-c{candidate['chain_id']}-s{candidate['step_id']}-hack",
            )
            parsed = extract_json_object(raw)
            errors = validate_attack(parsed)
        except (LLMCallError, Exception) as exc:
            raw = str(exc)
            parsed = {"attackable": False}
            errors = [str(exc)]
    return {
        "candidate": candidate,
        "attack": parsed,
        "raw": raw,
        "errors": errors,
        "sample_index": idx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="audit_candidates.jsonl path")
    parser.add_argument("--output", required=True, help="output JSONL with valid/invalid step examples")
    parser.add_argument("--responses-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-valid-pairs", action="store_true")
    parser.add_argument("--llm-provider", choices=["openai", "codex"], default="codex")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--llm-timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--reasoning", choices=["auto", "enabled", "disabled"], default="auto")
    parser.add_argument("--openai-reasoning-effort", choices=["high", "max"], default=None)
    parser.add_argument("--codex-reasoning-effort", default="high")
    parser.add_argument("--codex-sandbox", default="read-only")
    parser.add_argument("--codex-cwd", default=str(ROOT.parent))
    parser.add_argument("--codex-thread-id", default=None, help="resume one existing Codex session UUID for all hacker calls")
    parser.add_argument("--codex-thread-dir", default=None, help="store one Codex session UUID file per candidate")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    candidates = read_jsonl(Path(args.candidates))
    rng.shuffle(candidates)
    if args.limit is not None:
        candidates = candidates[: args.limit]

    reasoning = None
    if args.reasoning == "enabled":
        reasoning = True
    elif args.reasoning == "disabled":
        reasoning = False

    system_prompt = load_prompt("adversarial_hacker.md")
    worker_kwargs = {
        "system_prompt": system_prompt,
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
        "codex_thread_id": args.codex_thread_id,
        "codex_thread_dir": args.codex_thread_dir,
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(attack_one, (idx + 1, candidate), **worker_kwargs)
            for idx, candidate in enumerate(candidates)
        ]
        attack_results = [future.result() for future in futures]

    output_rows: list[dict[str, Any]] = []
    responses_dir = Path(args.responses_dir) if args.responses_dir else Path(args.output).parent / "hacker_responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict[str, Any]] = []
    for result in attack_results:
        candidate = result["candidate"]
        stem = safe_id(f"{candidate['id']}_c{candidate['chain_id']}_s{candidate['step_id']}")
        (responses_dir / f"{stem}.response.txt").write_text(result["raw"], encoding="utf-8")
        metadata_rows.append(
            {
                "id": candidate["id"],
                "chain_id": candidate["chain_id"],
                "step_id": candidate["step_id"],
                "attackable": result["attack"].get("attackable"),
                "errors": result["errors"],
                "response_file": str(responses_dir / f"{stem}.response.txt"),
            }
        )
        if args.include_valid_pairs:
            output_rows.append(build_valid_row(candidate))
        if result["errors"] or not result["attack"].get("attackable"):
            continue
        output_rows.append(build_invalid_row(candidate, result["attack"], result["sample_index"]))

    output_path = Path(args.output)
    write_jsonl(output_rows, output_path)
    write_jsonl(metadata_rows, output_path.with_suffix(".metadata.jsonl"))
    summary = {
        "candidates": len(candidates),
        "output_rows": len(output_rows),
        "invalid_rows": sum(1 for row in output_rows if row.get("gold_verdict") == "invalid"),
        "valid_rows": sum(1 for row in output_rows if row.get("gold_verdict") == "valid"),
        "attackable": sum(1 for row in metadata_rows if row.get("attackable") is True),
        "failed": sum(1 for row in metadata_rows if row.get("errors")),
        "output": str(output_path),
        "metadata": str(output_path.with_suffix(".metadata.jsonl")),
    }
    write_json(summary, output_path.with_suffix(".summary.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
