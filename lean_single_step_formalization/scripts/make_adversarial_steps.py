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
STEP_BLOCK_RE = re.compile(
    r"(\[Step\s+(\d+)\]\s*)(.*?)(?=\n\n\[Step\s+\d+\]|\n\n\[Final Answer\]|\Z)",
    re.I | re.S,
)


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
    full_cot = candidate.get("original_cot") or "(none)"
    score = candidate.get("selection_score") or {}
    return (
        f"题目：\n{candidate['question']}\n\n"
        f"完整 CoT：\n{full_cot}\n\n"
        f"局部 CoT 窗口：\n{context or '(none)'}\n\n"
        f"目标步骤序号：{candidate['step_id']}\n"
        f"目标步骤原文：\n{candidate['target_step']}\n\n"
        f"模型最终答案：\n{candidate.get('model_final_answer') or candidate.get('final_answer') or '(unknown)'}\n\n"
        f"标准答案：\n{candidate.get('gold_answer') or '(unknown)'}\n\n"
        f"步骤风险评分：\n{json.dumps(score, ensure_ascii=False, indent=2)}\n\n"
        "请生成一个隐蔽的错误版本，返回完整 modified_cot，并用 flawed_step 指出其中哪一步需要被评测。"
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
    target_step_id = int(candidate["step_id"])
    modified_cot = replace_steps_in_cot(
        candidate,
        {"flawed_step": flawed, "modified_steps": [{"step_id": target_step_id, "text": flawed}]},
    )
    return {
        "attackable": True,
        "flawed_step": target_step_id,
        "modified_cot": modified_cot,
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
    try:
        flawed_step = int(record.get("flawed_step"))
        if flawed_step < 1:
            errors.append("flawed_step must be positive integer")
    except Exception:
        errors.append("flawed_step must be integer")
    if not isinstance(record.get("modified_cot"), str) or not record.get("modified_cot", "").strip():
        errors.append("modified_cot must be non-empty string")
    if not isinstance(record.get("why_invalid"), str) or not record.get("why_invalid", "").strip():
        errors.append("why_invalid must be non-empty string")
    if "modified_steps" in record:
        if not isinstance(record.get("modified_steps"), list):
            errors.append("modified_steps must be list")
        else:
            for idx, item in enumerate(record.get("modified_steps") or []):
                if not isinstance(item, dict):
                    errors.append(f"modified_steps[{idx}] must be object")
                    continue
                try:
                    int(item.get("step_id"))
                except Exception:
                    errors.append(f"modified_steps[{idx}].step_id must be integer")
                if not isinstance(item.get("text"), str) or not item.get("text", "").strip():
                    errors.append(f"modified_steps[{idx}].text must be non-empty string")
    if "changed_elements" in record and not isinstance(record.get("changed_elements"), list):
        errors.append("changed_elements must be list")
    if "difficulty_for_judge" in record:
        difficulty_keys = ["difficulty_for_judge"]
    elif "difficulty_for_baseline" in record or "difficulty_for_lean_assisted" in record:
        difficulty_keys = ["difficulty_for_baseline", "difficulty_for_lean_assisted"]
    else:
        difficulty_keys = []
    for key in difficulty_keys:
        try:
            difficulty = int(record.get(key))
            if difficulty < 1 or difficulty > 5:
                errors.append(f"{key} out of range")
        except Exception:
            errors.append(f"{key} must be integer")
    return errors


def replacement_steps(attack: dict[str, Any]) -> dict[int, str]:
    replacements: dict[int, str] = {}
    for item in attack.get("modified_steps") or []:
        try:
            replacements[int(item.get("step_id"))] = str(item.get("text", "")).strip()
        except Exception:
            continue
    return {step_id: text for step_id, text in replacements.items() if text}


def attack_step_id(candidate: dict[str, Any], attack: dict[str, Any]) -> int:
    try:
        return int(attack.get("flawed_step"))
    except Exception:
        return int(candidate["step_id"])


def extract_step_text(cot: str, step_id: int) -> str:
    for match in STEP_BLOCK_RE.finditer(cot or ""):
        if int(match.group(2)) == step_id:
            return match.group(3).strip()
    return ""


def attack_step_text(candidate: dict[str, Any], attack: dict[str, Any]) -> str:
    step_id = attack_step_id(candidate, attack)
    mutated_cot = replace_steps_in_cot(candidate, attack)
    extracted = extract_step_text(mutated_cot, step_id)
    if extracted:
        return extracted
    flawed_step = attack.get("flawed_step")
    if isinstance(flawed_step, str) and flawed_step.strip() and not flawed_step.strip().isdigit():
        return flawed_step.strip()
    return str(candidate.get("target_step", "")).strip()


def replace_target_context(candidate: dict[str, Any], attack: dict[str, Any]) -> list[dict[str, Any]]:
    target_step_id = attack_step_id(candidate, attack)
    flawed_step = attack_step_text(candidate, attack)
    extra_replacements = replacement_steps(attack)
    replaced = []
    for step in candidate.get("context_steps", []):
        row = dict(step)
        if int(row.get("step_id")) == target_step_id:
            row["original_text"] = row.get("text")
            row["text"] = flawed_step
            row["is_selected"] = True
        elif int(row.get("step_id")) in extra_replacements:
            row["original_text"] = row.get("text")
            row["text"] = extra_replacements[int(row.get("step_id"))]
            row["is_selected"] = False
        else:
            row["is_selected"] = False
        replaced.append(row)
    return replaced


def replace_steps_in_cot(candidate: dict[str, Any], attack: dict[str, Any]) -> str:
    if isinstance(attack.get("modified_cot"), str) and attack.get("modified_cot", "").strip():
        return attack["modified_cot"].strip()
    mutated = candidate.get("original_cot", "")
    if not mutated:
        return mutated
    replacements = replacement_steps(attack)
    flawed_step = attack.get("flawed_step")
    if isinstance(flawed_step, str) and flawed_step.strip() and not flawed_step.strip().isdigit():
        replacements[int(candidate["step_id"])] = flawed_step.strip()

    def replace_match(match: re.Match[str]) -> str:
        step_id = int(match.group(2))
        if step_id not in replacements:
            return match.group(0)
        return f"{match.group(1)}{replacements[step_id]}"

    return STEP_BLOCK_RE.sub(replace_match, mutated)


def build_invalid_row(candidate: dict[str, Any], attack: dict[str, Any], sample_index: int) -> dict[str, Any]:
    target_step_id = attack_step_id(candidate, attack)
    attack_id = f"{candidate['id']}_c{candidate['chain_id']}_s{target_step_id}_adv{sample_index}"
    flawed_step = attack_step_text(candidate, attack)
    original_step = extract_step_text(candidate.get("original_cot", ""), target_step_id) or candidate["target_step"]
    mutated_cot = replace_steps_in_cot(candidate, attack)
    return {
        **candidate,
        "id": safe_id(attack_id),
        "source_id": candidate["id"],
        "source_chain_id": candidate["chain_id"],
        "source_step_id": target_step_id,
        "step_id": target_step_id,
        "adversarial": True,
        "gold_verdict": "invalid",
        "gold_issue_type": attack.get("flaw_type", "adversarial_flaw"),
        "gold_diagnosis": attack["why_invalid"],
        "gold_corrected_step": attack.get("corrected_step", original_step),
        "target_step": flawed_step,
        "original_target_step": original_step,
        "context_steps": replace_target_context(candidate, attack),
        "original_cot": candidate.get("original_cot", ""),
        "mutated_cot": mutated_cot,
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
    parser.add_argument("--codex-sandbox", default="danger-full-access")
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
