#!/usr/bin/env python3
"""End-to-end single-step Lean formalization loop.

The loop is intentionally small:

1. read natural-language math problems;
2. ask an LLM for structured CoT traces;
3. parse steps;
4. randomly select target steps;
5. ask an LLM to wrap each target step as a local premise -> conclusion claim;
6. ask an LLM to formalize each wrapped claim as a Lean transition contract;
7. optionally run `lake env lean` in a Mathlib project.

The script can run in `--mock` mode without API keys or Lean. This is useful for
checking file layout and downstream parsing before spending LLM calls.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
RUNS_DIR = ROOT / "experiments" / "runs"
PROMPTS_DIR = ROOT / "prompts"

sys.path.insert(0, str(SCRIPT_DIR))
from llm_client import call_llm, call_llm_batch


BRACKET_STEP_RE = re.compile(
    r"\[Step\s*(\d+)\]\s*(.*?)(?=(?:\n\[Step\s*\d+\])|\n\[Final\s*Answer\]|\Z)",
    re.I | re.S,
)
BRACKET_FINAL_RE = re.compile(r"\[Final\s*Answer\]\s*(.*)", re.I | re.S)
COLON_STEP_RE = re.compile(
    r"(?:^|\n)\s*Step\s*(\d+)\s*:\s*(.*?)(?=(?:\n\s*Step\s*\d+\s*:)|\n\s*Final\s*Answer\s*:|\Z)",
    re.I | re.S,
)
COLON_FINAL_RE = re.compile(r"Final\s*Answer\s*[:\-]?\s*(.*)", re.I | re.S)
LEAN_BLOCK_RE = re.compile(r"```\s*(?:lean4?|lean)?\s*\n(.*?)```", re.I | re.S)
JSON_BLOCK_RE = re.compile(r"```\s*(?:json)?\s*\n(.*?)```", re.I | re.S)
BANNED_LEAN_RE = re.compile(r"\b(sorry|admit)\b", re.I)
AXIOM_FALSE_RE = re.compile(r"\baxiom\s+\w+\s*:\s*False\b", re.I)
AXIOM_RE = re.compile(r"^\s*axiom\s+([A-Za-z_][A-Za-z0-9_']*)\b", re.M)
LOCAL_MISSING_RE = re.compile(r"\bh_missing_[A-Za-z0-9_']*\b")
WRAP_DIAGNOSTIC_RE = re.compile(r"(不成立|错误|有误|正确(的)?(版本|分解|结论)|反例|一般不|不能作为)")


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


def read_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YAML config requires PyYAML, or use a .json config file") from exc
    return yaml.safe_load(text) or {}


def cfg_get(config: dict[str, Any], dotted: str, default: Any) -> Any:
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def parse_reasoning(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "auto", "default", "none", "null"}:
        return None
    if text in {"true", "1", "yes", "y", "on", "enabled", "enable"}:
        return True
    if text in {"false", "0", "no", "n", "off", "disabled", "disable"}:
        return False
    raise ValueError(f"invalid reasoning value: {value!r}")


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8").strip()


def normalize_problem(row: dict[str, Any], idx: int) -> dict[str, str]:
    pid = str(row.get("id") or row.get("problem_id") or f"problem_{idx:04d}")
    question = row.get("question") or row.get("problem") or row.get("text")
    if not question:
        raise ValueError(f"Input row {idx} has no question/problem/text field")
    return {"id": pid, "question": str(question)}


def default_lean_project_dir() -> str:
    env_dir = os.environ.get("LEAN_PROJECT_DIR")
    if env_dir:
        return env_dir
    local_mathlib = Path("/root/mathlib4")
    if (local_mathlib / "lakefile.lean").exists() or (local_mathlib / "lakefile.toml").exists():
        return str(local_mathlib)
    return str(REPO_ROOT / "lean_fhis")


def existing_default_config() -> str | None:
    json_path = ROOT / "configs" / "default.json"
    if json_path.exists():
        return str(json_path)
    yaml_path = ROOT / "configs" / "default.yaml"
    return str(yaml_path) if yaml_path.exists() else None


def mock_cot(problem: dict[str, str], chain_index: int) -> str:
    q = problem["question"]
    return (
        f"[Step 1] 重述题目目标：{q[:120]}\n"
        "[Step 2] 引入相关量，并把已知条件翻译成等式或不等式。\n"
        "[Step 3] 合并这些关系，得到题目要求的量。\n"
        "[Final Answer] 模拟答案\n"
    )


def generate_cot(
    problems: list[dict[str, str]],
    *,
    provider: str,
    chains: int,
    model: str | None,
    mock: bool,
    max_workers: int,
    llm_timeout: int,
    cot_max_tokens: int,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
    codex_reasoning_effort: str,
    codex_sandbox: str,
    codex_cwd: str,
) -> list[dict[str, Any]]:
    prompt = load_prompt("cot_prompt.md")
    outputs: list[dict[str, Any]] = []

    for problem in problems:
        if mock:
            problem_chains = [
                {
                    "chain_index": i + 1,
                    "model": "mock",
                    "text": mock_cot(problem, i + 1),
                }
                for i in range(chains)
            ]
        else:
            tasks = []
            for i in range(chains):
                tasks.append(
                    {
                        "system_prompt": prompt,
                        "user_prompt": (
                            "题目：\n"
                            f"{problem['question']}\n\n"
                            "请只生成一条推理链。使用 [Step N] 标记，并以 [Final Answer] 结束。"
                            "除固定标记外，所有内容都用中文。"
                        ),
                        "provider": provider,
                        "model": model,
                        "temperature": 0.7,
                        "max_tokens": cot_max_tokens,
                        "timeout": llm_timeout,
                        "retries": 2,
                        "reasoning": reasoning,
                        "openai_reasoning_effort": openai_reasoning_effort,
                        "codex_reasoning_effort": codex_reasoning_effort,
                        "codex_sandbox": codex_sandbox,
                        "codex_cwd": codex_cwd,
                        "call_label": f"{problem['id']}-cot-{i + 1}",
                    }
                )
            texts = call_llm_batch(tasks, max_workers=min(max_workers, chains))
            problem_chains = [
                {"chain_index": i + 1, "model": model or f"{provider}:default", "text": text}
                for i, text in enumerate(texts)
            ]

        outputs.append(
            {
                "id": problem["id"],
                "question": problem["question"],
                "chains": problem_chains,
            }
        )
    return outputs


def parse_chain_text(text: str) -> tuple[list[dict[str, Any]], str | None]:
    steps: list[dict[str, Any]] = []
    matches = list(BRACKET_STEP_RE.finditer(text))
    if matches:
        for match in matches:
            steps.append({"step_index": int(match.group(1)), "text": match.group(2).strip()})
        final_match = BRACKET_FINAL_RE.search(text)
        return steps, final_match.group(1).strip() if final_match else None

    matches = list(COLON_STEP_RE.finditer(text))
    if matches:
        for match in matches:
            steps.append({"step_index": int(match.group(1)), "text": match.group(2).strip()})
        final_match = COLON_FINAL_RE.search(text)
        return steps, final_match.group(1).strip() if final_match else None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [{"step_index": i + 1, "text": line} for i, line in enumerate(lines)], None


def parse_cot_outputs(cot_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for item in cot_outputs:
        for chain in item.get("chains", []):
            steps, final_answer = parse_chain_text(chain.get("text", ""))
            parsed.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "chain_index": chain["chain_index"],
                    "model": chain.get("model"),
                    "original_text": chain.get("text", ""),
                    "steps": steps,
                    "final_answer": final_answer,
                }
            )
    return parsed


def select_steps(
    parsed_chains: list[dict[str, Any]],
    *,
    k: int,
    seed: int,
    include_final_answer: bool,
    context_before: int,
    context_after: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for chain in parsed_chains:
        steps = chain.get("steps", [])
        if not steps:
            continue
        chosen = rng.sample(steps, min(k, len(steps)))
        step_by_index = {step["step_index"]: step for step in steps}
        ordered_indices = sorted(step_by_index)
        for step in sorted(chosen, key=lambda s: s["step_index"]):
            selected_pos = ordered_indices.index(step["step_index"])
            context_indices = ordered_indices[
                max(0, selected_pos - context_before) : selected_pos + context_after + 1
            ]
            previous_steps = [
                step_by_index[i]["text"]
                for i in ordered_indices
                if i < step["step_index"]
            ]
            row = {
                "id": chain["id"],
                "question": chain["question"],
                "chain_id": chain["chain_index"],
                "step_id": step["step_index"],
                "target_step": step["text"],
                "previous_steps": previous_steps,
                "context_steps": [
                    {
                        "step_id": i,
                        "text": step_by_index[i]["text"],
                        "is_selected": i == step["step_index"],
                    }
                    for i in context_indices
                ],
                "final_answer": chain.get("final_answer") if include_final_answer else None,
            }
            selected.append(row)
    return selected


def build_selected_step_row(
    chain: dict[str, Any],
    step: dict[str, Any],
    *,
    include_final_answer: bool,
    context_before: int,
    context_after: int,
    selection_score: dict[str, Any] | None = None,
) -> dict[str, Any]:
    steps = chain.get("steps", [])
    step_by_index = {row["step_index"]: row for row in steps}
    ordered_indices = sorted(step_by_index)
    selected_pos = ordered_indices.index(step["step_index"])
    context_indices = ordered_indices[
        max(0, selected_pos - context_before) : selected_pos + context_after + 1
    ]
    previous_steps = [
        step_by_index[i]["text"]
        for i in ordered_indices
        if i < step["step_index"]
    ]
    row = {
        "id": chain["id"],
        "question": chain["question"],
        "chain_id": chain["chain_index"],
        "step_id": step["step_index"],
        "target_step": step["text"],
        "previous_steps": previous_steps,
        "context_steps": [
            {
                "step_id": i,
                "text": step_by_index[i]["text"],
                "is_selected": i == step["step_index"],
            }
            for i in context_indices
        ],
        "final_answer": chain.get("final_answer") if include_final_answer else None,
    }
    if selection_score:
        row["selection_score"] = selection_score
    return row


def heuristic_step_score(step: dict[str, Any]) -> dict[str, Any]:
    text = str(step.get("text", ""))
    has_math_symbol = bool(re.search(r"[=<>∣|≤≥≡∑∏√^_{}\\]", text))
    has_claim_word = any(
        word in text
        for word in [
            "得到",
            "推出",
            "因此",
            "所以",
            "必须",
            "等于",
            "整除",
            "同余",
            "不等式",
            "最大",
            "最小",
            "构造",
            "归纳",
        ]
    )
    low_value = any(
        phrase in text
        for phrase in ["重述", "设", "记", "题目要求", "题目目标", "我们需要", "尝试", "考虑", "先"]
    ) and not (has_math_symbol and has_claim_word)
    verification_value = 4 if has_math_symbol and has_claim_word else 2
    risk = 4 if any(word in text for word in ["显然", "必然", "所有", "任意", "存在", "唯一"]) else 2
    feasibility = 4 if has_math_symbol else 2
    if low_value:
        verification_value = min(verification_value, 2)
        risk = min(risk, 2)
    return {
        "step_id": int(step["step_index"]),
        "is_mathematical_claim": bool(has_math_symbol or has_claim_word),
        "low_value": bool(low_value),
        "verification_value": verification_value,
        "risk": risk,
        "lean_feasibility": feasibility,
        "reason": "启发式评分：根据数学符号、推理关键词和低价值短语估计。",
    }


def normalize_step_score(raw: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_step_score(step)
    score: dict[str, Any] = {
        "step_id": int(step["step_index"]),
        "is_mathematical_claim": bool(raw.get("is_mathematical_claim", fallback["is_mathematical_claim"])),
        "low_value": bool(raw.get("low_value", fallback["low_value"])),
        "reason": str(raw.get("reason") or fallback["reason"]).strip(),
    }
    for key in ["verification_value", "risk", "lean_feasibility"]:
        try:
            value = int(raw.get(key, fallback[key]))
        except Exception:
            value = int(fallback[key])
        score[key] = max(1, min(5, value))
    score["combined_score"] = (
        score["verification_value"] * score["risk"] + score["lean_feasibility"]
    )
    if score["low_value"] or not score["is_mathematical_claim"]:
        score["combined_score"] -= 20
    return score


def build_step_score_prompt(chain: dict[str, Any]) -> str:
    steps_text = "\n".join(
        f"{step['step_index']}. {step['text']}"
        for step in chain.get("steps", [])
    )
    final = chain.get("final_answer") or "(unknown)"
    return (
        f"题目：\n{chain['question']}\n\n"
        f"CoT 步骤：\n{steps_text}\n\n"
        f"最终答案：\n{final}\n"
    )


def mock_step_scores(chain: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": [
            heuristic_step_score(step)
            for step in chain.get("steps", [])
        ]
    }


def validate_step_scores(record: dict[str, Any], chain: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    steps = chain.get("steps", [])
    expected = {int(step["step_index"]) for step in steps}
    raw_scores = record.get("steps")
    if not isinstance(raw_scores, list):
        return ["steps must be a list"]
    seen: set[int] = set()
    for idx, row in enumerate(raw_scores):
        if not isinstance(row, dict):
            errors.append(f"score row {idx} must be an object")
            continue
        try:
            step_id = int(row.get("step_id"))
        except Exception:
            errors.append(f"score row {idx} missing integer step_id")
            continue
        if step_id not in expected:
            errors.append(f"score row {idx} has unknown step_id {step_id}")
        seen.add(step_id)
        for key in ["verification_value", "risk", "lean_feasibility"]:
            try:
                value = int(row.get(key))
            except Exception:
                errors.append(f"step {step_id} missing integer {key}")
                continue
            if value < 1 or value > 5:
                errors.append(f"step {step_id} {key} must be between 1 and 5")
        if not isinstance(row.get("is_mathematical_claim"), bool):
            errors.append(f"step {step_id} is_mathematical_claim must be boolean")
        if not isinstance(row.get("low_value"), bool):
            errors.append(f"step {step_id} low_value must be boolean")
    missing = sorted(expected - seen)
    if missing:
        errors.append(f"missing scores for step_id(s): {missing}")
    return errors


def score_cot_steps(
    parsed_chains: list[dict[str, Any]],
    *,
    provider: str,
    model: str | None,
    mock: bool,
    out_dir: Path,
    llm_timeout: int,
    score_max_tokens: int,
    max_workers: int,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
    codex_reasoning_effort: str,
    codex_sandbox: str,
    codex_cwd: str,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = load_prompt("score_cot_steps.md")
    score_records: list[dict[str, Any]] = []

    chains = [chain for chain in parsed_chains if chain.get("steps")]
    if mock:
        raw_texts = [
            json.dumps(mock_step_scores(chain), ensure_ascii=False, indent=2)
            for chain in chains
        ]
    else:
        tasks = [
            {
                "system_prompt": system_prompt,
                "user_prompt": build_step_score_prompt(chain),
                "provider": provider,
                "model": model,
                "temperature": 0.0,
                "max_tokens": score_max_tokens,
                "timeout": llm_timeout,
                "retries": 1,
                "reasoning": reasoning,
                "openai_reasoning_effort": openai_reasoning_effort,
                "codex_reasoning_effort": codex_reasoning_effort,
                "codex_sandbox": codex_sandbox,
                "codex_cwd": codex_cwd,
                "call_label": f"{chain['id']}-c{chain['chain_index']}-score",
            }
            for chain in chains
        ]
        raw_texts = call_llm_batch(tasks, max_workers=max_workers)

    for chain, raw in zip(chains, raw_texts):
        file_stem = safe_record_name(
            {"id": chain["id"], "chain_id": chain["chain_index"], "step_id": "scores"}
        )
        raw_path = out_dir / f"{file_stem}.response.txt"
        validation_path = out_dir / f"{file_stem}.validation.json"
        raw_path.write_text(raw, encoding="utf-8")
        try:
            parsed = extract_json_object(raw)
            validation_errors = validate_step_scores(parsed, chain)
        except Exception as exc:
            parsed = mock_step_scores(chain)
            validation_errors = [str(exc)]
        if validation_errors:
            parsed = mock_step_scores(chain)
        step_by_id = {int(step["step_index"]): step for step in chain.get("steps", [])}
        normalized = [
            normalize_step_score(row, step_by_id[int(row.get("step_id", -1))])
            for row in parsed.get("steps", [])
            if int(row.get("step_id", -1)) in step_by_id
        ]
        normalized_by_id = {row["step_id"]: row for row in normalized}
        for step in chain.get("steps", []):
            step_id = int(step["step_index"])
            if step_id not in normalized_by_id:
                normalized_by_id[step_id] = normalize_step_score({}, step)
        score_rows = [normalized_by_id[i] for i in sorted(normalized_by_id)]
        write_json({"ok": not validation_errors, "errors": validation_errors}, validation_path)
        score_records.append(
            {
                "id": chain["id"],
                "question": chain["question"],
                "chain_id": chain["chain_index"],
                "model": chain.get("model"),
                "final_answer": chain.get("final_answer"),
                "scores": score_rows,
                "score_response_file": str(raw_path),
                "score_validation_file": str(validation_path),
                "score_valid": not validation_errors,
                "score_errors": validation_errors,
            }
        )
    return score_records


def select_steps_by_value(
    parsed_chains: list[dict[str, Any]],
    score_records: list[dict[str, Any]],
    *,
    k: int,
    include_final_answer: bool,
    context_before: int,
    context_after: int,
) -> list[dict[str, Any]]:
    chain_by_key = {(chain["id"], chain["chain_index"]): chain for chain in parsed_chains}
    selected: list[dict[str, Any]] = []
    for record in score_records:
        chain = chain_by_key.get((record["id"], record["chain_id"]))
        if not chain:
            continue
        steps_by_id = {int(step["step_index"]): step for step in chain.get("steps", [])}
        ranked_scores = sorted(
            record.get("scores", []),
            key=lambda row: (
                row.get("combined_score", -999),
                row.get("verification_value", 0),
                row.get("risk", 0),
                row.get("lean_feasibility", 0),
            ),
            reverse=True,
        )
        chosen = [row for row in ranked_scores if int(row.get("step_id", -1)) in steps_by_id][:k]
        for score in chosen:
            step = steps_by_id[int(score["step_id"])]
            selected.append(
                build_selected_step_row(
                    chain,
                    step,
                    include_final_answer=include_final_answer,
                    context_before=context_before,
                    context_after=context_after,
                    selection_score=score,
                )
            )
    return selected


def extract_lean_code(text: str) -> str:
    match = LEAN_BLOCK_RE.search(text)
    code = match.group(1) if match else text
    code = code.strip()
    lines = code.splitlines()
    if not any(line.strip().startswith("import ") for line in lines):
        lines.insert(0, "import Mathlib")
    return "\n".join(lines).strip() + "\n"


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


def safe_lean_name(item: dict[str, Any]) -> str:
    raw = f"step_contract_{item['id']}_c{item['chain_id']}_s{item['step_id']}"
    name = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if not re.match(r"[A-Za-z_]", name):
        name = "step_contract_" + name
    return name[:180]


def safe_record_name(item: dict[str, Any]) -> str:
    return safe_lean_name(item).removeprefix("step_contract_")


def build_wrap_prompt(item: dict[str, Any]) -> str:
    context = item.get("context_steps") or [
        {"step_id": idx + 1, "text": text, "is_selected": False}
        for idx, text in enumerate(item.get("previous_steps") or [])
    ]
    if not any(row.get("is_selected") for row in context):
        context.append(
            {
                "step_id": item["step_id"],
                "text": item["target_step"],
                "is_selected": True,
            }
        )
    context_text = "\n".join(
        f"{row['step_id']}. {'[选中] ' if row.get('is_selected') else ''}{row['text']}"
        for row in context
    )
    final = item.get("final_answer") or "(unknown)"
    return (
        f"题目：\n{item['question']}\n\n"
        f"候选 CoT 上下文：\n{context_text}\n\n"
        f"选中的 CoT 步骤序号：{item['step_id']}\n"
        f"选中的 CoT 步骤原文：{item['target_step']}\n\n"
        f"最终答案：\n{final}\n"
    )


def mock_wrapped_claim(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "low_value": False,
        "low_value_reason": "",
        "used_cot": [
            {
                "step_id": item["step_id"],
                "is_selected": True,
                "text": item["target_step"],
                "role": "conclusion",
            }
        ],
        "wrapped_claim": {
            "premises": [
                {
                    "text": "题目与前序步骤中的相关条件成立。",
                    "source": "problem",
                }
            ],
            "conclusion": {
                "text": item["target_step"],
                "source": f"cot_step_{item['step_id']}",
            },
            "proof_description": "这是 mock 包装：把选中步骤作为局部结论，相关条件作为局部前提。",
        },
    }


def validate_wrapped_record(record: dict[str, Any], item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(record.get("low_value"), bool):
        errors.append("low_value must be a boolean")
    used_cot = record.get("used_cot")
    if not isinstance(used_cot, list) or not used_cot:
        errors.append("used_cot must be a non-empty list")
    else:
        selected = [row for row in used_cot if row.get("is_selected") is True]
        if len(selected) != 1:
            errors.append("used_cot must contain exactly one selected step")
        elif int(selected[0].get("step_id", -1)) != int(item["step_id"]):
            errors.append("selected used_cot step_id does not match target step")
        elif str(selected[0].get("text", "")).strip() != str(item["target_step"]).strip():
            errors.append("selected used_cot text must match target step exactly")

    wrapped = record.get("wrapped_claim")
    if not isinstance(wrapped, dict):
        errors.append("wrapped_claim must be an object")
        return errors
    premises = wrapped.get("premises")
    if not isinstance(premises, list) or not premises:
        errors.append("wrapped_claim.premises must be a non-empty list")
    else:
        for idx, premise in enumerate(premises):
            if not isinstance(premise, dict):
                errors.append(f"premise {idx} must be an object")
                continue
            if not str(premise.get("text", "")).strip():
                errors.append(f"premise {idx} missing text")
            source = str(premise.get("source", "")).strip()
            if not (
                source == "problem"
                or source == "standard_math"
                or re.fullmatch(r"cot_step_\d+", source)
            ):
                errors.append(f"premise {idx} has invalid source")
    conclusion = wrapped.get("conclusion")
    if not isinstance(conclusion, dict) or not str(conclusion.get("text", "")).strip():
        errors.append("wrapped_claim.conclusion must contain text")
    else:
        conclusion_text = str(conclusion.get("text", "")).strip()
        if WRAP_DIAGNOSTIC_RE.search(conclusion_text):
            errors.append("wrapped_claim.conclusion must assert the selected step, not diagnose or correct it")
        source = str(conclusion.get("source", "")).strip()
        if not (
            source == "problem"
            or source == "standard_math"
            or re.fullmatch(r"cot_step_\d+", source)
        ):
            errors.append("wrapped_claim.conclusion has invalid source")
    proof_description = str(wrapped.get("proof_description", "")).strip()
    if not proof_description:
        errors.append("wrapped_claim.proof_description is required")
    elif WRAP_DIAGNOSTIC_RE.search(proof_description):
        errors.append("wrapped_claim.proof_description must not diagnose, refute, or correct the selected step")
    return errors


def wrap_repair_prompt(raw: str, errors: list[str], item: dict[str, Any]) -> str:
    return (
        "上一次输出不是合格的 JSON 包装结果。请只返回修复后的合法 JSON。\n\n"
        f"选中步骤序号：{item['step_id']}\n"
        f"选中步骤原文：{item['target_step']}\n\n"
        "校验错误：\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\n\n上一次输出：\n"
        f"{raw[-6000:]}"
    )


def generate_wrapped_claims(
    selected_steps: list[dict[str, Any]],
    *,
    provider: str,
    model: str | None,
    mock: bool,
    out_dir: Path,
    llm_timeout: int,
    wrap_max_tokens: int,
    wrap_repair_rounds: int,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
    codex_reasoning_effort: str,
    codex_sandbox: str,
    codex_cwd: str,
) -> list[dict[str, Any]]:
    system_prompt = load_prompt("wrap_step_claim.md")
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for item in selected_steps:
        if mock:
            raw = json.dumps(mock_wrapped_claim(item), ensure_ascii=False, indent=2)
            parsed = mock_wrapped_claim(item)
            validation_errors: list[str] = []
        else:
            raw = call_llm(
                system_prompt=system_prompt,
                user_prompt=build_wrap_prompt(item),
                provider=provider,
                model=model,
                temperature=0.0,
                max_tokens=wrap_max_tokens,
                timeout=llm_timeout,
                retries=2,
                reasoning=reasoning,
                openai_reasoning_effort=openai_reasoning_effort,
                codex_reasoning_effort=codex_reasoning_effort,
                codex_sandbox=codex_sandbox,
                codex_cwd=codex_cwd,
                call_label=f"{item['id']}-c{item['chain_id']}-s{item['step_id']}-wrap",
            )
            parsed: dict[str, Any] = {}
            validation_errors = []
            for round_idx in range(wrap_repair_rounds + 1):
                try:
                    parsed = extract_json_object(raw)
                    validation_errors = validate_wrapped_record(parsed, item)
                except Exception as exc:
                    validation_errors = [str(exc)]
                if not validation_errors:
                    break
                if round_idx >= wrap_repair_rounds:
                    break
                raw = call_llm(
                    system_prompt=system_prompt,
                    user_prompt=wrap_repair_prompt(raw, validation_errors, item),
                    provider=provider,
                    model=model,
                    temperature=0.0,
                    max_tokens=wrap_max_tokens,
                    timeout=llm_timeout,
                    retries=1,
                    reasoning=reasoning,
                    openai_reasoning_effort=openai_reasoning_effort,
                    codex_reasoning_effort=codex_reasoning_effort,
                    codex_sandbox=codex_sandbox,
                    codex_cwd=codex_cwd,
                    call_label=f"{item['id']}-c{item['chain_id']}-s{item['step_id']}-wrap-repair-{round_idx + 1}",
                )

        file_stem = safe_record_name(item)
        raw_path = out_dir / f"{file_stem}.response.txt"
        validation_path = out_dir / f"{file_stem}.validation.json"
        raw_path.write_text(raw, encoding="utf-8")
        write_json({"ok": not validation_errors, "errors": validation_errors}, validation_path)
        if validation_errors:
            fallback = mock_wrapped_claim(item)
            fallback["low_value"] = True
            fallback["low_value_reason"] = "wrapper 输出未通过 JSON/结构校验，使用保底 wrapped_claim"
            parsed = fallback
        records.append(
            {
                **item,
                "low_value": bool(parsed.get("low_value")) if isinstance(parsed, dict) else True,
                "low_value_reason": parsed.get("low_value_reason", "") if isinstance(parsed, dict) else "",
                "used_cot": parsed.get("used_cot", []) if isinstance(parsed, dict) else [],
                "wrapped_claim": parsed.get("wrapped_claim", {}) if isinstance(parsed, dict) else {},
                "wrap_response_file": str(raw_path),
                "wrap_validation_file": str(validation_path),
                "wrap_valid": not validation_errors,
                "wrap_errors": validation_errors,
            }
        )
    return records


def build_lean_prompt(item: dict[str, Any], theorem_name: str) -> str:
    if item.get("wrapped_claim"):
        template = load_prompt("lean_wrapped_claim.md")
        return template.format(
            THEOREM_NAME=theorem_name,
            WRAPPED_CLAIM_JSON=json.dumps(item["wrapped_claim"], ensure_ascii=False, indent=2),
        )

    previous = "\n".join(
        f"{idx + 1}. {text}" for idx, text in enumerate(item.get("previous_steps") or [])
    )
    if not previous:
        previous = "(none)"
    final = item.get("final_answer") or "(unknown)"
    template = load_prompt("lean_step_contract.md")
    return template.format(
        THEOREM_NAME=theorem_name,
        QUESTION=item["question"],
        PREVIOUS_STEPS=previous,
        TARGET_STEP=item["target_step"],
        FINAL_ANSWER=final,
    )


def mock_lean_code(theorem_name: str) -> str:
    return (
        "import Mathlib\n\n"
        "namespace SingleStepFormalization\n\n"
        f"theorem {theorem_name} : True := by\n"
        "  trivial\n\n"
        "end SingleStepFormalization\n"
    )


def lean_repair_prompt(code: str, error_text: str) -> str:
    return (
        "下面的 Lean 代码没有通过编译。请修复它。\n\n"
        "要求：\n"
        "- 只返回一个 Lean 代码块，包含完整文件。\n"
        "- 保持 theorem 名字和目标语义不变。\n"
        "- 不使用 sorry、admit。\n"
        "- warning 不需要处理，只修复 error。\n"
        "- 如果缺失必要前提，优先加入局部 h_missing_* 假设；必要时可用具体的 obligation_* axiom。\n\n"
        "原代码：\n"
        f"```lean\n{code}\n```\n\n"
        "Lean 错误：\n"
        f"```text\n{error_text[-6000:]}\n```"
    )


def generate_lean_contracts(
    selected_steps: list[dict[str, Any]],
    *,
    provider: str,
    model: str | None,
    mock: bool,
    out_dir: Path,
    llm_timeout: int,
    lean_max_tokens: int,
    project_dir: Path,
    lean_timeout: int,
    repair_rounds: int,
    skip_lean_check: bool,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
    codex_reasoning_effort: str,
    codex_sandbox: str,
    codex_cwd: str,
) -> list[dict[str, Any]]:
    system_prompt = (
        "你是 Lean 4 形式化工程师。只返回一个 Lean 代码块。"
        "请把 wrapped_claim 形式化为局部 transition contract。"
        "Lean theorem 必须表达前提推出结论。"
        "不允许使用 sorry、admit。优先不用 axiom；如果需要，请优先写成 theorem 的局部 h_missing_* 假设。"
        "复杂高层标准数学概念如果 mathlib 名称不明确或接口成本太高，可以先定义清晰的局部谓词/结构作为接口。"
        "需要高层定理或库里难以找到的接口时，优先写成 theorem 的局部 h_missing_* 假设。"
        "只有在局部接口难以表达时，才允许使用具体、窄范围、可读的 obligation_* axiom，且不能直接断言最终结论。"
        "如果 Lean 代码里需要注释，注释请用中文。"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for item in selected_steps:
        theorem_name = safe_lean_name(item)
        if mock:
            raw = f"```lean\n{mock_lean_code(theorem_name)}```"
            code = mock_lean_code(theorem_name)
            repair_history: list[dict[str, Any]] = []
        else:
            user_prompt = build_lean_prompt(item, theorem_name)
            raw = call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
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
                call_label=f"{item['id']}-c{item['chain_id']}-s{item['step_id']}-lean",
            )
            code = extract_lean_code(raw)
            repair_history = []

            if not skip_lean_check:
                for round_idx in range(repair_rounds):
                    check = check_lean_code(
                        code,
                        project_dir=project_dir,
                        timeout=lean_timeout,
                        theorem_name=theorem_name,
                        stem=f"{safe_lean_name(item)}_repair_probe_{round_idx}",
                    )
                    repair_history.append(
                        {
                            "round": round_idx,
                            "ok": check.get("ok"),
                            "stderr": check.get("stderr", ""),
                            "stdout": check.get("stdout", ""),
                            "dependency_mode": check.get("dependency_mode"),
                        }
                    )
                    if check.get("ok") is True:
                        break
                    error_text = (check.get("stdout") or "") + "\n" + (check.get("stderr") or "")
                    raw = call_llm(
                        system_prompt=system_prompt,
                        user_prompt=lean_repair_prompt(code, error_text),
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
                        call_label=(
                            f"{item['id']}-c{item['chain_id']}-s{item['step_id']}"
                            f"-repair-{round_idx + 1}"
                        ),
                    )
                    code = extract_lean_code(raw)

        file_stem = safe_lean_name(item)
        lean_path = out_dir / f"{file_stem}.lean"
        raw_path = out_dir / f"{file_stem}.response.txt"
        repair_path = out_dir / f"{file_stem}.repair_history.json"
        lean_path.write_text(code, encoding="utf-8")
        raw_path.write_text(raw, encoding="utf-8")
        write_json(repair_history, repair_path)
        records.append(
            {
                **item,
                "theorem_name": theorem_name,
                "lean_file": str(lean_path),
                "response_file": str(raw_path),
                "repair_history_file": str(repair_path),
                "repair_rounds_used": len([r for r in repair_history if r.get("ok") is not True]),
            }
        )
    return records


def guess_theorem_full_name(code: str, theorem_name: str) -> str:
    namespace_stack: list[str] = []
    for line in code.splitlines():
        ns_match = re.match(r"\s*namespace\s+([A-Za-z0-9_.'\s]+)\s*$", line)
        if ns_match:
            namespace_stack.extend(ns_match.group(1).split())
            continue
        theorem_match = re.match(rf"\s*theorem\s+{re.escape(theorem_name)}\b", line)
        if theorem_match:
            return ".".join([*namespace_stack, theorem_name]) if namespace_stack else theorem_name
        end_match = re.match(r"\s*end(?:\s+[A-Za-z0-9_.'\s]+)?\s*$", line)
        if end_match and namespace_stack:
            namespace_stack.pop()
    return theorem_name


def parse_print_axioms_output(output: str) -> list[str]:
    match = re.search(r"depends on axioms:\s*\[([^\]]*)\]", output)
    if not match:
        return []
    return [part.strip() for part in match.group(1).split(",") if part.strip()]


def run_lean_command(code: str, *, project_dir: Path, timeout: int, stem: str) -> dict[str, Any]:
    tmp_dir = project_dir / ".single_step_formalization_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{stem}_{uuid.uuid4().hex[:8]}.lean"
    tmp_path.write_text(code, encoding="utf-8")
    start = time.time()
    try:
        proc = subprocess.Popen(
            ["lake", "env", "lean", str(tmp_path)],
            cwd=str(project_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            return {
                "ok": False,
                "returncode": None,
                "stdout": stdout,
                "stderr": f"timeout after {timeout}s\n{stderr}",
                "elapsed": time.time() - start,
            }
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed": time.time() - start,
        }
    except FileNotFoundError:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "lake command not found",
            "elapsed": time.time() - start,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


def check_lean_code(
    code: str,
    *,
    project_dir: Path,
    timeout: int,
    theorem_name: str | None = None,
    stem: str = "lean_check",
) -> dict[str, Any]:
    has_banned = bool(BANNED_LEAN_RE.search(code))
    has_false_axiom = bool(AXIOM_FALSE_RE.search(code))
    declared_axioms = AXIOM_RE.findall(code)
    local_missing = sorted(set(LOCAL_MISSING_RE.findall(code)))
    missing_theorem = bool(theorem_name) and not re.search(
        rf"\btheorem\s+{re.escape(theorem_name or '')}\b", code
    )

    if has_banned or has_false_axiom or missing_theorem:
        if missing_theorem:
            reason = f"Lean code does not declare theorem {theorem_name}"
        else:
            reason = "Lean code contains sorry/admit or a False axiom"
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": reason,
            "elapsed": 0.0,
            "declared_axioms": declared_axioms,
            "local_missing_hypotheses": local_missing,
            "dependency_mode": "invalid",
        }

    check = run_lean_command(code, project_dir=project_dir, timeout=timeout, stem=stem)
    result = {
        **check,
        "declared_axioms": declared_axioms,
        "local_missing_hypotheses": local_missing,
        "dependency_mode": "complete",
    }
    if declared_axioms:
        result["dependency_mode"] = "global_axiom_fallback"
    elif local_missing:
        result["dependency_mode"] = "local_missing_hypotheses"

    if check["ok"] and theorem_name:
        full_name = guess_theorem_full_name(code, theorem_name)
        axiom_probe_code = code + f"\n\n#print axioms {full_name}\n"
        axiom_probe = run_lean_command(
            axiom_probe_code,
            project_dir=project_dir,
            timeout=timeout,
            stem=f"{stem}_axioms",
        )
        raw_report = (axiom_probe.get("stdout") or "") + (axiom_probe.get("stderr") or "")
        result["theorem_full_name"] = full_name
        result["print_axioms_ok"] = axiom_probe["ok"]
        result["print_axioms_raw"] = raw_report
        result["kernel_axioms"] = parse_print_axioms_output(raw_report)
        if not axiom_probe["ok"]:
            result["ok"] = False
            result["dependency_mode"] = "invalid"
            result["stderr"] = (
                (result.get("stderr") or "")
                + "\n#print axioms failed; theorem may be missing or inaccessible:\n"
                + raw_report
            ).strip()
    return result


def run_lean_file(
    path: Path,
    *,
    project_dir: Path,
    timeout: int,
    theorem_name: str | None = None,
) -> dict[str, Any]:
    code = path.read_text(encoding="utf-8")
    result = check_lean_code(
        code,
        project_dir=project_dir,
        timeout=timeout,
        theorem_name=theorem_name,
        stem=path.stem,
    )
    return {"file": str(path), **result}


def verify_lean_outputs(
    generated: list[dict[str, Any]],
    *,
    project_dir: Path,
    timeout: int,
    skip: bool,
) -> list[dict[str, Any]]:
    if skip:
        return [
            {"file": item["lean_file"], "ok": None, "skipped": True, "stderr": "verification skipped"}
            for item in generated
        ]
    return [
        run_lean_file(
            Path(item["lean_file"]),
            project_dir=project_dir,
            timeout=timeout,
            theorem_name=item.get("theorem_name"),
        )
        for item in generated
    ]


def write_run_summary(
    path: Path,
    *,
    problems: list[dict[str, Any]],
    parsed: list[dict[str, Any]],
    selected: list[dict[str, Any]] | None = None,
    step_selector: str,
    step_scores: list[dict[str, Any]] | None = None,
    wrapped: list[dict[str, Any]] | None = None,
    generated: list[dict[str, Any]] | None = None,
    verification: list[dict[str, Any]] | None = None,
    stopped_after: str = "verification",
) -> dict[str, Any]:
    selected = selected or []
    step_scores = step_scores or []
    wrapped = wrapped or []
    generated = generated or []
    verification = verification or []
    summary = {
        "stopped_after": stopped_after,
        "problems": len(problems),
        "chains": len(parsed),
        "parsed_steps": sum(len(chain.get("steps", [])) for chain in parsed),
        "selected_steps": len(selected),
        "step_selector": step_selector,
        "scored_chains": len(step_scores),
        "score_valid": sum(1 for row in step_scores if row.get("score_valid") is True),
        "score_invalid": sum(1 for row in step_scores if row.get("score_valid") is not True),
        "wrapped_claims": len(wrapped),
        "wrap_valid": sum(1 for row in wrapped if row.get("wrap_valid") is True),
        "wrap_invalid": sum(1 for row in wrapped if row.get("wrap_valid") is not True),
        "low_value_steps": sum(1 for row in wrapped if row.get("low_value") is True),
        "lean_files": len(generated),
        "verified_ok": sum(1 for row in verification if row.get("ok") is True),
        "verified_failed": sum(1 for row in verification if row.get("ok") is False),
        "verified_skipped": sum(1 for row in verification if row.get("ok") is None),
        "complete_proofs": sum(
            1
            for row in verification
            if row.get("ok") is True and row.get("dependency_mode") == "complete"
        ),
        "local_missing_hypotheses": sum(
            1
            for row in verification
            if row.get("ok") is True and row.get("dependency_mode") == "local_missing_hypotheses"
        ),
        "global_axiom_fallbacks": sum(
            1
            for row in verification
            if row.get("ok") is True and row.get("dependency_mode") == "global_axiom_fallback"
        ),
    }
    write_json(summary, path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=existing_default_config(), help="YAML/JSON config path")
    parser.add_argument("--input", required=True, help="JSONL with id/question, problem, or text")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--chains", type=int, default=None, help="CoT traces per problem")
    parser.add_argument("--sample-steps", type=int, default=None, help="random target steps per chain")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "codex"],
        default=None,
        help="LLM backend: openai-compatible API or local Codex CLI",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true", help="no API calls; generate deterministic toy outputs")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--llm-timeout", type=int, default=None, help="single LLM call timeout in seconds")
    parser.add_argument("--cot-max-tokens", type=int, default=None)
    parser.add_argument("--wrap-max-tokens", type=int, default=None)
    parser.add_argument("--lean-max-tokens", type=int, default=None)
    parser.add_argument("--score-max-tokens", type=int, default=None)
    parser.add_argument(
        "--step-selector",
        choices=["random", "value"],
        default=None,
        help="random samples CoT steps; value asks the LLM to score verification value/risk",
    )
    parser.add_argument("--wrap-repair-rounds", type=int, default=None)
    parser.add_argument("--wrap-context-before", type=int, default=None)
    parser.add_argument("--wrap-context-after", type=int, default=None)
    parser.add_argument(
        "--reasoning",
        choices=["auto", "enabled", "disabled"],
        default=None,
        help="OpenAI-compatible thinking/reasoning switch; auto omits provider-specific parameter",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=["high", "max"],
        default=None,
        help="OpenAI-compatible reasoning_effort, e.g. DeepSeek high/max",
    )
    parser.add_argument("--codex-reasoning-effort", default=None)
    parser.add_argument("--codex-sandbox", default=None)
    parser.add_argument("--codex-cwd", default=None)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--lean-timeout", type=int, default=None)
    parser.add_argument("--repair-rounds", type=int, default=None)
    parser.add_argument("--skip-lean-check", action="store_true")
    parser.add_argument("--include-final-answer", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=["cot", "selection", "wrapped", "lean", "verification"],
        default=None,
        help="stop after a pipeline stage and write partial outputs/summary",
    )
    args = parser.parse_args()
    config = read_config(args.config)

    chains = args.chains if args.chains is not None else int(cfg_get(config, "run.chains", 3))
    sample_steps = (
        args.sample_steps if args.sample_steps is not None else int(cfg_get(config, "run.sample_steps", 2))
    )
    seed = args.seed if args.seed is not None else int(cfg_get(config, "run.seed", 42))
    llm_provider = args.llm_provider or os.environ.get("LLM_PROVIDER") or cfg_get(config, "llm.provider", "codex")
    max_workers = args.max_workers if args.max_workers is not None else int(cfg_get(config, "llm.max_workers", 4))
    llm_timeout = args.llm_timeout if args.llm_timeout is not None else int(cfg_get(config, "llm.timeout", 900))
    cot_max_tokens = args.cot_max_tokens if args.cot_max_tokens is not None else int(cfg_get(config, "llm.cot_max_tokens", 2048))
    wrap_max_tokens = args.wrap_max_tokens if args.wrap_max_tokens is not None else int(cfg_get(config, "llm.wrap_max_tokens", 4096))
    lean_max_tokens = args.lean_max_tokens if args.lean_max_tokens is not None else int(cfg_get(config, "llm.lean_max_tokens", 4096))
    score_max_tokens = args.score_max_tokens if args.score_max_tokens is not None else int(cfg_get(config, "llm.score_max_tokens", 8192))
    step_selector = args.step_selector or cfg_get(config, "run.step_selector", "random")
    stop_after = args.stop_after or cfg_get(config, "run.stop_after", "verification")
    reasoning = parse_reasoning(args.reasoning if args.reasoning is not None else cfg_get(config, "llm.reasoning", None))
    openai_reasoning_effort = args.openai_reasoning_effort or cfg_get(config, "llm.openai_reasoning_effort", None)
    codex_reasoning_effort = (
        args.codex_reasoning_effort
        or os.environ.get("CODEX_REASONING_EFFORT")
        or cfg_get(config, "llm.codex_reasoning_effort", "high")
    )
    codex_sandbox = args.codex_sandbox or os.environ.get("CODEX_SANDBOX") or cfg_get(config, "llm.codex_sandbox", "danger-full-access")
    codex_cwd = args.codex_cwd or cfg_get(config, "llm.codex_cwd", str(REPO_ROOT))
    project_dir = args.project_dir or cfg_get(config, "paths.lean_project_dir", default_lean_project_dir())
    lean_timeout = args.lean_timeout if args.lean_timeout is not None else int(cfg_get(config, "lean.timeout", 120))
    repair_rounds = args.repair_rounds if args.repair_rounds is not None else int(cfg_get(config, "lean.repair_rounds", 3))
    wrap_repair_rounds = (
        args.wrap_repair_rounds
        if args.wrap_repair_rounds is not None
        else int(cfg_get(config, "run.wrap_repair_rounds", 2))
    )
    wrap_context_before = (
        args.wrap_context_before
        if args.wrap_context_before is not None
        else int(cfg_get(config, "run.wrap_context_before", 5))
    )
    wrap_context_after = (
        args.wrap_context_after
        if args.wrap_context_after is not None
        else int(cfg_get(config, "run.wrap_context_after", 2))
    )
    skip_lean_check = args.skip_lean_check or bool(cfg_get(config, "lean.skip_check", False))
    include_final_answer = args.include_final_answer or bool(cfg_get(config, "run.include_final_answer", False))

    run_dir = RUNS_DIR / args.run_id
    input_dir = run_dir / "input"
    cot_dir = run_dir / "cot"
    selection_dir = run_dir / "selection"
    scores_dir = selection_dir / "scores"
    wrapped_dir = run_dir / "wrapped_claims"
    lean_dir = run_dir / "lean"
    verification_dir = run_dir / "verification"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_model = cfg_get(config, "llm.model", None)
    if llm_provider == "codex":
        model = args.model or os.environ.get("CODEX_MODEL") or os.environ.get("OPENAI_MODEL")
    else:
        model = args.model or config_model or os.environ.get("OPENAI_MODEL") or "gpt-4o"
    if llm_provider == "codex" and args.model is None and config_model:
        model = config_model
    raw_rows = read_jsonl(Path(args.input))
    problems = [normalize_problem(row, i) for i, row in enumerate(raw_rows)]
    write_jsonl(problems, input_dir / "problems.jsonl")

    config = {
        "run_id": args.run_id,
        "config": args.config,
        "input": args.input,
        "chains": chains,
        "sample_steps": sample_steps,
        "seed": seed,
        "model": "mock" if args.mock else (model or f"{args.llm_provider}:default"),
        "llm_provider": "mock" if args.mock else llm_provider,
        "llm_timeout": llm_timeout,
        "cot_max_tokens": cot_max_tokens,
        "wrap_max_tokens": wrap_max_tokens,
        "lean_max_tokens": lean_max_tokens,
        "score_max_tokens": score_max_tokens,
        "step_selector": step_selector,
        "stop_after": stop_after,
        "reasoning": reasoning,
        "openai_reasoning_effort": openai_reasoning_effort,
        "mock": args.mock,
        "codex_reasoning_effort": codex_reasoning_effort,
        "codex_sandbox": codex_sandbox,
        "codex_cwd": codex_cwd,
        "project_dir": project_dir,
        "wrap_repair_rounds": wrap_repair_rounds,
        "wrap_context_before": wrap_context_before,
        "wrap_context_after": wrap_context_after,
        "repair_rounds": repair_rounds,
        "skip_lean_check": skip_lean_check,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(config, run_dir / "run_config.json")

    cot_outputs = generate_cot(
        problems,
        provider=llm_provider,
        chains=chains,
        model=model,
        mock=args.mock,
        max_workers=max_workers,
        llm_timeout=llm_timeout,
        cot_max_tokens=cot_max_tokens,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
    )
    write_jsonl(cot_outputs, cot_dir / "cot_outputs.jsonl")

    parsed = parse_cot_outputs(cot_outputs)
    write_jsonl(parsed, cot_dir / "cot_steps.jsonl")
    if stop_after == "cot":
        summary = write_run_summary(
            run_dir / "run_summary.json",
            problems=problems,
            parsed=parsed,
            step_selector=step_selector,
            stopped_after=stop_after,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Run directory: {run_dir}")
        return

    step_scores: list[dict[str, Any]] = []
    if step_selector == "value":
        step_scores = score_cot_steps(
            parsed,
            provider=llm_provider,
            model=model,
            mock=args.mock,
            out_dir=scores_dir,
            llm_timeout=llm_timeout,
            score_max_tokens=score_max_tokens,
            max_workers=max_workers,
            reasoning=reasoning,
            openai_reasoning_effort=openai_reasoning_effort,
            codex_reasoning_effort=codex_reasoning_effort,
            codex_sandbox=codex_sandbox,
            codex_cwd=codex_cwd,
        )
        write_jsonl(step_scores, selection_dir / "step_scores.jsonl")
        selected = select_steps_by_value(
            parsed,
            step_scores,
            k=sample_steps,
            include_final_answer=include_final_answer,
            context_before=wrap_context_before,
            context_after=wrap_context_after,
        )
    else:
        selected = select_steps(
            parsed,
            k=sample_steps,
            seed=seed,
            include_final_answer=include_final_answer,
            context_before=wrap_context_before,
            context_after=wrap_context_after,
        )
    write_jsonl(selected, selection_dir / "steps_selected.jsonl")
    if stop_after == "selection":
        summary = write_run_summary(
            run_dir / "run_summary.json",
            problems=problems,
            parsed=parsed,
            selected=selected,
            step_selector=step_selector,
            step_scores=step_scores,
            stopped_after=stop_after,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Run directory: {run_dir}")
        return

    wrapped = generate_wrapped_claims(
        selected,
        provider=llm_provider,
        model=model,
        mock=args.mock,
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
    if stop_after == "wrapped":
        summary = write_run_summary(
            run_dir / "run_summary.json",
            problems=problems,
            parsed=parsed,
            selected=selected,
            step_selector=step_selector,
            step_scores=step_scores,
            wrapped=wrapped,
            stopped_after=stop_after,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Run directory: {run_dir}")
        return

    generated = generate_lean_contracts(
        wrapped,
        provider=llm_provider,
        model=model,
        mock=args.mock,
        out_dir=lean_dir,
        llm_timeout=llm_timeout,
        lean_max_tokens=lean_max_tokens,
        project_dir=Path(project_dir),
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
    if stop_after == "lean":
        summary = write_run_summary(
            run_dir / "run_summary.json",
            problems=problems,
            parsed=parsed,
            selected=selected,
            step_selector=step_selector,
            step_scores=step_scores,
            wrapped=wrapped,
            generated=generated,
            stopped_after=stop_after,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Run directory: {run_dir}")
        return

    verification = verify_lean_outputs(
        generated,
        project_dir=Path(project_dir),
        timeout=lean_timeout,
        skip=skip_lean_check,
    )
    write_json(verification, verification_dir / "verification.json")

    summary = write_run_summary(
        run_dir / "run_summary.json",
        problems=problems,
        parsed=parsed,
        selected=selected,
        step_selector=step_selector,
        step_scores=step_scores,
        wrapped=wrapped,
        generated=generated,
        verification=verification,
        stopped_after=stop_after,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
