from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"Could not find repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise
        return json.loads(text[start : end + 1])


def compact_steps(trace: dict[str, Any]) -> str:
    lines = []
    for step in trace.get("steps", []):
        text = str(step.get("text", "")).strip()
        lines.append(f"Step {step['index']}: {text}")
    return "\n\n".join(lines)


def build_prompt(trace: dict[str, Any]) -> str:
    rough = trace.get("rough_final_correct")
    rough_text = "unknown" if rough is None else str(bool(rough)).lower()
    return f"""You are creating high-quality FHIS labels for a mathematical reasoning probe.

Definitions:
- A harmful invalid step is the earliest generated step whose mathematical claim,
  transformation, computation, or conclusion is wrong and can plausibly cause the
  final answer to be wrong.
- If the final answer is correct and the generated reasoning has no harmful invalid
  step, set final_correct=true and first_invalid_step=null.
- If the final answer is wrong, first_invalid_step should be the earliest harmful
  invalid step. Do not choose a later step if an earlier harmful error exists.
- If the generated trace is incomplete, lacks enough information, or the first
  harmful invalid step cannot be determined, use confidence="low".
- Minor wording issues, missing rigor, or skipped algebra are not harmful invalid
  steps unless they introduce a false claim.

Return only JSON matching this schema:
{{
  "final_correct": true or false,
  "first_invalid_step": integer or null,
  "error_type": string or null,
  "reason": string,
  "confidence": "high" or "medium" or "low"
}}

Trace id:
{trace["trace_id"]}

Rough automatic final-answer match:
{rough_text}

Problem:
{trace["problem"]}

Reference answer:
{trace.get("reference_answer")}

Reference solution:
{trace.get("reference_solution")}

Generated final answer:
{trace.get("final_answer")}

Generated steps:
{compact_steps(trace)}
"""


def normalize_label(raw: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    first_invalid = raw.get("first_invalid_step")
    if first_invalid in ("", "null", "None"):
        first_invalid = None
    if first_invalid is not None:
        first_invalid = int(first_invalid)
    return {
        "trace_id": str(trace["trace_id"]),
        "problem_id": str(trace["problem_id"]),
        "final_correct": bool(raw.get("final_correct", False)),
        "first_invalid_step": first_invalid,
        "error_type": raw.get("error_type"),
        "reason": str(raw.get("reason", "")).strip(),
        "confidence": str(raw.get("confidence", "low")).lower(),
        "rough_final_correct": trace.get("rough_final_correct"),
        "num_steps": len(trace.get("steps", [])),
        "labeler": "local_codex",
        "labeler_model": "gpt-5.5",
        "labeler_reasoning_effort": "high",
    }


def is_training_candidate(trace: dict[str, Any], include_unknown: bool) -> bool:
    if not trace.get("steps"):
        return False
    if trace.get("rough_final_correct") is None and not include_unknown:
        return False
    if trace.get("final_answer") is None and not include_unknown:
        return False
    return True


def run_codex(prompt: str, schema_path: Path, model: str, reasoning_effort: str) -> dict[str, Any]:
    prompt = prompt.replace("\x00", "")
    env = os.environ.copy()
    env.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
    env.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
    env.setdefault("ALL_PROXY", "http://127.0.0.1:7890")

    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".json") as out:
        cmd = [
            "codex",
            "exec",
            "--cd",
            str(REPO_ROOT),
            "--sandbox",
            "read-only",
            "-m",
            model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
            "-c",
            'approval_policy="never"',
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            out.name,
            prompt,
        ]
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        out.seek(0)
        return parse_json_object(out.read())


def label_is_structurally_valid(label: dict[str, Any]) -> bool:
    if not label["reason"]:
        return False
    if label["confidence"] not in {"high", "medium", "low"}:
        return False
    first_invalid = label["first_invalid_step"]
    if first_invalid is None:
        return True
    return 1 <= int(first_invalid) <= int(label["num_steps"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Label FHIS with local Codex CLI.")
    parser.add_argument(
        "--traces",
        default="data_generation/qwen25_fhis/outputs/generated_traces.jsonl",
    )
    parser.add_argument("--output", default="data_generation/qwen25_fhis/labels/fhis_labels.jsonl")
    parser.add_argument(
        "--schema",
        default="data_generation/qwen25_fhis/schema/local_codex_label_schema.json",
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    traces = [
        trace
        for idx, trace in enumerate(read_jsonl(args.traces))
        if is_training_candidate(trace, include_unknown=args.include_unknown)
        and idx % int(args.num_shards) == int(args.shard_index)
    ]
    if args.resume:
        done = {row["trace_id"] for row in read_jsonl(args.output)}
        traces = [trace for trace in traces if trace["trace_id"] not in done]
    if args.limit is not None:
        traces = traces[: args.limit]

    schema_path = Path(args.schema)
    failures_path = Path(args.output).with_suffix(".failures.jsonl")
    for i, trace in enumerate(traces, start=1):
        prompt = build_prompt(trace)
        last_error = None
        for attempt in range(args.max_retries + 1):
            try:
                raw = run_codex(prompt, schema_path, args.model, args.reasoning_effort)
                label = normalize_label(raw, trace)
                if not label_is_structurally_valid(label):
                    raise ValueError(f"invalid label structure: {label}")
                append_jsonl(args.output, label)
                print(f"[{i}/{len(traces)}] labeled {trace['trace_id']}: {label['confidence']}")
                break
            except Exception as exc:
                last_error = exc
                if attempt >= args.max_retries:
                    append_jsonl(
                        failures_path,
                        {
                            "trace_id": trace["trace_id"],
                            "problem_id": trace["problem_id"],
                            "error": repr(last_error),
                        },
                    )
                    print(f"[{i}/{len(traces)}] failed {trace['trace_id']}: {last_error}")


if __name__ == "__main__":
    main()
