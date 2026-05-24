from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"Could not find repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fhis.labeling import (  # noqa: E402
    compact_steps,
    is_labeling_candidate,
    label_is_structurally_valid,
    normalize_label,
    parse_json_object,
    rough_final_correct_text,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def build_batch_prompt(traces: list[dict[str, Any]]) -> str:
    parts = [
        "You are creating high-quality FHIS labels for mathematical reasoning traces.",
        "",
        "Definitions:",
        "- A harmful invalid step is the earliest generated step whose mathematical claim, transformation, computation, or conclusion is wrong and can plausibly cause the final answer to be wrong.",
        "- final_correct records whether the generated final answer matches the reference.",
        "- first_invalid_step records the earliest harmful invalid step, even if the trace later recovers and reaches the correct final answer.",
        "- If the final answer is correct and the generated reasoning has no harmful invalid step, set first_invalid_step=null.",
        "- If the final answer is wrong, first_invalid_step should be the earliest harmful invalid step.",
        "- If the trace is incomplete or the first harmful invalid step cannot be determined, use confidence=\"low\".",
        "- Minor missing rigor is not a harmful invalid step unless it introduces a false claim.",
        "",
        "Return only valid JSON in this schema:",
        '{"labels":[{"trace_id":"...","final_correct":true,"first_invalid_step":null,"error_type":null,"reason":"...","confidence":"high"}]}',
        "",
    ]
    for idx, trace in enumerate(traces, start=1):
        parts.extend(
            [
                f"TRACE {idx}",
                f"trace_id: {trace.get('trace_id')}",
                f"rough automatic final-answer match: {rough_final_correct_text(trace)}",
                "Problem:",
                str(trace["problem"]),
                "Reference answer:",
                str(trace.get("reference_answer")),
                "Reference solution:",
                str(trace.get("reference_solution")),
                "Generated final answer:",
                str(trace.get("final_answer")),
                "Generated steps:",
                compact_steps(trace),
                "",
            ]
        )
    return "\n".join(parts)


def run_codex(prompt: str, schema_path: Path, model: str, reasoning_effort: str) -> dict[str, Any]:
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
            prompt.replace("\x00", ""),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-label FHIS traces with local Codex CLI.")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schema", default="data_generation/qwen25_fhis/schema/local_codex_batch_label_schema.json")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--split", default=None)
    parser.add_argument("--label-source", default="local_codex_batch_fhis")
    args = parser.parse_args()

    traces = [
        trace
        for trace in read_jsonl(args.traces)
        if is_labeling_candidate(trace, include_unknown=args.include_unknown)
    ]
    if args.resume:
        done = {row["trace_id"] for row in read_jsonl(args.output)}
        traces = [trace for trace in traces if trace["trace_id"] not in done]
    if args.limit is not None:
        traces = traces[: args.limit]

    schema_path = Path(args.schema)
    failures_path = Path(args.output).with_suffix(".failures.jsonl")
    trace_by_id = {str(trace["trace_id"]): trace for trace in traces}
    for batch_index, trace_batch in enumerate(batched(traces, args.batch_size), start=1):
        prompt = build_batch_prompt(trace_batch)
        last_error = None
        for attempt in range(args.max_retries + 1):
            try:
                raw = run_codex(prompt, schema_path, args.model, args.reasoning_effort)
                labels = raw.get("labels", [])
                if len(labels) != len(trace_batch):
                    raise ValueError(f"expected {len(trace_batch)} labels, got {len(labels)}")
                seen = set()
                for raw_label in labels:
                    tid = str(raw_label.get("trace_id"))
                    if tid not in trace_by_id:
                        raise ValueError(f"unexpected trace_id in label: {tid}")
                    seen.add(tid)
                    trace = trace_by_id[tid]
                    label = normalize_label(
                        raw_label,
                        trace=trace,
                        labeler="local_codex_batch",
                        labeler_model=args.model,
                        labeler_reasoning_effort=args.reasoning_effort,
                    )
                    if args.split is not None:
                        label["split"] = args.split
                    label["label_source"] = args.label_source
                    if not label_is_structurally_valid(label):
                        raise ValueError(f"invalid label structure: {label}")
                    append_jsonl(args.output, label)
                expected = {str(trace["trace_id"]) for trace in trace_batch}
                if seen != expected:
                    raise ValueError(f"missing trace ids: {sorted(expected - seen)}")
                print(f"[{batch_index}/{len(batched(traces, args.batch_size))}] labeled {len(trace_batch)} traces")
                break
            except Exception as exc:
                last_error = exc
                if attempt >= args.max_retries:
                    append_jsonl(
                        failures_path,
                        {
                            "batch_index": batch_index,
                            "trace_ids": [trace["trace_id"] for trace in trace_batch],
                            "error": repr(last_error),
                        },
                    )
                    print(f"[{batch_index}] failed: {last_error}")


if __name__ == "__main__":
    main()
