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
    build_label_prompt,
    is_labeling_candidate,
    label_is_structurally_valid,
    normalize_label,
    parse_json_object,
)


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


def read_trace_ids(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    trace_ids: set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trace_ids.add(line)
    return trace_ids


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    parser.add_argument(
        "--trace-id",
        action="append",
        default=[],
        help="Restrict labeling to this trace id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--trace-ids-file",
        default=None,
        help="Restrict labeling to newline-separated trace ids from this file.",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    target_trace_ids = set(args.trace_id) | read_trace_ids(args.trace_ids_file)
    traces = [
        trace
        for idx, trace in enumerate(read_jsonl(args.traces))
        if is_labeling_candidate(trace, include_unknown=args.include_unknown)
        and (not target_trace_ids or trace["trace_id"] in target_trace_ids)
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
        prompt = build_label_prompt(trace)
        last_error = None
        for attempt in range(args.max_retries + 1):
            try:
                raw = run_codex(prompt, schema_path, args.model, args.reasoning_effort)
                label = normalize_label(
                    raw,
                    trace=trace,
                    labeler="local_codex",
                    labeler_model=args.model,
                    labeler_reasoning_effort=args.reasoning_effort,
                )
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
