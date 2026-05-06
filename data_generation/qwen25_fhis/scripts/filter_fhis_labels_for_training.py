from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_training_usable(label: dict[str, Any]) -> bool:
    """Return whether the Codex FHIS label can supervise probe training."""
    if label.get("confidence") != "high":
        return False
    if not str(label.get("reason", "")).strip():
        return False
    first_invalid = label.get("first_invalid_step")
    if bool(label.get("final_correct")):
        return first_invalid is None
    if first_invalid is None:
        return False
    return 1 <= int(first_invalid) <= int(label.get("num_steps", 0))


def is_maxed_generation(trace: dict[str, Any]) -> bool:
    max_new_tokens = int((trace.get("generation_config") or {}).get("max_new_tokens", 0) or 0)
    token_ids = trace.get("token_ids") or []
    return max_new_tokens > 0 and len(token_ids) >= max_new_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter FHIS labels for probe training.")
    parser.add_argument("--input", default="data_generation/qwen25_fhis/labels/fhis_labels.jsonl")
    parser.add_argument(
        "--traces",
        default="data_generation/qwen25_fhis/outputs/generated_traces.jsonl",
        help="Generated traces used for quality filters such as max-token truncation.",
    )
    parser.add_argument(
        "--output",
        default="data_generation/qwen25_fhis/labels/fhis_labels_train_high.jsonl",
    )
    parser.add_argument(
        "--summary",
        default="data_generation/qwen25_fhis/labels/fhis_labels_train_high_summary.json",
    )
    parser.add_argument(
        "--keep-maxed-generations",
        action="store_true",
        help="Keep traces whose generated token count reached max_new_tokens.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    traces = {row["trace_id"]: row for row in read_jsonl(args.traces)}

    excluded_reasons: Counter[str] = Counter()
    usable: list[dict[str, Any]] = []
    rough_codex_conflicts = 0
    for row in rows:
        trace = traces.get(row["trace_id"])
        if (
            row.get("rough_final_correct") is not None
            and bool(row.get("rough_final_correct")) != bool(row.get("final_correct"))
        ):
            rough_codex_conflicts += 1

        if not is_training_usable(row):
            excluded_reasons["label_not_training_usable"] += 1
            continue
        if trace is None:
            excluded_reasons["missing_trace"] += 1
            continue
        if not args.keep_maxed_generations and is_maxed_generation(trace):
            excluded_reasons["maxed_generation"] += 1
            continue

        usable.append(row)

    write_jsonl(args.output, usable)

    train_conflicts = sum(
        1
        for row in usable
        if row.get("rough_final_correct") is not None
        and bool(row.get("rough_final_correct")) != bool(row.get("final_correct"))
    )
    payload = {
        "label_authority": "codex",
        "rough_final_correct_usage": "diagnostic_only",
        "input_labels": len(rows),
        "training_labels": len(usable),
        "correct_negative_traces": sum(1 for row in usable if row.get("final_correct") is True),
        "wrong_fhis_traces": sum(1 for row in usable if row.get("final_correct") is False),
        "excluded": len(rows) - len(usable),
        "excluded_reasons": dict(excluded_reasons),
        "rough_codex_final_correct_conflicts": rough_codex_conflicts,
        "rough_codex_final_correct_conflicts_in_training": train_conflicts,
    }
    Path(args.summary).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
