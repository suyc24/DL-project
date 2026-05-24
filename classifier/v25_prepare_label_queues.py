from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONTENT_KEYS = {"problem", "reference_solution", "prompt", "completion", "steps", "token_ids", "token_logprobs"}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} line {line_no}: {exc}") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_trace_metadata(trace: dict[str, Any], split: str) -> dict[str, Any]:
    steps = trace.get("steps") or []
    refs = trace.get("reference_answer") or []
    return {
        "trace_id": trace.get("trace_id"),
        "problem_id": trace.get("problem_id"),
        "dataset": trace.get("dataset"),
        "subset": trace.get("subset"),
        "source_id": trace.get("source_id"),
        "split": split,
        "generation_batch_id": trace.get("generation_batch_id"),
        "sample_index": trace.get("sample_index"),
        "model_name": trace.get("model_name"),
        "rough_final_correct": trace.get("rough_final_correct"),
        "final_answer_present": trace.get("final_answer") is not None,
        "reference_answer_count": len(refs) if isinstance(refs, list) else int(refs is not None),
        "num_steps": len(steps) if isinstance(steps, list) else 0,
        "step_parseable": isinstance(steps, list) and len(steps) > 0,
    }


def blank_label(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": row["trace_id"],
        "final_correct": None,
        "first_invalid_step": None,
        "error_type": None,
        "reason": "",
        "confidence": "",
        "labeler": "",
        "label_source": "offline_or_human",
        "split": row["split"],
    }


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    by_rough = Counter(str(r.get("rough_final_correct")) for r in rows)
    by_dataset = Counter(str(r.get("dataset")) for r in rows)
    by_subset = Counter(str(r.get("subset")) for r in rows)
    problem_to_traces: dict[str, int] = defaultdict(int)
    duplicate_trace_ids = []
    seen_trace_ids = set()
    missing_trace_ids = 0
    missing_problem_ids = 0
    step_counts = []
    for r in rows:
        tid = r.get("trace_id")
        pid = r.get("problem_id")
        if not tid:
            missing_trace_ids += 1
        elif tid in seen_trace_ids:
            duplicate_trace_ids.append(tid)
        else:
            seen_trace_ids.add(tid)
        if not pid:
            missing_problem_ids += 1
        else:
            problem_to_traces[str(pid)] += 1
        step_counts.append(int(r.get("num_steps") or 0))
    return {
        "split": split,
        "num_traces": len(rows),
        "num_unique_trace_ids": len(seen_trace_ids),
        "num_unique_problem_ids": len(problem_to_traces),
        "missing_trace_ids": missing_trace_ids,
        "missing_problem_ids": missing_problem_ids,
        "duplicate_trace_ids": duplicate_trace_ids[:50],
        "num_duplicate_trace_ids": len(duplicate_trace_ids),
        "rough_final_correct_counts": dict(sorted(by_rough.items())),
        "dataset_counts": dict(sorted(by_dataset.items())),
        "subset_counts_top20": dict(by_subset.most_common(20)),
        "num_step_parseable": sum(1 for r in rows if r.get("step_parseable")),
        "min_steps": min(step_counts) if step_counts else None,
        "max_steps": max(step_counts) if step_counts else None,
        "avg_steps": (sum(step_counts) / len(step_counts)) if step_counts else None,
        "problem_trace_count_min": min(problem_to_traces.values()) if problem_to_traces else None,
        "problem_trace_count_max": max(problem_to_traces.values()) if problem_to_traces else None,
    }


def load_split(path: Path, split: str) -> tuple[list[dict[str, Any]], set[str]]:
    rows = []
    problems = set()
    for trace in read_jsonl(path):
        meta = safe_trace_metadata(trace, split)
        rows.append(meta)
        if meta.get("problem_id"):
            problems.add(str(meta["problem_id"]))
    return rows, problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare safe metadata-only FHIS label queues for v2.5 traces.")
    parser.add_argument("--train-traces", type=Path, required=True)
    parser.add_argument("--clean-eval-traces", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete-clean-eval", action="store_true")
    parser.add_argument("--expected-train", type=int, default=2000)
    parser.add_argument("--expected-clean-eval", type=int, default=500)
    args = parser.parse_args()

    train_rows, train_problems = load_split(args.train_traces, "train_expansion")
    eval_rows, eval_problems = load_split(args.clean_eval_traces, "clean_eval_natural")

    if len(train_rows) != args.expected_train:
        raise SystemExit(f"train trace count {len(train_rows)} != expected {args.expected_train}")
    if len(eval_rows) != args.expected_clean_eval and not args.allow_incomplete_clean_eval:
        raise SystemExit(
            f"clean eval trace count {len(eval_rows)} != expected {args.expected_clean_eval}; "
            "pass --allow-incomplete-clean-eval only for progress snapshots"
        )

    overlap = sorted(train_problems & eval_problems)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(args.out_dir / "train_queue_metadata.jsonl", train_rows)
    write_jsonl(args.out_dir / "clean_eval_queue_metadata.jsonl", eval_rows)
    write_jsonl(args.out_dir / "train_blank_labels.jsonl", [blank_label(r) for r in train_rows])
    write_jsonl(args.out_dir / "clean_eval_blank_labels.jsonl", [blank_label(r) for r in eval_rows])

    manifest = {
        "train_traces": str(args.train_traces),
        "clean_eval_traces": str(args.clean_eval_traces),
        "outputs": {
            "train_queue_metadata": "train_queue_metadata.jsonl",
            "clean_eval_queue_metadata": "clean_eval_queue_metadata.jsonl",
            "train_blank_labels": "train_blank_labels.jsonl",
            "clean_eval_blank_labels": "clean_eval_blank_labels.jsonl",
        },
        "train_summary": summarize(train_rows, "train_expansion"),
        "clean_eval_summary": summarize(eval_rows, "clean_eval_natural"),
        "leakage_check": {
            "train_clean_eval_problem_overlap_count": len(overlap),
            "train_clean_eval_problem_overlap_sample": overlap[:50],
            "clean_eval_is_held_out": len(overlap) == 0,
        },
        "content_policy_note": (
            "This script writes metadata-only queues and blank label templates by default. "
            "It intentionally excludes problem, prompt, completion, steps, reference_solution, and token fields."
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "train_traces": len(train_rows),
        "clean_eval_traces": len(eval_rows),
        "problem_overlap": len(overlap),
        "clean_eval_held_out": len(overlap) == 0,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
