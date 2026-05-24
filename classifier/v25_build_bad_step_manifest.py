from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path} line {line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_hash_pct(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16) % 100


def split_for_problem(problem_id: str, train_pct: int, calibration_pct: int) -> str:
    bucket = stable_hash_pct(problem_id)
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + calibration_pct:
        return "calibration"
    return "hard_dev"


def normalized_confidence(label: dict[str, Any]) -> str:
    return str(label.get("confidence", "")).strip().lower()


def label_step_target(label: dict[str, Any], step_index: int) -> int | None:
    first_invalid = label.get("first_invalid_step")
    final_correct = bool(label.get("final_correct", False))
    if final_correct and first_invalid is None:
        return 0
    if first_invalid is None:
        return None
    first_invalid = int(first_invalid)
    if step_index < first_invalid:
        return 0
    if step_index == first_invalid:
        return 1
    return None


def example_type_for(label: dict[str, Any], target: int) -> str:
    if target == 1:
        return "fhis_positive"
    if bool(label.get("final_correct", False)) and label.get("first_invalid_step") is None:
        return "strong_correct_negative"
    return "strong_prefhis_negative"


def weight_for(example_type: str, args: argparse.Namespace) -> float:
    if example_type == "fhis_positive":
        return float(args.positive_weight)
    if example_type == "strong_correct_negative":
        return float(args.correct_negative_weight)
    if example_type == "strong_prefhis_negative":
        return float(args.prefhis_negative_weight)
    return 1.0


def build_manifest(
    traces: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels_by_trace = {str(row["trace_id"]): row for row in labels}
    manifest: list[dict[str, Any]] = []
    dropped_labels = Counter()
    trace_counts = Counter()
    split_counts = Counter()
    example_counts = Counter()
    target_counts = Counter()
    problem_to_traces: dict[str, set[str]] = defaultdict(set)

    for trace in traces:
        trace_id = str(trace["trace_id"])
        label = labels_by_trace.get(trace_id)
        if label is None:
            dropped_labels["missing_label"] += 1
            continue
        confidence = normalized_confidence(label)
        if confidence != args.keep_confidence:
            dropped_labels[f"confidence_{confidence or 'missing'}"] += 1
            continue
        steps = trace.get("steps") or []
        if not steps:
            dropped_labels["no_steps"] += 1
            continue
        problem_id = str(trace["problem_id"])
        split = split_for_problem(problem_id, int(args.train_pct), int(args.calibration_pct))
        first_invalid = label.get("first_invalid_step")
        for step in steps:
            step_index = int(step["index"])
            target = label_step_target(label, step_index)
            if target is None:
                continue
            example_type = example_type_for(label, target)
            row = {
                "trace_id": trace_id,
                "problem_id": problem_id,
                "step_index": step_index,
                "split": split,
                "target_bad_step": int(target),
                "sample_weight": weight_for(example_type, args),
                "example_type": example_type,
                "validity": "high_confidence_fhis",
                "label_confidence": confidence,
                "first_invalid_step": first_invalid,
                "trace_final_correct": bool(label.get("final_correct", False)),
                "rough_final_correct": trace.get("rough_final_correct"),
                "num_steps": len(steps),
                "label_source": label.get("label_source"),
                "labeler": label.get("labeler"),
                "generation_batch_id": trace.get("generation_batch_id"),
                "dataset": trace.get("dataset"),
                "subset": trace.get("subset"),
            }
            manifest.append(row)
            trace_counts[trace_id] += 1
            split_counts[split] += 1
            example_counts[example_type] += 1
            target_counts[str(target)] += 1
            problem_to_traces[problem_id].add(trace_id)

    all_trace_ids = {str(t["trace_id"]) for t in traces}
    labeled_trace_ids = set(labels_by_trace)
    used_trace_ids = set(trace_counts)
    incomplete_problem_batches = {
        pid: sorted(traces_)
        for pid, traces_ in problem_to_traces.items()
        if len(traces_) != 4
    }
    summary = {
        "trace_rows": len(traces),
        "label_rows": len(labels),
        "unique_trace_ids": len(all_trace_ids),
        "unique_label_trace_ids": len(labeled_trace_ids),
        "missing_label_trace_ids": sorted(all_trace_ids - labeled_trace_ids)[:100],
        "extra_label_trace_ids": sorted(labeled_trace_ids - all_trace_ids)[:100],
        "manifest_rows": len(manifest),
        "manifest_trace_count": len(used_trace_ids),
        "manifest_problem_count": len(problem_to_traces),
        "dropped_trace_counts": dict(dropped_labels),
        "split_counts": dict(split_counts),
        "example_type_counts": dict(example_counts),
        "target_bad_step_counts": dict(target_counts),
        "train_pct": int(args.train_pct),
        "calibration_pct": int(args.calibration_pct),
        "hard_dev_pct": 100 - int(args.train_pct) - int(args.calibration_pct),
        "keep_confidence": args.keep_confidence,
        "weights": {
            "fhis_positive": float(args.positive_weight),
            "strong_prefhis_negative": float(args.prefhis_negative_weight),
            "strong_correct_negative": float(args.correct_negative_weight),
        },
        "complete_problem_batches_of_4": sum(1 for traces_ in problem_to_traces.values() if len(traces_) == 4),
        "incomplete_problem_batch_count": len(incomplete_problem_batches),
        "incomplete_problem_batches_sample": dict(list(sorted(incomplete_problem_batches.items()))[:50]),
    }
    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2.5 bad-step per-step manifest from FHIS labels.")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--keep-confidence", default="high")
    parser.add_argument("--train-pct", type=int, default=70)
    parser.add_argument("--calibration-pct", type=int, default=15)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--prefhis-negative-weight", type=float, default=1.0)
    parser.add_argument("--correct-negative-weight", type=float, default=0.75)
    args = parser.parse_args()

    if args.train_pct + args.calibration_pct >= 100:
        raise SystemExit("--train-pct + --calibration-pct must be < 100")

    traces = read_jsonl(args.traces)
    labels = read_jsonl(args.labels)
    manifest, summary = build_manifest(traces, labels, args)
    write_jsonl(args.output, manifest)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
