from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def choose_step(row: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    steps = row.get("steps") or []
    candidates = []
    for step in steps:
        text = str(step.get("text", "")).strip()
        if not text:
            continue
        if len(text) > 1400:
            continue
        candidates.append(step)
    if not candidates:
        return None
    return rng.choice(candidates)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample a balanced batch of qwen25_fhis steps for StepFun formalization."
    )
    parser.add_argument(
        "--traces",
        default="data_generation/qwen25_fhis/outputs/generated_traces.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data_generation/qwen25_fhis/stepfun_remote_results/batch_100_samples.jsonl",
    )
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-previous-steps", type=int, default=6)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows_by_label: dict[bool, list[dict[str, Any]]] = {True: [], False: []}
    with Path(args.traces).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            label = row.get("rough_final_correct")
            if label in rows_by_label:
                rows_by_label[label].append(row)

    output_rows = []
    for label in (True, False):
        rows = rows_by_label[label][:]
        rng.shuffle(rows)
        taken = 0
        for row in rows:
            step = choose_step(row, rng)
            if step is None:
                continue
            steps = row.get("steps") or []
            step_index = int(step.get("index", 0))
            previous = [
                str(prev.get("text", "")).strip()
                for prev in steps
                if int(prev.get("index", 0)) < step_index and str(prev.get("text", "")).strip()
            ]
            if args.max_previous_steps >= 0:
                previous = previous[-args.max_previous_steps :]
            output_rows.append(
                {
                    "sample_id": f"{'trace_correct' if label else 'trace_incorrect'}-{taken:03d}",
                    "trace_id": row.get("trace_id"),
                    "problem_id": row.get("problem_id"),
                    "dataset": row.get("dataset"),
                    "rough_final_correct": label,
                    "problem": row.get("problem"),
                    "previous_steps": previous,
                    "current_step_index": step_index,
                    "current_step": f"Step {step_index}: {str(step.get('text', '')).strip()}",
                }
            )
            taken += 1
            if taken >= args.per_class:
                break
        if taken < args.per_class:
            raise RuntimeError(f"Only sampled {taken} examples for label={label}")

    rng.shuffle(output_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(output_rows)} samples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
