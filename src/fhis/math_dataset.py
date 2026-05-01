from __future__ import annotations

import argparse
import random
import re
from typing import Any

from fhis.config import load_config
from fhis.io import write_jsonl
from fhis.steps import extract_reference_answer


LEVEL_RE = re.compile(r"(\d+)")


def parse_level(value: Any) -> int | None:
    if value is None:
        return None
    match = LEVEL_RE.search(str(value))
    return int(match.group(1)) if match else None


def load_math_subset(config: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset_cfg = config["dataset"]
    seed = int(config.get("seed", 0))
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for subject, subject_cfg in dataset_cfg["subjects"].items():
        wanted_levels = set(int(x) for x in subject_cfg.get("levels", []))
        max_problems = int(subject_cfg.get("max_problems", 10**9))
        subject_rows: list[dict[str, Any]] = []

        for split in ("test", "train"):
            try:
                ds = load_dataset(dataset_cfg["hf_name"], subject, split=split)
            except Exception:
                continue
            for idx, item in enumerate(ds):
                level = parse_level(item.get("level"))
                if wanted_levels and level not in wanted_levels:
                    continue
                solution = str(item["solution"])
                subject_rows.append(
                    {
                        "problem_id": f"{subject}-{split}-{idx}",
                        "subject": subject,
                        "level": level,
                        "problem": str(item["problem"]),
                        "reference_solution": solution,
                        "reference_answer": extract_reference_answer(solution),
                        "source_split": split,
                    }
                )

        rng.shuffle(subject_rows)
        rows.extend(subject_rows[:max_problems])

    rng.shuffle(rows)
    target_total = int(dataset_cfg.get("target_total", len(rows)))
    if limit is not None:
        target_total = min(target_total, int(limit))
    return rows[:target_total]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the MATH subset for the v0 experiment.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output = args.output or config["paths"]["problems"]
    rows = load_math_subset(config, limit=args.limit)
    write_jsonl(output, rows)
    print(f"Wrote {len(rows)} problems to {output}")


if __name__ == "__main__":
    main()
