from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter FHIS labels for probe training.")
    parser.add_argument("--input", default="data_generation/qwen25_fhis/labels/fhis_labels.jsonl")
    parser.add_argument(
        "--output",
        default="data_generation/qwen25_fhis/labels/fhis_labels_train_high.jsonl",
    )
    parser.add_argument(
        "--summary",
        default="data_generation/qwen25_fhis/labels/fhis_labels_train_high_summary.json",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    usable = [row for row in rows if is_training_usable(row)]
    write_jsonl(args.output, usable)

    payload = {
        "input_labels": len(rows),
        "training_labels": len(usable),
        "correct_negative_traces": sum(1 for row in usable if row.get("final_correct") is True),
        "wrong_fhis_traces": sum(1 for row in usable if row.get("final_correct") is False),
        "excluded": len(rows) - len(usable),
    }
    Path(args.summary).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
