from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_train_usable(row: dict[str, Any]) -> bool:
    if row.get("confidence") != "high":
        return False
    if not str(row.get("reason", "")).strip():
        return False
    first_invalid = row.get("first_invalid_step")
    if row.get("final_correct") is True:
        return first_invalid is None
    if first_invalid is None:
        return False
    return 1 <= int(first_invalid) <= int(row.get("num_steps", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local Codex FHIS labels.")
    parser.add_argument("--labels", default="data_generation/qwen25_fhis/labels/fhis_labels.jsonl")
    parser.add_argument(
        "--output",
        default="data_generation/qwen25_fhis/labels/fhis_label_summary.json",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.labels)
    confidence = Counter(row.get("confidence") for row in rows)
    final_correct = Counter(row.get("final_correct") for row in rows)
    rough = Counter(row.get("rough_final_correct") for row in rows)
    usable = [row for row in rows if is_train_usable(row)]
    conflict = [
        row
        for row in rows
        if row.get("rough_final_correct") is not None
        and bool(row.get("rough_final_correct")) != bool(row.get("final_correct"))
    ]
    wrong_with_step = [
        row
        for row in usable
        if row.get("final_correct") is False and row.get("first_invalid_step") is not None
    ]
    payload = {
        "num_labels": len(rows),
        "confidence": dict(confidence),
        "final_correct": {str(k): v for k, v in final_correct.items()},
        "rough_final_correct": {str(k): v for k, v in rough.items()},
        "train_usable_high_confidence": len(usable),
        "high_confidence_wrong_with_first_invalid_step": len(wrong_with_step),
        "rough_codex_final_correct_conflicts": len(conflict),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
