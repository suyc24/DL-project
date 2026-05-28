#!/usr/bin/env python3
"""Download a small OlympiadBench text-only math sample via HF datasets-server."""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def fetch_rows(config: str, offset: int, length: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "dataset": "Hothan/OlympiadBench",
            "config": config,
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{params}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [row["row"] for row in payload.get("rows", [])]


def normalize(row: dict[str, Any], config: str, idx: int) -> dict[str, Any]:
    return {
        "id": f"{config}-{row.get('id', idx)}",
        "source": "Hothan/OlympiadBench",
        "config": config,
        "subject": row.get("subject"),
        "subfield": row.get("subfield"),
        "difficulty": row.get("difficulty"),
        "language": row.get("language"),
        "question": row.get("question") or row.get("problem"),
        "reference_answer": row.get("final_answer") or row.get("answer"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="OE_TO_maths_en_COMP")
    parser.add_argument("--out", default="lean_single_step_formalization/data/olympiadbench_non_geometry_20.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--skip-geometry", action="store_true", default=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < args.limit:
        rows = fetch_rows(args.config, offset, args.batch_size)
        if not rows:
            break
        for row in rows:
            subfield = str(row.get("subfield") or "")
            question = row.get("question") or row.get("problem")
            if not question:
                continue
            if args.skip_geometry and "geometry" in subfield.lower():
                continue
            selected.append(normalize(row, args.config, offset + len(selected)))
            if len(selected) >= args.limit:
                break
        offset += len(rows)

    with out_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} rows to {out_path}")


if __name__ == "__main__":
    main()
