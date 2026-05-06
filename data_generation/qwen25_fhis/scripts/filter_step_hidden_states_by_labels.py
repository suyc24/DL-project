from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter cached step hidden-state rows to match a label file."
    )
    parser.add_argument(
        "--input",
        default="data_generation/qwen25_fhis/features/step_hidden_states.pt",
    )
    parser.add_argument(
        "--labels",
        default="data_generation/qwen25_fhis/labels/fhis_labels_train_high.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data_generation/qwen25_fhis/features/step_hidden_states_codex_clean.pt",
    )
    parser.add_argument(
        "--summary",
        default="data_generation/qwen25_fhis/features/step_hidden_states_codex_clean_summary.json",
    )
    args = parser.parse_args()

    keep_trace_ids = {row["trace_id"] for row in read_jsonl(args.labels)}
    payload: dict[str, Any] = torch.load(args.input, map_location="cpu", weights_only=False)
    input_rows = payload["rows"]
    output_rows = [row for row in input_rows if row["trace_id"] in keep_trace_ids]

    output_payload = dict(payload)
    output_payload["rows"] = output_rows
    output_payload["filtered_from"] = str(args.input)
    output_payload["labels"] = str(args.labels)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)

    label_counts = Counter(int(row["label"]) for row in output_rows)
    trace_ids = {row["trace_id"] for row in output_rows}
    summary = {
        "input_rows": len(input_rows),
        "output_rows": len(output_rows),
        "input_trace_count": len({row["trace_id"] for row in input_rows}),
        "output_trace_count": len(trace_ids),
        "label_trace_count": len(keep_trace_ids),
        "missing_label_traces_in_features": sorted(keep_trace_ids - trace_ids),
        "label_counts": {str(k): v for k, v in sorted(label_counts.items())},
    }
    Path(args.summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
