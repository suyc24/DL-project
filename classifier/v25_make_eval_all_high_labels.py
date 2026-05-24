from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an eval-only label view that keeps all clean-eval labels during feature extraction."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["original_confidence"] = obj.get("confidence")
            obj["confidence"] = "high"
            obj["eval_confidence_policy"] = "all_clean_eval_labels_as_high_for_feature_extraction_only"
            rows.append(obj)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
