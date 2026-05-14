from __future__ import annotations

import argparse
import json
import operator
import random
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fhis.lean_verify import verify_lean_code  # noqa: E402


EQUATION_RE = re.compile(
    r"(?<![\w.])(-?\d+)\s*(\+|-|\\times|\*|×|\\cdot|cdot)\s*(-?\d+)\s*=\s*(-?\d+)(?![\w.])"
)
OPS = {
    "+": ("+", operator.add),
    "-": ("-", operator.sub),
    "*": ("*", operator.mul),
    "\\times": ("*", operator.mul),
    "×": ("*", operator.mul),
    "\\cdot": ("*", operator.mul),
    "cdot": ("*", operator.mul),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("version https://git-lfs.github.com"):
                rows.append(json.loads(line))
    return rows


def lean_code(a: int, op: str, b: int, c: int) -> str:
    lean_op, _ = OPS[op]
    return f"""example : ({a} : Int) {lean_op} ({b} : Int) = ({c} : Int) := by
  native_decide
"""


def find_candidates(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for trace in traces:
        for step in trace.get("steps") or []:
            text = str(step.get("text", ""))
            match = EQUATION_RE.search(text)
            if not match:
                continue
            a, op, b, c = match.groups()
            a_i, b_i, c_i = int(a), int(b), int(c)
            _, fn = OPS[op]
            expected_status = "proved" if fn(a_i, b_i) == c_i else "failed"
            candidates.append(
                {
                    "trace_id": trace.get("trace_id"),
                    "problem_id": trace.get("problem_id"),
                    "rough_final_correct": trace.get("rough_final_correct"),
                    "step_index": step.get("index"),
                    "step_text": text,
                    "localized_equation": match.group(0),
                    "a": a_i,
                    "op": op,
                    "b": b_i,
                    "c": c_i,
                    "expected_status": expected_status,
                    "lean_code": lean_code(a_i, op, b_i, c_i),
                }
            )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample localized arithmetic steps from qwen25_fhis traces and verify Lean templates."
    )
    parser.add_argument(
        "--traces",
        default="data_generation/qwen25_fhis/outputs/generated_traces.jsonl",
    )
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260507)
    args = parser.parse_args()

    traces = read_jsonl(Path(args.traces))
    candidates = find_candidates(traces)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sample = candidates[: args.sample_size]

    cases = []
    ok = True
    for item in sample:
        result = verify_lean_code(item["lean_code"], timeout_s=10.0)
        passed = result.status == item["expected_status"]
        ok = ok and passed
        cases.append(
            {
                **{k: v for k, v in item.items() if k != "lean_code"},
                "actual_status": result.status,
                "passed": passed,
                "lean_stderr": result.stderr.strip()[:1000],
                "lean_code": item["lean_code"],
            }
        )

    payload = {
        "passed": ok,
        "num_traces": len(traces),
        "num_candidates": len(candidates),
        "num_sampled": len(sample),
        "cases": cases,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
