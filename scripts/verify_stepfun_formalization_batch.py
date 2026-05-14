from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def normalize_lean_code(code: str) -> str:
    code = code.strip()
    fences = list(re.finditer(r"```(?:lean4?|Lean4?)?\s*(.*?)```", code, flags=re.S))
    if fences:
        code = fences[-1].group(1).strip()
    starts = [pos for pos in (code.find("import "), code.find("theorem localized_step_check")) if pos >= 0]
    if starts:
        return code[min(starts) :].strip()
    return code


def compile_lean(code: str, lean_executable: str, timeout_s: float) -> dict[str, Any]:
    code = normalize_lean_code(code)
    if not code.strip():
        return {"compile_ok": False, "returncode": None, "stderr": "empty Lean code"}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "Check.lean"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [lean_executable, str(path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "compile_ok": False,
                "returncode": None,
                "stderr": f"timeout after {timeout_s}s: {exc}",
            }
    return {
        "compile_ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-4000:],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counters = Counter()
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        code = row.get("lean_code", "")
        label = "trace_correct" if row.get("rough_final_correct") else "trace_incorrect"
        compile_ok = bool(row.get("lean_compile", {}).get("compile_ok"))
        theorem = "localized_step_check" in code
        by_sorry = ":= by sorry" in code.replace("\n", " ")
        has_sorry = "sorry" in code
        counters["compile_ok"] += compile_ok
        counters["theorem_present"] += theorem
        counters["ends_by_sorry_pattern"] += by_sorry
        counters["has_sorry"] += has_sorry
        by_label[label]["total"] += 1
        by_label[label]["compile_ok"] += compile_ok
        by_label[label]["theorem_present"] += theorem
        by_label[label]["ends_by_sorry_pattern"] += by_sorry
        by_label[label]["has_sorry"] += has_sorry

    def rate(count: int, denom: int = total) -> float:
        return round(count / denom, 4) if denom else 0.0

    label_summary = {}
    for label, counter in by_label.items():
        denom = counter["total"]
        label_summary[label] = {
            "total": denom,
            "compile_ok": counter["compile_ok"],
            "compile_ok_rate": rate(counter["compile_ok"], denom),
            "theorem_present": counter["theorem_present"],
            "theorem_present_rate": rate(counter["theorem_present"], denom),
            "ends_by_sorry_pattern": counter["ends_by_sorry_pattern"],
            "ends_by_sorry_pattern_rate": rate(counter["ends_by_sorry_pattern"], denom),
            "has_sorry": counter["has_sorry"],
            "has_sorry_rate": rate(counter["has_sorry"], denom),
        }
    return {
        "total": total,
        "compile_ok": counters["compile_ok"],
        "compile_ok_rate": rate(counters["compile_ok"]),
        "theorem_present": counters["theorem_present"],
        "theorem_present_rate": rate(counters["theorem_present"]),
        "ends_by_sorry_pattern": counters["ends_by_sorry_pattern"],
        "ends_by_sorry_pattern_rate": rate(counters["ends_by_sorry_pattern"]),
        "has_sorry": counters["has_sorry"],
        "has_sorry_rate": rate(counters["has_sorry"]),
        "by_trace_label": label_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile-check StepFun batch Lean outputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--lean-executable", default="lean")
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    rows = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["lean_code"] = normalize_lean_code(row.get("lean_code", ""))
            row["lean_compile"] = compile_lean(
                row.get("lean_code", ""),
                lean_executable=args.lean_executable,
                timeout_s=args.timeout_s,
            )
            rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    summary_path = Path(args.summary_output)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
