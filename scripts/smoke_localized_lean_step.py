from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fhis.lean_verify import verify_lean_code  # noqa: E402


CASES = [
    {
        "name": "arithmetic_step",
        "problem": "Compute 1 + 1.",
        "localized_step": "Step 1: We compute 1 + 1 = 2.",
        "lean_code": """
example : 1 + 1 = 2 := by
  rfl
""",
        "expected_status": "proved",
    },
    {
        "name": "linear_equation_step",
        "problem": "Given x = 4, compute x + 3.",
        "localized_step": "Step 1: Substituting x = 4 gives x + 3 = 7.",
        "lean_code": """
example (x : Nat) (h : x = 4) : x + 3 = 7 := by
  subst x
  rfl
""",
        "expected_status": "proved",
    },
    {
        "name": "detect_invalid_step",
        "problem": "Compute 1 + 1.",
        "localized_step": "Step 1: We compute 1 + 1 = 3.",
        "lean_code": """
example : 1 + 1 = 3 := by
  rfl
""",
        "expected_status": "failed",
    },
]


def main() -> int:
    rows = []
    ok = True
    for case in CASES:
        result = verify_lean_code(case["lean_code"], timeout_s=10.0)
        passed = result.status == case["expected_status"]
        ok = ok and passed
        rows.append(
            {
                "name": case["name"],
                "problem": case["problem"],
                "localized_step": case["localized_step"],
                "expected_status": case["expected_status"],
                "actual_status": result.status,
                "passed": passed,
                "stderr": result.stderr.strip()[:1000],
            }
        )

    print(json.dumps({"passed": ok, "cases": rows}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
