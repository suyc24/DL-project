from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fhis.localized_verify import extract_atomic_claims, verify_localized_step  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("version https://git-lfs.github.com"):
                continue
            if line.startswith("oid ") or line.startswith("size "):
                continue
            rows.append(json.loads(line))
    return rows


def load_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {str(row.get("trace_id")): row for row in read_jsonl(path)}


def fhis_position(label: dict[str, Any] | None, step_index: int) -> str:
    if not label:
        return "unlabeled"
    if label.get("final_correct") is True:
        return "correct_trace"
    first_invalid = label.get("first_invalid_step")
    if first_invalid is None:
        return "wrong_trace_unknown_fhis"
    first_invalid = int(first_invalid)
    if step_index < first_invalid:
        return "pre_fhis"
    if step_index == first_invalid:
        return "fhis"
    return "post_fhis"


def inc(counter: Counter[str], key: str, amount: int = 1) -> None:
    counter[key] += amount


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic localized arithmetic/approximation checks on qwen25 traces."
    )
    parser.add_argument(
        "--traces",
        default="data_generation/qwen25_fhis/outputs/generated_traces.jsonl",
    )
    parser.add_argument(
        "--labels",
        default="data_generation/qwen25_fhis/labels/fhis_labels_train_high.jsonl",
    )
    parser.add_argument("--limit-traces", type=int)
    parser.add_argument("--max-claims-per-step", type=int, default=16)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-limit-steps", type=int, default=50)
    parser.add_argument("--lean-executable", default="lean")
    parser.add_argument("--lean-workdir", default=None)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    args = parser.parse_args()

    traces = read_jsonl(Path(args.traces))
    if args.limit_traces is not None:
        traces = traces[: args.limit_traces]
    labels = load_labels(Path(args.labels) if args.labels else None)

    totals: Counter[str] = Counter()
    by_position: dict[str, Counter[str]] = defaultdict(Counter)
    by_rough: dict[str, Counter[str]] = defaultdict(Counter)
    verify_statuses: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    verified_steps = 0

    output_rows: list[dict[str, Any]] = []
    for trace in traces:
        trace_id = str(trace.get("trace_id"))
        label = labels.get(trace_id)
        rough = trace.get("rough_final_correct")
        rough_key = "rough_correct" if rough is True else "rough_wrong" if rough is False else "rough_unknown"
        steps = trace.get("steps") or []
        inc(totals, "traces")
        inc(totals, "steps", len(steps))
        for step in steps:
            step_index = int(step.get("index", 0))
            step_text = str(step.get("text", ""))
            position = fhis_position(label, step_index)
            claims = extract_atomic_claims(step_text, max_claims=args.max_claims_per_step)
            has_claim = bool(claims)
            false_claims = [claim for claim in claims if not claim.expected_truth]
            true_claims = [claim for claim in claims if claim.expected_truth]

            inc(totals, "steps_with_claims", int(has_claim))
            inc(totals, "claims", len(claims))
            inc(totals, "true_claims", len(true_claims))
            inc(totals, "false_claims", len(false_claims))
            inc(totals, "steps_with_false_claims", int(bool(false_claims)))

            for counter in (by_position[position], by_rough[rough_key]):
                inc(counter, "steps")
                inc(counter, "steps_with_claims", int(has_claim))
                inc(counter, "claims", len(claims))
                inc(counter, "true_claims", len(true_claims))
                inc(counter, "false_claims", len(false_claims))
                inc(counter, "steps_with_false_claims", int(bool(false_claims)))

            verification = None
            if args.verify and has_claim and verified_steps < args.verify_limit_steps:
                result = verify_localized_step(
                    step_text,
                    workdir=args.lean_workdir,
                    executable=args.lean_executable,
                    timeout_s=args.timeout_s,
                    max_claims=args.max_claims_per_step,
                )
                verification = result.to_dict()
                inc(verify_statuses, result.status)
                verified_steps += 1

            if false_claims and len(examples) < 20:
                examples.append(
                    {
                        "trace_id": trace_id,
                        "problem_id": trace.get("problem_id"),
                        "step_index": step_index,
                        "position": position,
                        "rough_final_correct": rough,
                        "step_text": step_text[:1200],
                        "false_claims": [claim.to_dict() for claim in false_claims],
                    }
                )

            if args.output is not None and has_claim:
                output_rows.append(
                    {
                        "trace_id": trace_id,
                        "problem_id": trace.get("problem_id"),
                        "step_index": step_index,
                        "position": position,
                        "rough_final_correct": rough,
                        "claims": [claim.to_dict() for claim in claims],
                        "verification": verification,
                    }
                )

    def summarize_counter(counter: Counter[str]) -> dict[str, Any]:
        steps = counter["steps"]
        claims = counter["claims"]
        return {
            **dict(counter),
            "step_claim_coverage": rate(counter["steps_with_claims"], steps),
            "false_claim_step_rate": rate(counter["steps_with_false_claims"], steps),
            "false_claim_rate": rate(counter["false_claims"], claims),
        }

    summary = {
        "traces_path": args.traces,
        "labels_path": args.labels,
        "limit_traces": args.limit_traces,
        "verify": args.verify,
        "verified_steps": verified_steps,
        "totals": summarize_counter(totals),
        "by_position": {key: summarize_counter(counter) for key, counter in sorted(by_position.items())},
        "by_rough": {key: summarize_counter(counter) for key, counter in sorted(by_rough.items())},
        "verify_statuses": dict(verify_statuses),
        "false_claim_examples": examples,
    }

    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.summary_output is not None:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
