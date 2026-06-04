#!/usr/bin/env python3
"""Run synthetic missing-premise use cases through GV v2."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"

sys.path.insert(0, str(SCRIPT_DIR))
from run_loop import cfg_get, parse_reasoning, read_config, write_json, write_jsonl  # noqa: E402
import run_adversarial_game_gv_v2 as gv2  # noqa: E402


def templates() -> list[dict[str, Any]]:
    return [
        {
            "name": "real_successive_equalities",
            "question": "Synthetic user-premise use case over real numbers. The only available assumptions are the supplied local premises.",
            "target_step": "Using only the supplied local equations, substitute successively to obtain \\(c=3\\).",
            "premises": [
                ("P1", "The supplied local premise is \\(a=1\\)."),
                ("P2", "The supplied local premise is \\(b=a+1\\)."),
                ("P3", "The supplied local premise is \\(c=b+1\\)."),
            ],
            "proof": "Substitute the supplied equations in order.",
            "conclusion": "\\(c=3\\)",
        },
        {
            "name": "real_order_chain",
            "question": "Synthetic user-premise use case over real numbers. The only available assumptions are the supplied local premises.",
            "target_step": "Using only the supplied local inequalities, chain the bounds to obtain \\(z\\ge10\\).",
            "premises": [
                ("P1", "The supplied local premise is \\(x\\ge2\\)."),
                ("P2", "The supplied local premise is \\(y\\ge x+3\\)."),
                ("P3", "The supplied local premise is \\(z\\ge y+5\\)."),
            ],
            "proof": "Chain the supplied lower bounds.",
            "conclusion": "\\(z\\ge10\\)",
        },
        {
            "name": "nat_successive_equalities",
            "question": "Synthetic user-premise use case over natural numbers. The only available assumptions are the supplied local premises.",
            "target_step": "Using only the supplied local equations, calculate \\(k=36\\).",
            "premises": [
                ("P1", "The supplied local premise is \\(m=6\\)."),
                ("P2", "The supplied local premise is \\(n=2m\\)."),
                ("P3", "The supplied local premise is \\(k=3n\\)."),
            ],
            "proof": "Substitute the supplied equations and evaluate.",
            "conclusion": "\\(k=36\\)",
        },
        {
            "name": "set_membership_chain",
            "question": "Synthetic user-premise use case for sets. The only available assumptions are the supplied local premises.",
            "target_step": "Using only the supplied local set-inclusion premises, conclude \\(x\\in C\\).",
            "premises": [
                ("P1", "The supplied local premise is \\(A\\subseteq B\\)."),
                ("P2", "The supplied local premise is \\(B\\subseteq C\\)."),
                ("P3", "The supplied local premise is \\(x\\in A\\)."),
            ],
            "proof": "Apply the two inclusions to the supplied membership.",
            "conclusion": "\\(x\\in C\\)",
        },
        {
            "name": "real_transitive_equality",
            "question": "Synthetic user-premise use case over real numbers. The only available assumptions are the supplied local premises.",
            "target_step": "Using only the supplied local equalities, conclude \\(p=7\\).",
            "premises": [
                ("P1", "The supplied local premise is \\(p=q\\)."),
                ("P2", "The supplied local premise is \\(q=r\\)."),
                ("P3", "The supplied local premise is \\(r=7\\)."),
            ],
            "proof": "Chain the supplied equalities.",
            "conclusion": "\\(p=7\\)",
        },
        {
            "name": "positive_product",
            "question": "Synthetic user-premise use case over real numbers. The only available assumptions are the supplied local premises.",
            "target_step": "Using only the supplied local premises, prove \\(z>0\\).",
            "premises": [
                ("P1", "The supplied local premise is \\(x>0\\)."),
                ("P2", "The supplied local premise is \\(y>0\\)."),
                ("P3", "The supplied local premise is \\(z=x y\\)."),
            ],
            "proof": "Use positivity of the two factors and substitute the product expression for z.",
            "conclusion": "\\(z>0\\)",
        },
        {
            "name": "le_transitive_chain",
            "question": "Synthetic user-premise use case over real numbers. The only available assumptions are the supplied local premises.",
            "target_step": "Using only the supplied local order premises, conclude \\(r\\le u\\).",
            "premises": [
                ("P1", "The supplied local premise is \\(r\\le s\\)."),
                ("P2", "The supplied local premise is \\(s\\le t\\)."),
                ("P3", "The supplied local premise is \\(t\\le u\\)."),
            ],
            "proof": "Apply transitivity of the supplied inequalities.",
            "conclusion": "\\(r\\le u\\)",
        },
        {
            "name": "real_product_evaluation",
            "question": "Synthetic user-premise use case over real numbers. The only available assumptions are the supplied local premises.",
            "target_step": "Using only the supplied local equations, evaluate \\(z=7\\).",
            "premises": [
                ("P1", "The supplied local premise is \\(x=2\\)."),
                ("P2", "The supplied local premise is \\(y=3\\)."),
                ("P3", "The supplied local premise is \\(z=xy+1\\)."),
            ],
            "proof": "Substitute the supplied values into the supplied expression for z.",
            "conclusion": "\\(z=7\\)",
        },
        {
            "name": "integer_additive_chain",
            "question": "Synthetic user-premise use case over integers. The only available assumptions are the supplied local premises.",
            "target_step": "Using only the supplied local equations, calculate \\(w=4\\).",
            "premises": [
                ("P1", "The supplied local premise is \\(u=1\\)."),
                ("P2", "The supplied local premise is \\(v=u+1\\)."),
                ("P3", "The supplied local premise is \\(w=v+2\\)."),
            ],
            "proof": "Substitute the supplied integer equations and evaluate.",
            "conclusion": "\\(w=4\\)",
        },
        {
            "name": "strict_order_chain",
            "question": "Synthetic user-premise use case over real numbers. The only available assumptions are the supplied local premises.",
            "target_step": "Using only the supplied local strict inequalities, conclude \\(a<d\\).",
            "premises": [
                ("P1", "The supplied local premise is \\(a<b\\)."),
                ("P2", "The supplied local premise is \\(b<c\\)."),
                ("P3", "The supplied local premise is \\(c<d\\)."),
            ],
            "proof": "Apply transitivity of the supplied strict inequalities.",
            "conclusion": "\\(a<d\\)",
        },
    ]


def make_row(template: dict[str, Any], missing_count: int, group_idx: int) -> dict[str, Any]:
    kept = template["premises"][: max(0, len(template["premises"]) - missing_count)]
    removed = template["premises"][len(kept) :]
    premises = [{"id": pid, "text": text, "source": "problem"} for pid, text in kept]
    premise_ids = [pid for pid, _ in kept]
    name = template["name"]
    return {
        "id": f"usecase_g{group_idx:02d}_{name}_missing_{missing_count}",
        "source_id": "synthetic_missing_premise_usecase",
        "question": template["question"],
        "chain_id": group_idx,
        "step_id": missing_count,
        "previous_steps": [],
        "context_steps": [{"step_id": missing_count, "text": template["target_step"], "is_selected": True}],
        "target_step": template["target_step"],
        "original_cot": template["target_step"],
        "gold_verdict": "valid" if missing_count == 0 else "invalid_missing_premises",
        "manual_annotation": {
            "source": "synthetic_user_premise_usecase",
            "label": "valid" if missing_count == 0 else "invalid_missing_premises",
            "annotation": {
                "label": "valid" if missing_count == 0 else "invalid",
                "reason": f"{missing_count} necessary premise(s) removed from the local premise list.",
                "confidence": 5,
            },
        },
        "step_decomposition": {
            "premises": premises,
            "proof_steps": [
                {
                    "id": "S1",
                    "text": template["proof"],
                    "uses": premise_ids,
                    "yields": template["conclusion"],
                }
            ],
            "conclusion": template["conclusion"],
            "confidence": 4,
        },
        "premise_usecase": {
            "template": name,
            "missing_count": missing_count,
            "provided_premise_ids": premise_ids,
            "removed_premises": [{"id": pid, "text": text} for pid, text in removed],
        },
    }


def final_generator_report(lean: dict[str, Any]) -> dict[str, Any]:
    events = lean.get("generator_events") or []
    if not events:
        return {}
    report = events[-1].get("report") if isinstance(events[-1], dict) else {}
    return report if isinstance(report, dict) else {}


def final_review(lean: dict[str, Any]) -> dict[str, Any]:
    decisions = lean.get("verifier_decisions") or []
    if not decisions:
        return {}
    decision = decisions[-1].get("decision") if isinstance(decisions[-1], dict) else {}
    return decision if isinstance(decision, dict) else {}


def run_one(
    row: dict[str, Any],
    *,
    run_dir: Path,
    provider: str,
    model: str | None,
    llm_timeout: int,
    lean_max_tokens: int,
    judge_max_tokens: int,
    project_dir: Path,
    lean_timeout: int,
    repair_rounds: int,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
    codex_reasoning_effort: str,
    codex_sandbox: str,
    codex_cwd: str,
    mock: bool,
) -> dict[str, Any]:
    case_dir = run_dir / row["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(row, case_dir / "row.json")
    thread_dir = case_dir / "threads"
    thread_dir.mkdir(parents=True, exist_ok=True)
    initial = {
        "verdict": "valid",
        "reason": "Use-case experiment: determine whether the user-supplied local premises suffice.",
        "confidence": 4,
    }
    started = time.time()
    lean = gv2.run_gv_lean_assist_v2(
        row,
        round_dir=case_dir,
        provider=provider,
        model=model,
        mock=mock,
        llm_timeout=llm_timeout,
        lean_max_tokens=lean_max_tokens,
        judge_max_tokens=judge_max_tokens,
        project_dir=project_dir,
        lean_timeout=lean_timeout,
        repair_rounds=repair_rounds,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
        generator_thread_file=str(thread_dir / "generator.thread"),
        verifier_thread_dir=thread_dir / "verifier_reviews",
        initial_override=initial,
        initial_errors_override=[],
        step_decomposition_override=row["step_decomposition"],
        step_decomposition_errors_override=[],
    )
    judgment = lean.get("judgment") or {}
    gen = final_generator_report(lean)
    review = final_review(lean)
    record = {
        "case_id": row["id"],
        "template": row["premise_usecase"]["template"],
        "missing_count": row["premise_usecase"]["missing_count"],
        "provided_premise_ids": row["premise_usecase"]["provided_premise_ids"],
        "removed_premises": row["premise_usecase"]["removed_premises"],
        "gv_verdict": judgment.get("verdict"),
        "gv_stage": lean.get("stage"),
        "gv_reason": judgment.get("reason"),
        "lean_evidence": judgment.get("lean_evidence"),
        "final_compile_ok": gen.get("compile_ok"),
        "final_compile_reason": gen.get("reason"),
        "generator_attempts": len(lean.get("generator_events") or []),
        "verifier_reviews": len(lean.get("verifier_decisions") or []),
        "final_review_action": review.get("action"),
        "final_review_reason": review.get("reason"),
        "elapsed_sec": round(time.time() - started, 3),
        "case_dir": str(case_dir),
    }
    write_json(record, case_dir / "case_result.json")
    return record


def write_report(run_dir: Path, records: list[dict[str, Any]]) -> None:
    by_missing = defaultdict(list)
    for record in records:
        by_missing[int(record["missing_count"])].append(record)
    lines = ["# Missing-Premise Use-Case GV v2 Run", ""]
    for missing in sorted(by_missing):
        items = by_missing[missing]
        verdicts = Counter(str(item.get("gv_verdict")) for item in items)
        compiles = Counter(str(item.get("final_compile_ok")) for item in items)
        lines.append(
            f"- missing={missing}: cases={len(items)}, verdicts={dict(verdicts)}, compile={dict(compiles)}"
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--parallel-cases", type=int, default=10)
    parser.add_argument("--llm-provider", choices=["openai", "codex"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--llm-timeout", type=int, default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=10000)
    parser.add_argument("--lean-max-tokens", type=int, default=None)
    parser.add_argument("--repair-rounds", type=int, default=3)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--lean-timeout", type=int, default=None)
    parser.add_argument("--reasoning", choices=["auto", "enabled", "disabled"], default=None)
    parser.add_argument("--openai-reasoning-effort", choices=["high", "max"], default=None)
    parser.add_argument("--codex-reasoning-effort", default=None)
    parser.add_argument("--codex-sandbox", default=None)
    parser.add_argument("--codex-cwd", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    provider = args.llm_provider or os.environ.get("LLM_PROVIDER") or cfg_get(config, "llm.provider", "codex")
    model = args.model or cfg_get(config, "llm.model", None)
    llm_timeout = args.llm_timeout if args.llm_timeout is not None else int(cfg_get(config, "llm.timeout", 900))
    lean_max_tokens = args.lean_max_tokens if args.lean_max_tokens is not None else int(cfg_get(config, "llm.lean_max_tokens", 4096))
    project_dir = Path(args.project_dir or cfg_get(config, "paths.lean_project_dir", "/root/mathlib4"))
    lean_timeout = args.lean_timeout if args.lean_timeout is not None else int(cfg_get(config, "lean.timeout", 120))
    reasoning = parse_reasoning(args.reasoning if args.reasoning is not None else cfg_get(config, "llm.reasoning", None))
    openai_reasoning_effort = args.openai_reasoning_effort or cfg_get(config, "llm.openai_reasoning_effort", None)
    codex_reasoning_effort = args.codex_reasoning_effort or os.environ.get("CODEX_REASONING_EFFORT") or cfg_get(config, "llm.codex_reasoning_effort", "high")
    codex_sandbox = args.codex_sandbox or os.environ.get("CODEX_SANDBOX") or cfg_get(config, "llm.codex_sandbox", "danger-full-access")
    codex_cwd = args.codex_cwd or cfg_get(config, "llm.codex_cwd", str(ROOT.parent))

    selected_templates = templates()[: args.groups]
    rows = [
        make_row(template, missing_count, idx)
        for idx, template in enumerate(selected_templates, start=1)
        for missing_count in [0, 1, 2, 3]
    ]
    run_dir = RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, run_dir / "input_rows.jsonl")
    write_json(
        {
            "run_id": args.run_id,
            "groups": len(selected_templates),
            "cases": len(rows),
            "parallel_cases": args.parallel_cases,
            "provider": "mock" if args.mock else provider,
            "model": "mock" if args.mock else model,
            "codex_reasoning_effort": codex_reasoning_effort,
            "prompt_dir": str(gv2.PROMPT_DIR),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        run_dir / "run_config.json",
    )

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel_cases) as executor:
        futures = [
            executor.submit(
                run_one,
                row,
                run_dir=run_dir,
                provider=provider,
                model=model,
                llm_timeout=llm_timeout,
                lean_max_tokens=lean_max_tokens,
                judge_max_tokens=args.judge_max_tokens,
                project_dir=project_dir,
                lean_timeout=lean_timeout,
                repair_rounds=args.repair_rounds,
                reasoning=reasoning,
                openai_reasoning_effort=openai_reasoning_effort,
                codex_reasoning_effort=codex_reasoning_effort,
                codex_sandbox=codex_sandbox,
                codex_cwd=codex_cwd,
                mock=args.mock,
            )
            for row in rows
        ]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                json.dumps(
                    {
                        "case_id": record["case_id"],
                        "missing": record["missing_count"],
                        "compile_ok": record["final_compile_ok"],
                        "verdict": record["gv_verdict"],
                        "stage": record["gv_stage"],
                        "action": record["final_review_action"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    records.sort(key=lambda item: (item["template"], item["missing_count"]))
    write_json(records, run_dir / "case_results.json")
    by_missing: dict[str, Any] = {}
    for missing in [0, 1, 2, 3]:
        subset = [record for record in records if record["missing_count"] == missing]
        by_missing[str(missing)] = {
            "cases": len(subset),
            "verdict_counts": dict(Counter(str(record.get("gv_verdict")) for record in subset)),
            "compile_counts": dict(Counter(str(record.get("final_compile_ok")) for record in subset)),
            "case_ids": [record["case_id"] for record in subset],
        }
    summary = {"run_dir": str(run_dir), "cases": len(records), "groups": len(selected_templates), "by_missing": by_missing}
    write_json(summary, run_dir / "summary.json")
    write_report(run_dir, records)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
