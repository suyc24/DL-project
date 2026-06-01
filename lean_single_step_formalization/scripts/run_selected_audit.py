#!/usr/bin/env python3
"""Run wrapper, Lean generation, and Lean verification for preselected steps."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"

sys.path.insert(0, str(SCRIPT_DIR))
from run_loop import (
    cfg_get,
    default_lean_project_dir,
    existing_default_config,
    generate_lean_contracts,
    generate_wrapped_claims,
    parse_reasoning,
    read_config,
    read_jsonl,
    verify_lean_outputs,
    write_json,
    write_jsonl,
)


def summarize(
    selected: list[dict[str, Any]],
    wrapped: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    verification: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "selected_steps": len(selected),
        "wrapped_claims": len(wrapped),
        "wrap_valid": sum(1 for row in wrapped if row.get("wrap_valid") is True),
        "wrap_invalid": sum(1 for row in wrapped if row.get("wrap_valid") is not True),
        "low_value_steps": sum(1 for row in wrapped if row.get("low_value") is True),
        "lean_files": len(generated),
        "verified_ok": sum(1 for row in verification if row.get("ok") is True),
        "verified_failed": sum(1 for row in verification if row.get("ok") is False),
        "verified_skipped": sum(1 for row in verification if row.get("ok") is None),
        "complete_proofs": sum(
            1 for row in verification
            if row.get("ok") is True and row.get("dependency_mode") == "complete"
        ),
        "local_missing_hypotheses": sum(
            1 for row in verification
            if row.get("ok") is True and row.get("dependency_mode") == "local_missing_hypotheses"
        ),
        "global_axiom_fallbacks": sum(
            1 for row in verification
            if row.get("ok") is True and row.get("dependency_mode") == "global_axiom_fallback"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=existing_default_config())
    parser.add_argument("--selected", required=True, help="selected-steps-compatible JSONL")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--llm-provider", choices=["openai", "codex"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--llm-timeout", type=int, default=None)
    parser.add_argument("--wrap-max-tokens", type=int, default=None)
    parser.add_argument("--lean-max-tokens", type=int, default=None)
    parser.add_argument("--wrap-repair-rounds", type=int, default=None)
    parser.add_argument("--repair-rounds", type=int, default=None)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--lean-timeout", type=int, default=None)
    parser.add_argument("--skip-lean-check", action="store_true")
    parser.add_argument("--reasoning", choices=["auto", "enabled", "disabled"], default=None)
    parser.add_argument("--openai-reasoning-effort", choices=["high", "max"], default=None)
    parser.add_argument("--codex-reasoning-effort", default=None)
    parser.add_argument("--codex-sandbox", default=None)
    parser.add_argument("--codex-cwd", default=None)
    parser.add_argument("--stop-after", choices=["wrapped", "lean", "verification"], default="verification")
    args = parser.parse_args()

    config = read_config(args.config)
    llm_provider = args.llm_provider or os.environ.get("LLM_PROVIDER") or cfg_get(config, "llm.provider", "codex")
    model = args.model or cfg_get(config, "llm.model", None)
    llm_timeout = args.llm_timeout if args.llm_timeout is not None else int(cfg_get(config, "llm.timeout", 900))
    wrap_max_tokens = args.wrap_max_tokens if args.wrap_max_tokens is not None else int(cfg_get(config, "llm.wrap_max_tokens", 4096))
    lean_max_tokens = args.lean_max_tokens if args.lean_max_tokens is not None else int(cfg_get(config, "llm.lean_max_tokens", 4096))
    wrap_repair_rounds = (
        args.wrap_repair_rounds
        if args.wrap_repair_rounds is not None
        else int(cfg_get(config, "run.wrap_repair_rounds", 2))
    )
    repair_rounds = args.repair_rounds if args.repair_rounds is not None else int(cfg_get(config, "lean.repair_rounds", 3))
    project_dir = Path(args.project_dir or cfg_get(config, "paths.lean_project_dir", default_lean_project_dir()))
    lean_timeout = args.lean_timeout if args.lean_timeout is not None else int(cfg_get(config, "lean.timeout", 120))
    skip_lean_check = args.skip_lean_check or bool(cfg_get(config, "lean.skip_check", False))
    reasoning = parse_reasoning(args.reasoning if args.reasoning is not None else cfg_get(config, "llm.reasoning", None))
    openai_reasoning_effort = args.openai_reasoning_effort or cfg_get(config, "llm.openai_reasoning_effort", None)
    codex_reasoning_effort = (
        args.codex_reasoning_effort
        or os.environ.get("CODEX_REASONING_EFFORT")
        or cfg_get(config, "llm.codex_reasoning_effort", "high")
    )
    codex_sandbox = args.codex_sandbox or os.environ.get("CODEX_SANDBOX") or cfg_get(config, "llm.codex_sandbox", "read-only")
    codex_cwd = args.codex_cwd or cfg_get(config, "llm.codex_cwd", str(ROOT.parent))

    run_dir = RUNS_DIR / args.run_id
    input_dir = run_dir / "input"
    selection_dir = run_dir / "selection"
    wrapped_dir = run_dir / "wrapped_claims"
    lean_dir = run_dir / "lean"
    verification_dir = run_dir / "verification"
    run_dir.mkdir(parents=True, exist_ok=True)

    selected = read_jsonl(Path(args.selected))
    write_jsonl(selected, input_dir / "selected_steps.jsonl")
    write_jsonl(selected, selection_dir / "steps_selected.jsonl")
    write_json(
        {
            "run_id": args.run_id,
            "selected": args.selected,
            "llm_provider": "mock" if args.mock else llm_provider,
            "model": "mock" if args.mock else model,
            "llm_timeout": llm_timeout,
            "wrap_max_tokens": wrap_max_tokens,
            "lean_max_tokens": lean_max_tokens,
            "project_dir": str(project_dir),
            "lean_timeout": lean_timeout,
            "repair_rounds": repair_rounds,
            "skip_lean_check": skip_lean_check,
            "reasoning": reasoning,
            "openai_reasoning_effort": openai_reasoning_effort,
            "codex_reasoning_effort": codex_reasoning_effort,
            "codex_sandbox": codex_sandbox,
            "codex_cwd": codex_cwd,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        run_dir / "run_config.json",
    )

    wrapped = generate_wrapped_claims(
        selected,
        provider=llm_provider,
        model=model,
        mock=args.mock,
        out_dir=wrapped_dir,
        llm_timeout=llm_timeout,
        wrap_max_tokens=wrap_max_tokens,
        wrap_repair_rounds=wrap_repair_rounds,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
    )
    write_jsonl(wrapped, wrapped_dir / "wrapped_claims.jsonl")
    if args.stop_after == "wrapped":
        summary = summarize(selected, wrapped, [], [])
        write_json(summary, run_dir / "run_summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Run directory: {run_dir}")
        return

    generated = generate_lean_contracts(
        wrapped,
        provider=llm_provider,
        model=model,
        mock=args.mock,
        out_dir=lean_dir,
        llm_timeout=llm_timeout,
        lean_max_tokens=lean_max_tokens,
        project_dir=project_dir,
        lean_timeout=lean_timeout,
        repair_rounds=repair_rounds,
        skip_lean_check=skip_lean_check,
        reasoning=reasoning,
        openai_reasoning_effort=openai_reasoning_effort,
        codex_reasoning_effort=codex_reasoning_effort,
        codex_sandbox=codex_sandbox,
        codex_cwd=codex_cwd,
    )
    write_jsonl(generated, lean_dir / "lean_generation_manifest.jsonl")
    if args.stop_after == "lean":
        summary = summarize(selected, wrapped, generated, [])
        write_json(summary, run_dir / "run_summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Run directory: {run_dir}")
        return

    verification = verify_lean_outputs(
        generated,
        project_dir=project_dir,
        timeout=lean_timeout,
        skip=skip_lean_check,
    )
    write_json(verification, verification_dir / "verification.json")
    summary = summarize(selected, wrapped, generated, verification)
    write_json(summary, run_dir / "run_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
