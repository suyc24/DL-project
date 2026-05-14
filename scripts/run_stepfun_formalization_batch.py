from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = """You are given the original problem, previous accepted reasoning steps, and one current complete step.

Your task is NOT to solve the full problem.
Your task is NOT to compute the final answer.
Your task is ONLY to formalize the mathematical claim made in the current step.

Use previous steps only as assumptions.
Do not introduce quantities that are not needed for the current step.
Do not correct the current step.

You may analyze the step first. At the very end, output a final Lean section in this exact format:
FINAL_LEAN:
theorem localized_step_check ... := by sorry

Rules for the FINAL_LEAN section:
- No Markdown.
- No explanation.
- No code fences.
- Use Lean core only.
- Do not import Mathlib or any external library.
- The theorem must be named localized_step_check.
- The theorem must end with `:= by sorry`.
- Do not put any text after the Lean code.

Original problem:
{problem}

Previous accepted steps:
{previous_steps}

Current complete step:
{current_step}
"""


def format_previous_steps(previous_steps: list[str]) -> str:
    if not previous_steps:
        return "(none)"
    return "\n".join(f"Step {i + 1}: {step}" for i, step in enumerate(previous_steps))


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    user_prompt = PROMPT_TEMPLATE.format(
        problem=row.get("problem", ""),
        previous_steps=format_previous_steps(row.get("previous_steps") or []),
        current_step=row.get("current_step", ""),
    )
    return [
        {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
        {"role": "user", "content": user_prompt},
    ]


def extract_lean(text: str) -> str:
    text = text.strip()
    marker = "FINAL_LEAN:"
    marker_pos = text.rfind(marker)
    if marker_pos >= 0:
        text = text[marker_pos + len(marker) :].strip()
    fences = list(re.finditer(r"```(?:lean4?|Lean4?)?\s*(.*?)```", text, flags=re.S))
    if fences:
        text = fences[-1].group(1).strip()
    starts = [pos for pos in (text.find("import "), text.find("theorem localized_step_check")) if pos >= 0]
    if starts:
        return text[min(starts) :].strip()
    theorem_pos = text.find("theorem localized_step_check")
    if theorem_pos >= 0:
        return text[theorem_pos:].strip()
    return text


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run StepFun-Formalizer-7B on a batch of localized qwen25_fhis steps."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/root/autodl-tmp/StepFun-Formalizer-7B")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    rows = load_rows(Path(args.input), args.limit)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"loaded_model_seconds={time.time() - t0:.2f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows, start=1):
            messages = build_messages(row)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            start = time.time()
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
            payload = {
                **row,
                "prompt_version": "current_step_only_by_sorry_v1",
                "raw_generation": raw,
                "lean_code": extract_lean(raw),
                "generation_seconds": round(time.time() - start, 3),
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{i}/{len(rows)} {row.get('sample_id')} seconds={payload['generation_seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
