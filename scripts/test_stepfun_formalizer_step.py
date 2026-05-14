from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fhis.lean_verify import verify_lean_code  # noqa: E402


def strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:lean4?|Lean4?)?\s*(.*?)```", text, flags=re.S)
    if match:
        return match.group(1).strip()
    return text


def build_step_formalization_prompt(problem: str, previous_steps: list[str], current_step: str) -> list[dict[str, str]]:
    previous = "\n".join(f"Step {i + 1}: {step}" for i, step in enumerate(previous_steps))
    if not previous:
        previous = "(none)"
    user_prompt = f"""Please autoformalize the following localized mathematical reasoning step in Lean 4.

We are not formalizing the final answer; we only want to check whether this one complete step is mathematically valid under the problem statement and previous accepted steps.

Requirements:
- Output a complete Lean 4 file only.
- Use theorem name: localized_step_check.
- Do not use `sorry` or `admit`.
- Prefer Lean core constructs only. Avoid `import Mathlib` unless absolutely necessary.
- If the step cannot be faithfully formalized, output exactly:
-- formalization_failed

Original problem:
{problem}

Previous accepted steps:
{previous}

Current complete step:
{current_step}

Your code should start with:
```Lean4

```
"""
    return [
        {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
        {"role": "user", "content": user_prompt},
    ]


def sample_simple_step_from_traces(path: Path) -> dict[str, Any]:
    pattern = re.compile(r"15\s*\+\s*10\s*=\s*25")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for step in row.get("steps") or []:
                text = str(step.get("text", ""))
                if pattern.search(text):
                    return {
                        "trace_id": row["trace_id"],
                        "problem_id": row["problem_id"],
                        "problem": row["problem"],
                        "previous_steps": [],
                        "current_step": f"Step {step['index']}: {text}",
                    }
    raise RuntimeError(f"No simple sample step found in {path}")


def load_model(
    model_name: str,
    device: str,
    dtype: str,
    offload_dir: str | None,
    load_in_8bit: bool,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    torch_dtype: Any = "auto"
    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if load_in_8bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    if device == "auto":
        kwargs["device_map"] = "auto"
    elif device == "cpu":
        kwargs["device_map"] = {"": "cpu"}
    elif device.startswith("cuda"):
        kwargs["device_map"] = device
    if offload_dir:
        Path(offload_dir).mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = offload_dir
        kwargs["max_memory"] = {"cpu": "8GiB"}

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def generate_lean(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<think>"
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    target_device = getattr(model, "device", None)
    if target_device is not None:
        encoded = {k: v.to(target_device) for k, v in encoded.items()}
    do_sample = temperature > 0
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, encoded["input_ids"].shape[1] :]
    return strip_code_fence(tokenizer.decode(generated, skip_special_tokens=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use StepFun-Formalizer-7B to formalize one complete qwen25_fhis step and check it with Lean."
    )
    parser.add_argument("--model", default="stepfun-ai/StepFun-Formalizer-7B")
    parser.add_argument(
        "--traces",
        default="data_generation/qwen25_fhis/outputs/generated_traces.jsonl",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--offload-dir", default="/tmp/stepfun_formalizer_offload")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lean-executable", default="lean")
    args = parser.parse_args()

    sample = sample_simple_step_from_traces(Path(args.traces))
    messages = build_step_formalization_prompt(
        sample["problem"],
        sample["previous_steps"],
        sample["current_step"],
    )
    tokenizer, model = load_model(
        args.model,
        args.device,
        args.dtype,
        args.offload_dir,
        load_in_8bit=args.load_in_8bit,
    )
    lean_code = generate_lean(
        tokenizer,
        model,
        messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    verification = verify_lean_code(lean_code, executable=args.lean_executable, timeout_s=30.0)
    payload = {
        **sample,
        "model": args.model,
        "lean_code": lean_code,
        "verification": verification.to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if verification.status == "proved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
