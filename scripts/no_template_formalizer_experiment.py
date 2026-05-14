from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fhis.semantic_lean_verify import SemanticLeanTask, SemanticLeanVerifier  # noqa: E402


@dataclass(frozen=True)
class Case:
    case_id: str
    expected_status: str
    problem: str
    prior_steps: list[str]
    current_step: str


def cases() -> list[Case]:
    return [
        Case(
            "terminating_unit_fraction_bad_n10",
            "invalid",
            "Find the least positive integer n such that exactly half of the fractions 1/k for 1 <= k <= n have terminating decimal expansions.",
            [
                "A unit fraction 1/k terminates in base 10 iff k divides a power of 10.",
                "We should count terminating fractions among k = 1, ..., n.",
            ],
            "Since n/2 = 5, n = 10 works.",
        ),
        Case(
            "sum_squares_bad_k48",
            "invalid",
            "Find the smallest positive integer k such that 1^2 + 2^2 + ... + k^2 is divisible by 200.",
            ["Use the exact finite sum of squares, not a decimal approximation."],
            "k = 48 is valid because the sum of squares is a multiple of 200.",
        ),
        Case(
            "vector_cos_bad_five_six",
            "invalid",
            "Two nonzero vectors p and q satisfy |p| = 1, |q| = 2, and p dot q = 3/4. Determine cos(theta), where theta is the angle between p and q.",
            [
                "cos(theta) = (p dot q) / (|p| |q|).",
                "Substituting the known values gives cos(theta) = (3/4)/(1*2).",
            ],
            "Therefore cos(theta) = 5/6.",
        ),
        Case(
            "crt_bad_pairwise_coprime",
            "invalid",
            "For k = 1, ..., 20, N and a_k are congruent modulo k. A solution claims CRT gives a unique N modulo 20!.",
            ["CRT with uniqueness modulo the product requires pairwise coprime moduli."],
            "This is possible by the Chinese Remainder Theorem, since the moduli 1,2,...,20 are pairwise coprime.",
        ),
        Case(
            "motion_relative_speed_valid",
            "valid",
            "Sam rides toward Marty at 15 km/hr, Marty rides toward Sam at 10 km/hr, and they start 100 km apart. Find Sam's travel distance when they meet.",
            ["They are moving toward each other, so relative speed is the sum."],
            "The relative speed is 15 + 10 = 25 km/hr.",
        ),
        Case(
            "permutation_absdiff_valid",
            "valid",
            "Let (a1,a2,a3,a4) range over permutations of {1,2,3,4}. Compute the average contribution of |a1-a2|.",
            ["The unordered pair {a1,a2} is uniformly distributed over the six two-element subsets."],
            "The six absolute differences are 1,2,3,1,2,1, so the expected value of |a1-a2| is 10/6 = 5/3.",
        ),
    ]


def prior_text(prior_steps: list[str]) -> str:
    return "\n".join(f"- {step}" for step in prior_steps) or "- none"


def build_direct_prompt(case: Case) -> str:
    return f"""You are autoformalizing one natural-language mathematical reasoning step into Lean 4.

Translate the complete CURRENT STEP into Lean. Include every definition and
condition needed for Lean to decide whether that exact step follows from the
PROBLEM and ACCEPTED PRIOR STEPS. Do not prove anything. Do not refute anything.
Do not replace the step by a smaller supporting calculation.
The current step may be false; formalize the proposition it asserts anyway.
When the step says something "works" or is "valid", use the PROBLEM and
ACCEPTED PRIOR STEPS to expand what property must work, rather than encoding
only the arithmetic phrase used as support.

Output exactly:

FINAL_COMMON_CODE:
<Lean 4 declarations, ending with `def current_step_claim : Prop := ...`>

General requirements:
- The Lean code must be self-contained.
- Define every identifier used in `current_step_claim`.
- Do not use theorem, lemma, example, proof terms, sorry, admit, axiom, constant, opaque, or unsafe.
- Do not import any library, including Mathlib.
- Do not use set-builder notation, Finset, big operators, Real, vector libraries, or theorem-style parameters.
- Stay inside this generic executable Lean-core subset when possible:
  `Nat`, `Int`, `Bool`, `List`, `List.range`, `map`, `filter`,
  `foldl`, `all`, `any`, `if`, `%`, `∣`, `Nat.gcd`, `decide`,
  products like `Nat × Nat`, and ordinary recursive-free definitions.
- Lean 4 core `List.range` has one argument and starts at 0. To encode
  1..20, use `List.map (fun i => i + 1) (List.range 20)` or an explicit list;
  never write `List.range 1 21`.
- Prefer executable finite definitions when the step is finite.
- If exact fractions are needed and the environment lacks libraries, use exact integer or natural-number equalities such as cross multiplication.
- For averages or ratios, encode equality by clearing denominators.
- If you cannot faithfully formalize the complete step, output `FINAL_COMMON_CODE:` followed by no code.

PROBLEM:
{case.problem}

ACCEPTED PRIOR STEPS:
{prior_text(case.prior_steps)}

CURRENT STEP:
{case.current_step}
"""


def build_contract_prompt(case: Case) -> str:
    return f"""Analyze one mathematical reasoning step for autoformalization.

Return only JSON with these keys:
- "claim": the complete mathematical assertion made by the current step.
- "must_encode": a list of concepts, quantities, and conditions that any faithful Lean statement must encode.
- "allowed_abstractions": concepts that may be represented by definitions rather than full libraries.
- "failure_if_missing": what would make a Lean statement unfaithful.

Do not write Lean code.

PROBLEM:
{case.problem}

ACCEPTED PRIOR STEPS:
{prior_text(case.prior_steps)}

CURRENT STEP:
{case.current_step}
"""


def build_from_contract_prompt(case: Case, contract: str) -> str:
    return f"""You are given a semantic contract for one reasoning step. Convert it into self-contained Lean 4 declarations.

Do not use examples or theorem templates. Do not prove anything.

Output exactly:

FINAL_COMMON_CODE:
<Lean 4 declarations, ending with `def current_step_claim : Prop := ...`>

Semantic contract:
{contract}

Original problem:
{case.problem}

Accepted prior steps:
{prior_text(case.prior_steps)}

Current step:
{case.current_step}
"""


def build_repair_prompt(case: Case, code: str, lean_error: str) -> str:
    return f"""Repair this Lean 4 formalization without changing the mathematical claim.

The previous Lean code failed to compile or verify. Preserve the complete
meaning of the CURRENT STEP; only fix syntax, missing definitions, types, or
executable encodings. Do not introduce theorem, lemma, example, proof terms,
sorry, admit, axiom, constant, opaque, or unsafe.

Output exactly:

FINAL_COMMON_CODE:
<repaired self-contained Lean declarations ending with `def current_step_claim : Prop := ...`>

PROBLEM:
{case.problem}

ACCEPTED PRIOR STEPS:
{prior_text(case.prior_steps)}

CURRENT STEP:
{case.current_step}

Previous Lean code:
```lean
{code}
```

Lean feedback:
```text
{lean_error[-4000:]}
```
"""


def build_alignment_prompt(case: Case, code: str) -> str:
    return f"""Judge whether this Lean statement faithfully formalizes the complete current natural-language step.

Return only JSON:
{{
  "faithful": true or false,
  "missing": [short strings],
  "invented": [short strings],
  "reason": "one short sentence"
}}

Mark false if the Lean code proves/refutes a different claim, omits a central
condition, assumes the current step, or checks only a smaller supporting atom.
Judge against what the CURRENT STEP actually asserts. Use the problem and
accepted prior steps only to disambiguate terms such as "works", "valid", or
"this"; do not demand final-answer minimality, uniqueness, or extra problem
requirements unless the CURRENT STEP itself asserts them.
Helper definitions, cross multiplication for rational equalities, and finite
executable expansions of a concrete finite claim are allowed when they preserve
the same mathematical assertion. Do not mark helper variable names as invented
merely because they are implementation names.

PROBLEM:
{case.problem}

ACCEPTED PRIOR STEPS:
{prior_text(case.prior_steps)}

CURRENT STEP:
{case.current_step}

Lean code:
```lean
{code}
```
"""


def build_semantic_repair_prompt(case: Case, code: str, alignment: dict[str, Any]) -> str:
    return f"""Repair this Lean 4 statement so it faithfully formalizes the complete CURRENT STEP.

The previous Lean code compiled or nearly compiled but was semantically
unfaithful. Use the semantic judge feedback below to add missing concepts,
remove invented concepts, and avoid checking only a smaller supporting atom.
Do not use theorem, lemma, example, proof terms, sorry, admit, axiom, constant,
opaque, unsafe, imports, or Mathlib. Do not solve a different problem.

Output exactly:

FINAL_COMMON_CODE:
<self-contained Lean declarations ending with `def current_step_claim : Prop := ...`>

PROBLEM:
{case.problem}

ACCEPTED PRIOR STEPS:
{prior_text(case.prior_steps)}

CURRENT STEP:
{case.current_step}

Previous Lean code:
```lean
{code}
```

Semantic judge feedback:
```json
{json.dumps(alignment, ensure_ascii=False, indent=2)[:4000]}
```
"""


def load_model(model_name: str, dtype: str, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map: dict[str, Any] = {"auto": "auto", "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if dtype != "auto":
        kwargs["dtype"] = dtype_map[dtype]
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device != "auto":
        model.to(device)
    model.eval()
    return tokenizer, model


def generate(tokenizer: Any, model: Any, prompt: str, max_new_tokens: int, assistant_prefix: str = "") -> str:
    import torch

    messages = [
        {"role": "system", "content": "You are precise, conservative, and fluent in Lean 4."},
        {"role": "user", "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt
    text += assistant_prefix
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    target_device = getattr(model, "device", None)
    if target_device is not None:
        enc = {k: v.to(target_device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)


def strip_fence(text: str) -> str:
    fences = list(re.finditer(r"```(?:lean4?|Lean4?|Lean|lean)?\s*(.*?)```", text, flags=re.S))
    if fences:
        return fences[-1].group(1).strip()
    return text.strip()


def extract_common_code(text: str) -> str | None:
    text = text.strip()
    chunks: list[str] = []
    marker = re.search(r"FINAL_COMMON_CODE:\s*(.*)", text, flags=re.S)
    if marker:
        chunks.append(marker.group(1))
    if "</think>" in text:
        chunks.append(text.split("</think>", 1)[0])
        chunks.append(text.split("</think>", 1)[1])
    chunks.append(text)
    for chunk in chunks:
        code = strip_fence(chunk)
        starts = [p for p in (code.find("set_option"), code.find("import "), code.find("def ")) if p >= 0]
        if starts:
            code = code[min(starts) :]
        code = trim_after_code(code)
        code = generic_normalize(code)
        if "def current_step_claim" in code and not forbidden_in_statement(code):
            return code.strip()
    return None


def trim_after_code(code: str) -> str:
    markers = ["\n\nLet me ", "\n\nThe problem ", "\n\nThis code ", "\n</think>", "\n# "]
    end = len(code)
    for marker in markers:
        pos = code.find(marker)
        if pos >= 0:
            end = min(end, pos)
    return code[:end].strip()


def generic_normalize(code: str) -> str:
    code = code.replace("ℕ", "Nat").replace("ℤ", "Int")
    code = code.replace("ℚ", "Rat").replace("ℝ", "Real")
    # Do not erase Mathlib-dependent content. If the environment cannot compile
    # it, the verifier/repair loop should expose that honestly.
    return code.strip()


def forbidden_in_statement(code: str) -> bool:
    return bool(re.search(r"\b(theorem|lemma|example|sorry|admit|axiom|constant|opaque|unsafe)\b", code))


def lean_def_names(code: str) -> list[str]:
    return [name for name in re.findall(r"(?m)^\s*def\s+([A-Za-z_][A-Za-z0-9_']*)\b", code) if name != "current_step_claim"]


def build_task(case: Case, common_code: str, proof_method: str = "native") -> SemanticLeanTask:
    defs = " ".join(["current_step_claim", *reversed(lean_def_names(common_code))])
    unfold = f"  unfold {defs}\n" if defs else ""
    if proof_method == "native":
        tactic = f"{unfold}  native_decide"
    elif proof_method == "simp":
        tactic = f"{unfold}  simp"
    else:
        tactic = f"{unfold}  native_decide"
    prove = f"{common_code}\n\ntheorem current_step_valid : current_step_claim := by\n{tactic}\n"
    refute = f"{common_code}\n\ntheorem current_step_invalid : ¬ current_step_claim := by\n{tactic}\n"
    return SemanticLeanTask(case.problem, case.prior_steps, case.current_step, prove, refute, {"case_id": case.case_id})


def judge_alignment(tokenizer: Any, model: Any, case: Case, code: str, max_new_tokens: int) -> dict[str, Any]:
    raw = generate(tokenizer, model, build_alignment_prompt(case, code), max_new_tokens=max_new_tokens)
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {"faithful": False, "raw": raw, "reason": "alignment judge did not return JSON"}
    try:
        data = json.loads(match.group(0))
    except Exception:
        return {"faithful": False, "raw": raw, "reason": "alignment judge returned invalid JSON"}
    data["raw"] = raw
    return data


def lean_error_text(result: Any) -> str:
    parts = []
    if result.prove:
        parts.append("PROVE STDOUT:\n" + result.prove.stdout[-2000:])
        parts.append("PROVE STDERR:\n" + result.prove.stderr[-2000:])
    if result.refute:
        parts.append("REFUTE STDOUT:\n" + result.refute.stdout[-2000:])
        parts.append("REFUTE STDERR:\n" + result.refute.stderr[-2000:])
    return "\n".join(parts)


def run_case(case: Case, tokenizer: Any, model: Any, args: argparse.Namespace) -> dict[str, Any]:
    verifier = SemanticLeanVerifier(args.lean_executable, timeout_s=args.timeout_s, workdir=args.lean_workdir)
    attempts: list[dict[str, Any]] = []
    start = time.time()

    if args.protocol == "contract":
        contract = generate(tokenizer, model, build_contract_prompt(case), args.max_new_tokens)
        prompt = build_from_contract_prompt(case, contract)
    else:
        contract = None
        prompt = build_direct_prompt(case)

    raw = generate(tokenizer, model, prompt, args.max_new_tokens, assistant_prefix=args.assistant_prefix)
    code = extract_common_code(raw)
    if code is None:
        return {
            "case_id": case.case_id,
            "expected_status": case.expected_status,
            "status": "extraction_failed",
            "matches_expected": False,
            "seconds": round(time.time() - start, 3),
            "contract": contract,
            "raw_generation": raw,
        }

    for round_idx in range(args.repair_rounds + 1):
        alignment = judge_alignment(tokenizer, model, case, code, args.judge_tokens) if args.alignment_judge else {"faithful": True}
        task = build_task(case, code)
        result = verifier.verify(task)
        attempt = {
            "round": round_idx,
            "code": code,
            "alignment": alignment,
            "status": result.status,
            "reason": result.reason,
            "prove_stdout": result.prove.stdout[-1200:] if result.prove else None,
            "refute_stdout": result.refute.stdout[-1200:] if result.refute else None,
        }
        attempts.append(attempt)
        if alignment.get("faithful") is not True:
            if round_idx < args.repair_rounds:
                repair_raw = generate(
                    tokenizer,
                    model,
                    build_semantic_repair_prompt(case, code, alignment),
                    args.max_new_tokens,
                    assistant_prefix=args.assistant_prefix,
                )
                repaired = extract_common_code(repair_raw)
                attempts[-1]["semantic_repair_raw"] = repair_raw
                if repaired is not None and repaired != code:
                    code = repaired
                    continue
            return {
                "case_id": case.case_id,
                "expected_status": case.expected_status,
                "status": "unsafe_formalization",
                "reason": "alignment judge rejected formalization",
                "matches_expected": False,
                "seconds": round(time.time() - start, 3),
                "contract": contract,
                "attempts": attempts,
            }
        if result.status in {"valid", "invalid"}:
            return {
                "case_id": case.case_id,
                "expected_status": case.expected_status,
                "status": result.status,
                "reason": result.reason,
                "matches_expected": result.status == case.expected_status,
                "seconds": round(time.time() - start, 3),
                "contract": contract,
                "attempts": attempts,
            }
        if round_idx >= args.repair_rounds:
            break
        repair_raw = generate(
            tokenizer,
            model,
            build_repair_prompt(case, code, lean_error_text(result)),
            args.max_new_tokens,
            assistant_prefix=args.assistant_prefix,
        )
        repaired = extract_common_code(repair_raw)
        attempts[-1]["repair_raw"] = repair_raw
        if repaired is None or repaired == code:
            break
        code = repaired

    final_status = attempts[-1]["status"] if attempts else "unknown"
    return {
        "case_id": case.case_id,
        "expected_status": case.expected_status,
        "status": final_status,
        "matches_expected": final_status == case.expected_status,
        "seconds": round(time.time() - start, 3),
        "contract": contract,
        "attempts": attempts,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = sorted({str(r.get("status")) for r in rows})
    return {
        "total": len(rows),
        "matches_expected": sum(1 for r in rows if r.get("matches_expected") is True),
        "statuses": {s: sum(1 for r in rows if r.get("status") == s) for s in statuses},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="No-template NL-to-Lean formalizer experiment.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--protocol", choices=["direct", "contract"], default="direct")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--judge-tokens", type=int, default=256)
    parser.add_argument("--repair-rounds", type=int, default=0)
    parser.add_argument("--alignment-judge", action="store_true")
    parser.add_argument("--assistant-prefix", default="")
    parser.add_argument("--lean-executable", default="lean")
    parser.add_argument("--lean-workdir", default=None)
    parser.add_argument("--timeout-s", type=float, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model, args.dtype, args.device)
    rows = [run_case(case, tokenizer, model, args) for case in cases()[: args.limit or None]]
    payload = {"summary": summarize(rows), "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
