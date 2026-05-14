from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fhis.semantic_lean_verify import SemanticLeanVerifier  # noqa: E402
from no_template_formalizer_experiment import (  # noqa: E402
    Case,
    build_alignment_prompt,
    build_direct_prompt,
    build_repair_prompt,
    build_task,
    cases as benchmark_cases,
    extract_common_code,
    lean_error_text,
)


@dataclass(frozen=True)
class SFTExample:
    problem: str
    prior_steps: list[str]
    current_step: str
    code: str
    family: str

    def as_case(self) -> Case:
        return Case(
            case_id=f"train_{self.family}",
            expected_status="unknown",
            problem=self.problem,
            prior_steps=self.prior_steps,
            current_step=self.current_step,
        )


def quote_list(xs: list[int]) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]"


def terminating_code(n: int) -> str:
    return f"""def pow10 : Nat -> Nat
  | 0 => 1
  | m + 1 => 10 * pow10 m

def naturalsUpTo (n : Nat) : List Nat :=
  List.map (fun i => i + 1) (List.range n)

def unitFractionTerminatesWithin (bound k : Nat) : Bool :=
  if k == 0 then false else List.any (List.range (bound + 1)) (fun m => pow10 m % k == 0)

def countTerminatesUpTo (n : Nat) : Nat :=
  List.length (List.filter (unitFractionTerminatesWithin (n + 1)) (naturalsUpTo n))

def current_step_claim : Prop :=
  2 * countTerminatesUpTo {n} = {n}"""


def sum_squares_code(k: int, m: int) -> str:
    return f"""def sumSquares (k : Nat) : Nat :=
  k * (k + 1) * (2 * k + 1) / 6

def current_step_claim : Prop :=
  sumSquares {k} % {m} = 0"""


def vector_code(pn: int, qn: int, dot_num: int, dot_den: int, claim_num: int, claim_den: int) -> str:
    return f"""def pNorm : Nat := {pn}
def qNorm : Nat := {qn}
def dotNumerator : Nat := {dot_num}
def dotDenominator : Nat := {dot_den}
def claimNumerator : Nat := {claim_num}
def claimDenominator : Nat := {claim_den}

def current_step_claim : Prop :=
  dotNumerator * claimDenominator = claimNumerator * dotDenominator * pNorm * qNorm"""


def relative_speed_code(a: int, b: int, claim: int) -> str:
    return f"""def firstSpeed : Nat := {a}
def secondSpeed : Nat := {b}
def relativeSpeed : Nat := firstSpeed + secondSpeed

def current_step_claim : Prop :=
  relativeSpeed = {claim}"""


def absdiff_average_code(values: list[int], num: int, den: int) -> str:
    pairs = [(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]
    terms = " + ".join(f"absDiff {a} {b}" for a, b in pairs)
    return f"""def absDiff (a b : Nat) : Nat :=
  if a <= b then b - a else a - b

def diffSum : Nat :=
  {terms}

def pairCount : Nat := {len(pairs)}
def claimNum : Nat := {num}
def claimDen : Nat := {den}

def current_step_claim : Prop :=
  claimDen * diffSum = claimNum * pairCount"""


def crt_code(moduli: list[int]) -> str:
    if moduli and moduli == list(range(1, len(moduli) + 1)):
        moduli_def = f"def moduli : List Nat := List.map (fun i => i + 1) (List.range {len(moduli)})"
    else:
        values = ", ".join(str(x) for x in moduli)
        moduli_def = f"def moduli : List Nat := [{values}]"
    return f"""{moduli_def}

def allCoprimeWith (x : Nat) : List Nat -> Bool
  | [] => true
  | y :: ys => (Nat.gcd x y == 1) && allCoprimeWith x ys

def pairwiseCoprime : List Nat -> Bool
  | [] => true
  | x :: xs => allCoprimeWith x xs && pairwiseCoprime xs

def current_step_claim : Prop :=
  pairwiseCoprime moduli = true"""


def build_training_examples(seed: int, n_per_family: int) -> list[SFTExample]:
    rng = random.Random(seed)
    rows: list[SFTExample] = []

    for _ in range(n_per_family):
        n = rng.choice([6, 8, 12, 14, 15, 16, 18, 20, 24, 25, 30])
        support = n // 2
        phrasing = rng.choice(
            [
                f"Since n/2 = {support}, n = {n} works.",
                f"For n = {n}, the required half-count would be {support}, so this value works.",
                f"Thus n = {n} is valid for the terminating-decimal condition.",
            ]
        )
        rows.append(
            SFTExample(
                problem="Find a positive integer n such that exactly half of the fractions 1/k for 1 <= k <= n have terminating decimal expansions.",
                prior_steps=[
                    "A unit fraction 1/k terminates in base 10 iff k divides a power of 10.",
                    "The step called works must check the count among k = 1, ..., n.",
                ],
                current_step=phrasing,
                code=terminating_code(n),
                family="terminating",
            )
        )

    for _ in range(n_per_family):
        k = rng.choice([12, 18, 24, 30, 36, 42, 54, 60, 72])
        m = rng.choice([50, 75, 100, 125, 150, 225, 250])
        rows.append(
            SFTExample(
                problem=f"Find a positive integer k such that 1^2 + 2^2 + ... + k^2 is divisible by {m}.",
                prior_steps=["Use the exact finite sum of squares."],
                current_step=rng.choice(
                    [
                        f"k = {k} is valid because the sum of squares is a multiple of {m}.",
                        f"The candidate k = {k} works: 1^2 + ... + {k}^2 is divisible by {m}.",
                        f"For k = {k}, the square-sum divisibility by {m} holds.",
                    ]
                ),
                code=sum_squares_code(k, m),
                family="sum_squares",
            )
        )

    for _ in range(n_per_family):
        pn = rng.choice([1, 2, 3, 4, 5])
        qn = rng.choice([1, 2, 3, 4, 6])
        dot_den = rng.choice([2, 3, 4, 5, 6, 8])
        dot_num = rng.randint(1, dot_den - 1)
        claim_den = rng.choice([3, 4, 5, 6, 7, 8, 9, 10])
        claim_num = rng.randint(1, claim_den - 1)
        rows.append(
            SFTExample(
                problem=f"Two nonzero vectors p and q satisfy |p| = {pn}, |q| = {qn}, and p dot q = {dot_num}/{dot_den}. Determine cos(theta).",
                prior_steps=["cos(theta) = (p dot q) / (|p| |q|)."],
                current_step=rng.choice(
                    [
                        f"Therefore cos(theta) = {claim_num}/{claim_den}.",
                        f"Substituting gives the value cos(theta) = {claim_num}/{claim_den}.",
                        f"Hence the cosine is {claim_num}/{claim_den}.",
                    ]
                ),
                code=vector_code(pn, qn, dot_num, dot_den, claim_num, claim_den),
                family="vector",
            )
        )

    for _ in range(n_per_family):
        a = rng.randint(3, 40)
        b = rng.randint(3, 40)
        distance = rng.choice([60, 72, 80, 90, 96, 100, 120, 144])
        claim = rng.choice([a + b, abs(a - b), a + b + rng.choice([-2, -1, 1, 2])])
        rows.append(
            SFTExample(
                problem=f"Two people start {distance} km apart and move toward each other with speeds {a} km/hr and {b} km/hr. Find the distance traveled by the first person when they meet.",
                prior_steps=["When moving toward each other, relative speed is the sum of their speeds."],
                current_step=rng.choice(
                    [
                        f"The relative speed is {a} + {b} = {claim}.",
                        f"Together they close the gap at {claim}.",
                        f"The closing speed is {claim}.",
                    ]
                ),
                code=relative_speed_code(a, b, claim),
                family="relative_speed",
            )
        )

    for _ in range(n_per_family):
        size = rng.choice([3, 4, 5])
        start = rng.randint(1, 5)
        values = list(range(start, start + size))
        pairs = [(values[i], values[j]) for i in range(size) for j in range(i + 1, size)]
        diff_sum = sum(abs(a - b) for a, b in pairs)
        den = len(pairs)
        num = rng.choice([diff_sum, diff_sum + rng.choice([-2, -1, 1, 2])])
        rows.append(
            SFTExample(
                problem=f"Let (a1,...,a{size}) range over permutations of {{{', '.join(map(str, values))}}}. Compute the average contribution of |a1-a2|.",
                prior_steps=[f"The unordered pair {{a1,a2}} is uniformly distributed over the {den} two-element subsets."],
                current_step=rng.choice(
                    [
                        f"The absolute differences over the unordered pairs have sum {num}, so the average is {num}/{den}.",
                        f"The expected value of |a1-a2| is {num}/{den}.",
                        f"Adding the pairwise differences gives {num}, hence the average contribution is {num}/{den}.",
                    ]
                ),
                code=absdiff_average_code(values, num, den),
                family="absdiff",
            )
        )

    for _ in range(n_per_family):
        roll = rng.random()
        if roll < 0.25:
            top = rng.choice([8, 10, 12, 15, 18, 20])
            moduli = list(range(1, top + 1))
        elif roll < 0.6:
            moduli = rng.sample([2, 3, 5, 7, 11, 13, 17, 19], rng.choice([3, 4, 5]))
        else:
            base = rng.choice([2, 3, 4, 5])
            moduli = [base, 2 * base, rng.choice([base + 1, base + 2, base + 3, base + 5])]
        moduli = sorted(set(moduli))
        rows.append(
            SFTExample(
                problem=f"A solution applies the Chinese Remainder Theorem to moduli {quote_list(moduli)}.",
                prior_steps=["CRT with uniqueness modulo the product requires the moduli to be pairwise coprime."],
                current_step=rng.choice(
                    [
                        f"The moduli {quote_list(moduli)} are pairwise coprime.",
                        f"This is possible by CRT because these moduli are pairwise coprime.",
                        f"The required CRT hypothesis holds for the listed moduli.",
                    ]
                ),
                code=crt_code(moduli),
                family="crt",
            )
        )

    rng.shuffle(rows)
    return rows


def theorem_to_claim_code(formal_statement: str) -> str | None:
    text = formal_statement.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("| ", "")
    text = re.sub(r":=\s*by\s*sorry\s*$", "", text).strip()
    match = re.match(r"theorem\s+\S+\s+(?P<body>.*)$", text)
    if not match:
        return None
    body = match.group("body").strip()
    params: list[str] = []
    while body.startswith("("):
        depth = 0
        end = None
        for i, ch in enumerate(body):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            return None
        params.append(body[1:end].strip())
        body = body[end + 1 :].strip()
    if not body.startswith(":"):
        return None
    proposition = body[1:].strip()
    if not proposition:
        return None

    forall_parts: list[str] = []
    hypotheses: list[str] = []
    type_heads = ("Nat", "Int", "Bool", "String", "List", "ℕ", "ℤ")
    for param in params:
        if ":" not in param:
            return None
        lhs, rhs = [part.strip() for part in param.split(":", 1)]
        if rhs.startswith(type_heads) or rhs in {"Prop"}:
            forall_parts.append(f"({lhs} : {rhs})")
        else:
            hypotheses.append(rhs)
    prefix = ""
    if forall_parts:
        prefix += "∀ " + " ".join(forall_parts) + ", "
    for hyp in hypotheses:
        prefix += f"({hyp}) -> "
    return "def current_step_claim : Prop :=\n  " + prefix + proposition


def build_workbook_examples(path: str | None, seed: int, limit: int) -> list[SFTExample]:
    if not path or limit <= 0:
        return []
    import pandas as pd

    rng = random.Random(seed)
    df = pd.read_parquet(path)
    rows: list[SFTExample] = []
    forbidden = re.compile(r"ℝ|Real|Set|Finset|Matrix|Polynomial|Complex|Rat|ℚ|Type|∑|∫|sqrt|√")
    wanted = re.compile(r"Nat|Int|ℕ|ℤ")
    for _, row in df.iterrows():
        formal = str(row.get("formal_statement") or "")
        nl = str(row.get("natural_language_statement") or "")
        if not nl.strip() or not formal.strip():
            continue
        if forbidden.search(formal) or not wanted.search(formal):
            continue
        code = theorem_to_claim_code(formal)
        if code is None:
            continue
        rows.append(
            SFTExample(
                problem=nl,
                prior_steps=[],
                current_step=nl,
                code=code,
                family="lean_workbook",
            )
        )
    rng.shuffle(rows)
    return rows[:limit]


class PromptDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, tokenizer: Any, examples: list[SFTExample], max_len: int) -> None:
        self.rows: list[dict[str, torch.Tensor]] = []
        for ex in examples:
            prompt = render_prompt(tokenizer, build_direct_prompt(ex.as_case()))
            target = "FINAL_COMMON_CODE:\n" + ex.code.strip() + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + target_ids)[-max_len:]
            prompt_kept = max(0, len(ids) - len(target_ids))
            labels = [-100] * prompt_kept + ids[prompt_kept:]
            self.rows.append(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.rows[idx]


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, row in enumerate(batch):
        n = len(row["input_ids"])
        input_ids[i, :n] = row["input_ids"]
        labels[i, :n] = row["labels"]
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def set_module(root: nn.Module, path: str, module: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def add_lora(model: nn.Module, r: int, alpha: int, dropout: float, target_names: tuple[str, ...]) -> list[str]:
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.endswith(target_names):
            set_module(model, name, LoRALinear(module, r, alpha, dropout))
            replaced.append(name)
    for name, param in model.named_parameters():
        param.requires_grad_("lora_A" in name or "lora_B" in name)
    return replaced


def render_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are precise, conservative, and fluent in Lean 4. Output code only."},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt


def generate_text(tokenizer: Any, model: nn.Module, prompt: str, max_new_tokens: int, assistant_prefix: str) -> str:
    text = render_prompt(tokenizer, prompt) + assistant_prefix
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)


def judge_alignment(tokenizer: Any, model: nn.Module, case: Case, code: str, max_new_tokens: int) -> dict[str, Any]:
    raw = generate_text(tokenizer, model, build_alignment_prompt(case, code), max_new_tokens, "")
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {"faithful": False, "raw": raw, "reason": "alignment judge did not return JSON"}
    try:
        data = json.loads(match.group(0))
    except Exception:
        return {"faithful": False, "raw": raw, "reason": "alignment judge returned invalid JSON"}
    data["raw"] = raw
    return data


def evaluate(
    tokenizer: Any,
    model: nn.Module,
    args: argparse.Namespace,
    rows: list[Case],
) -> list[dict[str, Any]]:
    verifier = SemanticLeanVerifier(args.lean_executable, timeout_s=args.timeout_s)
    out: list[dict[str, Any]] = []
    for case in rows:
        raw = generate_text(tokenizer, model, build_direct_prompt(case), args.eval_tokens, "FINAL_COMMON_CODE:\n")
        code = extract_common_code("FINAL_COMMON_CODE:\n" + raw)
        if code is None:
            out.append(
                {
                    "case_id": case.case_id,
                    "expected_status": case.expected_status,
                    "status": "extraction_failed",
                    "matches_expected": False,
                    "raw_generation": raw,
                }
            )
            continue
        result = verifier.verify(build_task(case, code))
        repair_raw = None
        for _ in range(args.repair_rounds):
            if result.status in {"valid", "invalid"}:
                break
            repair_raw = generate_text(
                tokenizer,
                model,
                build_repair_prompt(case, code, lean_error_text(result)),
                args.eval_tokens,
                "FINAL_COMMON_CODE:\n",
            )
            repaired = extract_common_code("FINAL_COMMON_CODE:\n" + repair_raw)
            if repaired is None or repaired == code:
                break
            code = repaired
            result = verifier.verify(build_task(case, code))
        alignment = judge_alignment(tokenizer, model, case, code, args.judge_tokens) if args.alignment_judge else {"faithful": True}
        status = result.status if alignment.get("faithful") is True else "unsafe_formalization"
        out.append(
            {
                "case_id": case.case_id,
                "expected_status": case.expected_status,
                "status": status,
                "lean_status": result.status,
                "matches_expected": status == case.expected_status,
                "alignment": alignment,
                "code": code,
                "raw_generation": raw,
                "repair_raw": repair_raw,
                "prove_stdout": result.prove.stdout[-1000:] if result.prove else None,
                "refute_stdout": result.refute.stdout[-1000:] if result.refute else None,
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = sorted({str(row.get("status")) for row in rows})
    return {
        "total": len(rows),
        "matches_expected": sum(1 for row in rows if row.get("matches_expected") is True),
        "statuses": {status: sum(1 for row in rows if row.get("status") == status) for status in statuses},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual Qwen3 LoRA SFT for no-template whole-step Lean formalization.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter-output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-per-family", type=int, default=40)
    parser.add_argument("--workbook-parquet", default=None)
    parser.add_argument("--workbook-examples", type=int, default=0)
    parser.add_argument("--synthetic-examples", action="store_true")
    parser.add_argument("--max-len", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--eval-tokens", type=int, default=1024)
    parser.add_argument("--judge-tokens", type=int, default=384)
    parser.add_argument("--repair-rounds", type=int, default=0)
    parser.add_argument("--alignment-judge", action="store_true")
    parser.add_argument("--lean-executable", default="lean")
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device=args.device, dtype=dtype)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    replaced = add_lora(
        model,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    )
    model.to(device=args.device, dtype=dtype)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    trainable = [param for param in model.parameters() if param.requires_grad]
    print(json.dumps({"lora_modules": len(replaced), "trainable_params": sum(p.numel() for p in trainable)}))

    examples = []
    if args.synthetic_examples:
        examples.extend(build_training_examples(args.seed, args.n_per_family))
    examples.extend(build_workbook_examples(args.workbook_parquet, args.seed, args.workbook_examples))
    if not examples:
        examples = build_training_examples(args.seed, args.n_per_family)
    dataset = PromptDataset(tokenizer, examples, args.max_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    optim = torch.optim.AdamW(trainable, lr=args.lr)
    model.train()
    step = 0
    total_loss = 0.0
    started = time.time()
    optim.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            total_loss += float(loss.detach().cpu()) * args.grad_accum
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(json.dumps({"step": step, "loss": total_loss / (step * args.grad_accum), "elapsed_s": round(time.time() - started, 1)}))
                if step >= args.max_steps:
                    break
        if step >= args.max_steps:
            break

    model.eval()
    eval_rows = evaluate(tokenizer, model, args, benchmark_cases())
    payload = {
        "train": {
            "num_examples": len(examples),
            "steps": step,
            "lora_modules": len(replaced),
            "trainable_params": sum(p.numel() for p in trainable),
        },
        "summary": summarize(eval_rows),
        "rows": eval_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.adapter_output:
        adapter_dir = Path(args.adapter_output)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        state = {name: param.detach().cpu() for name, param in model.named_parameters() if "lora_A" in name or "lora_B" in name}
        torch.save(
            {
                "state_dict": state,
                "config": {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "target_names": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                },
            },
            adapter_dir / "adapter.pt",
        )
        (adapter_dir / "training_summary.json").write_text(json.dumps(payload["train"], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
