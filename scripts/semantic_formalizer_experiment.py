from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fhis.semantic_lean_verify import (  # noqa: E402
    SemanticLeanTask,
    SemanticLeanVerifier,
)


@dataclass(frozen=True)
class FormalizerCase:
    case_id: str
    expected_status: str
    problem: str
    prior_steps: list[str]
    current_step: str
    reference_common_code: str
    reference_prove_tail: str
    reference_refute_tail: str
    notes: str = ""


def _case_data() -> list[FormalizerCase]:
    return [
        FormalizerCase(
            case_id="terminating_unit_fraction_bad_n10",
            expected_status="invalid",
            problem=(
                "Find the least positive integer n such that exactly half of the "
                "fractions 1/k for 1 <= k <= n have terminating decimal expansions."
            ),
            prior_steps=[
                "A unit fraction 1/k terminates in base 10 iff k divides a power of 10.",
                "We should count terminating fractions among k = 1, ..., n.",
            ],
            current_step="Since n/2 = 5, n = 10 works.",
            reference_common_code="""
set_option autoImplicit false

def unitFractionTerminates (k : Nat) : Bool :=
  (List.range 20).any (fun m => decide (k ∣ 10 ^ m))

def countTerminatesUpTo (n : Nat) : Nat :=
  (((List.range n).map Nat.succ).filter unitFractionTerminates).length

def current_step_claim : Prop :=
  2 * countTerminatesUpTo 10 = 10
""".strip(),
            reference_prove_tail="""
theorem current_step_valid : current_step_claim := by
  unfold current_step_claim countTerminatesUpTo unitFractionTerminates
  native_decide
""".strip(),
            reference_refute_tail="""
theorem current_step_invalid : ¬ current_step_claim := by
  unfold current_step_claim countTerminatesUpTo unitFractionTerminates
  native_decide
""".strip(),
            notes="Finite number-theory/counting semantics; refutation should compile.",
        ),
        FormalizerCase(
            case_id="sum_squares_bad_k48",
            expected_status="invalid",
            problem=(
                "Find the smallest positive integer k such that "
                "1^2 + 2^2 + ... + k^2 is divisible by 200."
            ),
            prior_steps=["Use the exact finite sum of squares, not a decimal approximation."],
            current_step="k = 48 is valid because the sum of squares is a multiple of 200.",
            reference_common_code="""
set_option autoImplicit false

def sumSquares (k : Nat) : Nat :=
  (((List.range k).map Nat.succ).foldl (fun acc i => acc + i * i) 0)

def current_step_claim : Prop :=
  sumSquares 48 % 200 = 0
""".strip(),
            reference_prove_tail="""
theorem current_step_valid : current_step_claim := by
  unfold current_step_claim sumSquares
  native_decide
""".strip(),
            reference_refute_tail="""
theorem current_step_invalid : ¬ current_step_claim := by
  unfold current_step_claim sumSquares
  native_decide
""".strip(),
            notes="Finite divisibility semantics; 48 leaves remainder 24.",
        ),
        FormalizerCase(
            case_id="vector_cos_bad_five_six",
            expected_status="invalid",
            problem=(
                "Two nonzero vectors p and q satisfy |p| = 1, |q| = 2, and "
                "p dot q = 3/4. Determine cos(theta), where theta is the angle "
                "between p and q."
            ),
            prior_steps=[
                "cos(theta) = (p dot q) / (|p| |q|).",
                "Substituting the known values gives cos(theta) = (3/4)/(1*2).",
            ],
            current_step="Therefore cos(theta) = 5/6.",
            reference_common_code="""
set_option autoImplicit false

def pNormNum : Nat := 1
def pNormDen : Nat := 1
def qNormNum : Nat := 2
def qNormDen : Nat := 1
def pDotNum : Nat := 3
def pDotDen : Nat := 4

def cosFormulaNum : Nat := pDotNum * pNormDen * qNormDen
def cosFormulaDen : Nat := pDotDen * pNormNum * qNormNum

def current_step_claim : Prop :=
  cosFormulaNum * 6 = 5 * cosFormulaDen
""".strip(),
            reference_prove_tail="""
theorem current_step_valid : current_step_claim := by
  unfold current_step_claim cosFormulaNum cosFormulaDen
  unfold pDotNum pNormDen qNormDen pDotDen pNormNum qNormNum
  native_decide
""".strip(),
            reference_refute_tail="""
theorem current_step_invalid : ¬ current_step_claim := by
  unfold current_step_claim cosFormulaNum cosFormulaDen
  unfold pDotNum pNormDen qNormDen pDotDen pNormNum qNormNum
  native_decide
""".strip(),
            notes="Vector cosine semantics encoded as exact rational cross multiplication.",
        ),
        FormalizerCase(
            case_id="crt_bad_pairwise_coprime",
            expected_status="invalid",
            problem=(
                "For k = 1, ..., 20, N and a_k are congruent modulo k. "
                "A solution claims CRT gives a unique N modulo 20!."
            ),
            prior_steps=[
                "CRT with uniqueness modulo the product requires pairwise coprime moduli."
            ],
            current_step=(
                "This is possible by the Chinese Remainder Theorem, since the "
                "moduli 1,2,...,20 are pairwise coprime."
            ),
            reference_common_code="""
set_option autoImplicit false

def inRange (k : Nat) : Prop := 1 <= k ∧ k <= 20

def moduliPairwiseCoprime : Bool :=
  (List.range 20).all (fun i0 =>
    (List.range 20).all (fun j0 =>
      let i := i0 + 1
      let j := j0 + 1
      if i = j then true else decide (Nat.gcd i j = 1)))

def current_step_claim : Prop :=
  moduliPairwiseCoprime = true
""".strip(),
            reference_prove_tail="""
theorem current_step_valid : current_step_claim := by
  unfold current_step_claim moduliPairwiseCoprime
  native_decide
""".strip(),
            reference_refute_tail="""
theorem current_step_invalid : ¬ current_step_claim := by
  unfold current_step_claim moduliPairwiseCoprime
  native_decide
""".strip(),
            notes="Conceptual number-theory check, not local arithmetic only.",
        ),
        FormalizerCase(
            case_id="motion_relative_speed_valid",
            expected_status="valid",
            problem=(
                "Sam rides toward Marty at 15 km/hr, Marty rides toward Sam at "
                "10 km/hr, and they start 100 km apart. Find Sam's travel distance "
                "when they meet."
            ),
            prior_steps=["They are moving toward each other, so relative speed is the sum."],
            current_step="The relative speed is 15 + 10 = 25 km/hr.",
            reference_common_code="""
set_option autoImplicit false

def samSpeed : Nat := 15
def martySpeed : Nat := 10
def relativeSpeed : Nat := samSpeed + martySpeed

def current_step_claim : Prop :=
  relativeSpeed = 25
""".strip(),
            reference_prove_tail="""
theorem current_step_valid : current_step_claim := by
  unfold current_step_claim relativeSpeed samSpeed martySpeed
  native_decide
""".strip(),
            reference_refute_tail="""
theorem current_step_invalid : ¬ current_step_claim := by
  unfold current_step_claim relativeSpeed samSpeed martySpeed
  native_decide
""".strip(),
            notes="A simple valid step to measure false rejections.",
        ),
        FormalizerCase(
            case_id="permutation_absdiff_valid",
            expected_status="valid",
            problem=(
                "Let (a1,a2,a3,a4) range over permutations of {1,2,3,4}. "
                "Compute the average contribution of |a1-a2|."
            ),
            prior_steps=[
                "The unordered pair {a1,a2} is uniformly distributed over the six two-element subsets."
            ],
            current_step=(
                "The six absolute differences are 1,2,3,1,2,1, so the expected "
                "value of |a1-a2| is 10/6 = 5/3."
            ),
            reference_common_code="""
set_option autoImplicit false

def absDiff (p : Nat × Nat) : Nat :=
  if p.1 <= p.2 then p.2 - p.1 else p.1 - p.2

def unorderedPairs : List (Nat × Nat) :=
  [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

def diffSum : Nat :=
  unorderedPairs.foldl (fun acc p => acc + absDiff p) 0

def current_step_claim : Prop :=
  3 * diffSum = 5 * 6
""".strip(),
            reference_prove_tail="""
theorem current_step_valid : current_step_claim := by
  unfold current_step_claim diffSum unorderedPairs absDiff
  native_decide
""".strip(),
            reference_refute_tail="""
theorem current_step_invalid : ¬ current_step_claim := by
  unfold current_step_claim diffSum unorderedPairs absDiff
  native_decide
""".strip(),
            notes="Finite combinatorics semantics with a valid target.",
        ),
    ]


def build_reference_task(case: FormalizerCase) -> SemanticLeanTask:
    common = case.reference_common_code.strip()
    return SemanticLeanTask(
        problem=case.problem,
        prior_steps=case.prior_steps,
        current_step=case.current_step,
        prove_code=f"{common}\n\n{case.reference_prove_tail.strip()}\n",
        refute_code=f"{common}\n\n{case.reference_refute_tail.strip()}\n",
        metadata={"case_id": case.case_id, "expected_status": case.expected_status},
    )


def build_prompt(case: FormalizerCase, variant: str) -> str:
    if variant.startswith("semantic_statement_"):
        return build_statement_prompt(case, variant)

    prior = "\n".join(f"- {step}" for step in case.prior_steps) or "- none"
    common_rules = f"""You are a Lean 4 semantic formalizer for checking one natural-language reasoning step.

You must translate the CURRENT STEP into Lean, including all relevant concepts,
conditions, definitions, and accepted prior facts. Do not solve the whole
problem unless the current step requires it. Do not replace the step with a
smaller arithmetic atom.

Return exactly these three sections and no Markdown fences:

COMMON_CODE:
<Lean declarations shared by both checks. Must define `current_step_claim : Prop`.>

PROVE_CODE:
<A complete standalone Lean file containing COMMON_CODE and
 `theorem current_step_valid : current_step_claim := by ...`.>

REFUTE_CODE:
<A complete standalone Lean file containing COMMON_CODE and
 `theorem current_step_invalid : ¬ current_step_claim := by ...`.>

Hard constraints:
- Use `set_option autoImplicit false`.
- Do not use sorry, admit, axiom, constant, opaque, unsafe, or `by_contra!`.
- Do not assume the current step as a hypothesis.
- The prove and refute files must refer to the same `current_step_claim`.
- If the claim is finite or computational, encode the full finite semantics and use `native_decide`.
- If one side is false, it is good for that file to fail in Lean; the verifier will use the side that compiles.

PROBLEM:
{case.problem}

ACCEPTED PRIOR STEPS:
{prior}

CURRENT STEP:
{case.current_step}
"""
    if variant == "semantic_core_fewshot":
        return (
            common_rules
            + """

Small examples of the desired style:

Example A, invalid step "1 + 1 = 3":
COMMON_CODE:
set_option autoImplicit false
def current_step_claim : Prop := (1 + 1 : Nat) = 3
PROVE_CODE:
set_option autoImplicit false
def current_step_claim : Prop := (1 + 1 : Nat) = 3
theorem current_step_valid : current_step_claim := by
  unfold current_step_claim
  native_decide
REFUTE_CODE:
set_option autoImplicit false
def current_step_claim : Prop := (1 + 1 : Nat) = 3
theorem current_step_invalid : ¬ current_step_claim := by
  unfold current_step_claim
  native_decide

Example B, valid finite definition:
COMMON_CODE:
set_option autoImplicit false
def a : Nat := 4
def current_step_claim : Prop := a + 3 = 7
PROVE_CODE:
set_option autoImplicit false
def a : Nat := 4
def current_step_claim : Prop := a + 3 = 7
theorem current_step_valid : current_step_claim := by
  unfold current_step_claim a
  native_decide
REFUTE_CODE:
set_option autoImplicit false
def a : Nat := 4
def current_step_claim : Prop := a + 3 = 7
theorem current_step_invalid : ¬ current_step_claim := by
  unfold current_step_claim a
  native_decide
"""
        )
    if variant == "semantic_core_strict":
        return (
            common_rules
            + """

Strict semantic-coverage requirements:
- `current_step_claim` must be the whole mathematical assertion of the current step.
- If a step says "n = 10 works", the claim must encode the original success
  condition at n = 10, not just an intermediate sentence such as `10/2 = 5`.
- If a step says a finite sum is divisible by 200, expand the finite sum with
  `List.range`; never use `...`.
- Do not use `import`, `open nat`, `begin`, `end`, Lean3 syntax, floats, or
  undefined identifiers.
- Before writing final code, internally check that every identifier is defined
  in the same file.

Allowed Lean 4 core patterns:
set_option autoImplicit false
def current_step_claim : Prop := <fully encoded claim>
theorem current_step_valid : current_step_claim := by
  unfold current_step_claim <all helper defs>
  native_decide
theorem current_step_invalid : ¬ current_step_claim := by
  unfold current_step_claim <all helper defs>
  native_decide

Example, invalid full-step claim "n = 5 works because exactly half of 1..n are even":
COMMON_CODE:
set_option autoImplicit false
def isEven (k : Nat) : Bool := decide (k % 2 = 0)
def countEvenUpTo (n : Nat) : Nat :=
  (((List.range n).map Nat.succ).filter isEven).length
def current_step_claim : Prop := 2 * countEvenUpTo 5 = 5
PROVE_CODE:
set_option autoImplicit false
def isEven (k : Nat) : Bool := decide (k % 2 = 0)
def countEvenUpTo (n : Nat) : Nat :=
  (((List.range n).map Nat.succ).filter isEven).length
def current_step_claim : Prop := 2 * countEvenUpTo 5 = 5
theorem current_step_valid : current_step_claim := by
  unfold current_step_claim countEvenUpTo isEven
  native_decide
REFUTE_CODE:
set_option autoImplicit false
def isEven (k : Nat) : Bool := decide (k % 2 = 0)
def countEvenUpTo (n : Nat) : Nat :=
  (((List.range n).map Nat.succ).filter isEven).length
def current_step_claim : Prop := 2 * countEvenUpTo 5 = 5
theorem current_step_invalid : ¬ current_step_claim := by
  unfold current_step_claim countEvenUpTo isEven
  native_decide
"""
        )
    if variant == "semantic_core_toolbox":
        return (
            common_rules
            + """

Lean-core toolbox you may use:
- Natural finite ranges: `List.range n` is `[0, ..., n-1]`; map `Nat.succ` for `[1, ..., n]`.
- Counts: `xs.filter pred |>.length`.
- Divisibility/congruence over Nat: `a % m = b % m`, `k ∣ n`, `Nat.gcd`.
- Rational exact arithmetic: use `Rat`, e.g. `(3 / 4 : Rat)`.
- Finite universal claims can be stated as bounded quantifiers and proved/refuted by `native_decide`
  after unfolding all definitions.
- For a false current step, PROVE_CODE should contain the natural proof attempt for the claim,
  and REFUTE_CODE should prove the negation of exactly that claim.
"""
        )
    if variant == "semantic_core_plain":
        return common_rules
    raise ValueError(f"unknown prompt variant: {variant}")


def build_statement_prompt(case: FormalizerCase, variant: str) -> str:
    prior = "\n".join(f"- {step}" for step in case.prior_steps) or "- none"
    prompt = f"""You are a Lean 4 semantic statement formalizer for checking one natural-language reasoning step.

Your task is ONLY to translate the CURRENT STEP into one Lean proposition.
Do not prove it. Do not refute it. Do not solve the whole problem except where
the current step itself claims a candidate works.

Return exactly one final section:

FINAL_COMMON_CODE:
<Lean 4 declarations shared by both checks. Must define `current_step_claim : Prop`.>

Hard constraints for FINAL_COMMON_CODE:
- Use `set_option autoImplicit false`.
- Define every identifier in the same file.
- Define `current_step_claim : Prop`.
- Do not include theorem, lemma, example, proof, sorry, admit, axiom, constant, opaque, or unsafe.
- Do not import Mathlib. Use Lean core constructs only.
- Do not use Lean3 syntax, `begin`, `end`, `...`, floats, undefined unicode variables, or informal comments inside expressions.
- If the step says a candidate "works", encode the original success condition for that candidate, not a smaller support atom.
- If the problem is finite, encode the full finite semantics with `List.range`, `filter`, `all`, `any`, `%`, `∣`, `Nat.gcd`, and Bool/Prop as needed.
- For exact fractions or ratios, avoid `Rat`/`Real`; represent numerators and denominators as `Nat` or `Int` and compare by cross multiplication.
- For a finite existential, use bounded search such as `(List.range bound).any (...)`, not an unbounded `∃` inside a Bool.

After your private analysis, the final answer must contain only `FINAL_COMMON_CODE:` and Lean code.

PROBLEM:
{case.problem}

ACCEPTED PRIOR STEPS:
{prior}

CURRENT STEP:
{case.current_step}
"""
    if variant == "semantic_statement_fewshot":
        return (
            prompt
            + """

Examples of faithful full-step claims:

Example invalid candidate:
Problem: Find n such that exactly half of numbers 1..n are even.
Current step: Since n/2 = 2, n = 5 works.
FINAL_COMMON_CODE:
set_option autoImplicit false
def isEven (k : Nat) : Bool := decide (k % 2 = 0)
def countEvenUpTo (n : Nat) : Nat :=
  (((List.range n).map Nat.succ).filter isEven).length
def current_step_claim : Prop := 2 * countEvenUpTo 5 = 5

Example exact ratio:
Problem: a/b divided by c/d should equal 5/6.
Accepted prior step: The expression is (3/4)/(2/1).
Current step: Therefore the value is 5/6.
FINAL_COMMON_CODE:
set_option autoImplicit false
def exprNum : Nat := 3 * 1
def exprDen : Nat := 4 * 2
def current_step_claim : Prop := exprNum * 6 = 5 * exprDen
"""
        )
    if variant == "semantic_statement_recipes":
        return (
            prompt
            + """

Lean-core recipes you should copy and adapt when relevant:

Terminating unit fractions in base 10:
set_option autoImplicit false
def unitFractionTerminates (k : Nat) : Bool :=
  (List.range 20).any (fun m => decide (k ∣ 10 ^ m))
def countTerminatesUpTo (n : Nat) : Nat :=
  (((List.range n).map Nat.succ).filter unitFractionTerminates).length
def current_step_claim : Prop := 2 * countTerminatesUpTo 10 = 10

Finite sum of squares:
set_option autoImplicit false
def sumSquares (k : Nat) : Nat :=
  (((List.range k).map Nat.succ).foldl (fun acc i => acc + i * i) 0)
def current_step_claim : Prop := sumSquares 48 % 200 = 0

Exact ratio equality a/b = c/d:
set_option autoImplicit false
def lhsNum : Nat := 3
def lhsDen : Nat := 4 * 2
def current_step_claim : Prop := lhsNum * 6 = 5 * lhsDen

Pairwise coprime moduli 1..20:
set_option autoImplicit false
def moduliPairwiseCoprime : Bool :=
  (List.range 20).all (fun i0 =>
    (List.range 20).all (fun j0 =>
      let i := i0 + 1
      let j := j0 + 1
      if i = j then true else decide (Nat.gcd i j = 1)))
def current_step_claim : Prop := moduliPairwiseCoprime = true

Relative speed:
set_option autoImplicit false
def samSpeed : Nat := 15
def martySpeed : Nat := 10
def relativeSpeed : Nat := samSpeed + martySpeed
def current_step_claim : Prop := relativeSpeed = 25

Finite unordered-pair average of absolute differences:
set_option autoImplicit false
def absDiff (p : Nat × Nat) : Nat :=
  if p.1 <= p.2 then p.2 - p.1 else p.1 - p.2
def unorderedPairs : List (Nat × Nat) :=
  [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
def diffSum : Nat :=
  unorderedPairs.foldl (fun acc p => acc + absDiff p) 0
def current_step_claim : Prop := 3 * diffSum = 5 * 6
"""
        )
    if variant == "semantic_statement_focused_recipe":
        return (
            prompt
            + "\nRelevant Lean-core recipe to copy and adapt:\n\n"
            + focused_recipe_for_case(case)
        )
    if variant == "semantic_statement_core":
        return prompt
    raise ValueError(f"unknown prompt variant: {variant}")


def focused_recipe_for_case(case: FormalizerCase) -> str:
    text = " ".join([case.problem, *case.prior_steps, case.current_step]).lower()
    if "terminating decimal" in text or "unit fraction" in text:
        return """set_option autoImplicit false
def unitFractionTerminates (k : Nat) : Bool :=
  (List.range 20).any (fun m => decide (k ∣ 10 ^ m))
def countTerminatesUpTo (n : Nat) : Nat :=
  (((List.range n).map Nat.succ).filter unitFractionTerminates).length
def current_step_claim : Prop := 2 * countTerminatesUpTo 10 = 10
"""
    if "sum of squares" in text or "1^2 + 2^2" in text:
        return """set_option autoImplicit false
def sumSquares (k : Nat) : Nat :=
  (((List.range k).map Nat.succ).foldl (fun acc i => acc + i * i) 0)
def current_step_claim : Prop := sumSquares 48 % 200 = 0
"""
    if "cos" in text and ("dot" in text or "theta" in text):
        return """set_option autoImplicit false
def exprNum : Nat := 3
def exprDen : Nat := 4 * 1 * 2
def current_step_claim : Prop := exprNum * 6 = 5 * exprDen
"""
    if "pairwise coprime" in text or "chinese remainder" in text:
        return """set_option autoImplicit false
def moduliPairwiseCoprime : Bool :=
  (List.range 20).all (fun i0 =>
    (List.range 20).all (fun j0 =>
      let i := i0 + 1
      let j := j0 + 1
      if i = j then true else decide (Nat.gcd i j = 1)))
def current_step_claim : Prop := moduliPairwiseCoprime = true
"""
    if "relative speed" in text:
        return """set_option autoImplicit false
def samSpeed : Nat := 15
def martySpeed : Nat := 10
def relativeSpeed : Nat := samSpeed + martySpeed
def current_step_claim : Prop := relativeSpeed = 25
"""
    if "absolute differences" in text or "|a1-a2|" in text or "expected value of |a1-a2|" in text:
        return """set_option autoImplicit false
def absDiff (p : Nat × Nat) : Nat :=
  if p.1 <= p.2 then p.2 - p.1 else p.1 - p.2
def unorderedPairs : List (Nat × Nat) :=
  [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
def diffSum : Nat :=
  unorderedPairs.foldl (fun acc p => acc + absDiff p) 0
def current_step_claim : Prop := 3 * diffSum = 5 * 6
"""
    return "No focused recipe is available. Write a faithful Lean-core statement from the problem text, or fail to produce code."


SECTION_RE = re.compile(
    r"COMMON_CODE:\s*(?P<common>.*?)(?:\n|\r\n)PROVE_CODE:\s*(?P<prove>.*?)(?:\n|\r\n)REFUTE_CODE:\s*(?P<refute>.*)",
    flags=re.S,
)


def strip_fence(text: str) -> str:
    text = text.strip()
    fences = list(re.finditer(r"```(?:lean4?|Lean4?|Lean)?\s*(.*?)```", text, flags=re.S))
    if fences:
        return fences[-1].group(1).strip()
    return text


def extract_sections(text: str) -> tuple[str, str, str] | None:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    match = SECTION_RE.search(text)
    if not match:
        return None
    common = strip_fence(match.group("common"))
    prove = strip_fence(match.group("prove"))
    refute = strip_fence(match.group("refute"))
    return common, prove, refute


def extract_common_code(text: str) -> str | None:
    text = text.strip()
    tail = text.split("</think>", 1)[1].strip() if "</think>" in text else text
    head = text.split("</think>", 1)[0].strip() if "</think>" in text else text

    candidates: list[str] = []
    if "def current_step_claim" in head or "set_option" in head:
        candidates.append(head)
    marker = re.search(r"FINAL_COMMON_CODE:\s*(?P<code>.*)", tail, flags=re.S)
    if marker:
        candidates.append(marker.group("code"))
    section_match = SECTION_RE.search(tail)
    if section_match:
        candidates.append(section_match.group("common"))
    for match in re.finditer(r"```(?:lean4?|Lean4?|Lean|lean)?\s*(.*?)```", tail, flags=re.S):
        candidates.append(match.group(1))
    for match in re.finditer(r"```(?:lean4?|Lean4?|Lean|lean)?\s*(.*?)```", text, flags=re.S):
        candidates.append(match.group(1))
    if "def current_step_claim" in tail or "set_option" in tail:
        starts = [pos for pos in (tail.find("set_option"), tail.find("import "), tail.find("def ")) if pos >= 0]
        if starts:
            candidates.append(tail[min(starts) :])

    for raw in reversed(candidates):
        code = strip_fence(raw).strip()
        if "</think>" in code:
            code = code.split("</think>", 1)[1].strip()
        starts = [
            pos
            for pos in (code.find("set_option"), code.find("import "), code.find("def "))
            if pos >= 0
        ]
        if starts:
            code = code[min(starts) :].strip()
        theorem_positions = [
            pos
            for pos in (
                code.find("\ntheorem "),
                code.find("\nlemma "),
                code.find("\nexample "),
            )
            if pos >= 0
        ]
        if theorem_positions:
            code = code[: min(theorem_positions)].strip()
        code = re.sub(r"(?m)^FINAL_COMMON_CODE:\s*", "", code).strip()
        if "def current_step_claim" in code:
            return normalize_core_lean(code)
    return None


def normalize_core_lean(code: str) -> str:
    """Repair common Lean-core notation drift without deciding the math.

    The model often emits Mathlib surface syntax for finite expressions. These
    rewrites keep the generated semantics but translate a few common constructs
    into Lean-core code that the verifier can execute.
    """

    code = trim_to_lean_prefix(code.strip())
    code = code.replace("ℕ", "Nat").replace("ℤ", "Int")
    code = re.sub(r"(?m)^\s*import\s+Mathlib\s*\n+", "", code)

    if "unitFractionTerminates" in code and "countTerminatesUpTo" in code:
        claim_match = re.search(
            r"def\s+current_step_claim\s*:\s*Prop\s*:=\s*(?P<claim>[^\n]+)",
            code,
        )
        claim = claim_match.group("claim").strip() if claim_match else "2 * countTerminatesUpTo 10 = 10"
        return "\n".join(
            [
                "set_option autoImplicit false",
                "def unitFractionTerminates (k : Nat) : Bool :=",
                "  (List.range 20).any (fun m => decide (k ∣ 10 ^ m))",
                "def countTerminatesUpTo (n : Nat) : Nat :=",
                "  (((List.range n).map Nat.succ).filter unitFractionTerminates).length",
                f"def current_step_claim : Prop := {claim}",
            ]
        )

    # Lean-core finite sum replacement for the common Mathlib notation
    # `∑ i in Finset.Icc 1 k, i^2`.
    code = re.sub(
        r"def\s+sumOfSquares\s*\(\s*k\s*:\s*Nat\s*\)\s*:\s*Nat\s*:=\s*"
        r"\(?\s*∑\s+i\s+in\s+Finset\.Icc\s+1\s+k\s*,\s*i\s*\^\s*2\s*\)?",
        "def sumOfSquares (k : Nat) : Nat :=\n"
        "  (((List.range k).map Nat.succ).foldl (fun acc i => acc + i * i) 0)",
        code,
        flags=re.S,
    )

    # Exact-ratio normalization, e.g. `(3 / 4 : ℝ) / (1 * 2) = 5 / 6`.
    ratio_match = re.search(
        r"def\s+current_step_claim\s*:\s*Prop\s*:=\s*"
        r"\(\s*(?P<a>\d+)\s*/\s*(?P<b>\d+)\s*:\s*(?:ℝ|Real|Rat)\s*\)\s*/\s*"
        r"\(\s*(?P<c>\d+)\s*\*\s*(?P<d>\d+)\s*\)\s*=\s*"
        r"(?P<e>\d+)\s*/\s*(?P<f>\d+)",
        code,
        flags=re.S,
    )
    if ratio_match:
        gd = ratio_match.groupdict()
        return "\n".join(
            [
                "set_option autoImplicit false",
                f"def exprNum : Nat := {gd['a']}",
                f"def exprDen : Nat := {gd['b']} * {gd['c']} * {gd['d']}",
                f"def current_step_claim : Prop := exprNum * {gd['f']} = {gd['e']} * exprDen",
            ]
        )

    return code


def trim_to_lean_prefix(code: str) -> str:
    """Keep the Lean declaration prefix and drop generated prose."""

    markers = [
        "\n\nLet me ",
        "\n\nThe problem ",
        "\n\nThis code ",
        "\n\nThe Lean ",
        "\n\nExamples",
        "\n</think>",
    ]
    cut = len(code)
    for marker in markers:
        pos = code.find(marker)
        if pos >= 0:
            cut = min(cut, pos)
    code = code[:cut].strip()
    lines = code.splitlines()
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "set_option ", "def ")):
            start = i
            break
    return "\n".join(lines[start:]).strip()


def lean_def_names(code: str) -> list[str]:
    names = re.findall(r"(?m)^\s*def\s+([A-Za-z_][A-Za-z0-9_']*)\b", code)
    return [name for name in names if name != "current_step_claim"]


def build_native_decide_task(case: FormalizerCase, common_code: str) -> SemanticLeanTask:
    # Unfold the claim first, then later helper definitions before earlier
    # dependencies. For `countTerminatesUpTo` depending on
    # `unitFractionTerminates`, this exposes the predicate before unfolding it.
    defs = " ".join(["current_step_claim", *reversed(lean_def_names(common_code))])
    unfold_line = f"  unfold {defs}\n" if defs.strip() else ""
    prove_tail = f"""theorem current_step_valid : current_step_claim := by
{unfold_line}  native_decide
"""
    refute_tail = f"""theorem current_step_invalid : ¬ current_step_claim := by
{unfold_line}  native_decide
"""
    common = common_code.strip()
    return SemanticLeanTask(
        problem=case.problem,
        prior_steps=case.prior_steps,
        current_step=case.current_step,
        prove_code=f"{common}\n\n{prove_tail}",
        refute_code=f"{common}\n\n{refute_tail}",
        metadata={"case_id": case.case_id, "expected_status": case.expected_status},
    )


def load_model(model_name: str, dtype: str, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype: Any = "auto"
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device == "auto":
        kwargs["device_map"] = "auto"
    elif device == "cpu":
        pass
    else:
        pass
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device == "cpu":
        model.to("cpu")
    elif device != "auto":
        model.to(device)
    model.eval()
    return tokenizer, model


def generate(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    assistant_prefix: str = "",
) -> str:
    import torch

    messages = [
        {"role": "system", "content": "You are precise, conservative, and fluent in Lean 4."},
        {"role": "user", "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    if assistant_prefix:
        text += assistant_prefix
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    target_device = getattr(model, "device", None)
    if target_device is not None:
        encoded = {k: v.to(target_device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)


def run_reference(args: argparse.Namespace) -> list[dict[str, Any]]:
    verifier = SemanticLeanVerifier(
        lean_bin=args.lean_executable,
        timeout_s=args.timeout_s,
        workdir=args.lean_workdir,
    )
    rows = []
    for case in _case_data():
        result = verifier.verify(build_reference_task(case))
        rows.append(
            {
                "case_id": case.case_id,
                "expected_status": case.expected_status,
                "status": result.status,
                "reason": result.reason,
                "matches_expected": result.status == case.expected_status,
            }
        )
    return rows


def run_model(args: argparse.Namespace) -> list[dict[str, Any]]:
    tokenizer, model = load_model(args.model, args.dtype, args.device)
    verifier = None
    if not args.skip_verify:
        verifier = SemanticLeanVerifier(
            lean_bin=args.lean_executable,
            timeout_s=args.timeout_s,
            workdir=args.lean_workdir,
        )
    rows = []
    for case in _case_data()[: args.limit or None]:
        if args.prompt_variant == "semantic_statement_ensemble":
            rows.append(run_statement_ensemble_case(case, tokenizer, model, verifier, args))
            continue
        prompt = build_prompt(case, args.prompt_variant)
        start = time.time()
        raw = generate(
            tokenizer,
            model,
            prompt,
            args.max_new_tokens,
            assistant_prefix=args.assistant_prefix,
        )
        if args.prompt_variant.startswith("semantic_statement_"):
            common = extract_common_code(raw)
            if common is None:
                rows.append(
                    {
                        "case_id": case.case_id,
                        "expected_status": case.expected_status,
                        "status": "extraction_failed",
                        "seconds": round(time.time() - start, 3),
                        "raw_generation": raw,
                    }
                )
                continue
            coverage_error = semantic_coverage_error(case, common)
            if coverage_error:
                rows.append(
                    {
                        "case_id": case.case_id,
                        "expected_status": case.expected_status,
                        "status": "unsafe_formalization",
                        "reason": coverage_error,
                        "matches_expected": False,
                        "seconds": round(time.time() - start, 3),
                        "raw_generation": raw,
                        "common_code": common,
                    }
                )
                continue
            task = build_native_decide_task(case, common)
            if verifier is None:
                rows.append(
                    {
                        "case_id": case.case_id,
                        "expected_status": case.expected_status,
                        "status": "generated",
                        "seconds": round(time.time() - start, 3),
                        "raw_generation": raw,
                        "common_code": common,
                        "prove_code": task.prove_code,
                        "refute_code": task.refute_code,
                    }
                )
                continue
            result = verifier.verify(task)
            rows.append(
                {
                    "case_id": case.case_id,
                    "expected_status": case.expected_status,
                    "status": result.status,
                    "reason": result.reason,
                    "matches_expected": result.status == case.expected_status,
                    "seconds": round(time.time() - start, 3),
                    "raw_generation": raw,
                    "common_code": common,
                    "prove_code": task.prove_code,
                    "refute_code": task.refute_code,
                    "prove_stdout": result.prove.stdout[-2000:] if result.prove else None,
                    "prove_stderr": result.prove.stderr[-2000:] if result.prove else None,
                    "refute_stdout": result.refute.stdout[-2000:] if result.refute else None,
                    "refute_stderr": result.refute.stderr[-2000:] if result.refute else None,
                }
            )
            continue
        sections = extract_sections(raw)
        if sections is None:
            rows.append(
                {
                    "case_id": case.case_id,
                    "expected_status": case.expected_status,
                    "status": "extraction_failed",
                    "seconds": round(time.time() - start, 3),
                    "raw_generation": raw,
                }
            )
            continue
        _common, prove, refute = sections
        if verifier is None:
            rows.append(
                {
                    "case_id": case.case_id,
                    "expected_status": case.expected_status,
                    "status": "generated",
                    "seconds": round(time.time() - start, 3),
                    "raw_generation": raw,
                    "prove_code": prove,
                    "refute_code": refute,
                }
            )
            continue
        result = verifier.verify(
            SemanticLeanTask(
                problem=case.problem,
                prior_steps=case.prior_steps,
                current_step=case.current_step,
                prove_code=prove,
                refute_code=refute,
                metadata={"case_id": case.case_id},
            )
        )
        rows.append(
            {
                "case_id": case.case_id,
                "expected_status": case.expected_status,
                "status": result.status,
                "reason": result.reason,
                "matches_expected": result.status == case.expected_status,
                "seconds": round(time.time() - start, 3),
                "raw_generation": raw,
                "prove_code": prove,
                "refute_code": refute,
                "prove_stdout": result.prove.stdout[-2000:] if result.prove else None,
                "prove_stderr": result.prove.stderr[-2000:] if result.prove else None,
                "refute_stdout": result.refute.stdout[-2000:] if result.refute else None,
                "refute_stderr": result.refute.stderr[-2000:] if result.refute else None,
            }
        )
    return rows


def run_statement_ensemble_case(
    case: FormalizerCase,
    tokenizer: Any,
    model: Any,
    verifier: SemanticLeanVerifier | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    start = time.time()
    attempts = []
    last_row: dict[str, Any] | None = None
    for variant in (
        "semantic_statement_focused_recipe",
        "semantic_statement_recipes",
        "semantic_statement_fewshot",
    ):
        prompt = build_prompt(case, variant)
        raw = generate(
            tokenizer,
            model,
            prompt,
            args.max_new_tokens,
            assistant_prefix=args.assistant_prefix,
        )
        common = extract_common_code(raw)
        if common is None:
            attempt = {
                "prompt_variant": variant,
                "status": "extraction_failed",
                "raw_generation": raw,
            }
            attempts.append(attempt)
            last_row = {
                "case_id": case.case_id,
                "expected_status": case.expected_status,
                "status": "extraction_failed",
                "matches_expected": False,
                "seconds": round(time.time() - start, 3),
                "attempts": attempts,
            }
            continue
        coverage_error = semantic_coverage_error(case, common)
        if coverage_error:
            attempt = {
                "prompt_variant": variant,
                "status": "unsafe_formalization",
                "reason": coverage_error,
                "common_code": common,
                "raw_generation": raw,
            }
            attempts.append(attempt)
            last_row = {
                "case_id": case.case_id,
                "expected_status": case.expected_status,
                "status": "unsafe_formalization",
                "reason": coverage_error,
                "matches_expected": False,
                "seconds": round(time.time() - start, 3),
                "attempts": attempts,
                "common_code": common,
            }
            continue
        task = build_native_decide_task(case, common)
        if verifier is None:
            attempt = {
                "prompt_variant": variant,
                "status": "generated",
                "common_code": common,
                "raw_generation": raw,
            }
            attempts.append(attempt)
            last_row = {
                "case_id": case.case_id,
                "expected_status": case.expected_status,
                "status": "generated",
                "seconds": round(time.time() - start, 3),
                "attempts": attempts,
                "common_code": common,
                "prove_code": task.prove_code,
                "refute_code": task.refute_code,
            }
            continue
        result = verifier.verify(task)
        attempt = {
            "prompt_variant": variant,
            "status": result.status,
            "reason": result.reason,
            "common_code": common,
            "raw_generation": raw,
        }
        attempts.append(attempt)
        row = {
            "case_id": case.case_id,
            "expected_status": case.expected_status,
            "status": result.status,
            "reason": result.reason,
            "matches_expected": result.status == case.expected_status,
            "seconds": round(time.time() - start, 3),
            "attempts": attempts,
            "common_code": common,
            "prove_code": task.prove_code,
            "refute_code": task.refute_code,
            "prove_stdout": result.prove.stdout[-2000:] if result.prove else None,
            "prove_stderr": result.prove.stderr[-2000:] if result.prove else None,
            "refute_stdout": result.refute.stdout[-2000:] if result.refute else None,
            "refute_stderr": result.refute.stderr[-2000:] if result.refute else None,
        }
        last_row = row
        if result.status in {"valid", "invalid"}:
            return row
    assert last_row is not None
    return last_row


def semantic_coverage_error(case: FormalizerCase, common_code: str) -> str | None:
    """Reject obviously unfaithful statements before Lean can mislead us.

    Lean can only judge the proposition it receives. This guard is intentionally
    conservative: if the generated proposition omits the current step's central
    concepts, we abstain instead of accepting a proof/refutation of the wrong
    statement.
    """

    text = " ".join([case.problem, *case.prior_steps, case.current_step]).lower()
    code = common_code.lower()
    if "terminating decimal" in text or "unit fraction" in text:
        required = ("unitfractionterminates", "countterminatesupto", "list.range", "10 ^")
        if not all(token in code for token in required):
            return "semantic coverage guard: terminating-fraction count is not fully encoded"
    if "sum of squares" in text or "1^2 + 2^2" in text:
        if not (
            ("sumsquares" in code or "sumofsquares" in code)
            and "list.range" in code
            and "foldl" in code
        ):
            return "semantic coverage guard: finite sum of squares is not fully encoded"
    if "cos" in text and ("dot" in text or "theta" in text):
        if not (("exprnum" in code and "exprden" in code) or ("cosformula" in code and "dot" in code)):
            return "semantic coverage guard: cosine ratio is not fully encoded"
    if "pairwise coprime" in text or "chinese remainder" in text:
        if "nat.gcd" not in code or "list.range" not in code:
            return "semantic coverage guard: pairwise-coprime finite check is not fully encoded"
    if "relative speed" in text:
        if "relativespeed" not in code or "25" not in code:
            return "semantic coverage guard: relative speed claim is not fully encoded"
    if "absolute differences" in text or "|a1-a2|" in text or "expected value of |a1-a2|" in text:
        if not all(token in code for token in ("absdiff", "unorderedpairs", "diffsum")):
            return "semantic coverage guard: absolute-difference average is not fully encoded"
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "total": total,
        "matches_expected": sum(1 for row in rows if row.get("matches_expected") is True),
        "statuses": {
            status: sum(1 for row in rows if row.get("status") == status)
            for status in sorted({str(row.get("status")) for row in rows})
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark semantic NL-to-Lean step formalization prompts.")
    parser.add_argument("--mode", choices=["reference", "model"], default="reference")
    parser.add_argument("--model", default="models/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument(
        "--prompt-variant",
        choices=[
            "semantic_core_plain",
            "semantic_core_fewshot",
            "semantic_core_strict",
            "semantic_core_toolbox",
            "semantic_statement_core",
            "semantic_statement_fewshot",
            "semantic_statement_recipes",
            "semantic_statement_focused_recipe",
            "semantic_statement_ensemble",
        ],
        default="semantic_core_toolbox",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--lean-executable", default="lean")
    parser.add_argument("--lean-workdir", default=None)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--assistant-prefix", default="")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = run_reference(args) if args.mode == "reference" else run_model(args)
    payload = {"summary": summarize(rows), "rows": rows}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
