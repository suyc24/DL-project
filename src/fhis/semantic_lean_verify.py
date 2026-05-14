"""Whole-step Lean verification for FHIS routing.

This verifier checks a complete natural-language step, not a convenient
arithmetic fragment. A formalizer emits two standalone Lean programs:

* ``current_step_valid`` proves the current step from the problem and accepted
  prior steps.
* ``current_step_invalid`` proves the negation/refutation of the current step.

If Lean proves the first theorem, the step is accepted. If Lean proves the
second theorem, the step is rejected. If neither side compiles, the result is
``unknown`` rather than mathematical failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Literal


SemanticLeanStatus = Literal[
    "valid",
    "invalid",
    "unknown",
    "formalization_failed",
    "unsafe_formalization",
    "inconsistent_formalization",
]
SemanticLeanDecision = Literal["accept", "reject", "abstain"]


FORBIDDEN_LEAN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsorry\b"),
    re.compile(r"\badmit\b"),
    re.compile(r"\baxiom\b"),
    re.compile(r"\bconstant\b"),
    re.compile(r"\bopaque\b"),
    re.compile(r"\bunsafe\b"),
    re.compile(r"\bby_contra!\b"),
    re.compile(r"set_option\s+autoImplicit\s+true"),
)


@dataclass(frozen=True)
class LeanRun:
    ok: bool
    stdout: str
    stderr: str
    elapsed_s: float
    returncode: int


@dataclass(frozen=True)
class SemanticLeanTask:
    problem: str
    prior_steps: list[str]
    current_step: str
    prove_code: str
    refute_code: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticLeanResult:
    status: SemanticLeanStatus
    reason: str
    prove: LeanRun | None = None
    refute: LeanRun | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticLeanPolicy:
    """Policy for combining probe risk with Lean's semantic result."""

    abstain_on_high_risk_unknown: bool = True

    def decide(self, result: SemanticLeanResult, *, probe_high_risk: bool) -> SemanticLeanDecision:
        if result.status == "valid":
            return "accept"
        if result.status == "invalid":
            return "reject"
        if probe_high_risk and self.abstain_on_high_risk_unknown:
            return "abstain"
        return "accept"


def strip_lean_comments(code: str) -> str:
    code = re.sub(r"/--.*?-/", "", code, flags=re.S)
    code = re.sub(r"/-.*?-/", "", code, flags=re.S)
    return re.sub(r"--.*", "", code)


def lean_command(workdir: str | Path | None = None, executable: str = "lean") -> list[str]:
    root = Path(workdir or ".")
    executable_parts = shlex.split(executable)
    if not executable_parts:
        executable_parts = ["lean"]
    if (root / "lakefile.lean").exists() or (root / "lakefile.toml").exists():
        return ["lake", "env", *executable_parts]
    return executable_parts


class SemanticLeanVerifier:
    """Run paired proof/refutation Lean checks for a full reasoning step."""

    def __init__(
        self,
        lean_bin: str = "lean",
        timeout_s: float = 20.0,
        workdir: str | Path | None = None,
    ) -> None:
        self.lean_bin = lean_bin
        self.timeout_s = timeout_s
        self.workdir = Path(workdir) if workdir is not None else None

    def lint(self, code: str, required_theorem: str) -> str | None:
        text = strip_lean_comments(code)
        for pattern in FORBIDDEN_LEAN_PATTERNS:
            match = pattern.search(text)
            if match:
                return f"forbidden Lean construct: {match.group(0)}"
        if not re.search(rf"\btheorem\s+{re.escape(required_theorem)}\b", text):
            return f"missing theorem `{required_theorem}`"
        theorem_head = re.search(
            rf"\btheorem\s+{re.escape(required_theorem)}\b(?P<head>.*?):=",
            text,
            flags=re.S,
        )
        if theorem_head and re.search(
            r"\((h|hyp|claim|step)\s*:\s*[^)]*current",
            theorem_head.group("head"),
            re.I,
        ):
            return "current step appears to be assumed in the theorem header"
        return None

    def run_lean(self, code: str) -> LeanRun:
        start = time.time()
        proc = subprocess.run(
            [*lean_command(self.workdir, self.lean_bin), "--stdin"],
            cwd=self.workdir,
            input=code,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_s,
            check=False,
        )
        return LeanRun(
            ok=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_s=time.time() - start,
            returncode=proc.returncode,
        )

    def verify(self, task: SemanticLeanTask) -> SemanticLeanResult:
        for code, theorem in (
            (task.prove_code, "current_step_valid"),
            (task.refute_code, "current_step_invalid"),
        ):
            lint_error = self.lint(code, theorem)
            if lint_error:
                return SemanticLeanResult(
                    status="unsafe_formalization",
                    reason=f"{theorem}: {lint_error}",
                    metadata=task.metadata,
                )

        try:
            prove = self.run_lean(task.prove_code)
        except subprocess.TimeoutExpired:
            prove = LeanRun(False, "", "timeout", self.timeout_s, 124)
        try:
            refute = self.run_lean(task.refute_code)
        except subprocess.TimeoutExpired:
            refute = LeanRun(False, "", "timeout", self.timeout_s, 124)

        if prove.ok and refute.ok:
            return SemanticLeanResult(
                "inconsistent_formalization",
                "both proof and refutation compiled",
                prove,
                refute,
                task.metadata,
            )
        if prove.ok:
            return SemanticLeanResult(
                "valid",
                "Lean proved the full current-step theorem",
                prove,
                refute,
                task.metadata,
            )
        if refute.ok:
            return SemanticLeanResult(
                "invalid",
                "Lean proved the full current-step refutation",
                prove,
                refute,
                task.metadata,
            )
        prove_text = f"{prove.stdout}\n{prove.stderr}"
        refute_text = f"{refute.stdout}\n{refute.stderr}"
        if "error:" in prove_text or "error:" in refute_text:
            return SemanticLeanResult(
                "unknown",
                "Lean could not prove either side; do not treat as math-invalid",
                prove,
                refute,
                task.metadata,
            )
        return SemanticLeanResult(
            "formalization_failed",
            "Lean produced no decisive theorem result",
            prove,
            refute,
            task.metadata,
        )


def build_semantic_formalizer_prompt(
    problem: str,
    prior_steps: list[str],
    current_step: str,
) -> str:
    prior = "\n".join(f"- {step}" for step in prior_steps) or "- none"
    return f"""You are a Lean 4 formalizer for chain-of-thought verification.

Goal: formalize the CURRENT STEP, not a convenient subclaim. Include every
concept and condition needed for Lean to recognize whether the step follows
from the problem and accepted prior steps.

Rules:
1. Produce two standalone Lean 4 files: PROVE_CODE and REFUTE_CODE.
2. PROVE_CODE must contain theorem `current_step_valid` proving the current step.
3. REFUTE_CODE must contain theorem `current_step_invalid` proving the negation
   or refutation of the current step.
4. You may encode accepted prior steps as hypotheses or previously proved
   theorems, but you must not assume the current step.
5. Do not use sorry, admit, axiom, constant, unsafe, or new axiomatized facts.
6. Prefer computable finite definitions when the problem is finite; otherwise
   use faithful mathematical definitions and Lean tactics/theorems.
7. If a proof is not possible, still return the best faithful formalization
   with an incomplete proof attempt that Lean will reject; the verifier will
   mark it unknown rather than invalid.

PROBLEM:
{problem}

ACCEPTED PRIOR STEPS:
{prior}

CURRENT STEP:
{current_step}
"""
