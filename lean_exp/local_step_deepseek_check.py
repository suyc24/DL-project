"""Ask DeepSeek v4 Pro to generate Lean checkers for individual local steps."""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "local_step_checks"
LEAN_CWD = ROOT.parent / "lean_fhis"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-pro"

SYSTEM = """You are a Lean 4 transition-contract generator.
Return ONLY Lean code, no markdown and no prose.
Given a trusted prefix and a natural-language local proof step, generate a Lean theorem that checks the step.
Do not use sorry, axiom, admit, or placeholder global lemmas.
If the natural-language step is not directly provable from the current context, expose the missing facts as explicit theorem hypotheses named h_missing_*.
"""

PREFIX = """
import Mathlib

namespace LocalStepCheck

open Finset

noncomputable def PairSet2023 (a : Fin 2023 → ℝ) : Finset (Fin 2023 × Fin 2023) := by
  classical
  exact ((Finset.univ : Finset (Fin 2023)).product (Finset.univ : Finset (Fin 2023))).filter
    (fun ij => ij.1.val ≤ ij.2.val ∧ 1 ≤ a ij.1 * a ij.2)

noncomputable def LargeSet2023 (a : Fin 2023 → ℝ) : Finset (Fin 2023) := by
  classical
  exact (Finset.univ : Finset (Fin 2023)).filter (fun i => 1 ≤ a i * a i)

theorem final_arithmetic_contract (m k total : ℕ)
    (h_eq : total + k = 2 * m)
    (h_total : total ≤ 10000)
    (h_k : k ≤ 100) :
    m ≤ 5050 := by
  omega
"""

TASKS = {
    "valid_counting_step": """
Current proof state:
- a : Fin 2023 → ℝ
- total : ℕ
- h_ordered : total + (LargeSet2023 a).card = 2 * (PairSet2023 a).card
- h_total : total ≤ 10000
- h_large : (LargeSet2023 a).card ≤ 100
Goal: (PairSet2023 a).card ≤ 5050

Natural-language local step:
Using ordered-pair counting, total≤10000, and large≤100, conclude |A|≤5050.

Generate a Lean theorem that checks exactly this transition.
""",
    "suspicious_concentration_step": """
Current proof state:
- a : Fin 2023 → ℝ
- h_nonneg : ∀ i, 0 ≤ a i
- h_sum : (∑ i : Fin 2023, a i) = (100 : ℝ)
Goal: ∃ b : Fin 2023 → ℝ,
  (∀ i, 0 ≤ b i) ∧
  (∑ i : Fin 2023, b i) = (100 : ℝ) ∧
  (PairSet2023 a).card ≤ (PairSet2023 b).card ∧
  (∃ m : ℕ, ∀ i : Fin 2023, b i = 0 ∨ b i = (100 : ℝ) / (m : ℝ))

Natural-language local step:
The maximum occurs when the total mass is concentrated into several equal positive entries and all other entries are zero.

Generate a Lean theorem that checks this transition. If the step is not directly provable, expose the missing mathematical claim as explicit h_missing_* hypotheses.
""",
}


def call(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2200,
        "stream": False,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"], "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        data = json.loads(resp.read().decode())
    msg = data["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content", "")


def clean(text: str) -> str:
    m = re.search(r"```(?:lean4?|lean)?\n(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    return text.strip() + "\n"


def check(path: pathlib.Path) -> tuple[bool, str]:
    cmd = ["bash", "-lc", f"source \"$HOME/.elan/env\" && cd '{LEAN_CWD}' && lake env lean '{path}'"]
    proc = subprocess.run(cmd, cwd=ROOT.parent, text=True, capture_output=True, timeout=180)
    return proc.returncode == 0, proc.stdout + proc.stderr


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name, task in TASKS.items():
        prompt = "Trusted Lean prefix:\n```lean\n" + PREFIX + "\n```\n\n" + task
        raw = call(prompt)
        code = clean(raw)
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "raw.md").write_text(raw, encoding="utf-8")
        lean = PREFIX + "\n" + code + "\nend LocalStepCheck\n"
        lean_path = d / "checker.lean"
        lean_path.write_text(lean, encoding="utf-8")
        ok, log = check(lean_path)
        (d / "checker.log").write_text(log, encoding="utf-8")
        results.append({"name": name, "ok": ok, "path": str(lean_path), "log": log[-1000:]})
    (OUT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
