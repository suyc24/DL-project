"""Per-step DeepSeek v4 Pro Lean transition-contract suite.

This script evaluates local proof-state transitions. Each task supplies a trusted
Lean prefix and asks DeepSeek v4 Pro to generate only the proof/code for one
local theorem. Repair uses compiler feedback only.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "pro_step_suite"
LEAN_CWD = ROOT.parent / "lean_fhis"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-pro"

SYSTEM = """You are a Lean 4 proof-term generator. Return ONLY Lean code, no markdown and no prose.
You are proving one local transition contract under a trusted prefix. Do not redefine trusted names.
No sorry, no axiom, no admit, no placeholder global lemmas."""

DAY1_PREFIX = """
import Mathlib

namespace StepSuite

open Finset

noncomputable def Day1P2NormSq (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  ∑ i : Fin n, (x i)^2

noncomputable def Day1P2QForm (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  ∑ i : Fin n, ∑ j : Fin n,
    (((n : ℝ) - (Nat.dist i.val j.val : ℝ)) * x i * x j)

noncomputable def Day1P2GoodConstant (c : ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → ∀ x : Fin n → ℝ,
    Day1P2QForm n x ≥ c * Day1P2NormSq n x

noncomputable def Day1P2Answer : ℝ := (1 : ℝ) / 2

noncomputable def Day1P2Target : Prop :=
  Day1P2GoodConstant Day1P2Answer ∧
    ∀ c : ℝ, Day1P2GoodConstant c → c ≤ Day1P2Answer
"""

DAY2_PREFIX = """
import Mathlib

namespace StepSuite

open Finset

noncomputable def PairSet2023 (a : Fin 2023 → ℝ) : Finset (Fin 2023 × Fin 2023) := by
  classical
  exact ((Finset.univ : Finset (Fin 2023)).product (Finset.univ : Finset (Fin 2023))).filter
    (fun ij => ij.1.val ≤ ij.2.val ∧ 1 ≤ a ij.1 * a ij.2)

noncomputable def JSet2023 (a : Fin 2023 → ℝ) (i : Fin 2023) : Finset (Fin 2023) := by
  classical
  exact (Finset.univ : Finset (Fin 2023)).filter (fun j => 1 ≤ a i * a j)

noncomputable def LargeSet2023 (a : Fin 2023 → ℝ) : Finset (Fin 2023) := by
  classical
  exact (Finset.univ : Finset (Fin 2023)).filter (fun i => 1 ≤ a i * a i)

noncomputable def Day2P4Upper : Prop :=
  ∀ a : Fin 2023 → ℝ,
    (∀ i, 0 ≤ a i) →
    (∑ i : Fin 2023, a i) = (100 : ℝ) →
    (PairSet2023 a).card ≤ 5050

noncomputable def Day2P4Achievable : Prop :=
  ∃ a : Fin 2023 → ℝ,
    (∀ i, 0 ≤ a i) ∧
    (∑ i : Fin 2023, a i) = (100 : ℝ) ∧
    (PairSet2023 a).card = 5050

noncomputable def Day2P4Target : Prop :=
  Day2P4Upper ∧ Day2P4Achievable
"""

OTHER_PREFIX = """
import Mathlib

namespace StepSuite

open Finset Equiv

-- Day 2 Problem 6: arrangements on 99-gon modulo rotation.
abbrev Arrangement99 := Equiv.Perm (Fin 99)

-- A deliberately minimal formal placeholder for rotation equivalence.
-- The hard domain-specific definitions are part of statement formalization.
def SameUpToRotation99 (_σ _τ : Arrangement99) : Prop := True

-- Day 1 Problem 1 skeleton: factor-choice predicate, intentionally abstract.
def Day1P1GoodLambda (_lambda : ℝ) : Prop := True

-- Day 1 Problem 3 skeleton: good set predicate, intentionally abstract.
def Day1P3GoodSet (_p : ℕ) (_A : Finset ℕ) : Prop := True
"""

TASKS = [
  {
    "id": "day1_step1_define_target",
    "problem": "Day1 P2",
    "nl_step": "把原题形式化为 GoodConstant c，并把答案写成 Day1P2Target：1/2 可行且任何可行 c 都不超过 1/2。",
    "prefix": DAY1_PREFIX,
    "signature": "theorem day1_step1_target_unfold_contract :\n    Day1P2Target ↔\n      (Day1P2GoodConstant Day1P2Answer ∧\n        ∀ c : ℝ, Day1P2GoodConstant c → c ≤ Day1P2Answer) := by\n  ...",
  },
  {
    "id": "day1_step2_final_intro",
    "problem": "Day1 P2",
    "nl_step": "若已证明 1/2 可行并证明任意可行常数 c≤1/2，则原目标成立。",
    "prefix": DAY1_PREFIX,
    "signature": "theorem day1_step2_final_intro_contract\n    (h_good : Day1P2GoodConstant Day1P2Answer)\n    (h_upper : ∀ c : ℝ, Day1P2GoodConstant c → c ≤ Day1P2Answer) :\n    Day1P2Target := by\n  ...",
  },
  {
    "id": "day1_step3_lower_from_key_lemma",
    "problem": "Day1 P2",
    "nl_step": "为证明 1/2 可行，只需证明关键二次型不等式对所有 n,x 成立。",
    "prefix": DAY1_PREFIX,
    "signature": "theorem day1_step3_lower_from_key_contract\n    (h_key : ∀ n : ℕ, 0 < n → ∀ x : Fin n → ℝ,\n      Day1P2QForm n x ≥ Day1P2Answer * Day1P2NormSq n x) :\n    Day1P2GoodConstant Day1P2Answer := by\n  ...",
  },
  {
    "id": "day1_step4_upper_from_counterexamples",
    "problem": "Day1 P2",
    "nl_step": "为证明上界，只需证明任何 c>1/2 都不是 GoodConstant。",
    "prefix": DAY1_PREFIX,
    "signature": "theorem day1_step4_upper_from_counterexamples_contract\n    (h_bad_above : ∀ c : ℝ, Day1P2Answer < c → ¬ Day1P2GoodConstant c) :\n    ∀ c : ℝ, Day1P2GoodConstant c → c ≤ Day1P2Answer := by\n  ...",
  },
  {
    "id": "day1_step5_counterexample_to_not_good",
    "problem": "Day1 P2",
    "nl_step": "若对某 c 找到 n>0 和 x 使 Q < c·normSq，则 c 不是 GoodConstant。",
    "prefix": DAY1_PREFIX,
    "signature": "theorem day1_step5_counterexample_to_not_good_contract\n    (c : ℝ)\n    (h_ex : ∃ n : ℕ, ∃ hn : 0 < n, ∃ x : Fin n → ℝ,\n      Day1P2QForm n x < c * Day1P2NormSq n x) :\n    ¬ Day1P2GoodConstant c := by\n  ...",
  },
  {
    "id": "day1_step6_detect_wrong_constant",
    "problem": "Day1 P2",
    "nl_step": "错误候选：若答案是 2 - sqrt 3，也应能由 1/2 的目标推出；这个 transition 应失败或暴露额外数学义务。",
    "prefix": DAY1_PREFIX,
    "signature": "theorem day1_step6_wrong_constant_contract\n    (h_target : Day1P2Target) :\n    Day1P2GoodConstant (2 - Real.sqrt 3) ∧\n      ∀ c : ℝ, Day1P2GoodConstant c → c ≤ 2 - Real.sqrt 3 := by\n  ...",
  },
  {
    "id": "day2_step1_target_intro",
    "problem": "Day2 P4",
    "nl_step": "最大值为 5050 等价于上界 Day2P4Upper 与可达性 Day2P4Achievable 同时成立。",
    "prefix": DAY2_PREFIX,
    "signature": "theorem day2_step1_target_intro_contract\n    (h_upper : Day2P4Upper)\n    (h_ach : Day2P4Achievable) :\n    Day2P4Target := by\n  ...",
  },
  {
    "id": "day2_step2_upper_from_counting",
    "problem": "Day2 P4",
    "nl_step": "上界证明的最后组合：由 ordered-pair 计数、total≤10000、large≤100 推出 |A|≤5050。",
    "prefix": DAY2_PREFIX + "\n" + "theorem final_arithmetic_contract (m k total : ℕ)\n    (h_eq : total + k = 2 * m)\n    (h_total : total ≤ 10000)\n    (h_k : k ≤ 100) :\n    m ≤ 5050 := by\n  omega\n",
    "signature": "theorem day2_step2_upper_from_counting_contract\n    (a : Fin 2023 → ℝ)\n    (total : ℕ)\n    (h_ordered : total + (LargeSet2023 a).card = 2 * (PairSet2023 a).card)\n    (h_total : total ≤ 10000)\n    (h_large : (LargeSet2023 a).card ≤ 100) :\n    (PairSet2023 a).card ≤ 5050 := by\n  ...",
  },
  {
    "id": "day2_step3_achievable_from_witness_facts",
    "problem": "Day2 P4",
    "nl_step": "为证明可达性，只需给出一个 a，并证明非负、和为 100、PairSet cardinal 为 5050。",
    "prefix": DAY2_PREFIX,
    "signature": "theorem day2_step3_achievable_from_witness_contract\n    (a : Fin 2023 → ℝ)\n    (h_nonneg : ∀ i, 0 ≤ a i)\n    (h_sum : (∑ i : Fin 2023, a i) = (100 : ℝ))\n    (h_card : (PairSet2023 a).card = 5050) :\n    Day2P4Achievable := by\n  ...",
  },
  {
    "id": "day2_step4_bad_concentration_claim",
    "problem": "Day2 P4",
    "nl_step": "可疑/错误步骤：最大值必在把总和集中到若干个相等正数，其余为 0 的构型中取得。尝试把它作为无额外假设的 Lean transition。",
    "prefix": DAY2_PREFIX,
    "signature": "theorem day2_step4_bad_concentration_claim_contract :\n    ∀ a : Fin 2023 → ℝ,\n      (∀ i, 0 ≤ a i) →\n      (∑ i : Fin 2023, a i) = (100 : ℝ) →\n      ∃ b : Fin 2023 → ℝ,\n        (∀ i, 0 ≤ b i) ∧\n        (∑ i : Fin 2023, b i) = (100 : ℝ) ∧\n        (PairSet2023 a).card ≤ (PairSet2023 b).card ∧\n        (∃ m : ℕ, ∀ i : Fin 2023, b i = 0 ∨ b i = (100 : ℝ) / (m : ℝ)) := by\n  ...",
  },
  {
    "id": "other_day2_p6_rotation_equiv_gap",
    "problem": "Day2 P6",
    "nl_step": "其他题尝试：把‘旋转后重合视为相同’作为形式化对象。当前 prefix 只给了 True 占位，说明 statement formalization 未完成，transition 可能 vacuous。",
    "prefix": OTHER_PREFIX,
    "signature": "theorem day2_p6_vacuous_rotation_contract (σ τ : Arrangement99) : SameUpToRotation99 σ τ := by\n  ...",
  },
]


def call_deepseek(messages: list[dict[str, str]], max_tokens: int = 1200) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    payload = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=240) as resp:
        data = json.loads(resp.read().decode())
    msg = data["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content", "")


def clean_code(text: str) -> str:
    m = re.search(r"```(?:lean4?|lean)?\n(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    return text.strip() + "\n"


def lean_check(path: pathlib.Path) -> tuple[bool, str]:
    cmd = ["bash", "-lc", f"source \"$HOME/.elan/env\" && cd '{LEAN_CWD}' && lake env lean '{path}'"]
    proc = subprocess.run(cmd, cwd=ROOT.parent, text=True, capture_output=True, timeout=180)
    return proc.returncode == 0, proc.stdout + proc.stderr


def run_task(task: dict[str, str], max_rounds: int = 3) -> dict[str, object]:
    d = OUT / task["id"]
    d.mkdir(parents=True, exist_ok=True)
    prompt = "Trusted Lean prefix:\n```lean\n" + task["prefix"] + "\n```\n\nNatural-language step:\n" + task["nl_step"] + "\n\nGenerate exactly this theorem and proof:\n```lean\n" + task["signature"] + "\n```"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    last = ""
    for r in range(1, max_rounds + 1):
        raw = call_deepseek(messages)
        code = clean_code(raw)
        (d / f"round{r}.raw.md").write_text(raw, encoding="utf-8")
        candidate = task["prefix"] + "\n" + code + "\nend StepSuite\n"
        lean_path = d / f"round{r}.lean"
        lean_path.write_text(candidate, encoding="utf-8")
        ok, log = lean_check(lean_path)
        (d / f"round{r}.lean.log").write_text(log, encoding="utf-8")
        if ok:
            accepted = d / "accepted.lean"
            accepted.write_text(candidate, encoding="utf-8")
            return {"id": task["id"], "problem": task["problem"], "ok": True, "rounds": r, "accepted": str(accepted), "nl_step": task["nl_step"]}
        last = log[-6000:]
        messages.append({"role": "assistant", "content": code})
        messages.append({"role": "user", "content": "Lean compiler rejected the code. Repair using ONLY this compiler feedback. Do not change trusted prefix or theorem signature.\n\n" + last})
    return {"id": task["id"], "problem": task["problem"], "ok": False, "rounds": max_rounds, "last_error": last, "nl_step": task["nl_step"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = [run_task(t) for t in TASKS]
    (OUT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
