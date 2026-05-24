"""All-step local transition-contract suite using DeepSeek v4 Pro."""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "all_step_checks"
LEAN_CWD = ROOT.parent / "lean_fhis"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-pro"

SYSTEM = """You are a Lean 4 proof generator. Return ONLY Lean code, no markdown, no prose.
Complete the exact theorem proof under the trusted prefix. Do not redefine trusted names.
No sorry, no axiom, no admit."""

PREFIX = """
import Mathlib

namespace AllStepChecks

open Finset
set_option linter.unusedVariables false

noncomputable def Day1P1GoodLambda (lambda : ℝ) : Prop :=
  ∀ n : ℕ, 0 < n →
    ∃ x : Fin 2023 → ℕ,
      (∀ i, 0 < x i) ∧
      (∏ i : Fin 2023, x i) = n ∧
      ∀ i : Fin 2023,
        Nat.Prime (x i) ∨ (x i : ℝ) ≤ Real.rpow (n : ℝ) lambda

noncomputable def Day1P1Target (lambda : ℝ) : Prop :=
  Day1P1GoodLambda lambda ∧
    ∀ mu : ℝ, Day1P1GoodLambda mu → lambda ≤ mu

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

def Rval (p : ℕ) (x y : Fin p) : ℕ :=
  if x.val ≤ y.val then y.val - x.val else y.val + p - x.val

def Fval (p : ℕ) (A : Finset (Fin p)) : ℕ :=
  A.sum (fun x => A.sum (fun y => (Rval p x y) ^ 2))

def Day1P3GoodSet (p : ℕ) (A : Finset (Fin p)) : Prop :=
  0 < A.card ∧ A.card < p ∧
    ∀ B : Finset (Fin p), B.card = A.card → Fval p B ≥ Fval p A

def Day1P3Chain (p l : ℕ) : Prop :=
  ∃ As : Fin l → Finset (Fin p),
    (∀ i, Day1P3GoodSet p (As i)) ∧
    (∀ i j : Fin l, i.val ≤ j.val → As i ⊆ As j) ∧
    (∀ i j : Fin l, i ≠ j → As i ≠ As j)

def Day1P3Target (answer : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → 5 ≤ p →
    Day1P3Chain p answer ∧ ∀ l : ℕ, Day1P3Chain p l → l ≤ answer

noncomputable def PairSet2023 (a : Fin 2023 → ℝ) : Finset (Fin 2023 × Fin 2023) := by
  classical
  exact ((Finset.univ : Finset (Fin 2023)).product (Finset.univ : Finset (Fin 2023))).filter
    (fun ij => ij.1.val ≤ ij.2.val ∧ 1 ≤ a ij.1 * a ij.2)

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

abbrev Point := ℝ × ℝ

def Angle (_A _B _C : Point) : ℝ := 0

def Dist (_A _B : Point) : ℝ := 0

def Day2P5Config (_A _B _C _K _P _Q _T : Point) : Prop := True

def Day2P5Goal1 (A B C K P Q T : Point) : Prop :=
  Angle B T C + Angle A P B = Angle C Q A

def Day2P5Goal2 (A B C K P Q T : Point) : Prop :=
  Dist A P * Dist B T * Dist C Q = Dist A Q * Dist C T * Dist B P

def Day2P5Target (A B C K P Q T : Point) : Prop :=
  Day2P5Config A B C K P Q T →
    Day2P5Goal1 A B C K P Q T ∧ Day2P5Goal2 A B C K P Q T

abbrev Arrangement99 := Equiv.Perm (ZMod 99)

def Rotation99 (k : ZMod 99) : Equiv.Perm (ZMod 99) where
  toFun x := x + k
  invFun x := x - k
  left_inv := by intro x; simp
  right_inv := by intro x; simp

def SameUpToRotation99 (σ τ : Arrangement99) : Prop :=
  ∃ k : ZMod 99, τ = σ * Rotation99 k

def OneAdjacentSwap99 (σ τ : Arrangement99) : Prop :=
  ∃ v : ZMod 99, τ = Equiv.swap v (v + 1) * σ

def ReachWithin99 : ℕ → Arrangement99 → Arrangement99 → Prop
  | 0, σ, τ => σ = τ
  | n + 1, σ, τ => σ = τ ∨ ∃ ρ, OneAdjacentSwap99 σ ρ ∧ ReachWithin99 n ρ τ

def Day2P6Upper (N : ℕ) : Prop :=
  ∀ σ τ : Arrangement99, ∃ τ' : Arrangement99,
    SameUpToRotation99 τ τ' ∧ ReachWithin99 N σ τ'

def Day2P6Target (N : ℕ) : Prop :=
  Day2P6Upper N ∧ ∀ M : ℕ, Day2P6Upper M → N ≤ M
"""

TASKS = [
("d1p1_factor_good", "valid", "theorem d1p1_factor_good (lambda : ℝ)\n    (h_missing_factor : ∀ n : ℕ, 0 < n →\n      ∃ x : Fin 2023 → ℕ,\n        (∀ i, 0 < x i) ∧\n        (∏ i : Fin 2023, x i) = n ∧\n        ∀ i : Fin 2023,\n          Nat.Prime (x i) ∨ (x i : ℝ) ≤ Real.rpow (n : ℝ) lambda) :\n    Day1P1GoodLambda lambda := by\n  ..."),
("d1p1_bad_witness", "valid", "theorem d1p1_bad_witness (lambda : ℝ)\n    (h_missing_bad : ∃ n : ℕ, ∃ hn : 0 < n,\n      ¬ ∃ x : Fin 2023 → ℕ,\n        (∀ i, 0 < x i) ∧\n        (∏ i : Fin 2023, x i) = n ∧\n        ∀ i : Fin 2023,\n          Nat.Prime (x i) ∨ (x i : ℝ) ≤ Real.rpow (n : ℝ) lambda) :\n    ¬ Day1P1GoodLambda lambda := by\n  ..."),
("d1p1_minimality", "valid", "theorem d1p1_minimality (lambda : ℝ)\n    (h_missing_bad_below : ∀ mu : ℝ, mu < lambda → ¬ Day1P1GoodLambda mu) :\n    ∀ mu : ℝ, Day1P1GoodLambda mu → lambda ≤ mu := by\n  ..."),
("d1p1_target", "valid", "theorem d1p1_target (lambda : ℝ)\n    (h_good : Day1P1GoodLambda lambda)\n    (h_min : ∀ mu : ℝ, Day1P1GoodLambda mu → lambda ≤ mu) :\n    Day1P1Target lambda := by\n  ..."),
("d1p2_target_unfold", "valid", "theorem d1p2_target_unfold :\n    Day1P2Target ↔\n      (Day1P2GoodConstant Day1P2Answer ∧\n        ∀ c : ℝ, Day1P2GoodConstant c → c ≤ Day1P2Answer) := by\n  ..."),
("d1p2_lower", "valid", "theorem d1p2_lower\n    (h_missing_key : ∀ n : ℕ, 0 < n → ∀ x : Fin n → ℝ,\n      Day1P2QForm n x ≥ Day1P2Answer * Day1P2NormSq n x) :\n    Day1P2GoodConstant Day1P2Answer := by\n  ..."),
("d1p2_upper", "valid", "theorem d1p2_upper\n    (h_missing_bad_above : ∀ c : ℝ, Day1P2Answer < c → ¬ Day1P2GoodConstant c) :\n    ∀ c : ℝ, Day1P2GoodConstant c → c ≤ Day1P2Answer := by\n  ..."),
("d1p2_counterexample", "valid", "theorem d1p2_counterexample (c : ℝ)\n    (h_ex : ∃ n : ℕ, ∃ hn : 0 < n, ∃ x : Fin n → ℝ,\n      Day1P2QForm n x < c * Day1P2NormSq n x) :\n    ¬ Day1P2GoodConstant c := by\n  ..."),
("d1p2_target", "valid", "theorem d1p2_target\n    (h_good : Day1P2GoodConstant Day1P2Answer)\n    (h_upper : ∀ c : ℝ, Day1P2GoodConstant c → c ≤ Day1P2Answer) :\n    Day1P2Target := by\n  ..."),
("d1p3_good_set", "valid", "theorem d1p3_good_set (p : ℕ) (A : Finset (Fin p))\n    (h_nonempty : 0 < A.card)\n    (h_proper : A.card < p)\n    (h_missing_min : ∀ B : Finset (Fin p), B.card = A.card → Fval p B ≥ Fval p A) :\n    Day1P3GoodSet p A := by\n  ..."),
("d1p3_chain", "valid", "theorem d1p3_chain (p l : ℕ)\n    (As : Fin l → Finset (Fin p))\n    (h_good : ∀ i, Day1P3GoodSet p (As i))\n    (h_mono : ∀ i j : Fin l, i.val ≤ j.val → As i ⊆ As j)\n    (h_distinct : ∀ i j : Fin l, i ≠ j → As i ≠ As j) :\n    Day1P3Chain p l := by\n  ..."),
("d1p3_upper", "valid", "theorem d1p3_upper (p answer : ℕ)\n    (h_missing_no_longer : ∀ l : ℕ, answer < l → ¬ Day1P3Chain p l) :\n    ∀ l : ℕ, Day1P3Chain p l → l ≤ answer := by\n  ..."),
("d1p3_target", "valid", "theorem d1p3_target (answer : ℕ)\n    (h_missing_all_p : ∀ p : ℕ, Nat.Prime p → 5 ≤ p →\n      Day1P3Chain p answer ∧ ∀ l : ℕ, Day1P3Chain p l → l ≤ answer) :\n    Day1P3Target answer := by\n  ..."),
("d2p4_target", "valid", "theorem d2p4_target\n    (h_upper : Day2P4Upper)\n    (h_ach : Day2P4Achievable) :\n    Day2P4Target := by\n  ..."),
("d2p4_arith", "valid", "theorem d2p4_arith (m k total : ℕ)\n    (h_eq : total + k = 2 * m)\n    (h_total : total ≤ 10000)\n    (h_k : k ≤ 100) :\n    m ≤ 5050 := by\n  ..."),
("d2p4_upper", "valid", "theorem d2p4_upper\n    (a : Fin 2023 → ℝ)\n    (total : ℕ)\n    (h_ordered : total + (LargeSet2023 a).card = 2 * (PairSet2023 a).card)\n    (h_total : total ≤ 10000)\n    (h_large : (LargeSet2023 a).card ≤ 100) :\n    (PairSet2023 a).card ≤ 5050 := by\n  ..."),
("d2p4_achievable", "valid", "theorem d2p4_achievable\n    (a : Fin 2023 → ℝ)\n    (h_nonneg : ∀ i, 0 ≤ a i)\n    (h_sum : (∑ i : Fin 2023, a i) = (100 : ℝ))\n    (h_card : (PairSet2023 a).card = 5050) :\n    Day2P4Achievable := by\n  ..."),
("d2p4_concentration_reduced", "reduced", "theorem d2p4_concentration_reduced\n    (a : Fin 2023 → ℝ)\n    (h_nonneg : ∀ i, 0 ≤ a i)\n    (h_sum : (∑ i : Fin 2023, a i) = (100 : ℝ))\n    (h_missing_concentration : ∃ b : Fin 2023 → ℝ,\n      (∀ i, 0 ≤ b i) ∧\n      (∑ i : Fin 2023, b i) = (100 : ℝ) ∧\n      (PairSet2023 a).card ≤ (PairSet2023 b).card ∧\n      (∃ m : ℕ, ∀ i : Fin 2023, b i = 0 ∨ b i = (100 : ℝ) / (m : ℝ))) :\n    ∃ b : Fin 2023 → ℝ,\n      (∀ i, 0 ≤ b i) ∧\n      (∑ i : Fin 2023, b i) = (100 : ℝ) ∧\n      (PairSet2023 a).card ≤ (PairSet2023 b).card ∧\n      (∃ m : ℕ, ∀ i : Fin 2023, b i = 0 ∨ b i = (100 : ℝ) / (m : ℝ)) := by\n  ..."),
("d2p5_goal_split", "reduced", "theorem d2p5_goal_split (A B C K P Q T : Point)\n    (h_goal1 : Day2P5Config A B C K P Q T → Day2P5Goal1 A B C K P Q T)\n    (h_goal2 : Day2P5Config A B C K P Q T → Day2P5Goal2 A B C K P Q T) :\n    Day2P5Target A B C K P Q T := by\n  ..."),
("d2p5_goal1_from_angle_lemma", "reduced", "theorem d2p5_goal1_from_angle_lemma (A B C K P Q T : Point)\n    (h_missing_angle : Day2P5Config A B C K P Q T →\n      Angle B T C + Angle A P B = Angle C Q A) :\n    Day2P5Config A B C K P Q T → Day2P5Goal1 A B C K P Q T := by\n  ..."),
("d2p5_goal2_from_length_lemma", "reduced", "theorem d2p5_goal2_from_length_lemma (A B C K P Q T : Point)\n    (h_missing_length : Day2P5Config A B C K P Q T →\n      Dist A P * Dist B T * Dist C Q = Dist A Q * Dist C T * Dist B P) :\n    Day2P5Config A B C K P Q T → Day2P5Goal2 A B C K P Q T := by\n  ..."),
("d2p6_reach_succ", "valid", "theorem d2p6_reach_succ (n : ℕ) (σ ρ τ : Arrangement99)\n    (h_swap : OneAdjacentSwap99 σ ρ)\n    (h_reach : ReachWithin99 n ρ τ) :\n    ReachWithin99 (n + 1) σ τ := by\n  ..."),
("d2p6_upper", "valid", "theorem d2p6_upper (N : ℕ)\n    (h_missing_representatives : ∀ σ τ : Arrangement99, ∃ τ' : Arrangement99,\n      SameUpToRotation99 τ τ' ∧ ReachWithin99 N σ τ') :\n    Day2P6Upper N := by\n  ..."),
("d2p6_minimality", "valid", "theorem d2p6_minimality (N : ℕ)\n    (h_missing_no_smaller : ∀ M : ℕ, M < N → ¬ Day2P6Upper M) :\n    ∀ M : ℕ, Day2P6Upper M → N ≤ M := by\n  ..."),
("d2p6_target", "valid", "theorem d2p6_target (N : ℕ)\n    (h_upper : Day2P6Upper N)\n    (h_min : ∀ M : ℕ, Day2P6Upper M → N ≤ M) :\n    Day2P6Target N := by\n  ..."),
]

def call(messages: list[dict[str, str]], max_tokens: int = 900) -> str:
    payload = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"], "Content-Type": "application/json"}, method="POST")
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
    proc = subprocess.run(["bash", "-lc", f"source \"$HOME/.elan/env\" && cd '{LEAN_CWD}' && lake env lean '{path}'"], cwd=ROOT.parent, text=True, capture_output=True, timeout=180)
    return proc.returncode == 0, proc.stdout + proc.stderr

def run_task(name: str, classification: str, signature: str) -> dict[str, object]:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    prompt = "Trusted prefix:\n```lean\n" + PREFIX + "\n```\n\nComplete exactly this theorem proof by replacing ...:\n```lean\n" + signature + "\n```"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    last = ""
    for r in range(1, 4):
        raw = call(messages)
        code = clean(raw)
        (d / f"round{r}.raw.md").write_text(raw, encoding="utf-8")
        candidate = PREFIX + "\n" + code + "\nend AllStepChecks\n"
        path = d / f"round{r}.lean"
        path.write_text(candidate, encoding="utf-8")
        ok, log = check(path)
        (d / f"round{r}.lean.log").write_text(log, encoding="utf-8")
        if ok:
            accepted = d / "accepted.lean"
            accepted.write_text(candidate, encoding="utf-8")
            return {"name": name, "classification": classification, "ok": True, "rounds": r, "accepted": str(accepted)}
        last = log[-3000:]
        messages.append({"role": "assistant", "content": code})
        messages.append({"role": "user", "content": "Lean rejected the code. Repair using only this compiler feedback. Keep the theorem statement exactly.\n" + last})
    return {"name": name, "classification": classification, "ok": False, "rounds": 3, "last_error": last}

def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    results = [run_task(*task) for task in TASKS]
    (OUT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    accepted_codes = []
    for r in results:
        if r.get("ok"):
            text = pathlib.Path(r["accepted"]).read_text(encoding="utf-8")
            body = text.split(PREFIX, 1)[1].rsplit("end AllStepChecks", 1)[0].strip()
            accepted_codes.append(body)
    consolidated = PREFIX + "\n\n" + "\n\n".join(accepted_codes) + "\n\nend AllStepChecks\n"
    (OUT / "all_accepted.lean").write_text(consolidated, encoding="utf-8")
    ok, log = check(OUT / "all_accepted.lean")
    (OUT / "all_accepted.lean.log").write_text(log, encoding="utf-8")
    print(json.dumps({"results": results, "all_accepted_ok": ok, "all_accepted_log": log[-1000:]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
