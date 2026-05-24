import Mathlib

namespace NonGeometryStepContracts

open Finset
set_option linter.unusedVariables false

/-!
Successful local transition contracts for the non-geometry problems in the two
supplied CMO 2023 pages. Hard olympiad lemmas are represented as the next
proof-state subgoals/hypotheses; each theorem checks that the local transition
itself is sound.
-/

section Day1P1

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

theorem day1_p1_step1_good_from_factor_contract (lambda : ℝ)
    (h_factor : ∀ n : ℕ, 0 < n →
      ∃ x : Fin 2023 → ℕ,
        (∀ i, 0 < x i) ∧
        (∏ i : Fin 2023, x i) = n ∧
        ∀ i : Fin 2023,
          Nat.Prime (x i) ∨ (x i : ℝ) ≤ Real.rpow (n : ℝ) lambda) :
    Day1P1GoodLambda lambda := by
  exact h_factor

theorem day1_p1_step2_bad_witness_contract (lambda : ℝ)
    (h_bad : ∃ n : ℕ, ∃ hn : 0 < n,
      ¬ ∃ x : Fin 2023 → ℕ,
        (∀ i, 0 < x i) ∧
        (∏ i : Fin 2023, x i) = n ∧
        ∀ i : Fin 2023,
          Nat.Prime (x i) ∨ (x i : ℝ) ≤ Real.rpow (n : ℝ) lambda) :
    ¬ Day1P1GoodLambda lambda := by
  intro hgood
  rcases h_bad with ⟨n, hn, hnot⟩
  exact hnot (hgood n hn)

theorem day1_p1_step3_min_from_bad_below_contract (lambda : ℝ)
    (h_bad_below : ∀ mu : ℝ, mu < lambda → ¬ Day1P1GoodLambda mu) :
    ∀ mu : ℝ, Day1P1GoodLambda mu → lambda ≤ mu := by
  intro mu hgood
  by_contra! hlt
  exact h_bad_below mu hlt hgood

theorem day1_p1_step4_target_intro_contract (lambda : ℝ)
    (h_good : Day1P1GoodLambda lambda)
    (h_min : ∀ mu : ℝ, Day1P1GoodLambda mu → lambda ≤ mu) :
    Day1P1Target lambda := by
  exact ⟨h_good, h_min⟩

end Day1P1

section Day1P2

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

theorem day1_p2_step1_target_unfold_contract :
    Day1P2Target ↔
      (Day1P2GoodConstant Day1P2Answer ∧
        ∀ c : ℝ, Day1P2GoodConstant c → c ≤ Day1P2Answer) := by
  rfl

theorem day1_p2_step2_lower_from_key_contract
    (h_key : ∀ n : ℕ, 0 < n → ∀ x : Fin n → ℝ,
      Day1P2QForm n x ≥ Day1P2Answer * Day1P2NormSq n x) :
    Day1P2GoodConstant Day1P2Answer := by
  exact h_key

theorem day1_p2_step3_upper_from_counterexamples_contract
    (h_bad_above : ∀ c : ℝ, Day1P2Answer < c → ¬ Day1P2GoodConstant c) :
    ∀ c : ℝ, Day1P2GoodConstant c → c ≤ Day1P2Answer := by
  intro c hgood
  by_contra! hlt
  exact h_bad_above c hlt hgood

theorem day1_p2_step4_counterexample_contract (c : ℝ)
    (h_ex : ∃ n : ℕ, ∃ hn : 0 < n, ∃ x : Fin n → ℝ,
      Day1P2QForm n x < c * Day1P2NormSq n x) :
    ¬ Day1P2GoodConstant c := by
  intro hgood
  rcases h_ex with ⟨n, hn, x, hx⟩
  exact not_le_of_gt hx (hgood n hn x)

theorem day1_p2_step5_target_intro_contract
    (h_good : Day1P2GoodConstant Day1P2Answer)
    (h_upper : ∀ c : ℝ, Day1P2GoodConstant c → c ≤ Day1P2Answer) :
    Day1P2Target := by
  exact ⟨h_good, h_upper⟩

end Day1P2

section Day1P3

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

theorem day1_p3_step1_good_set_intro_contract (p : ℕ) (A : Finset (Fin p))
    (h_nonempty : 0 < A.card)
    (h_proper : A.card < p)
    (h_min : ∀ B : Finset (Fin p), B.card = A.card → Fval p B ≥ Fval p A) :
    Day1P3GoodSet p A := by
  exact ⟨h_nonempty, h_proper, h_min⟩

theorem day1_p3_step2_chain_intro_contract (p l : ℕ)
    (As : Fin l → Finset (Fin p))
    (h_good : ∀ i, Day1P3GoodSet p (As i))
    (h_mono : ∀ i j : Fin l, i.val ≤ j.val → As i ⊆ As j)
    (h_distinct : ∀ i j : Fin l, i ≠ j → As i ≠ As j) :
    Day1P3Chain p l := by
  exact ⟨As, h_good, h_mono, h_distinct⟩

theorem day1_p3_step3_upper_from_no_longer_chain_contract (p answer : ℕ)
    (h_no_longer : ∀ l : ℕ, answer < l → ¬ Day1P3Chain p l) :
    ∀ l : ℕ, Day1P3Chain p l → l ≤ answer := by
  intro l hchain
  by_contra! hlt
  exact h_no_longer l hlt hchain

theorem day1_p3_step4_target_intro_contract (answer : ℕ)
    (h : ∀ p : ℕ, Nat.Prime p → 5 ≤ p →
      Day1P3Chain p answer ∧ ∀ l : ℕ, Day1P3Chain p l → l ≤ answer) :
    Day1P3Target answer := by
  exact h

end Day1P3

section Day2P4

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

theorem day2_p4_step1_target_intro_contract
    (h_upper : Day2P4Upper)
    (h_ach : Day2P4Achievable) :
    Day2P4Target := by
  exact ⟨h_upper, h_ach⟩

theorem day2_p4_step2_final_arithmetic_contract (m k total : ℕ)
    (h_eq : total + k = 2 * m)
    (h_total : total ≤ 10000)
    (h_k : k ≤ 100) :
    m ≤ 5050 := by
  omega

theorem day2_p4_step3_upper_from_counting_contract
    (a : Fin 2023 → ℝ)
    (total : ℕ)
    (h_ordered : total + (LargeSet2023 a).card = 2 * (PairSet2023 a).card)
    (h_total : total ≤ 10000)
    (h_large : (LargeSet2023 a).card ≤ 100) :
    (PairSet2023 a).card ≤ 5050 := by
  exact day2_p4_step2_final_arithmetic_contract
    (PairSet2023 a).card (LargeSet2023 a).card total h_ordered h_total h_large

theorem day2_p4_step4_achievable_witness_contract
    (a : Fin 2023 → ℝ)
    (h_nonneg : ∀ i, 0 ≤ a i)
    (h_sum : (∑ i : Fin 2023, a i) = (100 : ℝ))
    (h_card : (PairSet2023 a).card = 5050) :
    Day2P4Achievable := by
  exact ⟨a, h_nonneg, h_sum, h_card⟩

end Day2P4

section Day2P6

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

theorem day2_p6_step1_reach_succ_contract (n : ℕ) (σ ρ τ : Arrangement99)
    (h_swap : OneAdjacentSwap99 σ ρ)
    (h_reach : ReachWithin99 n ρ τ) :
    ReachWithin99 (n + 1) σ τ := by
  right
  exact ⟨ρ, h_swap, h_reach⟩

theorem day2_p6_step2_upper_intro_contract (N : ℕ)
    (h : ∀ σ τ : Arrangement99, ∃ τ' : Arrangement99,
      SameUpToRotation99 τ τ' ∧ ReachWithin99 N σ τ') :
    Day2P6Upper N := by
  exact h

theorem day2_p6_step3_min_from_no_smaller_contract (N : ℕ)
    (h_no_smaller : ∀ M : ℕ, M < N → ¬ Day2P6Upper M) :
    ∀ M : ℕ, Day2P6Upper M → N ≤ M := by
  intro M hM
  by_contra! hlt
  exact h_no_smaller M hlt hM

theorem day2_p6_step4_target_intro_contract (N : ℕ)
    (h_upper : Day2P6Upper N)
    (h_min : ∀ M : ℕ, Day2P6Upper M → N ≤ M) :
    Day2P6Target N := by
  exact ⟨h_upper, h_min⟩

end Day2P6

end NonGeometryStepContracts
