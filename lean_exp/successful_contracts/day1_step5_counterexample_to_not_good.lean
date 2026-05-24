
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

theorem day1_step5_counterexample_to_not_good_contract
    (c : ℝ)
    (h_ex : ∃ n : ℕ, ∃ hn : 0 < n, ∃ x : Fin n → ℝ,
      Day1P2QForm n x < c * Day1P2NormSq n x) :
    ¬ Day1P2GoodConstant c := by
  rcases h_ex with ⟨n, hn, x, hlt⟩
  intro hgood
  have hge := hgood n hn x
  linarith

end StepSuite
