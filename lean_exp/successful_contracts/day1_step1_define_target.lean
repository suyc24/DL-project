
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

theorem day1_step1_target_unfold_contract :
    Day1P2Target ↔
      (Day1P2GoodConstant Day1P2Answer ∧
        ∀ c : ℝ, Day1P2GoodConstant c → c ≤ Day1P2Answer) := by
  rfl

end StepSuite
