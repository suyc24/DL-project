import Mathlib

namespace NegativeErrorExamples

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

/--
If the trusted target says the maximum constant is 1/2, then the earlier
DeepSeek-style answer `2 - sqrt 3` cannot also satisfy the same greatestness
contract. This is a clean negative transition example.
-/
theorem wrong_constant_greatestness_impossible
    (h_target : Day1P2Target) :
    ¬ (Day1P2GoodConstant (2 - Real.sqrt 3) ∧
        ∀ c : ℝ, Day1P2GoodConstant c → c ≤ 2 - Real.sqrt 3) := by
  intro h_wrong
  have h_le : Day1P2Answer ≤ 2 - Real.sqrt 3 := h_wrong.2 Day1P2Answer h_target.1
  unfold Day1P2Answer at h_le
  have hbound : Real.sqrt 3 ≤ (3 : ℝ) / 2 := by
    nlinarith
  have hnonneg : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3
  have hsqle : Real.sqrt 3 * Real.sqrt 3 ≤ ((3 : ℝ) / 2) * ((3 : ℝ) / 2) :=
    mul_self_le_mul_self hnonneg hbound
  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by
    rw [Real.sq_sqrt]
    norm_num
  nlinarith [hsqle, hs]

end NegativeErrorExamples
