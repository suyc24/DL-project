
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

theorem day2_step3_achievable_from_witness_contract
    (a : Fin 2023 → ℝ)
    (h_nonneg : ∀ i, 0 ≤ a i)
    (h_sum : (∑ i : Fin 2023, a i) = (100 : ℝ))
    (h_card : (PairSet2023 a).card = 5050) :
    Day2P4Achievable := by
  refine ⟨a, h_nonneg, h_sum, h_card⟩

end StepSuite
