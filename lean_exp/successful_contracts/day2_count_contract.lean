
import Mathlib

namespace ProCompilerLoop

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

theorem final_arithmetic_contract (m k total : ℕ)
    (h_eq : total + k = 2 * m)
    (h_total : total ≤ 10000)
    (h_k : k ≤ 100) :
    m ≤ 5050 := by
  omega

theorem day2_upper_from_count_contract
    (a : Fin 2023 → ℝ)
    (total : ℕ)
    (h_ordered : total + (LargeSet2023 a).card = 2 * (PairSet2023 a).card)
    (h_total : total ≤ 10000)
    (h_large : (LargeSet2023 a).card ≤ 100) :
    (PairSet2023 a).card ≤ 5050 := by
  exact final_arithmetic_contract (PairSet2023 a).card (LargeSet2023 a).card total h_ordered h_total h_large

end ProCompilerLoop
