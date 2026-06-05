import Mathlib

/-!
Second local check for the repaired IMO 2024 P3 proof.

The repaired proof claimed that, under the proposed finite-state update rule,
within at most r steps the current minimum count increases and the current
maximum count increases at most once. The following concrete r = 2 state
refutes that local dynamical claim.
-/

def nextB (c1 c2 b : Nat) : Nat :=
  let target := if b = 1 then c1 else c2
  (if target ≤ c1 then 1 else 0) + (if target ≤ c2 then 1 else 0)

def step (state : Nat × Nat × Nat) : Nat × Nat × Nat :=
  match state with
  | (c1, c2, b) =>
      let c1' := if b = 1 then c1 + 1 else c1
      let c2' := if b = 2 then c2 + 1 else c2
      (c1', c2', nextB c1' c2' b)

theorem two_steps_keep_choosing_one :
    step (step (10, 0, 1)) = (12, 0, 1) := by
  norm_num [step, nextB]

theorem min_count_does_not_increase_in_two_steps :
    Nat.min 12 0 = Nat.min 10 0 := by
  norm_num
