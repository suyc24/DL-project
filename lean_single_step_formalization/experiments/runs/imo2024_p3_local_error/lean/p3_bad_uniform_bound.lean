import Mathlib

/-!
Local check for the failed step in a proposed IMO 2024 P3 proof.

The natural-language step tried to infer a uniform bound from pointwise
finiteness of counts. This file checks a distilled counterclaim:
not every natural-valued count function is eventually bounded by a constant.
-/

def BadLocalClaim : Prop :=
  ∀ count : Nat → Nat,
    (∀ n : Nat, ∃ k : Nat, count n = k) →
    ∃ M : Nat, ∀ n : Nat, count n < M

theorem p3_bad_uniform_bound_false : ¬ BadLocalClaim := by
  intro h
  have hfinite : ∀ n : Nat, ∃ k : Nat, (fun x : Nat => x) n = k := by
    intro n
    exact ⟨n, rfl⟩
  rcases h (fun x : Nat => x) hfinite with ⟨M, hM⟩
  exact Nat.lt_irrefl M (hM M)
