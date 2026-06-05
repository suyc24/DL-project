def BeforeH (H c : Nat) : Prop :=
  c < H

def BecomesH (H c : Nat) : Prop :=
  c + 1 = H

theorem at_most_before_not_exact_before :
    ¬ (∀ H c : Nat, H = 5 → BeforeH H c → BecomesH H c) := by
  intro h
  have hBefore : BeforeH 5 2 := by
    unfold BeforeH
    decide
  have hExact : BecomesH 5 2 := h 5 2 rfl hBefore
  unfold BecomesH at hExact
  have hNot : ¬ (2 + 1 = 5) := by
    decide
  exact hNot hExact

