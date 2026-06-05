def b (n : Nat) : Nat :=
  match n with
  | 0 => 3
  | 1 => 2
  | 2 => 3
  | 3 => 1
  | 4 => 2
  | _ => 0

theorem local_lex_decrease_without_next_repeat :
    b 0 = b 2 ∧ b 3 < b 1 ∧ b 4 ≠ b 2 := by
  native_decide

