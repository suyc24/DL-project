def smallCountAtTrigger (s : Nat) : Nat :=
  if s = 1 then 5 else 0

def bigCountAtTrigger : Nat := 0

def rhsAtTrigger : Nat :=
  (if 5 ≤ smallCountAtTrigger 1 then 1 else 0) +
  (if 5 ≤ smallCountAtTrigger 2 then 1 else 0)

theorem no_delay_correspondence_false :
    bigCountAtTrigger ≠ rhsAtTrigger := by
  native_decide

