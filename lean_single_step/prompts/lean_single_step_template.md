You are a Lean 4 expert.  You will receive the previous steps of a natural‑language proof and one **target step** to formalise.

Formalise the target step as a **transition contract**: a theorem that states the conclusion of this step, assuming the conclusions of all previous steps are already true.  If the step depends on a mathematical fact that has not yet been proved, introduce it as an explicit hypothesis named `h_missing_*` (e.g., `h_missing_concentration`).  Do **not** write a full proof of the whole problem — only the transition from the previous steps to the current one.

Do not assume the final conclusion as a hypothesis. The hypotheses should only contain facts established in previous steps. If the conclusion is logically equivalent to one of the hypotheses, you are missing a reasoning step.

Output format:
- A single fenced code block  ```lean … ```  containing the Lean 4 code.
- Inside the block, include `import Mathlib` (already available) if needed.
- Use `Finset.prod` / `Finset.sum` instead of ∏ / ∑ notation to avoid parsing errors.
- The code must be **compilable** by `lake env lean`.

---
Previous steps (in order):
{PREVIOUS_STEPS}

Target step (natural language):
{TARGET_STEP}

Produce exactly one theorem inside the fenced block. Example:

```lean
import Mathlib

theorem step_3 (primes : Finset ℕ) (h_all_prime : ∀ p ∈ primes, Nat.Prime p) : … := by
  -- your proof using previous facts
  sorry   -- you may use sorry if the proof is not complete, but prefer to add h_missing_* hypotheses
```

Important: Use explicit arguments for all necessary hypotheses. Do not use the variables command.

