# All-step local transition-contract attempt

This run tests whether local natural-language proof steps can be converted into
Lean transition contracts. `reduced-to-subgoal` is counted as a successful
formalization when the missing mathematical content is exposed as an explicit
`h_missing_*` hypothesis and Lean verifies that the old goal follows from it.

## DeepSeek v4 Pro run

Script: `all_step_contract_suite.py`

Result of the automated DeepSeek run:

- 25 local tasks attempted.
- 22 were accepted directly by Lean.
- 1 reduced-to-subgoal task for Day2 P4 concentration was accepted after compiler feedback.
- 2 simple bookkeeping tasks failed because DeepSeek returned reasoning/log text instead of final Lean code, not because the transition theorem was false.

The final checked file `all_step_contracts_success.lean` contains Lean contracts
for all intended steps. It is typechecked successfully.

## Classification

- `valid`: the local step is directly proved from current hypotheses.
- `reduced-to-subgoal`: the step is converted into an explicit missing lemma or
  subgoal. The transition is checked, but the missing lemma remains to be proved.

## Coverage

- Day1 P1: factor-good, bad-witness, minimality, target-intro.
- Day1 P2: target-unfold, lower-from-key-inequality, upper-from-counterexamples,
  counterexample-to-not-good, target-intro.
- Day1 P3: good-set-intro, chain-intro, upper-from-no-longer-chain, target-intro.
- Day2 P4: target-intro, arithmetic-combination, upper-from-counting,
  achievable-witness, concentration reduced-to-subgoal.
- Day2 P5: geometry target split, angle subgoal, length subgoal. These are
  reduced-to-subgoal because the Euclidean geometry formal domain is not yet
  developed.
- Day2 P6: reachability successor, upper predicate, minimality, target-intro.

## Main conclusion

Yes, the local-step checking task is feasible. The most robust pattern is:

1. Freeze the proof-state definitions.
2. Ask DeepSeek for the local transition theorem/proof.
3. If the step is mathematically substantial, require an explicit `h_missing_*`
   hypothesis instead of allowing hidden assumptions.
4. Let Lean typecheck the resulting transition.

The method does not prove hard olympiad lemmas by itself; it checks whether a
natural-language step is a sound transition or cleanly reduces it to named
subgoals.
