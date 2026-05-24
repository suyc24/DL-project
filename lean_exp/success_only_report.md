# Success-only Lean transition contracts

This folder has been cleaned so that raw failed trajectories and earlier
exploratory outputs are removed. The successful Lean contracts are collected in
`lean_exp/successful_contracts/`.

## What changed

The earlier experiments only formalized statement shapes and a few final
assembly steps. The new file `non_geometry_step_contracts.lean` collects
successful local transition contracts for every non-geometry problem in the two
supplied CMO 2023 pages:

- Day 1 Problem 1
- Day 1 Problem 2
- Day 1 Problem 3
- Day 2 Problem 4
- Day 2 Problem 6

Day 2 Problem 5 is the geometry problem and is not included here because the
Euclidean geometry domain, angle conventions, line extension assumptions, circle
intersection, and product-of-lengths vocabulary need a separate audited geometry
formalization layer first.

## Meaning of “step formalization”

Each theorem checks a local transition contract: if the next proof-state
subgoals are proved, then the current goal follows. The hard olympiad lemmas are
not silently assumed as global axioms; they appear as local hypotheses of the
corresponding transition theorem.

So these files do not claim to solve the whole olympiad sheet. They check the
legality of the local proof-state operations: introducing witnesses, splitting a
maximum statement into upper/lower parts, turning counterexamples into negated
universal claims, and combining counting inequalities.

## Successful files

- `successful_contracts/non_geometry_step_contracts.lean`: consolidated
  successful contracts for all non-geometry problems.
- `successful_contracts/negative_error_examples.lean`: successful Lean proof
  that the earlier wrong Day1 P2 candidate `2 - sqrt 3` is incompatible with the
  trusted `1/2` greatestness target.
- The remaining files in `successful_contracts/` are the previously accepted
  per-step contracts copied out of the old raw output tree.

## Coverage Summary

### Day1 P1

Steps covered:

1. A factorization construction implies `GoodLambda lambda`.
2. A bad witness `n` implies `¬ GoodLambda lambda`.
3. Badness below `lambda` implies minimality of `lambda`.
4. Goodness plus minimality implies the target.

### Day1 P2

Steps covered:

1. Target unfolds to `GoodConstant 1/2` plus greatestness.
2. The lower bound follows from the key inequality subgoal.
3. The upper bound follows from showing every `c > 1/2` is not good.
4. A concrete counterexample implies `¬ GoodConstant c`.
5. Lower bound plus upper bound implies the target.

### Day1 P3

Steps covered:

1. Good-set introduction from nonempty/proper/minimality facts.
2. Chain introduction from good sets, monotonicity, and distinctness.
3. No longer chain implies the upper bound on chain length.
4. Existence plus upper bound implies the target answer.

### Day2 P4

Steps covered:

1. Target splits into upper bound plus achievability.
2. Arithmetic contract: `total + k = 2m`, `total ≤ 10000`, `k ≤ 100` imply
   `m ≤ 5050`.
3. Counting facts imply the pair-set upper bound.
4. A witness with nonnegativity, sum, and cardinality facts implies
   achievability.

### Day2 P6

Steps covered:

1. One adjacent swap followed by an `n`-step reachability proof gives an
   `(n+1)`-step reachability proof.
2. A representative reachability statement gives the upper-bound predicate.
3. No smaller upper bound gives minimality.
4. Upper bound plus minimality gives the target.

## Error-catching retained as a successful proof

The failed raw attempts were deleted, but the useful negative result was kept as
a successful theorem: `negative_error_examples.lean` shows that the earlier
DeepSeek-style answer `2 - sqrt 3` cannot satisfy the same greatestness contract
as the trusted answer `1/2`.
