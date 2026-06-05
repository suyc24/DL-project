# IMO 2024 P3 Local Lean Feedback Experiment Report

## 1. Objective

This experiment tests whether a solver-auditor-formalizer loop can reduce premature convergence to confident but wrong mathematical reasoning.

The target problem was IMO 2024 Problem 3:

> Let \(a_1,a_2,\ldots\) be an infinite sequence of positive integers, and let \(N\) be a positive integer. For every \(n\ge N\), \(a_n\) is the number of occurrences of \(a_{n-1}\) among \(a_1,\ldots,a_{n-1}\). Prove that at least one of the two subsequences \(a_1,a_3,a_5,\ldots\) and \(a_2,a_4,a_6,\ldots\) is eventually periodic.

The intended workflow was:

1. A solver model generates a natural-language proof.
2. An auditor model identifies only the first problematic local step, without solving the problem.
3. The problematic step is distilled into a small Lean statement or counterexample.
4. Lean verifies that the local step is indeed invalid.
5. Only this local feedback is returned to the solver.
6. The solver continues from the corrected state.
7. The loop repeats until a correct proof is produced.

The experiment deliberately avoided web search. The sub-agents were instructed not to browse or query external sources.

## 2. Infrastructure

The experiment followed the style of `lean_single_step_formalization`, rather than the older `lean_single_step_formalization_v1` flow.

Local Lean project:

```text
/home/suyc24/Python/DL-project/lean_fhis
```

Experiment directory:

```text
/home/suyc24/Python/DL-project/lean_single_step_formalization/experiments/runs/imo2024_p3_local_error
```

Remote Lean host requested by the user:

```text
ssh -p 36685 root@connect.westd.seetacloud.com
```

The remote host initially did not have Lean available. A minimal standalone Lean 4.30 runtime was prepared and transferred to:

```text
/root/autodl-tmp/lean-4.30.0-linux/bin/lean
```

All final local-error Lean files compiled both locally and remotely.

## 3. Iteration Log

### Iteration 1: Pointwise Finite vs Uniformly Bounded

The first attempted proof used the step:

> Since every non-recurrent number appears only finitely many times, their occurrence counts are eventually uniformly bounded.

The auditor identified this as a local quantifier error. Pointwise finiteness does not imply a uniform bound.

Lean file:

```text
lean/p3_bad_uniform_bound.lean
```

Distilled Lean idea:

```lean
def BadLocalClaim : Prop :=
  forall count : Nat -> Nat,
    (forall n : Nat, exists k : Nat, count n = k) ->
    exists M : Nat, forall n : Nat, count n < M
```

Lean refuted this with the counterexample `count n = n`.

Feedback returned to the solver:

> Pointwise finite occurrence counts do not imply a global uniform bound. A local model such as `count n = n` satisfies pointwise finiteness but has no uniform bound.

This forced the solver away from a common but invalid compactness-style shortcut.

### Iteration 2: False Finite-State Drift Argument

The next proof tried to show that a finite set of recurrent counters must remain controlled by arguing informally that the minimum counter increases regularly.

The auditor found a local update error:

> It is possible to keep choosing one high counter repeatedly while another counter remains low, so the minimum need not increase every bounded number of steps.

Lean file:

```text
lean/p3_bad_count_gap_update.lean
```

Distilled finite counterexample:

```lean
theorem two_steps_keep_choosing_one :
    step (step (10, 0, 1)) = (12, 0, 1) := by
  native_decide

theorem min_count_does_not_increase_in_two_steps :
    Nat.min 12 0 = Nat.min 10 0 := by
  native_decide
```

Feedback returned to the solver:

> A bounded number of counter updates need not raise the minimum. The abstract update system can repeatedly select the same high counter.

This blocked a premature finite-state conclusion and forced the proof to use the recurrence of all small labels more carefully.

### Iteration 3: One-Step Delay in the Big-Number Correspondence

The solver then introduced a more structural proof, but wrote:

> For every time \(t\), if \(g\) is large, then
> \[
> C_t(g)=\#\{s:C_t(s)\ge g\}.
> \]

The auditor identified a precise one-step delay. If a small recurrent value \(s\) has just reached its \(g\)-th occurrence at time \(r\), then the right-hand side already counts \(s\), but the generated large value \(g\) appears only at time \(r+1\).

Lean file:

```text
lean/p3_bad_no_delay_correspondence.lean
```

Distilled Lean counterexample:

```lean
def smallCountAtTrigger (s : Nat) : Nat :=
  if s = 1 then 5 else 0

def bigCountAtTrigger : Nat := 0

def rhsAtTrigger : Nat :=
  (if 5 <= smallCountAtTrigger 1 then 1 else 0) +
  (if 5 <= smallCountAtTrigger 2 then 1 else 0)

theorem no_delay_correspondence_false :
    bigCountAtTrigger != rhsAtTrigger := by
  native_decide
```

Feedback returned to the solver:

> The correspondence is valid only after the generated large number has appeared. It cannot be used at the exact moment when the small counter reaches the threshold.

This corrected the time indexing and prevented a subtle off-by-one proof error.

### Iteration 4: At-Most \(H-1\) vs Exactly \(H-1\)

The next proof tried to order recurrent values by the first time they reached a large height \(H\). It used:

> Before this point, no recurrent value has reached \(H\), so when value \(1\) appears next it is exactly its \(H\)-th occurrence.

The auditor found a local arithmetic/logical error:

> Knowing a count is less than \(H\) does not imply that after one more occurrence it becomes exactly \(H\).

Lean file:

```text
lean/p3_bad_at_most_to_exact_fixed.lean
```

Distilled Lean counterexample:

```lean
def BeforeH (H c : Nat) : Prop :=
  c < H

def BecomesH (H c : Nat) : Prop :=
  c + 1 = H

theorem at_most_before_not_exact_before :
    not (forall H c : Nat, H = 5 -> BeforeH H c -> BecomesH H c) := by
  intro h
  have hBefore : BeforeH 5 2 := by
    unfold BeforeH
    decide
  have hExact : BecomesH 5 2 := h 5 2 rfl hBefore
  unfold BecomesH at hExact
  have hNot : not (2 + 1 = 5) := by
    decide
  exact hNot hExact
```

Feedback returned to the solver:

> From \(c<H\), one cannot infer \(c+1=H\). The count might be far below \(H-1\).

This invalidated an overly rigid synchronization argument and pushed the solver toward a more robust dynamical-system proof.

### Iteration 5: Lexicographic Minimal Block Misuse

The solver then proposed a "first repeated distance" argument. It compared two repeated blocks and claimed that if the next value becomes smaller, this creates a lexicographically smaller repeated block.

The auditor identified the local flaw:

> The second block is not known to be another repeated block. To use lexicographic minimality, one would need \(b_{u+2d}=b_{u+d}\), which has not been proved.

Lean file:

```text
lean/p3_bad_lex_block_not_repeat.lean
```

Distilled Lean counterexample:

```lean
def b (n : Nat) : Nat :=
  match n with
  | 0 => 3
  | 1 => 2
  | 2 => 3
  | 3 => 1
  | 4 => 2
  | _ => 0

theorem local_lex_decrease_without_next_repeat :
    b 0 = b 2 /\ b 3 < b 1 /\ b 4 != b 2 := by
  native_decide
```

Feedback returned to the solver:

> A smaller following value does not by itself produce a smaller repeated block. The candidate block must first be shown to have the same repeat structure.

This prevented a polished but invalid extremal argument from being accepted.

## 4. Final Successful Proof Strategy

After the local feedback loop, the solver abandoned the fragile synchronization and repeated-block arguments. The final proof used a sorting dynamics on recurrent counters.

The proof has three conceptual stages.

### Stage 1: Recurrent Values Form a Finite Initial Segment

Let a number be recurrent if it appears infinitely often.

Choose \(K\) larger than the initial segment. No value \(y\ge K\) can appear \(K\) times: otherwise its first \(K\)-th occurrence would require \(K\) distinct predecessor values that had each already appeared at least \(y\ge K\) times. Either one of them is also at least \(K\), contradicting minimality, or all are below \(K\), impossible because there are only \(K-1\) positive integers below \(K\).

Thus recurrent values are finite. They are nonempty, and downward closed: if \(k\) is recurrent, then every \(j<k\) is recurrent. Hence the recurrent set is:

```text
{1, 2, ..., r}
```

### Stage 2: The Tail Alternates Between Recurrent and Large Non-Recurrent Values

For all sufficiently large \(g\), the only values that can occur at least \(g\) times are the recurrent values \(1,\ldots,r\). Each recurrent value eventually has a \(g\)-th occurrence and then produces \(g\) one step later.

Therefore each sufficiently large \(g\) appears exactly \(r\) times.

After deleting a finite prefix:

```text
recurrent value, large value, recurrent value, large value, ...
```

So all sufficiently late recurrent values lie in one parity class. It remains to prove that the recurrent subsequence is eventually periodic.

### Stage 3: Sorting Dynamics of Recurrent Counters

Let the recurrent subsequence be:

```text
b_1, b_2, b_3, ...
```

where each \(b_t\in\{1,\ldots,r\}\). Let \(d_i(t)\) be the total count of recurrent value \(i\) after the first \(t\) recurrent terms.

If \(b_t=i\), then \(d_i(t)\) is the large value generated after this recurrent term. The next recurrent value is the number of recurrent counters whose counts are at least \(d_i(t)\):

\[
b_{t+1}=\#\{j:d_j(t)\ge d_i(t)\}.
\]

Now sort the counter values:

\[
x_1(t)\ge x_2(t)\ge\cdots\ge x_r(t).
\]

The key observation is that the next transition depends only on this sorted profile up to an additive constant vector. More precisely, there is a fixed vector \(A\) such that:

\[
d_k(t+1)=x_k(t)+A_k,
\qquad
x(t+1)=\operatorname{sort}_{\downarrow}(x(t)+A).
\]

This converts the recurrent subsequence into a finite-dimensional deterministic sorting system.

The adjacent gaps \(x_s(t)-x_{s+1}(t)\) are bounded. If a gap exceeds

\[
\Delta=\max A_i-\min A_i,
\]

then adding \(A\) cannot mix the upper and lower blocks. If the upper block permanently gains total mass while the lower block does not, some recurrent labels would stop appearing, contradicting recurrence. Hence large gaps cannot keep increasing. Since a single step can enlarge a gap by at most \(\Delta\), all normalized sorted states are bounded.

Therefore

\[
(x_1(t)-x_r(t),\ldots,x_{r-1}(t)-x_r(t),0)
\]

takes only finitely many values and evolves deterministically. It is eventually periodic. This implies the increment coordinate, hence \(b_t\), is eventually periodic.

Since the recurrent subsequence occupies one parity class in the original sequence, one of

```text
a_1, a_3, a_5, ...
a_2, a_4, a_6, ...
```

is eventually periodic.

## 5. Final Verification Result

The final sorting-dynamics proof was sent to an independent GPT-5.5 auditor with instructions:

- no web search;
- do not provide a replacement proof;
- only identify the earliest local error if one exists;
- otherwise return `verdict: correct`.

The auditor returned:

```text
verdict: correct
```

Additional local random stress tests were run on the abstract ranking dynamics. They did not find counterexamples to:

- the sorted-dynamics identity;
- eventual periodicity in small finite states;
- the gap inequality used in the final proof.

These tests are not a proof, but they helped catch earlier false proof ideas and increased confidence before the final auditor pass.

## 6. Why This Is a Good Example for the Method

This experiment shows the value of local formal feedback in exactly the intended way.

The solver repeatedly produced plausible high-level mathematical arguments. Several of them looked polished and could easily pass a superficial natural-language review. However, each failed at a small local inference:

- exchanging pointwise and uniform boundedness;
- assuming a minimum counter must regularly increase;
- ignoring a one-step delay;
- turning \(c<H\) into \(c+1=H\);
- using lexicographic minimality on an object not known to be in the comparison class.

None of these errors required formalizing the whole IMO problem in Lean. Each was distilled into a small finite or first-order Lean check. This made the feedback precise, cheap, and hard to ignore.

The most important behavior was not that Lean solved the problem. It did not. Instead, Lean acted as a local error detector that prevented the solver from stopping at a confident but invalid proof. The solver then had to revise the proof structure. After several iterations, the argument changed qualitatively: it moved from informal synchronization claims to a robust sorting-dynamics invariant.

This supports the proposed story:

> A strong LLM may initially stop at a convincing but false proof. A local verifier does not need to solve the whole problem; it only needs to invalidate the fragile step. Repeated local invalidation can steer the solver away from shallow proof patterns and toward a genuinely stable argument.

## 7. Takeaways for the Pipeline

This example suggests the following design principles for the full system:

1. The reviewer should identify the first local error, not repair the proof globally.
2. The formalizer should target the smallest checkable abstraction of that error.
3. The Lean check can be a counterexample, not necessarily a theorem from the original problem.
4. Feedback to the solver should be minimal and local.
5. Iteration should continue until the proof changes structure, not merely wording.
6. A final independent reviewer is useful after the last repair.

The strongest evidence from this run is that the final proof was not a minor edit of the first proof. The local Lean feedback forced the solver to abandon multiple attractive but invalid routes and eventually produce a higher-quality proof based on a more stable invariant.

